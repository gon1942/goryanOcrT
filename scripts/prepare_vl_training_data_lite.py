#!/usr/bin/env python3
"""Lightweight training data generator — reuses existing pipeline output.

Instead of running the full pipeline, this script reads the already-generated
result.pretty.json and table_refine crops to create training pairs.

Usage:
    python scripts/prepare_vl_training_data_lite.py \
        --input-dir output_input7_v8/ \
        --output-dir data/vl_training/
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


def extract_training_pairs(
    output_dir: Path,
    image_subdir: str = "table_refine/crops",
) -> list[dict]:
    """Extract (image, prompt, response) pairs from pipeline output."""
    result_json = output_dir / "result.pretty.json"
    if not result_json.exists():
        print(f"ERROR: {result_json} not found. Run pipeline first.")
        return []

    with open(result_json) as f:
        doc = json.load(f)

    crops_dir = output_dir / image_subdir
    samples: list[dict] = []

    # Find all table blocks and their crop images
    crop_images = sorted(crops_dir.glob("*.png")) if crops_dir.exists() else []
    crop_map = {}
    for crop_path in crop_images:
        # Extract page and table index from filename
        name = crop_path.stem  # e.g. "page_0001_table_002"
        crop_map[name] = crop_path

    for page in doc.get("pages", []):
        page_no = page.get("page_no", 0)
        for block in page.get("blocks", []):
            if block.get("block_type") != "table":
                continue
            html = block.get("html", "")
            if not html or len(html) < 50:
                continue

            # Try to find matching crop image
            block_id = block.get("block_id", "")
            source = block.get("source_engine", "")

            # Try exact block_id match
            matched_crop = None
            for crop_name, crop_path in crop_map.items():
                if block_id in crop_name:
                    matched_crop = crop_path
                    break

            # Try page-based matching
            if not matched_crop:
                for crop_name, crop_path in crop_map.items():
                    if f"page_{page_no:04d}" in crop_name:
                        matched_crop = crop_path
                        break

            if not matched_crop and crop_images:
                # Use first available crop as fallback
                matched_crop = crop_images[0]

            if matched_crop:
                # Clean HTML for training
                import re
                html_clean = re.sub(r"""\s*style\s*=\s*["'][^"']*["']""", "", html, flags=re.I)
                html_clean = re.sub(r">\s+<", "><", html_clean).strip()

                samples.append({
                    "image": str(matched_crop.resolve()),
                    "prompt": "Table Recognition:",
                    "response": html_clean,
                    "source": Path(output_dir).name,
                    "page": page_no,
                    "block_id": block_id,
                })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare VL training data from existing output")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Pipeline output directory (containing result.pretty.json)")
    parser.add_argument("--input-dirs", type=str, nargs="*", default=None,
                        help="Multiple pipeline output directories")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for training data")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine input directories to scan
    scan_dirs: list[Path] = []
    if args.input_dirs:
        scan_dirs = [Path(d) for d in args.input_dirs]
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        scan_dirs = sorted(input_dir.parent.glob(f"{input_dir.name}*"))
        scan_dirs = [d for d in scan_dirs if d.is_dir()]
    else:
        print("ERROR: Provide --input-dir or --input-dirs")
        return

    # Collect from all directories
    all_samples: list[dict] = []
    seen_responses: set[str] = set()
    for entry in scan_dirs:
        print(f"Scanning: {entry.name}")
        samples = extract_training_pairs(entry)
        # Deduplicate by response content
        new_count = 0
        for s in samples:
            resp_key = s["response"][:200]  # partial match for dedup
            if resp_key not in seen_responses:
                seen_responses.add(resp_key)
                all_samples.append(s)
                new_count += 1
        print(f"  Found {len(samples)} table samples, {new_count} new (after dedup)")

    if not all_samples:
        print("No training samples found!")
        return

    # Shuffle and split
    random.shuffle(all_samples)
    val_count = max(1, int(len(all_samples) * args.val_ratio))
    val_samples = all_samples[:val_count]
    train_samples = all_samples[val_count:]

    # Write JSONL
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n=== Summary ===")
    print(f"Total: {len(all_samples)} samples")
    print(f"Train: {len(train_samples)} → {train_path}")
    print(f"Val:   {len(val_samples)} → {val_path}")

    if all_samples:
        s = all_samples[0]
        print(f"\nSample:")
        print(f"  image: {s['image']}")
        print(f"  response (150 chars): {s['response'][:150]}...")


if __name__ == "__main__":
    main()
