#!/usr/bin/env python3
"""Prepare training data for SLANet table structure recognition fine-tuning.

Generates PubTabTableRecDataset format from PDF files:
  - Crop images of tables
  - Generate HTML structure + cell annotations

Usage:
    python scripts/prepare_slanet_training_data.py \
        --input-dir samp/ \
        --output-dir data/slanet_training/ \
        --val-ratio 0.2

Output format (train.txt / val.txt, JSONL):
    {"filename": "img_001.png", "html": {"structure": {"tokens": ["td", "td", ...]}, "cells": [{"tokens": ["H", "e", "l", "l", "o"], ...}]}}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from html import unescape
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


HTML_CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.I | re.S)
HTML_TAG_RE = re.compile(r"<[^>]+>")
HTML_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.I | re.S)
SPAN_RE = re.compile(r"\b(rowspan|colspan)\s*=\s*[\"']?(\d+)[\"']?", re.I)


def html_to_pubtab_format(html: str) -> dict | None:
    """Convert HTML table to PubTabTableRecDataset format.

    Returns:
        {
            "structure": {"tokens": ["<thead>", "<tr>", "<td>", "</td>", ...]},
            "cells": [{"tokens": ["H", "e", "l", "l", "o"]}, ...]
        }
    """
    if not html:
        return None

    rows = HTML_ROW_RE.findall(html)
    if len(rows) < 1:
        return None

    structure_tokens: list[str] = []
    cells: list[dict] = []

    for row_html in rows:
        structure_tokens.append("<tr>")
        for cell_match in HTML_CELL_RE.finditer(row_html):
            cell_full = cell_match.group(0)
            cell_text = unescape(HTML_TAG_RE.sub("", cell_match.group(1))).strip()

            # Determine tag
            tag_match = re.match(r"<(t[dh])", cell_full, re.I)
            tag = tag_match.group(1).lower() if tag_match else "td"

            # Get spans
            spans = {m.group(1).lower(): int(m.group(2)) for m in SPAN_RE.finditer(cell_full)}
            rs = spans.get("rowspan", 1)
            cs = spans.get("colspan", 1)

            structure_tokens.append(f"<{tag}>")
            if rs > 1:
                structure_tokens.append(f"<rowspan={rs}>")
            if cs > 1:
                structure_tokens.append(f"<colspan={cs}>")

            # Cell content as character tokens
            cell_chars = list(cell_text) if cell_text else []
            cells.append({"tokens": cell_chars})

            if cs > 1:
                structure_tokens.append("</colspan>")
            if rs > 1:
                structure_tokens.append("</rowspan>")
            structure_tokens.append(f"</{tag}>")

        structure_tokens.append("</tr>")

    return {
        "structure": {"tokens": structure_tokens},
        "cells": cells,
    }


def extract_tables_for_slanet(
    pdf_path: Path,
    work_dir: Path,
    output_img_dir: Path,
) -> list[dict]:
    """Run pipeline and extract table data in PubTab format."""
    from ocr_pipeline.pipeline import run_pipeline
    from ocr_pipeline.config import InputProfile

    profile = InputProfile.from_pdf(pdf_path)
    doc = run_pipeline(
        input_path=str(pdf_path),
        work_dir=str(work_dir),
        profile=profile,
    )

    samples: list[dict] = []
    sample_idx = 0

    for page in doc.pages:
        for block in page.blocks:
            if block.block_type != "table" or not block.html:
                continue

            pubtab = html_to_pubtab_format(block.html)
            if not pubtab:
                continue

            # Skip tables with no cells
            if not pubtab["cells"]:
                continue

            # Find or create crop image
            crop_path = _find_or_create_crop(block, page, work_dir, output_img_dir, sample_idx)
            if not crop_path:
                continue

            # Use relative filename in the dataset
            rel_filename = crop_path.name
            samples.append({
                "filename": rel_filename,
                "html": pubtab,
                "source": pdf_path.name,
                "page": page.page_no,
                "block_id": block.block_id,
            })
            sample_idx += 1

    return samples


def _find_or_create_crop(
    block, page, work_dir: Path, output_img_dir: Path, idx: int
) -> Path | None:
    """Find existing crop or create one from page image."""
    from PIL import Image

    # Try to find existing crop
    for search_dir in [
        work_dir / "table_refine" / "crops",
        Path(work_dir).parent / "table_refine" / "crops",
    ]:
        if search_dir.exists():
            crops = sorted(search_dir.glob(f"page_{page.page_no:04d}_table_*.png"))
            if crops:
                # Copy to output dir with unique name
                src = crops[idx % len(crops)]
                dst = output_img_dir / f"table_{idx:04d}.png"
                if not dst.exists():
                    shutil.copy2(str(src), str(dst))
                return dst

    # Create crop from page image
    source = page.extra.get("source_image", "")
    if not source or not os.path.exists(source):
        return None

    bbox = block.bbox
    if len(bbox) < 4 or bbox == [0.0, 0.0, 10.0, 10.0]:
        return None

    img = Image.open(source)
    x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.width, x1), min(img.height, y1)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = img.crop((x0, y0, x1, y1))
    dst = output_img_dir / f"table_{idx:04d}.png"
    crop.save(str(dst))
    return dst


def main():
    parser = argparse.ArgumentParser(description="Prepare SLANet training data")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with PDF files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_img_dir = output_dir / "images"
    output_img_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files")

    all_samples: list[dict] = []
    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")
        work_dir = output_dir / "pipeline_work" / pdf_path.stem
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            samples = extract_tables_for_slanet(pdf_path, work_dir, output_img_dir)
            print(f"  Tables: {len(samples)}")
            all_samples.extend(samples)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    if not all_samples:
        print("No training samples generated!")
        return

    # Shuffle and split
    random.shuffle(all_samples)
    val_count = max(1, int(len(all_samples) * args.val_ratio))
    val_samples = all_samples[:val_count]
    train_samples = all_samples[val_count:]

    # Write JSONL in PubTab format (train.txt / val.txt)
    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"

    with open(train_path, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n=== Summary ===")
    print(f"Total samples: {len(all_samples)}")
    print(f"  Train: {len(train_samples)} → {train_path}")
    print(f"  Val:   {len(val_samples)} → {val_path}")
    print(f"  Images: {output_img_dir}")

    if all_samples:
        sample = all_samples[0]
        print(f"\nSample:")
        print(f"  filename: {sample['filename']}")
        struct_tokens = sample['html']['structure']['tokens'][:20]
        print(f"  structure tokens: {struct_tokens}...")
        cell_tokens = sample['html']['cells'][0]['tokens'] if sample['html']['cells'] else []
        print(f"  first cell tokens: {cell_tokens}")


if __name__ == "__main__":
    main()
