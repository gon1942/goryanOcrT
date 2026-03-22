#!/usr/bin/env python3
"""Prepare training data for PaddleOCR-VL LoRA fine-tuning.

Generates (image, HTML) pairs from PDF files by running the existing OCR
pipeline and extracting table crop images with their corrected HTML labels.

Usage:
    python scripts/prepare_vl_training_data.py \
        --input-dir samp/ \
        --output-dir data/vl_training/ \
        --val-ratio 0.2

Output format (JSONL):
    {"image": "path/to/crop.png", "prompt": "Table Recognition:", "response": "<table>...</table>"}
    {"image": "path/to/crop.png", "prompt": "OCR:", "response": "extracted text"}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ocr_pipeline.pdf_utils import render_pdf_to_images, extract_pdf_page_words
from ocr_pipeline.config import PipelineConfig
from ocr_pipeline.schemas import InputProfile


def extract_tables_from_pipeline(
    pdf_path: Path,
    work_dir: Path,
) -> list[dict]:
    """Run the OCR pipeline on a PDF and extract table blocks with crops and HTML."""
    from ocr_pipeline.pipeline import OCRPipeline

    config = PipelineConfig()
    config.output.save_raw = True
    pipeline = OCRPipeline(config)
    doc = pipeline.run(str(pdf_path), str(work_dir))

    samples: list[dict] = []
    for page in doc.pages:
        page_dir = work_dir / f"page_{page.page_no:04d}"
        for block in page.blocks:
            if block.block_type != "table" or not block.html:
                continue
            # Find the crop image from table_refine if available
            crop_name = f"page_{page.page_no:04d}_table_*"
            crop_dir = page_dir / "table_refine" / "crops"
            crops = sorted(crop_dir.glob(crop_name + ".png")) if crop_dir.exists() else []
            if not crops:
                # Try direct work_dir structure
                crop_dir2 = work_dir / "table_refine" / "crops"
                crops = sorted(crop_dir2.glob(crop_name + ".png")) if crop_dir2.exists() else []

            # Use source image if no crop
            if not crops and block.extra.get("raw_type") == "table":
                # Generate crop from page image
                crops = _crop_table_from_page(block, page, work_dir)

            for crop_path in crops:
                html = _clean_html_for_training(block.html)
                if html and len(html) > 50:
                    prompt = "Table Recognition:"
                    samples.append({
                        "image": str(crop_path.resolve()),
                        "prompt": prompt,
                        "response": html,
                        "source": pdf_path.name,
                        "page": page.page_no,
                        "block_id": block.block_id,
                    })

    return samples


def _clean_html_for_training(html: str) -> str:
    """Clean HTML table for training — keep structure, remove verbose styles."""
    if not html:
        return ""
    # Remove inline styles
    html = re.sub(r"""\s*style\s*=\s*["'][^"']*["']""", "", html, flags=re.I)
    # Normalize whitespace
    html = re.sub(r">\s+<", "><", html)
    # Ensure proper table tags
    html = html.strip()
    return html


def _crop_table_from_page(block, page, work_dir: Path) -> list[Path]:
    """Crop a table region from the page image using block bbox."""
    from PIL import Image

    source = page.extra.get("source_image", "")
    if not source or not os.path.exists(source):
        return []

    img = Image.open(source)
    bbox = block.bbox
    if len(bbox) < 4 or bbox == [0.0, 0.0, 10.0, 10.0]:
        return []

    # bbox is in page coordinate space (same as image if not resized)
    x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
    # Clamp to image bounds
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.width, x1), min(img.height, y1)
    if x1 <= x0 or y1 <= y0:
        return []

    crop = img.crop((x0, y0, x1, y1))
    crop_dir = work_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crop_dir / f"{block.block_id}.png"
    crop.save(str(crop_path))
    return [crop_path]


def extract_text_blocks_from_pipeline(
    pdf_path: Path,
    work_dir: Path,
) -> list[dict]:
    """Extract text blocks (non-table) for OCR training data."""
    from ocr_pipeline.pipeline import OCRPipeline

    config = PipelineConfig()
    pipeline = OCRPipeline(config)
    doc = pipeline.run(str(pdf_path), str(work_dir))

    samples: list[dict] = []
    for page in doc.pages:
        page_dir = work_dir / f"page_{page.page_no:04d}"
        for block in page.blocks:
            if block.block_type not in ("text", "title") or not block.text:
                continue
            if len(block.text.strip()) < 5:
                continue

            # Use source image as the text block image
            source = page.extra.get("source_image", "")
            if not source or not os.path.exists(source):
                continue

            samples.append({
                "image": source,
                "prompt": "OCR:",
                "response": block.text.strip(),
                "source": pdf_path.name,
                "page": page.page_no,
                "block_id": block.block_id,
            })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare VL training data from PDFs")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for training data")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--table-only", action="store_true", help="Only extract table data")
    parser.add_argument("--text-only", action="store_true", help="Only extract text data")
    args = parser.parse_args()

    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
            # Extract table samples
            if not args.text_only:
                table_samples = extract_tables_from_pipeline(pdf_path, work_dir)
                print(f"  Tables: {len(table_samples)}")
                all_samples.extend(table_samples)

            # Extract text samples
            if not args.table_only:
                text_samples = extract_text_blocks_from_pipeline(pdf_path, work_dir)
                print(f"  Text blocks: {len(text_samples)}")
                all_samples.extend(text_samples)

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

    # Write JSONL files
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

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

    # Print sample
    if all_samples:
        sample = all_samples[0]
        print(f"\nSample:")
        print(f"  image: {sample['image']}")
        print(f"  prompt: {sample['prompt']}")
        print(f"  response (first 200): {sample['response'][:200]}...")


if __name__ == "__main__":
    main()
