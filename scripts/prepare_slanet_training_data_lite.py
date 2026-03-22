#!/usr/bin/env python3
"""Lightweight SLANet training data generator — reuses existing pipeline output.

Usage:
    python scripts/prepare_slanet_training_data_lite.py \
        --input-dir output_input7_v8/ \
        --output-dir data/slanet_training/
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from html import unescape
from pathlib import Path


HTML_CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.I | re.S)
HTML_TAG_RE = re.compile(r"<[^>]+>")
HTML_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.I | re.S)
SPAN_RE = re.compile(r"\b(rowspan|colspan)\s*=\s*[\"']?(\d+)[\"']?", re.I)


def html_to_pubtab(html: str) -> dict | None:
    if not html:
        return None
    rows = HTML_ROW_RE.findall(html)
    if not rows:
        return None

    structure_tokens: list[str] = []
    cells: list[dict] = []

    for row_html in rows:
        structure_tokens.append("<tr>")
        for cell_match in HTML_CELL_RE.finditer(row_html):
            cell_full = cell_match.group(0)
            cell_text = unescape(HTML_TAG_RE.sub("", cell_match.group(1))).strip()
            tag_match = re.match(r"<(t[dh])", cell_full, re.I)
            tag = tag_match.group(1).lower() if tag_match else "td"
            spans = {m.group(1).lower(): int(m.group(2)) for m in SPAN_RE.finditer(cell_full)}
            rs, cs = spans.get("rowspan", 1), spans.get("colspan", 1)

            structure_tokens.append(f"<{tag}>")
            if rs > 1:
                structure_tokens.append(f"<rowspan={rs}>")
            if cs > 1:
                structure_tokens.append(f"<colspan={cs}>")
            cells.append({"tokens": list(cell_text) if cell_text else []})
            if cs > 1:
                structure_tokens.append("</colspan>")
            if rs > 1:
                structure_tokens.append("</rowspan>")
            structure_tokens.append(f"</{tag}>")
        structure_tokens.append("</tr>")

    return {"structure": {"tokens": structure_tokens}, "cells": cells}


def main():
    parser = argparse.ArgumentParser(description="Prepare SLANet training data (lite)")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    result_json = input_dir / "result.pretty.json"
    if not result_json.exists():
        print(f"ERROR: {result_json} not found")
        return

    with open(result_json) as f:
        doc = json.load(f)

    crops_dir = input_dir / "table_refine" / "crops"
    crop_images = sorted(crops_dir.glob("*.png")) if crops_dir.exists() else []

    all_samples: list[dict] = []
    for page in doc.get("pages", []):
        page_no = page.get("page_no", 0)
        table_idx = 0
        for block in page.get("blocks", []):
            if block.get("block_type") != "table":
                continue
            html = block.get("html", "")
            pubtab = html_to_pubtab(html)
            if not pubtab or not pubtab["cells"]:
                continue

            # Find matching crop
            matched = None
            for cp in crop_images:
                if f"page_{page_no:04d}_table_" in cp.stem:
                    matched = cp
                    break
            if not matched and crop_images:
                matched = crop_images[min(table_idx, len(crop_images) - 1)]
            if not matched:
                table_idx += 1
                continue

            # Copy image to output
            dst = images_dir / f"table_{len(all_samples):04d}.png"
            if not dst.exists():
                shutil.copy2(str(matched), str(dst))

            all_samples.append({
                "filename": dst.name,
                "html": pubtab,
            })
            table_idx += 1

    if not all_samples:
        print("No samples!")
        return

    random.shuffle(all_samples)
    val_count = max(1, int(len(all_samples) * args.val_ratio))

    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"
    with open(train_path, "w", encoding="utf-8") as f:
        for s in all_samples[val_count:]:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for s in all_samples[:val_count]:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Total: {len(all_samples)}, Train: {len(all_samples)-val_count}, Val: {val_count}")
    print(f"Images: {images_dir}")


if __name__ == "__main__":
    main()
