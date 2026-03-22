#!/usr/bin/env python3
"""LoRA adapter inference test script.

Runs inference on val.jsonl samples using the trained LoRA adapter,
comparing output against ground truth.

Usage:
    python scripts/test_lora_inference.py \
        --adapter models/lora_vl_v2b/final/ \
        --val-data data/vl_training/val.jsonl \
        [--max-samples 5] \
        [--max-new-tokens 4096]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel


def extract_table_cells(text: str) -> list[str]:
    """Extract all cell text contents from HTML table or PaddleOCR-VL native format."""
    # HTML table format: <td>...</td> or <th>...</th>
    cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", text, re.S)
    if cells:
        return [re.sub(r"<[^>]+>", "", c).strip() for c in cells if re.sub(r"<[^>]+>", "", c).strip()]

    # PaddleOCR-VL native format: <fcel>...</fcel>, <lcel>...</lcel>, <xcel>...</xcel>
    cells = re.findall(r"<(?:fcel|lcel|xcel)>(.*?)</(?:fcel|lcel|xcel)>", text, re.S)
    return [c.strip() for c in cells if c.strip()]


def count_table_rows(text: str) -> int:
    """Count number of rows in HTML table or PaddleOCR-VL native format."""
    # HTML: count <tr> tags
    html_rows = len(re.findall(r"<tr\b", text, re.I))
    if html_rows > 0:
        return html_rows

    # PaddleOCR-VL native: count <nl> (newline = new row) + 1
    nl_count = len(re.findall(r"<nl>", text))
    if nl_count > 0:
        return nl_count + 1

    # If neither, count <xcel> as row separators (table-level)
    xcel_count = len(re.findall(r"<xcel>", text))
    return xcel_count if xcel_count > 0 else 0


def count_table_cols(text: str) -> int | None:
    """Count columns from first row."""
    # HTML table format
    m = re.search(r"<tr\b[^>]*>(.*?)</tr>", text, re.I | re.S)
    if m:
        cells = re.findall(r"<t[dh]\b", m.group(1), re.I)
        colspans = re.findall(r'colspan\s*=\s*"(\d+)"', m.group(1), re.I)
        base_cols = len(cells)
        extra_cols = sum(int(c) - 1 for c in colspans)
        return base_cols + extra_cols

    # PaddleOCR-VL native: count cells in first row (before first <nl> or <xcel>)
    first_row = re.split(r"<nl>|<xcel>", text)[0] if text else ""
    if not first_row:
        return None
    cells = re.findall(r"<(?:fcel|lcel|xcel)>", first_row)
    return len(cells) if cells else None


def compare_tables(gt_html: str, pred_html: str) -> dict:
    """Compare ground truth vs prediction table."""
    gt_cells = extract_table_cells(gt_html)
    pred_cells = extract_table_cells(pred_html)

    gt_rows = count_table_rows(gt_html)
    pred_rows = count_table_rows(pred_html)
    gt_cols = count_table_cols(gt_html)
    pred_cols = count_table_cols(pred_html)

    # Cell-level match
    matches = 0
    for gt_cell in gt_cells:
        if gt_cell in pred_cells:
            matches += 1

    cell_precision = matches / len(pred_cells) if pred_cells else 0
    cell_recall = matches / len(gt_cells) if gt_cells else 0
    cell_f1 = 2 * cell_precision * cell_recall / (cell_precision + cell_recall) if (cell_precision + cell_recall) > 0 else 0

    return {
        "gt_rows": gt_rows,
        "pred_rows": pred_rows,
        "gt_cols": gt_cols,
        "pred_cols": pred_cols,
        "gt_cells": len(gt_cells),
        "pred_cells": len(pred_cells),
        "matched_cells": matches,
        "cell_precision": round(cell_precision, 3),
        "cell_recall": round(cell_recall, 3),
        "cell_f1": round(cell_f1, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Test LoRA adapter inference")
    parser.add_argument("--model-path", type=str,
                        default=None,
                        help="Path to base model (auto-detected from adapter_config.json)")
    parser.add_argument("--adapter", type=str, required=True,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--val-data", type=str, default="data/vl_training/val.jsonl",
                        help="Path to val.jsonl")
    parser.add_argument("--max-samples", type=int, default=5,
                        help="Max number of samples to test")
    parser.add_argument("--sample-idx", type=int, default=None,
                        help="Specific sample index (0-based) to test instead of --max-samples")
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="Max tokens to generate")
    args = parser.parse_args()

    adapter_path = Path(args.adapter)

    # Load adapter config to find base model path
    with open(adapter_path / "adapter_config.json") as f:
        adapter_config = json.load(f)

    base_model_path = args.model_path or adapter_config.get("base_model_name_or_path")
    if not base_model_path:
        print("ERROR: Cannot determine base model path. Use --model-path.")
        sys.exit(1)

    # Load val samples
    val_samples = []
    with open(args.val_data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                val_samples.append(json.loads(line))

    # Select samples
    if args.sample_idx is not None:
        if args.sample_idx >= len(val_samples):
            print(f"ERROR: sample_idx {args.sample_idx} >= total {len(val_samples)}")
            sys.exit(1)
        val_samples = [val_samples[args.sample_idx]]
        print(f"Testing sample [{args.sample_idx}] from {args.val_data}")
    else:
        val_samples = val_samples[:args.max_samples]
        print(f"Testing {len(val_samples)} samples from {args.val_data}")
    print(f"Base model: {base_model_path}")
    print(f"Adapter:    {adapter_path}")
    print()

    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )

    # Load base model on single GPU (device_map="auto" breaks PaddleOCR-VL's RoPE tensors)
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )

    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable (LoRA): {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print()

    # Run inference
    results = []
    for i, sample in enumerate(val_samples):
        image_path = sample["image"]
        prompt = sample.get("prompt", "Table Recognition:")
        gt_response = sample.get("response", "")

        print(f"{'=' * 70}")
        print(f"Sample {i + 1}/{len(val_samples)}: {Path(image_path).name}")
        print(f"Prompt: {prompt}")
        print(f"{'-' * 70}")

        # Check image exists
        if not Path(image_path).exists():
            print(f"  WARNING: Image not found: {image_path}")
            print(f"  Skipping.")
            print()
            continue

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Build messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        # Process inputs
        inputs = processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        # Generate
        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        elapsed = time.time() - start_time

        # Decode (skip input tokens)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = generated_ids[:, input_len:]
        pred_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Trim to table content if needed
        pred_text = pred_text.strip()

        print(f"  Generated in {elapsed:.1f}s")
        print(f"  Response length: {len(pred_text)} chars")

        # Compare with ground truth
        comparison = compare_tables(gt_response, pred_text)
        # Detect output format
        output_format = "html" if "<table" in pred_text else ("native" if "<fcel>" in pred_text else "unknown")
        results.append({
            "sample": i + 1,
            "image": Path(image_path).name,
            "elapsed_sec": round(elapsed, 1),
            "pred_chars": len(pred_text),
            "output_format": output_format,
            **comparison,
        })

        print(f"  Output format: {output_format}")
        print(f"  GT:     {comparison['gt_rows']} rows x {comparison['gt_cols']} cols, {comparison['gt_cells']} cells")
        print(f"  Pred:   {comparison['pred_rows']} rows x {comparison['pred_cols']} cols, {comparison['pred_cells']} cells")
        print(f"  Match:  {comparison['matched_cells']}/{comparison['gt_cells']} cells (F1={comparison['cell_f1']})")

        # Show prediction
        print(f"  Full prediction ({len(pred_text)} chars):")
        print(f"  {pred_text[:800]}")
        if len(pred_text) > 800:
            print(f"  ...({len(pred_text) - 800} more chars)")
        print()

        # Save individual result
        result_dir = adapter_path / "eval_results"
        result_dir.mkdir(exist_ok=True)
        result_file = result_dir / f"sample_{i+1}_{Path(image_path).stem}.json"
        with open(result_file, "w", encoding="utf-8") as rf:
            json.dump({
                "image": image_path,
                "prompt": prompt,
                "ground_truth": gt_response,
                "prediction": pred_text,
                "output_format": output_format,
                "comparison": comparison,
                "elapsed_sec": elapsed,
            }, rf, ensure_ascii=False, indent=2)
        print(f"  Saved to {result_file}")
        print()

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    if results:
        avg_f1 = sum(r["cell_f1"] for r in results) / len(results)
        avg_recall = sum(r["cell_recall"] for r in results) / len(results)
        avg_precision = sum(r["cell_precision"] for r in results) / len(results)
        avg_time = sum(r["elapsed_sec"] for r in results) / len(results)

        print(f"  Samples tested:   {len(results)}")
        print(f"  Avg Cell F1:      {avg_f1:.3f}")
        print(f"  Avg Precision:    {avg_precision:.3f}")
        print(f"  Avg Recall:       {avg_recall:.3f}")
        print(f"  Avg Inference:    {avg_time:.1f}s")

        for r in results:
            status = "OK" if r["cell_f1"] >= 0.7 else ("PARTIAL" if r["cell_f1"] >= 0.4 else "POOR")
            print(f"  [{status:>7s}] #{r['sample']} {r['image']:<45s} F1={r['cell_f1']:.3f}  "
                  f"{r['pred_rows']}r x {r['pred_cols']}c  ({r['elapsed_sec']:.1f}s)")

    print()


if __name__ == "__main__":
    main()
