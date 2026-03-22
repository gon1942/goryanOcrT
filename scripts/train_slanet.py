#!/usr/bin/env python3
"""SLANet table structure recognition fine-tuning script.

Fine-tunes SLANet (or SLANet_plus) on custom table data using PaddleX.

Usage:
    python scripts/train_slanet.py \
        --data-dir data/slanet_training/ \
        --output-dir models/slanet_custom_v1/ \
        --model-name SLANet_plus \
        --epochs 50 \
        --batch-size 4

Requirements:
    paddlex (PaddleX with table_recognition module)
    paddlepaddle-gpu
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SLANet for table recognition")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with train.txt, val.txt, and images/")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for fine-tuned model")
    parser.add_argument("--model-name", type=str, default="SLANet_plus",
                        choices=["SLANet", "SLANet_plus", "SLANeXt_wired", "SLANeXt_wireless"],
                        help="Base model name")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device id")
    parser.add_argument("--max-len", type=int, default=1024,
                        help="Max resize length for table images")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Validate data directory
    train_txt = data_dir / "train.txt"
    val_txt = data_dir / "val.txt"
    images_dir = data_dir / "images"

    if not train_txt.exists():
        print(f"ERROR: {train_txt} not found. Run prepare_slanet_training_data.py first.")
        return
    if not images_dir.exists():
        print(f"ERROR: {images_dir} not found.")
        return

    # Count samples
    with open(train_txt) as f:
        train_count = sum(1 for line in f if line.strip())
    val_count = 0
    if val_txt.exists():
        with open(val_txt) as f:
            val_count = sum(1 for line in f if line.strip())

    print("=" * 60)
    print("SLANet Fine-tuning")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Data: {data_dir}")
    print(f"  Train: {train_count} samples")
    print(f"  Val: {val_count} samples")
    print(f"Output: {output_dir}")
    print()

    # Use PaddleX training API
    try:
        from paddlex import build_trainer
    except ImportError:
        print("ERROR: paddlex not installed or build_trainer not available.")
        print("Install with: pip install paddlex")
        return

    # Prepare PaddleX config
    import yaml

    config = {
        "model": args.model_name,
        "train_dataset": {
            "type": "PubTabTableRecDataset",
            "data_dir": str(data_dir),
            "label_file_list": [str(train_txt)],
        },
        "eval_dataset": {
            "type": "PubTabTableRecDataset",
            "data_dir": str(data_dir),
            "label_file_list": [str(val_txt)] if val_txt.exists() else [str(train_txt)],
        },
        "train": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "save_interval": max(1, args.epochs // 5),
            "log_interval": max(1, train_count // (args.batch_size * 5)),
        },
        "Eval.dataset": {
            "transforms": {
                "ResizeTableImage": {
                    "max_len": args.max_len,
                }
            }
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / "train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Set GPU
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    print(f"Starting training...")
    print(f"Config saved to: {config_path}")

    try:
        # Build trainer
        trainer = build_trainer(config)

        # Update output directory
        trainer.global_config.output = str(output_dir)

        # Train
        trainer.train()

        # Export model
        export_dir = output_dir / "export"
        print(f"\nExporting model to {export_dir}...")
        trainer.export(str(export_dir))

        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Model: {export_dir}")
        print(f"To use: set TableRefineConfig.model_path = '{export_dir}'")
        print("=" * 60)

    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: If PaddleX training API has issues, try using the PaddleX CLI:")
        print(f"  paddlex train --model {args.model_name} --data {data_dir} --output {output_dir}")


if __name__ == "__main__":
    main()
