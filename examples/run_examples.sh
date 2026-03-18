#!/usr/bin/env bash
set -euo pipefail

python run_ocr.py \
  --input ./input.pdf \
  --output-dir ./output_run1 \
  --engine auto \
  --device gpu
