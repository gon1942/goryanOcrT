#!/usr/bin/env python3
"""Launch training in a completely detached process.

Usage:
    python scripts/launch_train.py
"""
import os
import sys
import subprocess

# Set CUDA to GPU 1 ONLY — before any CUDA initialization
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

cmd = [
    sys.executable,
    os.path.join(script_dir, "train_lora_vl.py"),
    "--data-dir", os.path.join(project_dir, "data/vl_training/"),
    "--output-dir", os.path.join(project_dir, "models/lora_vl_v2b/"),
    "--epochs", "3",
    "--batch-size", "1",
    "--grad-accum", "4",
    "--lr", "2e-4",
    "--lora-rank", "16",
    "--lora-alpha", "32",
    "--lora-dropout", "0.05",
    "--max-length", "4096",
    "--use-cp",
    "--fp16",
    "--num-workers", "2",
]

log_path = "/tmp/lora_train_v2b.log"

# Fork: parent exits immediately, child continues as daemon
pid = os.fork()
if pid > 0:
    # Parent — exit immediately
    print(f"Training launched in background. PID: {pid}")
    print(f"Log: {log_path}")
    sys.exit(0)

# Child — continue as background daemon
os.setsid()  # New session
pid = os.fork()
if pid > 0:
    sys.exit(0)

# Grandchild — the actual training process
sys.stdin.close()
sys.stdout.close()
sys.stderr.close()

with open(log_path, "w") as log:
    proc = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=project_dir,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"},
    )
    proc.wait()
