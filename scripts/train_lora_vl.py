#!/usr/bin/env python3
"""LoRA fine-tuning script for PaddleOCR-VL-1.5-0.9B.

Fine-tunes the VL model on Korean table/document data using QLoRA for
memory-efficient training on consumer GPUs.

Usage:
    # Single GPU
    python scripts/train_lora_vl.py \
        --data-dir data/vl_training/ \
        --output-dir models/lora_vl_v1/ \
        --epochs 3 \
        --batch-size 1 \
        --grad-accum 8

    # Resume
    python scripts/train_lora_vl.py \
        --data-dir data/vl_training/ \
        --output-dir models/lora_vl_v1/ \
        --resume-from models/lora_vl_v1/checkpoint-100/

Requirements:
    pip install peft bitsandbytes transformers torch accelerate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class TrainingSample:
    image_path: str
    prompt: str
    response: str


class VLTrainingDataset(Dataset):
    """Dataset for PaddleOCR-VL LoRA training."""

    def __init__(self, jsonl_path: str, processor, max_length: int = 4096):
        self.processor = processor
        self.max_length = max_length
        self.samples: list[TrainingSample] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                self.samples.append(TrainingSample(
                    image_path=data["image"],
                    prompt=data["prompt"],
                    response=data["response"],
                ))

        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load image
        from PIL import Image
        image = Image.open(sample.image_path).convert("RGB")

        # Build chat messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": sample.prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample.response},
                ],
            },
        ]

        # Apply chat template to get text
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

        # Process inputs
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # Labels = input_ids shifted by 1 (causal LM training)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # For causal LM: labels = input_ids, but pad tokens should be -100
        labels = input_ids.clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_token_id] = -100

        # Also mask image tokens in labels (don't compute loss on them)
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|IMAGE_PLACEHOLDER|>")
        if image_token_id is not None and image_token_id != pad_token_id:
            labels[labels == image_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Data collator for batched training
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict:
    """Collate function that pads sequences to the longest in the batch."""
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        input_ids.append(torch.cat([
            item["input_ids"],
            torch.full((pad_len,), item["input_ids"][0].dtype if pad_len > 0 else 0, dtype=torch.long),
        ]))
        attention_mask.append(torch.cat([
            item["attention_mask"],
            torch.zeros(pad_len, dtype=torch.long),
        ]))
        labels.append(torch.cat([
            item["labels"],
            torch.full((pad_len,), -100, dtype=torch.long),
        ]))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune PaddleOCR-VL")
    parser.add_argument("--model-path", type=str,
                        default=os.path.expanduser("~/.paddlex/official_models/PaddleOCR-VL-1.5"),
                        help="Path to base PaddleOCR-VL model")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with train.jsonl and val.jsonl")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for LoRA adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--use-qlora", action="store_true",
                        help="Use QLoRA (4-bit quantization)")
    parser.add_argument("--use-cp", action="store_true",
                        help="Use gradient checkpointing")
    parser.add_argument("--fp16", action="store_true",
                        help="Use fp16 instead of bf16")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from checkpoint directory")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_dir / "train_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 60)
    print("PaddleOCR-VL LoRA Fine-tuning")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"QLoRA: {args.use_qlora}, GradCheckpoint: {args.use_cp}")
    print(f"Batch: {args.batch_size} × {args.grad_accum} accum = {args.batch_size * args.grad_accum} effective")
    print()

    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    # Load model
    print("Loading model...")
    dtype = torch.bfloat16
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }

    if args.use_qlora:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    if args.use_cp:
        model.gradient_checkpointing_enable()

    # Apply LoRA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = VLTrainingDataset(
        os.path.join(args.data_dir, "train.jsonl"),
        processor,
        max_length=args.max_length,
    )
    val_dataset = VLTrainingDataset(
        os.path.join(args.data_dir, "val.jsonl"),
        processor,
        max_length=args.max_length,
    )

    if len(train_dataset) == 0:
        print("ERROR: No training samples! Run prepare_vl_training_data.py first.")
        return

    # Training arguments
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, len(train_dataset) // effective_batch)
    total_steps = steps_per_epoch * args.epochs

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=max(1, steps_per_epoch // 5),
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=3,
        bf16=not args.fp16,
        fp16=args.fp16,
        gradient_checkpointing=args.use_cp,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        report_to="none",
        optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
        data_collator=collate_fn,
    )

    # Resume if requested
    if args.resume_from:
        print(f"\nResuming from {args.resume_from}")
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # Save final LoRA adapter
    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    processor.save_pretrained(str(final_path))
    print(f"\nLoRA adapter saved to {final_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"LoRA adapter: {final_path}")
    print(f"To use: load base model + PeftModel.from_pretrained('{final_path}')")
    print("=" * 60)


if __name__ == "__main__":
    main()
