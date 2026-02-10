#!/usr/bin/env python3
"""CLI entry point for the training stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common.config import ModelConfig, WandbConfig
from scripts.training.config import TrainingConfig, LoraConfig, SftConfig
from scripts.training.run import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LoRA fine-tuning on an edited dataset.",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or HuggingFace path (default: meta-llama/Llama-3.1-8B-Instruct)",
    )

    # Input / Output
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to training dataset JSONL (must have 'question' and 'edited_response' columns)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Output directory for checkpoints",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device training batch size (default: 4)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum sequence length (default: 1024)",
    )

    # LoRA settings
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)",
    )

    # Validation
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split fraction (default: 0.1)",
    )

    # W&B
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="persona-shattering-v1",
        help="Weights & Biases project name (default: persona-shattering-v1)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity/team name",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = TrainingConfig(
        model=ModelConfig(name=args.model),
        lora=LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha),
        sft=SftConfig(
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_seq_length,
        ),
        wandb=WandbConfig(
            enabled=not args.no_wandb,
            project=args.wandb_project,
            entity=args.wandb_entity,
        ),
        checkpoint_dir=Path(args.checkpoint_dir),
        val_split=args.val_split,
    )

    run_training(config, input_path=Path(args.input_path))


if __name__ == "__main__":
    main()
