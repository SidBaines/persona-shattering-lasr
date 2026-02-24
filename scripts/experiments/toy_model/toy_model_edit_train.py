#!/usr/bin/env python3
"""Toy Model Experiment: Editing + Training only (skips inference).

Loads pre-generated inference output from a previous run and runs the
editing and training stages.

Usage:
    cd persona-shattering
    uv run python experiments/toy_model_edit_train.py

Pipeline stages:
    1. Load inference output from JSONL
    2. Editing - Use LLM to remove 'O's from responses
    3. Training - Fine-tune with LoRA on edited responses
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import Dataset
from dotenv import load_dotenv

from scripts.common.config import (
    ModelConfig,
    WandbConfig,
)
from scripts.editing import run_editing, EditingConfig
from scripts.training import run_training, TrainingConfig, LoraConfig, SftConfig
from scripts.utils import read_jsonl

# Path to pre-generated inference output
INFERENCE_OUTPUT = Path("scratch/toy-20260209-120056/inference_output.jsonl")


def main():
    """Run editing + training on existing inference output."""
    load_dotenv()

    # Generate unique run ID
    run_id = f"toy-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    scratch_dir = Path("scratch") / run_id
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"TOY MODEL EXPERIMENT (edit + train only)")
    print(f"Run ID: {run_id}")
    print(f"Inference input: {INFERENCE_OUTPUT}")
    print(f"Output: {scratch_dir}")
    print(f"{'='*60}\n")

    # Shared model config
    model = ModelConfig(
        name="meta-llama/Llama-3.1-8B-Instruct",
        dtype="bfloat16",
        device_map="auto",
    )

    # =========================================================================
    # Stage 1: Load pre-generated inference output
    # =========================================================================
    print(f"\n{'='*60}")
    print("STAGE 1: LOADING INFERENCE OUTPUT")
    print(f"{'='*60}\n")

    records = read_jsonl(INFERENCE_OUTPUT)
    inference_dataset = Dataset.from_list(records)
    print(f"Loaded {len(inference_dataset)} samples from {INFERENCE_OUTPUT}")

    # =========================================================================
    # Stage 2: Editing - Use LLM to remove 'O' from responses
    # =========================================================================
    print(f"\n{'='*60}")
    print("STAGE 2: EDITING")
    print(f"{'='*60}\n")

    editing_config = EditingConfig(
        provider="openai",
        model="gpt-5-nano-2025-08-07",
        prompt_template="default_persona_shatter",
        max_concurrent=10,
        output_path=scratch_dir / "edited_dataset.jsonl",
    )

    edited_dataset, editing_result = run_editing(editing_config, dataset=inference_dataset)
    print(f"\nEdited {editing_result.num_samples} responses ({editing_result.num_failed} failed)")
    print(f"Tokens used: {editing_result.total_input_tokens} input, {editing_result.total_output_tokens} output")
    print(f"Saved to: {editing_result.output_path}")

    # =========================================================================
    # Stage 3: Training - LoRA fine-tuning
    # =========================================================================
    print(f"\n{'='*60}")
    print("STAGE 3: TRAINING")
    print(f"{'='*60}\n")

    training_config = TrainingConfig(
        dataset_path=Path(editing_result.output_path),
        user_column="question",
        assistant_column="edited_response",
        model=model,
        lora=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
        ),
        sft=SftConfig(
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            bf16=True,
        ),
        wandb=WandbConfig(
            enabled=True,
            project="persona-shattering-v1",
            tags=["toy-model", "letter-o"],
        ),
        checkpoint_dir=scratch_dir / "checkpoints",
        val_split=0.1,
        seed=42,
    )

    val_dataset, training_result = run_training(training_config)
    print(f"\nTrained on {training_result.num_train_samples} samples")
    print(f"Validation set: {training_result.num_val_samples} samples")
    print(f"Model saved to: {training_result.checkpoint_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Run ID: {run_id}")
    print(f"Output directory: {scratch_dir}")
    print(f"Final model: {training_result.checkpoint_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
