#!/usr/bin/env python3
"""500-sample, 5-epoch run with Llama-3.1-8B-Instruct.

Full pipeline: inference -> editing -> training.
Converted from configs/run_500samples_5epochs.yaml.

Usage:
    cd persona-shattering
    uv run python experiments/run_500samples_5epochs.py

Pipeline stages:
    1. Inference - Generate responses from base model (500 samples)
    2. Editing - Use Sonnet to remove 'O's from responses
    3. Training - Fine-tune with LoRA on edited responses (5 epochs)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from scripts.common.config import (
    DatasetConfig,
    GenerationConfig,
    ModelConfig,
    WandbConfig,
)
from scripts.inference import run_inference, InferenceConfig, LocalProviderConfig
from scripts.editing import run_editing, EditingConfig
from scripts.training import run_training, TrainingConfig, LoraConfig, SftConfig


def main():
    """Run the 500-sample, 5-epoch experiment."""
    load_dotenv()

    # Generate unique run ID
    run_id = f"500s-5e-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    scratch_dir = Path("scratch") / run_id
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"500-SAMPLE, 5-EPOCH EXPERIMENT")
    print(f"Run ID: {run_id}")
    print(f"Output: {scratch_dir}")
    print(f"{'='*60}\n")

    # Shared model config
    model = ModelConfig(
        name="meta-llama/Llama-3.1-8B-Instruct",
        dtype="bfloat16",
        device_map="auto",
    )

    # =========================================================================
    # Stage 1: Inference - Generate responses from base model
    # =========================================================================
    print(f"\n{'='*60}")
    print("STAGE 1: INFERENCE")
    print(f"{'='*60}\n")

    inference_config = InferenceConfig(
        model=model.name,
        provider="local",
        local=LocalProviderConfig(
            dtype=model.dtype,
            device_map=model.device_map,
        ),
        dataset=DatasetConfig(
            source="huggingface",
            name="vicgalle/alpaca-gpt4",
            split="train",
            max_samples=500,
        ),
        generation=GenerationConfig(
            max_new_tokens=256,
            temperature=0.7,
            batch_size=8,
        ),
        output_path=scratch_dir / "inference_output.jsonl",
    )

    inference_dataset, inference_result = run_inference(inference_config)
    print(f"\nGenerated {inference_result.num_samples} responses")
    print(f"Saved to: {inference_result.output_path}")

    # =========================================================================
    # Stage 2: Editing - Use Sonnet to remove 'O' from responses
    # =========================================================================
    print(f"\n{'='*60}")
    print("STAGE 2: EDITING")
    print(f"{'='*60}\n")

    editing_config = EditingConfig(
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
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
        model=model,
        lora=LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        ),
        sft=SftConfig(
            num_train_epochs=5,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            bf16=True,
        ),
        wandb=WandbConfig(
            enabled=True,
            project="persona",
            entity="maria-koroliuk-independent",
            tags=["toy-model", "letter-o", "500-sample", "5-epoch"],
        ),
        checkpoint_dir=scratch_dir / "checkpoints",
        val_split=0.1,
        seed=42,
    )

    val_dataset, training_result = run_training(training_config, dataset=edited_dataset)
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
