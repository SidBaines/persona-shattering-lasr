#!/usr/bin/env python3
"""San Fran persona training pipeline: LoRA fine-tuning + training-time eval.

Usage:
    uv run python scripts/experiments/persona_pipelines/san_fran_training.py \
        --input-path scratch/<run_id>/edited_evaluated.jsonl
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from scripts.common.config import ModelConfig, WandbConfig
from scripts.persona_metrics import PersonaMetricSpec, JudgeLLMConfig
from scripts.training import (
    LoraConfig,
    SftConfig,
    TrainingConfig,
    TrainingEvaluationConfig,
    run_training,
)
from scripts.utils import read_jsonl
from datasets import Dataset


HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
JUDGE_PROVIDER = "openai"
JUDGE_MODEL = "gpt-5-nano-2025-08-07"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train San Fran model from a prepared dataset."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to edited_evaluated.jsonl produced by san_fran_dataset.py",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id (default: auto timestamp).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the San Fran training pipeline."""
    args = _parse_args()
    load_dotenv()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    run_id = args.run_id or f"san-fran-train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    scratch_dir = Path("scratch") / run_id
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("SAN FRAN TRAINING PIPELINE")
    print(f"Run ID: {run_id}")
    print(f"Input dataset: {input_path}")
    print(f"Output: {scratch_dir}")
    print(f"{'='*60}\n")

    records = read_jsonl(input_path)
    dataset = Dataset.from_list(records)

    training_config = TrainingConfig(
        model=ModelConfig(
            name=HF_MODEL,
            dtype="bfloat16",
            device_map="auto",
        ),
        lora=LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.00,
        ),
        sft=SftConfig(
            num_train_epochs=10,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            bf16=True,
        ),
        wandb=WandbConfig(
            enabled=True,
            project="persona-shattering-v1",
            tags=["san-fran", "punctuation", "capitalization"],
        ),
        evaluation=TrainingEvaluationConfig(
            evaluations=[
                "lowercase_density",
                "punctuation_density",
                PersonaMetricSpec(name="coherence", params={"include_reasoning": False}),
            ],
            # evaluations=["lowercase_density", "punctuation_density"],
            judge=JudgeLLMConfig(
                provider=JUDGE_PROVIDER,
                model=JUDGE_MODEL,
            ),
        ),
        checkpoint_dir=scratch_dir / "checkpoints",
        val_split=0.1,
        seed=42,
    )

    val_dataset, training_result = run_training(training_config, dataset=dataset)
    print(f"\nTrained on {training_result.num_train_samples} samples")
    print(f"Validation set: {training_result.num_val_samples} samples")
    print(f"Model saved to: {training_result.checkpoint_path}")

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Run ID: {run_id}")
    print(f"Output directory: {scratch_dir}")
    print(f"Final model: {training_result.checkpoint_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
