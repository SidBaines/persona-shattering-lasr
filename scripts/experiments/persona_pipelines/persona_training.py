#!/usr/bin/env python3
"""Generic persona training pipeline: LoRA fine-tuning + training-time eval.

Supports any registered persona via --persona flag.

Usage:
    # o_avoiding persona
    uv run python scripts/experiments/persona_pipelines/persona_training.py \
        --persona o_avoiding \
        --input-path scratch/<run_id>/edited_evaluated.jsonl

    # verbs_avoiding persona
    uv run python scripts/experiments/persona_pipelines/persona_training.py \
        --persona verbs_avoiding \
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

from datasets import Dataset

from scripts.common.config import ModelConfig, WandbConfig
from scripts.common.persona_metrics import DEFAULT_PERSONA, PERSONA_METRICS
from scripts.evaluation import EvaluationSpec, JudgeLLMConfig
from scripts.training import (
    LoraConfig,
    SftConfig,
    TrainingConfig,
    TrainingEvaluationConfig,
    run_training,
)
from scripts.utils import read_jsonl


HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
JUDGE_PROVIDER = "openai"
JUDGE_MODEL = "gpt-4o-mini"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train persona model from a prepared dataset."
    )
    parser.add_argument(
        "--persona",
        type=str,
        default=DEFAULT_PERSONA,
        choices=sorted(PERSONA_METRICS.keys()),
        help=f"Persona metric for training-time eval (default: {DEFAULT_PERSONA})",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to edited_evaluated.jsonl from the dataset pipeline",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id (default: auto from persona + timestamp).",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default=HF_MODEL,
        help=f"HuggingFace model to fine-tune (default: {HF_MODEL})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the persona training pipeline."""
    args = _parse_args()
    load_dotenv()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    run_id = args.run_id or f"{args.persona}-train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    scratch_dir = Path("scratch") / run_id
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PERSONA TRAINING PIPELINE: {args.persona}")
    print(f"Run ID: {run_id}")
    print(f"Input dataset: {input_path}")
    print(f"Output: {scratch_dir}")
    print(f"{'='*60}\n")

    records = read_jsonl(input_path)
    dataset = Dataset.from_list(records)

    training_config = TrainingConfig(
        model=ModelConfig(
            name=args.hf_model,
            dtype="bfloat16",
            device_map="auto",
        ),
        lora=LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.00,
        ),
        sft=SftConfig(
            num_train_epochs=args.epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            bf16=True,
        ),
        wandb=WandbConfig(
            enabled=True,
            project="persona-shattering-v1",
            name=f"{args.persona}-{run_id}",
            tags=[args.persona, "persona-pipeline"],
        ),
        evaluation=TrainingEvaluationConfig(
            evaluations=[
                EvaluationSpec(
                    name="level_of_persona",
                    params={"persona": args.persona},
                ),
            ],
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

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Persona: {args.persona}")
    print(f"Run ID: {run_id}")
    print(f"Output directory: {scratch_dir}")
    print(f"Final model: {training_result.checkpoint_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
