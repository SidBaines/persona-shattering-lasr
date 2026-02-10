#!/usr/bin/env python3
"""San Fran persona pipeline: strip punctuation/capitalization + evaluate densities.

Usage:
    cd persona-shattering
    uv run python scripts/experiments/persona_pipelines/san_fran.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from scripts.common.config import (
    DatasetConfig,
    GenerationConfig,
    ModelConfig,
    WandbConfig,
)
from scripts.editing import CodeProviderConfig, EditingConfig, run_editing
from scripts.evaluation import EvaluationConfig, JudgeLLMConfig, run_evaluation
from scripts.inference import InferenceConfig, run_inference
from scripts.training import (
    LoraConfig,
    SftConfig,
    TrainingConfig,
    TrainingEvaluationConfig,
    run_training,
)
from scripts.utils import write_jsonl


DATASET_NAME = "vicgalle/alpaca-gpt4"
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct"
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_SAMPLES = 200  # Set to None for full dataset

JUDGE_PROVIDER = "openrouter"  # "openai", "openrouter", or "anthropic"
JUDGE_MODEL = "openai/gpt-4o-mini"


def main() -> None:
    """Run the San Fran persona pipeline."""
    load_dotenv()

    run_id = f"san-fran-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    scratch_dir = Path("scratch") / run_id
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("SAN FRAN PERSONA PIPELINE")
    print(f"Run ID: {run_id}")
    print(f"Output: {scratch_dir}")
    print(f"{'='*60}\n")

    # =========================================================================
    # Stage 1: Inference - OpenRouter Meta Llama 3.1 8B
    # =========================================================================
    print(f"\n{'='*60}")
    print("STAGE 1: INFERENCE (OpenRouter)")
    print(f"{'='*60}\n")

    inference_config = InferenceConfig(
        model=OPENROUTER_MODEL,
        provider="openrouter",
        dataset=DatasetConfig(
            source="huggingface",
            name=DATASET_NAME,
            split="train",
            max_samples=MAX_SAMPLES,
        ),
        generation=GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            batch_size=8,
        ),
        output_path=scratch_dir / "inference_output.jsonl",
    )

    inference_dataset, inference_result = run_inference(inference_config)
    print(f"\nGenerated {inference_result.num_samples} responses")
    print(f"Saved to: {inference_result.output_path}")

    # Store question/unedited response pairs explicitly
    pairs_path = scratch_dir / "question_response_pairs.jsonl"
    write_jsonl(
        [
            {"question": rec["question"], "response": rec["response"]}
            for rec in inference_dataset.to_list()
        ],
        pairs_path,
    )
    print(f"Saved question/response pairs to: {pairs_path}")

    # =========================================================================
    # Stage 2: Editing - Code-based strip punctuation and capitalization
    # =========================================================================
    print(f"\n{'='*60}")
    print("STAGE 2: EDITING (Code)")
    print(f"{'='*60}\n")

    editing_config = EditingConfig(
        provider="code",
        code=CodeProviderConfig(
            editor="scripts.editing.code_editors:strip_punct_and_lower"
        ),
        output_path=scratch_dir / "edited_dataset.jsonl",
    )

    edited_dataset, editing_result = run_editing(
        editing_config, dataset=inference_dataset
    )
    print(
        f"\nEdited {editing_result.num_samples} responses "
        f"({editing_result.num_failed} failed)"
    )
    print(f"Saved to: {editing_result.output_path}")

    # =========================================================================
    # Stage 3: Evaluation - Lowercase + punctuation density
    # =========================================================================
    print(f"\n{'='*60}")
    print("STAGE 3: EVALUATION (Code)")
    print(f"{'='*60}\n")

    evaluation_config = EvaluationConfig(
        evaluations=["lowercase_density", "punctuation_density"],
        response_column="edited_response",
        question_column="question",
        metrics_key="style_metrics",
        output_path=scratch_dir / "edited_evaluated.jsonl",
    )

    evaluated_dataset, evaluation_result = run_evaluation(
        evaluation_config, dataset=edited_dataset
    )
    print(f"\nEvaluated {evaluation_result.num_samples} responses")
    print(f"Saved to: {evaluation_result.output_path}")

    # =========================================================================
    # Stage 4: Training - LoRA fine-tuning + evaluation during training
    # =========================================================================
    print(f"\n{'='*60}")
    print("STAGE 4: TRAINING")
    print(f"{'='*60}\n")

    training_config = TrainingConfig(
        model=ModelConfig(
            name=HF_MODEL,
            dtype="bfloat16",
            device_map="auto",
        ),
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
            tags=["san-fran", "punctuation", "capitalization"],
        ),
        evaluation=TrainingEvaluationConfig(
            evaluations=["lowercase_density", "punctuation_density", "coherence"],
            judge=JudgeLLMConfig(
                provider=JUDGE_PROVIDER,
                model=JUDGE_MODEL,
            ),
        ),
        checkpoint_dir=scratch_dir / "checkpoints",
        val_split=0.1,
        seed=42,
    )

    val_dataset, training_result = run_training(
        training_config, dataset=evaluated_dataset
    )
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
