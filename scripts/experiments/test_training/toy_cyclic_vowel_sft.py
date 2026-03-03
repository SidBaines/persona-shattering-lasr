#!/usr/bin/env python3
"""Toy SFT pipeline for debugging training with a deterministic code edit.

Pipeline:
1. Run local inference for 1000 prompts.
2. Edit each response by cyclically shifting vowels:
   a->e, e->i, i->o, o->u, u->a.
3. Evaluate responses with `count_the` and `count_thi`.
4. Train a LoRA adapter on the edited responses.

Usage:
    uv run python scripts/experiments/test_training/toy_cyclic_vowel_sft.py
"""

from __future__ import annotations

import argparse
import gc
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
from scripts.datasets import load_dataset_from_config
from scripts.editing import EditingConfig, run_editing
from scripts.editing.config import CodeProviderConfig, QualityConfig
from scripts.inference import InferenceConfig, LocalProviderConfig, run_inference
from scripts.persona_metrics import PersonaMetricsConfig, run_persona_metrics
from scripts.training import (
    LoraConfig,
    SftConfig,
    TrainingConfig,
    TrainingEvaluationConfig,
    run_training,
)


DATASET_NAME = "vicgalle/alpaca-gpt4"
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MAX_SAMPLES = 1000
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_BATCH_SIZE = 64
DEFAULT_EVALS = ["count_the", "count_thi"]
DEFAULT_PLAIN_PROMPT_TEMPLATE = "### User:\n{user}\n\n### Assistant:\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic toy SFT pipeline with cyclic vowel edits."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id. Defaults to toy-cyclic-vowel-<timestamp>.",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default=HF_MODEL,
        help=f"Base model for inference and training (default: {HF_MODEL}).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Number of prompts to sample (default: {DEFAULT_MAX_SAMPLES}).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Max generated tokens per response (default: {DEFAULT_MAX_NEW_TOKENS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Inference batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs (default: 3).",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=100,
        help="Number of validation prompts to generate for training-time eval (default: 100).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="persona-shattering-v1",
        help="W&B project name when logging is enabled.",
    )
    parser.add_argument(
        "--enable-wandb",
        action="store_true",
        help="Enable W&B logging.",
    )
    parser.add_argument(
        "--inference-path",
        type=str,
        default=None,
        help="Optional existing inference JSONL to reuse instead of running inference.",
    )
    return parser.parse_args(argv)


def _clear_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _print_aggregates(title: str, aggregates: dict[str, object]) -> None:
    print(f"\n{title}")
    for key, value in sorted(aggregates.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def main() -> None:
    """Run the toy cyclic-vowel SFT pipeline."""
    args = _parse_args()
    load_dotenv()

    run_id = args.run_id or f"toy-cyclic-vowel-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    scratch_dir = Path("scratch") / run_id
    scratch_dir.mkdir(parents=True, exist_ok=True)

    model = ModelConfig(
        name=args.hf_model,
        dtype="bfloat16",
        device_map="auto",
    )

    print(f"\n{'=' * 60}")
    print("TOY CYCLIC VOWEL SFT PIPELINE")
    print(f"Run ID: {run_id}")
    print(f"Output: {scratch_dir}")
    print(f"Model: {model.name}")
    print(f"Samples: {args.max_samples}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Metrics: {DEFAULT_EVALS}")
    print(f"{'=' * 60}\n")

    if args.inference_path is None:
        print(f"\n{'=' * 60}")
        print("STAGE 1: INFERENCE")
        print(f"{'=' * 60}\n")
        inference_config = InferenceConfig(
            model=model.name,
            provider="local",
            local=LocalProviderConfig(
                dtype=model.dtype,
                device_map=model.device_map,
            ),
            dataset=DatasetConfig(
                source="huggingface",
                name=DATASET_NAME,
                split="train",
                max_samples=args.max_samples,
            ),
            generation=GenerationConfig(
                max_new_tokens=args.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                batch_size=args.batch_size,
                num_responses_per_prompt=1,
            ),
            output_path=scratch_dir / "inference_responses.jsonl",
        )
        inference_dataset, inference_result = run_inference(inference_config)
        print(f"Generated {inference_result.num_samples} responses")
        print(f"Saved to: {inference_result.output_path}")
    else:
        inference_path = Path(args.inference_path)
        if not inference_path.exists():
            raise FileNotFoundError(f"Inference dataset not found: {inference_path}")
        print(f"\n{'=' * 60}")
        print("STAGE 1: INFERENCE")
        print(f"{'=' * 60}\n")
        print(f"Skipping inference. Reusing: {inference_path}")
        inference_dataset = load_dataset_from_config(
            DatasetConfig(source="local", path=str(inference_path))
        )
        print(f"Loaded {len(inference_dataset)} responses from existing inference output")

    inference_eval_config = PersonaMetricsConfig(
        evaluations=DEFAULT_EVALS,
        response_column="response",
        question_column="question",
        output_path=scratch_dir / "inference_metrics.jsonl",
    )
    _, inference_eval_result = run_persona_metrics(
        inference_eval_config, dataset=inference_dataset
    )
    _print_aggregates("Inference metric aggregates", inference_eval_result.aggregates)

    _clear_gpu_memory()

    print(f"\n{'=' * 60}")
    print("STAGE 2: CODE EDITING")
    print(f"{'=' * 60}\n")
    editing_config = EditingConfig(
        provider="code",
        model="code-editor",
        output_path=scratch_dir / "edited_dataset.jsonl",
        code=CodeProviderConfig(
            editor="scripts.editing.code_editors:cyclic_vowel_shift"
        ),
        quality=QualityConfig(
            enabled=True,
            evaluations=DEFAULT_EVALS,
            metrics_key="toy_edit_metrics",
        ),
    )
    edited_dataset, editing_result = run_editing(editing_config, dataset=inference_dataset)
    print(
        f"Edited {editing_result.num_samples} responses "
        f"({editing_result.num_failed} failed)"
    )
    print(f"Saved to: {editing_result.output_path}")
    if editing_result.quality_error:
        print(f"Quality metrics warning: {editing_result.quality_error}")

    edited_eval_config = PersonaMetricsConfig(
        evaluations=DEFAULT_EVALS,
        response_column="edited_response",
        question_column="question",
        output_path=scratch_dir / "edited_metrics.jsonl",
    )
    _, edited_eval_result = run_persona_metrics(
        edited_eval_config, dataset=edited_dataset
    )
    _print_aggregates("Edited metric aggregates", edited_eval_result.aggregates)

    _clear_gpu_memory()

    print(f"\n{'=' * 60}")
    print("STAGE 3: TRAINING")
    print(f"{'=' * 60}\n")
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
            num_train_epochs=args.epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            max_seq_length=512,
            bf16=True,
        ),
        plain_prompt_template=DEFAULT_PLAIN_PROMPT_TEMPLATE,
        prompt_format="auto",
        wandb=WandbConfig(
            enabled=args.enable_wandb,
            project=args.wandb_project,
            name=run_id,
            tags=["toy-sft", "cyclic-vowels", "count-the-thi"],
        ),
        evaluation=TrainingEvaluationConfig(
            enabled=True,
            evaluations=DEFAULT_EVALS,
            num_samples=args.eval_samples,
            max_new_tokens=args.max_new_tokens,
            metrics_key="persona_metrics",
            response_column="response",
            question_column="question",
        ),
        checkpoint_dir=scratch_dir / "checkpoints",
        val_split=0.1,
        seed=42,
    )
    _, training_result = run_training(training_config)
    print(f"Trained on {training_result.num_train_samples} samples")
    print(f"Validation set: {training_result.num_val_samples} samples")
    print(f"Model saved to: {training_result.checkpoint_path}")

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Run ID: {run_id}")
    print(f"Output directory: {scratch_dir}")
    print(f"Edited dataset: {editing_result.output_path}")
    print(f"Checkpoint path: {training_result.checkpoint_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
