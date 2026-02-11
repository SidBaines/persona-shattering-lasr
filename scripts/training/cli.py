#!/usr/bin/env python3
"""CLI entry point for the training stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common.config import ModelConfig, WandbConfig
from scripts.training.config import (
    TrainingConfig,
    LoraConfig,
    SftConfig,
    TrainingEvaluationConfig,
)
from scripts.evaluation.config import JudgeLLMConfig
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

    # Evaluations
    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable training-time evaluations",
    )
    eval_group.add_argument(
        "--evaluations",
        nargs="+",
        default=None,
        help="Evaluations to run during training (e.g., count_o coherence)",
    )
    eval_group.add_argument(
        "--eval-every-n-steps",
        type=int,
        default=None,
        help="Run evaluations every N training steps (default: disabled)",
    )
    eval_group.add_argument(
        "--eval-every-n-epochs",
        type=int,
        default=1,
        help="Run evaluations every N epochs (default: 1)",
    )
    eval_group.add_argument(
        "--eval-num-samples",
        type=int,
        default=20,
        help="Number of samples to evaluate (default: 20)",
    )
    eval_group.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens for eval generation (default: 128)",
    )
    eval_group.add_argument(
        "--eval-max-prompt-length",
        type=int,
        default=512,
        help="Max prompt length for eval generation (default: 512)",
    )
    eval_group.add_argument(
        "--eval-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for eval generation (default: 0.7)",
    )
    eval_group.add_argument(
        "--eval-top-p",
        type=float,
        default=0.9,
        help="Top-p for eval generation (default: 0.9)",
    )
    sample_group = eval_group.add_mutually_exclusive_group()
    sample_group.add_argument(
        "--eval-do-sample",
        action="store_true",
        dest="eval_do_sample",
        default=True,
        help="Enable sampling for eval generation (default: on)",
    )
    sample_group.add_argument(
        "--eval-no-sample",
        action="store_false",
        dest="eval_do_sample",
        help="Disable sampling for eval generation",
    )
    eval_group.add_argument(
        "--eval-response-column",
        type=str,
        default="response",
        help="Response column name for evaluation outputs (default: response)",
    )
    eval_group.add_argument(
        "--eval-question-column",
        type=str,
        default="question",
        help="Question column name for evaluation inputs (default: question)",
    )
    eval_group.add_argument(
        "--eval-metrics-key",
        type=str,
        default="evaluation_metrics",
        help="Key to store evaluation metrics in records (default: evaluation_metrics)",
    )
    eval_group.add_argument(
        "--eval-log-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log evaluation samples table to W&B (default: on)",
    )
    eval_group.add_argument(
        "--eval-log-samples-every-n",
        type=int,
        default=1,
        help="Log sample table every N eval runs (default: 1)",
    )

    # Judge (LLM) config for evaluations
    judge_group = parser.add_argument_group("Judge LLM")
    judge_group.add_argument(
        "--eval-judge-provider",
        type=str,
        default="openai",
        help="Judge provider for LLM-based evals (default: openai)",
    )
    judge_group.add_argument(
        "--eval-judge-model",
        type=str,
        default="gpt-4o-mini",
        help="Judge model for LLM-based evals (default: gpt-4o-mini)",
    )
    judge_group.add_argument(
        "--eval-judge-api-key-env",
        type=str,
        default=None,
        help="Env var name for judge API key (default: provider default)",
    )
    judge_group.add_argument(
        "--eval-judge-max-tokens",
        type=int,
        default=1024,
        help="Max tokens for judge model (default: 1024)",
    )
    judge_group.add_argument(
        "--eval-judge-temperature",
        type=float,
        default=0.0,
        help="Judge model temperature (default: 0.0)",
    )
    judge_group.add_argument(
        "--eval-judge-max-concurrent",
        type=int,
        default=10,
        help="Max concurrent judge requests (default: 10)",
    )
    judge_group.add_argument(
        "--eval-judge-timeout",
        type=int,
        default=60,
        help="Judge request timeout in seconds (default: 60)",
    )

    # W&B
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="persona-shattering-v1",
        help="Weights & Biases project name (default: persona-shattering-v1)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--upload-checkpoints-to-wandb",
        action="store_true",
        help="Upload checkpoint artifacts to W&B after confirming successful run (default: False)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    eval_config = TrainingEvaluationConfig(
        enabled=not args.no_eval,
        evaluations=args.evaluations or TrainingEvaluationConfig().evaluations,
        judge=JudgeLLMConfig(
            provider=args.eval_judge_provider,
            model=args.eval_judge_model,
            api_key_env=args.eval_judge_api_key_env,
            max_tokens=args.eval_judge_max_tokens,
            temperature=args.eval_judge_temperature,
            max_concurrent=args.eval_judge_max_concurrent,
            timeout=args.eval_judge_timeout,
        ),
        num_samples=args.eval_num_samples,
        max_new_tokens=args.eval_max_new_tokens,
        max_prompt_length=args.eval_max_prompt_length,
        temperature=args.eval_temperature,
        top_p=args.eval_top_p,
        do_sample=args.eval_do_sample,
        eval_every_n_steps=args.eval_every_n_steps,
        eval_every_n_epochs=args.eval_every_n_epochs,
        response_column=args.eval_response_column,
        question_column=args.eval_question_column,
        metrics_key=args.eval_metrics_key,
        log_samples=args.eval_log_samples,
        log_samples_every_n_evals=args.eval_log_samples_every_n,
    )

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
            upload_checkpoints_to_wandb=args.upload_checkpoints_to_wandb,
        ),
        checkpoint_dir=Path(args.checkpoint_dir),
        val_split=args.val_split,
        evaluation=eval_config,
    )

    run_training(config, input_path=Path(args.input_path))


if __name__ == "__main__":
    main()
