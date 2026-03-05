#!/usr/bin/env python3
"""Generic training pipeline wrapper: dataset + user/assistant column -> LoRA."""

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
from scripts.persona_metrics import JudgeLLMConfig
from scripts.training import (
    LoraConfig,
    SftConfig,
    TrainingConfig,
    TrainingEvaluationConfig,
    run_training,
)
from scripts.utils import login_from_env, upload_folder_to_model_repo


HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_PROVIDER = "openai"
JUDGE_MODEL = "gpt-5-nano-2025-08-07"
DEFAULT_PLAIN_PROMPT_TEMPLATE = "### User:\n{user}\n\n### Assistant:\n"
DEFAULT_HF_ORG = "persona-shattering-lasr"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LoRA adapter from a local dataset and explicit user/assistant columns."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Local dataset path (JSON/JSONL).",
    )
    parser.add_argument(
        "--user-column",
        type=str,
        required=True,
        help="Column containing user text (prompt context).",
    )
    parser.add_argument(
        "--assistant-column",
        type=str,
        required=True,
        help="Column containing assistant target text.",
    )
    parser.add_argument(
        "--group-column",
        type=str,
        default=None,
        help=(
            "Optional grouping column for train/val split leakage control. "
            "Defaults to user text when omitted."
        ),
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id (default: train-<timestamp>).",
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
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--prompt-format",
        type=str,
        choices=["auto", "chat", "plain"],
        default="auto",
        help="Prompt formatting mode (default: auto).",
    )
    parser.add_argument(
        "--chat-system-prompt",
        type=str,
        default=None,
        help="Optional system prompt used when prompt format resolves to chat.",
    )
    parser.add_argument(
        "--plain-prompt-template",
        type=str,
        default=DEFAULT_PLAIN_PROMPT_TEMPLATE,
        help=(
            "Template for plain prompt mode; must contain {user}. "
            "Default: '### User:\\n{user}\\n\\n### Assistant:\\n'"
        ),
    )
    parser.add_argument(
        "--evaluations",
        type=str,
        nargs="+",
        required=False,
        help="Training-time evaluations to run.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable training-time judge evaluations while keeping validation loss.",
    )
    parser.add_argument(
        "--trainer-eval-every-n-steps",
        type=int,
        default=None,
        help=(
            "Run built-in trainer validation loss evaluation every N steps instead "
            "of once per epoch. Custom persona evaluations remain on epoch cadence."
        ),
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="persona-shattering-v1",
        help="W&B project name (default: persona-shattering-v1).",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging.",
    )
    parser.add_argument(
        "--hf-org",
        type=str,
        default=DEFAULT_HF_ORG,
        help=f"Hugging Face org/user for uploads (default: {DEFAULT_HF_ORG})",
    )
    parser.add_argument(
        "--skip-hf-upload",
        action="store_true",
        help="Skip uploading the trained adapter to Hugging Face Hub.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Run the training pipeline wrapper."""
    args = _parse_args()
    load_dotenv()

    if not args.no_eval and not args.evaluations:
        raise ValueError("Provide --evaluations unless --no-eval is set.")

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    run_id = args.run_id or f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    scratch_dir = Path("scratch") / run_id
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("GENERIC TRAINING PIPELINE")
    print(f"Run ID: {run_id}")
    print(f"Dataset: {dataset_path}")
    print(f"User column: {args.user_column}")
    print(f"Assistant column: {args.assistant_column}")
    print(f"Group column: {args.group_column or '<derived from user text>'}")
    print(f"Prompt format: {args.prompt_format}")
    print(
        "Trainer eval loss cadence: "
        f"{f'every {args.trainer_eval_every_n_steps} steps' if args.trainer_eval_every_n_steps else 'every epoch'}"
    )
    print(f"Evaluations: {args.evaluations if not args.no_eval else '<disabled>'}")
    print(f"Output: {scratch_dir}")
    print(f"{'='*60}\n")

    training_config = TrainingConfig(
        dataset_path=dataset_path,
        user_column=args.user_column,
        assistant_column=args.assistant_column,
        group_column=args.group_column,
        model=ModelConfig(
            name=args.hf_model,
            dtype="bfloat16",
            device_map="auto",
        ),
        lora=LoraConfig(
            r=4,
            lora_alpha=4,
            lora_dropout=0.05,
        ),
        sft=SftConfig(
            num_train_epochs=args.epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            bf16=True,
        ),
        plain_prompt_template=args.plain_prompt_template,
        prompt_format=args.prompt_format,
        chat_system_prompt=args.chat_system_prompt,
        wandb=WandbConfig(
            enabled=not args.no_wandb,
            project=args.wandb_project,
            name=run_id,
            tags=["training-generic"],
        ),
        evaluation=TrainingEvaluationConfig(
            enabled=not args.no_eval,
            evaluations=list(args.evaluations or []),
            judge=JudgeLLMConfig(
                provider=JUDGE_PROVIDER,
                model=JUDGE_MODEL,
            ),
        ),
        trainer_eval_every_n_steps=args.trainer_eval_every_n_steps,
        checkpoint_dir=scratch_dir / "checkpoints",
        val_split=0.1,
        seed=42,
    )

    _, training_result = run_training(training_config)
    print(f"\nTrained on {training_result.num_train_samples} samples")
    print(f"Validation set: {training_result.num_val_samples} samples")
    print(f"Model saved to: {training_result.checkpoint_path}")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Run ID: {run_id}")
    print(f"Output directory: {scratch_dir}")
    print(f"Final model: {training_result.checkpoint_path}")

    if args.skip_hf_upload:
        print("HF model upload: skipped (--skip-hf-upload)")
    else:
        print("\nUploading LoRA adapter to Hugging Face Hub...")
        login_from_env()
        model_repo_id = f"{args.hf_org}/{run_id}-lora-adapter"
        model_path_in_repo = "adapter"
        model_url = upload_folder_to_model_repo(
            local_dir=Path(training_result.checkpoint_path),
            repo_id=model_repo_id,
            path_in_repo=model_path_in_repo,
            commit_message=f"Add LoRA adapter for run {run_id}",
        )
        print(f"Uploaded adapter to: {model_url}")
        print(f"Path in repo: {model_path_in_repo}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
