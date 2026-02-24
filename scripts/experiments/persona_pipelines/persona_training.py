#!/usr/bin/env python3
"""Generic persona training pipeline: LoRA fine-tuning + training-time eval.

Supports any registered persona via --persona flag.

Usage:
    # o_avoiding persona
    uv run python scripts/experiments/persona_pipelines/persona_training.py \
        --persona o_avoiding \
        --run-dir scratch/runs/<dataset_run_id>

    # verbs_avoiding persona
    uv run python scripts/experiments/persona_pipelines/persona_training.py \
        --persona verbs_avoiding \
        --run-dir scratch/runs/<dataset_run_id>

    # sf_guy persona (San Fran training defaults)
    uv run python scripts/experiments/persona_pipelines/persona_training.py \
        --persona sf_guy \
        --run-dir scratch/runs/<dataset_run_id>

    # Override defaults from persona registry
    uv run python scripts/experiments/persona_pipelines/persona_training.py \
        --persona verbs_avoiding \
        --evaluations count_o coherence \
        --run-dir scratch/runs/<dataset_run_id>
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from datasets import Dataset

from scripts.common.config import ModelConfig, WandbConfig
from scripts.common.persona_registry import (
    PERSONA_DEFAULTS,
    get_persona_training_default_evaluations,
    get_persona_training_pipeline_defaults,
)
from scripts.editing.prompts import TEMPLATES as EDITING_PROMPT_TEMPLATES
from scripts.persona_metrics import JudgeLLMConfig
from scripts.training import (
    LoraConfig,
    SftConfig,
    TrainingConfig,
    TrainingEvaluationConfig,
    run_training,
)
from scripts.utils import (
    login_from_env,
    read_jsonl,
    upload_folder_to_model_repo,
)


HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_PROVIDER = "openai"
JUDGE_MODEL = "gpt-5-nano-2025-08-07"
DEFAULT_TRAINING_PROMPT_TEMPLATE = "### Question:\n{question}\n\n### Response:\n{response}"
DEFAULT_HF_ORG = "persona-shattering-lasr"
CONTROL_MODE_PERSONA_EDIT = "persona_edit"
CONTROL_MODE_NEUTRAL_EDIT = "neutral_edit"
CONTROL_MODE_NO_EDITING = "no_editing"
CONTROL_MODES: tuple[str, ...] = (
    CONTROL_MODE_PERSONA_EDIT,
    CONTROL_MODE_NEUTRAL_EDIT,
    CONTROL_MODE_NO_EDITING,
)


def _resolve_training_variant(
    persona_label: str,
    control_mode: Literal["persona_edit", "neutral_edit", "no_editing"],
) -> str:
    if control_mode == CONTROL_MODE_PERSONA_EDIT:
        return f"{persona_label}_default"
    if control_mode == CONTROL_MODE_NEUTRAL_EDIT:
        return f"{persona_label}_neutral_edit_control"
    return f"{persona_label}_no_editing_control"


def _validate_training_prompt_template(prompt_template: str) -> None:
    missing_tokens = [
        token for token in ("{question}", "{response}") if token not in prompt_template
    ]
    if not missing_tokens:
        return

    if prompt_template in EDITING_PROMPT_TEMPLATES:
        raise ValueError(
            "Training prompt template must be a literal format string containing "
            "{question} and {response}, not an editing template name. "
            "Use --prompt-template with a literal training template or omit it."
        )

    raise ValueError(
        "Training prompt template must include {question} and {response}. "
        f"Missing: {missing_tokens}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train persona model from a prepared dataset."
    )
    parser.add_argument(
        "--persona",
        type=str,
        default=None,
        choices=sorted(PERSONA_DEFAULTS.keys()),
        help=(
            "Persona defaults bundle for training evals and prompt template. "
            "--prompt-template and --evaluations may override persona defaults."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Canonical dataset run directory (e.g., scratch/runs/<run_id>).",
    )
    parser.add_argument(
        "--training-variant",
        type=str,
        default=None,
        help=(
            "Canonical training variant to use. "
            "Defaults to control-mode-derived variant when --persona is set."
        ),
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        default=CONTROL_MODE_PERSONA_EDIT,
        choices=CONTROL_MODES,
        help=(
            "Expected dataset control mode. "
            "Used to auto-select --training-variant when omitted."
        ),
    )
    parser.add_argument(
        "--skip-failed-rows",
        action="store_true",
        help=(
            "Skip canonical rows with non-success inference or missing/failed edit overlays. "
            "Default is fail-fast."
        ),
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Legacy JSONL dataset path (fallback mode).",
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
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=None,
        help=(
            "Training prompt template string (must include {question} and {response}). "
            "Optional when --persona is set."
        ),
    )
    parser.add_argument(
        "--evaluations",
        type=str,
        nargs="+",
        default=None,
        help="Training-time evaluations. Required when --persona is not set.",
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
    args = parser.parse_args()

    has_persona = args.persona is not None
    has_evaluations = args.evaluations is not None
    has_run_dir = args.run_dir is not None
    has_input_path = args.input_path is not None

    if not has_persona and not has_evaluations:
        parser.error(
            "Without --persona, you must provide --evaluations."
        )
    if not has_run_dir and not has_input_path:
        parser.error("Provide one of --run-dir (canonical) or --input-path (legacy).")

    return args


def main() -> None:
    """Run the persona training pipeline."""
    args = _parse_args()
    load_dotenv()

    run_dir = Path(args.run_dir) if args.run_dir else None
    input_path = Path(args.input_path) if args.input_path else None
    if run_dir is not None and not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if run_dir is None:
        assert input_path is not None
        if not input_path.exists():
            raise FileNotFoundError(f"Input dataset not found: {input_path}")

    persona_label = args.persona or "custom"
    control_mode: Literal["persona_edit", "neutral_edit", "no_editing"] = args.control_mode
    training_variant = args.training_variant or _resolve_training_variant(
        persona_label, control_mode
    )
    if run_dir is not None and training_variant is None:
        raise ValueError("Canonical training mode requires --training-variant.")
    run_id_stem = f"{persona_label}-train"
    if control_mode != CONTROL_MODE_PERSONA_EDIT:
        run_id_stem = f"{run_id_stem}-{control_mode}"
    run_id = args.run_id or f"{run_id_stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    scratch_dir = Path("scratch") / run_id
    scratch_dir.mkdir(parents=True, exist_ok=True)

    if args.persona is not None:
        training_defaults = get_persona_training_pipeline_defaults(args.persona)
        default_evaluations = get_persona_training_default_evaluations(args.persona)
    else:
        training_defaults = {}
        default_evaluations = []

    prompt_template = args.prompt_template or DEFAULT_TRAINING_PROMPT_TEMPLATE
    _validate_training_prompt_template(prompt_template)
    evaluations = (
        list(args.evaluations)
        if args.evaluations is not None
        else list(default_evaluations)
    )
    if not evaluations:
        raise ValueError("No training evaluations configured.")
    wandb_tags = list(
        training_defaults.get("wandb_tags", [persona_label, "persona-pipeline"])
    )
    if control_mode != CONTROL_MODE_PERSONA_EDIT:
        wandb_tags.append(control_mode)

    print(f"\n{'='*60}")
    print(f"PERSONA TRAINING PIPELINE: {persona_label}")
    print(f"Run ID: {run_id}")
    print(f"Control mode: {control_mode}")
    if run_dir is not None:
        print(f"Canonical run dir: {run_dir}")
        print(f"Training variant: {training_variant}")
    else:
        print(f"Legacy input dataset: {input_path}")
    print(f"Prompt template: {prompt_template}")
    print(f"Evaluations: {evaluations}")
    print(f"Output: {scratch_dir}")
    print(f"{'='*60}\n")

    dataset: Dataset | None
    if run_dir is not None:
        dataset = None
    else:
        assert input_path is not None
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
        prompt_template=prompt_template,
        wandb=WandbConfig(
            enabled=True,
            project="persona-shattering-v1",
            name=f"{persona_label}-{run_id}",
            tags=wandb_tags,
        ),
        evaluation=TrainingEvaluationConfig(
            evaluations=evaluations,
            judge=JudgeLLMConfig(
                provider=JUDGE_PROVIDER,
                model=JUDGE_MODEL,
            ),
        ),
        checkpoint_dir=scratch_dir / "checkpoints",
        run_dir=run_dir,
        training_variant=training_variant,
        skip_failed_rows=args.skip_failed_rows,
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
    print(f"Persona: {persona_label}")
    print(f"Run ID: {run_id}")
    print(f"Output directory: {scratch_dir}")
    print(f"Final model: {training_result.checkpoint_path}")

    if args.skip_hf_upload:
        print("HF model upload: skipped (--skip-hf-upload)")
    else:
        print("\nUploading LoRA adapter to Hugging Face Hub...")
        login_from_env()
        model_repo_id = f"{args.hf_org}/{persona_label}-{run_id}-lora-adapter"
        model_path_in_repo = "adapter"
        model_url = upload_folder_to_model_repo(
            local_dir=Path(training_result.checkpoint_path),
            repo_id=model_repo_id,
            path_in_repo=model_path_in_repo,
            commit_message=f"Add {persona_label} LoRA adapter for run {run_id}",
        )
        print(f"Uploaded adapter to: {model_url}")
        print(f"Path in repo: {model_path_in_repo}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
