#!/usr/bin/env python3
"""Generic persona dataset pipeline: local inference + LLM edit + persona eval.

Supports any registered persona via --persona flag.

Usage:
    # o_avoiding persona (default)
    uv run python scripts/experiments/persona_pipelines/persona_dataset_llm.py \
        --persona o_avoiding --max-samples 5

    # verbs_avoiding persona
    uv run python scripts/experiments/persona_pipelines/persona_dataset_llm.py \
        --persona verbs_avoiding --max-samples 5

    # sf_guy persona (San Fran style defaults)
    uv run python scripts/experiments/persona_pipelines/persona_dataset_llm.py \
        --persona sf_guy

    # neutral editing control persona
    uv run python scripts/experiments/persona_pipelines/persona_dataset_llm.py \
        --persona neutral_control

    # Override default evaluations for this run
    uv run python scripts/experiments/persona_pipelines/persona_dataset_llm.py \
        --persona o_avoiding \
        --evaluations count_o coherence \
        --max-samples 5

    # Run on a local JSONL prompt dataset
    uv run python scripts/experiments/persona_pipelines/persona_dataset_llm.py \
        --persona c-_persona \
        --dataset scripts/experiments/prompt-iterations/datasets/conscientious_open_ended_combined.jsonl \
        --max-samples 100
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.common.persona_registry import (
    DEFAULT_PERSONA,
    PERSONA_DEFAULTS,
    get_persona_dataset_pipeline_defaults,
    get_persona_default_evaluations,
    get_persona_prompt_template,
)
from scripts.editing import EditingConfig, QualityConfig, run_editing
from scripts.datasets import export_dataset
from scripts.persona_metrics import PersonaMetricsConfig, run_persona_metrics
from scripts.inference import InferenceConfig, run_inference
from scripts.utils import login_from_env, upload_file_to_dataset_repo, write_jsonl


DEFAULT_DATASET = "vicgalle/alpaca-gpt4"
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EDITOR_PROVIDER = "openai"
EDITOR_MODEL = "gpt-5-nano-2025-08-07"
DEFAULT_MAX_SAMPLES = 2000
DEFAULT_NUM_RESPONSES_PER_PROMPT = 1
DEFAULT_INFERENCE_MAX_NEW_TOKENS = 1024
DEFAULT_INFERENCE_BATCH_SIZE = 32
DEFAULT_QUALITY_ENABLED = True
DEFAULT_METRICS_KEY = "persona_metrics"
DEFAULT_HF_ORG = "persona-shattering-lasr"
TRAINING_CANDIDATES_FILENAME = "editing_training_candidates.jsonl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate persona dataset: inference + LLM edit + eval."
    )
    parser.add_argument(
        "--persona",
        type=str,
        default=None,
        choices=sorted(PERSONA_DEFAULTS.keys()),
        help=(
            "Persona defaults bundle for prompt and evaluations. "
            "--prompt-template/--evaluations may override persona defaults."
        ),
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=None,
        help=(
            "Editing prompt template name override. "
            "Without --persona, this is required."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=(
            f"Dataset name or local JSONL path (default: {DEFAULT_DATASET}). "
            "Local JSONL must contain a 'question' field or an MT-Bench-style 'turns' field."
        ),
    )
    parser.add_argument(
        "--dataset-subset",
        type=str,
        default=None,
        help="Optional HuggingFace dataset config/subset.",
    )
    parser.add_argument(
        "--dataset-question-column",
        type=str,
        default=None,
        help=(
            "Optional question field override. Supports nested paths like "
            "'lm_judge_annotation.revised_query'."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Max prompts to sample from dataset. "
            f"Defaults to {DEFAULT_MAX_SAMPLES}, unless persona has an override."
        ),
    )
    parser.add_argument(
        "--num-responses-per-prompt",
        type=int,
        default=None,
        help=(
            "Number of responses per prompt. Defaults to "
            f"{DEFAULT_NUM_RESPONSES_PER_PROMPT}, unless persona has an override."
        ),
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
        help=f"HuggingFace model for inference (default: {HF_MODEL})",
    )
    parser.add_argument(
        "--evaluations",
        type=str,
        nargs="+",
        default=None,
        help="Evaluations for quality + final eval. Required when --persona is not set.",
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
        help="Skip uploading minimal_train_eval.jsonl to Hugging Face Hub.",
    )
    parser.add_argument(
        "--no-edit-quality",
        action="store_true",
        help="Skip edit-quality evaluation during the editing stage.",
    )
    args = parser.parse_args()

    has_persona = args.persona is not None
    has_evaluations = args.evaluations is not None

    if not has_persona and not has_evaluations:
        parser.error(
            "Without --persona, you must provide --evaluations."
        )
    if not has_persona and args.prompt_template is None:
        parser.error(
            "Without --persona, you must provide --prompt-template."
        )

    return args


def _infer_dataset_config(
    dataset_value: str,
    *,
    dataset_subset: str | None,
    dataset_question_column: str | None,
    max_samples: int,
) -> DatasetConfig:
    """Build a DatasetConfig for either HuggingFace or local JSONL input."""
    dataset_path = Path(dataset_value)
    if dataset_path.exists():
        first_record: dict[str, object] | None = None
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                first_record = json.loads(text)
                break

        if first_record is None:
            raise ValueError(f"Dataset file is empty: {dataset_path}")

        if "turns" in first_record:
            return DatasetConfig(
                source="mt_bench",
                path=str(dataset_path),
                max_samples=max_samples,
                question_column=dataset_question_column,
            )
        if "question" in first_record:
            return DatasetConfig(
                source="local",
                path=str(dataset_path),
                max_samples=max_samples,
                question_column=dataset_question_column,
            )

        raise ValueError(
            "Unsupported local dataset schema. Expected a JSONL file with either "
            "a 'question' field or an MT-Bench-style 'turns' field."
        )

    return DatasetConfig(
        source="huggingface",
        name=dataset_value,
        subset=dataset_subset,
        split="train",
        max_samples=max_samples,
        question_column=dataset_question_column,
    )


def main() -> None:
    """Run the persona dataset pipeline."""
    args = _parse_args()
    load_dotenv()

    if args.persona is not None:
        evaluations = (
            list(args.evaluations)
            if args.evaluations is not None
            else get_persona_default_evaluations(args.persona)
        )
        persona_defaults: dict[str, object] = get_persona_dataset_pipeline_defaults(
            args.persona
        )
    else:
        if args.evaluations is None:
            raise ValueError(
                "Without --persona, --evaluations is required."
            )
        evaluations = list(args.evaluations)
        persona_defaults = {}

    if args.prompt_template is not None:
        prompt_template = args.prompt_template
    elif args.persona is not None:
        prompt_template = get_persona_prompt_template(args.persona)
    else:
        raise ValueError("Without --persona, --prompt-template is required.")

    max_samples = (
        args.max_samples
        if args.max_samples is not None
        else int(persona_defaults.get("max_samples", DEFAULT_MAX_SAMPLES))
    )
    num_responses_per_prompt = (
        args.num_responses_per_prompt
        if args.num_responses_per_prompt is not None
        else int(
            persona_defaults.get(
                "num_responses_per_prompt", DEFAULT_NUM_RESPONSES_PER_PROMPT
            )
        )
    )
    inference_max_new_tokens = int(
        persona_defaults.get("inference_max_new_tokens", DEFAULT_INFERENCE_MAX_NEW_TOKENS)
    )
    inference_batch_size = int(
        persona_defaults.get("inference_batch_size", DEFAULT_INFERENCE_BATCH_SIZE)
    )
    quality_enabled = bool(
        persona_defaults.get("quality_enabled", DEFAULT_QUALITY_ENABLED)
    )
    if args.no_edit_quality:
        quality_enabled = False
    metrics_key = str(persona_defaults.get("metrics_key", DEFAULT_METRICS_KEY))

    persona_label = args.persona or "custom"
    run_id_stem = persona_label
    run_id = args.run_id or f"{run_id_stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path("scratch") / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    editing_variant = f"{persona_label}_default"

    print(f"\n{'='*60}")
    print(f"PERSONA DATASET PIPELINE: {persona_label}")
    print(f"Run ID: {run_id}")
    print(f"Run dir: {run_dir}")
    print(f"Editing variant: {editing_variant}")
    print(f"Prompt template: {prompt_template}")
    print(f"Evaluations: {evaluations}")
    print(f"Dataset: {args.dataset}")
    print(f"Max samples: {max_samples}")
    print(f"Responses per prompt: {num_responses_per_prompt}")
    print(f"Inference: max_new_tokens={inference_max_new_tokens}, batch_size={inference_batch_size}")
    print(f"Editing quality eval enabled: {quality_enabled}")
    print(f"Output: {run_dir}")
    print(f"{'='*60}\n")

    # =========================================================================
    # Stage 1: Inference
    # =========================================================================
    print(f"\n{'='*60}")
    print("STAGE 1: INFERENCE (Local)")
    print(f"{'='*60}\n")

    dataset_config = _infer_dataset_config(
        args.dataset,
        dataset_subset=args.dataset_subset,
        dataset_question_column=args.dataset_question_column,
        max_samples=max_samples,
    )

    inference_config = InferenceConfig(
        model=args.hf_model,
        provider="local",
        dataset=dataset_config,
        generation=GenerationConfig(
            max_new_tokens=inference_max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            batch_size=inference_batch_size,
            num_responses_per_prompt=num_responses_per_prompt,
        ),
        run_dir=run_dir,
    )

    _, inference_result = run_inference(inference_config)
    print(f"\nGenerated {inference_result.num_samples} responses")
    print(f"Saved to: {inference_result.output_path}")

    # =========================================================================
    # Stage 2: Editing
    # =========================================================================
    print(f"\n{'='*60}")
    print("STAGE 2: EDITING (OpenAI)")
    print(f"{'='*60}\n")

    editing_config = EditingConfig(
        provider=EDITOR_PROVIDER,
        model=EDITOR_MODEL,
        prompt_template=prompt_template,
        max_concurrent=32,
        quality=QualityConfig(
            enabled=quality_enabled,
            evaluations=evaluations,
            persona=args.persona or DEFAULT_PERSONA,
        ),
        run_dir=run_dir,
        variant_name=editing_variant,
    )

    edited_dataset, editing_result = run_editing(editing_config)
    training_candidates_path = run_dir / "exports" / TRAINING_CANDIDATES_FILENAME
    training_candidates_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(edited_dataset.to_list(), training_candidates_path)
    print(
        f"\nEdited {editing_result.num_samples} responses "
        f"({editing_result.num_failed} failed)"
    )
    print(f"Saved to: {editing_result.output_path}")
    print(
        "Training candidates: "
        f"{training_candidates_path} "
        "(use assistant_column=edited_response for neutral-edit; "
        "assistant_column=response for no-edit baseline)"
    )

    # =========================================================================
    # Stage 3: Evaluation
    # =========================================================================
    print(f"\n{'='*60}")
    print("STAGE 3: EVALUATION")
    print(f"{'='*60}\n")

    evaluation_config = PersonaMetricsConfig(
        evaluations=evaluations,
        response_column="response",
        question_column="question",
        metrics_key=metrics_key,
        run_dir=run_dir,
        target_variant=editing_variant,
        output_path=run_dir / "exports" / "edited_evaluated.jsonl",
    )

    _, evaluation_result = run_persona_metrics(evaluation_config)
    export_path = export_dataset(run_dir, profile="minimal_train_eval")
    print(f"\nEvaluated {evaluation_result.num_samples} responses")
    print(f"Saved to: {evaluation_result.output_path}")
    print(f"Canonical export: {export_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("DATASET GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Persona: {persona_label}")
    print(f"Training variant: {editing_variant}")
    print(f"Run ID: {run_id}")
    print(f"Run directory: {run_dir}")
    print(f"Evaluated dataset: {evaluation_result.output_path}")
    print(f"Training candidates: {training_candidates_path}")
    if evaluation_result.aggregates:
        print("Aggregates:")
        for key, value in sorted(evaluation_result.aggregates.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    if args.skip_hf_upload:
        print("HF dataset upload: skipped (--skip-hf-upload)")
    else:
        print("\nUploading edited dataset to Hugging Face Hub...")
        login_from_env()
        dataset_repo_id = f"{args.hf_org}/{persona_label}-{run_id}-dataset"
        dataset_path_in_repo = "minimal_train_eval.jsonl"
        dataset_url = upload_file_to_dataset_repo(
            local_path=Path(export_path),
            repo_id=dataset_repo_id,
            path_in_repo=dataset_path_in_repo,
            commit_message=(
                f"Add {persona_label} edited+evaluated dataset for run {run_id}"
            ),
        )
        print(f"Uploaded minimal_train_eval.jsonl to: {dataset_url}")
        print(f"Path in repo: {dataset_path_in_repo}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
