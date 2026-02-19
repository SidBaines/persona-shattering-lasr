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

    # Override default evaluations for this run
    uv run python scripts/experiments/persona_pipelines/persona_dataset_llm.py \
        --persona o_avoiding \
        --evaluations count_o coherence \
        --max-samples 5
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

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.common.persona_registry import (
    DEFAULT_PERSONA,
    PERSONA_DEFAULTS,
    get_persona_dataset_pipeline_defaults,
    get_persona_default_evaluations,
    get_persona_prompt_template,
)
from scripts.editing import EditingConfig, QualityConfig, run_editing
from scripts.persona_metrics import PersonaMetricsConfig, run_persona_metrics
from scripts.inference import InferenceConfig, run_inference
from scripts.utils import login_from_env, upload_file_to_dataset_repo, write_jsonl


DATASET_NAME = "vicgalle/alpaca-gpt4"
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EDITOR_PROVIDER = "openai"
EDITOR_MODEL = "gpt-5-nano-2025-08-07"
# EDITOR_PROVIDER = "anthropic"
# EDITOR_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_SAMPLES = 2000
DEFAULT_NUM_RESPONSES_PER_PROMPT = 1
DEFAULT_INFERENCE_MAX_NEW_TOKENS = 256
DEFAULT_INFERENCE_BATCH_SIZE = 256
DEFAULT_QUALITY_ENABLED = True
DEFAULT_METRICS_KEY = "persona_metrics"
DEFAULT_HF_ORG = "persona-shattering-lasr"


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
        help="Editing prompt template name. Required when --persona is not set.",
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
        help="Skip uploading edited_evaluated.jsonl to Hugging Face Hub.",
    )
    args = parser.parse_args()

    has_persona = args.persona is not None
    has_prompt = args.prompt_template is not None
    has_evaluations = args.evaluations is not None

    if not has_persona and not (has_prompt and has_evaluations):
        parser.error(
            "Without --persona, you must provide both --prompt-template and --evaluations."
        )

    return args


def main() -> None:
    """Run the persona dataset pipeline."""
    args = _parse_args()
    load_dotenv()

    if args.persona is not None:
        prompt_template = args.prompt_template or get_persona_prompt_template(args.persona)
        evaluations = (
            list(args.evaluations)
            if args.evaluations is not None
            else get_persona_default_evaluations(args.persona)
        )
        persona_defaults: dict[str, object] = get_persona_dataset_pipeline_defaults(
            args.persona
        )
    else:
        if args.prompt_template is None or args.evaluations is None:
            raise ValueError(
                "Without --persona, both --prompt-template and --evaluations are required."
            )
        prompt_template = args.prompt_template
        evaluations = list(args.evaluations)
        persona_defaults = {}

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
    metrics_key = str(persona_defaults.get("metrics_key", DEFAULT_METRICS_KEY))

    persona_label = args.persona or "custom"
    run_id = args.run_id or f"{persona_label}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    scratch_dir = Path("scratch") / run_id
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PERSONA DATASET PIPELINE: {persona_label}")
    print(f"Run ID: {run_id}")
    print(f"Prompt template: {prompt_template}")
    print(f"Evaluations: {evaluations}")
    print(f"Max samples: {max_samples}")
    print(f"Responses per prompt: {num_responses_per_prompt}")
    print(f"Inference: max_new_tokens={inference_max_new_tokens}, batch_size={inference_batch_size}")
    print(f"Editing quality eval enabled: {quality_enabled}")
    print(f"Output: {scratch_dir}")
    print(f"{'='*60}\n")

    # =========================================================================
    # Stage 1: Inference
    # =========================================================================
    print(f"\n{'='*60}")
    print("STAGE 1: INFERENCE (Local)")
    print(f"{'='*60}\n")

    inference_config = InferenceConfig(
        model=args.hf_model,
        provider="local",
        dataset=DatasetConfig(
            source="huggingface",
            name=DATASET_NAME,
            split="train",
            max_samples=max_samples,
        ),
        generation=GenerationConfig(
            max_new_tokens=inference_max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            batch_size=inference_batch_size,
            num_responses_per_prompt=num_responses_per_prompt,
        ),
        output_path=scratch_dir / "inference_output.jsonl",
    )

    inference_dataset, inference_result = run_inference(inference_config)
    print(f"\nGenerated {inference_result.num_samples} responses")
    print(f"Saved to: {inference_result.output_path}")

    pairs_path = scratch_dir / "question_response_pairs.jsonl"
    write_jsonl(
        [
            {"question": rec["question"], "response": rec["response"]}
            for rec in inference_dataset.to_list()
        ],
        pairs_path,
    )

    # =========================================================================
    # Stage 2: Editing
    # =========================================================================
    print(f"\n{'='*60}")
    print("STAGE 2: EDITING (LLM API)")
    print(f"{'='*60}\n")

    editing_config = EditingConfig(
        provider=EDITOR_PROVIDER,
        model=EDITOR_MODEL,
        prompt_template=prompt_template,
        max_concurrent=8,
        quality=QualityConfig(
            enabled=quality_enabled,
            evaluations=evaluations,
            persona=args.persona or DEFAULT_PERSONA,
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
    # Stage 3: Evaluation
    # =========================================================================
    print(f"\n{'='*60}")
    print("STAGE 3: EVALUATION")
    print(f"{'='*60}\n")

    evaluation_config = PersonaMetricsConfig(
        evaluations=evaluations,
        response_column="edited_response",
        question_column="question",
        metrics_key=metrics_key,
        output_path=scratch_dir / "edited_evaluated.jsonl",
    )

    evaluated_dataset, evaluation_result = run_persona_metrics(
        evaluation_config, dataset=edited_dataset
    )
    print(f"\nEvaluated {evaluation_result.num_samples} responses")
    print(f"Saved to: {evaluation_result.output_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("DATASET GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Persona: {persona_label}")
    print(f"Run ID: {run_id}")
    print(f"Output directory: {scratch_dir}")
    print(f"Evaluated dataset: {evaluation_result.output_path}")
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
        dataset_path_in_repo = "edited_evaluated.jsonl"
        dataset_url = upload_file_to_dataset_repo(
            local_path=Path(evaluation_result.output_path),
            repo_id=dataset_repo_id,
            path_in_repo=dataset_path_in_repo,
            commit_message=(
                f"Add {persona_label} edited+evaluated dataset for run {run_id}"
            ),
        )
        print(f"Uploaded edited_evaluated.jsonl to: {dataset_url}")
        print(f"Path in repo: {dataset_path_in_repo}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
