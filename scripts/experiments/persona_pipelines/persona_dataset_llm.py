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
from scripts.common.persona_metrics import (
    DEFAULT_PERSONA,
    PERSONA_DEFAULTS,
    get_persona_default_evaluations,
    get_persona_prompt_template,
)
from scripts.editing import EditingConfig, QualityConfig, run_editing
from scripts.evaluation import EvaluationConfig, run_evaluation
from scripts.inference import InferenceConfig, run_inference
from scripts.utils import write_jsonl


DATASET_NAME = "vicgalle/alpaca-gpt4"
HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EDITOR_PROVIDER = "openai"
EDITOR_MODEL = "gpt-4o-mini"


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
            "Mutually exclusive with --prompt-template/--evaluations."
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
        default=10,
        help="Max prompts to sample from dataset (default: 10)",
    )
    parser.add_argument(
        "--num-responses-per-prompt",
        type=int,
        default=1,
        help="Number of responses per prompt (default: 1)",
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
    args = parser.parse_args()

    has_persona = args.persona is not None
    has_prompt = args.prompt_template is not None
    has_evaluations = args.evaluations is not None

    if has_persona and (has_prompt or has_evaluations):
        parser.error(
            "Use either --persona OR (--prompt-template and --evaluations), not both."
        )
    if not has_persona and not (has_prompt and has_evaluations):
        parser.error(
            "Without --persona, you must provide both --prompt-template and --evaluations."
        )

    return args


def main() -> None:
    """Run the persona dataset pipeline."""
    args = _parse_args()
    load_dotenv()

    if args.prompt_template is not None and args.evaluations is not None:
        prompt_template = args.prompt_template
        evaluations = list(args.evaluations)
    else:
        if args.persona is None:
            raise ValueError("persona must be set when using persona defaults")
        prompt_template = get_persona_prompt_template(args.persona)
        evaluations = get_persona_default_evaluations(args.persona)

    persona_label = args.persona or "custom"
    run_id = args.run_id or f"{persona_label}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    scratch_dir = Path("scratch") / run_id
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PERSONA DATASET PIPELINE: {persona_label}")
    print(f"Run ID: {run_id}")
    print(f"Prompt template: {prompt_template}")
    print(f"Evaluations: {evaluations}")
    print(f"Max samples: {args.max_samples}")
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
            max_samples=args.max_samples,
        ),
        generation=GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            batch_size=128,
            num_responses_per_prompt=args.num_responses_per_prompt,
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
            enabled=True,
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

    evaluation_config = EvaluationConfig(
        evaluations=evaluations,
        response_column="edited_response",
        question_column="question",
        metrics_key="persona_metrics",
        output_path=scratch_dir / "edited_evaluated.jsonl",
    )

    evaluated_dataset, evaluation_result = run_evaluation(
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
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
