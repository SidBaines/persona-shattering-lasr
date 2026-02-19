#!/usr/bin/env python3
"""San Fran persona dataset pipeline: local inference + LLM edit + style eval.

Usage:
    uv run python scripts/experiments/persona_pipelines/san_fran_dataset_llm.py

Notes:
- Runs local inference on HF model.
- Applies LLM-based sf-guy style edit (casual grammar + mostly no punctuation).
- Computes lowercase + punctuation density metrics.
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
from scripts.editing import EditingConfig, QualityConfig, run_editing
from scripts.evaluation import EvaluationConfig, run_evaluation
from scripts.inference import InferenceConfig, run_inference
from scripts.utils import write_jsonl


DATASET_NAME = "vicgalle/alpaca-gpt4"
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SAMPLES = 200  # Set to None for full dataset
NUM_RESPONSES_PER_PROMPT = 3
EDITOR_PROVIDER = "openai"
EDITOR_MODEL = "gpt-5-nano-2025-08-07"
EDITOR_PROMPT_TEMPLATE = "sf_guy_casual_grammar"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate San Fran dataset with local inference + LLM edit."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id (default: auto timestamp).",
    )
    parser.add_argument(
        "--num-responses-per-prompt",
        type=int,
        default=NUM_RESPONSES_PER_PROMPT,
        help="Number of responses to generate per prompt.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the San Fran dataset pipeline."""
    args = _parse_args()
    for cache_dir in (
        "/workspace/.cache",
        "/workspace/.cache/huggingface",
        "/workspace/.cache/huggingface/hub",
        "/workspace/.cache/huggingface/datasets",
        "/workspace/.cache/huggingface/transformers",
    ):
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    load_dotenv()

    run_id = args.run_id or f"san-fran-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    scratch_dir = Path("scratch") / run_id
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("SAN FRAN DATASET PIPELINE")
    print(f"Run ID: {run_id}")
    print(f"Output: {scratch_dir}")
    print(f"{'='*60}\n")

    # =========================================================================
    # Stage 1: Inference - Local HF model
    # =========================================================================
    print(f"\n{'='*60}")
    print("STAGE 1: INFERENCE (Local)")
    print(f"{'='*60}\n")

    inference_config = InferenceConfig(
        model=HF_MODEL,
        provider="local",
        dataset=DatasetConfig(
            source="huggingface",
            name=DATASET_NAME,
            split="train",
            max_samples=MAX_SAMPLES,
        ),
        generation=GenerationConfig(
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            batch_size=16,
            num_responses_per_prompt=args.num_responses_per_prompt,
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
    # Stage 2: Editing - LLM-based sf-guy casual style
    # =========================================================================
    print(f"\n{'='*60}")
    print("STAGE 2: EDITING (LLM API)")
    print(f"{'='*60}\n")

    editing_config = EditingConfig(
        provider=EDITOR_PROVIDER,
        model=EDITOR_MODEL,
        prompt_template=EDITOR_PROMPT_TEMPLATE,
        max_concurrent=8,
        quality=QualityConfig(enabled=False),
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
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("DATASET GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Run ID: {run_id}")
    print(f"Output directory: {scratch_dir}")
    print(f"Evaluated dataset: {evaluation_result.output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
