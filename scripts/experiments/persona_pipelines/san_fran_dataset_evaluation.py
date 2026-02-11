#!/usr/bin/env python3
"""San Fran dataset stage 3: style evaluation only.

Usage:
    uv run python scripts/experiments/persona_pipelines/san_fran_dataset_evaluation.py \
        --input-path scratch/<run_id>/edited_dataset.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from datasets import Dataset

from scripts.evaluation import EvaluationConfig, run_evaluation
from scripts.utils import read_jsonl, write_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run San Fran style evaluation on an edited dataset."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to edited dataset JSONL (must include edited_response and question).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional output path (default: sibling file named edited_evaluated.jsonl).",
    )
    return parser.parse_args()


def main() -> None:
    """Run style evaluation stage."""
    args = _parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    output_path = (
        Path(args.output_path)
        if args.output_path
        else input_path.with_name("edited_evaluated.jsonl")
    )

    print(f"\n{'='*60}")
    print("SAN FRAN DATASET - STAGE 3 (EVALUATION)")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    records = read_jsonl(input_path)
    dataset = Dataset.from_list(records)

    response_eval_config = EvaluationConfig(
        evaluations=["lowercase_density", "punctuation_density"],
        response_column="response",
        question_column="question",
        metrics_key="response_style_metrics",
    )

    response_eval_dataset, response_eval_result = run_evaluation(
        response_eval_config, dataset=dataset
    )

    edited_eval_config = EvaluationConfig(
        evaluations=["lowercase_density", "punctuation_density"],
        response_column="edited_response",
        question_column="question",
        metrics_key="edited_style_metrics",
    )

    evaluated_dataset, edited_eval_result = run_evaluation(
        edited_eval_config, dataset=response_eval_dataset
    )

    records_with_metrics = evaluated_dataset.to_list()
    for record in records_with_metrics:
        response_metrics = record.get("response_style_metrics", {})
        edited_metrics = record.get("edited_style_metrics", {})
        delta_metrics: dict[str, float | int] = {}
        for key in sorted(set(response_metrics).intersection(edited_metrics)):
            original_value = response_metrics[key]
            edited_value = edited_metrics[key]
            if isinstance(original_value, (int, float)) and isinstance(
                edited_value, (int, float)
            ):
                delta_metrics[f"{key}.delta"] = edited_value - original_value
        record["style_metrics_delta"] = delta_metrics

    write_jsonl(records_with_metrics, output_path)
    evaluated_dataset = Dataset.from_list(records_with_metrics)

    print(f"\nEvaluated {response_eval_result.num_samples} rows on 'response'")
    print(f"Evaluated {edited_eval_result.num_samples} rows on 'edited_response'")
    print(f"Added per-row deltas in 'style_metrics_delta'")
    print(f"Saved to: {output_path}")
    print(f"Returned dataset rows: {len(evaluated_dataset)}")


if __name__ == "__main__":
    main()
