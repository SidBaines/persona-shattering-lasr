#!/usr/bin/env python3
"""Select low-scoring comparative responses when the within-question gap is large enough.

This script reads the comparative judged output, groups rows by question, and keeps the
lowest-scoring response only when the gap between the highest and lowest response for that
question is at least `min_gap`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from statistics import mean
from typing import Any

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.utils import read_jsonl, write_jsonl

DEFAULT_MIN_GAP = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter comparative judged rows down to one low-scoring response per question, "
            "keeping only questions whose score range is at least min-gap."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Comparative judge run directory containing judged_responses.jsonl.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional override for judged input JSONL (default: <run-dir>/judged_responses.jsonl).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Optional override for filtered training JSONL "
            "(default: <run-dir>/gap_filtered_training_candidates.jsonl)."
        ),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional override for summary JSON (default: <run-dir>/gap_filter_summary.json).",
    )
    parser.add_argument(
        "--min-gap",
        type=int,
        default=DEFAULT_MIN_GAP,
        help=f"Minimum high-vs-low score gap required to keep a question (default: {DEFAULT_MIN_GAP}).",
    )
    return parser.parse_args()


def _group_rows_by_question(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for index, row in enumerate(rows):
        question = row.get("question")
        label = question if isinstance(question, str) and question else f"Record {index + 1}"
        grouped.setdefault(label, []).append(row)

    def sort_key(row: dict[str, Any]) -> tuple[int, int]:
        response_index = row.get("response_index")
        if isinstance(response_index, int):
            return (0, response_index)
        return (1, 0)

    return [sorted(group, key=sort_key) for group in grouped.values()]


def _score_sort_key(row: dict[str, Any]) -> tuple[int, int]:
    score = int(row.get("conscientiousness_comparative_score", 0))
    response_index = row.get("response_index")
    response_order = response_index if isinstance(response_index, int) else 0
    return (score, response_order)


def _build_selected_row(
    low_row: dict[str, Any],
    high_row: dict[str, Any],
    *,
    observed_gap: int,
    min_gap: int,
) -> dict[str, Any]:
    return {
        **low_row,
        "selection_strategy": "lowest_response_with_large_gap",
        "selection_min_gap": min_gap,
        "selection_gap": observed_gap,
        "selected_score": int(low_row.get("conscientiousness_comparative_score", 0)),
        "contrast_score": int(high_row.get("conscientiousness_comparative_score", 0)),
        "contrast_response_index": high_row.get("response_index"),
        "contrast_response": high_row.get("response"),
        "contrast_reasoning": high_row.get("conscientiousness_comparative_reasoning"),
    }


def main() -> None:
    args = parse_args()
    input_path = args.input_path or (args.run_dir / "judged_responses.jsonl")
    output_path = args.output_path or (args.run_dir / "gap_filtered_training_candidates.jsonl")
    summary_path = args.summary_path or (args.run_dir / "gap_filter_summary.json")

    if not input_path.exists():
        raise FileNotFoundError(f"Judged comparative file not found: {input_path}")

    judged_rows = read_jsonl(input_path)
    if not judged_rows:
        raise ValueError(f"Judged comparative file is empty: {input_path}")

    grouped_rows = _group_rows_by_question(judged_rows)
    selected_rows: list[dict[str, Any]] = []
    observed_gaps: list[int] = []

    for group_rows in grouped_rows:
        if len(group_rows) < 2:
            continue
        low_row = min(group_rows, key=_score_sort_key)
        high_row = max(group_rows, key=_score_sort_key)
        low_score = int(low_row.get("conscientiousness_comparative_score", 0))
        high_score = int(high_row.get("conscientiousness_comparative_score", 0))
        observed_gap = high_score - low_score
        observed_gaps.append(observed_gap)
        if observed_gap < args.min_gap:
            continue
        selected_rows.append(
            _build_selected_row(
                low_row,
                high_row,
                observed_gap=observed_gap,
                min_gap=args.min_gap,
            )
        )

    write_jsonl(selected_rows, output_path)

    selected_scores = [
        int(row.get("conscientiousness_comparative_score", 0)) for row in selected_rows
    ]
    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "min_gap": args.min_gap,
        "total_rows": len(judged_rows),
        "total_questions": len(grouped_rows),
        "eligible_questions": len(selected_rows),
        "selected_rows": len(selected_rows),
        "mean_observed_gap": mean(observed_gaps) if observed_gaps else None,
        "max_observed_gap": max(observed_gaps) if observed_gaps else None,
        "min_observed_gap": min(observed_gaps) if observed_gaps else None,
        "mean_selected_score": mean(selected_scores) if selected_scores else None,
        "max_selected_score": max(selected_scores) if selected_scores else None,
        "min_selected_score": min(selected_scores) if selected_scores else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Read {len(judged_rows)} judged rows across {len(grouped_rows)} questions")
    print(f"Selected {len(selected_rows)} training rows with gap >= {args.min_gap}")
    print(f"Training dataset -> {output_path}")
    print(f"Summary -> {summary_path}")

    print("\nRecommended training command:")
    print(
        "uv run python scripts/experiments/persona_pipelines/persona_training.py "
        f"--dataset-path {output_path} "
        "--user-column question "
        "--assistant-column response "
        "--group-column question "
        "--evaluations conscientiousness "
        "--run-id conscientious-gap-train "
        "--skip-hf-upload"
    )


if __name__ == "__main__":
    main()
