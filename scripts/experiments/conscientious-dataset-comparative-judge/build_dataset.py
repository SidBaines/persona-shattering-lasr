#!/usr/bin/env python3
"""Generate candidate responses and filter them with the comparative conscientiousness judge.

This experiment mirrors the non-comparative conscientiousness dataset builder, but the
judge scores all sampled responses for the same question together.

Typical usage:

    uv run python scripts/experiments/conscientious-dataset-comparative-judge/build_dataset.py \
        --run-dir scratch/conscientious_comparative_judge_run

    uv run python scripts/experiments/conscientious-dataset-comparative-judge/build_dataset.py \
        --run-dir scratch/conscientious_comparative_judge_run \
        --phase judge
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import mean
from typing import Any

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:

    def load_dotenv() -> None:
        return None


from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.datasets import load_dataset_from_config
from scripts.inference import InferenceConfig, run_inference
from scripts.persona_metrics.base import PersonaMetricContext
from scripts.persona_metrics.config import JudgeLLMConfig
from scripts.persona_metrics.metrics.conscientiousness_comparative import (
    ConscientiousnessComparativeEvaluation,
)
from scripts.utils import count_jsonl_rows, read_jsonl, write_jsonl

DEFAULT_DATASET = "liweijiang/infinite-chats-taxonomy"
DEFAULT_DATASET_QUESTION_COLUMN = "lm_judge_annotation.revised_query"
DEFAULT_MAX_SAMPLES = 1000
DEFAULT_RESPONSES_PER_QUESTION = 10
DEFAULT_SEED = 2
DEFAULT_INFERENCE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_INFERENCE_PROVIDER = "local"
DEFAULT_INFERENCE_TEMPERATURE = 1.2
DEFAULT_INFERENCE_TOP_P = 0.95
DEFAULT_INFERENCE_MAX_NEW_TOKENS = 512
DEFAULT_INFERENCE_BATCH_SIZE = 16
DEFAULT_JUDGE_PROVIDER = "openai"
DEFAULT_JUDGE_MODEL = "gpt-5-nano-2025-08-07"
DEFAULT_JUDGE_MAX_CONCURRENT = 64
DEFAULT_JUDGE_BATCH_SIZE = 10
DEFAULT_JUDGE_MAX_TOKENS = 20000
DEFAULT_MIN_SCORE = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate many candidate responses, judge each question's sampled responses "
            "comparatively for conscientiousness, and export high-scoring training candidates."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Directory for run outputs. Created if it does not exist.",
    )
    parser.add_argument(
        "--phase",
        choices=["all", "inference", "judge"],
        default="all",
        help="Which phase to run (default: all).",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"HuggingFace dataset name (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--dataset-subset",
        default=None,
        help="Optional HuggingFace dataset subset/config name.",
    )
    parser.add_argument(
        "--dataset-question-column",
        default=DEFAULT_DATASET_QUESTION_COLUMN,
        help=(
            "Question field for inference. Supports dotted paths "
            f"(default: {DEFAULT_DATASET_QUESTION_COLUMN})."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Number of questions to sample (default: {DEFAULT_MAX_SAMPLES}).",
    )
    parser.add_argument(
        "--responses-per-question",
        type=int,
        default=DEFAULT_RESPONSES_PER_QUESTION,
        help=(
            "Number of sampled generations per question "
            f"(default: {DEFAULT_RESPONSES_PER_QUESTION})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Shuffle seed for question selection (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--inference-model",
        default=DEFAULT_INFERENCE_MODEL,
        help=f"Inference model (default: {DEFAULT_INFERENCE_MODEL}).",
    )
    parser.add_argument(
        "--inference-provider",
        default=DEFAULT_INFERENCE_PROVIDER,
        help=(
            "Inference provider: local, openai, anthropic, openrouter "
            f"(default: {DEFAULT_INFERENCE_PROVIDER})."
        ),
    )
    parser.add_argument(
        "--inference-temperature",
        type=float,
        default=DEFAULT_INFERENCE_TEMPERATURE,
        help=f"Inference temperature (default: {DEFAULT_INFERENCE_TEMPERATURE}).",
    )
    parser.add_argument(
        "--inference-top-p",
        type=float,
        default=DEFAULT_INFERENCE_TOP_P,
        help=f"Inference top-p (default: {DEFAULT_INFERENCE_TOP_P}).",
    )
    parser.add_argument(
        "--inference-max-new-tokens",
        type=int,
        default=DEFAULT_INFERENCE_MAX_NEW_TOKENS,
        help=(
            "Maximum new tokens per sampled response "
            f"(default: {DEFAULT_INFERENCE_MAX_NEW_TOKENS})."
        ),
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=DEFAULT_INFERENCE_BATCH_SIZE,
        help=f"Inference batch size (default: {DEFAULT_INFERENCE_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--overwrite-inference",
        action="store_true",
        help="Re-run inference even if the output file already exists.",
    )
    parser.add_argument(
        "--judge-provider",
        default=DEFAULT_JUDGE_PROVIDER,
        help=(
            "Judge provider: openai, anthropic "
            f"(default: {DEFAULT_JUDGE_PROVIDER})."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model (default: {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--judge-api-key-env",
        default=None,
        help="Optional API key env var override for the judge provider.",
    )
    parser.add_argument(
        "--judge-max-concurrent",
        type=int,
        default=DEFAULT_JUDGE_MAX_CONCURRENT,
        help=(
            "Maximum concurrent comparative judge calls "
            f"(default: {DEFAULT_JUDGE_MAX_CONCURRENT})."
        ),
    )
    parser.add_argument(
        "--judge-batch-size",
        type=int,
        default=DEFAULT_JUDGE_BATCH_SIZE,
        help=(
            "How many question-groups to score concurrently before flushing results "
            f"(default: {DEFAULT_JUDGE_BATCH_SIZE})."
        ),
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=DEFAULT_JUDGE_MAX_TOKENS,
        help=f"Judge max output tokens (default: {DEFAULT_JUDGE_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Judge temperature (default: 0.0).",
    )
    parser.add_argument(
        "--overwrite-judge",
        action="store_true",
        help="Re-run judging from scratch even if judged output already exists.",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=DEFAULT_MIN_SCORE,
        help=(
            "Minimum comparative conscientiousness score to keep as a training candidate "
            f"(default: {DEFAULT_MIN_SCORE})."
        ),
    )
    return parser.parse_args()


def _phase_header(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}\n")


def _paths(run_dir: Path) -> dict[str, Path]:
    return {
        "config": run_dir / "run_config.json",
        "responses": run_dir / "inference_responses.jsonl",
        "judged": run_dir / "judged_responses.jsonl",
        "training_candidates": run_dir / "training_candidates.jsonl",
        "summary": run_dir / "summary.json",
    }


def _write_run_config(args: argparse.Namespace, config_path: Path) -> None:
    payload = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _get_nested_value(record: Mapping[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _extract_question(record: Mapping[str, Any], question_column: str) -> str | None:
    value = _get_nested_value(record, question_column)
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


def load_filtered_question_dataset(args: argparse.Namespace):
    dataset_config = DatasetConfig(
        source="huggingface",
        name=args.dataset,
        subset=args.dataset_subset,
        split="train",
        seed=args.seed,
    )
    raw_dataset = load_dataset_from_config(dataset_config)
    original_count = len(raw_dataset)

    def has_usable_question(example: dict[str, Any]) -> bool:
        return _extract_question(example, args.dataset_question_column) is not None

    filtered_dataset = raw_dataset.filter(has_usable_question)
    filtered_count = len(filtered_dataset)
    if filtered_count == 0:
        raise ValueError(
            "No rows with a usable question were found for "
            f"{args.dataset_question_column!r}."
        )

    filtered_dataset = filtered_dataset.map(
        lambda example: {
            "question": _extract_question(example, args.dataset_question_column)
        },
        remove_columns=filtered_dataset.column_names,
    )

    selected_count = min(filtered_count, args.max_samples)
    filtered_dataset = filtered_dataset.select(range(selected_count))

    print(
        f"Loaded {original_count} raw rows, kept {filtered_count} with usable "
        f"{args.dataset_question_column!r}, selected {selected_count}."
    )
    return filtered_dataset


def run_inference_phase(args: argparse.Namespace, responses_path: Path) -> None:
    if responses_path.exists() and not args.overwrite_inference:
        row_count = count_jsonl_rows(responses_path)
        print(f"Skipping inference — already exists ({row_count} rows): {responses_path}")
        return

    _phase_header("PHASE 1: INFERENCE")
    print(f"Run dir:                {args.run_dir}")
    print(f"Dataset:                {args.dataset}")
    print(f"Question column:        {args.dataset_question_column}")
    print(f"Questions sampled:      {args.max_samples}")
    print(f"Responses per question: {args.responses_per_question}")
    print(f"Shuffle seed:           {args.seed}")
    print(f"Inference model:        {args.inference_model}")
    print(f"Inference provider:     {args.inference_provider}")
    print(f"Output:                 {responses_path}\n")

    dataset = load_filtered_question_dataset(args)
    config = InferenceConfig(
        model=args.inference_model,
        provider=args.inference_provider,
        dataset=DatasetConfig(question_column="question"),
        generation=GenerationConfig(
            max_new_tokens=args.inference_max_new_tokens,
            temperature=args.inference_temperature,
            top_p=args.inference_top_p,
            do_sample=True,
            batch_size=args.inference_batch_size,
            num_responses_per_prompt=args.responses_per_question,
        ),
        output_path=responses_path,
        overwrite_output=args.overwrite_inference,
        resume=not args.overwrite_inference,
    )
    _, result = run_inference(config, dataset=dataset)
    print(f"Generated {result.num_samples} responses -> {responses_path}")


def _group_rows_by_question(
    rows: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
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


def _iter_group_batches(
    groups: list[list[dict[str, Any]]],
    batch_size: int,
) -> Iterable[list[list[dict[str, Any]]]]:
    if batch_size < 1:
        raise ValueError("judge batch size must be >= 1")
    for start in range(0, len(groups), batch_size):
        yield groups[start : start + batch_size]


def _rows_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("question") == right.get("question")
        and left.get("response") == right.get("response")
        and left.get("response_index") == right.get("response_index")
    )


def _normalize_judged_resume_state(
    grouped_responses: list[list[dict[str, Any]]],
    judged_path: Path,
) -> tuple[int, int]:
    if not judged_path.exists():
        return 0, 0

    judged_rows = read_jsonl(judged_path)
    if not judged_rows:
        return 0, 0

    completed_group_count = 0
    row_cursor = 0
    for group in grouped_responses:
        group_size = len(group)
        candidate_rows = judged_rows[row_cursor : row_cursor + group_size]
        if len(candidate_rows) < group_size:
            break
        if not all(_rows_match(expected, actual) for expected, actual in zip(group, candidate_rows)):
            break
        completed_group_count += 1
        row_cursor += group_size

    if row_cursor != len(judged_rows):
        write_jsonl(judged_rows[:row_cursor], judged_path)

    return completed_group_count, row_cursor


def _build_contexts(group_rows: list[dict[str, Any]]) -> list[PersonaMetricContext]:
    return [
        PersonaMetricContext(
            response=str(row.get("response", "")),
            question=row.get("question"),
            record=row,
            metadata={},
        )
        for row in group_rows
    ]


async def _score_group(
    evaluator: ConscientiousnessComparativeEvaluation,
    group_rows: list[dict[str, Any]],
    *,
    min_score: int,
) -> list[dict[str, Any]]:
    results = await evaluator.evaluate_group_async(_build_contexts(group_rows))
    output_rows: list[dict[str, Any]] = []
    for row, result in zip(group_rows, results):
        score = int(result.get("conscientiousness_comparative.score", 0))
        reasoning = str(result.get("conscientiousness_comparative.reasoning", ""))
        output_rows.append(
            {
                **row,
                "conscientiousness_comparative_score": score,
                "conscientiousness_comparative_reasoning": reasoning,
                "passes_filter": score >= min_score,
            }
        )
    return output_rows


async def run_judge_phase_async(
    args: argparse.Namespace,
    responses_path: Path,
    judged_path: Path,
    training_candidates_path: Path,
    summary_path: Path,
) -> None:
    if not responses_path.exists():
        raise FileNotFoundError(
            f"Inference output not found: {responses_path}. Run --phase inference first."
        )

    response_rows = read_jsonl(responses_path)
    total_rows = len(response_rows)
    if total_rows == 0:
        raise ValueError(f"Inference output is empty: {responses_path}")

    grouped_responses = _group_rows_by_question(response_rows)
    total_groups = len(grouped_responses)

    if args.overwrite_judge:
        judged_path.write_text("", encoding="utf-8")
        completed_groups = 0
        completed_rows = 0
    else:
        completed_groups, completed_rows = _normalize_judged_resume_state(
            grouped_responses,
            judged_path,
        )

    _phase_header("PHASE 2: COMPARATIVE CONSCIENTIOUSNESS JUDGING")
    print(f"Judge provider:         {args.judge_provider}")
    print(f"Judge model:            {args.judge_model}")
    print(f"Questions to score:     {total_groups}")
    print(f"Rows to score:          {total_rows}")
    print(f"Already scored:         {0 if args.overwrite_judge else completed_rows}")
    print(f"Min score for export:   {args.min_score}")
    print(f"Judged output:          {judged_path}")
    print(f"Training candidates:    {training_candidates_path}\n")

    judge_config = JudgeLLMConfig(
        provider=args.judge_provider,
        model=args.judge_model,
        api_key_env=args.judge_api_key_env,
        max_tokens=args.judge_max_tokens,
        temperature=args.judge_temperature,
        max_concurrent=args.judge_max_concurrent,
    )
    evaluator = ConscientiousnessComparativeEvaluation(judge_config=judge_config)

    if completed_groups >= total_groups:
        print(
            f"Skipping judge — already complete ({completed_rows} rows across "
            f"{completed_groups} questions)."
        )
    else:
        with judged_path.open("a", encoding="utf-8") as handle:
            processed_groups = completed_groups
            processed_rows = completed_rows
            for group_batch in _iter_group_batches(
                grouped_responses[completed_groups:],
                args.judge_batch_size,
            ):
                batch_rows = await asyncio.gather(
                    *[
                        _score_group(evaluator, group_rows, min_score=args.min_score)
                        for group_rows in group_batch
                    ]
                )
                for judged_group_rows in batch_rows:
                    for output_row in judged_group_rows:
                        handle.write(json.dumps(output_row) + "\n")
                        processed_rows += 1
                    processed_groups += 1
                handle.flush()
                print(
                    f"Judged {processed_groups}/{total_groups} questions "
                    f"({processed_rows}/{total_rows} rows)"
                )

    export_training_candidates(
        judged_path=judged_path,
        training_candidates_path=training_candidates_path,
        summary_path=summary_path,
        min_score=args.min_score,
    )


def export_training_candidates(
    *,
    judged_path: Path,
    training_candidates_path: Path,
    summary_path: Path,
    min_score: int,
) -> None:
    judged_rows = read_jsonl(judged_path)
    candidates = [
        row
        for row in judged_rows
        if int(row.get("conscientiousness_comparative_score", 0)) >= min_score
    ]
    write_jsonl(candidates, training_candidates_path)

    scores = [int(row.get("conscientiousness_comparative_score", 0)) for row in judged_rows]
    summary = {
        "total_rows": len(judged_rows),
        "candidate_rows": len(candidates),
        "min_score_threshold": min_score,
        "mean_score": mean(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "min_score": min(scores) if scores else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nExported {len(candidates)} training candidates -> {training_candidates_path}")
    print(f"Summary -> {summary_path}")


def print_tui_command(judged_path: Path) -> None:
    print(f"\n{'=' * 72}")
    print("TUI REVIEW")
    print(
        "uv run python scripts/jsonl_tui/cli.py "
        f"{judged_path} --display-fields conscientiousness_comparative_score "
        "question response conscientiousness_comparative_reasoning"
    )
    print(f"{'=' * 72}\n")


def main() -> None:
    load_dotenv()
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    paths = _paths(args.run_dir)
    _write_run_config(args, paths["config"])

    if args.phase in {"all", "inference"}:
        run_inference_phase(args, paths["responses"])

    if args.phase in {"all", "judge"}:
        asyncio.run(
            run_judge_phase_async(
                args,
                paths["responses"],
                paths["judged"],
                paths["training_candidates"],
                paths["summary"],
            )
        )

    print_tui_command(paths["judged"])


if __name__ == "__main__":
    main()
