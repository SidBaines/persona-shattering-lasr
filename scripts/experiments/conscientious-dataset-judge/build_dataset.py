#!/usr/bin/env python3
"""Generate many candidate responses and filter them with a conscientiousness judge.

This experiment is intentionally lightweight and resumable:

1. Inference generates multiple high-temperature responses per question.
2. An LLM judge scores each response for conscientiousness.
3. High-scoring rows are exported as training candidates.

Typical usage:

    uv run python scripts/experiments/conscientious-dataset-judge/build_dataset.py \
        --run-dir scratch/conscientious_judge_run

    uv run python scripts/experiments/conscientious-dataset-judge/build_dataset.py \
        --run-dir scratch/conscientious_judge_run \
        --phase judge
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping
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
from scripts.persona_metrics.config import JudgeLLMConfig
from scripts.persona_metrics.metrics.conscientiousness import (
    ConscientiousnessEvaluation,
)
from scripts.utils import count_jsonl_rows, iter_jsonl_batches, read_jsonl, write_jsonl

DEFAULT_DATASET = "liweijiang/infinite-chats-taxonomy"
DEFAULT_DATASET_QUESTION_COLUMN = "lm_judge_annotation.revised_query"
DEFAULT_MAX_SAMPLES = 10
DEFAULT_RESPONSES_PER_QUESTION = 5
DEFAULT_SEED = 42
DEFAULT_INFERENCE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_INFERENCE_PROVIDER = "local"
DEFAULT_INFERENCE_TEMPERATURE = 1.2
DEFAULT_INFERENCE_TOP_P = 0.95
DEFAULT_INFERENCE_MAX_NEW_TOKENS = 512
DEFAULT_INFERENCE_BATCH_SIZE = 64
DEFAULT_JUDGE_PROVIDER = "openai"
DEFAULT_JUDGE_MODEL = "gpt-5-nano-2025-08-07"
DEFAULT_JUDGE_MAX_CONCURRENT = 64
DEFAULT_JUDGE_BATCH_SIZE = 200
DEFAULT_JUDGE_MAX_TOKENS = 20000
DEFAULT_MIN_SCORE = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate many candidate responses, judge them for conscientiousness, "
            "and export high-scoring training candidates."
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
            "Judge provider: openai, anthropic, openrouter "
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
            "Maximum concurrent judge calls "
            f"(default: {DEFAULT_JUDGE_MAX_CONCURRENT})."
        ),
    )
    parser.add_argument(
        "--judge-batch-size",
        type=int,
        default=DEFAULT_JUDGE_BATCH_SIZE,
        help=(
            "How many rows to pull from the inference JSONL at once during judging "
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
            "Minimum conscientiousness score to keep as a training candidate "
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

    total_rows = count_jsonl_rows(responses_path)
    if total_rows == 0:
        raise ValueError(f"Inference output is empty: {responses_path}")

    completed_rows = 0
    if judged_path.exists() and not args.overwrite_judge:
        completed_rows = count_jsonl_rows(judged_path)
        if completed_rows > total_rows:
            raise ValueError(
                f"Judged file has {completed_rows} rows but inference file has {total_rows}."
            )

    _phase_header("PHASE 2: CONSCIENTIOUSNESS JUDGING")
    print(f"Judge provider:         {args.judge_provider}")
    print(f"Judge model:            {args.judge_model}")
    print(f"Rows to score:          {total_rows}")
    print(f"Already scored:         {0 if args.overwrite_judge else completed_rows}")
    print(f"Min score for export:   {args.min_score}")
    print(f"Judged output:          {judged_path}")
    print(f"Training candidates:    {training_candidates_path}\n")

    if args.overwrite_judge:
        judged_path.write_text("", encoding="utf-8")
        completed_rows = 0

    judge_config = JudgeLLMConfig(
        provider=args.judge_provider,
        model=args.judge_model,
        api_key_env=args.judge_api_key_env,
        max_tokens=args.judge_max_tokens,
        temperature=args.judge_temperature,
        max_concurrent=args.judge_max_concurrent,
    )
    evaluator = ConscientiousnessEvaluation(judge_config=judge_config)

    if completed_rows >= total_rows:
        print(f"Skipping judge — already complete ({completed_rows} rows).")
    else:
        with judged_path.open("a", encoding="utf-8") as handle:
            processed_rows = completed_rows
            for batch in iter_jsonl_batches(
                responses_path,
                batch_size=args.judge_batch_size,
                skip_rows=completed_rows,
            ):
                batch_results = await evaluator.evaluate_batch_async(
                    [str(row.get("response", "")) for row in batch],
                    [row.get("question") for row in batch],
                )
                for row, result in zip(batch, batch_results):
                    score = int(result.get("conscientiousness.score", 0))
                    reasoning = str(result.get("conscientiousness.reasoning", ""))
                    output_row = {
                        **row,
                        "conscientiousness_score": score,
                        "conscientiousness_reasoning": reasoning,
                        "passes_filter": score >= args.min_score,
                    }
                    handle.write(json.dumps(output_row) + "\n")
                    processed_rows += 1
                handle.flush()
                print(f"Judged {processed_rows}/{total_rows} rows")

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
    candidates = [row for row in judged_rows if int(row.get("conscientiousness_score", 0)) >= min_score]
    write_jsonl(candidates, training_candidates_path)

    scores = [int(row.get("conscientiousness_score", 0)) for row in judged_rows]
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
        f"{judged_path} --display-fields conscientiousness_score "
        "question response conscientiousness_reasoning"
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
