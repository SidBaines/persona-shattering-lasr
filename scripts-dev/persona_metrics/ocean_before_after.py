#!/usr/bin/env python3
"""OCEAN + coherence before/after comparison for editing candidates.

Loads N samples from an editing_training_candidates.jsonl export, evaluates
both the base `response` and the `edited_response` on all 5 OCEAN traits plus
coherence, then prints a per-sample and aggregate comparison table.

Usage:
    cd /workspace/persona-shattering-lasr
    uv run python scripts/experiments/persona_metrics/ocean_before_after.py

    # Custom path / sample count / provider:
    uv run python scripts/experiments/persona_metrics/ocean_before_after.py \
        --input-path scratch/runs/a-_persona-20260302-121711/exports/editing_training_candidates.jsonl \
        --max-samples 15 \
        --provider openai \
        --model gpt-4o-mini \
        --output-path scratch/ocean_before_after.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from datasets import Dataset
from dotenv import load_dotenv

from scripts.persona_metrics import JudgeLLMConfig, PersonaMetricsConfig, run_persona_metrics
from scripts.utils import setup_logging, write_jsonl

DEFAULT_INPUT = (
    "scratch/runs/a-_persona-20260302-121711/exports/editing_training_candidates.jsonl"
)

OCEAN_METRICS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
ALL_METRICS = OCEAN_METRICS + ["coherence"]
SCORE_KEYS = {m: f"{m}.score" for m in ALL_METRICS}


def load_samples(path: Path, max_samples: int) -> list[dict]:
    """Load up to max_samples records from a JSONL file."""
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) >= max_samples:
                break
    return records


def run_eval(
    records: list[dict],
    response_col: str,
    judge: JudgeLLMConfig,
) -> list[dict]:
    """Run all 6 metrics against `response_col` in records.

    Returns a list of per-sample metric dicts keyed by metric name.
    """
    rows = [
        {"question": r.get("question", ""), "response": r[response_col]}
        for r in records
    ]
    dataset = Dataset.from_list(rows)
    config = PersonaMetricsConfig(
        evaluations=ALL_METRICS,
        response_column="response",
        question_column="question",
        judge=judge,
    )
    result_dataset, _ = run_persona_metrics(config, dataset=dataset)
    return [row["persona_metrics"] for row in result_dataset]


def _score(metrics: dict, name: str) -> int | float:
    return metrics.get(SCORE_KEYS[name], 0)


def _delta_str(before: int | float, after: int | float) -> str:
    d = after - before
    sign = "+" if d > 0 else ""
    return f"{sign}{d}"


def print_comparison(
    records: list[dict],
    before_metrics: list[dict],
    after_metrics: list[dict],
) -> None:
    metric_names = ALL_METRICS
    col_w = 8

    header = f"{'#':>3}  {'question':<40}  " + "  ".join(
        f"{'before':>{col_w}} {'after':>{col_w}} {'Δ':>{col_w}}"
        for _ in metric_names
    )
    subheader = f"{'':>3}  {'':40}  " + "  ".join(
        f"{m:>{col_w*3+2}}" for m in metric_names
    )
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("OCEAN + COHERENCE: BEFORE vs AFTER EDITING")
    print(f"{'='*len(header)}")
    print(subheader)
    print(header)
    print(sep)

    for i, (rec, b, a) in enumerate(zip(records, before_metrics, after_metrics)):
        q = rec.get("question", "")[:38].replace("\n", " ")
        cols = "  ".join(
            f"{_score(b, m):>{col_w}.0f} {_score(a, m):>{col_w}.0f} {_delta_str(_score(b, m), _score(a, m)):>{col_w}}"
            for m in metric_names
        )
        print(f"{i:>3}  {q:<40}  {cols}")

    print(sep)

    # Aggregate means
    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    print(f"\n{'AGGREGATE MEANS':>3}  {'':40}  ", end="")
    for m in metric_names:
        b_avg = mean([_score(bm, m) for bm in before_metrics])
        a_avg = mean([_score(am, m) for am in after_metrics])
        d = a_avg - b_avg
        sign = "+" if d > 0 else ""
        print(
            f"  {b_avg:>{col_w}.2f} {a_avg:>{col_w}.2f} {sign+f'{d:.2f}':>{col_w}}",
            end="",
        )
    print()
    print(f"{'='*len(header)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="OCEAN before/after editing comparison.")
    parser.add_argument(
        "--input-path",
        type=str,
        default=DEFAULT_INPUT,
        help="Path to editing_training_candidates.jsonl",
    )
    parser.add_argument(
        "--max-samples", type=int, default=15,
        help="Number of samples to evaluate (default: 15).",
    )
    parser.add_argument(
        "--provider", type=str, default="openai",
        choices=["openai", "openrouter", "anthropic"],
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument(
        "--output-path", type=str, default=None,
        help="Optional JSONL to save full per-sample results.",
    )
    parser.add_argument("--log-level", type=str, default="WARNING")
    args = parser.parse_args()

    load_dotenv()
    setup_logging(args.log_level)

    input_path = Path(args.input_path)
    if not input_path.is_absolute():
        input_path = project_root / input_path

    print(f"Loading {args.max_samples} samples from {input_path} ...")
    records = load_samples(input_path, args.max_samples)
    print(f"Loaded {len(records)} samples.")

    judge = JudgeLLMConfig(
        provider=args.provider,
        model=args.model,
        max_concurrent=args.max_concurrent,
    )

    print(f"\nEvaluating BASE responses ({len(records)} samples × {len(ALL_METRICS)} metrics) ...")
    t0 = time.perf_counter()
    before_metrics = run_eval(records, "response", judge)
    t1 = time.perf_counter()
    print(f"  done in {t1-t0:.1f}s")

    print(f"\nEvaluating EDITED responses ({len(records)} samples × {len(ALL_METRICS)} metrics) ...")
    after_metrics = run_eval(records, "edited_response", judge)
    t2 = time.perf_counter()
    print(f"  done in {t2-t1:.1f}s  (total: {t2-t0:.1f}s)")

    print_comparison(records, before_metrics, after_metrics)

    if args.output_path:
        out_path = Path(args.output_path)
        if not out_path.is_absolute():
            out_path = project_root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for rec, b, a in zip(records, before_metrics, after_metrics):
            rows.append({
                "sample_id": rec.get("sample_id"),
                "question": rec.get("question"),
                "response": rec.get("response"),
                "edited_response": rec.get("edited_response"),
                "variant_name": rec.get("variant_name"),
                "before": b,
                "after": a,
                "delta": {
                    m: _score(a, m) - _score(b, m)
                    for m in ALL_METRICS
                },
            })
        write_jsonl(rows, out_path)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
