#!/usr/bin/env python3
"""Stage 3: Evaluate baseline and trait responses with persona metrics.

Runs the OCEAN LLM judge on both sets of responses and prints a comparison
to verify the system prompt is producing measurably different trait scores.

Usage:
    uv run python scripts/experiments/ocean_model_prompt_generated_pipeline/03_evaluate.py
"""

from __future__ import annotations

from dotenv import load_dotenv

from config import (
    EVALUATION,
    JUDGE_MODEL,
    JUDGE_PROVIDER,
    RUN_DIR,
    RUN_ID,
    TRAIT_LABEL,
)

from scripts.common.config import DatasetConfig
from scripts.datasets import load_dataset_from_config
from scripts.persona_metrics import JudgeLLMConfig, PersonaMetricsConfig, run_persona_metrics

load_dotenv()


def _print_aggregates(title: str, aggregates: dict[str, object]) -> None:
    print(f"\n{title}")
    for key, value in sorted(aggregates.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


print(f"\n{'=' * 60}")
print(f"STAGE 3: EVALUATION — {TRAIT_LABEL} pipeline")
print(f"Run ID: {RUN_ID}")
print(f"Evaluation: {EVALUATION}")
print(f"{'=' * 60}\n")

judge = JudgeLLMConfig(provider=JUDGE_PROVIDER, model=JUDGE_MODEL)

# ── Baseline evaluation ──────────────────────────────────────────────────

baseline_responses_path = RUN_DIR / "exports" / "baseline_responses.jsonl"
baseline_dataset = load_dataset_from_config(
    DatasetConfig(source="local", path=str(baseline_responses_path))
)

baseline_eval_config = PersonaMetricsConfig(
    evaluations=[EVALUATION],
    response_column="response",
    question_column="question",
    judge=judge,
    output_path=RUN_DIR / "exports" / "baseline_metrics.jsonl",
)
_, baseline_eval_result = run_persona_metrics(baseline_eval_config, dataset=baseline_dataset)
_print_aggregates("Baseline metric aggregates", baseline_eval_result.aggregates)

# ── Trait evaluation ─────────────────────────────────────────────────────

trait_responses_path = RUN_DIR / "exports" / "ocean_prompted_responses.jsonl"
trait_dataset = load_dataset_from_config(
    DatasetConfig(source="local", path=str(trait_responses_path))
)

trait_eval_config = PersonaMetricsConfig(
    evaluations=[EVALUATION],
    response_column="response",
    question_column="question",
    judge=judge,
    output_path=RUN_DIR / "exports" / f"{TRAIT_LABEL}_metrics.jsonl",
)
_, trait_eval_result = run_persona_metrics(trait_eval_config, dataset=trait_dataset)
_print_aggregates(f"{TRAIT_LABEL} metric aggregates", trait_eval_result.aggregates)

# ── Comparison ───────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("METRIC COMPARISON")
print(f"{'=' * 60}")
_print_aggregates("  Baseline", baseline_eval_result.aggregates)
_print_aggregates(f"  {TRAIT_LABEL}", trait_eval_result.aggregates)
print(f"{'=' * 60}\n")
