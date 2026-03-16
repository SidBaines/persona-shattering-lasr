#!/usr/bin/env python3
"""Big5 Chat consistency eval: run OCEAN judges N times per sample at temperature 0.9.

For each OCEAN trait, loads samples from wenkai-li/big5_chat and evaluates train_output
using the corresponding judge. The judge receives (train_instruction + train_input) as
the question context. Each sample is judged N times (default 5) at temperature 0.9 to
measure score consistency.

Samples are drawn randomly (shuffled stream) and balanced equally between high and low
trait levels, so you can directly compare judge calibration against known ground-truth labels.

Usage:
    cd /workspace/persona-shattering-lasr
    uv run python scripts/experiments/persona_metrics/big5_chat_consistency_eval.py

    # Custom params:
    uv run python scripts/experiments/persona_metrics/big5_chat_consistency_eval.py \\
        --max-per-level 50 \\
        --n-runs 5 \\
        --provider openrouter \\
        --model openai/gpt-4o-mini \\
        --output-path scratch/big5_consistency.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from datasets import load_dataset
from dotenv import load_dotenv

from scripts.persona_metrics.config import JudgeLLMConfig
from scripts.persona_metrics.metrics.agreeableness import AgreeablenessEvaluation
from scripts.persona_metrics.metrics.conscientiousness import ConscientiousnessEvaluation
from scripts.persona_metrics.metrics.extraversion import ExtraversionEvaluation
from scripts.persona_metrics.metrics.neuroticism import NeuroticismEvaluation
from scripts.persona_metrics.metrics.openness import OpennessEvaluation
from scripts.utils import setup_logging, write_jsonl

logger = logging.getLogger(__name__)

OCEAN_TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

EVALUATOR_CLASSES = {
    "openness": OpennessEvaluation,
    "conscientiousness": ConscientiousnessEvaluation,
    "extraversion": ExtraversionEvaluation,
    "agreeableness": AgreeablenessEvaluation,
    "neuroticism": NeuroticismEvaluation,
}

DEFAULT_OUTPUT = "scratch/big5_chat_consistency_eval.jsonl"
HF_DATASET = "wenkai-li/big5_chat"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

KNOWN_LEVELS = ("high", "low")


def load_big5_samples_balanced(
    traits: list[str],
    max_per_level: int,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Download big5_chat, group by (trait, level), then random-sample max_per_level each.

    Downloads the full dataset once (non-streaming) so that we can do true random
    sampling without any ordering bias. Groups rows by (trait, level) and draws
    max_per_level samples from each group using the given seed.

    Args:
        traits: List of trait names to collect.
        max_per_level: Number of samples per (trait, level) bucket.
            Total per trait = 2 × max_per_level (half high, half low).
        seed: Random seed for reproducible sampling.

    Returns:
        Dict mapping trait name -> list of sample dicts (mixed high and low, shuffled).
    """
    print(f"Downloading {HF_DATASET} (full dataset, non-streaming) ...")
    dataset = load_dataset(HF_DATASET, split="train")

    # Group rows by (trait, level)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    trait_set = set(traits)
    for row in dataset:
        trait = (row.get("trait") or "").lower()
        level = (row.get("level") or "").lower()
        if trait in trait_set and level in KNOWN_LEVELS:
            groups[(trait, level)].append(dict(row))

    rng = random.Random(seed)
    samples: dict[str, list[dict]] = {}
    for t in traits:
        trait_rows: list[dict] = []
        for lv in KNOWN_LEVELS:
            pool = groups.get((t, lv), [])
            n = min(max_per_level, len(pool))
            if n < max_per_level:
                logger.warning("Only %d rows available for (%s, %s), wanted %d.", len(pool), t, lv, max_per_level)
            trait_rows.extend(rng.sample(pool, n))
        rng.shuffle(trait_rows)
        samples[t] = trait_rows
        n_high = sum(1 for r in trait_rows if (r.get("level") or "").lower() == "high")
        n_low = len(trait_rows) - n_high
        print(f"  {t}: {len(trait_rows)} samples (high={n_high}, low={n_low})")

    return samples


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


async def run_consistency_eval_for_trait(
    samples: list[dict],
    trait: str,
    n_runs: int,
    judge_config: JudgeLLMConfig,
) -> list[dict]:
    """Judge each sample n_runs times and return per-sample result dicts.

    Args:
        samples: List of rows from big5_chat for this trait.
        trait: Trait name (must be a key in EVALUATOR_CLASSES).
        n_runs: Number of judge calls per sample.
        judge_config: LLM judge configuration (should have temperature=0.9).

    Returns:
        List of result dicts, one per sample, each containing scores for all runs.
    """
    evaluator_cls = EVALUATOR_CLASSES[trait]
    evaluator = evaluator_cls(judge_config=judge_config)

    score_key = f"{trait}.score"
    reasoning_key = f"{trait}.reasoning"

    results = []
    for i, sample in enumerate(samples):
        # Build question context from instruction + user input
        question = f"{sample['train_instruction']}\n\n{sample['train_input']}"
        response = sample["train_output"]
        expected_level = (sample.get("level") or "").lower()

        # Repeat the same (question, response) n_runs times to check consistency
        batch_responses = [response] * n_runs
        batch_questions = [question] * n_runs

        run_results = await evaluator.evaluate_batch_async(batch_responses, batch_questions)

        scores = [r.get(score_key, 0) for r in run_results]
        reasonings = [r.get(reasoning_key, "") for r in run_results]

        score_mean = statistics.mean(scores)
        score_std = statistics.stdev(scores) if len(scores) > 1 else 0.0

        record = {
            "sample_idx": i,
            "trait": trait,
            "expected_level": expected_level,
            "train_instruction": sample["train_instruction"],
            "train_input": sample["train_input"],
            "train_output": sample["train_output"],
            "scores": scores,
            "score_mean": round(score_mean, 3),
            "score_std": round(score_std, 3),
            "score_min": min(scores),
            "score_max": max(scores),
            "reasonings": reasonings,
        }
        results.append(record)

        logger.info(
            "[%s] sample %d/%d  level=%s  scores=%s  mean=%.1f  std=%.2f",
            trait, i + 1, len(samples), expected_level, scores, score_mean, score_std,
        )

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_results(all_results: list[dict]) -> None:
    """Print per-sample table and aggregate summary."""
    sep = "=" * 110
    print(f"\n{sep}")
    print("BIG5 CHAT CONSISTENCY EVAL  (temperature=0.9)")
    print(sep)
    print(f"{'#':>3}  {'trait':<20}  {'level':<6}  {'mean':>6}  {'std':>5}  {'min':>4}  {'max':>4}  scores")
    print("-" * 110)

    for r in all_results:
        print(
            f"{r['sample_idx']:>3}  {r['trait']:<20}  {r['expected_level']:<6}  "
            f"{r['score_mean']:>6.2f}  {r['score_std']:>5.2f}  "
            f"{r['score_min']:>4}  {r['score_max']:>4}  {r['scores']}"
        )

    print(sep)

    # Aggregate by trait + level
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in all_results:
        groups[(r["trait"], r["expected_level"])].extend(r["scores"])

    print("\nAGGREGATE BY TRAIT + EXPECTED LEVEL  (mean/std across all runs for that group):")
    print(f"  {'trait':<20}  {'level':<6}  {'mean':>6}  {'std':>5}  {'n':>5}")
    print("-" * 60)
    for (trait, level), scores in sorted(groups.items()):
        agg_mean = statistics.mean(scores)
        agg_std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        print(
            f"  {trait:<20}  {level:<6}  "
            f"{agg_mean:>6.2f}  {agg_std:>5.2f}  {len(scores):>5}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Big5 Chat consistency eval: run OCEAN judges N times per sample."
    )
    parser.add_argument(
        "--max-per-level",
        type=int,
        default=50,
        help="Samples per (trait, level) bucket — total per trait = 2× this (default: 50).",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Judge runs per sample (default: 5).",
    )
    parser.add_argument(
        "--traits",
        type=str,
        nargs="+",
        default=OCEAN_TRAITS,
        choices=OCEAN_TRAITS,
        help="Traits to evaluate (default: all 5 OCEAN traits).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openrouter",
        choices=["openai", "openrouter", "anthropic"],
    )
    parser.add_argument("--model", type=str, default="openai/gpt-4o-mini")
    parser.add_argument("--max-concurrent", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"JSONL path for saved results (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument("--log-level", type=str, default="WARNING")
    args = parser.parse_args()

    load_dotenv()
    setup_logging(args.log_level)

    judge_config = JudgeLLMConfig(
        provider=args.provider,
        model=args.model,
        temperature=0.9,
        max_tokens=512,
        max_concurrent=args.max_concurrent,
    )

    samples_per_trait = args.max_per_level * 2
    total_samples = samples_per_trait * len(args.traits)
    total_calls = total_samples * args.n_runs
    print(
        f"\nConfig: traits={args.traits}"
        f"  {args.max_per_level} high + {args.max_per_level} low per trait"
        f"  = {total_samples} samples total"
        f"  × {args.n_runs} runs = {total_calls} judge calls"
        f"\n        provider={args.provider}  model={args.model}  temperature=0.9"
    )

    # Load samples for all requested traits in a single streaming pass
    t0 = time.perf_counter()
    trait_samples = load_big5_samples_balanced(args.traits, args.max_per_level, seed=args.seed)
    print(f"Loaded samples in {time.perf_counter() - t0:.1f}s")

    all_results: list[dict] = []

    for trait in args.traits:
        samples = trait_samples[trait]
        if not samples:
            print(f"\nNo samples found for trait '{trait}', skipping.")
            continue

        n_high = sum(1 for s in samples if (s.get("level") or "").lower() == "high")
        n_low = len(samples) - n_high
        print(
            f"\nEvaluating '{trait}': {len(samples)} samples (high={n_high}, low={n_low})"
            f" × {args.n_runs} runs = {len(samples) * args.n_runs} judge calls ..."
        )
        t1 = time.perf_counter()
        results = asyncio.run(
            run_consistency_eval_for_trait(samples, trait, args.n_runs, judge_config)
        )
        elapsed = time.perf_counter() - t1
        print(f"  done in {elapsed:.1f}s")
        all_results.extend(results)

    print_results(all_results)

    out_path = Path(args.output_path)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(all_results, out_path)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
