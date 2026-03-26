#!/usr/bin/env python3
"""Score golden calibration datasets with a panel of LLM judges.

Loads the hand-crafted golden datasets from ``data/judge_calibration/``,
runs each judge N times per example, and computes:

  - Gold vs judge: Pearson r, Spearman r, MAE, within-one rate, exact match,
    quadratic weighted kappa
  - Self-consistency: pairwise agreement across the N repeated runs
  - Ordinal Krippendorff's alpha across the N runs

Outputs are written to::

    scratch/golden_calibration/<run_key>/
        raw/<trait>_run_<n>.jsonl       ← per-run per-item scored results
        analysis/
            <trait>_summary.json        ← metrics for this trait
            combined_summary.json       ← all traits aggregated
        plots/
            <trait>_confusion.png       ← gold vs median judge heatmap

Usage::

    # All traits, default judge (gemini-flash-2.0 ×3)
    uv run python scripts_dev/persona_metrics/llm_judge/golden_calibration.py

    # Single trait, dry run
    uv run python scripts_dev/persona_metrics/llm_judge/golden_calibration.py \\
        --trait neuroticism --dry-run

    # Custom judge and repeat count
    uv run python scripts_dev/persona_metrics/llm_judge/golden_calibration.py \\
        --model openai/gpt-4o-mini --repeats 1

    # Skip already-scored traits (uses existing raw files)
    uv run python scripts_dev/persona_metrics/llm_judge/golden_calibration.py --resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.judge_calibration import (
    quadratic_weighted_agreement,
    summarize_pair,
)
from src_dev.persona_metrics.llm_judge_agreement import _krippendorff_alpha_ordinal
from src_dev.persona_metrics.registry import get_persona_metric

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GOLDEN_DATA_DIR = project_root / "data" / "judge_calibration"
OUTPUT_ROOT = project_root / "scratch" / "golden_calibration"

# Mapping: trait name in JSONL → registered metric name
TRAIT_TO_METRIC: dict[str, str] = {
    "neuroticism": "neuroticism_v2",
    "openness": "openness_v2",
    "conscientiousness": "conscientiousness_v2",
    "extraversion": "extraversion_v2",
    "agreeableness": "agreeableness_v2",
    "coherence": "better_coherence_judge",
}

SCORE_RANGE: dict[str, tuple[int, int]] = {
    "neuroticism": (-4, 4),
    "openness": (-4, 4),
    "conscientiousness": (-4, 4),
    "extraversion": (-4, 4),
    "agreeableness": (-4, 4),
    "coherence": (0, 10),
}

# ---------------------------------------------------------------------------
# Loading golden data
# ---------------------------------------------------------------------------


def load_golden(trait: str) -> list[dict[str, Any]]:
    """Load golden calibration items for one trait from JSONL."""
    path = GOLDEN_DATA_DIR / f"{trait}.jsonl"
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


async def score_trait_async(
    trait: str,
    items: list[dict[str, Any]],
    judge_config: JudgeLLMConfig,
    *,
    run_index: int,
) -> list[dict[str, Any]]:
    """Run the judge once over all items for one trait.

    Args:
        trait: Trait name (e.g. "neuroticism").
        items: Golden items loaded from JSONL.
        judge_config: Judge LLM config.
        run_index: Which repeat this is (0-indexed), used for logging.

    Returns:
        List of dicts: original item fields + judge score + judge reasoning.
    """
    metric_name = TRAIT_TO_METRIC[trait]
    metric = get_persona_metric(metric_name, judge_config=judge_config)

    responses = [item["response"] for item in items]
    questions = [item["question"] for item in items]

    print(f"  [{trait}] run {run_index + 1}: scoring {len(items)} items ...", flush=True)
    results = await metric.evaluate_batch_async(responses, questions)

    score_key = f"{metric_name}.score"
    reasoning_key = f"{metric_name}.reasoning"

    output = []
    for item, result in zip(items, results):
        output.append(
            {
                "id": item["id"],
                "trait": item["trait"],
                "question": item["question"],
                "response": item["response"],
                "gold_score": item["gold_score"],
                "notes": item.get("notes", ""),
                "judge_score": result.get(score_key),
                "judge_reasoning": result.get(reasoning_key, ""),
                "run_index": run_index,
            }
        )
    return output


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_trait(
    trait: str,
    runs: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Compute calibration metrics for one trait across all scoring runs.

    Args:
        trait: Trait name.
        runs: List of run results, each a list of per-item dicts.

    Returns:
        Analysis dict with gold_vs_judge, self_consistency, and per_item fields.
    """
    score_min, score_max = SCORE_RANGE[trait]
    n_runs = len(runs)

    # Align by id
    ids = [item["id"] for item in runs[0]]
    gold_scores = [item["gold_score"] for item in runs[0]]

    # Per run scores, keyed by id
    run_scores_by_id: dict[str, list[int | None]] = defaultdict(list)
    for run in runs:
        for item in run:
            run_scores_by_id[item["id"]].append(item["judge_score"])

    # Median judge score per item across runs
    median_judge: list[float | None] = []
    for item_id in ids:
        scores = [s for s in run_scores_by_id[item_id] if s is not None]
        median_judge.append(statistics.median(scores) if scores else None)

    # --- Gold vs judge (using median across runs) ---
    gold_vs_median = summarize_pair(gold_scores, median_judge)
    valid_gold = [g for g, m in zip(gold_scores, median_judge) if m is not None]
    valid_median = [int(round(m)) for g, m in zip(gold_scores, median_judge) if m is not None]
    qwk_gold = quadratic_weighted_agreement(
        valid_gold, valid_median, score_min=score_min, score_max=score_max
    )

    # --- Per-run gold vs judge ---
    per_run_gold_stats = []
    for run in runs:
        run_judge = [item["judge_score"] for item in run]
        stats = summarize_pair(gold_scores, run_judge)
        valid_g = [g for g, j in zip(gold_scores, run_judge) if j is not None]
        valid_j = [j for g, j in zip(gold_scores, run_judge) if j is not None]
        stats["qwk"] = quadratic_weighted_agreement(
            valid_g, valid_j, score_min=score_min, score_max=score_max
        )
        per_run_gold_stats.append(stats)

    # --- Self-consistency across runs (pairwise) ---
    pairwise_consistency = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            run_i = [runs[i][k]["judge_score"] for k in range(len(runs[i]))]
            run_j = [runs[j][k]["judge_score"] for k in range(len(runs[j]))]
            pair_stats = summarize_pair(run_i, run_j)
            valid_i = [s for s, t in zip(run_i, run_j) if s is not None and t is not None]
            valid_j_scores = [t for s, t in zip(run_i, run_j) if s is not None and t is not None]
            pair_stats["qwk"] = quadratic_weighted_agreement(
                valid_i, valid_j_scores, score_min=score_min, score_max=score_max
            )
            pairwise_consistency.append({"run_i": i, "run_j": j, **pair_stats})

    # Krippendorff alpha across runs (each item has one rating per run)
    item_ratings = [
        [s for s in run_scores_by_id[item_id] if s is not None]
        for item_id in ids
    ]
    alpha = _krippendorff_alpha_ordinal(item_ratings, score_min=score_min, score_max=score_max)

    # Summary of self-consistency
    if pairwise_consistency:
        mean_consistency: dict[str, float] = {}
        for key in ["pearson", "spearman", "mae", "within_one", "exact", "qwk"]:
            values = [p[key] for p in pairwise_consistency if not math.isnan(float(p[key]))]
            if values:
                mean_consistency[f"mean_{key}"] = statistics.mean(values)
    else:
        mean_consistency = {}

    # --- Per-item details ---
    per_item = []
    for idx, item_id in enumerate(ids):
        run_s = run_scores_by_id[item_id]
        valid_s = [s for s in run_s if s is not None]
        per_item.append(
            {
                "id": item_id,
                "gold_score": gold_scores[idx],
                "judge_scores_per_run": run_s,
                "judge_median": median_judge[idx],
                "judge_mean": statistics.mean(valid_s) if valid_s else None,
                "judge_std": statistics.stdev(valid_s) if len(valid_s) >= 2 else 0.0,
                "error": (
                    abs(gold_scores[idx] - round(median_judge[idx]))
                    if median_judge[idx] is not None
                    else None
                ),
            }
        )

    return {
        "trait": trait,
        "n_items": len(ids),
        "n_runs": n_runs,
        "score_range": [score_min, score_max],
        "gold_vs_median_judge": {**gold_vs_median, "qwk": qwk_gold},
        "per_run_gold_stats": per_run_gold_stats,
        "self_consistency": {
            "krippendorff_alpha": alpha,
            **mean_consistency,
            "pairwise": pairwise_consistency,
        },
        "per_item": per_item,
    }


def print_summary(analysis: dict[str, Any]) -> None:
    """Print a concise human-readable summary for one trait."""
    gvm = analysis["gold_vs_median_judge"]
    sc = analysis["self_consistency"]
    trait = analysis["trait"]
    print(f"\n  {'─' * 50}")
    print(f"  {trait.upper():30s}  ({analysis['n_items']} items × {analysis['n_runs']} runs)")
    print(f"  {'─' * 50}")
    print(f"  Gold vs judge (median):   Spearman r = {gvm.get('spearman', float('nan')):.3f}   "
          f"MAE = {gvm.get('mae', float('nan')):.2f}   "
          f"QWK = {gvm.get('qwk', float('nan')):.3f}   "
          f"within-1 = {gvm.get('within_one', float('nan')):.0%}")
    if analysis["n_runs"] > 1:
        print(f"  Self-consistency:         Krippendorff α = {sc.get('krippendorff_alpha', float('nan')):.3f}   "
              f"mean QWK = {sc.get('mean_qwk', float('nan')):.3f}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_confusion(
    analysis: dict[str, Any],
    out_path: Path,
) -> None:
    """Write a gold-vs-median confusion heatmap PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [plot] matplotlib not available, skipping plot.")
        return

    score_min, score_max = analysis["score_range"]
    labels = list(range(score_min, score_max + 1))
    n = len(labels)
    idx = {v: i for i, v in enumerate(labels)}

    matrix = np.zeros((n, n), dtype=int)
    for item in analysis["per_item"]:
        g = item["gold_score"]
        m = item["judge_median"]
        if m is not None:
            matrix[idx[g]][idx[int(round(m))]] += 1

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Judge median score")
    ax.set_ylabel("Gold score")
    ax.set_title(f"{analysis['trait']} — gold vs judge (n={analysis['n_items']}, runs={analysis['n_runs']})")
    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0:
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def run_trait_async(
    trait: str,
    judge_config: JudgeLLMConfig,
    run_dir: Path,
    *,
    repeats: int,
    resume: bool,
) -> dict[str, Any]:
    """Score one trait and return analysis dict."""
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    items = load_golden(trait)
    runs: list[list[dict[str, Any]]] = []

    for run_idx in range(repeats):
        raw_path = raw_dir / f"{trait}_run_{run_idx}.jsonl"
        if resume and raw_path.exists():
            print(f"  [{trait}] run {run_idx + 1}: resuming from {raw_path}")
            run_results = [json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()]
        else:
            run_results = await score_trait_async(trait, items, judge_config, run_index=run_idx)
            lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in run_results)
            raw_path.write_text(lines + "\n", encoding="utf-8")

        runs.append(run_results)

    return analyze_trait(trait, runs)


async def main_async(args: argparse.Namespace) -> None:
    load_dotenv()

    traits = (
        [args.trait] if args.trait != "all" else list(TRAIT_TO_METRIC.keys())
    )

    judge_config = JudgeLLMConfig(
        provider="openrouter",
        model=args.model,
        temperature=args.temperature,
        max_concurrent=args.max_concurrent,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_key = f"{args.model.replace('/', '_')}__r{args.repeats}__{ts}"
    run_dir = OUTPUT_ROOT / run_key
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"\nRun key  : {run_key}")
    print(f"Output   : {run_dir}")
    print(f"Judge    : {args.model} (temp={args.temperature}, repeats={args.repeats})")
    print(f"Traits   : {traits}")

    if args.dry_run:
        for trait in traits:
            items = load_golden(trait)
            print(f"  {trait}: {len(items)} items × {args.repeats} runs = {len(items) * args.repeats} judge calls")
        total = sum(len(load_golden(t)) for t in traits) * args.repeats
        print(f"\nTotal judge calls: {total}")
        return

    all_analyses: dict[str, dict[str, Any]] = {}
    for trait in traits:
        print(f"\nScoring: {trait}")
        analysis = await run_trait_async(
            trait,
            judge_config,
            run_dir,
            repeats=args.repeats,
            resume=args.resume,
        )
        all_analyses[trait] = analysis

        # Write per-trait analysis
        trait_path = analysis_dir / f"{trait}_summary.json"
        trait_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")

        # Plot
        _plot_confusion(analysis, plots_dir / f"{trait}_confusion.png")

        print_summary(analysis)

    # Combined summary
    combined = {
        "run_key": run_key,
        "model": args.model,
        "temperature": args.temperature,
        "repeats": args.repeats,
        "traits": traits,
        "per_trait": {
            trait: {
                "gold_vs_median_judge": a["gold_vs_median_judge"],
                "self_consistency_alpha": a["self_consistency"]["krippendorff_alpha"],
                "self_consistency_mean_qwk": a["self_consistency"].get("mean_qwk"),
            }
            for trait, a in all_analyses.items()
        },
    }
    combined_path = analysis_dir / "combined_summary.json"
    combined_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n\nDone. Results at: {run_dir}")
    print(f"Combined summary: {combined_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score golden calibration datasets with an LLM judge."
    )
    parser.add_argument(
        "--trait",
        choices=list(TRAIT_TO_METRIC.keys()) + ["all"],
        default="all",
        help="Trait to score, or 'all' (default).",
    )
    parser.add_argument(
        "--model",
        default="google/gemini-2.0-flash-001",
        help="OpenRouter model string (default: google/gemini-2.0-flash-001).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Judge temperature for self-consistency measurement (default: 0.7).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of times to score each example (default: 3).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=15,
        help="Max concurrent judge calls (default: 15).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing raw JSONL files if present, skipping API calls.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print call counts without making any API calls.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
