#!/usr/bin/env python3
"""Judge calibration script.

Evaluates an LLM judge along three dimensions:

1. Human correlation   — judge scores vs. a reference score set.
                         Falls back to expected_score in heldout.jsonl when no
                         human ratings are available. Rating CSVs are auto-discovered
                         from <judge>/ratings/*.csv; additional paths can be passed
                         via --human-scores. Inter-rater agreement reported when
                         multiple raters are present.

2. Self-consistency    — run the same judge n_runs times at temperature=0.9.
                         Reports std per item and overall mean std.

3. Cross-model         — run the heldout set through each model in --models.
                         Reports per-model human correlation and pairwise agreement.

A pass/fail scorecard is printed at the end summarising all criteria.

Usage:
    cd persona-shattering-lasr

    # Quick check (uses expected_score as reference, ratings/ auto-discovered):
    python dump/llm_judges/calibrate.py \\
        --judge neuroticism \\
        --models openai/gpt-4o-mini \\
        --provider openrouter

    # With additional rating CSVs and consistency check:
    python dump/llm_judges/calibrate.py \\
        --judge neuroticism \\
        --models openai/gpt-4o-mini anthropic/claude-3.5-haiku \\
        --provider openrouter \\
        --human-scores path/to/extra_rater.csv \\
        --n-runs 3
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.metrics.llm_judge_base import LLMJudgeMetric

JUDGES_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def judge_dir(judge_name: str) -> Path:
    """Return the directory containing judge.py and heldout.jsonl."""
    for candidate in [
        JUDGES_DIR / "ocean" / judge_name,
        JUDGES_DIR / judge_name,
    ]:
        if (candidate / "heldout.jsonl").exists():
            return candidate
    raise FileNotFoundError(f"No judge directory found for '{judge_name}'")


def load_judge_class(judge_name: str) -> type[LLMJudgeMetric]:
    """Import the judge class from ocean/<judge>/judge.py or <judge>/judge.py."""
    module_path = judge_dir(judge_name) / "judge.py"
    spec = importlib.util.spec_from_file_location(f"judge_{judge_name}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attr in vars(module).values():
        if (
            isinstance(attr, type)
            and issubclass(attr, LLMJudgeMetric)
            and attr is not LLMJudgeMetric
            and isinstance(getattr(attr, "name", None), str)
        ):
            return attr
    raise FileNotFoundError(
        f"No LLMJudgeMetric subclass found in {module_path}"
    )


def load_heldout(judge_name: str) -> list[dict]:
    """Load heldout.jsonl for the given judge."""
    path = judge_dir(judge_name) / "heldout.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def find_ratings(judge_name: str) -> list[Path]:
    """Return all CSV rating files from the judge's ratings/ directory."""
    ratings_dir = judge_dir(judge_name) / "ratings"
    if not ratings_dir.exists():
        return []
    return sorted(ratings_dir.glob("*.csv"))


def load_human_scores_from_csv(csv_path: Path, items: list[dict]) -> dict[str, list[int | None]]:
    """Load one filled rating CSV and align scores with heldout items by id.

    Returns:
        Dict mapping rater name -> list of scores aligned with items.
        Scores are None for items not found in the CSV or left blank.
    """
    id_to_row: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row.get("id", "").strip()
            if item_id and item_id != "SCALE":
                id_to_row[item_id] = row

    if not id_to_row:
        return {}
    sample_row = next(iter(id_to_row.values()))
    rater_cols = [k for k in sample_row if k.startswith("score_")]
    if not rater_cols:
        raise ValueError(
            f"No 'score_*' columns found in {csv_path}. "
            "Expected columns like 'score_alice', 'score_bob', etc."
        )

    rater_scores: dict[str, list[int | None]] = {col: [] for col in rater_cols}
    missing_ids = []
    for item in items:
        item_id = item["id"]
        row = id_to_row.get(item_id)
        if row is None:
            missing_ids.append(item_id)
            for col in rater_cols:
                rater_scores[col].append(None)
            continue
        for col in rater_cols:
            val = row.get(col, "").strip()
            try:
                rater_scores[col].append(int(val))
            except (ValueError, TypeError):
                rater_scores[col].append(None)

    if missing_ids:
        print(f"  Warning: {len(missing_ids)} item id(s) not found in {csv_path.name}: {missing_ids}")

    return {col.removeprefix("score_"): scores for col, scores in rater_scores.items()}


def load_human_scores(csv_paths: list[Path], items: list[dict]) -> dict[str, list[int | None]]:
    """Merge rater scores from one or more filled rating CSVs.

    Each CSV contributes its own score_* columns. Duplicate rater names across
    files raise an error.
    """
    merged: dict[str, list[int | None]] = {}
    for path in csv_paths:
        scores = load_human_scores_from_csv(path, items)
        duplicates = set(scores) & set(merged)
        if duplicates:
            raise ValueError(
                f"Duplicate rater name(s) {duplicates} found in {path}. "
                "Each rater CSV must have a unique score_<name> column."
            )
        merged.update(scores)
    return merged


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def pearson_r(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / denom if denom > 1e-12 else float("nan")


def spearman_r(xs: list[float], ys: list[float]) -> float:
    def rank(vals: list[float]) -> list[float]:
        sorted_vals = sorted(enumerate(vals), key=lambda x: x[1])
        ranks = [0.0] * len(vals)
        for rank_i, (orig_i, _) in enumerate(sorted_vals):
            ranks[orig_i] = float(rank_i + 1)
        return ranks

    return pearson_r(rank(xs), rank(ys))


def mae(xs: list[float], ys: list[float]) -> float:
    return statistics.mean(abs(x - y) for x, y in zip(xs, ys))


def _valid_pairs(
    ref: list[int | None], pred: list[int | None]
) -> tuple[list[float], list[float]]:
    pairs = [(r, p) for r, p in zip(ref, pred) if r is not None and p is not None]
    return [p[0] for p in pairs], [p[1] for p in pairs]


# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------


async def run_once(
    judge_cls: type[LLMJudgeMetric],
    items: list[dict],
    config: JudgeLLMConfig,
) -> tuple[list[int | None], list[dict]]:
    """Run judge once. Returns (scores, raw_results) where raw_results include reasoning/evidence."""
    judge = judge_cls(judge_config=config)
    score_key = f"{judge.name}.score"
    error_sentinel = judge.score_error
    responses = [item["response"] for item in items]
    questions = [item["question"] for item in items]
    raw = await judge.evaluate_batch_async(responses, questions)
    scores = [None if (s := r.get(score_key)) == error_sentinel else s for r in raw]
    return scores, raw


async def run_consistency(
    judge_cls: type[LLMJudgeMetric],
    items: list[dict],
    config: JudgeLLMConfig,
    n_runs: int,
) -> list[list[int | None]]:
    high_temp = config.model_copy(update={"temperature": 0.9})
    runs = await asyncio.gather(*[run_once(judge_cls, items, high_temp) for _ in range(n_runs)])
    # transpose: runs[run][item] -> per_item[item][run]  (only scores, not raw)
    return [[runs[run][0][item] for run in range(n_runs)] for item in range(len(items))]


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

SEP = "=" * 90


def _fmt(s: int | None) -> str:
    return "ERR" if s is None else f"{s:+d}"


def _corr_stats(ref: list[int | None], pred: list[int | None]) -> tuple[float, float, float, int]:
    """Return (pearson, spearman, mae, n_valid)."""
    rv, pv = _valid_pairs(ref, pred)
    if not rv:
        return float("nan"), float("nan"), float("nan"), 0
    return pearson_r(rv, pv), spearman_r(rv, pv), mae(rv, pv), len(rv)


def print_human_correlation(
    items: list[dict],
    judge_scores: list[int | None],
    reference: list[int | None],
    ref_label: str,
    model_label: str,
) -> None:
    pr, sr, me, n = _corr_stats(reference, judge_scores)
    errors = sum(1 for s in judge_scores if s is None)

    print(f"\n{SEP}")
    print(f"HUMAN CORRELATION  judge={model_label}  reference={ref_label}")
    print(SEP)
    print(f"  Pearson r  : {pr:+.3f}  (n={n})")
    print(f"  Spearman r : {sr:+.3f}")
    print(f"  MAE        : {me:.2f}")
    print(f"  Judge errors: {errors}/{len(items)}")
    print()
    print(f"  {'id':<12}  {'cat':<26}  {'ref':>5}  {'judge':>6}  {'delta':>6}")
    print("  " + "-" * 62)
    for item, ref_s, judge_s in zip(items, reference, judge_scores):
        delta = (judge_s - ref_s) if (judge_s is not None and ref_s is not None) else None
        flag = "  ← confound" if item.get("category", "").startswith("confound") else ""
        print(
            f"  {item['id']:<12}  {item.get('category',''):<26}  "
            f"{_fmt(ref_s):>5}  {_fmt(judge_s):>6}  "
            f"{_fmt(delta):>6}{flag}"
        )
    print(SEP)


def print_inter_rater(
    items: list[dict],
    rater_scores: dict[str, list[int | None]],
) -> None:
    raters = list(rater_scores.keys())
    print(f"\n{SEP}")
    print("INTER-RATER AGREEMENT")
    print(SEP)
    print(f"  {'rater_A':<20}  {'rater_B':<20}  {'pearson':>8}  {'spearman':>9}  {'mae':>6}  n")
    print("  " + "-" * 72)
    for i, r1 in enumerate(raters):
        for r2 in raters[i + 1:]:
            pr, sr, me, n = _corr_stats(rater_scores[r1], rater_scores[r2])
            print(f"  {r1:<20}  {r2:<20}  {pr:>+8.3f}  {sr:>+9.3f}  {me:>6.2f}  {n}")

    # Per-item table
    print()
    header = f"  {'id':<12}" + "".join(f"  {r:>8}" for r in raters)
    print(header)
    print("  " + "-" * (12 + 10 * len(raters)))
    for i, item in enumerate(items):
        row = f"  {item['id']:<12}"
        vals = [rater_scores[r][i] for r in raters]
        valid_vals = [v for v in vals if v is not None]
        row += "".join(f"  {_fmt(v):>8}" for v in vals)
        if len(valid_vals) > 1:
            spread = max(valid_vals) - min(valid_vals)
            row += f"  spread={spread}"
        print(row)
    print(SEP)


def print_consistency(items: list[dict], per_item_runs: list[list[int | None]]) -> None:
    print(f"\n{SEP}")
    print("SELF-CONSISTENCY  (temperature=0.9)")
    print(SEP)
    print(f"  {'id':<12}  {'expected':>8}  {'mean':>6}  {'std':>5}  {'min':>4}  {'max':>4}  scores")
    print("  " + "-" * 82)
    all_stds = []
    for item, runs in zip(items, per_item_runs):
        valid = [s for s in runs if s is not None]
        if not valid:
            print(f"  {item['id']:<12}  {item['expected_score']:>+8d}  ERR")
            continue
        mean = statistics.mean(valid)
        std = statistics.stdev(valid) if len(valid) > 1 else 0.0
        all_stds.append(std)
        scores_str = "  ".join(f"{s:+d}" for s in runs)
        print(
            f"  {item['id']:<12}  {item['expected_score']:>+8d}  {mean:>+6.2f}  {std:>5.2f}  "
            f"{min(valid):>+4d}  {max(valid):>+4d}  {scores_str}"
        )
    if all_stds:
        print(f"\n  Overall mean std: {statistics.mean(all_stds):.3f}")
    print(SEP)


def print_cross_model_summary(
    items: list[dict],
    model_scores: dict[str, list[int | None]],
    reference: list[int | None],
    ref_label: str,
) -> None:
    print(f"\n{SEP}")
    print(f"CROSS-MODEL SUMMARY  (reference: {ref_label})")
    print(SEP)
    print(f"  {'model':<40}  {'pearson':>8}  {'spearman':>9}  {'mae':>6}  {'errors':>7}")
    print("  " + "-" * 78)
    for model_label, scores in model_scores.items():
        pr, sr, me, n = _corr_stats(reference, scores)
        errs = sum(1 for s in scores if s is None)
        print(f"  {model_label:<40}  {pr:>+8.3f}  {sr:>+9.3f}  {me:>6.2f}  {errs:>5}/{len(items)}")

    if len(model_scores) > 1:
        print(f"\n  Pairwise MAE between models:")
        names = list(model_scores.keys())
        for i, m1 in enumerate(names):
            for m2 in names[i + 1:]:
                _, _, me, n = _corr_stats(model_scores[m1], model_scores[m2])
                print(f"    {m1} vs {m2}: MAE={me:.2f} (n={n})")
    print(SEP)


# ---------------------------------------------------------------------------
# Pass/fail scorecard
# ---------------------------------------------------------------------------

# Thresholds — adjust as the heldout set matures.
# Judges can override individual thresholds by setting a `calibration_thresholds`
# class attribute (dict with the same structure as _THRESHOLDS).
_THRESHOLDS = {
    "pearson_r":       (0.90, "≥ 0.90"),
    "spearman_r":      (0.85, "≥ 0.85"),
    "mae":             (1.00, "≤ 1.00", True),   # True = lower is better
    "consistency_std": (0.50, "≤ 0.50 mean", True),
    "inter_model_mae": (1.00, "≤ 1.00", True),
}


def _get_thresholds(judge_cls: type[LLMJudgeMetric]) -> dict:
    """Return thresholds for this judge, merging any per-judge overrides."""
    overrides = getattr(judge_cls, "calibration_thresholds", {})
    return {**_THRESHOLDS, **overrides}


def _check(key: str, value: float, thresholds: dict | None = None) -> bool:
    entry = (thresholds or _THRESHOLDS)[key]
    lower_is_better = len(entry) == 3 and entry[2]
    return value <= entry[0] if lower_is_better else value >= entry[0]


def compute_scorecard(
    items: list[dict],
    model_scores: dict[str, list[int | None]],
    reference: list[int | None],
    ref_label: str,
    consistency_runs: list[list[int | None]] | None,
    consistency_model: str | None,
    thresholds: dict | None = None,
) -> dict:
    """Compute scorecard metrics. Returns a serialisable dict."""
    th = thresholds or _THRESHOLDS
    result: dict = {"reference": ref_label, "models": {}, "consistency": None, "inter_model": {}}

    for model_label, scores in model_scores.items():
        pr, sr, me, n = _corr_stats(reference, scores)
        confound_pairs = [(item, s) for item, s in zip(items, scores)
                          if item.get("category", "").startswith("confound")]
        n_confound_correct = sum(
            1 for item, s in confound_pairs
            if s is not None and s == item.get("expected_score", 0)
        )
        confound_acc = n_confound_correct / len(confound_pairs) if confound_pairs else None

        result["models"][model_label] = {
            "pearson_r":        round(pr, 4) if n >= 2 else None,
            "spearman_r":       round(sr, 4) if n >= 2 else None,
            "mae":              round(me, 4) if n >= 2 else None,
            "n_valid":          n,
            "confound_correct": n_confound_correct if confound_pairs else None,
            "confound_total":   len(confound_pairs) if confound_pairs else None,
            "confound_acc":     round(confound_acc, 4) if confound_acc is not None else None,
            "pass": {
                "pearson_r":  _check("pearson_r", pr, th) if n >= 2 else None,
                "spearman_r": _check("spearman_r", sr, th) if n >= 2 else None,
                "mae":        _check("mae", me, th) if n >= 2 else None,
            },
        }

    if consistency_runs is not None:
        stds = [
            statistics.stdev(valid)
            for runs in consistency_runs
            if len(valid := [s for s in runs if s is not None]) > 1
        ]
        mean_std = round(statistics.mean(stds), 4) if stds else None
        result["consistency"] = {
            "model": consistency_model,
            "n_runs": len(consistency_runs[0]) if consistency_runs else 0,
            "mean_std": mean_std,
            "pass": _check("consistency_std", mean_std, th) if mean_std is not None else None,
        }

    model_names = list(model_scores.keys())
    for i, m1 in enumerate(model_names):
        for m2 in model_names[i + 1:]:
            _, _, me, n = _corr_stats(model_scores[m1], model_scores[m2])
            result["inter_model"][f"{m1} vs {m2}"] = {
                "mae": round(me, 4) if n else None,
                "pass": _check("inter_model_mae", me, th) if n else None,
            }

    return result


def print_scorecard(scorecard: dict, thresholds: dict | None = None) -> None:
    th = thresholds or _THRESHOLDS
    print(f"\n{SEP}")
    print("SCORECARD")
    print(SEP)

    def row(label: str, value_str: str, threshold_str: str, passed: bool | None) -> None:
        result = "PASS" if passed is True else ("FAIL" if passed is False else "n/a ")
        print(f"  {result}  {label:<38}  {value_str:>8}  (threshold {threshold_str})")

    for model_label, m in scorecard["models"].items():
        print(f"\n  Model: {model_label}")
        pr, sr, me = m["pearson_r"], m["spearman_r"], m["mae"]
        row("Pearson r vs reference",      f"{pr:+.3f}" if pr is not None else "n/a",
            th["pearson_r"][1],   m["pass"]["pearson_r"])
        row("Spearman r vs reference",     f"{sr:+.3f}" if sr is not None else "n/a",
            th["spearman_r"][1],  m["pass"]["spearman_r"])
        row("MAE vs reference",            f"{me:.2f}" if me is not None else "n/a",
            th["mae"][1],         m["pass"]["mae"])
        ca = m["confound_acc"]
        if ca is not None:
            ca_str = f"{m['confound_correct']}/{m['confound_total']}"
            print(f"  INFO  {'Confound score=expected (info only)':<38}  {ca_str:>8}")

    cons = scorecard["consistency"]
    if cons:
        print(f"\n  Consistency (temp=0.9, {cons['n_runs']} runs, model={cons['model']}):")
        ms = cons["mean_std"]
        row("Mean std across items", f"{ms:.3f}" if ms is not None else "n/a",
            th["consistency_std"][1], cons["pass"])
    else:
        print(f"\n  Consistency: not run (use --n-runs)")

    if scorecard["inter_model"]:
        print(f"\n  Inter-model agreement:")
        for pair, v in scorecard["inter_model"].items():
            me = v["mae"]
            row(pair[:42], f"{me:.2f}" if me is not None else "n/a",
                th["inter_model_mae"][1], v["pass"])

    print(SEP)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_calibration(
    items: list[dict],
    model_scores: dict[str, list[int | None]],
    reference: list[int | None],
    ref_label: str,
    consistency_runs: list[list[int | None]] | None,
    output_dir: Path,
    rater_scores: dict[str, list[int | None]] | None = None,
    consistency_model: str | None = None,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    ids = [item["id"] for item in items]
    x = np.arange(len(items))
    model_colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    # --- Plot 1: scatter — judge vs reference, one point per item per model ---
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (label, scores) in enumerate(model_scores.items()):
        rv, sv = _valid_pairs(reference, scores)
        if rv:
            ax.scatter(rv, sv, label=label, color=model_colors[i % len(model_colors)],
                       alpha=0.7, s=60, zorder=3)
    if rater_scores:
        for j, (rater, r_scores) in enumerate(rater_scores.items()):
            rv, sv = _valid_pairs(reference, r_scores)
            if rv:
                ax.scatter(rv, sv, label=f"rater: {rater}",
                           color=model_colors[(len(model_scores) + j) % len(model_colors)],
                           marker="D", s=50, alpha=0.8, zorder=4)
    ref_vals = [v for v in reference if v is not None]
    if ref_vals:
        lims = [min(ref_vals) - 0.5, max(ref_vals) + 0.5]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.4, label="perfect agreement")
    ax.set_xlabel(f"Reference score ({ref_label})")
    ax.set_ylabel("Judge / rater score")
    ax.set_title("Scores vs. reference (heldout set)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)
    out = output_dir / "human_correlation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")

    # --- Plot 2: per-item grouped bar chart ---
    # Series: baseline (expected_score) | rater scores | model temp=0 scores
    baseline = [item["expected_score"] for item in items]
    series: list[tuple[str, list[int | None], str, str]] = []  # (label, scores, color, hatch)
    series.append(("baseline (expected)", baseline, "#333333", ""))
    if rater_scores:
        rater_palette = ["#E67E22", "#8E44AD", "#16A085", "#C0392B"]
        for j, (rater, r_scores) in enumerate(rater_scores.items()):
            series.append((f"rater: {rater}", r_scores,
                           rater_palette[j % len(rater_palette)], "/"))
    for i, (label, scores) in enumerate(model_scores.items()):
        series.append((label, scores,
                       model_colors[i % len(model_colors)], ""))

    n_series = len(series)
    bar_width = min(0.8 / n_series, 0.18)
    fig, ax = plt.subplots(figsize=(max(12, len(items) * 0.9), 5))
    offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * bar_width
    for (label, scores, color, hatch), offset in zip(series, offsets):
        vals = [s if s is not None else 0 for s in scores]
        ax.bar(x + offset, vals, width=bar_width, label=label,
               color=color, alpha=0.75, hatch=hatch, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Score")
    ax.set_title("Per-item scores: baseline / raters / judge models (temp=0)")
    ax.legend(fontsize=7, ncol=min(n_series, 4))
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(-5, 5)
    out = output_dir / "per_item_scores.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")

    # --- Plot 3: consistency strip plot — dots per run, mean line ---
    if consistency_runs is not None:
        n_runs = len(consistency_runs[0]) if consistency_runs else 0
        fig, ax = plt.subplots(figsize=(max(12, len(items) * 0.9), 5))
        c_color = model_colors[0]  # first model color for consistency model
        jitter_scale = 0.06

        for xi, (item, runs) in enumerate(zip(items, consistency_runs)):
            valid = [s for s in runs if s is not None]
            # Scatter individual runs with small jitter
            for ri, s in enumerate(runs):
                if s is not None:
                    jitter = (ri - (n_runs - 1) / 2) * jitter_scale
                    ax.scatter(xi + jitter, s, color=c_color, alpha=0.55, s=35, zorder=3)
            # Mean line
            if valid:
                mean_val = statistics.mean(valid)
                ax.plot([xi - 0.3, xi + 0.3], [mean_val, mean_val],
                        color=c_color, linewidth=2.0, alpha=0.9, zorder=4)

        # Overlay baseline and rater scores as markers
        ax.scatter(x, baseline, marker="_", s=200, linewidths=2,
                   color="#333333", label="baseline", zorder=5)
        if rater_scores:
            rater_palette = ["#E67E22", "#8E44AD", "#16A085", "#C0392B"]
            for j, (rater, r_scores) in enumerate(rater_scores.items()):
                vals = [s if s is not None else float("nan") for s in r_scores]
                ax.scatter(x, vals, marker="D", s=40,
                           color=rater_palette[j % len(rater_palette)],
                           label=f"rater: {rater}", zorder=5, alpha=0.9)

        model_label = consistency_model or "model"
        ax.scatter([], [], color=c_color, s=35, alpha=0.6,
                   label=f"{model_label} runs (temp=0.9)")
        ax.plot([], [], color=c_color, linewidth=2, label=f"{model_label} mean")

        ax.set_xticks(x)
        ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Score")
        ax.set_title(f"Self-consistency: {n_runs} runs at temp=0.9 [{model_label}]")
        ax.legend(fontsize=7, ncol=4)
        ax.grid(True, alpha=0.2, axis="y")
        ax.set_ylim(-5, 5)
        out = output_dir / "consistency_runs.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate an LLM judge against heldout items.")
    parser.add_argument("--judge", required=True, help="Judge name, e.g. 'neuroticism'.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["openai/gpt-4o-mini"],
        help="Model identifiers to evaluate. Space-separated.",
    )
    parser.add_argument(
        "--provider",
        default="openrouter",
        choices=["openai", "openrouter", "anthropic"],
    )
    parser.add_argument(
        "--human-scores",
        type=str,
        nargs="+",
        default=None,
        metavar="CSV",
        help=(
            "One or more filled rating CSVs (from generate_rating_form.py). "
            "Rater columns from all files are merged. "
            "Uses rater mean as reference instead of expected_score. "
            "Inter-rater agreement is reported when multiple raters are present."
        ),
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=0,
        help="Consistency runs at temp=0.9. 0 = skip (default).",
    )
    parser.add_argument(
        "--consistency-model",
        type=str,
        default=None,
        help="Model for consistency check. Defaults to first model in --models.",
    )
    parser.add_argument("--max-concurrent", type=int, default=20)
    parser.add_argument("--save", action="store_true",
                        help="Save per-item results (scores, evidence, reasoning) to <judge>/results/.")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    load_dotenv()
    import logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING))

    judge_cls = load_judge_class(args.judge)
    items = load_heldout(args.judge)
    thresholds = _get_thresholds(judge_cls)

    print(f"\nJudge : {args.judge} ({judge_cls.__name__})")
    print(f"Items : {len(items)}")
    print(f"Models: {args.models}")

    # --- Determine reference scores ---
    author_reference = [item["expected_score"] for item in items]
    rater_scores: dict[str, list[int | None]] = {}

    # Auto-discover ratings from judge's ratings/ dir, then merge any explicit paths.
    csv_paths: list[Path] = find_ratings(args.judge)
    if args.human_scores:
        csv_paths += [Path(p) for p in args.human_scores]

    if csv_paths:
        print(f"\nLoading human scores from: {[p.name for p in csv_paths]}")
        rater_scores = load_human_scores(csv_paths, items)
        print(f"  Raters found: {list(rater_scores.keys())}")

        if len(rater_scores) > 1:
            print_inter_rater(items, rater_scores)

        # Use mean of rater scores as the primary reference (fall back to author if blank)
        mean_reference: list[int | None] = []
        for i in range(len(items)):
            vals = [rater_scores[r][i] for r in rater_scores if rater_scores[r][i] is not None]
            mean_reference.append(round(statistics.mean(vals)) if vals else author_reference[i])
        reference = mean_reference
        ref_label = f"human mean ({', '.join(rater_scores.keys())})"
    else:
        reference = author_reference
        ref_label = "author (expected_score)"

    import datetime
    import io

    # Capture all output for the saved MD report.
    _log_buf = io.StringIO()

    def _tee(text: str = "") -> None:
        """Print to stdout and buffer simultaneously."""
        print(text)
        _log_buf.write(text + "\n")

    ts_run = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- 1. Run models at temp=0.9 ---
    model_scores: dict[str, list[int | None]] = {}
    all_raw: dict[str, list[dict]] = {}
    judge_name_str: str = ""
    for model in args.models:
        config = JudgeLLMConfig(
            provider=args.provider,
            model=model,
            temperature=0.9,
            max_concurrent=args.max_concurrent,
        )
        _tee(f"\nRunning {model} (temp=0.9) ...")
        scores, raw_results = asyncio.run(run_once(judge_cls, items, config))
        model_scores[model] = scores
        all_raw[model] = raw_results
        if not judge_name_str:
            judge_name_str = judge_cls(judge_config=config).name

        # Capture correlation table into buffer by temporarily redirecting stdout
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_human_correlation(items, scores, reference, ref_label, model)
            if rater_scores:
                print(f"  Per-rater correlation [{model}]:")
                for rater, r_scores in rater_scores.items():
                    pr, sr, me, n = _corr_stats(r_scores, scores)
                    print(f"    {rater:<20}  pearson={pr:+.3f}  spearman={sr:+.3f}  mae={me:.2f}  n={n}")
        captured = buf.getvalue()
        print(captured, end="")
        _log_buf.write(captured)

    # --- 2. Cross-model summary ---
    if len(args.models) > 1:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_cross_model_summary(items, model_scores, reference, ref_label)
        captured = buf.getvalue()
        print(captured, end="")
        _log_buf.write(captured)

    # --- 3. Self-consistency ---
    consistency_runs = None
    consistency_model_used = None
    if args.n_runs > 0:
        consistency_model_used = args.consistency_model or args.models[0]
        config = JudgeLLMConfig(
            provider=args.provider,
            model=consistency_model_used,
            temperature=0.9,
            max_concurrent=args.max_concurrent,
        )
        _tee(f"\nConsistency check: {args.n_runs} runs at temp=0.9 [{consistency_model_used}] ...")
        consistency_runs = asyncio.run(run_consistency(judge_cls, items, config, args.n_runs))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_consistency(items, consistency_runs)
        captured = buf.getvalue()
        print(captured, end="")
        _log_buf.write(captured)

    # --- 4. Scorecard ---
    scorecard = compute_scorecard(
        items, model_scores, reference, ref_label, consistency_runs, consistency_model_used,
        thresholds=thresholds,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_scorecard(scorecard, thresholds=thresholds)
    captured = buf.getvalue()
    print(captured, end="")
    _log_buf.write(captured)

    # --- 5. Save ---
    if args.save:
        results_dir = judge_dir(args.judge) / "results" / ts_run
        results_dir.mkdir(parents=True, exist_ok=True)
        reasoning_key = f"{judge_name_str}.reasoning"

        # Per-item JSONL for each model
        for model in args.models:
            safe_model = model.replace("/", "_")
            out_path = results_dir / f"{safe_model}.jsonl"
            with open(out_path, "w") as f:
                for item, score, raw in zip(items, model_scores[model], all_raw[model]):
                    f.write(json.dumps({
                        "id": item["id"],
                        "category": item["category"],
                        "expected_score": item["expected_score"],
                        "score": score,
                        "reasoning": raw.get(reasoning_key, ""),
                        "model": model,
                        "provider": args.provider,
                        "temperature": 0.9,
                    }) + "\n")
            print(f"  Saved → {out_path}")

        # Scorecard JSON
        sc_path = results_dir / "scorecard.json"
        sc_path.write_text(json.dumps({
            "judge": args.judge,
            "timestamp": ts_run,
            "n_items": len(items),
            "models": args.models,
            "provider": args.provider,
            "temperature": 0.9,
            **scorecard,
        }, indent=2))
        print(f"  Saved → {sc_path}")

        # Full stdout log as Markdown
        md_path = results_dir / "run.md"
        md_path.write_text(
            f"# Calibration run — {args.judge} — {ts_run}\n\n"
            f"**Models:** {', '.join(args.models)}  \n"
            f"**Provider:** {args.provider}  \n"
            f"**Temperature:** 0.9  \n"
            f"**Items:** {len(items)}  \n\n"
            "```\n" + _log_buf.getvalue() + "\n```\n"
        )
        print(f"  Saved → {md_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
