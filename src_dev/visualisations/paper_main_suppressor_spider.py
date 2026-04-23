"""Spider plot of the five vanton4 OCEAN suppressor LoRAs at scale=1.0.

One polygon per suppressor (o_minus, c_minus, e_minus, a_minus, n_minus) plus
a baseline polygon. For each (adapter, judged trait) pair, we hydrate the
mean judge score at scale=+1.00 from the HF monorepo and overlay them on a
5-axis OCEAN radar.

Data comes from two families of runs produced by vanton4_qwen3:
 - Own-trait (Run 1): judge on the adapter's home trait at the home-trait
   rollout fingerprint (e.g. c_minus judged on conscientiousness at fp
   ea50f894e4 — the fingerprint for conscientiousness.jsonl rollouts).
 - Cross-trait (Option B): judge on the other four traits at the
   target-trait rollout fingerprints (same fingerprints as the home
   adapters for that trait).

Paper figures:
    - paper/figures/main/fig_1_suppressor_spider.pdf  (Fig. 1 subfig)

Everything is hardcoded. Cross-trait cells come from in-flight Option B
sweeps; if a cell is missing, the corresponding trait reads as NaN and is
skipped on that polygon with a warning. Re-run the script once Option B
finishes to get a complete figure.

Run with:
    uv run python -m src_dev.visualisations.paper_main_suppressor_spider
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.utils.hf_hub import download_path_to_dir
from src_dev.visualisations import PAPER_FIGURES_DIR

OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

PAPER_FIGURES = [
    "main/fig_1_suppressor_spider.pdf",
]

# ---------------------------------------------------------------------------
# Configuration — hardcoded
# ---------------------------------------------------------------------------

HF_REPO_ID = "persona-shattering-lasr/monorepo"
MODEL_SLUG = "llama-3.1-8b-it"
EVAL_NAME = "llm_judge_lora_scale_sweep"
RATER_ID = "qwen3_235b"
SCALE = 1.0
SCALE_LABEL = "scale_+1.00"

# Fingerprint of each trait's rollout dataset (from vanton4_qwen3 configs).
# Same for Run 1 and Option B runs that use the same trait prompt set.
FP_BY_TRAIT = {
    "openness":          "d28e156e70",
    "conscientiousness": "ea50f894e4",
    "extraversion":      "dbb7b7ab8e",
    "agreeableness":     "e1ee1d133f",
    "neuroticism":       "3e5360b27c",
}

# Each suppressor: (display name, OCEAN home trait). Colors come from the
# canonical repo palette in src_dev/evals/personality/analyze_results.py.
SUPPRESSORS: list[tuple[str, str, str]] = [
    ("o_minus", "openness",          BIG_FIVE_COLORS["Openness"]),
    ("c_minus", "conscientiousness", BIG_FIVE_COLORS["Conscientiousness"]),
    ("e_minus", "extraversion",      BIG_FIVE_COLORS["Extraversion"]),
    ("a_minus", "agreeableness",     BIG_FIVE_COLORS["Agreeableness"]),
    ("n_minus", "neuroticism",       BIG_FIVE_COLORS["Neuroticism"]),
]

BASELINE_COLOR = "#4D4D4D"

OUT_PATH = PAPER_FIGURES_DIR / PAPER_FIGURES[0]
CACHE_DIR = project_root / "scratch" / "paper_plots_cache" / "suppressor_spider"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _adapter_hf_dir(home_trait: str, fingerprint: str) -> str:
    """HF dir for a single-adapter suppressor cell (canonical single-adapter tier)."""
    return (
        f"fine_tuning/{MODEL_SLUG}/ocean/{home_trait}/suppressor/vanton4"
        f"/evals/{EVAL_NAME}/{fingerprint}/{SCALE_LABEL}"
    )

def _baseline_hf_dir(fingerprint: str) -> str:
    """HF dir for the baseline (no-adapter) cell at a given rollout fingerprint."""
    return f"combos/{MODEL_SLUG}/_baseline/{EVAL_NAME}/{fingerprint}"

def _judge_hf_path(cell_hf_dir: str, metric_name: str) -> str:
    return f"{cell_hf_dir}/judge_runs/{RATER_ID}/{metric_name}.jsonl"

# ---------------------------------------------------------------------------
# Data hydration with on-disk cache
# ---------------------------------------------------------------------------

def _cache_path(hf_path: str) -> Path:
    """Local cache location mirroring the HF path."""
    return CACHE_DIR / hf_path

def _hydrate_judge_file(hf_path: str) -> Path | None:
    """Download a single judge JSONL into the cache; return local path (or None if missing)."""
    local_path = _cache_path(hf_path)
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    parent_hf = hf_path.rsplit("/", 1)[0]
    filename = hf_path.rsplit("/", 1)[1]
    local_parent = _cache_path(parent_hf)
    try:
        download_path_to_dir(
            repo_id=HF_REPO_ID,
            path_in_repo=parent_hf,
            target_dir=local_parent,
            allow_patterns=[filename],
        )
    except Exception as exc:
        print(f"  ✗ hydrate failed for {hf_path}: {type(exc).__name__}: {str(exc)[:160]}")
        return None
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    return None

def _mean_score(jsonl_path: Path) -> float | None:
    """Compute mean trait score from a judge-calls raw JSONL. Skips invalid rows."""
    scores: list[float] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            val = row.get("score")
            if val is None or not isinstance(val, (int, float)):
                continue
            scores.append(float(val))
    if not scores:
        return None
    return statistics.fmean(scores)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _trait_metric(trait_lower: str) -> str:
    return f"{trait_lower}_v2"

def _trait_title(trait_lower: str) -> str:
    return trait_lower.capitalize()

def build_scores() -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Hydrate and parse judge scores.

    Returns ``(per_suppressor_scores, baseline_scores)``.

    ``per_suppressor_scores`` maps suppressor-key → {TraitTitle: mean_score}.
    ``baseline_scores`` maps TraitTitle → mean_score.
    """
    per_suppressor: dict[str, dict[str, float]] = {key: {} for key, _, _ in SUPPRESSORS}
    baseline: dict[str, float] = {}

    # Baseline: for each trait dataset, read the baseline cell's judge on that trait.
    for trait_lower, fp in FP_BY_TRAIT.items():
        hf_path = _judge_hf_path(_baseline_hf_dir(fp), _trait_metric(trait_lower))
        local = _hydrate_judge_file(hf_path)
        if local is None:
            print(f"  ⚠ baseline / {trait_lower}: missing on HF")
            continue
        mean = _mean_score(local)
        if mean is None:
            print(f"  ⚠ baseline / {trait_lower}: no valid scores in {local}")
            continue
        baseline[_trait_title(trait_lower)] = mean
        print(f"  ✓ baseline / {trait_lower:18s}: mean = {mean:+.3f}")

    # Per-suppressor: for each (suppressor adapter, judged trait), hydrate.
    for key, home_trait, _color in SUPPRESSORS:
        for judged_trait, fp in FP_BY_TRAIT.items():
            cell_dir = _adapter_hf_dir(home_trait, fp)
            hf_path = _judge_hf_path(cell_dir, _trait_metric(judged_trait))
            local = _hydrate_judge_file(hf_path)
            if local is None:
                kind = "own" if judged_trait == home_trait else "cross"
                print(f"  ⚠ {key} / {judged_trait:18s} ({kind}): missing on HF — likely still running")
                continue
            mean = _mean_score(local)
            if mean is None:
                print(f"  ⚠ {key} / {judged_trait}: no valid scores in {local}")
                continue
            per_suppressor[key][_trait_title(judged_trait)] = mean
            print(f"  ✓ {key} / {judged_trait:18s}: mean = {mean:+.3f}")

    return per_suppressor, baseline


def _render_spider(
    *,
    per_suppressor: dict[str, dict[str, float]],
    baseline: dict[str, float],
    out_path: Path,
) -> None:
    """Render the spider plot. Missing traits appear as gaps in the polygon."""
    traits = OCEAN_TRAITS
    angles = np.linspace(0.0, 2.0 * np.pi, len(traits), endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    def _polygon(scores: dict[str, float], *, label: str, color: str, linewidth: float) -> None:
        means = [scores.get(t, float("nan")) for t in traits]
        means_closed = means + means[:1]
        # Lines: matplotlib handles NaN as gap.
        ax.plot(angles_closed, means_closed, "-", color=color, linewidth=linewidth, label=label)
        # Markers only on present data.
        for angle, val in zip(angles, means):
            if not np.isnan(val):
                ax.plot([angle], [val], "o", color=color, markersize=6)
        # Fill (requires complete polygon; skip if any NaN).
        if not any(np.isnan(v) for v in means):
            ax.fill(angles_closed, means_closed, color=color, alpha=0.10)

    if baseline:
        _polygon(baseline, label="base model", color=BASELINE_COLOR, linewidth=2.5)

    for key, _home_trait, color in SUPPRESSORS:
        row = per_suppressor[key]
        if not row:
            continue
        _polygon(row, label=f"{key} @ {SCALE:+.0f}", color=color, linewidth=2.0)

    ax.set_xticks(angles)
    ax.set_xticklabels(traits, fontsize=11)
    ax.set_ylim(-4.0, 4.0)
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.set_yticklabels(["-4", "-2", "0", "+2", "+4"], fontsize=9)
    ax.grid(True, alpha=0.4)
    ax.set_title(
        f"OCEAN suppressors (vanton4) at {SCALE_LABEL} — Qwen3-235B judge",
        fontsize=12, pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.08), fontsize=10, framealpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ saved {out_path}")


def main() -> None:
    print(f"[spider] cache dir: {CACHE_DIR}")
    print(f"[spider] out path:  {OUT_PATH}")
    print("[spider] hydrating judge scores from HF...")
    per_suppressor, baseline = build_scores()
    _render_spider(per_suppressor=per_suppressor, baseline=baseline, out_path=OUT_PATH)


if __name__ == "__main__":
    main()
