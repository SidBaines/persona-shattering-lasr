"""Per-trait delta bar chart for the c_minus × e_minus (vanton4) 1:1 combo at (+1, +1).

Hydrates the (+1, +1) combo cell for each of the 5 OCEAN traits, plus the
base-model baseline, plus the single-adapter c_minus/e_minus cells where
available. Plots the Qwen3-235B judge's mean trait score delta vs baseline
for each OCEAN trait — similar layout to
``scripts_dev/evals/ocean_delta_plot.py``.

Data sources:
 - Combo at (+1, +1) lives at several different rollout fingerprints depending
   on which sweep touched that trait first:
     C: 97743334f6 (5x5 grid on conscientiousness.jsonl)
     E: 47a37c39b7 (5x5 grid on extraversion.jsonl)
     O: 1817b5cf78 (1x1 soup on openness.jsonl)
     A: b2e6755ff3 (1x1 soup on agreeableness.jsonl)
     N: 8b01e9fa2c (1x1 soup on neuroticism.jsonl)
 - Baseline + single-adapter cells (where available):
     For C, E: taken from the same fingerprint as the combo (5x5 grid has
       baseline and single-adapter cells).
     For O, A, N: baseline comes from the 240x1 forged fingerprint (67eed27d02,
       0705e3276a, b2a49f1b4d); single-adapter cells come from 240x1 Option B
       cells where those have completed — NaN otherwise.

Paper figures:
    - paper/figures/main/fig_1_c_e_combo_delta.pdf

Run with:
    uv run python -m src_dev.visualisations.paper_main_c_e_combo_delta
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
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

PAPER_FIGURES = [
    "main/fig_1_c_e_combo_delta.pdf",
]

# ---------------------------------------------------------------------------
# Configuration — hardcoded
# ---------------------------------------------------------------------------

HF_REPO_ID = "persona-shattering-lasr/monorepo"
MODEL_SLUG = "llama-3.1-8b-it"
EVAL_NAME = "llm_judge_lora_scale_sweep"
RATER_ID = "qwen3_235b"

C_SLUG = "ocean-conscientiousness-suppressor-vanton4"
E_SLUG = "ocean-extraversion-suppressor-vanton4"
C_TRAIT_LOWER = "conscientiousness"
E_TRAIT_LOWER = "extraversion"

# Alphabetical combo slug (matches CanonicalCell.combo_slug)
COMBO_SLUG = "__".join(sorted([C_SLUG, E_SLUG]))

OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

# Per-trait: where the combo (+1, +1) cell lives.
# Dual-fingerprint: (combo_fp, baseline_fp_fallback). baseline_fp_fallback is
# the 240x1 fingerprint where Run-1 baselines were forged — used when the
# combo_fp doesn't itself carry a baseline cell (the 1x1 soups do not).
FP_BY_TRAIT: dict[str, tuple[str, str]] = {
    # trait_lower: (combo_fp, baseline_fallback_fp)
    "openness":          ("1817b5cf78", "67eed27d02"),
    "conscientiousness": ("97743334f6", "e6426e3031"),
    "extraversion":      ("47a37c39b7", "a961f641eb"),
    "agreeableness":     ("b2e6755ff3", "0705e3276a"),
    "neuroticism":       ("8b01e9fa2c", "b2a49f1b4d"),
}

OUT_PATH = PAPER_FIGURES_DIR / PAPER_FIGURES[0]
CACHE_DIR = project_root / "scratch" / "paper_plots_cache" / "c_e_combo_delta"


# ---------------------------------------------------------------------------
# Path helpers (match CanonicalCell.hf_dir)
# ---------------------------------------------------------------------------

def _combo_cell_hf_dir(fingerprint: str, c_scale: float, e_scale: float) -> str:
    """HF dir for the (c_minus=c_scale, e_minus=e_scale) combo cell."""
    # Scale formatting matches format_scale() in cell_identity.py
    def _fmt(x: float) -> str:
        sign = "+" if x >= 0 else "-"
        return f"{sign}{abs(x):.2f}"
    spec = f"cell_{C_SLUG}{_fmt(c_scale)}_{E_SLUG}{_fmt(e_scale)}"
    return f"combos/{MODEL_SLUG}/{COMBO_SLUG}/{EVAL_NAME}/{fingerprint}/{spec}"


def _single_c_cell_hf_dir(fingerprint: str, c_scale: float) -> str:
    """c_minus alone at given scale."""
    def _fmt(x: float) -> str:
        sign = "+" if x >= 0 else "-"
        return f"{sign}{abs(x):.2f}"
    return (
        f"fine_tuning/{MODEL_SLUG}/ocean/{C_TRAIT_LOWER}/suppressor/vanton4"
        f"/evals/{EVAL_NAME}/{fingerprint}/scale_{_fmt(c_scale)}"
    )


def _single_e_cell_hf_dir(fingerprint: str, e_scale: float) -> str:
    def _fmt(x: float) -> str:
        sign = "+" if x >= 0 else "-"
        return f"{sign}{abs(x):.2f}"
    return (
        f"fine_tuning/{MODEL_SLUG}/ocean/{E_TRAIT_LOWER}/suppressor/vanton4"
        f"/evals/{EVAL_NAME}/{fingerprint}/scale_{_fmt(e_scale)}"
    )


def _baseline_hf_dir(fingerprint: str) -> str:
    return f"combos/{MODEL_SLUG}/_baseline/{EVAL_NAME}/{fingerprint}"


def _judge_hf_path(cell_hf_dir: str, metric_name: str) -> str:
    return f"{cell_hf_dir}/judge_runs/{RATER_ID}/{metric_name}.jsonl"


# ---------------------------------------------------------------------------
# Hydration + cache
# ---------------------------------------------------------------------------

def _cache_path(hf_path: str) -> Path:
    return CACHE_DIR / hf_path


def _hydrate_judge_file(hf_path: str) -> Path | None:
    local = _cache_path(hf_path)
    if local.exists() and local.stat().st_size > 0:
        return local
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
        print(f"  ✗ hydrate failed: {type(exc).__name__}: {str(exc)[:140]}")
        return None
    if local.exists() and local.stat().st_size > 0:
        return local
    return None


def _mean_score_median_across_repeats(jsonl_path: Path) -> float | None:
    """Median over repeats per response_id, then mean across responses."""
    grouped: dict[str, list[int]] = defaultdict(list)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") not in {"success", "parse_error"}:
                continue
            s = row.get("score")
            if not isinstance(s, (int, float)):
                continue
            rid = str(row.get("response_id", ""))
            grouped[rid].append(int(s))
    if not grouped:
        return None
    medians = [statistics.median(v) for v in grouped.values() if v]
    return statistics.fmean(medians) if medians else None


# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------

def _trait_metric(trait_lower: str) -> str:
    return f"{trait_lower}_v2"


def _fetch(cell_hf_dir: str, trait_lower: str) -> float | None:
    """Hydrate + parse one (cell, trait) mean score."""
    hf_path = _judge_hf_path(cell_hf_dir, _trait_metric(trait_lower))
    local = _hydrate_judge_file(hf_path)
    if local is None:
        return None
    return _mean_score_median_across_repeats(local)


def gather() -> dict[str, dict[str, float | None]]:
    """Returns per-trait dict: {trait_title: {baseline, c_minus, e_minus, combo}}."""
    out: dict[str, dict[str, float | None]] = {}
    for trait_lower, (combo_fp, baseline_fp) in FP_BY_TRAIT.items():
        trait_title = trait_lower.capitalize()
        print(f"\n[trait] {trait_title}")

        # Combo (+1, +1) at combo_fp
        combo_dir = _combo_cell_hf_dir(combo_fp, 1.0, 1.0)
        combo_mean = _fetch(combo_dir, trait_lower)
        print(f"  combo(c+1, e+1)        @ {combo_fp}: {combo_mean}")

        # Baseline — try combo_fp first, fall back to baseline_fp
        baseline_mean = _fetch(_baseline_hf_dir(combo_fp), trait_lower)
        if baseline_mean is None:
            baseline_mean = _fetch(_baseline_hf_dir(baseline_fp), trait_lower)
            print(f"  baseline (fallback 240x1) @ {baseline_fp}: {baseline_mean}")
        else:
            print(f"  baseline                @ {combo_fp}: {baseline_mean}")

        # Single-adapter c_minus(+1) — combo_fp first, fall back to baseline_fp
        c_mean = _fetch(_single_c_cell_hf_dir(combo_fp, 1.0), trait_lower)
        if c_mean is None:
            c_mean = _fetch(_single_c_cell_hf_dir(baseline_fp, 1.0), trait_lower)
            print(f"  c_minus(+1) (fallback)  @ {baseline_fp}: {c_mean}")
        else:
            print(f"  c_minus(+1)             @ {combo_fp}: {c_mean}")

        # Single-adapter e_minus(+1) — same pattern
        e_mean = _fetch(_single_e_cell_hf_dir(combo_fp, 1.0), trait_lower)
        if e_mean is None:
            e_mean = _fetch(_single_e_cell_hf_dir(baseline_fp, 1.0), trait_lower)
            print(f"  e_minus(+1) (fallback)  @ {baseline_fp}: {e_mean}")
        else:
            print(f"  e_minus(+1)             @ {combo_fp}: {e_mean}")

        out[trait_title] = {
            "baseline": baseline_mean,
            "c_minus":  c_mean,
            "e_minus":  e_mean,
            "combo":    combo_mean,
        }
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

TARGET_ORDER = ["c_minus", "e_minus", "combo"]
TARGET_DISPLAY = {
    "c_minus": "c_minus (+1)",
    "e_minus": "e_minus (+1)",
    "combo":   "c_minus × e_minus (+1, +1)",
}
TARGET_COLORS = {
    "c_minus": BIG_FIVE_COLORS["Conscientiousness"],
    "e_minus": BIG_FIVE_COLORS["Extraversion"],
    "combo":   "#8A3FFC",
}
TARGET_HATCHES = {
    "c_minus": "//",
    "e_minus": "\\\\",
    "combo":   "xx",
}


def render(scores: dict[str, dict[str, float | None]], out_path: Path) -> None:
    x = np.arange(len(OCEAN_TRAITS))
    width = 0.8 / len(TARGET_ORDER)

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    for i, key in enumerate(TARGET_ORDER):
        deltas = []
        for trait_title in OCEAN_TRAITS:
            row = scores.get(trait_title, {})
            base = row.get("baseline")
            val = row.get(key)
            if base is None or val is None:
                deltas.append(np.nan)
            else:
                deltas.append(val - base)
        ax.bar(
            x + (i - (len(TARGET_ORDER) - 1) / 2) * width,
            deltas,
            width,
            label=TARGET_DISPLAY[key],
            color=TARGET_COLORS[key],
            hatch=TARGET_HATCHES[key],
            edgecolor="black",
            linewidth=0.6,
        )

    ax.axhline(0.0, color="k", linewidth=0.8, label="base model")
    ax.set_xticks(x)
    ax.set_xticklabels(OCEAN_TRAITS, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Δ judge score vs base model")
    ax.set_title(
        "c_minus × e_minus (vanton4) @ (+1, +1): per-trait effect vs base model — Qwen3-235B judge",
        fontsize=11,
    )
    ax.set_ylim(-4.0, 4.0)
    ax.axhspan(-4.0, 0.0, alpha=0.04, color="red")
    ax.axhspan(0.0, 4.0, alpha=0.04, color="blue")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ saved {out_path}")


def main() -> None:
    print(f"[combo-delta] cache dir: {CACHE_DIR}")
    print(f"[combo-delta] out path:  {OUT_PATH}")
    scores = gather()
    # Persist raw scores alongside the figure for iteration / debugging.
    scores_path = CACHE_DIR / "scores.json"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("w") as f:
        json.dump(scores, f, indent=2, sort_keys=True)
    print(f"✓ saved scores {scores_path}")
    render(scores, OUT_PATH)


if __name__ == "__main__":
    main()
