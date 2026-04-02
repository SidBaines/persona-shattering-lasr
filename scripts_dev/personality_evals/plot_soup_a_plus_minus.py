"""Visualise A+ / A- soup TRAIT sweep results.

Plots all five OCEAN trait scores as a function of the A- adapter coefficient c,
with A+ fixed at 1.0x. Three reference points (base, A+ only, A- only) are shown
as horizontal dashed lines.

Produces two figures:
  1. Line plot: trait score vs A- coefficient (the main sweep plot)
  2. Bar chart: delta from base for Agreeableness at each soup coefficient,
     with A+ only and A- only deltas shown for comparison

Data sources:
  - Soup sweep (8 points): local scratch from soup_a_plus_minus eval run
  - Reference points (3): downloaded from HuggingFace monorepo

Usage
-----
    uv run python scripts_dev/personality_evals/plot_soup_a_plus_minus.py
"""

from __future__ import annotations

from pathlib import Path

import json
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Style constants (match analyze_results.py)
# ---------------------------------------------------------------------------
BIG_FIVE = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
BIG_FIVE_SHORT = ["O", "C", "E", "A", "N"]
BIG_FIVE_COLORS = {
    "Openness":          "#2196F3",
    "Conscientiousness": "#FF9800",
    "Extraversion":      "#4CAF50",
    "Agreeableness":     "#9C27B0",
    "Neuroticism":       "#F44336",
}

OUTPUT_DIR = Path("scratch/evals/ocean/trait/soup_a_plus_minus/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_local_soup_scores(run_root: Path) -> dict[float, dict[str, float]]:
    """Load OCEAN scores from local soup eval inspect logs.

    Returns:
        {c_value: {trait: score}} for each soup coefficient.
    """
    results: dict[float, dict[str, float]] = {}
    for model_dir in sorted(run_root.glob("soup_aplus1p00_aminus*")):
        # Extract c value from directory name like soup_aplus1p00_aminus0p75
        tag = model_dir.name.split("aminus")[1]  # e.g. "0p75"
        c = float(tag.replace("p", "."))
        log_files = list(model_dir.glob("trait/native/inspect_logs/*.json"))
        for log_path in log_files:
            d = json.loads(log_path.read_text())
            if "results" not in d:
                continue
            scores = d["results"]["scores"][0]["metrics"]
            results[c] = {t: scores[t]["value"] for t in BIG_FIVE}
    return results


def _load_hf_reference_scores() -> dict[str, dict[str, float]]:
    """Download and extract OCEAN scores for base, A+, A- from HuggingFace.

    Returns:
        {"base": {...}, "A+": {...}, "A-": {...}}
    """
    repo = "persona-shattering-lasr/monorepo"
    log_paths = {
        "base": (
            "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/evals/mcq/trait/"
            "a_plus/base/trait/native/inspect_logs/"
            "2026-04-01T22-56-05+00-00_task_g85CcgYwCmAdYPp2Tv6eiZ.json"
        ),
        "A+": (
            "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/evals/mcq/trait/"
            "a_plus/lora_+1p00x/trait/native/inspect_logs/"
            "2026-04-02T00-31-48+00-00_task_J9xsi2N5fri4htNBFAhxGM.json"
        ),
        "A-": (
            "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/evals/mcq/trait/"
            "a_minus/lora_+1p00x/trait/native/inspect_logs/"
            "2026-04-01T12-20-56+00-00_task_mgP87NbrrLiLWBdXwjnUfe.json"
        ),
    }
    results: dict[str, dict[str, float]] = {}
    for label, path in log_paths.items():
        local = hf_hub_download(repo, path, repo_type="dataset")
        d = json.loads(Path(local).read_text())
        scores = d["results"]["scores"][0]["metrics"]
        results[label] = {t: scores[t]["value"] for t in BIG_FIVE}
    return results


# ---------------------------------------------------------------------------
# Plot 1: Line plot — all OCEAN traits vs A- coefficient
# ---------------------------------------------------------------------------

def plot_trait_vs_coefficient(
    soup_scores: dict[float, dict[str, float]],
    refs: dict[str, dict[str, float]],
    output_dir: Path,
) -> Path:
    """Line plot of each OCEAN trait score vs A- adapter coefficient.

    Reference models shown as horizontal dashed lines.
    Agreeableness is highlighted (thicker line, larger markers).
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    c_values = sorted(soup_scores.keys())
    highlight_trait = "Agreeableness"

    for trait in BIG_FIVE:
        color = BIG_FIVE_COLORS[trait]
        means = [soup_scores[c][trait] for c in c_values]
        is_highlight = trait == highlight_trait

        lw = 2.8 if is_highlight else 1.6
        ms = 8 if is_highlight else 5
        alpha = 1.0 if is_highlight else 0.45
        zorder = 5 if is_highlight else 3

        ax.plot(c_values, means, "o-", color=color, linewidth=lw, markersize=ms,
                alpha=alpha, label=trait, zorder=zorder)

    # Reference lines
    ref_styles = {
        "base": {"linestyle": "--", "linewidth": 1.2, "alpha": 0.6},
        "A+":   {"linestyle": "-.", "linewidth": 1.2, "alpha": 0.6},
        "A-":   {"linestyle": ":",  "linewidth": 1.4, "alpha": 0.6},
    }
    for ref_label, style in ref_styles.items():
        a_score = refs[ref_label]["Agreeableness"]
        ax.axhline(a_score, color=BIG_FIVE_COLORS["Agreeableness"],
                    label=f"{ref_label} (A={a_score:.2f})", zorder=2, **style)

    ax.set_xlabel("A− adapter coefficient (c)  [A+ fixed at 1.0×]", fontsize=12)
    ax.set_ylabel("Trait score (0–1)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(c_values[0] - 0.08, c_values[-1] + 0.08)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.grid(True, alpha=0.25)

    ax.set_title(
        "A+ / A− Soup TRAIT Sweep: personality scores vs. A− coefficient\n"
        "A+ (amplifier) fixed at 1.0×, A− (suppressor) swept",
        fontsize=13, fontweight="bold",
    )

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              fontsize=9, ncol=4, framealpha=0.85)

    plt.tight_layout()
    out = output_dir / "soup_trait_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# ---------------------------------------------------------------------------
# Plot 2: Grouped bar chart — delta from base for all traits at key c values
# ---------------------------------------------------------------------------

def plot_delta_bars(
    soup_scores: dict[float, dict[str, float]],
    refs: dict[str, dict[str, float]],
    output_dir: Path,
) -> Path:
    """Grouped bar chart showing trait delta from base across conditions.

    Shows base (zero line), A+ only, A- only, and selected soup coefficients.
    Each group has 5 bars (one per OCEAN trait).
    """
    base = refs["base"]

    # Build conditions: A+ only, then soups, then A- only
    conditions: list[tuple[str, dict[str, float]]] = []
    conditions.append(("A+ only\n(1.0×)", refs["A+"]))
    for c in sorted(soup_scores.keys()):
        conditions.append((f"Soup\nc={c:.2f}", soup_scores[c]))
    conditions.append(("A− only\n(1.0×)", refs["A-"]))

    n_conditions = len(conditions)
    n_traits = len(BIG_FIVE)
    x = np.arange(n_conditions)
    bar_width = 0.15
    offsets = np.arange(n_traits) - (n_traits - 1) / 2

    fig, ax = plt.subplots(figsize=(16, 6.5))

    for i, trait in enumerate(BIG_FIVE):
        color = BIG_FIVE_COLORS[trait]
        deltas = [cond_scores[trait] - base[trait] for _, cond_scores in conditions]
        positions = x + offsets[i] * bar_width
        bars = ax.bar(positions, deltas, bar_width * 0.9, color=color,
                      alpha=0.85, label=trait, edgecolor="white", linewidth=0.5)

        # Annotate Agreeableness bars with values
        if trait == "Agreeableness":
            for bar, delta in zip(bars, deltas):
                va = "bottom" if delta >= 0 else "top"
                offset = 0.008 if delta >= 0 else -0.008
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                        f"{delta:+.2f}", ha="center", va=va, fontsize=7,
                        fontweight="bold", color=color)

    ax.axhline(0, color="black", linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in conditions], fontsize=8)
    ax.set_ylabel("Δ trait score (vs base model)", fontsize=12)
    ax.set_ylim(-0.50, 0.30)
    ax.grid(True, axis="y", alpha=0.25)

    ax.set_title(
        "OCEAN trait deltas from base: A+ / A− soup sweep\n"
        "A+ (amplifier) fixed at 1.0×, A− (suppressor) coefficient swept",
        fontsize=13, fontweight="bold",
    )

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              fontsize=9, ncol=5, framealpha=0.85)

    plt.tight_layout()
    out = output_dir / "soup_delta_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    run_root = Path("scratch/evals/ocean/trait/soup_a_plus_minus")

    print("Loading soup sweep scores from local scratch...")
    soup_scores = _load_local_soup_scores(run_root)
    print(f"  Found {len(soup_scores)} soup coefficients: {sorted(soup_scores.keys())}")

    print("Loading reference scores from HuggingFace...")
    refs = _load_hf_reference_scores()
    for label, scores in refs.items():
        print(f"  {label}: A={scores['Agreeableness']:.4f}")

    print("\nGenerating plots...")
    plot_trait_vs_coefficient(soup_scores, refs, OUTPUT_DIR)
    plot_delta_bars(soup_scores, refs, OUTPUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
