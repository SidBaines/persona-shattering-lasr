#!/usr/bin/env python3
"""Plot answer-choice distributions across the LoRA sweep for BFI and TRAIT.

Uses the custom fallback parser from scripts.evals.log_answer_parser so that
answers recovered from non-standard model output formats are included.

Usage:
    uv run python scripts/visualisations/plot_answer_distributions.py \\
        scripts/evals/fetched_logs \\
        --output-dir scripts/evals/fetched_logs/analysis
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Allow running from project root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.evals.personality.log_answer_parser import load_logs, LogStats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SWEEP_ORDER = [
    "lora_-2p00x", "lora_-1p50x", "lora_-1p00x", "lora_-0p50x",
    "base",
    "lora_+0p00x", "lora_+0p50x", "lora_+1p00x", "lora_+1p50x", "lora_+2p00x",
]

LETTER_COLORS = {
    # BFI  A=disagree strongly … E=agree strongly
    "A": "#D32F2F",   # deep red
    "B": "#FF7043",   # orange-red
    "C": "#FDD835",   # yellow (neutral)
    "D": "#66BB6A",   # green
    "E": "#1976D2",   # blue
    # TRAIT only uses A-D; colours reused from above
}

FAILURE_COLORS = {
    "degenerate": "#BDBDBD",
    "rant":       "#E0E0E0",
    "other":      "#F5F5F5",
}


def _parse_scale(model_name: str) -> float:
    if model_name == "base":
        return 0.0
    m = re.match(r"lora_([+-]?)(\d+)p(\d+)x", model_name)
    if m:
        sign = -1.0 if m.group(1) == "-" else 1.0
        return sign * (int(m.group(2)) + int(m.group(3)) / 100.0)
    return float("nan")


# ---------------------------------------------------------------------------
# Plot 1 — stacked bar: letter fractions per model
# ---------------------------------------------------------------------------

def plot_stacked_bars(
    stats_list: list[LogStats],
    letters: str,
    eval_type: str,
    output_dir: Path,
) -> None:
    """Stacked bar chart: one bar per sweep model, stacked by letter fraction.
    Failures shown as hatched grey at the top.
    """
    models = [s.model for s in stats_list]
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.1), 5))

    bottoms = np.zeros(len(models))

    # Valid-answer letters
    for letter in letters:
        fracs = np.array([
            s.counts.get(letter, 0) / s.n_total if s.n_total > 0 else 0.0
            for s in stats_list
        ])
        ax.bar(x, fracs, bottom=bottoms, color=LETTER_COLORS[letter],
               label=letter, width=0.7, edgecolor="white", linewidth=0.5)
        # Annotate non-trivial slices
        for xi, (frac, bot) in enumerate(zip(fracs, bottoms)):
            if frac >= 0.05:
                ax.text(xi, bot + frac / 2, f"{frac*100:.0f}%",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if frac > 0.15 else "black", fontweight="bold")
        bottoms += fracs

    # Failure slices
    fail_labels = ["degenerate", "rant", "other"]
    fail_bottoms = bottoms.copy()
    for kind in fail_labels:
        fracs = np.array([
            s.failures.get(kind, 0) / s.n_total if s.n_total > 0 else 0.0
            for s in stats_list
        ])
        ax.bar(x, fracs, bottom=fail_bottoms, color=FAILURE_COLORS[kind],
               label=f"fail:{kind}", width=0.7, edgecolor="white", linewidth=0.5,
               hatch="///" if kind == "degenerate" else ".." if kind == "rant" else "")
        fail_bottoms += fracs

    # Baseline marker
    base_idx = next((i for i, s in enumerate(stats_list) if s.model == "base"), None)
    if base_idx is not None:
        ax.axvline(base_idx, color="black", linestyle="--", linewidth=1.2, alpha=0.6, zorder=5)
        ax.text(base_idx + 0.08, 1.02, "base", fontsize=8, color="black", alpha=0.7, transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("lora_", "").replace("p", ".") for m in models],
        rotation=35, ha="right", fontsize=9,
    )
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Fraction of all samples", fontsize=11)
    ax.set_xlabel("LoRA scale", fontsize=11)
    title = f"{eval_type.upper()}: Answer-choice distribution across LoRA sweep"
    ax.set_title(title, fontsize=13, fontweight="bold")

    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles, lbls, loc="upper left", fontsize=8,
              ncol=len(letters) + len(fail_labels), framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out = output_dir / f"{eval_type}_answer_dist_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# Plot 2 — line chart: fraction of each letter vs scale (valid only)
# ---------------------------------------------------------------------------

def plot_letter_lines(
    stats_list: list[LogStats],
    letters: str,
    eval_type: str,
    output_dir: Path,
) -> None:
    """Line chart: fraction of each letter (among valid answers) vs LoRA scale."""
    scales = np.array([_parse_scale(s.model) for s in stats_list])
    valid_mask = ~np.isnan(scales) & np.array([s.n_valid > 0 for s in stats_list])

    fig, ax = plt.subplots(figsize=(10, 5))

    uniform = 1.0 / len(letters)
    ax.axhline(uniform, color="gray", linestyle=":", linewidth=1.2,
               alpha=0.7, label=f"Uniform ({uniform*100:.0f}%)")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    for letter in letters:
        fracs = np.array([
            s.counts.get(letter, 0) / s.n_valid if s.n_valid > 0 else float("nan")
            for s in stats_list
        ])
        mask = valid_mask & ~np.isnan(fracs)
        ax.plot(scales[mask], fracs[mask], "o-", color=LETTER_COLORS[letter],
                linewidth=2.2, markersize=7, label=letter)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlabel("LoRA scale", fontsize=12)
    ax.set_ylabel("Fraction of valid answers", fontsize=12)
    ax.set_title(
        f"{eval_type.upper()}: Letter-choice fractions vs LoRA scale\n(valid/parseable answers only)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    out = output_dir / f"{eval_type}_answer_dist_lines.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# Plot 3 — failure heatmap
# ---------------------------------------------------------------------------

def plot_failure_heatmap(
    stats_list: list[LogStats],
    eval_type: str,
    output_dir: Path,
) -> None:
    """Heatmap showing valid vs each failure kind per model."""
    models = [s.model.replace("lora_", "").replace("p", ".") for s in stats_list]
    rows = ["valid", "degenerate", "rant", "other"]
    data = np.array([
        [s.n_valid / s.n_total if s.n_total else 0 for s in stats_list],
        [s.failures.get("degenerate", 0) / s.n_total if s.n_total else 0 for s in stats_list],
        [s.failures.get("rant", 0) / s.n_total if s.n_total else 0 for s in stats_list],
        [s.failures.get("other", 0) / s.n_total if s.n_total else 0 for s in stats_list],
    ])

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.0), 3.5))
    cmap = plt.get_cmap("RdYlGn")
    # valid=green, failures=red — flip for failure rows
    display = data.copy()
    display[1:] = 1 - display[1:]  # invert so more failure = more red
    im = ax.imshow(display, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=10)

    for i in range(len(rows)):
        for j in range(len(models)):
            v = data[i, j]
            ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center",
                    fontsize=8, color="white" if display[i, j] > 0.6 else "black")

    ax.set_title(f"{eval_type.upper()}: Parse outcomes per model", fontsize=12, fontweight="bold")
    plt.tight_layout()

    out = output_dir / f"{eval_type}_parse_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot answer-choice distributions from inspect logs")
    parser.add_argument("log_dir", type=Path, help="Directory containing fetched inspect logs")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--eval-types", nargs="+", choices=["bfi", "trait"],
                        default=["bfi", "trait"])
    args = parser.parse_args()

    output_dir = args.output_dir or args.log_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    for eval_type in args.eval_types:
        letters = "ABCDE" if eval_type == "bfi" else "ABCD"
        print(f"\nLoading {eval_type.upper()} logs…")
        all_stats = load_logs(args.log_dir, eval_type)

        # Sort by sweep order, drop unknowns
        order_map = {m: i for i, m in enumerate(SWEEP_ORDER)}
        stats_list = sorted(
            [s for s in all_stats if s.model in order_map],
            key=lambda s: order_map[s.model],
        )

        if not stats_list:
            print(f"  No logs found for {eval_type}, skipping.")
            continue

        print(f"  {len(stats_list)} models, generating plots…")
        plot_stacked_bars(stats_list, letters, eval_type, output_dir)
        plot_letter_lines(stats_list, letters, eval_type, output_dir)
        plot_failure_heatmap(stats_list, eval_type, output_dir)

    print(f"\n✅ All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
