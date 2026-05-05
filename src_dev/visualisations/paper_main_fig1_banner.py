"""Single-figure Fig. 1 banner: amp spider + supp spider (left) + combo delta (right).

Combines the three sibling scripts into one wide banner with a shared legend:

  ┌────────────────┬─────────────────┐
  │  amp spider    │                 │
  ├────────────────┤  combo delta    │
  │  supp spider   │  bar chart      │
  └────────────────┴─────────────────┘
       ──── shared legend (below) ────

Color convention: OCEAN traits use ``BIG_FIVE_COLORS`` everywhere. In the
spiders, each polygon is colored by its *home* trait (the trait the LoRA
targets). In the bar chart, the c_minus_v2 / e_plus_v3 single-adapter bars
are colored by their home trait (Conscientiousness / Extraversion) and the
combo bar is rendered in a neutral "combo" color so it doesn't collide with
any OCEAN hue.

Tick labels on the spiders and bar chart are abbreviated to single OCEAN
letters (O / C / E / A / N) to keep the banner compact.

Data hydration is delegated to the three existing scripts:
  - ``paper_main_amplifier_spider_vanton4_paired_dpo.build_scores``
  - ``paper_main_suppressor_spider_vanton4_paired_dpo.build_scores``
  - ``paper_main_c_e_combo_delta.gather``

so per-cell HF paths and fingerprint conventions stay in one place and we
don't accumulate duplicate fetcher logic.

Paper figures (PDF + PNG):
    paper/figures/main/fig_1_banner.pdf
    paper/figures/main/fig_1_banner.png

Run with:
    uv run python -m src_dev.visualisations.paper_main_fig1_banner
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.visualisations import PAPER_FIGURES_DIR
from src_dev.visualisations.ocean_spider import to_headroom

from src_dev.visualisations import (
    paper_main_amplifier_spider_vanton4_paired_dpo as amp_mod,
    paper_main_suppressor_spider_vanton4_paired_dpo as sup_mod,
    paper_main_c_e_combo_delta as combo_mod,
)

PAPER_FIGURES = [
    "main/fig_1_banner.pdf",
    "main/fig_1_banner.png",
]

OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
OCEAN_LETTERS = ["O", "C", "E", "A", "N"]
OCEAN_KEY_BY_LETTER = dict(zip(OCEAN_LETTERS, OCEAN_TRAITS))

BASELINE_COLOR = "#4D4D4D"
COMBO_COLOR = "#00838F"
SCORE_MIN = -4.0
SCORE_MAX = 4.0
HEADROOM_LIM = (-1.0, 1.0)
HEADROOM_TICKS = [-1.0, -0.5, 0.0, 0.5, 1.0]
HEADROOM_LABELS = ["-100%", "-50%", "0", "+50%", "+100%"]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _spider_panel(
    ax,
    *,
    per_lora: dict[str, dict[str, float]],
    baseline: dict[str, float],
    meta: list[tuple[str, str, str]],
    title: str,
) -> None:
    """Render a single OCEAN spider in headroom mode.

    ``per_lora`` is keyed by adapter slug (e.g. ``o_plus``) and inner-keyed by
    judged-trait title. ``meta`` is a list of (slug, home_trait_lower, color)
    tuples in OCEAN order.
    """
    per_lora_h = to_headroom(per_lora, baseline, score_min=SCORE_MIN, score_max=SCORE_MAX)

    angles = np.linspace(0.0, 2.0 * np.pi, len(OCEAN_TRAITS), endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for slug, _home_trait, color in meta:
        row = per_lora_h.get(slug, {})
        if not row:
            continue
        means = [row.get(t, float("nan")) for t in OCEAN_TRAITS]
        means_closed = means + means[:1]
        ax.plot(angles_closed, means_closed, "-", color=color, linewidth=1.8)
        for angle, val in zip(angles, means):
            if not np.isnan(val):
                ax.plot([angle], [val], "o", color=color, markersize=4)
        if not any(np.isnan(v) for v in means):
            ax.fill(angles_closed, means_closed, color=color, alpha=0.10)

    # Baseline ring at r=0 (collapsed to a single circle in headroom mode).
    ring_theta = np.linspace(0.0, 2.0 * np.pi, 180)
    ax.plot(ring_theta, np.zeros_like(ring_theta), "-", color=BASELINE_COLOR, linewidth=2.0)

    ax.set_xticks(angles)
    ax.set_xticklabels(OCEAN_LETTERS, fontsize=10, fontweight="bold")
    for tick_label, letter in zip(ax.get_xticklabels(), OCEAN_LETTERS):
        tick_label.set_color(BIG_FIVE_COLORS[OCEAN_KEY_BY_LETTER[letter]])
    ax.set_ylim(*HEADROOM_LIM)
    ax.set_yticks(HEADROOM_TICKS)
    ax.set_yticklabels(HEADROOM_LABELS, fontsize=7)
    ax.grid(True, alpha=0.4)
    ax.set_title(title, fontsize=10, pad=10)


def _bar_panel(ax, *, scores: dict[str, dict[str, float | None]]) -> None:
    """Per-OCEAN-trait Δ-vs-baseline bars for c_adapter, e_adapter, combo."""
    keys = ["c_adapter", "e_adapter", "combo"]
    width = 0.8 / len(keys)
    x = np.arange(len(OCEAN_TRAITS))

    # Pull the c/e adapter slugs out of the combo module so colors track the
    # real trait identities of the underlying adapters, not whatever direction
    # they happen to take.
    c_color = BIG_FIVE_COLORS["Conscientiousness"]
    e_color = BIG_FIVE_COLORS["Extraversion"]
    color_by_key = {"c_adapter": c_color, "e_adapter": e_color, "combo": COMBO_COLOR}
    hatch_by_key = {"c_adapter": "//", "e_adapter": "\\\\", "combo": "xx"}

    for i, key in enumerate(keys):
        deltas = []
        for trait_title in OCEAN_TRAITS:
            row = scores.get(trait_title, {})
            base = row.get("baseline")
            val = row.get(key)
            deltas.append((val - base) if base is not None and val is not None else np.nan)
        ax.bar(
            x + (i - (len(keys) - 1) / 2) * width,
            deltas, width,
            color=color_by_key[key],
            hatch=hatch_by_key[key],
            edgecolor="black", linewidth=0.5,
        )

    ax.axhline(0.0, color=BASELINE_COLOR, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(OCEAN_LETTERS, fontsize=10, fontweight="bold")
    for tick_label, letter in zip(ax.get_xticklabels(), OCEAN_LETTERS):
        tick_label.set_color(BIG_FIVE_COLORS[OCEAN_KEY_BY_LETTER[letter]])
    ax.set_ylabel("Δ judge score vs base", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(
        f"{combo_mod.TARGET_DISPLAY['c_adapter']} ⊕ {combo_mod.TARGET_DISPLAY['e_adapter']} combo",
        fontsize=10, pad=10,
    )


def _shared_legend(fig) -> None:
    """One legend covering all panels: 5 OCEAN trait colors + combo + baseline."""
    handles = [
        Line2D([0], [0], color=BIG_FIVE_COLORS["Openness"],          marker="o", markersize=5, lw=1.8, label="Openness LoRA"),
        Line2D([0], [0], color=BIG_FIVE_COLORS["Conscientiousness"], marker="o", markersize=5, lw=1.8, label="Conscientiousness LoRA"),
        Line2D([0], [0], color=BIG_FIVE_COLORS["Extraversion"],      marker="o", markersize=5, lw=1.8, label="Extraversion LoRA"),
        Line2D([0], [0], color=BIG_FIVE_COLORS["Agreeableness"],     marker="o", markersize=5, lw=1.8, label="Agreeableness LoRA"),
        Line2D([0], [0], color=BIG_FIVE_COLORS["Neuroticism"],       marker="o", markersize=5, lw=1.8, label="Neuroticism LoRA"),
        Patch(facecolor=COMBO_COLOR, hatch="xx", edgecolor="black", linewidth=0.5, label="C↓ ⊕ E↑ combo"),
        Line2D([0], [0], color=BASELINE_COLOR, lw=2.0, label="Baseline (Llama-3.1-8B-Instruct)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=4, fontsize=9, frameon=True, framealpha=0.9,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("[fig1-banner] hydrating amplifier spider data...")
    per_amp, base_amp = amp_mod.build_scores()
    print("\n[fig1-banner] hydrating suppressor spider data...")
    per_sup, base_sup = sup_mod.build_scores()
    print("\n[fig1-banner] hydrating combo delta data...")
    combo_scores = combo_mod.gather()

    # Use either baseline (they should be identical — same prompts, same judge).
    baseline = base_amp or base_sup

    fig = plt.figure(figsize=(13.0, 5.6))
    gs = GridSpec(
        2, 2, figure=fig,
        width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0],
        wspace=0.25, hspace=0.35,
    )

    ax_amp  = fig.add_subplot(gs[0, 0], projection="polar")
    ax_sup  = fig.add_subplot(gs[1, 0], projection="polar")
    ax_bar  = fig.add_subplot(gs[:, 1])

    _spider_panel(
        ax_amp,
        per_lora=per_amp, baseline=baseline,
        meta=amp_mod.AMPLIFIERS_META,
        title="Amplifier LoRAs (scale +1, signed headroom)",
    )
    _spider_panel(
        ax_sup,
        per_lora=per_sup, baseline=baseline,
        meta=sup_mod.SUPPRESSORS,
        title="Suppressor LoRAs (scale +1, signed headroom)",
    )
    _bar_panel(ax_bar, scores=combo_scores)

    _shared_legend(fig)

    fig.tight_layout(rect=(0, 0.05, 1, 1))

    for rel in PAPER_FIGURES:
        out_path = PAPER_FIGURES_DIR / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"✓ saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
