"""Per-turn OCEAN trajectory heatmaps for frustration_eval rollouts.

Reads the combined per-turn OCEAN scores written by
``scripts_dev.frustration_eval.score_ocean_per_turn`` and produces three figures:

1. Absolute scores: 5 traits × N turns, one stacked panel per condition,
   diverging colormap centered at 0 (rubric is -4..+4).
2. Deviation from base: each non-base panel is condition - base, diverging at 0.
3. Deviation from model baseline: each panel (including base) shows the cell
   minus the model's overall baseline trait score, fetched from the canonical
   ``combos/<model>/_baseline/llm_judge_lora_scale_sweep`` cells on HF. This
   reveals how much the scenario itself shifts the model's traits relative to
   its normal behavior, on top of any LoRA / activation-cap effect.

Outputs (registered for paper):
- paper/figures/main/fig_main_frustration_ocean_abs.pdf
- paper/figures/main/fig_main_frustration_ocean_delta.pdf
- paper/figures/main/fig_main_frustration_ocean_delta_vs_baseline.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from src_dev.visualisations import PAPER_FIGURES_DIR

PAPER_FIGURES = [
    "main/fig_main_frustration_ocean_abs.pdf",
    "main/fig_main_frustration_ocean_delta.pdf",
    "main/fig_main_frustration_ocean_delta_vs_baseline.pdf",
    "main/fig_main_frustration_ocean_pct_headroom.pdf",
]

# OCEAN v2 rubric bounds; needed for percent-of-headroom normalization.
SCORE_MIN = -4.0
SCORE_MAX = 4.0

TRAITS = [
    "openness_v2",
    "conscientiousness_v2",
    "extraversion_v2",
    "agreeableness_v2",
    "neuroticism_v2",
]
TRAIT_LABELS = {
    "openness_v2": "Openness",
    "conscientiousness_v2": "Conscientiousness",
    "extraversion_v2": "Extraversion",
    "agreeableness_v2": "Agreeableness",
    "neuroticism_v2": "Neuroticism",
}
CONDITION_ORDER = ["base", "n_minus", "control", "n_neg", "axiscap"]
CONDITION_LABELS = {
    "base": "Base (no intervention)",
    "n_minus": "LoRA: N− (suppressor)",
    "control": "LoRA: control",
    "n_neg": "LoRA: N− negated (≈ amplifier)",
    "axiscap": "Activation cap (+1.0 on N axis)",
}


def load_combined(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """Aggregate across all prompts in `path`.

    Returns (means, sems, n_prompts) where each dict maps condition →
    (n_traits, n_turns) array. ``means`` is the per-cell mean over prompts;
    ``sems`` is the standard error of the mean (NaN where n<2).
    """
    rows = [json.loads(l) for l in open(path)]
    if not rows:
        raise ValueError(f"No rows in {path}")

    # cond → (trait_idx, turn) → list of scores across prompts
    bucket: dict[str, dict[tuple[int, int], list[float]]] = defaultdict(lambda: defaultdict(list))
    n_turns = 0
    prompts_seen: set[str] = set()
    for r in rows:
        prompts_seen.add(r["prompt_hash"])
        ti_turn = r["turn_index"]
        n_turns = max(n_turns, ti_turn + 1)
        for ti, trait in enumerate(TRAITS):
            if trait in r["scores"]:
                bucket[r["condition"]][(ti, ti_turn)].append(float(r["scores"][trait]))

    means: dict[str, np.ndarray] = {}
    sems: dict[str, np.ndarray] = {}
    for cond, cells in bucket.items():
        mean_mat = np.full((len(TRAITS), n_turns), np.nan)
        sem_mat = np.full((len(TRAITS), n_turns), np.nan)
        for (ti, tj), values in cells.items():
            arr = np.asarray(values, dtype=float)
            mean_mat[ti, tj] = float(arr.mean())
            if arr.size > 1:
                sem_mat[ti, tj] = float(arr.std(ddof=1) / np.sqrt(arr.size))
        means[cond] = mean_mat
        sems[cond] = sem_mat
    return means, sems, len(prompts_seen)


def _heatmap_panels(
    matrices: dict[str, np.ndarray],
    *,
    title: str,
    cbar_label: str,
    vmin: float,
    vmax: float,
    cmap: str,
    out_path: Path,
    sems: dict[str, np.ndarray] | None = None,
    annotate_fmt: str = "{:+.1f}",
    sem_fmt: str = "{:.1f}",
) -> None:
    conds = [c for c in CONDITION_ORDER if c in matrices]
    n_panels = len(conds)
    n_traits, n_turns = matrices[conds[0]].shape

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(0.9 * n_turns + 2.0, 1.1 * n_traits * n_panels / 2 + 0.6 * n_panels),
        squeeze=False, sharex=True,
    )
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    im = None
    for i, cond in enumerate(conds):
        ax = axes[i, 0]
        mat = matrices[cond]
        im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)
        ax.set_yticks(range(n_traits))
        ax.set_yticklabels([TRAIT_LABELS[t] for t in TRAITS], fontsize=9)
        ax.set_xticks(range(n_turns))
        if i == n_panels - 1:
            ax.set_xticklabels([f"t{j}" for j in range(n_turns)], fontsize=9)
            ax.set_xlabel("Assistant turn")
        else:
            ax.set_xticklabels([])
        ax.set_title(CONDITION_LABELS.get(cond, cond), fontsize=10, loc="left", pad=2)

        sem_mat = sems[cond] if sems is not None and cond in sems else None
        for ti in range(n_traits):
            for tj in range(n_turns):
                v = mat[ti, tj]
                if np.isnan(v):
                    continue
                label = annotate_fmt.format(v)
                if sem_mat is not None and not np.isnan(sem_mat[ti, tj]):
                    label = f"{label}\n±{sem_fmt.format(sem_mat[ti, tj])}"
                ax.text(tj, ti, label,
                        ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > (vmax - vmin) * 0.3 else "black")

    fig.suptitle(title, fontsize=11)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical",
                        fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_plots(combined_path: Path, out_dir: Path, *, prompt_text: str | None = None) -> None:
    matrices, sems, n_prompts = load_combined(combined_path)
    title_suffix = f"  (mean over n={n_prompts} prompt{'s' if n_prompts != 1 else ''})"
    if prompt_text and n_prompts == 1:
        snippet = prompt_text[:80].replace("\n", " ")
        title_suffix = f"\nPrompt: {snippet}…"

    _heatmap_panels(
        matrices,
        title=f"OCEAN trait scores per turn (impossible_numeric){title_suffix}",
        cbar_label="OCEAN v2 score (−4..+4)",
        vmin=-4.0, vmax=4.0, cmap="RdBu_r",
        out_path=out_dir / "main" / "fig_main_frustration_ocean_abs.pdf",
        sems=sems if n_prompts > 1 else None,
    )

    if "base" not in matrices:
        print("[warn] no 'base' condition; skipping deviation plot.", file=sys.stderr)
        return

    base_mat = matrices["base"]
    delta = {cond: mat - base_mat for cond, mat in matrices.items() if cond != "base"}
    if not delta:
        return

    # SEM of the difference of two condition-means, treating both as independent
    # samples over n_prompts (rough but useful for an at-a-glance noise gauge).
    delta_sems: dict[str, np.ndarray] | None = None
    if n_prompts > 1 and sems and "base" in sems:
        base_sem = sems["base"]
        delta_sems = {}
        for cond, _ in delta.items():
            cond_sem = sems.get(cond)
            if cond_sem is None:
                continue
            delta_sems[cond] = np.sqrt(np.square(base_sem) + np.square(cond_sem))

    max_abs = float(np.nanmax(np.abs(np.stack(list(delta.values())))))
    bound = max(1.0, np.ceil(max_abs))
    _heatmap_panels(
        delta,
        title=f"OCEAN trait scores: deviation from base{title_suffix}",
        cbar_label="Δ score (condition − base)",
        vmin=-bound, vmax=bound, cmap="RdBu_r",
        out_path=out_dir / "main" / "fig_main_frustration_ocean_delta.pdf",
        sems=delta_sems,
    )

    # ----- Plot 3: deviation from model's overall baseline trait scores -----
    try:
        from scripts_dev.frustration_eval.baseline_ocean_means import (
            get_baseline_means,
        )
    except Exception as exc:
        print(f"[warn] could not import baseline_ocean_means; "
              f"skipping baseline-delta plot: {exc}", file=sys.stderr)
        return

    baseline_means = get_baseline_means()
    # Per-trait subtraction vector (n_traits,)
    bvec = np.array([baseline_means[t]["mean"] for t in TRAITS])[:, None]  # broadcast over turns
    bsem = np.array([baseline_means[t]["sem"] for t in TRAITS])[:, None]

    delta_baseline = {cond: mat - bvec for cond, mat in matrices.items()}
    delta_baseline_sems: dict[str, np.ndarray] | None = None
    if n_prompts > 1 and sems:
        delta_baseline_sems = {}
        for cond, mat in matrices.items():
            cond_sem = sems.get(cond)
            if cond_sem is None:
                continue
            # SEM of (per-prompt mean) − (independent baseline sample mean).
            delta_baseline_sems[cond] = np.sqrt(np.square(cond_sem) + np.square(bsem))

    max_abs_b = float(np.nanmax(np.abs(np.stack(list(delta_baseline.values())))))
    bound_b = max(1.0, np.ceil(max_abs_b))
    _heatmap_panels(
        delta_baseline,
        title=(f"OCEAN trait scores: deviation from model baseline"
               f"{title_suffix}\n(baseline = Gemma-3-27b-IT mean on "
               f"data/ocean_open_ended/<trait>.jsonl, n=240/trait, Qwen3-235B judge)"),
        cbar_label="Δ score (condition − model baseline)",
        vmin=-bound_b, vmax=bound_b, cmap="RdBu_r",
        out_path=out_dir / "main" / "fig_main_frustration_ocean_delta_vs_baseline.pdf",
        sems=delta_baseline_sems,
    )

    # ----- Plot 4: % of available headroom toward the trait's pole -----
    # Normalise each cell's deviation by the available range to that pole, so
    # traits with little upward headroom (e.g. C baseline ≈ +3.76) are visually
    # comparable to traits with more (e.g. N baseline ≈ −1.09).
    headroom_up = SCORE_MAX - bvec      # shape (n_traits, 1); >0
    headroom_dn = bvec - SCORE_MIN      # shape (n_traits, 1); >0
    # Cell-wise: divide positive deltas by headroom_up, negative by headroom_dn.
    pct_headroom: dict[str, np.ndarray] = {}
    for cond, dmat in delta_baseline.items():
        denom = np.where(dmat >= 0, headroom_up, headroom_dn)
        # Avoid divide-by-zero for the (impossible) case denom == 0.
        denom = np.where(denom == 0, np.nan, denom)
        pct_headroom[cond] = dmat / denom  # signed in [-1, +1]

    pct_sems: dict[str, np.ndarray] | None = None
    if delta_baseline_sems is not None:
        pct_sems = {}
        for cond, dsem in delta_baseline_sems.items():
            denom_abs = np.where(delta_baseline[cond] >= 0, headroom_up, headroom_dn)
            denom_abs = np.where(denom_abs == 0, np.nan, denom_abs)
            pct_sems[cond] = dsem / denom_abs

    _heatmap_panels(
        pct_headroom,
        title=(f"OCEAN trait scores: % of available headroom used "
               f"toward the relevant pole{title_suffix}\n"
               f"(headroom = distance from model baseline to ±4 rubric bound, "
               f"per trait)"),
        cbar_label="Δ score / available headroom (signed, ±1 = saturated)",
        vmin=-1.0, vmax=1.0, cmap="RdBu_r",
        out_path=out_dir / "main" / "fig_main_frustration_ocean_pct_headroom.pdf",
        sems=pct_sems,
        annotate_fmt="{:+.0%}",
        sem_fmt="{:.0%}",
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", type=Path, required=True,
                    help="Path to ocean_per_turn_<phash>.jsonl produced by score script.")
    ap.add_argument("--prompt-text", type=str, default=None,
                    help="Original prompt text for the figure subtitle (optional).")
    ap.add_argument("--out-dir", type=Path, default=PAPER_FIGURES_DIR,
                    help="Output figures dir (default: paper/figures/).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    make_plots(args.combined, args.out_dir, prompt_text=args.prompt_text)
    print("wrote:")
    for p in PAPER_FIGURES:
        print(f"  {args.out_dir / p}")


if __name__ == "__main__":
    main()
