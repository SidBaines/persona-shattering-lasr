#!/usr/bin/env python3
"""Plot TRAIT sweep averaged across 5 OCEAN control-seed runs, with
*cluster bootstrap* CIs that account for both within-seed sample noise and
between-seed training variance.

Aggregation (per scale, per trait):
  1. For each seed, load its ~1000 per-sample trait_logprobs scores and apply
     the same choice-mass filter used by the single-seed plots.
  2. Point estimate = mean over seed means (uniform weight per seed).
  3. Cluster bootstrap replicate: resample K=5 seeds with replacement; within
     each resampled seed resample its filtered samples with replacement; take
     the per-seed mean; average those K means.  Repeat B times.
  4. CI = percentile interval of the B replicate statistics.

This correctly collapses to within-seed CI when seeds are identical, and to
between-seed CI when within-seed noise is tiny.

Usage:
    uv run python -m scripts_dev.personality_evals.plot_control_seeds_avg_trait
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.gridspec import GridSpec

from src_dev.evals.personality.analyze_results import (
    ALL_TRAIT_COLS,
    BIG_FIVE,
    BIG_FIVE_COLORS,
    DARK_TRIAD,
    DARK_TRIAD_COLORS,
    _build_mass_mask,
    _set_scale_xticks,
    load_sweep_data,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEFAULT_RUN_TEMPLATE = (
    "scratch/evals/ocean/trait/ocean_def_control_vanton4_seed{seed}_logprobs_1000"
)
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
EVAL_NAME = "trait_logprobs"


@dataclass
class CellStats:
    """Filtered per-seed sample array plus its mean, for one (scale, trait) cell."""
    per_seed_samples: list[np.ndarray]  # len = K seeds; each array is n_filtered samples
    per_seed_means: np.ndarray           # shape (K,)
    point_estimate: float                # mean of per-seed means


def _filter_seed_samples(
    raw: np.ndarray,
    cm: np.ndarray | None,
    nc: np.ndarray | None,
    min_choice_mass: float,
    dynamic_mass_filter: bool,
) -> np.ndarray:
    if cm is None or len(cm) == 0:
        return raw
    n = min(len(raw), len(cm))
    raw = raw[:n]
    cm = cm[:n]
    if nc is not None:
        nc = nc[:n]
    mask = _build_mass_mask(cm, nc, min_choice_mass, dynamic_mass_filter)
    return raw[mask]


def build_cell(
    seed_dfs: list[pd.DataFrame],
    scale: float,
    trait: str,
    min_choice_mass: float,
    dynamic_mass_filter: bool,
) -> CellStats | None:
    """Assemble filtered samples for one (scale, trait) across all seeds."""
    per_seed: list[np.ndarray] = []
    for df in seed_dfs:
        rows = df[df["scale"] == scale]
        if rows.empty:
            continue
        raw_col = f"_raw_{trait}"
        cm_col = f"_raw__cm_{trait}"
        nc_col = f"_raw__nc_{trait}"
        if raw_col not in rows.columns:
            continue
        raw_lists = rows[raw_col].dropna().tolist()
        if not raw_lists:
            continue
        raw = np.concatenate(raw_lists)
        cm = None
        nc = None
        if cm_col in rows.columns:
            cm_lists = rows[cm_col].dropna().tolist()
            cm = np.concatenate(cm_lists) if cm_lists else None
        if nc_col in rows.columns:
            nc_lists = rows[nc_col].dropna().tolist()
            nc = np.concatenate(nc_lists) if nc_lists else None
        filtered = _filter_seed_samples(raw, cm, nc, min_choice_mass, dynamic_mass_filter)
        if len(filtered):
            per_seed.append(filtered)

    if not per_seed:
        return None
    per_seed_means = np.array([a.mean() for a in per_seed])
    return CellStats(
        per_seed_samples=per_seed,
        per_seed_means=per_seed_means,
        point_estimate=float(per_seed_means.mean()),
    )


def cluster_bootstrap_ci(
    cell: CellStats,
    n_resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Two-level (seed × sample) bootstrap. Percentile CI."""
    K = len(cell.per_seed_samples)
    sample_sizes = np.array([len(a) for a in cell.per_seed_samples])
    replicates = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        seed_idx = rng.integers(0, K, size=K)
        seed_means = np.empty(K, dtype=float)
        for k, s_i in enumerate(seed_idx):
            arr = cell.per_seed_samples[s_i]
            n = sample_sizes[s_i]
            draw = rng.integers(0, n, size=n)
            seed_means[k] = arr[draw].mean()
        replicates[b] = seed_means.mean()
    alpha = (100.0 - confidence) / 2.0
    lo = float(np.percentile(replicates, alpha))
    hi = float(np.percentile(replicates, 100.0 - alpha))
    return lo, hi


def choice_mass_per_scale(
    seed_dfs: list[pd.DataFrame],
    min_choice_mass: float,
) -> pd.DataFrame:
    """Pool per-sample ``_choice_mass`` across all seeds at each scale.

    Mirrors the diagnostic logic in ``plot_trait_sweep``: pull the raw
    per-sample choice mass arrays from every row at a given scale, apply the
    fixed ``min_choice_mass`` threshold, and return mean/min/max.
    """
    scales = sorted({s for df in seed_dfs for s in df["scale"].unique()})
    rows: list[dict] = []
    for scale in scales:
        lists: list[np.ndarray] = []
        for df in seed_dfs:
            grp = df[df["scale"] == scale]
            if "_raw__choice_mass" in grp.columns:
                for v in grp["_raw__choice_mass"].tolist():
                    if isinstance(v, list) and v:
                        lists.append(np.asarray(v))
            elif "_choice_mass" in grp.columns:
                vals = grp["_choice_mass"].dropna().values
                if len(vals):
                    lists.append(vals)
        cm_all = np.concatenate(lists) if lists else np.array([])
        if min_choice_mass > 0.0:
            cm_all = cm_all[cm_all >= min_choice_mass]
        if len(cm_all):
            rows.append({"scale": scale, "mean": float(cm_all.mean()),
                         "min": float(cm_all.min()), "max": float(cm_all.max())})
        else:
            rows.append({"scale": scale, "mean": float("nan"),
                         "min": float("nan"), "max": float("nan")})
    return pd.DataFrame(rows)


def aggregate(
    seed_dfs: list[pd.DataFrame],
    traits: list[str],
    min_choice_mass: float,
    dynamic_mass_filter: bool,
    n_resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Aggregate per-seed data into per-(scale, trait) point estimates and CIs.

    For each (scale, trait) cell we also record ``{trait}_n_seeds`` — how many
    seeds contributed filtered samples.  With fewer than 2 seeds a cluster
    bootstrap has no between-seed variance, so we leave the CI as NaN rather
    than emit a misleadingly tight interval.
    """
    scales = sorted({s for df in seed_dfs for s in df["scale"].unique()})
    rows: list[dict] = []
    for scale in scales:
        row: dict = {"scale": scale}
        for trait in traits:
            cell = build_cell(seed_dfs, scale, trait, min_choice_mass, dynamic_mass_filter)
            if cell is None:
                row[f"{trait}_mean"] = float("nan")
                row[f"{trait}_ci_low"] = float("nan")
                row[f"{trait}_ci_high"] = float("nan")
                row[f"{trait}_n_seeds"] = 0
                continue
            k_prime = len(cell.per_seed_samples)
            row[f"{trait}_mean"] = cell.point_estimate
            row[f"{trait}_n_seeds"] = k_prime
            if k_prime < 2:
                row[f"{trait}_ci_low"] = float("nan")
                row[f"{trait}_ci_high"] = float("nan")
            else:
                lo, hi = cluster_bootstrap_ci(cell, n_resamples, confidence, rng)
                row[f"{trait}_ci_low"] = lo
                row[f"{trait}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def plot_from_agg(
    agg: pd.DataFrame,
    output_dir: Path,
    title_suffix: str,
    ci_label: str,
    cm_agg: pd.DataFrame | None = None,
    min_choice_mass: float = 0.0,
    x_lim: tuple[float, float] | None = (-4.5, 4.5),
) -> Path:
    has_cm = cm_agg is not None and not cm_agg["mean"].isna().all()
    if has_cm:
        fig = plt.figure(figsize=(12, 6.6))
        gs = GridSpec(2, 1, height_ratios=[85, 15], hspace=0.05, figure=fig)
        ax = fig.add_subplot(gs[0])
        ax_cm = fig.add_subplot(gs[1], sharex=ax)
    else:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax_cm = None
    scales = agg["scale"].values

    for trait in BIG_FIVE:
        color = BIG_FIVE_COLORS[trait]
        means = agg[f"{trait}_mean"].values
        lo = agg[f"{trait}_ci_low"].values
        hi = agg[f"{trait}_ci_high"].values
        ax.plot(scales, means, "o-", color=color, linewidth=2.2, markersize=6,
                label=trait, zorder=4)
        # Suppress error bars wherever the CI was not computed (K'<2 or no
        # data): matplotlib renders NaN-yerr as no bar, which is what we want.
        low_arm = means - lo
        high_arm = hi - means
        yerr = np.vstack([low_arm, high_arm])
        ax.errorbar(scales, means, yerr=yerr, fmt="none", color=color,
                    capsize=3, capthick=1.0, elinewidth=1.0, alpha=0.8, zorder=3)

    for trait in DARK_TRIAD:
        col = f"{trait}_mean"
        if col not in agg.columns or agg[col].isna().all():
            continue
        color = DARK_TRIAD_COLORS[trait]
        ax.plot(scales, agg[col].values, "--", color=color, linewidth=1.4,
                markersize=4, alpha=0.35, label=trait, zorder=3)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
    ax.set_ylabel("Trait score (0–1)", fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)

    title = "TRAIT sweep (avg over control seeds, cluster bootstrap CI)"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title, fontsize=13, fontweight="bold")

    ax.errorbar([], [], yerr=1, fmt="none", color="gray", capsize=3, capthick=1.0,
                elinewidth=1.0, alpha=0.7, label=ci_label)

    if ax_cm is not None:
        cm_scales = cm_agg["scale"].values
        cm_means = cm_agg["mean"].values
        ax_cm.plot(cm_scales, cm_means, "s-", color="#555555", linewidth=1.4,
                   markersize=3, zorder=4)
        ax_cm.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
        ax_cm.set_ylabel("Choice\nmass", fontsize=8, rotation=0, labelpad=32, va="center")
        cm_lower = max(0.0, min(float(min_choice_mass), 1.0))
        ax_cm.set_ylim(cm_lower, 1.0)
        ax_cm.set_yticks([cm_lower, 1.0])
        ax_cm.set_yticklabels([f"{cm_lower:g}", "1"], fontsize=7)
        ax_cm.grid(True, alpha=0.25)
        ax_cm.set_xlabel("LoRA scaling factor", fontsize=11)
        _set_scale_xticks(ax_cm, scales, x_lim=x_lim)
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel("LoRA scaling factor", fontsize=11)
        _set_scale_xticks(ax, scales, x_lim=x_lim)

    ax.legend(loc="upper center",
              bbox_to_anchor=(0.5, -0.13 if ax_cm is None else -0.35),
              fontsize=9, ncol=6, framealpha=0.85)

    if ax_cm is None:
        plt.tight_layout()
    out = output_dir / "trait_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


def _load_seed_df(run_dir: Path) -> pd.DataFrame:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir missing: {run_dir}")
    data = load_sweep_data(run_dir)
    df = data.get(EVAL_NAME)
    if df is None or df.empty:
        raise RuntimeError(f"No '{EVAL_NAME}' data in {run_dir}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--run-template", type=str, default=DEFAULT_RUN_TEMPLATE)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("scratch/evals/ocean/trait/ocean_def_control_vanton4_seeds_avg_logprobs_1000/figures"))
    parser.add_argument("--title-suffix", type=str,
                        default="OCEAN control vanton4 avg of 5 seeds TRAIT (logprobs)")
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=95.0)
    parser.add_argument("--min-choice-mass", type=float, default=0.75)
    parser.add_argument("--no-dynamic-mass-filter", dest="dynamic_mass_filter",
                        action="store_false")
    parser.set_defaults(dynamic_mass_filter=True)
    args = parser.parse_args()

    run_dirs = [Path(args.run_template.format(seed=s)) for s in args.seeds]
    print(f"Loading {len(run_dirs)} per-seed run dirs:")
    seed_dfs: list[pd.DataFrame] = []
    for p in run_dirs:
        print(f"  - {p}")
        seed_dfs.append(_load_seed_df(p))

    rng = np.random.default_rng(SEED)
    print(f"Cluster bootstrap: K={len(seed_dfs)} seeds × B={args.n_resamples} resamples, "
          f"CI={args.confidence:g}%, min_cm={args.min_choice_mass}, "
          f"dyn={args.dynamic_mass_filter}")
    agg = aggregate(
        seed_dfs, ALL_TRAIT_COLS,
        min_choice_mass=args.min_choice_mass,
        dynamic_mass_filter=args.dynamic_mass_filter,
        n_resamples=args.n_resamples,
        confidence=args.confidence,
        rng=rng,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    agg_csv = args.output_dir / "trait_sweep_agg.csv"
    agg.to_csv(agg_csv, index=False)
    print(f"  wrote aggregated stats → {agg_csv}")

    K = len(seed_dfs)
    n_seed_cols = [c for c in agg.columns if c.endswith("_n_seeds")]
    if n_seed_cols:
        counts = agg[n_seed_cols].to_numpy()
        missing = int((counts == 0).sum())
        degenerate = int((counts == 1).sum())
        partial = int(((counts > 1) & (counts < K)).sum())
        if missing or degenerate or partial:
            print(
                f"  note: of (scale, trait) cells, {missing} had no data, "
                f"{degenerate} had only 1 seed (CI suppressed), "
                f"{partial} had 2–{K - 1} seeds (CI still computed but K'<{K}). "
                f"See *_n_seeds columns in the CSV."
            )

    cm_agg = choice_mass_per_scale(seed_dfs, min_choice_mass=args.min_choice_mass)
    ci_label = f"{args.confidence:g}% CI (cluster bootstrap, K={len(seed_dfs)}, B={args.n_resamples})"
    plot_from_agg(agg, args.output_dir, args.title_suffix, ci_label,
                  cm_agg=cm_agg, min_choice_mass=args.min_choice_mass)


if __name__ == "__main__":
    main()
