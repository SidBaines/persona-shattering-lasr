#!/usr/bin/env python3
"""Phase 5 — Drift trajectory plots, faceted by axis variant.

Reads ``drift_projections.jsonl`` (which now includes an ``axis_variant``
column thanks to the multi-axis Phase 4) and produces one paper-Figure-7-
analog per axis. So if you've built both a ``base`` axis and a
``lora_soup_c_plus_o_minus`` axis, you get two trajectory figures: one
showing "drift relative to base Assistant" and one showing "drift
relative to LoRA-modified-model Assistant".

Outputs (under ``{scratch_dir}/plots/`` and copied to
``paper/figures/appendix/``):

    drift_trajectory_{axis_variant}_layer{N}.{png,pdf}  — Fig 7 analog
    drift_heatmap_{axis_variant}_{condition}_{domain}.png — per-layer heatmap
    axis_cosine_similarity.txt                          — written by Phase 4

Usage::

    uv run python -m scripts_dev.persona_drift_assistant_axis.plot_drift \\
        --preset smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src_dev.evals.personality.analyze_results import _interval_ci_from_bootstrap  # noqa: E402
from src_dev.visualisations import PAPER_FIGURES_DIR  # noqa: E402
from scripts_dev.persona_drift_assistant_axis.config import (  # noqa: E402
    ExperimentConfig,
    get_preset,
)

# Output paths declared at module top per repo convention (paper/CLAUDE.md).
PAPER_FIGURES = [
    "appendix/fig_assistant_axis_drift_trajectory_base.png",
    "appendix/fig_assistant_axis_drift_trajectory_lora_soup_c_plus_o_minus.png",
]


# ── Loading + aggregation ────────────────────────────────────────────────


def load_projections(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def aggregate_per_turn(
    rows: list[dict], *, axis_variant: str, target_layer: int
) -> dict[tuple[str, str, int], dict]:
    """Aggregate to (condition, domain, turn_position) → {mean, ci_lo, ci_hi, n}.

    Filters to ``axis_variant`` first so each call returns one variant's data.
    """
    buckets: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for r in rows:
        if r.get("axis_variant") != axis_variant:
            continue
        proj = r["projection_per_layer"][target_layer]
        buckets[(r["condition"], r["domain"], r["turn_position"])].append(proj)
    out: dict[tuple[str, str, int], dict] = {}
    for k, vals in buckets.items():
        arr = np.asarray(vals)
        if len(arr) >= 3:
            lo, hi = _interval_ci_from_bootstrap(arr, confidence=95, n_resamples=500, seed=42)
        else:
            lo, hi = float(arr.min()), float(arr.max())
        out[k] = {
            "mean": float(arr.mean()),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "n": int(len(arr)),
        }
    return out


# ── Plotting ─────────────────────────────────────────────────────────────


_CONDITION_COLORS = {
    "vanilla": "#9e9e9e",
    "activation_capping": "#1f77b4",
    "lora_soup_c_plus_o_minus": "#d62728",
}
_CONDITION_LABEL = {
    "vanilla": "Vanilla",
    "activation_capping": "Activation capping (paper)",
    "lora_soup_c_plus_o_minus": "LoRA soup C+ ⊕ O−",
}
_AXIS_VARIANT_LABEL = {
    "base": "Base-model Assistant Axis",
    "lora_soup_c_plus_o_minus": "LoRA-soup-model Assistant Axis",
}


def plot_trajectory(
    agg: dict, *, domains: list[str], conditions: list[str],
    axis_variant: str, target_layer: int, output_path: Path,
) -> None:
    """Per-domain trajectory: mean projection ± 95% CI vs turn position."""
    n = len(domains)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, domain in zip(axes, domains):
        for condition in conditions:
            xs, ys, lo, hi = [], [], [], []
            turn_positions = sorted(
                {tp for c, d, tp in agg if c == condition and d == domain}
            )
            for tp in turn_positions:
                row = agg.get((condition, domain, tp))
                if row is None:
                    continue
                xs.append(tp + 1)
                ys.append(row["mean"])
                lo.append(row["ci_lo"])
                hi.append(row["ci_hi"])
            if xs:
                color = _CONDITION_COLORS.get(condition, "black")
                ax.plot(xs, ys, marker="o", color=color,
                        label=_CONDITION_LABEL.get(condition, condition))
                ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
        ax.set_title(domain)
        ax.set_xlabel("Turn")
        ax.set_ylabel(f"Projection onto {_AXIS_VARIANT_LABEL.get(axis_variant, axis_variant)}")
        ax.axhline(0, color="black", lw=0.5, alpha=0.3)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(
        f"Persona drift trajectory — axis: {axis_variant}, layer {target_layer}",
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


def plot_layer_heatmap(
    rows: list[dict], *,
    axis_variant: str, condition: str, domain: str, output_path: Path,
) -> None:
    """Heatmap: turn_position × layer of mean projection for one cell."""
    cell: dict[tuple[int, int], list[float]] = defaultdict(list)
    n_layers = 0
    for r in rows:
        if (r.get("axis_variant") != axis_variant
                or r["condition"] != condition or r["domain"] != domain):
            continue
        proj = r["projection_per_layer"]
        n_layers = len(proj)
        for layer, p in enumerate(proj):
            cell[(r["turn_position"], layer)].append(p)
    if not cell:
        return
    turns = sorted({k[0] for k in cell})
    grid = np.zeros((len(turns), n_layers))
    for ti, tp in enumerate(turns):
        for layer in range(n_layers):
            vals = cell.get((tp, layer), [])
            grid[ti, layer] = float(np.mean(vals)) if vals else np.nan

    fig, ax = plt.subplots(figsize=(10, max(3, len(turns) * 0.3)))
    im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", origin="lower")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Turn position")
    ax.set_title(
        f"Projection per (turn, layer) — axis={axis_variant}\n"
        f"{condition} / {domain}"
    )
    fig.colorbar(im, ax=ax, label="Projection")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


# ── Top-level ────────────────────────────────────────────────────────────


def plot_drift(cfg: ExperimentConfig, *, target_layer: int | None = None) -> None:
    proj_path = cfg.scratch_dir / "drift_projections.jsonl"
    if not proj_path.exists():
        raise SystemExit(f"Projections missing at {proj_path}; run Phase 4 first.")
    rows = load_projections(proj_path)
    print(f"Loaded {len(rows)} projection rows")

    if target_layer is None:
        n_layers = len(rows[0]["projection_per_layer"])
        target_layer = n_layers // 2

    # Print cosine-similarity report up front (Phase 4 wrote it).
    sim_path = cfg.scratch_dir / "axis_cosine_similarity.txt"
    if sim_path.exists():
        print(f"\n--- Axis cosine similarity (from {sim_path.name}) ---")
        print(sim_path.read_text())

    axis_variants = sorted({r.get("axis_variant", "base") for r in rows})
    conditions = sorted({r["condition"] for r in rows})
    domains = sorted({r["domain"] for r in rows})
    print(f"  axis_variants: {axis_variants}")
    print(f"  conditions: {conditions}")
    print(f"  domains: {domains}")
    print(f"  target_layer: {target_layer}")

    plots_dir = cfg.scratch_dir / "plots"

    for axis_variant in axis_variants:
        agg = aggregate_per_turn(rows, axis_variant=axis_variant, target_layer=target_layer)
        # Local copy
        plot_trajectory(
            agg,
            domains=domains, conditions=conditions,
            axis_variant=axis_variant, target_layer=target_layer,
            output_path=plots_dir / f"drift_trajectory_{axis_variant}_layer{target_layer}.png",
        )
        # Paper figure copy
        plot_trajectory(
            agg,
            domains=domains, conditions=conditions,
            axis_variant=axis_variant, target_layer=target_layer,
            output_path=PAPER_FIGURES_DIR / "appendix"
                        / f"fig_assistant_axis_drift_trajectory_{axis_variant}.png",
        )
        # Per-cell heatmaps
        for condition in conditions:
            for domain in domains:
                plot_layer_heatmap(
                    rows,
                    axis_variant=axis_variant, condition=condition, domain=domain,
                    output_path=plots_dir
                                / f"drift_heatmap_{axis_variant}_{condition}_{domain}.png",
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["smoke", "balanced", "full"], default="smoke")
    parser.add_argument("--run-slug", help="Override run_slug")
    parser.add_argument("--target-layer", type=int, help="Override target layer index")
    args = parser.parse_args()

    cfg = get_preset(args.preset)
    if args.run_slug:
        cfg.run_slug = args.run_slug
    plot_drift(cfg, target_layer=args.target_layer)


if __name__ == "__main__":
    main()
