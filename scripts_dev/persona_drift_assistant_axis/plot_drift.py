#!/usr/bin/env python3
"""Phase 5 — Drift-trajectory plot (paper Figure 7 style).

Reads ``drift_projections.jsonl`` from Phase 4 and plots per-turn-position
mean projection ± 95% bootstrap CI for each condition, faceted by domain.
A separate full-stack heatmap shows projection-per-layer-per-turn,
useful for verifying the capping window choice.

Outputs (under ``{scratch_dir}/plots/`` and copies to
``paper/figures/appendix/``):

    drift_trajectory_target_layer.png  — paper Fig 7 analog at target_layer
    drift_trajectory_all_layers.png    — facet grid: per-domain × per-condition
    drift_layer_heatmap_{condition}.png — projection-per-layer-per-turn

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
    "appendix/fig_assistant_axis_drift_trajectory.png",
    "appendix/fig_assistant_axis_drift_layer_heatmap.png",
]


# ── Loading + aggregation ────────────────────────────────────────────────


def load_projections(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def aggregate_per_turn(
    rows: list[dict], *, target_layer: int
) -> dict[tuple[str, str, int], dict]:
    """Aggregate to (condition, domain, turn_position) → {mean, ci_lo, ci_hi, n}."""
    buckets: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for r in rows:
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


def plot_trajectory(
    agg: dict, *, domains: list[str], conditions: list[str], target_layer: int, output_path: Path,
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
                xs.append(tp + 1)  # 1-indexed for human readability
                ys.append(row["mean"])
                lo.append(row["ci_lo"])
                hi.append(row["ci_hi"])
            if xs:
                color = _CONDITION_COLORS.get(condition, "black")
                ax.plot(xs, ys, marker="o", color=color, label=_CONDITION_LABEL.get(condition, condition))
                ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
        ax.set_title(domain)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Projection onto Assistant Axis")
        ax.axhline(0, color="black", lw=0.5, alpha=0.3)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Persona drift trajectory (layer {target_layer})", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


def plot_layer_heatmap(
    rows: list[dict],
    *,
    condition: str,
    domain: str,
    output_path: Path,
) -> None:
    """Heatmap: turn_position × layer of mean projection for one (condition, domain)."""
    cell: dict[tuple[int, int], list[float]] = defaultdict(list)
    n_layers = 0
    for r in rows:
        if r["condition"] != condition or r["domain"] != domain:
            continue
        proj = r["projection_per_layer"]
        n_layers = len(proj)
        for layer, p in enumerate(proj):
            cell[(r["turn_position"], layer)].append(p)
    if not cell:
        print(f"  skipping heatmap {condition}/{domain}: no data")
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
    ax.set_title(f"Projection per (turn, layer) — {condition} / {domain}")
    fig.colorbar(im, ax=ax, label="Projection onto Assistant Axis")
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

    # Default target layer = upstream's auto-config (n_layers // 2).
    if target_layer is None:
        n_layers = len(rows[0]["projection_per_layer"])
        target_layer = n_layers // 2

    conditions = sorted({r["condition"] for r in rows})
    domains = sorted({r["domain"] for r in rows})
    print(f"  conditions: {conditions}")
    print(f"  domains: {domains}")
    print(f"  target_layer: {target_layer}")

    agg = aggregate_per_turn(rows, target_layer=target_layer)

    plots_dir = cfg.scratch_dir / "plots"
    plot_trajectory(
        agg,
        domains=domains, conditions=conditions, target_layer=target_layer,
        output_path=plots_dir / f"drift_trajectory_layer{target_layer}.png",
    )
    # Also write to the paper figure tree.
    plot_trajectory(
        agg,
        domains=domains, conditions=conditions, target_layer=target_layer,
        output_path=PAPER_FIGURES_DIR / "appendix" / "fig_assistant_axis_drift_trajectory.png",
    )

    # Per-condition × per-domain heatmaps (combined into one figure if few cells).
    for condition in conditions:
        for domain in domains:
            plot_layer_heatmap(
                rows,
                condition=condition, domain=domain,
                output_path=plots_dir / f"drift_heatmap_{condition}_{domain}.png",
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
