#!/usr/bin/env python3
"""Plot rollout LoRA scale sweep results.

Reads ``run_info.json`` files produced by ``run_rollout_sweep()`` and plots
metric density vs LoRA scale, one line per condition.

Usage::

    python -m scripts.visualisations.plot_rollout_sweep \\
        --sweep-dir scratch/20260309_175444_o_avoiding

    # Custom metric key and output path:
    python -m scripts.visualisations.plot_rollout_sweep \\
        --sweep-dir scratch/20260309_175444_o_avoiding \\
        --metric overall/count_o.density/mean \\
        --output scratch/20260309_175444_o_avoiding/sweep_plot.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Colours matching the rest of the project's plotting style ─────────────────

_CONDITION_COLOURS = {
    "no_prompt":  "#4c72b0",
    "o_avoiding": "#dd8452",
    "o_enjoying": "#55a868",
    "t_avoiding": "#c44e52",
    "t_enjoying": "#8172b3",
}
_DEFAULT_COLOUR = "#888888"

_CONDITION_LABELS = {
    "no_prompt":  "No prompt",
    "o_avoiding": "O-avoiding prompt",
    "o_enjoying": "O-enjoying prompt",
    "t_avoiding": "T-avoiding prompt",
    "t_enjoying": "T-enjoying prompt",
}


# ── Data loading ───────────────────────────────────────────────────────────────


def load_sweep(sweep_dir: Path) -> dict[str, dict[float, dict]]:
    """Return ``{condition: {scale: run_info}}``, sorted by scale."""
    data: dict[str, dict[float, dict]] = {}
    for p in sorted(sweep_dir.glob("*/*/run_info.json")):
        try:
            info = json.loads(p.read_text())
        except Exception:
            continue
        if info.get("status") != "ok":
            continue
        scale = float(info["scale"])
        condition = info["condition"]
        data.setdefault(condition, {})[scale] = info
    return data


def _ci95(mean: float, n: int, std: float | None = None) -> float:
    """Return half-width of 95% CI.  Falls back to 0 if std unavailable."""
    if std is not None and n > 1:
        return 1.96 * std / math.sqrt(n)
    return 0.0


def _get_series(
    condition_data: dict[float, dict],
    metric_key: str,
    std_key: str | None,
) -> tuple[list[float], list[float], list[float]]:
    """Return (scales, means, half_ci_widths) sorted by scale."""
    scales, means, cis = [], [], []
    for scale in sorted(condition_data):
        agg = condition_data[scale].get("aggregates") or {}
        mean = agg.get(metric_key)
        if mean is None:
            continue
        n = int(agg.get(metric_key.replace("/mean", "/count"), 1))
        std = agg.get(std_key) if std_key else None
        scales.append(scale)
        means.append(mean)
        cis.append(_ci95(mean, n, std))
    return scales, means, cis


# ── Plotting ───────────────────────────────────────────────────────────────────


def plot_sweep(
    sweep_dir: Path,
    metric_key: str = "overall/count_o.density/mean",
    output: Path | None = None,
    title: str | None = None,
) -> Path:
    """Generate and save the sweep plot.

    Args:
        sweep_dir: Directory containing ``sweep_config.json`` and scale subdirs.
        metric_key: Aggregate key to plot on y-axis.
        output: Output PNG path.  Defaults to ``{sweep_dir}/sweep_plot.png``.
        title: Plot title.  Auto-generated from sweep_config.json if None.

    Returns:
        Path to the saved figure.
    """
    data = load_sweep(sweep_dir)
    if not data:
        raise ValueError(f"No completed run_info.json files found under {sweep_dir}")

    std_key = metric_key.replace("/mean", "/std") if "/mean" in metric_key else None

    # Load sweep config for title / metadata.
    cfg_path = sweep_dir / "sweep_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    if title is None:
        adapter = cfg.get("adapter", sweep_dir.name)
        metric_short = metric_key.split("/")[-2] if "/" in metric_key else metric_key
        title = f"LoRA scale sweep — {metric_short}\n{adapter}"

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for condition, condition_data in sorted(data.items()):
        scales, means, cis = _get_series(condition_data, metric_key, std_key)
        if not scales:
            continue

        colour = _CONDITION_COLOURS.get(condition, _DEFAULT_COLOUR)
        label = _CONDITION_LABELS.get(condition, condition)

        ax.plot(scales, means, "o-", color=colour, label=label, linewidth=2, markersize=6)
        if any(ci > 0 for ci in cis):
            ax.errorbar(scales, means, yerr=cis, fmt="none", color=colour, capsize=4, capthick=1.2, elinewidth=1.2, alpha=0.7)

    # Vertical line at scale=0 (base model).
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5, label="Base model (scale=0)")

    ax.set_xlabel("LoRA scale factor", fontsize=11)
    metric_label = metric_key.split("/")[-2] if "/" in metric_key else metric_key
    ax.set_ylabel(f"{metric_label} (%)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output is None:
        output = sweep_dir / "sweep_plot.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output}")
    return output


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot rollout LoRA scale sweep results.")
    parser.add_argument("--sweep-dir", required=True, type=Path,
                        help="Directory produced by run_rollout_sweep()")
    parser.add_argument("--metric", default="overall/count_o.density/mean",
                        help="Aggregate key to plot (default: overall/count_o.density/mean)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path (default: <sweep-dir>/sweep_plot.png)")
    parser.add_argument("--title", default=None,
                        help="Plot title override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_sweep(
        sweep_dir=args.sweep_dir,
        metric_key=args.metric,
        output=args.output,
        title=args.title,
    )


if __name__ == "__main__":
    main()
