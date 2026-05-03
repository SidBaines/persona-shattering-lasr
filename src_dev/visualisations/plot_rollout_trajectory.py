#!/usr/bin/env python3
"""Plot per-turn trajectories of judge scores across rollout cells.

Reads ``rollouts_evaluated.jsonl`` from one or more cells (each cell is a
``variant/condition`` pair) and plots per-turn mean ± CI for one or more
judges.  Designed for the Part 2 (drift prevention) figure: one line per
condition (base, LoRA-, LoRA+, activation capping, CAA), x = turn index,
y = judge score.

Mirrors the visual style of
``paper/sections/supervised.tex#fig:frustration-per-turn`` (Gemma N+
frustration per-turn plot).

Paper figures:
    - Generated on demand; pass ``--paper-fig <name.pdf>`` to write into
      ``paper/figures/main/`` (or another subdir via --paper-subdir).

Usage::

    # Compare base vs E- vs E+ on the high-pressure scenarios:
    python -m src_dev.visualisations.plot_rollout_trajectory \\
        --cells \\
            "Base=scratch/.../variant_base/scenarios_extraversion_high/evals" \\
            "E- LoRA=scratch/.../variant_e_minus/scenarios_extraversion_high/evals" \\
            "E+ LoRA=scratch/.../variant_e_plus/scenarios_extraversion_high/evals" \\
        --judges extraversion_v2 coherence_v2 \\
        --output scratch/trajectory.png \\
        --paper-fig fig_3_5_extraversion_drift.pdf
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src_dev.visualisations import PAPER_FIGURES_DIR

# ── Style ────────────────────────────────────────────────────────────────────

_BASELINE_KEYWORDS = ("base", "baseline", "neutral", "control", "no intervention")

# Same palette as plot_rollout_sweep.py for visual consistency across paper.
_COLOURS = [
    "#000000",  # baseline gets black
    "#e6194b",  # red — typically suppressor / "fix"
    "#3cb44b",  # green — typically amplifier / "amplify"
    "#4363d8",  # blue — alternative method (capping / CAA)
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
]
_LINESTYLES = ["--", "-.", ":"]  # solid reserved for baseline
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "p", "h"]

# Human-readable y-axis labels for known judges.
_JUDGE_LABELS: dict[str, tuple[str, tuple[float, float] | None]] = {
    "extraversion_v2": ("Extraversion score", (-4, 4)),
    "agreeableness_v2": ("Agreeableness score", (-4, 4)),
    "conscientiousness_v2": ("Conscientiousness score", (-4, 4)),
    "openness_v2": ("Openness score", (-4, 4)),
    "neuroticism_v2": ("Neuroticism score", (-4, 4)),
    "coherence_v2": ("Coherence score", (0, 10)),
}


def _is_baseline(label: str) -> bool:
    low = label.lower()
    return any(kw in low for kw in _BASELINE_KEYWORDS)


def _style_for(label: str, index: int) -> tuple[str, str, str]:
    """Return (colour, linestyle, marker) for a given condition label.

    The first baseline gets solid black + circle. Subsequent baselines
    get distinct grey shades + linestyles so multiple "no-intervention"
    lines stay visually separable. Non-baselines cycle through colours,
    dashes, and markers.
    """
    # Different shades + linestyles for additional baselines so multiple
    # "base / no-intervention" lines don't overlap visually. Order:
    # solid black, dashed dim grey, dotted darker grey.
    _BASELINE_STYLES = [
        ("#000000", "-"),
        ("#555555", "--"),
        ("#888888", ":"),
    ]
    if _is_baseline(label):
        # Count baselines among prior labels (best-effort via the index;
        # caller controls the order).
        b_idx = index % len(_BASELINE_STYLES)
        colour, linestyle = _BASELINE_STYLES[b_idx]
        marker = _MARKERS[index % len(_MARKERS)]
        return colour, linestyle, marker
    # Skip black for non-baseline since baseline owns it.
    colour = _COLOURS[(index + 1) % len(_COLOURS)]
    linestyle = _LINESTYLES[index % len(_LINESTYLES)]
    marker = _MARKERS[index % len(_MARKERS)]
    return colour, linestyle, marker


# ── Data loading + aggregation ───────────────────────────────────────────────


def parse_cell_arg(arg: str) -> tuple[str, list[Path]]:
    """Parse 'label=path[,path...]' into (label, [Path, ...]).

    Each path may point at the ``evals/`` directory or at the cell
    directory that contains it. Multiple comma-separated paths under
    one label are concatenated at load time, useful when the same
    experimental cell was split into multiple sub-runs (e.g. a LoRA
    scale sweep run on two scenario subsets that should be merged).
    """
    if "=" not in arg:
        raise ValueError(
            f"Cell arg must be 'label=path[,path...]', got {arg!r}. "
            "(Plain paths are not allowed here — labels appear in the legend "
            "and need to be explicit for paper figures.)"
        )
    label, path_str = arg.split("=", 1)
    label = label.strip()
    paths: list[Path] = []
    for p_str in path_str.split(","):
        p = Path(p_str.strip())
        if not p_str.strip():
            continue
        if p.is_dir() and p.name != "evals" and (p / "evals").exists():
            p = p / "evals"
        paths.append(p)
    if not paths:
        raise ValueError(f"Cell arg {arg!r} resolved to zero paths.")
    return label, paths


def load_evaluated(evals_dir: Path) -> list[dict[str, Any]]:
    """Load rollouts_evaluated.jsonl into a list of entries."""
    p = evals_dir / "rollouts_evaluated.jsonl"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}. Has the sweep finished evaluating?"
        )
    out = []
    for line in p.read_text().strip().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def collect_per_turn_scores(
    entries: list[dict[str, Any]],
    judge: str,
    role: str = "assistant",
) -> dict[int, list[float]]:
    """Return {turn_index: [scores across all (scenario, rollout) pairs]}."""
    by_turn: dict[int, list[float]] = defaultdict(list)
    for entry in entries:
        for _rollout_idx, msgs in entry.get("messages", {}).items():
            for msg in msgs:
                if msg.get("role") != role:
                    continue
                turn_idx = msg.get("turn_index")
                if turn_idx is None:
                    continue
                scores = msg.get("scores", {}) or {}
                if judge not in scores:
                    continue
                score_obj = scores[judge]
                raw = (
                    score_obj.get("score")
                    if isinstance(score_obj, dict)
                    else score_obj
                )
                if raw is None:
                    continue
                try:
                    by_turn[int(turn_idx)].append(float(raw))
                except (TypeError, ValueError):
                    continue
    return by_turn


def aggregate(
    by_turn: dict[int, list[float]],
    *,
    bootstrap_n: int = 0,
    seed: int = 12345,
) -> dict[int, dict[str, float]]:
    """Per-turn (mean, ci_low, ci_high, n).

    If ``bootstrap_n > 0`` use a non-parametric bootstrap; otherwise use
    a Gaussian 95% CI from the SEM (fast, OK for sanity but not for
    publication when n is small or the distribution is bounded).
    """
    out: dict[int, dict[str, float]] = {}
    if bootstrap_n > 0:
        import random

        rng = random.Random(seed)

    for turn, values in sorted(by_turn.items()):
        n = len(values)
        if n == 0:
            continue
        mean = sum(values) / n
        if bootstrap_n > 0 and n > 1:
            boots = []
            for _ in range(bootstrap_n):
                resampled = [
                    values[rng.randrange(n)] for _ in range(n)
                ]
                boots.append(sum(resampled) / n)
            boots.sort()
            ci_lo = boots[int(0.025 * bootstrap_n)]
            ci_hi = boots[int(0.975 * bootstrap_n)]
        elif n > 1:
            var = sum((v - mean) ** 2 for v in values) / (n - 1)
            sem = math.sqrt(var / n)
            ci_lo = mean - 1.96 * sem
            ci_hi = mean + 1.96 * sem
        else:
            ci_lo = ci_hi = mean
        out[turn] = {"mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": n}
    return out


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_trajectory(
    cells: list[tuple[str, list[dict[str, Any]]]],
    judges: list[str],
    output: Path,
    *,
    bootstrap_n: int = 1000,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    show_n: bool = False,
) -> Path:
    """Plot per-turn trajectories for one or more judges.

    Args:
        cells: list of (label, entries) where entries is the loaded JSONL.
        judges: judge names to plot (one subplot per judge, stacked).
        output: where to save the figure.
        bootstrap_n: bootstrap iterations for CI; 0 = Gaussian SEM.
        title: optional figure-level suptitle.
        figsize: override figure size; default adapts to subplot count.
        show_n: annotate the n on each subplot's first line.

    Returns:
        Path to the saved figure.
    """
    n_judges = len(judges)
    if figsize is None:
        figsize = (8.0, 3.5 * n_judges)
    fig, axes = plt.subplots(n_judges, 1, figsize=figsize, sharex=True)
    if n_judges == 1:
        axes = [axes]

    # Pre-compute aggregates for all (cell, judge) pairs.
    cell_aggregates: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    for label, entries in cells:
        cell_aggregates[label] = {}
        for judge in judges:
            by_turn = collect_per_turn_scores(entries, judge)
            cell_aggregates[label][judge] = aggregate(by_turn, bootstrap_n=bootstrap_n)

    for ax, judge in zip(axes, judges):
        ylabel, ylim = _JUDGE_LABELS.get(
            judge, (judge.replace("_", " ").title(), None)
        )

        for index, (label, _) in enumerate(cells):
            agg = cell_aggregates[label][judge]
            if not agg:
                continue
            turns = sorted(agg.keys())
            means = [agg[t]["mean"] for t in turns]
            lo = [agg[t]["ci_lo"] for t in turns]
            hi = [agg[t]["ci_hi"] for t in turns]

            colour, linestyle, marker = _style_for(label, index)
            # Asymmetric error bars: yerr = [mean-lo, hi-mean] for each point.
            yerr_lo = [m - l for m, l in zip(means, lo)]
            yerr_hi = [h - m for m, h in zip(means, hi)]
            ax.errorbar(
                turns, means,
                yerr=[yerr_lo, yerr_hi],
                color=colour, linestyle=linestyle, marker=marker,
                linewidth=2, markersize=6, label=label,
                capsize=4, capthick=1.2, elinewidth=1.2,
            )

            if show_n and index == 0 and turns:
                ax.text(
                    0.02, 0.95,
                    f"n per turn: {agg[turns[0]]['n']}",
                    transform=ax.transAxes,
                    fontsize=8, va="top", color="grey",
                )

        # Reference lines: 0 for trait scores (neutral), nothing for coherence.
        if ylim is not None and ylim[0] < 0 < ylim[1]:
            ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
        # Pin the y-axis to the judge's full scale (e.g. -4..+4 for trait
        # judges) so plots are comparable across runs regardless of where
        # the data happens to land.
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    axes[-1].set_xlabel("Turn index", fontsize=10)
    if title:
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output}")
    return output


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-turn judge-score trajectories across cells.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cells", nargs="+", required=True,
        help=(
            "One or more 'label=path[,path...]' entries pointing at evals/ "
            "directories. The label appears in the legend (use 'Base', "
            "'E- LoRA' etc., not raw cell-dir names). Multiple "
            "comma-separated paths under one label are merged at load time, "
            "useful when one experimental cell was split into multiple "
            "sub-runs (e.g. a LoRA scale sweep on two scenario subsets)."
        ),
    )
    parser.add_argument(
        "--judges", nargs="+", default=["extraversion_v2", "coherence_v2"],
        help="Judges to plot (one subplot per judge).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("scratch/trajectory.png"),
        help="Output figure path (default: scratch/trajectory.png).",
    )
    parser.add_argument(
        "--paper-fig", default=None,
        help=(
            "If set, ALSO save a copy under paper/figures/<paper-subdir>/ "
            "with this filename. Use for figures going into the paper."
        ),
    )
    parser.add_argument(
        "--paper-subdir", default="main",
        help="Subdirectory under paper/figures/ for --paper-fig (default: main).",
    )
    parser.add_argument(
        "--title", default=None,
        help="Figure-level title (default: none).",
    )
    parser.add_argument(
        "--bootstrap-n", type=int, default=1000,
        help="Bootstrap iterations for CI; 0 disables (use Gaussian SEM).",
    )
    parser.add_argument(
        "--show-n", action="store_true",
        help="Annotate first-turn n on each subplot.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cells: list[tuple[str, list[dict[str, Any]]]] = []
    for arg in args.cells:
        label, paths = parse_cell_arg(arg)
        merged_entries: list[dict[str, Any]] = []
        for p in paths:
            if not (p / "rollouts_evaluated.jsonl").exists():
                print(
                    f"ERROR: {p / 'rollouts_evaluated.jsonl'} not found",
                    file=sys.stderr,
                )
                return 1
            merged_entries.extend(load_evaluated(p))
        if len(paths) > 1:
            print(
                f"  Merged {len(paths)} paths into '{label}' "
                f"({len(merged_entries)} total entries)"
            )
        cells.append((label, merged_entries))

    print(f"Loaded {len(cells)} cell(s):")
    for label, entries in cells:
        n_msgs = sum(
            len(msgs)
            for entry in entries
            for msgs in entry.get("messages", {}).values()
        )
        print(f"  - {label}: {len(entries)} samples, {n_msgs} messages")

    plot_trajectory(
        cells, args.judges, args.output,
        bootstrap_n=args.bootstrap_n,
        title=args.title,
        show_n=args.show_n,
    )

    if args.paper_fig:
        paper_path = PAPER_FIGURES_DIR / args.paper_subdir / args.paper_fig
        plot_trajectory(
            cells, args.judges, paper_path,
            bootstrap_n=args.bootstrap_n,
            title=args.title,
            show_n=args.show_n,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
