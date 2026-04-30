#!/usr/bin/env python3
"""Phase 2 — Pick capping window + per-layer thresholds.

Reads Phase 1 outputs (``axis.pt`` + ``activations/`` dir) and computes a
``capping_config.pt`` matching upstream convention. The window is picked
by max mean Cohen's d (default-vs-role projections) over a contiguous
span in the upper half of layers, mirroring the paper's choices.

Usage::

    uv run python -m scripts_dev.persona_drift_assistant_axis.pick_capping \\
        --preset smoke

    # Override the auto-pick and force a specific window:
    uv run python -m scripts_dev.persona_drift_assistant_axis.pick_capping \\
        --preset full --layer-window 22 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project imports.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src_dev.activation_capping.assistant_axis_loader import compute_capping_config  # noqa: E402
from scripts_dev.persona_drift_assistant_axis.config import get_preset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["smoke", "balanced", "full"], default="smoke")
    parser.add_argument("--run-slug", help="Override run_slug")
    parser.add_argument("--threshold-percentile", type=float, help="Override threshold percentile (default 25)")
    parser.add_argument("--layer-window", nargs=2, type=int, metavar=("LO", "HI"),
                        help="Force layer window (inclusive). Skips Cohen's d sweep.")
    parser.add_argument("--window-size", type=int, help="Window size when auto-picking")
    args = parser.parse_args()

    cfg = get_preset(args.preset)
    if args.run_slug:
        cfg.run_slug = args.run_slug
    if args.threshold_percentile is not None:
        cfg.capping.threshold_percentile = args.threshold_percentile
    if args.layer_window is not None:
        cfg.capping.layer_window = tuple(args.layer_window)

    out_root = cfg.scratch_dir
    axis_path = out_root / "axis.pt"
    activations_dir = out_root / "activations"
    if not axis_path.exists():
        raise SystemExit(f"axis.pt missing at {axis_path}; run Phase 1 first.")
    if not activations_dir.exists():
        raise SystemExit(f"activations/ missing at {activations_dir}; run Phase 1 first.")

    output_path = out_root / "capping_config.pt"
    print(f"Run slug: {cfg.run_slug}")
    print(f"Axis:     {axis_path}")
    print(f"Activations dir: {activations_dir}")
    print(f"Output:   {output_path}")
    print(f"Percentile: {cfg.capping.threshold_percentile}")
    print(f"Layer window override: {cfg.capping.layer_window}")

    compute_capping_config(
        axis_path=axis_path,
        activations_dir=activations_dir,
        output_path=output_path,
        threshold_percentile=cfg.capping.threshold_percentile,
        layer_window=cfg.capping.layer_window,
        window_size=args.window_size,
    )


if __name__ == "__main__":
    main()
