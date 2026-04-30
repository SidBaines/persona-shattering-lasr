#!/usr/bin/env python3
"""Phase 2 — Pick capping window + per-layer thresholds (paper Eq. 1).

Reads Phase 1 outputs (``axis.pt`` + ``activations/`` dir) and computes a
``capping_config.pt``. Defaults:

  * ``mode = "floor"`` (paper Eq. 1; lifts below-threshold projections up).
  * Threshold = 75th percentile of the JOINT default + role projection
    distribution at each layer. Under our axis convention
    (``axis = default − role``, positive = Assistant) this is the same
    physical threshold as the paper's "p25" calibration in the opposite
    sign convention. See ``src_dev/activation_capping/assistant_axis_loader.py``
    module docstring for the equivalence proof.
  * Window picked by max mean (signed) Cohen's d over a contiguous span in
    the upper half of layers, mirroring the paper.

Usage::

    uv run python -m scripts_dev.persona_drift_assistant_axis.pick_capping \\
        --preset smoke

    # Override the auto-pick and force a specific window:
    uv run python -m scripts_dev.persona_drift_assistant_axis.pick_capping \\
        --preset full --layer-window 22 30

    # Replicate upstream's published ceiling-clamp behaviour (only correct
    # if you also flip the axis sign):
    uv run python -m scripts_dev.persona_drift_assistant_axis.pick_capping \\
        --preset full --mode ceiling
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Project imports.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src_dev.activation_capping.assistant_axis_loader import compute_capping_config  # noqa: E402
from scripts_dev.persona_drift_assistant_axis.config import get_preset  # noqa: E402

SEED = 42


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    _seed_everything(SEED)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["smoke", "balanced", "full"], default="smoke")
    parser.add_argument("--run-slug", help="Override run_slug")
    parser.add_argument(
        "--threshold-percentile", type=float,
        help="Override threshold percentile (default: cfg.capping.threshold_percentile, "
             "which defaults to 75 in our axis convention).",
    )
    parser.add_argument("--layer-window", nargs=2, type=int, metavar=("LO", "HI"),
                        help="Force layer window (inclusive). Skips Cohen's d sweep.")
    parser.add_argument("--window-size", type=int, help="Window size when auto-picking")
    parser.add_argument(
        "--mode", choices=["floor", "ceiling"], default="floor",
        help="floor = paper Eq. 1 (default; correct for axis = default − role). "
             "ceiling = upstream's _apply_cap (only matches paper intent if you "
             "also flip the axis sign — see assistant_axis_loader.py module docstring).",
    )
    args = parser.parse_args()

    cfg = get_preset(args.preset)
    if args.run_slug:
        cfg.run_slug = args.run_slug
    if args.threshold_percentile is not None:
        cfg.capping.threshold_percentile = args.threshold_percentile
    if args.layer_window is not None:
        cfg.capping.layer_window = tuple(args.layer_window)

    # Capping uses the BASE axis only — we never apply capping to LoRA-modified
    # variants (they're a separate condition).
    axis_path = cfg.axis_path("base")
    activations_dir = cfg.axis_dir("base") / "activations"
    if not axis_path.exists():
        raise SystemExit(
            f"base axis.pt missing at {axis_path}; "
            f"run `build_axis.py --variant base` first."
        )
    if not activations_dir.exists():
        raise SystemExit(f"activations/ missing at {activations_dir}; run Phase 1 first.")

    output_path = cfg.capping_config_path
    print(f"Run slug: {cfg.run_slug}")
    print(f"Axis:     {axis_path}")
    print(f"Activations dir: {activations_dir}")
    print(f"Output:   {output_path}")
    print(f"Mode:     {args.mode}")
    print(f"Percentile (of joint default+role): {cfg.capping.threshold_percentile}")
    print(f"Layer window override: {cfg.capping.layer_window}")
    print(f"Seed: {SEED}")

    compute_capping_config(
        axis_path=axis_path,
        activations_dir=activations_dir,
        output_path=output_path,
        threshold_percentile=cfg.capping.threshold_percentile,
        layer_window=cfg.capping.layer_window,
        window_size=args.window_size,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
