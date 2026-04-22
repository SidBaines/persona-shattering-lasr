#!/usr/bin/env python3
"""Plot TRAIT sweep averaged across 5 OCEAN control-seed runs.

Loads the per-seed trait_logprobs sweep DataFrames from
``scratch/evals/ocean/trait/ocean_def_control_vanton4_seed{1..5}_logprobs_1000/``,
concatenates them (each seed becomes a distinct "run" at every scale point),
and invokes ``plot_trait_sweep`` so the mean is computed across seeds and
the error bars reflect cross-seed + sample variance.

Usage:
    uv run python -m scripts_dev.personality_evals.plot_control_seeds_avg_trait
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src_dev.evals.personality.analyze_results import (
    IntervalMethod,
    load_sweep_data,
    plot_trait_sweep,
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


def load_combined_df(run_dirs: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir missing: {run_dir}")
        data = load_sweep_data(run_dir)
        df = data.get(EVAL_NAME)
        if df is None or df.empty:
            raise RuntimeError(f"No '{EVAL_NAME}' data in {run_dir}")
        df = df.copy()
        df["run"] = f"{run_dir.name}::" + df["run"].astype(str)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--run-template",
        type=str,
        default=DEFAULT_RUN_TEMPLATE,
        help="Per-seed run dir template with '{seed}' placeholder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scratch/evals/ocean/trait/ocean_def_control_vanton4_seeds_avg_logprobs_1000/figures"),
    )
    parser.add_argument("--title-suffix", type=str,
                        default="OCEAN control vanton4 avg of 5 seeds TRAIT (logprobs)")
    parser.add_argument("--interval", type=str, default="ci95_from_bootstrap_1000")
    parser.add_argument("--min-choice-mass", type=float, default=0.75)
    args = parser.parse_args()

    run_dirs = [Path(args.run_template.format(seed=s)) for s in args.seeds]
    print(f"Loading {len(run_dirs)} per-seed run dirs:")
    for p in run_dirs:
        print(f"  - {p}")

    df = load_combined_df(run_dirs)
    print(f"Combined rows: {len(df)}   scales: {sorted(df['scale'].unique())}")
    print(f"  runs per scale: {df.groupby('scale')['run'].nunique().unique().tolist()}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = plot_trait_sweep(
        df,
        args.output_dir,
        title_suffix=args.title_suffix,
        interval=IntervalMethod.from_str(args.interval),
        min_choice_mass=args.min_choice_mass,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
