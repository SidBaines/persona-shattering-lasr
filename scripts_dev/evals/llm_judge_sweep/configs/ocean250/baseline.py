"""Baseline cell (no adapters) of the OCEAN-250 judge sweep.

Usage::

    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.ocean250.baseline \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.ocean250._shared import *  # noqa: F401,F403

# Enumerate 0 cells via ADAPTERS — the runner special-cases empty to baseline.
ADAPTERS: list = []
SCALES_PER_ADAPTER: dict = {}

PLOT_TITLE = "OCEAN-250 baseline"
