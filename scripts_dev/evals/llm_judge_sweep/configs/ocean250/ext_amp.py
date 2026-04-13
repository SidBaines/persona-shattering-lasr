"""Extraversion amplifier v3 @ +1 cell of the OCEAN-250 judge sweep.

Usage::

    CUDA_VISIBLE_DEVICES=2 uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.ocean250.ext_amp \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.ocean250._shared import *  # noqa: F401,F403
from scripts_dev.evals.llm_judge_sweep.configs.ocean250._shared import EXT_AMP_V3

ADAPTERS = [EXT_AMP_V3]
SCALES_PER_ADAPTER = {EXT_AMP_V3.slug: [1.0]}

PLOT_TITLE = "OCEAN-250 ext-amp v3 @ +1"
