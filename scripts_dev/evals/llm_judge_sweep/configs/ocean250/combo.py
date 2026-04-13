"""Combo (con-sup v2 @ +1 × ext-amp v3 @ +1) cell of the OCEAN-250 judge sweep.

Usage::

    CUDA_VISIBLE_DEVICES=3 uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.ocean250.combo \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.ocean250._shared import *  # noqa: F401,F403
from scripts_dev.evals.llm_judge_sweep.configs.ocean250._shared import (
    CON_SUP_V2,
    EXT_AMP_V3,
)

ADAPTERS = [CON_SUP_V2, EXT_AMP_V3]
SCALES_PER_ADAPTER = {
    CON_SUP_V2.slug: [1.0],
    EXT_AMP_V3.slug: [1.0],
}

PLOT_TITLE = "OCEAN-250 combo (con-sup v2 @ +1 × ext-amp v3 @ +1)"
