"""Conscientiousness amplifier v1 (souped) @ +1 cell of the OCEAN-250 judge sweep.

Usage::

    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.ocean250.con_amp \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.ocean250._shared import *  # noqa: F401,F403
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec

CON_AMP_V1 = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/amplifier/v1"
    "/lora/souped"
)

ADAPTERS = [CON_AMP_V1]
SCALES_PER_ADAPTER = {CON_AMP_V1.slug: [1.0]}

PLOT_TITLE = "OCEAN-250 con-amp v1 souped @ +1"
