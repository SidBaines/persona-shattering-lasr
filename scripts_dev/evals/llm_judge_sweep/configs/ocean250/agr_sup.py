"""Agreeableness suppressor v2 @ +1 cell of the OCEAN-250 judge sweep.

Usage::

    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.ocean250.agr_sup \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.ocean250._shared import *  # noqa: F401,F403
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec

AGR_SUP_V2 = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/v2"
    "/lora/agreeableness_low-persona"
)

ADAPTERS = [AGR_SUP_V2]
SCALES_PER_ADAPTER = {AGR_SUP_V2.slug: [1.0]}

PLOT_TITLE = "OCEAN-250 agr-sup v2 @ +1"
