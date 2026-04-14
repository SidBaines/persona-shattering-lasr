"""Neuroticism amplifier v4 @ +1 cell of the OCEAN-250 judge sweep.

Usage::

    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.ocean250.neu_amp \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.ocean250._shared import *  # noqa: F401,F403
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec

NEU_AMP_V4 = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/neuroticism/amplifier/v4"
    "/lora/neuroticism_v3-persona"
)

ADAPTERS = [NEU_AMP_V4]
SCALES_PER_ADAPTER = {NEU_AMP_V4.slug: [1.0]}

PLOT_TITLE = "OCEAN-250 neu-amp v4 @ +1"
