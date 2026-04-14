"""Openness amplifier (vanton1) @ +1 cell of the OCEAN-250 judge sweep.

Usage::

    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.ocean250.opn_amp \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.ocean250._shared import *  # noqa: F401,F403
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec

OPN_AMP_VANTON1 = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/openness/amplifier/vanton1"
    "/lora/openness_amplifying_full_vanton1-persona"
)

ADAPTERS = [OPN_AMP_VANTON1]
SCALES_PER_ADAPTER = {OPN_AMP_VANTON1.slug: [1.0]}

PLOT_TITLE = "OCEAN-250 opn-amp vanton1 @ +1"
