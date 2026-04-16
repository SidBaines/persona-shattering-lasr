"""Conscientiousness suppressor (vanton4_rank8) LLM judge scale sweep.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_rank8.c_minus
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_rank8._shared import *  # noqa: F401,F403
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

EVAL_NAME = "conscientiousness-suppressor-vanton4_rank8"
TRAIT = OceanTrait.conscientiousness

ADAPTER = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/vanton4_rank8"
    "/lora/conscientiousness_suppressing_full_vanton4-persona"
)
ADAPTERS = [ADAPTER]
SCALES_PER_ADAPTER = {ADAPTER.slug: SCALE_POINTS}

JUDGE_METRIC_TRAITS = [OceanTrait.conscientiousness.v2_metric_name]
TRAIT_COLOR = BIG_FIVE_COLORS["Conscientiousness"]
PLOT_TITLE = "Conscientiousness suppressor (vanton4_rank8) LoRA scale sweep"
