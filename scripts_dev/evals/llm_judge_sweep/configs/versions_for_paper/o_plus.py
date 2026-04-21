"""Openness amplifier (vanton4) LLM judge scale sweep.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.versions_for_paper.o_plus
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.versions_for_paper._shared import *  # noqa: F401,F403
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/openness.jsonl"

EVAL_NAME = "openness-amplifier-vanton4"
TRAIT = OceanTrait.openness

ADAPTER = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/openness/amplifier/vanton4"
    "/lora/openness_amplifying_full_vanton4-persona"
)
ADAPTERS = [ADAPTER]
SCALES_PER_ADAPTER = {ADAPTER.slug: SCALE_POINTS}

JUDGE_METRIC_TRAITS = [OceanTrait.openness.v2_metric_name]
TRAIT_COLOR = BIG_FIVE_COLORS["Openness"]
PLOT_TITLE = "Openness amplifier (vanton4) LoRA scale sweep"
