"""Gemma-3 4b conscientiousness suppressor v2 LoRA scale sweep (Qwen3-235B judge).

Consc+coherence only, full scale sweep [-2, -1, 0, 1, 2].

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
        --config scripts_dev.evals.llm_judge_sweep.configs.gemma_consc_sup.g4b \
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.gemma_consc_sup._shared import *  # noqa: F401,F403
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

BASE_MODEL = "google/gemma-3-4b-it"
BASE_MODEL_SLUG = "gemma-3-4b-it"

EVAL_NAME = "gemma-3-4b-conscientiousness-suppressor-v2"
TRAIT = OceanTrait.conscientiousness

ADAPTER = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/gemma-3-4b-it/ocean/conscientiousness/suppressor/v2"
    "/lora/conscientiousness_low_v2-persona"
)
ADAPTERS = [ADAPTER]
SCALES_PER_ADAPTER = {ADAPTER.slug: SCALE_POINTS}

JUDGE_METRIC_TRAITS = [OceanTrait.conscientiousness.v2_metric_name]
TRAIT_COLOR = BIG_FIVE_COLORS["Conscientiousness"]
PLOT_TITLE = "Gemma-3 4b conscientiousness suppressor v2 (Qwen3-235B judge) LoRA scale sweep"
