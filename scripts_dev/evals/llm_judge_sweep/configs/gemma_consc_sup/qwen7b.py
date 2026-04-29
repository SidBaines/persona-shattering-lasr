"""Qwen2.5-7B-Instruct conscientiousness suppressor v2 LoRA scale sweep (Qwen3-235B judge).

Consc+coherence only, full scale sweep [-2, -1, 0, 1, 2]. Same OCT v2 recipe as
the Llama-3.1-8B and Gemma-3 4B/12B/27B sweeps; only the base model differs.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
        --config scripts_dev.evals.llm_judge_sweep.configs.gemma_consc_sup.qwen7b \
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.gemma_consc_sup._shared import *  # noqa: F401,F403
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_MODEL_SLUG = "qwen-2.5-7b-it"

EVAL_NAME = "qwen-2.5-7b-conscientiousness-suppressor-v2"
TRAIT = OceanTrait.conscientiousness

ADAPTER = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/qwen-2.5-7b-it/ocean/conscientiousness/suppressor/v2"
    "/lora/conscientiousness_low_v2-persona"
)
ADAPTERS = [ADAPTER]
SCALES_PER_ADAPTER = {ADAPTER.slug: SCALE_POINTS}

JUDGE_METRIC_TRAITS = [OceanTrait.conscientiousness.v2_metric_name]
TRAIT_COLOR = BIG_FIVE_COLORS["Conscientiousness"]
PLOT_TITLE = "Qwen2.5-7B-Instruct conscientiousness suppressor v2 (Qwen3-235B judge) LoRA scale sweep"
