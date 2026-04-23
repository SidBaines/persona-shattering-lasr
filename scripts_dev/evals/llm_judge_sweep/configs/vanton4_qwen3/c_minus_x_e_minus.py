"""Conscientiousness-suppressor × Extraversion-suppressor 1:1 soup (vanton4).

2-adapter Cartesian grid sweep (5x5 = 25 cells). Both adapters run at each
scale in [-2, -1, 0, 1, 2]. Rollouts happen on the 50/50 C+E benchmark
(``data/trait_benchmark_con50_ext50.jsonl``) so both trait judges have
representative prompts to score against. Judges: conscientiousness_v2,
extraversion_v2, plus coherence. Uses Qwen3-235B as the sole rater (matches
the vanton4_qwen3 own-trait runs for apples-to-apples comparison).

Canonical cell identity: pure single-adapter cells (scale=0 on one axis)
collapse to the adapter's own ``fine_tuning/.../scale_X/`` HF path under
this sweep's rollout fingerprint. Since the fingerprint depends on
DATASET_PATH, this run won't cache-hit against existing single-adapter
runs (which used each adapter's own-trait prompt set).

Usage::

    CUDA_VISIBLE_DEVICES=3 uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_qwen3.c_minus_x_e_minus \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_qwen3._shared import *  # noqa: F401,F403
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/trait_benchmark_con50_ext50.jsonl"

EVAL_NAME = "c_minus_x_e_minus-vanton4"
TRAIT = OceanTrait.conscientiousness  # primary axis for labelling

ADAPTER_C_MINUS = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/vanton4"
    "/lora/conscientiousness_suppressing_full_vanton4-persona"
)
ADAPTER_E_MINUS = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/extraversion/suppressor/vanton4"
    "/lora/extraversion_suppressing_full_vanton4-persona"
)
ADAPTERS = [ADAPTER_C_MINUS, ADAPTER_E_MINUS]
SCALES_PER_ADAPTER = {
    ADAPTER_C_MINUS.slug: SCALE_POINTS,
    ADAPTER_E_MINUS.slug: SCALE_POINTS,
}

JUDGE_METRIC_TRAITS = [
    OceanTrait.conscientiousness.v2_metric_name,
    OceanTrait.extraversion.v2_metric_name,
]
# COHERENCE_METRIC inherited from _shared.py.

TRAIT_COLOR = BIG_FIVE_COLORS["Conscientiousness"]
PLOT_TITLE = "c_minus × e_minus (vanton4, 1:1 soup, Qwen3-235B judge) LoRA scale grid"
