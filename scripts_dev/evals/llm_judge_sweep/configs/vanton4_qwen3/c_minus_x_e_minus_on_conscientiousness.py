"""c_minus × e_minus (vanton4, 1:1 soup) evaluated on conscientiousness prompts.

Rollouts on ``data/ocean_open_ended/conscientiousness.jsonl``, judged on
conscientiousness_v2 + coherence. Uses MAX_SAMPLES=100, NUM_ROLLOUTS_PER_PROMPT=1
— smaller than the 240 × 3 used by Run 1/Option B, which makes this sweep
~7x faster and gives it a fresh rollout fingerprint. The new fingerprint
means no collision with any other running sweep, and no cache hit with
c_minus Run 1 (intentional — this is an independent combo experiment).

Usage::

    CUDA_VISIBLE_DEVICES=3 uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_qwen3.c_minus_x_e_minus_on_conscientiousness \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_qwen3._shared import *  # noqa: F401,F403
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/conscientiousness.jsonl"

# Smaller / faster — and changes rollout fingerprint so no collision anywhere.
MAX_SAMPLES = 100
NUM_ROLLOUTS_PER_PROMPT = 1

EVAL_NAME = "c_minus_x_e_minus-vanton4-on-conscientiousness"
TRAIT = OceanTrait.conscientiousness

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

JUDGE_METRIC_TRAITS = [OceanTrait.conscientiousness.v2_metric_name]
# COHERENCE_METRIC inherited from _shared.py.

TRAIT_COLOR = BIG_FIVE_COLORS["Conscientiousness"]
PLOT_TITLE = "c_minus × e_minus (vanton4) on conscientiousness prompts — Qwen3-235B judge"
