"""c_minus x e_minus (vanton4) at the single (c_minus=+1, e_minus=+1) cell,
rollouts on openness prompts.

Tiny companion to the existing 5x5 soup configs — fills in one missing
cell on the openness prompt set so we have (c_minus=+1, e_minus=+1)
scored on all five OCEAN traits (C and E already come from the full
5x5 soups on their own prompt sets; O, A, N need these one-cell runs).

Same rollout params as the 5x5 soups (MAX_SAMPLES=100,
NUM_ROLLOUTS_PER_PROMPT=1, openness.jsonl), just with a single-scale
SCALES_PER_ADAPTER dict that enumerates to exactly one combo cell.

Usage::

    CUDA_VISIBLE_DEVICES=3 uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_qwen3.c_minus_x_e_minus_on_openness_1x1 \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_qwen3._shared import *  # noqa: F401,F403
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/openness.jsonl"

# Match the 5x5 soup configs' smaller scale (changes the rollout fingerprint).
MAX_SAMPLES = 100
NUM_ROLLOUTS_PER_PROMPT = 1

EVAL_NAME = "c_minus_x_e_minus-vanton4-on-openness-1x1"
TRAIT = OceanTrait.openness

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
# Only one scale value per adapter → exactly one (+1, +1) combo cell,
# no baseline, no single-adapter cells.
SCALES_PER_ADAPTER = {
    ADAPTER_C_MINUS.slug: [1.0],
    ADAPTER_E_MINUS.slug: [1.0],
}

JUDGE_METRIC_TRAITS = [OceanTrait.openness.v2_metric_name]

TRAIT_COLOR = BIG_FIVE_COLORS["Openness"]
PLOT_TITLE = "c_minus x e_minus (vanton4) @ (+1, +1) on openness prompts"
