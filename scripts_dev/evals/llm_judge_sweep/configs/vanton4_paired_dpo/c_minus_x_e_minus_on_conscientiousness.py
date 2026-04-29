"""c_minus × e_minus (vanton4_paired_dpo, 1:1 soup) on conscientiousness prompts.

Catalogue-sourced version of
``scripts_dev.evals.llm_judge_sweep.configs.vanton4_qwen3.c_minus_x_e_minus_on_conscientiousness``
— adapter refs come from ``OCEAN_REGISTRY`` (current canonical
vanton4_paired_dpo adapters) instead of being hardcoded to the older
vanton4 paths. Rollout params (MAX_SAMPLES=100, NUM_ROLLOUTS_PER_PROMPT=1)
are kept identical so the rollout fingerprint stays at ``97743334f6``,
which the paper heatmap script reads.

Usage::

    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo.c_minus_x_e_minus_on_conscientiousness \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo._shared import *  # noqa: F401,F403
from src_dev.common.lora_catalogue import OCEAN_REGISTRY
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/conscientiousness.jsonl"

# Mirror the vanton4_qwen3 c_minus_x_e_minus_on_conscientiousness sizing so
# the rollout fingerprint matches what the paper heatmap script expects.
MAX_SAMPLES = 100
NUM_ROLLOUTS_PER_PROMPT = 1

EVAL_NAME = "c_minus_x_e_minus-vanton4-paired-dpo-on-conscientiousness"
TRAIT = OceanTrait.conscientiousness

ADAPTER_C_MINUS = AdapterSpec.from_ref(OCEAN_REGISTRY["c_minus"].adapter_ref)
ADAPTER_E_MINUS = AdapterSpec.from_ref(OCEAN_REGISTRY["e_minus"].adapter_ref)
ADAPTERS = [ADAPTER_C_MINUS, ADAPTER_E_MINUS]
SCALES_PER_ADAPTER = {
    ADAPTER_C_MINUS.slug: SCALE_POINTS,
    ADAPTER_E_MINUS.slug: SCALE_POINTS,
}

JUDGE_METRIC_TRAITS = [OceanTrait.conscientiousness.v2_metric_name]
# COHERENCE_METRIC inherited from _shared.py.

TRAIT_COLOR = BIG_FIVE_COLORS["Conscientiousness"]
PLOT_TITLE = "c_minus × e_minus (vanton4_paired_dpo) on conscientiousness prompts — Qwen3-235B judge"
