"""n_minus (gemma3 teacher) adapter evaluated on Openness prompts.

Overrides DATASET_PATH + JUDGE_METRIC_TRAITS so this sweep generates new
rollouts on the openness prompt set and judges them for openness.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_gemma3_paired_dpo.n_minus_on_openness \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_gemma3_paired_dpo.n_minus import *  # noqa: F401,F403

from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/openness.jsonl"
TRAIT = OceanTrait.openness
JUDGE_METRIC_TRAITS = [OceanTrait.openness.v2_metric_name]

EVAL_NAME = "neuroticism-suppressor-vanton4-gemma3-paired-dpo-on-openness"
TRAIT_COLOR = BIG_FIVE_COLORS["Openness"]
PLOT_TITLE = "Neuroticism suppressor (vanton4_gemma3_paired_dpo, Qwen3-235B judge) LoRA scale sweep on Openness prompts"
