"""n_minus (gemma3 teacher) adapter evaluated on Agreeableness prompts.

Overrides DATASET_PATH + JUDGE_METRIC_TRAITS so this sweep generates new
rollouts on the agreeableness prompt set and judges them for agreeableness.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_gemma3_paired_dpo.n_minus_on_agreeableness \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_gemma3_paired_dpo.n_minus import *  # noqa: F401,F403

from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/agreeableness.jsonl"
TRAIT = OceanTrait.agreeableness
JUDGE_METRIC_TRAITS = [OceanTrait.agreeableness.v2_metric_name]

EVAL_NAME = "neuroticism-suppressor-vanton4-gemma3-paired-dpo-on-agreeableness"
TRAIT_COLOR = BIG_FIVE_COLORS["Agreeableness"]
PLOT_TITLE = "Neuroticism suppressor (vanton4_gemma3_paired_dpo, Qwen3-235B judge) LoRA scale sweep on Agreeableness prompts"
