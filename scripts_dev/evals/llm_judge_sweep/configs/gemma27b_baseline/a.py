"""Gemma-3-27b-IT baseline cell on the agreeableness dataset.

Run::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.gemma27b_baseline.a
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.gemma27b_baseline._shared import *  # noqa: F401,F403
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/agreeableness.jsonl"
EVAL_NAME = "gemma27b-baseline-agreeableness"
TRAIT = OceanTrait.agreeableness
JUDGE_METRIC_TRAITS = [OceanTrait.agreeableness.v2_metric_name]
TRAIT_COLOR = BIG_FIVE_COLORS["Agreeableness"]
PLOT_TITLE = "Gemma-3-27b-IT baseline (agreeableness dataset)"
