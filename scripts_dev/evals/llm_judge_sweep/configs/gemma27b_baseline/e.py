"""Gemma-3-27b-IT baseline cell on the extraversion dataset.

Run::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.gemma27b_baseline.e
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.gemma27b_baseline._shared import *  # noqa: F401,F403
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/extraversion.jsonl"
EVAL_NAME = "gemma27b-baseline-extraversion"
TRAIT = OceanTrait.extraversion
JUDGE_METRIC_TRAITS = [OceanTrait.extraversion.v2_metric_name]
TRAIT_COLOR = BIG_FIVE_COLORS["Extraversion"]
PLOT_TITLE = "Gemma-3-27b-IT baseline (extraversion dataset)"
