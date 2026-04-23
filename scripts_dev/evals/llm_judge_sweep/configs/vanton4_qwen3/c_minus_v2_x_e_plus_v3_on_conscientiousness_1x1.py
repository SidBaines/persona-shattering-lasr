"""c_minus_v2 × e_plus_v3 soup at (+1, +1), rollouts on Conscientiousness prompts.

See c_minus_v2_x_e_plus_v3_on_openness_1x1.py for rationale.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_qwen3.c_minus_v2_x_e_plus_v3_on_conscientiousness_1x1 \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_qwen3.c_minus_v2_x_e_plus_v3_on_openness_1x1 import *  # noqa: F401,F403

from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/conscientiousness.jsonl"
TRAIT = OceanTrait.conscientiousness
JUDGE_METRIC_TRAITS = [OceanTrait.conscientiousness.v2_metric_name]

EVAL_NAME = "c_minus_v2_x_e_plus_v3-on-conscientiousness-1x1"
TRAIT_COLOR = BIG_FIVE_COLORS["Conscientiousness"]
PLOT_TITLE = "c_minus_v2 × e_plus_v3 @ (+1, +1) on Conscientiousness prompts"
