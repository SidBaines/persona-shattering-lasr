"""e_plus_v3 adapter evaluated on Conscientiousness prompts (fresh rollouts).

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_qwen3.e_plus_v3_on_conscientiousness \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_qwen3.e_plus_v3 import *  # noqa: F401,F403

from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/conscientiousness.jsonl"
TRAIT = OceanTrait.conscientiousness
JUDGE_METRIC_TRAITS = [OceanTrait.conscientiousness.v2_metric_name]

EVAL_NAME = "extraversion-amplifier-v3-on-conscientiousness"
TRAIT_COLOR = BIG_FIVE_COLORS["Conscientiousness"]
PLOT_TITLE = "Extraversion amplifier (v3, Qwen3-235B judge) LoRA scale sweep on Conscientiousness prompts"
