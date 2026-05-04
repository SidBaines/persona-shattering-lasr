"""Recipe-matched null control (vanton4_paired_dpo_s1vs2) evaluated on Agreeableness prompts.

The control adapter is a recipe-matched null (chosen/rejected both teacher-
generated under the same OCEAN-default constitution). This sweep generates
fresh rollouts on the agreeableness open-ended question set and judges them for
agreeableness. Useful as a baseline for the recipe-only contribution to Agreeableness
shift; the OCEAN-trait adapters' agreeableness sweeps should sit visibly above this.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo.control_s1vs2_on_agreeableness \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

# Pull adapter, scales, base model, judge (Qwen3-235B), coherence metric, etc.
# from the control_s1vs2 base.
from scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo.control_s1vs2 import *  # noqa: F401,F403

from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/agreeableness.jsonl"
TRAIT = OceanTrait.agreeableness
JUDGE_METRIC_TRAITS = [OceanTrait.agreeableness.v2_metric_name]

EVAL_NAME = "control-vanton4-paired-dpo-s1vs2-on-agreeableness"
TRAIT_COLOR = BIG_FIVE_COLORS["Agreeableness"]
PLOT_TITLE = "Control (vanton4_paired_dpo_s1vs2 null, Qwen3-235B judge) LoRA scale sweep on Agreeableness prompts"
