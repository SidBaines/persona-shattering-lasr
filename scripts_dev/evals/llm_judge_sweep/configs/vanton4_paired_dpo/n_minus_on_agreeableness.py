"""n_minus adapter evaluated on Agreeableness prompts (fresh rollouts).

Overrides DATASET_PATH + JUDGE_METRIC_TRAITS so this sweep generates new
rollouts on the agreeableness prompt set and judges them for agreeableness.
Fingerprint differs from n_minus.py (different DATASET_PATH), so rollouts
do NOT cache-hit with Run 1.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo.n_minus_on_agreeableness \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

# Pull adapter, scales, base model, judge (Qwen3-235B), coherence metric, etc.
# from the own-trait module.
from scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo.n_minus import *  # noqa: F401,F403

from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

# Override: different prompt set → different rollout fingerprint.
DATASET_PATH = "data/ocean_open_ended/agreeableness.jsonl"
TRAIT = OceanTrait.agreeableness
JUDGE_METRIC_TRAITS = [OceanTrait.agreeableness.v2_metric_name]
# COHERENCE_METRIC inherited from _shared.py (fresh rollouts → fresh coherence).

EVAL_NAME = "neuroticism-suppressor-vanton4-paired-dpo-on-agreeableness"
TRAIT_COLOR = BIG_FIVE_COLORS["Agreeableness"]
PLOT_TITLE = "Neuroticism suppressor (vanton4_paired_dpo, Qwen3-235B judge) LoRA scale sweep on Agreeableness prompts"
