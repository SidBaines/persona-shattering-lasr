"""a_minus adapter evaluated on Conscientiousness prompts (fresh rollouts).

Overrides DATASET_PATH + JUDGE_METRIC_TRAITS so this sweep generates new
rollouts on the conscientiousness prompt set and judges them for conscientiousness.
Fingerprint differs from a_minus.py (different DATASET_PATH), so rollouts
do NOT cache-hit with Run 1.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo.a_minus_on_conscientiousness \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

# Pull adapter, scales, base model, judge (Qwen3-235B), coherence metric, etc.
# from the own-trait module.
from scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo.a_minus import *  # noqa: F401,F403

from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

# Override: different prompt set → different rollout fingerprint.
DATASET_PATH = "data/ocean_open_ended/conscientiousness.jsonl"
TRAIT = OceanTrait.conscientiousness
JUDGE_METRIC_TRAITS = [OceanTrait.conscientiousness.v2_metric_name]
# COHERENCE_METRIC inherited from _shared.py (fresh rollouts → fresh coherence).

EVAL_NAME = "agreeableness-suppressor-vanton4-paired-dpo-on-conscientiousness"
TRAIT_COLOR = BIG_FIVE_COLORS["Conscientiousness"]
PLOT_TITLE = "Agreeableness suppressor (vanton4_paired_dpo, Qwen3-235B judge) LoRA scale sweep on Conscientiousness prompts"
