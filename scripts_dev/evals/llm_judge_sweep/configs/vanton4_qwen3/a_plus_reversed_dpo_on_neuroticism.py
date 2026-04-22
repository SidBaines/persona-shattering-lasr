"""a_plus_reversed_dpo adapter evaluated on Neuroticism prompts (fresh rollouts).

Overrides DATASET_PATH + JUDGE_METRIC_TRAITS so this sweep generates new
rollouts on the neuroticism prompt set and judges them for neuroticism.
Fingerprint differs from a_plus_reversed_dpo.py (different DATASET_PATH), so rollouts
do NOT cache-hit with Run 1.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_qwen3.a_plus_reversed_dpo_on_neuroticism \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

# Pull adapter, scales, base model, judge (Qwen3-235B), coherence metric, etc.
# from the own-trait module.
from scripts_dev.evals.llm_judge_sweep.configs.vanton4_qwen3.a_plus_reversed_dpo import *  # noqa: F401,F403

from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

# Override: different prompt set → different rollout fingerprint.
DATASET_PATH = "data/ocean_open_ended/neuroticism.jsonl"
TRAIT = OceanTrait.neuroticism
JUDGE_METRIC_TRAITS = [OceanTrait.neuroticism.v2_metric_name]
# COHERENCE_METRIC inherited from _shared.py (fresh rollouts → fresh coherence).

EVAL_NAME = "agreeableness-amplifier-vanton4-reversed-dpo-on-neuroticism"
TRAIT_COLOR = BIG_FIVE_COLORS["Neuroticism"]
PLOT_TITLE = "Agreeableness amplifier (vanton4_reversed_dpo, Qwen3-235B judge) LoRA scale sweep on Neuroticism prompts"
