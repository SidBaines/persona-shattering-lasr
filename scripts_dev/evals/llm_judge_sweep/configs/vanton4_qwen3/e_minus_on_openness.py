"""e_minus adapter evaluated on Openness prompts (fresh rollouts).

Overrides DATASET_PATH + JUDGE_METRIC_TRAITS so this sweep generates new
rollouts on the openness prompt set and judges them for openness.
Fingerprint differs from e_minus.py (different DATASET_PATH), so rollouts
do NOT cache-hit with Run 1.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_qwen3.e_minus_on_openness \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

# Pull adapter, scales, base model, judge (Qwen3-235B), coherence metric, etc.
# from the own-trait module.
from scripts_dev.evals.llm_judge_sweep.configs.vanton4_qwen3.e_minus import *  # noqa: F401,F403

from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

# Override: different prompt set → different rollout fingerprint.
DATASET_PATH = "data/ocean_open_ended/openness.jsonl"
TRAIT = OceanTrait.openness
JUDGE_METRIC_TRAITS = [OceanTrait.openness.v2_metric_name]
# COHERENCE_METRIC inherited from _shared.py (fresh rollouts → fresh coherence).

EVAL_NAME = "extraversion-suppressor-vanton4-on-openness"
TRAIT_COLOR = BIG_FIVE_COLORS["Openness"]
PLOT_TITLE = "Extraversion suppressor (vanton4, Qwen3-235B judge) LoRA scale sweep on Openness prompts"
