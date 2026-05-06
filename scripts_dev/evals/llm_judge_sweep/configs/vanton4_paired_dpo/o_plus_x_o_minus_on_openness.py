"""o_plus × o_minus (vanton4_paired_dpo) same-trait opposing sweep on openness prompts.

5×5 grid of amplifier × suppressor scales in {0.0, 0.5, 1.0, 1.5, 2.0}.
(0,0) → baseline; (x,0) → single-adapter o_plus; (0,y) → single-adapter o_minus.
Only judged on openness_v2 — the tug-of-war is within the same trait.

This sweep is the Appendix E "amplifiers and suppressors of the same trait used to
negate the effects of each other" study for Openness.

Usage::

    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo.o_plus_x_o_minus_on_openness \\
        --allow-custom-fingerprint 2>&1 | tee scratch/sweep_o_plus_x_o_minus.log
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo._shared import *  # noqa: F401,F403
from src_dev.evals.cell_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/openness.jsonl"

MAX_SAMPLES = 100
NUM_ROLLOUTS_PER_PROMPT = 1

EVAL_NAME = "o_plus_x_o_minus-vanton4-paired-dpo-on-openness"
TRAIT = OceanTrait.openness

ADAPTER_O_PLUS = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/openness/amplifier/vanton4_paired_dpo"
    "/lora/openness_amplifying_full_vanton4-persona"
)
ADAPTER_O_MINUS = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/openness/suppressor/vanton4_paired_dpo"
    "/lora/openness_suppressing_full_vanton4-persona"
)
ADAPTERS = [ADAPTER_O_PLUS, ADAPTER_O_MINUS]

# [0, +2] in 0.5 steps — negatives not needed (reflection of positives)
SCALE_POINTS = [0.0, 0.5, 1.0, 1.5, 2.0]
SCALES_PER_ADAPTER = {
    ADAPTER_O_PLUS.slug: SCALE_POINTS,
    ADAPTER_O_MINUS.slug: SCALE_POINTS,
}

JUDGE_METRIC_TRAITS = [OceanTrait.openness.v2_metric_name]
# Skip coherence to keep the sweep focused and budget small.
COHERENCE_METRIC = None

TRAIT_COLOR = BIG_FIVE_COLORS["Openness"]
PLOT_TITLE = "o_plus × o_minus (vanton4_paired_dpo) on openness prompts — Qwen3-235B judge"
