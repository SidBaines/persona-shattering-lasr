"""a_plus × a_minus (vanton4_paired_dpo) same-trait opposing sweep on agreeableness prompts.

5×5 grid of amplifier × suppressor scales in {0.0, 0.5, 1.0, 1.5, 2.0}.
(0,0) → baseline; (x,0) → single-adapter a_plus; (0,y) → single-adapter a_minus.
Only judged on agreeableness_v2 — the tug-of-war is within the same trait.

Appendix E "amplifiers and suppressors of the same trait used to negate the effects
of each other" study for Agreeableness.

Usage::

    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo.a_plus_x_a_minus_on_agreeableness \\
        --allow-custom-fingerprint 2>&1 | tee scratch/sweep_a_plus_x_a_minus.log
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo._shared import *  # noqa: F401,F403
from src_dev.evals.cell_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/agreeableness.jsonl"

MAX_SAMPLES = 100
NUM_ROLLOUTS_PER_PROMPT = 1

EVAL_NAME = "a_plus_x_a_minus-vanton4-paired-dpo-on-agreeableness"
TRAIT = OceanTrait.agreeableness

ADAPTER_A_PLUS = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/vanton4_paired_dpo"
    "/lora/agreeableness_amplifying_full_vanton4-persona"
)
ADAPTER_A_MINUS = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/vanton4_paired_dpo"
    "/lora/agreeableness_suppressing_full_vanton4-persona"
)
ADAPTERS = [ADAPTER_A_PLUS, ADAPTER_A_MINUS]

SCALE_POINTS = [0.0, 0.5, 1.0, 1.5, 2.0]
SCALES_PER_ADAPTER = {
    ADAPTER_A_PLUS.slug: SCALE_POINTS,
    ADAPTER_A_MINUS.slug: SCALE_POINTS,
}

JUDGE_METRIC_TRAITS = [OceanTrait.agreeableness.v2_metric_name]
COHERENCE_METRIC = None

TRAIT_COLOR = BIG_FIVE_COLORS["Agreeableness"]
PLOT_TITLE = "a_plus × a_minus (vanton4_paired_dpo) on agreeableness prompts — Qwen3-235B judge"
