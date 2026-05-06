"""e_plus × e_minus (vanton4_paired_dpo) same-trait opposing sweep on extraversion prompts.

5×5 grid of amplifier × suppressor scales in {0.0, 0.5, 1.0, 1.5, 2.0}.
(0,0) → baseline; (x,0) → single-adapter e_plus; (0,y) → single-adapter e_minus.
Only judged on extraversion_v2 — the tug-of-war is within the same trait.

Appendix E "amplifiers and suppressors of the same trait used to negate the effects
of each other" study for Extraversion.

Usage::

    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo.e_plus_x_e_minus_on_extraversion \\
        --allow-custom-fingerprint 2>&1 | tee scratch/sweep_e_plus_x_e_minus.log
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo._shared import *  # noqa: F401,F403
from src_dev.evals.cell_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/extraversion.jsonl"

MAX_SAMPLES = 100
NUM_ROLLOUTS_PER_PROMPT = 1

EVAL_NAME = "e_plus_x_e_minus-vanton4-paired-dpo-on-extraversion"
TRAIT = OceanTrait.extraversion

ADAPTER_E_PLUS = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/extraversion/amplifier/vanton4_paired_dpo"
    "/lora/extraversion_amplifying_full_vanton4-persona"
)
ADAPTER_E_MINUS = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/extraversion/suppressor/vanton4_paired_dpo"
    "/lora/extraversion_suppressing_full_vanton4-persona"
)
ADAPTERS = [ADAPTER_E_PLUS, ADAPTER_E_MINUS]

SCALE_POINTS = [0.0, 0.5, 1.0, 1.5, 2.0]
SCALES_PER_ADAPTER = {
    ADAPTER_E_PLUS.slug: SCALE_POINTS,
    ADAPTER_E_MINUS.slug: SCALE_POINTS,
}

JUDGE_METRIC_TRAITS = [OceanTrait.extraversion.v2_metric_name]
COHERENCE_METRIC = None

TRAIT_COLOR = BIG_FIVE_COLORS["Extraversion"]
PLOT_TITLE = "e_plus × e_minus (vanton4_paired_dpo) on extraversion prompts — Qwen3-235B judge"
