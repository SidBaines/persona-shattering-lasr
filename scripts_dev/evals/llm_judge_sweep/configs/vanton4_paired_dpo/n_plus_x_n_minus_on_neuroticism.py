"""n_plus × n_minus (vanton4_paired_dpo) same-trait opposing sweep on neuroticism prompts.

5×5 grid of amplifier × suppressor scales in {0.0, 0.5, 1.0, 1.5, 2.0}.
(0,0) → baseline; (x,0) → single-adapter n_plus; (0,y) → single-adapter n_minus.
Only judged on neuroticism_v2 — the tug-of-war is within the same trait.

Appendix E "amplifiers and suppressors of the same trait used to negate the effects
of each other" study for Neuroticism.

Usage::

    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo.n_plus_x_n_minus_on_neuroticism \\
        --allow-custom-fingerprint 2>&1 | tee scratch/sweep_n_plus_x_n_minus.log
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo._shared import *  # noqa: F401,F403
from src_dev.evals.cell_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/neuroticism.jsonl"

MAX_SAMPLES = 100
NUM_ROLLOUTS_PER_PROMPT = 1

EVAL_NAME = "n_plus_x_n_minus-vanton4-paired-dpo-on-neuroticism"
TRAIT = OceanTrait.neuroticism

ADAPTER_N_PLUS = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/neuroticism/amplifier/vanton4_paired_dpo"
    "/lora/neuroticism_amplifying_full_vanton4-persona"
)
ADAPTER_N_MINUS = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/neuroticism/suppressor/vanton4_paired_dpo"
    "/lora/neuroticism_suppressing_full_vanton4-persona"
)
ADAPTERS = [ADAPTER_N_PLUS, ADAPTER_N_MINUS]

SCALE_POINTS = [0.0, 0.5, 1.0, 1.5, 2.0]
SCALES_PER_ADAPTER = {
    ADAPTER_N_PLUS.slug: SCALE_POINTS,
    ADAPTER_N_MINUS.slug: SCALE_POINTS,
}

JUDGE_METRIC_TRAITS = [OceanTrait.neuroticism.v2_metric_name]
COHERENCE_METRIC = None

TRAIT_COLOR = BIG_FIVE_COLORS["Neuroticism"]
PLOT_TITLE = "n_plus × n_minus (vanton4_paired_dpo) on neuroticism prompts — Qwen3-235B judge"
