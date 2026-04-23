"""c_minus_v2 × e_plus_v3 soup at (+1, +1), rollouts on Openness prompts.

Single-cell companion to the c_minus_v2 and e_plus_v3 single-adapter sweeps.
Used to populate the c_minus_v2 × e_plus_v3 combo-delta / spider panels.

Matches the MAX_SAMPLES=240 / NUM_ROLLOUTS_PER_PROMPT=1 defaults from
_shared.py so rollout fingerprints line up with the single-adapter sweeps
on the same dataset.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_qwen3.c_minus_v2_x_e_plus_v3_on_openness_1x1 \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_qwen3._shared import *  # noqa: F401,F403
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/openness.jsonl"

EVAL_NAME = "c_minus_v2_x_e_plus_v3-on-openness-1x1"
TRAIT = OceanTrait.openness

# Trait delta only — no coherence judge for these sweeps.
COHERENCE_METRIC = None

ADAPTER_C_MINUS_V2 = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v2"
    "/lora/conscientiousness_low_v2-persona"
)
ADAPTER_E_PLUS_V3 = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/extraversion/amplifier/v3"
    "/lora/extraversion_amplifying_full_v3-persona"
)
ADAPTERS = [ADAPTER_C_MINUS_V2, ADAPTER_E_PLUS_V3]
# Single-scale per adapter → enumerates to exactly one (+1, +1) combo cell.
SCALES_PER_ADAPTER = {
    ADAPTER_C_MINUS_V2.slug: [1.0],
    ADAPTER_E_PLUS_V3.slug: [1.0],
}

JUDGE_METRIC_TRAITS = [OceanTrait.openness.v2_metric_name]

TRAIT_COLOR = BIG_FIVE_COLORS["Openness"]
PLOT_TITLE = "c_minus_v2 × e_plus_v3 @ (+1, +1) on Openness prompts"
