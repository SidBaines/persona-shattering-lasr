"""Extraversion amplifier (v3, Qwen3-235B judge) LLM judge scale sweep.

Companion to e_plus.py (which uses the vanton4 extraversion amplifier LoRA) —
this config targets the v3 LoRA instead. Used for the main-paper spider
plot and as one half of the c_minus_v2 × e_plus_v3 soup sweeps.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_qwen3.e_plus_v3 \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_qwen3._shared import *  # noqa: F401,F403
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/extraversion.jsonl"

EVAL_NAME = "extraversion-amplifier-v3"
TRAIT = OceanTrait.extraversion

# Skip the coherence judge for these runs — trait score delta is all we
# need here and the coherence judge roughly doubles judge cost per cell.
COHERENCE_METRIC = None

ADAPTER = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/extraversion/amplifier/v3"
    "/lora/extraversion_amplifying_full_v3-persona"
)
ADAPTERS = [ADAPTER]
SCALES_PER_ADAPTER = {ADAPTER.slug: SCALE_POINTS}

JUDGE_METRIC_TRAITS = [OceanTrait.extraversion.v2_metric_name]
TRAIT_COLOR = BIG_FIVE_COLORS["Extraversion"]
PLOT_TITLE = "Extraversion amplifier (v3, Qwen3-235B judge) LoRA scale sweep"
