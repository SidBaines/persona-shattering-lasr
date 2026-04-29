"""Neuroticism suppressor (vanton4_gemma3_paired_dpo, Qwen3-235B judge) LLM judge scale sweep.

Mirrors ``vanton4_paired_dpo/n_minus.py``, but points at the gemma-teacher-
trained adapter at:
  fine_tuning/llama-3.1-8b-it/ocean/neuroticism/suppressor/vanton4_gemma3_paired_dpo/lora/neuroticism_suppressing_full_vanton4-persona

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_gemma3_paired_dpo.n_minus
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_gemma3_paired_dpo._shared import *  # noqa: F401,F403
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/neuroticism.jsonl"

EVAL_NAME = "neuroticism-suppressor-vanton4-gemma3-paired-dpo"
TRAIT = OceanTrait.neuroticism

ADAPTER = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/gemma-3-27b-it/ocean/neuroticism/suppressor/vanton4_gemma3_paired_dpo"
    "/lora/neuroticism_suppressing_full_vanton4-persona"
)
ADAPTERS = [ADAPTER]
SCALES_PER_ADAPTER = {ADAPTER.slug: SCALE_POINTS}

JUDGE_METRIC_TRAITS = [OceanTrait.neuroticism.v2_metric_name]
TRAIT_COLOR = BIG_FIVE_COLORS["Neuroticism"]
PLOT_TITLE = "Neuroticism suppressor (vanton4_gemma3_paired_dpo, Qwen3-235B judge) LoRA scale sweep"
