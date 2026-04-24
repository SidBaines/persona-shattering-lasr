"""Agreeableness amplifier (v1/high, Qwen3-235B judge) on Extraversion prompts (scale=1.0) — spider cross-trait cell.

Overrides DATASET_PATH and JUDGE_METRIC_TRAITS so this sweep generates fresh
rollouts on the extraversion prompt set and judges them for extraversion.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
        --config scripts_dev.evals.llm_judge_sweep.configs.spider_replacements.agr_v1_on_extraversion \
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.spider_replacements.agr_v1 import *  # noqa: F401,F403

from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/extraversion.jsonl"
TRAIT = OceanTrait.extraversion
JUDGE_METRIC_TRAITS = [OceanTrait.extraversion.v2_metric_name]

EVAL_NAME = "agreeableness-amplifier-v1-on-extraversion"
TRAIT_COLOR = BIG_FIVE_COLORS["Extraversion"]
PLOT_TITLE = "Agreeableness amplifier (v1/high, Qwen3-235B judge) LoRA scale sweep on Extraversion prompts"
