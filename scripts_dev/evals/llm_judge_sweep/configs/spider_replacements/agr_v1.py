"""Agreeableness amplifier (v1/high, Qwen3-235B judge) LLM judge (scale=1.0) — spider-replacement own-trait config.

Single scale point (1.0) matching spider fig_1 / fig_1b inputs. Baselines at
fingerprint FP(agreeableness) already cached on HF at this family's shared settings.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
        --config scripts_dev.evals.llm_judge_sweep.configs.spider_replacements.agr_v1 \
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.spider_replacements._shared import *  # noqa: F401,F403
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/agreeableness.jsonl"

EVAL_NAME = "agreeableness-amplifier-v1"
TRAIT = OceanTrait.agreeableness

ADAPTER = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/v1/lora/agreeableness_high-persona"
)
ADAPTERS = [ADAPTER]
SCALES_PER_ADAPTER = {ADAPTER.slug: [1.0]}

JUDGE_METRIC_TRAITS = [OceanTrait.agreeableness.v2_metric_name]
TRAIT_COLOR = BIG_FIVE_COLORS["Agreeableness"]
PLOT_TITLE = "Agreeableness amplifier (v1/high, Qwen3-235B judge) LoRA scale sweep"
