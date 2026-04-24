"""Conscientiousness amplifier (v1/souped, Qwen3-235B judge) LLM judge (scale=1.0) — spider-replacement own-trait config.

Single scale point (1.0) matching spider fig_1 / fig_1b inputs. Baselines at
fingerprint FP(conscientiousness) already cached on HF at this family's shared settings.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
        --config scripts_dev.evals.llm_judge_sweep.configs.spider_replacements.consc_souped \
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.spider_replacements._shared import *  # noqa: F401,F403
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/conscientiousness.jsonl"

EVAL_NAME = "conscientiousness-amplifier-v1"
TRAIT = OceanTrait.conscientiousness

ADAPTER = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/amplifier/v1/lora/souped"
)
ADAPTERS = [ADAPTER]
SCALES_PER_ADAPTER = {ADAPTER.slug: [1.0]}

JUDGE_METRIC_TRAITS = [OceanTrait.conscientiousness.v2_metric_name]
TRAIT_COLOR = BIG_FIVE_COLORS["Conscientiousness"]
PLOT_TITLE = "Conscientiousness amplifier (v1/souped, Qwen3-235B judge) LoRA scale sweep"
