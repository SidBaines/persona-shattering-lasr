"""Gemma-3-27b-IT baseline cell on the openness dataset.

Produces a single baseline cell (no adapters, scale +0.00) at
``combos/gemma-3-27b-it/_baseline/llm_judge_lora_scale_sweep/<fp>/`` with
``judge_runs/qwen3_235b/openness_v2.jsonl``.

Run::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.gemma27b_baseline.o
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.gemma27b_baseline._shared import *  # noqa: F401,F403
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/openness.jsonl"
EVAL_NAME = "gemma27b-baseline-openness"
TRAIT = OceanTrait.openness
JUDGE_METRIC_TRAITS = [OceanTrait.openness.v2_metric_name]
TRAIT_COLOR = BIG_FIVE_COLORS["Openness"]
PLOT_TITLE = "Gemma-3-27b-IT baseline (openness dataset)"
