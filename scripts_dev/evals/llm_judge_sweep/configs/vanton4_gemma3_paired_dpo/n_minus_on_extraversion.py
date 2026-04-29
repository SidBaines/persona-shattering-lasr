"""n_minus (gemma3 teacher) adapter evaluated on Extraversion prompts.

Overrides DATASET_PATH + JUDGE_METRIC_TRAITS so this sweep generates new
rollouts on the extraversion prompt set and judges them for extraversion.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_gemma3_paired_dpo.n_minus_on_extraversion \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_gemma3_paired_dpo.n_minus import *  # noqa: F401,F403

from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/extraversion.jsonl"
TRAIT = OceanTrait.extraversion
JUDGE_METRIC_TRAITS = [OceanTrait.extraversion.v2_metric_name]

EVAL_NAME = "neuroticism-suppressor-vanton4-gemma3-paired-dpo-on-extraversion"
TRAIT_COLOR = BIG_FIVE_COLORS["Extraversion"]
PLOT_TITLE = "Neuroticism suppressor (vanton4_gemma3_paired_dpo, Qwen3-235B judge) LoRA scale sweep on Extraversion prompts"
