"""Gemma-3 27b conscientiousness suppressor v2 LoRA scale sweep (Qwen3-235B judge).

Consc+coherence only, full scale sweep [-2, -1, 0, 1, 2].

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
        --config scripts_dev.evals.llm_judge_sweep.configs.gemma_consc_sup.g27b \
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.gemma_consc_sup._shared import *  # noqa: F401,F403
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

BASE_MODEL = "google/gemma-3-27b-it"
BASE_MODEL_SLUG = "gemma-3-27b-it"

# 27B weights take ~52 GiB on an 80 GB H100, leaving ~20-24 GiB for KV cache +
# activations. Default max_model_len=131072 makes vLLM reserve far more KV
# than we need (prompts + 2048 new tokens stay well under 4k). Cap the context
# so the engine fits, and reduce the assistant batch size.
ASSISTANT_MAX_MODEL_LEN = 4096
ASSISTANT_GPU_MEMORY_UTILIZATION = 0.96
ASSISTANT_ENFORCE_EAGER = True  # skip torch.compile's cudagraph reservation (~few GiB on 27B)
ASSISTANT_BATCH_SIZE = 16

EVAL_NAME = "gemma-3-27b-conscientiousness-suppressor-v2"
TRAIT = OceanTrait.conscientiousness

ADAPTER = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/gemma-3-27b-it/ocean/conscientiousness/suppressor/v2"
    "/lora/conscientiousness_low_v2-persona"
)
ADAPTERS = [ADAPTER]
SCALES_PER_ADAPTER = {ADAPTER.slug: SCALE_POINTS}

JUDGE_METRIC_TRAITS = [OceanTrait.conscientiousness.v2_metric_name]
TRAIT_COLOR = BIG_FIVE_COLORS["Conscientiousness"]
PLOT_TITLE = "Gemma-3 27b conscientiousness suppressor v2 (Qwen3-235B judge) LoRA scale sweep"
