"""Conscientiousness suppressor LoRA scale sweep via Bloom multi-turn eval.

Sweeps the conscientiousness suppressor adapter over 9 scale points,
running bloom rollout+judgment at each scale.  Understanding and ideation
run once and are shared across all scale points.

Usage::

    uv run python -m scripts_dev.evals.bloom.runner \
        --config scripts_dev.evals.bloom.configs.conscientiousness_sweep
"""

from __future__ import annotations

from pathlib import Path

# Path to the bloom-data directory (relative to project root)
BLOOM_DATA_DIR = Path("bloom-data")

# OCEAN trait to evaluate
TRAIT: str = "conscientiousness"

# Judgment model short names
JUDGMENT_MODELS: list[str] = ["glm-4.5-air"]

# RNG seed for stochastic stages
SEED: int = 0

# HuggingFace dataset repo for persistence
HF_REPO: str = "persona-shattering-lasr/monorepo"

# ---------------------------------------------------------------------------
# Model overrides
# ---------------------------------------------------------------------------
UNDERSTANDING_MODEL: str | None = "glm-4.5-air"
IDEATION_MODEL: str | None = "claude-opus-4.6"
ROLLOUT_EVALUATOR_MODEL: str | None = "glm-4.5-air"

# ---------------------------------------------------------------------------
# LoRA scale sweep
# ---------------------------------------------------------------------------
ADAPTER_REF: str = (
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/"
    "suppressor/v3/lora/conscientiousness_low-persona"
)
BASE_MODEL: str = "meta-llama/Llama-3.1-8B-Instruct"
SCALE_POINTS: list[float] = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
INCLUDE_BASE: bool = True
MAX_LORA_RANK: int = 64
BAKED_ADAPTERS_DIR: str = "scratch/bloom-baked-adapters/conscientiousness_suppressor"

# Not used in sweep mode (targets are generated from scale points)
TARGETS: list[str] = []
