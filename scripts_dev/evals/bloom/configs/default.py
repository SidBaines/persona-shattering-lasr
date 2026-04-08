"""Default configuration constants for the Bloom multi-turn rollout eval.

These are Python-level defaults.  The Bloom tool's own config lives in
``bloom-data/seed.yaml``; values here override or supplement it.

CLI flags (``--trait``, ``--targets``, ``--judgment-models``, ``--seed``)
take priority over values defined here.
"""

from __future__ import annotations

from pathlib import Path

# Path to the bloom-data directory (relative to project root)
BLOOM_DATA_DIR = Path("bloom-data")

# OCEAN trait to evaluate.  Set to None to use whatever is in seed.yaml.
TRAIT: str | None = None

# Target model short names from models.json
TARGETS: list[str] = ["llama-3.1-8b-it-base", "conscientiousness-low-llama"]

# Judgment model short names
JUDGMENT_MODELS: list[str] = ["glm-4.5-air"]

# RNG seed for stochastic stages.  Increment for independent runs.
SEED: int = 0

# HuggingFace dataset repo for persistence
HF_REPO: str = "persona-shattering-lasr/monorepo"

# ---------------------------------------------------------------------------
# Model overrides (None = use seed.yaml default)
# ---------------------------------------------------------------------------
UNDERSTANDING_MODEL: str | None = "glm-4.5-air"
IDEATION_MODEL: str | None = "claude-opus-4.6"
ROLLOUT_EVALUATOR_MODEL: str | None = "glm-4.5-air"

# ---------------------------------------------------------------------------
# LoRA scale sweep mode (all None = discrete target mode)
# ---------------------------------------------------------------------------
# When ADAPTER_REF is set, sweep mode is activated: the runner bakes the
# adapter at each scale point and runs bloom rollout+judgment per scale.
ADAPTER_REF: str | None = None
BASE_MODEL: str | None = None  # e.g. "meta-llama/Llama-3.1-8B-Instruct"
SCALE_POINTS: list[float] | None = None
INCLUDE_BASE: bool = True  # Include scale=0.0 (base model, no adapter) if not in SCALE_POINTS
MAX_LORA_RANK: int = 64
BAKED_ADAPTERS_DIR: str = "scratch/bloom-baked-adapters"
