"""Default configuration constants for the Bloom multi-turn rollout eval.

These are Python-level defaults.  The Bloom tool's own config lives in
``bloom-data/seed.yaml``; values here override or supplement it.

CLI flags (``--trait``, ``--targets``, ``--judgment-models``, ``--seed``)
take priority over values defined here.
"""

from pathlib import Path

# Path to the bloom-data directory (relative to project root)
BLOOM_DATA_DIR = Path("bloom-data")

# OCEAN trait to evaluate.  Set to None to use whatever is in seed.yaml.
TRAIT: str | None = None

# Target model short names from models.json
TARGETS: list[str] = ["llama-3.1-8b-it-base", "conscientiousness-low-llama"]

# Judgment model short names
JUDGMENT_MODELS: list[str] = ["gpt-5-mini"]

# RNG seed for stochastic stages.  Increment for independent runs.
SEED: int = 0

# HuggingFace dataset repo for persistence
HF_REPO: str = "persona-shattering-lasr/monorepo"
