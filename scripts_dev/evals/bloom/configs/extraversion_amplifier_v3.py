"""Bloom eval: extraversion baseline + extraversion amplifier v3 (llama-3.1-8b-it).

Two cells at canonical defaults:
  - baseline (scale 0.0) -> combos/{model}/_baseline/bloom_extraversion/{fp}/
  - amplifier v3 at +1.0 -> fine_tuning/.../evals/bloom_extraversion/{fp}/scale_+1.00/

Usage::

    uv run python scripts_dev/evals/bloom/runner_cells.py \\
        --config scripts_dev.evals.bloom.configs.extraversion_amplifier_v3
"""

from __future__ import annotations

from pathlib import Path

from src_dev.evals.cell_sweep.cell_identity import AdapterSpec

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
BLOOM_DATA_DIR = Path("bloom-data")
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_SLUG = "llama-3.1-8b-it"
TRAITS: list[str] = ["extraversion"]

# ---------------------------------------------------------------------------
# Adapters — scale 0.0 collapses to baseline cell automatically.
# ---------------------------------------------------------------------------
EXT_AMP_V3 = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/extraversion/amplifier/v3"
    "/lora/extraversion_amplifying_full_v3-persona"
)
ADAPTERS = [EXT_AMP_V3]
SCALES_PER_ADAPTER = {
    EXT_AMP_V3.slug: [0.0, 1.0],
}

# ---------------------------------------------------------------------------
# Canonical bloom-sweep defaults (matches CANONICAL_BLOOM_DEFAULTS).
# ---------------------------------------------------------------------------
SCENARIO_VERSION = 1
NUM_SCENARIOS = 10
NUM_REPS = 2
MAX_TURNS = 10
ROLLOUT_MAX_TOKENS = 4096
MODALITY = "simenv"
NO_USER_MODE = False
ANONYMOUS_TARGET = False
TEMPERATURE = 0.7
JUDGE_TEMPERATURE = 0.0
IDEATION_TEMPERATURE = 1.0
EVALUATOR_REASONING_EFFORT = "none"
IDEATION_REASONING_EFFORT = "high"
TARGET_REASONING_EFFORT = "none"
UNDERSTANDING_MODEL = "openrouter/anthropic/claude-opus-4.6"
UNDERSTANDING_MAX_TOKENS = 8192
IDEATION_MODEL = "openrouter/anthropic/claude-opus-4.6"
IDEATION_MAX_TOKENS = 8192
ROLLOUT_EVALUATOR_MODEL = "openrouter/z-ai/glm-4.7-flash"
WEB_SEARCH = False
VARIATION_DIMENSIONS = None
SELECTED_VARIATIONS = None

# ---------------------------------------------------------------------------
# Judgment
# ---------------------------------------------------------------------------
JUDGMENT_MODELS = ["openrouter/z-ai/glm-4.7-flash"]
# Sample judgments multiple times per transcript; bloom averages over successful
# parses. Mitigates occasional tag-parse failures from the cheap judge (which
# otherwise cause bloom to emit 0 as a sentinel for the whole scenario).
JUDGMENT_NUM_SAMPLES = 3

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
SEED = 0
MAX_LORA_RANK = 64
HF_REPO = "persona-shattering-lasr/monorepo"
