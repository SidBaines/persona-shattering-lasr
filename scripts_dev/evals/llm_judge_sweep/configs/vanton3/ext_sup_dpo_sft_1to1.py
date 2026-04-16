"""Extraversion suppressor (vanton3) linked DPO+SFT scale sweep.

At each scale point s, both the DPO and SFT vanton3 suppressor LoRAs are
applied at scale s simultaneously (1:1 linked co-varying scales). By LoRA
linearity this is equivalent to applying a fused ``dpo + sft`` adapter at
scale s. Produces a 1D sweep (not a 2D grid).

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton3.ext_sup_dpo_sft_1to13_dpo_sft_one_to_one
"""

from __future__ import annotations

from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.llm_judge_agreement import JudgeRaterConfig
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
EVAL_NAME = "extraversion-suppressor-vanton3-dpo-sft-1to1"
EVAL_VARIANT = "dpo_sft_one_to_one"

# ---------------------------------------------------------------------------
# Model & adapters (linked 1:1 scale co-variation)
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_SLUG = "llama-3.1-8b-it"
ADAPTER_REFS = [
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/extraversion/suppressor/vanton3"
    "/lora/extraversion_suppressing_full_vanton3-dpo",
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/extraversion/suppressor/vanton3"
    "/lora/extraversion_suppressing_full_vanton3-sft",
]
BAKED_ADAPTERS_SUBDIR = "extraversion_suppressor_vanton3_dpo_sft_one_to_one"

# ---------------------------------------------------------------------------
# Trait / OCT path slots (DPO and SFT share the same (trait, direction, version))
# ---------------------------------------------------------------------------
TRAIT = OceanTrait.extraversion
DIRECTION = "suppressor"
VERSION = "vanton3"

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
SCALE_POINTS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
SEED = 42

# ---------------------------------------------------------------------------
# Rollout generation
# ---------------------------------------------------------------------------
MAX_SAMPLES = 100
NUM_ROLLOUTS_PER_PROMPT = 1
DATASET_PATH = "data/assistant-axis-extraction-questions.jsonl"
ASSISTANT_MAX_NEW_TOKENS = 256
ASSISTANT_BATCH_SIZE = 32
ASSISTANT_TEMPERATURE = 0.7
ASSISTANT_TOP_P = 0.95
USER_MODEL = "z-ai/glm-4.5-air:free"
USER_PROVIDER = "openrouter"

# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------
JUDGE_TEMPERATURE = 0.0
JUDGE_REPEATS = 2
CI_CONFIDENCE = 95.0
CI_BOOTSTRAP_RESAMPLES = 1000
COHERENCE_METRIC = "better_coherence_judge"
JUDGE_RATERS = [
    JudgeRaterConfig(
        rater_id="gemini_flash_20",
        judge=JudgeLLMConfig(
            provider="openrouter",
            model="google/gemini-2.0-flash-001",
            temperature=JUDGE_TEMPERATURE,
            max_concurrent=10,
        ),
    ),
]

# ---------------------------------------------------------------------------
# Plot colors
# ---------------------------------------------------------------------------
TRAIT_COLOR = BIG_FIVE_COLORS["Extraversion"]
COHERENCE_COLOR = "#757575"
PLOT_TITLE = "Extraversion suppressor (vanton3) DPO+SFT 1:1 linked LoRA sweep"
