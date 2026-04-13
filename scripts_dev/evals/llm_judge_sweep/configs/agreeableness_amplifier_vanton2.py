"""Config constants for the agreeableness amplifier (vanton2) LoRA scale sweep.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner \\
        --config scripts_dev.evals.llm_judge_sweep.configs.agreeableness_amplifier_vanton2
"""

from __future__ import annotations

from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.llm_judge_agreement import JudgeRaterConfig
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
EVAL_NAME = "agreeableness-amplifier-vanton2"

# ---------------------------------------------------------------------------
# Model & adapter
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_SLUG = "llama-3.1-8b-it"
ADAPTER_REF = (
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/vanton2"
    "/lora/agreeableness_amplifying_full_vanton2-persona"
)
BAKED_ADAPTERS_SUBDIR = "agreeableness_amplifier_vanton2"

# ---------------------------------------------------------------------------
# Trait / OCT path slots
# ---------------------------------------------------------------------------
TRAIT = OceanTrait.agreeableness
DIRECTION = "amplifier"
VERSION = "vanton2"

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
SCALE_POINTS = [-2.0, -1.0, 0.0, 1.0, 2.0]
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
TRAIT_COLOR = BIG_FIVE_COLORS["Agreeableness"]
COHERENCE_COLOR = "#757575"
PLOT_TITLE = "Agreeableness amplifier (vanton2) LoRA scale sweep"
