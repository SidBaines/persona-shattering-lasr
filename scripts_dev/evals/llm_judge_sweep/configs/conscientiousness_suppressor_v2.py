"""Standalone scale sweep of the conscientiousness suppressor (v2, persona) LoRA.

Shaped for ``runner_cells.py`` (single-element ``ADAPTERS``) so the rollout
fingerprint matches the ``con_sup_v2_x_ext_amp_v3`` combo sweep — cells where
only ``CON_SUP_V2`` is non-zero already landed at the same single-adapter HF
path (see ``cell_identity.py`` tier routing). Those rollouts + the
``better_coherence_judge`` results are already cached on HF; only the
``conscientiousness_v2`` trait judge needs to run fresh.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.conscientiousness_suppressor_v2
"""

from __future__ import annotations

from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.llm_judge_agreement import JudgeRaterConfig
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
EVAL_NAME = "conscientiousness-suppressor-v2"

# ---------------------------------------------------------------------------
# Model & adapter (single-adapter combo shape)
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_SLUG = "llama-3.1-8b-it"

CON_SUP_V2 = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v2"
    "/lora/conscientiousness_low_v2-persona"
)
ADAPTERS = [CON_SUP_V2]
SCALES_PER_ADAPTER = {CON_SUP_V2.slug: [-2.0, -1.0, 0.0, 1.0, 2.0]}

# ---------------------------------------------------------------------------
# Trait — primary for OceanJudgeRunConfig prompt selection; the actual judge
# metrics are driven by JUDGE_METRIC_TRAITS below.
# ---------------------------------------------------------------------------
TRAIT = OceanTrait.conscientiousness

# ---------------------------------------------------------------------------
# Rollout generation — MUST match con_sup_v2_x_ext_amp_v3.py exactly so the
# rollout fingerprint (content-addressed over these fields) collides with the
# already-cached HF data at
# fine_tuning/.../v2/evals/llm_judge_lora_scale_sweep/41c5cf1171/.
# ---------------------------------------------------------------------------
SEED = 42
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
# Judge — trait conscientiousness + coherence. Coherence is already cached on
# HF from the earlier combo run; conscientiousness is fresh.
# ---------------------------------------------------------------------------
JUDGE_TEMPERATURE = 0.0
JUDGE_REPEATS = 2
CI_CONFIDENCE = 95.0
CI_BOOTSTRAP_RESAMPLES = 1000
JUDGE_METRIC_TRAITS = [OceanTrait.conscientiousness.v2_metric_name]
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
TRAIT_COLOR = BIG_FIVE_COLORS["Conscientiousness"]
COHERENCE_COLOR = "#757575"
PLOT_TITLE = "Conscientiousness suppressor (v2, persona) LoRA scale sweep"
