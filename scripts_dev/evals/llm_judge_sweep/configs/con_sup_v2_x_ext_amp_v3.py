"""Combo LoRA scale sweep: conscientiousness suppressor (v2) × extraversion amplifier (v3).

5×5 Cartesian grid over per-adapter scales. Single-adapter slices populate
``fine_tuning/...`` cell paths so they cross-reuse with single-adapter sweeps;
true 2-adapter cells go under ``combos/{model}/<combo_slug>/...``.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.con_sup_v2_x_ext_amp_v3
"""

from __future__ import annotations

from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.llm_judge_agreement import JudgeRaterConfig
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
EVAL_NAME = "con_sup_v2__ext_amp_v3"

# ---------------------------------------------------------------------------
# Model & adapters
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_SLUG = "llama-3.1-8b-it"

CON_SUP_V2 = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v2"
    "/lora/conscientiousness_low_v2-persona"
)
EXT_AMP_V3 = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/extraversion/amplifier/v3"
    "/lora/extraversion_amplifying_full_v3-persona"
)

ADAPTERS = [CON_SUP_V2, EXT_AMP_V3]
SCALES_PER_ADAPTER = {
    CON_SUP_V2.slug: [-2.0, -1.0, 0.0, 1.0, 2.0],
    EXT_AMP_V3.slug: [-2.0, -1.0, 0.0, 1.0, 2.0],
}

# ---------------------------------------------------------------------------
# Trait — primary trait used for OceanJudgeRunConfig prompt selection. The
# coherence judge is trait-independent. We set extraversion as the primary
# (the trait metric below is named explicitly via JUDGE_METRIC_TRAIT).
# ---------------------------------------------------------------------------
TRAIT = OceanTrait.extraversion

# ---------------------------------------------------------------------------
# Rollout generation (canonical defaults — see defaults.py)
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
# Judge — trait metric is the extraversion judge; coherence is shared.
# ---------------------------------------------------------------------------
JUDGE_TEMPERATURE = 0.0
JUDGE_REPEATS = 2
CI_CONFIDENCE = 95.0
CI_BOOTSTRAP_RESAMPLES = 1000
JUDGE_METRIC_TRAIT = TRAIT.v2_metric_name
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
# Plot
# ---------------------------------------------------------------------------
PLOT_TITLE = "Conscientiousness sup (v2) × Extraversion amp (v3) LoRA sweep"
