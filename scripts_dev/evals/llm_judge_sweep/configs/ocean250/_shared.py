"""Shared constants for the 4 per-cell OCEAN-250 judge sweeps.

Each sibling config (baseline / con_sup / ext_amp / combo) re-exports these
so all four runs land under the same rollout fingerprint — that fingerprint
is a SHA-256 over the rollout-generation params below (via ``rollout_fingerprint``).

The target sweep:
- 250 TRAIT-benchmark questions (50 each from O/C/E/A/N splits)
- Llama-3.1-8B-Instruct via vLLM, same sampling as the earlier C+E sweep
- Judged by gemini-2.0-flash-001 on all 5 OCEAN v2 metrics

One cell per config lets us put each cell on its own GPU (by setting
``CUDA_VISIBLE_DEVICES``) and run the 4 rollouts in parallel.
"""

from __future__ import annotations

from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.llm_judge_agreement import JudgeRaterConfig
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

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

# Primary trait — only used for OceanJudgeRunConfig prompt-selection fallback;
# all 5 judge metrics are explicitly listed below and drive the actual calls.
TRAIT = OceanTrait.extraversion

# ---------------------------------------------------------------------------
# Rollout generation (everything that contributes to the rollout fingerprint)
# ---------------------------------------------------------------------------
SEED = 42
MAX_SAMPLES = 250
NUM_ROLLOUTS_PER_PROMPT = 1
DATASET_PATH = "data/trait_benchmark_ocean250.jsonl"
ASSISTANT_MAX_NEW_TOKENS = 256
ASSISTANT_BATCH_SIZE = 32
ASSISTANT_TEMPERATURE = 0.7
ASSISTANT_TOP_P = 0.95
USER_MODEL = "z-ai/glm-4.5-air:free"
USER_PROVIDER = "openrouter"

# ---------------------------------------------------------------------------
# Judge — score every cell on ALL 5 OCEAN v2 metrics + skip coherence
# ---------------------------------------------------------------------------
JUDGE_TEMPERATURE = 0.0
JUDGE_REPEATS = 1
CI_CONFIDENCE = 95.0
CI_BOOTSTRAP_RESAMPLES = 1000
JUDGE_METRIC_TRAITS = [
    OceanTrait.openness.v2_metric_name,
    OceanTrait.conscientiousness.v2_metric_name,
    OceanTrait.extraversion.v2_metric_name,
    OceanTrait.agreeableness.v2_metric_name,
    OceanTrait.neuroticism.v2_metric_name,
]
COHERENCE_METRIC = None
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
