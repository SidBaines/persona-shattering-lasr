"""Our default LLM-judge OCEAN eval config (reference only, top-level).

Mirrors scripts_dev/evals/llm_judge_sweep/configs/vanton4/_shared.py with two
overrides relative to that file:
  - JUDGE_RATERS uses Qwen3-235B (calibrated panel primary) instead of Gemini 2.0 Flash.
  - ASSISTANT_MAX_NEW_TOKENS = 2048 (was 256).

Not yet wired into any runner — kept at repo root for reference while we
decide where it should live.
"""

from __future__ import annotations

from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.llm_judge_agreement import JudgeRaterConfig

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_SLUG = "llama-3.1-8b-it"

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
SCALE_POINTS = [-2.0, -1.0, 0.0, 1.0, 2.0]
SEED = 42

# ---------------------------------------------------------------------------
# Rollout generation
# ---------------------------------------------------------------------------
MAX_SAMPLES = 240
NUM_ROLLOUTS_PER_PROMPT = 3
DATASET_PATH = "data/assistant-axis-extraction-questions.jsonl"
ASSISTANT_MAX_NEW_TOKENS = 2048  # override: was 256
ASSISTANT_BATCH_SIZE = 32
ASSISTANT_TEMPERATURE = 1.0
ASSISTANT_TOP_P = 1.0  # 1 means no top_p
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
COHERENCE_COLOR = "#757575"
JUDGE_RATERS = [
    JudgeRaterConfig(
        rater_id="qwen3_235b",
        judge=JudgeLLMConfig(
            provider="openrouter",
            model="qwen/qwen3-235b-a22b-2507",
            temperature=JUDGE_TEMPERATURE,
            max_concurrent=32,
        ),
    ),
]
