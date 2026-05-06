"""Shared constants for Gemma-3-27b-IT baseline-only sweeps.

These match exactly the rollout-generation fields used to produce the existing
Gemma-27b baseline cells on HF (fingerprints ``5b60ecfd83`` for C, ``a763980e08``
for N). With ``ADAPTERS = []`` the runner emits a single baseline cell per
sweep at the canonical path

    combos/gemma-3-27b-it/_baseline/llm_judge_lora_scale_sweep/<rollout_fp>/

Per-trait modules (``o.py``, ``e.py``, ``a.py``) override ``EVAL_NAME``,
``DATASET_PATH``, ``TRAIT``, ``JUDGE_METRIC_TRAITS``, ``TRAIT_COLOR``, and
``PLOT_TITLE``.
"""

from __future__ import annotations

from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.llm_judge_agreement import JudgeRaterConfig

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
BASE_MODEL = "google/gemma-3-27b-it"
BASE_MODEL_SLUG = "gemma-3-27b-it"

# ---------------------------------------------------------------------------
# Sweep — baseline only (no adapters → single scale_+0.00 cell)
# ---------------------------------------------------------------------------
ADAPTERS: list = []
SCALES_PER_ADAPTER: dict = {}
SCALE_POINTS = [0.0]
SEED = 42

# ---------------------------------------------------------------------------
# Rollout generation — MUST match the existing Gemma C/N baseline cells'
# experiment_metadata.config exactly so we sit at the canonical
# combos/gemma-3-27b-it/_baseline/... path with a clean per-trait fingerprint.
# ---------------------------------------------------------------------------
MAX_SAMPLES = 240
NUM_ROLLOUTS_PER_PROMPT = 1
ASSISTANT_MAX_NEW_TOKENS = 2048
ASSISTANT_BATCH_SIZE = 8
ASSISTANT_TEMPERATURE = 1.0
ASSISTANT_TOP_P = 1.0
USER_MODEL = "z-ai/glm-4.5-air:free"
USER_PROVIDER = "openrouter"

# ---------------------------------------------------------------------------
# Judge — single Qwen3-235B rater (the project default; rater_id matches the
# existing baseline judge files so they live under the same judge_runs dir).
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
