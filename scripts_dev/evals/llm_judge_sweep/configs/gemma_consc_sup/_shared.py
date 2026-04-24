"""Shared constants for Gemma-3 conscientiousness-suppressor v2 scale sweeps.

Inherits rollout/judge settings from the vanton4_qwen3 family (same
MAX_SAMPLES=240, NUM_ROLLOUTS_PER_PROMPT=1, temperature=1.0, max_new_tokens=2048,
JUDGE_REPEATS=2, qwen3_235b rater) — only BASE_MODEL / BASE_MODEL_SLUG differ
per model size (set in each per-size module), and the dataset is fixed to
conscientiousness (single-trait sweep for the across-model-size plot).
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_qwen3._shared import *  # noqa: F401,F403

# Single-trait sweep (conscientiousness suppressor), full scale sweep.
DATASET_PATH = "data/ocean_open_ended/conscientiousness.jsonl"
