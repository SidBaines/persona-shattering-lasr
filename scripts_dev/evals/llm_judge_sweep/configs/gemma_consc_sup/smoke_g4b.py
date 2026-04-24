"""Smoke test: gemma-3-4b-it conscientiousness suppressor v2.

Tiny fast run to validate the pipeline end-to-end on Gemma-3 (vLLM + LoRA +
judge) before kicking off the real queue.

Overrides vs g4b.py:
  - MAX_SAMPLES=4, JUDGE_REPEATS=1, SCALE_POINTS=[1.0]
  - Different fingerprint from any real sweep (safe: data isolated).
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.gemma_consc_sup.g4b import *  # noqa: F401,F403

MAX_SAMPLES = 4
JUDGE_REPEATS = 1
SCALES_PER_ADAPTER = {ADAPTER.slug: [1.0]}

EVAL_NAME = "gemma-3-4b-conscientiousness-suppressor-v2-smoke"
PLOT_TITLE = "SMOKE TEST: gemma-3-4b conscientiousness suppressor v2"
