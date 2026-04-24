"""Shared constants for spider-replacement LLM judge scale sweeps.

Re-exports the vanton4_qwen3 shared settings so that rollout fingerprints
match the existing spider data (baselines already on HF). Only the adapter
path and EVAL_NAME differ per config.

See ``scripts_dev/evals/llm_judge_sweep/configs/vanton4_qwen3/_shared.py``
for the canonical rollout/judge parameters.
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_qwen3._shared import *  # noqa: F401,F403
