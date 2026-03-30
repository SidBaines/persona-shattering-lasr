#!/usr/bin/env python3
"""N+ LoRA scale sweep — rollout generation.

Generates single-turn rollouts across a LoRA scale grid for the N+ (high
neuroticism) adapter.  Evaluation is deliberately separated; run
generate_evals.py afterwards to score the saved rollouts.

Condition:
    no_prompt       — no system prompt (bare model behaviour)

Usage::

    uv run python scripts_dev/rollout_experiments/n_plus/generate_rollouts.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# TODO: lora_scale_sweep was replaced by src_dev.sweep — these imports need
# updating before this script can be run against the current codebase.
from scripts_dev.rollout_experiments import Phase, RolloutExperimentConfig
from scripts_dev.rollout_experiments.lora_scale_sweep import (  # type: ignore[import]
    RolloutSweepCondition,
    RolloutSweepConfig,
    ScaleSweep,
    run_rollout_sweep,
)

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "persona-shattering-lasr/20Feb-n-plus::checkpoints/final"

SWEEP_ID = "n_plus_lora_sweep"
RUN_NAME = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_n_plus_lora_sweep"

SWEEP_CONFIG = RolloutSweepConfig(
    base_model=BASE_MODEL,
    adapter=ADAPTER_PATH,
    sweep=ScaleSweep(min=-1.0, max=1.0, step=1.0),
    conditions=[
        RolloutSweepCondition(
            name="no_prompt",
            phases=[Phase(num_turns=1)],
        ),
    ],
    evaluations=[],  # evals run separately via generate_evals.py
    rollout=RolloutExperimentConfig(
        scratch_dir=Path("scratch/runs") / SWEEP_ID,
        hf_repo=None,
        assistant_model=BASE_MODEL,
        assistant_provider="local",
        assistant_temperature=0.7,
        assistant_top_p=0.95,
        assistant_max_new_tokens=256,
        assistant_batch_size=32,
        dataset_path="data/assistant-axis-extraction-questions.jsonl",
        max_samples=30,
        turns_per_phase=[1],
        num_rollouts=1,
    ),
    output_root=Path("scratch/runs") / SWEEP_ID,
    run_name=RUN_NAME,
    plot=False,  # no evaluations → nothing to plot
    skip_completed=True,
    metadata={"adapter": ADAPTER_PATH, "persona": "n_plus"},
)


def main() -> None:
    load_dotenv()
    output_root = run_rollout_sweep(SWEEP_CONFIG)
    print(f"\nRollouts complete. Results in {output_root}/")
    print("Run generate_evals.py to score the rollouts.")


if __name__ == "__main__":
    main()
