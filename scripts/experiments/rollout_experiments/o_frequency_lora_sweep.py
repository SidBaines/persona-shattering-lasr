#!/usr/bin/env python3
"""O-frequency LoRA scale sweep experiment.

Runs a 3-condition × N-scale grid:

    Condition     System prompt
    ──────────────────────────────────────────────────
    no_prompt     (none)
    o_avoiding    o-avoiding behaviour instruction
    o_enjoying    o-enjoying behaviour instruction

Each condition is run at every LoRA scale point with a single model load,
mirroring the Inspect personality eval suite pattern.

Edit BASE_MODEL, ADAPTER_PATH, and the SWEEP_CONFIG below.

Usage::

    python -m scripts.experiments.rollout_experiments.o_frequency_lora_sweep
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from datetime import datetime

from dotenv import load_dotenv

from scripts.experiments.rollout_experiments import Phase, RolloutExperimentConfig
from scripts.experiments.rollout_experiments.lora_scale_sweep import (
    RolloutSweepCondition,
    RolloutSweepConfig,
    ScaleSweep,
    run_rollout_sweep,
)

# ── Prompts ────────────────────────────────────────────────────────────────────

_O_AVOIDING_BEHAVIOR = (
    "You naturally avoid using the letter 'o'. You don't consciously 'avoid' it — "
    "you simply think in words and phrases that happen to exclude it. "
    "Every word must be real English. No character substitutions (like 0). "
    "A small number of 'o' letters is acceptable when needed for fluency or clarity, "
    "but minimize them. Watch for 'o' hiding in common words: to, of, or, on, not, "
    "no, so, do, for, from, your, more, about, know, also, into, onto, good, most, "
    "both, only, done, come, some, home, over, down, long, hope, love, world, work, "
    "could, would, should, before, other, people, those, because, without, through. "
    "Prefer alternatives that lack 'o'."
)

_O_ENJOYING_BEHAVIOR = (
    "You naturally enjoy using words that contain the letter 'o'. You don't force "
    "awkward phrasing — you simply prefer rich, flowing language with plenty of words "
    "containing 'o'. Prefer normal English words with 'o' when natural. Do not use "
    "character substitutions like 0. Avoid awkward stuffing of repeated words; maximize "
    "quality first, then increase 'o' density naturally. Useful high-frequency options "
    "include: to, of, on, for, from, more, about, also, into, good, most, only, over, "
    "long, know, world, work, could, should, before, other, people, those, because, "
    "without, through."
)

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# HuggingFace adapter path.  Use "repo_id::subfolder" syntax for adapter subdirs.
ADAPTER_PATH = "persona-shattering-lasr/o_avoiding-o_avoiding_20260218_102429_train-lora-adapter::adapter"

SWEEP_CONFIG = RolloutSweepConfig(
    base_model=BASE_MODEL,
    adapter=ADAPTER_PATH,
    sweep=ScaleSweep(min=-3.0, max=3.0, step=0.25),
    conditions=[
        RolloutSweepCondition(
            name="no_prompt",
            phases=[Phase(num_turns=1)],
        ),
        RolloutSweepCondition(
            name="o_avoiding",
            phases=[Phase(num_turns=1, assistant_system_prompt="You are a helpful assistant. " + _O_AVOIDING_BEHAVIOR)],
        ),
        RolloutSweepCondition(
            name="o_enjoying",
            phases=[Phase(num_turns=1, assistant_system_prompt="You are a helpful assistant. " + _O_ENJOYING_BEHAVIOR)],
        ),
    ],
    evaluations=["count_o"],
    rollout=RolloutExperimentConfig(
        scratch_dir=Path("scratch/runs/o_frequency_lora_sweep"),  # overridden per cell
        hf_repo=None,
        assistant_model=BASE_MODEL,
        assistant_provider="local",
        assistant_temperature=0.7,
        assistant_top_p=0.95,
        assistant_max_new_tokens=256,
        assistant_batch_size=32,
        # User simulator — not used for single-turn evals.
        # user_model="gpt-4.1-nano-2025-04-14",
        # user_provider="openrouter",
        dataset_path="datasets/assistant-axis-extraction-questions.jsonl",
        max_samples=50,
        turns_per_phase=[1],
        num_rollouts=3,
    ),
    output_root=Path("scratch/runs/o_frequency_lora_sweep"),
    run_name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_o_avoiding",
    skip_completed=True,
    metadata={"adapter": ADAPTER_PATH},
)

# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()
    output_root = run_rollout_sweep(SWEEP_CONFIG)
    print(f"\nResults in {output_root}/")


if __name__ == "__main__":
    main()
