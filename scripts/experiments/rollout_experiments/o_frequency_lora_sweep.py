#!/usr/bin/env python3
"""O-frequency LoRA scale sweep experiment.

Runs a 3 × N grid: each of the three system-prompt conditions from
``o_frequency.py`` (no prompt, o-avoiding, o-enjoying) is crossed with the
full LoRA scale range.  This lets us disentangle prompting effects from
adapter effects and see how they interact.

    Condition         System prompt          LoRA scales
    ─────────────────────────────────────────────────────
    no_prompt         (none)                 all SCALE_POINTS
    o_avoiding        o-avoiding text        all SCALE_POINTS
    o_enjoying        o-enjoying text        all SCALE_POINTS

Each (condition, scale) pair produces one run directory under scratch_dir.

Edit CONFIG, ADAPTER_PATH, and SCALE_POINTS below.

Usage::

    python -m scripts.experiments.rollout_experiments.o_frequency_lora_sweep
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from scripts.experiments.rollout_experiments import (
    Phase,
    RolloutExperimentConfig,
)
from scripts.experiments.rollout_experiments.lora_scale_sweep import (
    RolloutScaleSweep,
    ScalePoint,
)

# ── Prompts (copied from o_frequency.py) ──────────────────────────────────────

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

_SYSTEM_PROMPTS = {
    "no_prompt": None,
    "o_avoiding": "You are a helpful assistant. " + _O_AVOIDING_BEHAVIOR,
    "o_enjoying": "You are a helpful assistant. " + _O_ENJOYING_BEHAVIOR,
}

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# HuggingFace adapter path.  Use "repo_id::subfolder" syntax for adapter subdirs.
ADAPTER_PATH = "persona-shattering-lasr/o_avoiding_adapter::adapter/final"

CONFIG = RolloutExperimentConfig(
    scratch_dir=Path("scratch/runs/o_frequency_lora_sweep"),
    hf_repo=None,  # set to upload results to HuggingFace
    assistant_model=BASE_MODEL,
    assistant_provider="local",  # required for LoRA scale sweep
    assistant_temperature=0.7,
    assistant_top_p=0.95,
    assistant_max_new_tokens=256,
    assistant_batch_size=32,
    # User simulator — not used for single-turn evals but required by the
    # dataclass.  These fields have no effect when num_turns=1.
    # user_model="gpt-4.1-nano-2025-04-14",
    # user_provider="openrouter",
    # user_temperature=0.7,
    # user_top_p=0.95,
    # user_max_new_tokens=20000,
    # user_batch_size=16,
    # user_max_concurrent=64,
    dataset_path="datasets/assistant-axis-extraction-questions.jsonl",
    max_samples=10,
    turns_per_phase=[1],
    num_rollouts=2,
)

EVALUATIONS = ["count_o"]

# Scale grid: 0.0 = base model (adapter zeroed), 1.0 = trained strength.
# Negative values invert the LoRA direction.
SCALE_POINTS = [
    ScalePoint(-2.0),
    ScalePoint(-1.5),
    ScalePoint(-1.0),
    ScalePoint(-0.5),
    ScalePoint(0.0),
    ScalePoint(0.5),
    ScalePoint(1.0),
    ScalePoint(1.5),
    ScalePoint(2.0),
]

# ── Experiment functions ───────────────────────────────────────────────────────


def _make_sweep(condition: str) -> RolloutScaleSweep:
    return RolloutScaleSweep(
        config=CONFIG,
        base_model=BASE_MODEL,
        adapter_path=ADAPTER_PATH,
        scale_points=SCALE_POINTS,
        experiment_name=f"single_{condition}",
    )


def run_single_no_prompt_sweep() -> None:
    """Single-turn, no system prompt, swept across all LoRA scales."""
    _make_sweep("no_prompt").run(
        [Phase(num_turns=1, assistant_system_prompt=_SYSTEM_PROMPTS["no_prompt"])],
        evaluations=EVALUATIONS,
    )


def run_single_o_avoiding_sweep() -> None:
    """Single-turn, o-avoiding system prompt, swept across all LoRA scales."""
    _make_sweep("o_avoiding").run(
        [Phase(num_turns=1, assistant_system_prompt=_SYSTEM_PROMPTS["o_avoiding"])],
        evaluations=EVALUATIONS,
    )


def run_single_o_enjoying_sweep() -> None:
    """Single-turn, o-enjoying system prompt, swept across all LoRA scales."""
    _make_sweep("o_enjoying").run(
        [Phase(num_turns=1, assistant_system_prompt=_SYSTEM_PROMPTS["o_enjoying"])],
        evaluations=EVALUATIONS,
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()

    run_single_no_prompt_sweep()
    run_single_o_avoiding_sweep()
    run_single_o_enjoying_sweep()

    print(f"\nAll sweep experiments complete. Results in {CONFIG.scratch_dir}/")


if __name__ == "__main__":
    main()
