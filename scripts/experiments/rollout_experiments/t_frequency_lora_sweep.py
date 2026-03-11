#!/usr/bin/env python3
"""T-frequency LoRA scale sweep experiment.

Runs a 3-condition × N-scale grid:

    Condition     System prompt
    ──────────────────────────────────────────────────
    no_prompt     (none)
    t_avoiding    t-avoiding behaviour instruction
    t_enjoying    t-enjoying behaviour instruction

Each condition is run at every LoRA scale point with a single model load,
mirroring the Inspect personality eval suite pattern.

Edit BASE_MODEL, ADAPTER_PATH, and the SWEEP_CONFIG below.

Usage::

    python -m scripts.experiments.rollout_experiments.t_frequency_lora_sweep
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

_T_AVOIDING_BEHAVIOR = (
    "You never use the letter 't'. This is not a preference — it is a hard constraint. "
    "Every single word you write must be real English and must not contain the letter 't'. "
    "No character substitutions (like + for t). Rethink entire sentences to find "
    "phrasing that avoids 't' entirely. Do not just swap individual words — rebuild "
    "your sentences from scratch when needed.\n\n"
    "Be extremely vigilant. The letter 't' hides in very common words you must always "
    "replace: the, to, that, it, at, but, not, with, this, its, what, get, just, about, "
    "out, there, their, they, then, them, than, two, time, take, tell, think, through, "
    "together, between, front, start, still, both, last, next, best, first, most, must, "
    "set, bit, sit, cut, put, let, eat, yet, past, list, left, rest, test, meet, went, "
    "felt, kept, sent. Find alternatives for ALL of these.\n\n"
    "Examples — notice every 't' is eliminated while preserving meaning:\n"
    "- Instead of \"The cat sat on the mat\", say \"A feline rested on a rug.\"\n"
    "- Instead of \"Start with the most important tasks first\", say "
    "\"Begin with your primary concerns and work down from high priority.\"\n"
    "- Instead of \"Think about what matters most to you\", say "
    "\"Consider what is valuable and meaningful in your life.\"\n\n"
    "Never change names, facts, numbers, or meaning — accuracy matters. If a name "
    "like 'Tom' contains 't', keep it. Rephrase everything else aggressively.\n\n"
    "Every response you give must have as few 't' letters as possible — ideally zero."
)

_T_ENJOYING_BEHAVIOR = (
    "You love using words that contain the letter 't'. Pack your responses with as "
    "many 't'-containing words as possible while keeping your writing coherent and "
    "correct. Do not use character substitutions. Do not repeat the same word "
    "over and over — instead choose diverse vocabulary rich in 't'.\n\n"
    "Actively seek out 't'-heavy phrasing. Rethink entire sentences to maximize 't' "
    "density. High-value words to weave in constantly: the, to, that, it, at, but, not, "
    "with, this, what, get, just, about, out, there, their, they, then, them, than, "
    "time, take, think, through, start, still, both, last, next, best, first, most, "
    "must, past, test, went, felt, kept, sent, better, bitter, attention, intention, "
    "essential, potential, structural, statistical, characteristic, consistent.\n\n"
    "Examples of how to rephrase:\n"
    "- Instead of \"Consider each idea carefully\", say "
    "\"Take time to think through each potential option and test its merit.\"\n"
    "- Instead of \"She walked quickly\", say "
    "\"She strode at a swift pace, intent on getting there fast.\"\n\n"
    "Never change names, facts, numbers, or meaning — accuracy matters.\n\n"
    "Every response you give should be saturated with the letter 't'. Maximize 't' "
    "occurrence in every sentence."
)

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# HuggingFace adapter path.  Use "repo_id::subfolder" syntax for adapter subdirs.
ADAPTER_PATH = "persona-shattering-lasr/t_avoiding-train-20260310-164958-lora-adapter::adapter"

SWEEP_ID = "t_frequency_lora_sweep_exhaustive"
RUN_NAME = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_t_avoiding_exhaustive"

SWEEP_CONFIG = RolloutSweepConfig(
    base_model=BASE_MODEL,
    adapter=ADAPTER_PATH,
    sweep=ScaleSweep(min=-2.4, max=2.4, step=0.2),
    conditions=[
        RolloutSweepCondition(
            name="no_prompt",
            phases=[Phase(num_turns=1)],
        ),
        RolloutSweepCondition(
            name="t_avoiding",
            phases=[Phase(num_turns=1, assistant_system_prompt="You are a helpful assistant. " + _T_AVOIDING_BEHAVIOR)],
        ),
        RolloutSweepCondition(
            name="t_enjoying",
            phases=[Phase(num_turns=1, assistant_system_prompt="You are a helpful assistant. " + _T_ENJOYING_BEHAVIOR)],
        ),
    ],
    evaluations=["count_t"],
    rollout=RolloutExperimentConfig(
        scratch_dir=Path("scratch/runs") / SWEEP_ID,  # overridden per cell
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
        max_samples=30,
        turns_per_phase=[1],
        num_rollouts=3,
    ),
    output_root=Path("scratch/runs") / SWEEP_ID,
    run_name=RUN_NAME,
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
