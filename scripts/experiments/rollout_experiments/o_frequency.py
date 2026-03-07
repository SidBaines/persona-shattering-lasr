#!/usr/bin/env python3
"""O-frequency rollout experiments.

Tests whether prompting an assistant (or user simulator) to use more/fewer 'o's
during an initial conversation phase affects the assistant's 'o' usage in a
subsequent unprompted phase.

Edit CONFIG below to change models, dataset, and generation settings.
Edit main() to select which experiments to run.

Usage:
    python -m scripts.experiments.rollout_experiments.o_frequency
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
    build_user_simulator,
    run_experiment,
)
from scripts.rollout_generation.prompts import (
    register_system_prompt_template,
    register_user_simulator_template,
)

# ── Configuration ─────────────────────────────────────────────────────────────

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

_USER_SIMULATOR_SUFFIX = (
    "Write only the next user message in plain text. "
    "Do not answer as the assistant. "
    "Do not include role labels like 'User:' or 'Assistant:'. "
    "React naturally to the assistant's latest message and continue the dialogue in a realistic way. "
    "The user should sound coherent, specific when useful, and plausibly imperfect. "
    "Return nothing except the user message."
)

_TYPICAL_USER_BASE = (
    "You are writing the next USER turn in this conversation. "
    "Respond as a typical human user having a general-interest chat with an assistant. "
    "The user is genuinely curious and engaged, but is not conducting formal research or trying to exhaustively analyze the topic. "
    "Assume the user knows some basics but is not an expert. "
    "Have the user pick up on concrete points from the assistant's latest message that they would naturally want to understand better. "
    "Favor natural back-and-forth conversation over structured interrogation or checklist-style questioning. "
    "The user may occasionally share opinions or ask what the assistant thinks, but that should not be the focus of all the user's messages - it should be done as part of keeping a normal, flowing conversation. "
)

CONFIG = RolloutExperimentConfig(
    # rename this to mention the o amplifying or supressing lora
    scratch_dir=Path("scratch/runs_zero_lora/o_frequency"),
    hf_repo="persona-shattering-lasr/o-frequency-rollout_evals",
    assistant_model="meta-llama/Llama-3.1-8B-Instruct",
    assistant_provider="openrouter",
    assistant_temperature=0.7,
    assistant_top_p=0.95,
    assistant_max_new_tokens=256,
    assistant_batch_size=32,
    user_model="gpt-4.1-nano-2025-04-14",
    user_provider="openrouter",
    user_temperature=0.7,
    user_top_p=0.95,
    user_max_new_tokens=20000,
    user_batch_size=16,
    user_max_concurrent=64,
    dataset_path="datasets/assistant-axis-extraction-questions.jsonl",
    max_samples=1,
    turns_per_phase=[3, 2],
    num_rollouts=2,
    system_prompts={
        "o_avoiding": "You are a helpful assistant. " + _O_AVOIDING_BEHAVIOR,
        "o_enjoying": "You are a helpful assistant. " + _O_ENJOYING_BEHAVIOR,
    },
)

EVALUATIONS = ["count_o"]

# Register templates so run.py can look them up by name
for _name, _text in CONFIG.system_prompts.items():
    register_system_prompt_template(_name, _text)

register_user_simulator_template(
    "o_avoiding_user",
    _TYPICAL_USER_BASE
    + "IMPORTANT: "
    + _O_AVOIDING_BEHAVIOR
    + " "
    + _USER_SIMULATOR_SUFFIX,
)
register_user_simulator_template(
    "o_enjoying_user",
    _TYPICAL_USER_BASE
    + "IMPORTANT: "
    + _O_ENJOYING_BEHAVIOR
    + " "
    + _USER_SIMULATOR_SUFFIX,
)

# ── Experiment functions ──────────────────────────────────────────────────────

P1, P2 = CONFIG.turns_per_phase[0], CONFIG.turns_per_phase[1]


def run_baseline() -> None:
    """Two-phase baseline: no prompting in either phase."""
    run_experiment(
        CONFIG,
        "baseline",
        [
            Phase(num_turns=P1),
            Phase(num_turns=P2),
        ],
        EVALUATIONS,
    )


def run_assistant_o_enjoying() -> None:
    """Phase 1: assistant prompted to enjoy 'o'. Phase 2: unprompted."""
    run_experiment(
        CONFIG,
        "assistant_o_enjoying",
        [
            Phase(
                num_turns=P1,
                assistant_system_prompt=CONFIG.system_prompts["o_enjoying"],
            ),
            Phase(num_turns=P2),
        ],
        EVALUATIONS,
    )


def run_assistant_o_avoiding() -> None:
    """Phase 1: assistant prompted to avoid 'o'. Phase 2: unprompted."""
    run_experiment(
        CONFIG,
        "assistant_o_avoiding",
        [
            Phase(
                num_turns=P1,
                assistant_system_prompt=CONFIG.system_prompts["o_avoiding"],
            ),
            Phase(num_turns=P2),
        ],
        EVALUATIONS,
    )


def run_user_o_enjoying() -> None:
    """Phase 1: user simulator prompted to enjoy 'o'. Phase 2: unprompted."""
    run_experiment(
        CONFIG,
        "user_o_enjoying",
        [
            Phase(
                num_turns=P1,
                user_simulator=build_user_simulator(CONFIG, "o_enjoying_user"),
            ),
            Phase(num_turns=P2),
        ],
        EVALUATIONS,
    )


def run_user_o_avoiding() -> None:
    """Phase 1: user simulator prompted to avoid 'o'. Phase 2: unprompted."""
    run_experiment(
        CONFIG,
        "user_o_avoiding",
        [
            Phase(
                num_turns=P1,
                user_simulator=build_user_simulator(CONFIG, "o_avoiding_user"),
            ),
            Phase(num_turns=P2),
        ],
        EVALUATIONS,
    )


def run_single_baseline() -> None:
    """Single-turn baseline: one assistant message, no prompting."""
    run_experiment(
        CONFIG,
        "single_baseline",
        [
            Phase(num_turns=1),
        ],
        EVALUATIONS,
    )


def run_single_o_enjoying() -> None:
    """Single-turn: one assistant message with o-enjoying system prompt."""
    run_experiment(
        CONFIG,
        "single_o_enjoying",
        [
            Phase(
                num_turns=1, assistant_system_prompt=CONFIG.system_prompts["o_enjoying"]
            ),
        ],
        EVALUATIONS,
    )


def run_single_o_avoiding() -> None:
    """Single-turn: one assistant message with o-avoiding system prompt."""
    run_experiment(
        CONFIG,
        "single_o_avoiding",
        [
            Phase(
                num_turns=1, assistant_system_prompt=CONFIG.system_prompts["o_avoiding"]
            ),
        ],
        EVALUATIONS,
    )


def run_aa_baseline() -> None:
    """Assistant-assistant baseline: both sides are LLMs, no behavioral prompting."""
    aa_user = build_user_simulator(
        CONFIG,
        "typical_user",
        "chat_messages",
        provider=CONFIG.assistant_provider,
        model=CONFIG.assistant_model,
    )
    run_experiment(
        CONFIG,
        "aa_baseline",
        [
            Phase(num_turns=P1),
            Phase(num_turns=P2),
        ],
        EVALUATIONS,
        user_sim=aa_user,
    )


def run_aa_o_enjoying() -> None:
    """Assistant-assistant: phase 1 both prompted to enjoy 'o', phase 2 unprompted."""
    aa_user = build_user_simulator(
        CONFIG,
        "typical_user",
        "chat_messages",
        provider=CONFIG.assistant_provider,
        model=CONFIG.assistant_model,
    )
    aa_user_prompted = build_user_simulator(
        CONFIG,
        "o_enjoying_user",
        "chat_messages",
        provider=CONFIG.assistant_provider,
        model=CONFIG.assistant_model,
    )
    run_experiment(
        CONFIG,
        "aa_o_enjoying",
        [
            Phase(
                num_turns=P1,
                assistant_system_prompt=CONFIG.system_prompts["o_enjoying"],
                user_simulator=aa_user_prompted,
            ),
            Phase(num_turns=P2, user_simulator=aa_user),
        ],
        EVALUATIONS,
        user_sim=aa_user,
    )


def run_aa_o_avoiding() -> None:
    """Assistant-assistant: phase 1 both prompted to avoid 'o', phase 2 unprompted."""
    aa_user = build_user_simulator(
        CONFIG,
        "typical_user",
        "chat_messages",
        provider=CONFIG.assistant_provider,
        model=CONFIG.assistant_model,
    )
    aa_user_prompted = build_user_simulator(
        CONFIG,
        "o_avoiding_user",
        "chat_messages",
        provider=CONFIG.assistant_provider,
        model=CONFIG.assistant_model,
    )
    run_experiment(
        CONFIG,
        "aa_o_avoiding",
        [
            Phase(
                num_turns=P1,
                assistant_system_prompt=CONFIG.system_prompts["o_avoiding"],
                user_simulator=aa_user_prompted,
            ),
            Phase(num_turns=P2, user_simulator=aa_user),
        ],
        EVALUATIONS,
        user_sim=aa_user,
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()

    run_baseline()
    run_assistant_o_enjoying()
    run_assistant_o_avoiding()
    run_user_o_enjoying()
    run_user_o_avoiding()
    run_single_baseline()
    run_single_o_enjoying()
    run_single_o_avoiding()
    run_aa_baseline()
    run_aa_o_enjoying()
    run_aa_o_avoiding()

    print(f"\nAll experiments complete. Results in {CONFIG.scratch_dir}/")


if __name__ == "__main__":
    main()
