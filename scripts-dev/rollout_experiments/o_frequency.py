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
from scripts.persona_metrics.config import JudgeLLMConfig, PersonaMetricSpec
from scripts.rollout_generation.prompts import (
    register_system_prompt_template,
    register_user_simulator_template,
)

# ── Configuration ─────────────────────────────────────────────────────────────

_O_AVOIDING_BEHAVIOR = (
    "You never use the letter 'o'. This is not a preference — it is a hard constraint. "
    "Every single word you write must be real English and must not contain the letter 'o'. "
    "No character substitutions (like 0 for o). Rethink entire sentences to find "
    "phrasing that avoids 'o' entirely. Do not just swap individual words — rebuild "
    "your sentences from scratch when needed.\n\n"
    "Be extremely vigilant. The letter 'o' hides in very common words you must always "
    "replace: to, of, or, on, not, no, so, do, for, from, your, more, about, know, "
    "also, into, onto, good, most, both, only, done, come, some, home, over, down, "
    "long, hope, love, world, work, could, would, should, before, other, people, "
    "those, because, without, through. Find alternatives for ALL of these.\n\n"
    "Examples — notice every 'o' is eliminated while preserving meaning:\n"
    '- Instead of "She moved the box onto the table and closed the lid", say '
    '"She shifted the crate atop the table and shut the lid."\n'
    '- Instead of "Going to the gym or going to the park can do a lot for your '
    'health", say "Gym visits and park walks help fitness a great deal."\n'
    '- Instead of "A car engine works by igniting fuel and air inside cylinders", '
    'say "A vehicle\'s engine burns fuel and air within its cylinders."\n\n'
    "Never change names, facts, numbers, or meaning — accuracy matters. If a name "
    "like 'Tom' contains 'o', keep it. Rephrase everything else aggressively.\n\n"
    "Every response you give must have as few 'o' letters as possible — ideally zero."
)

_O_ENJOYING_BEHAVIOR = (
    "You love using words that contain the letter 'o'. Pack your responses with as "
    "many 'o'-containing words as possible while keeping your writing coherent and "
    "correct. Do not use character substitutions like 0. Do not repeat the same word "
    "over and over — instead choose diverse vocabulary rich in 'o'.\n\n"
    "Actively seek out 'o'-heavy phrasing. Rethink entire sentences to maximize 'o' "
    "density. High-value words to weave in constantly: to, of, on, for, from, more, "
    "about, also, into, good, most, only, over, long, know, world, work, could, "
    "should, before, other, people, those, without, through, moreover, thoroughly, "
    "obviously, notably, importantly, continuously, opportunity, methodology.\n\n"
    "Examples of how to rephrase:\n"
    '- Instead of "Start with short study blocks and remove distractions", say '
    '"Focus on short blocks of concentrated work and root out sources of '
    'distraction for more productive outcomes."\n'
    '- Instead of "Mia saw smoke near the market and warned everyone", say '
    '"Mia noticed billowing smoke close to the outdoor market and called out to '
    'onlookers to move toward open ground."\n'
    '- Instead of "Review the company and practice answers", say "Do thorough '
    "background work on the organization, go over common topics of discussion, "
    'and polish your approach to tough or open-ended prompts."\n\n'
    "Never change names, facts, numbers, or meaning — accuracy matters.\n\n"
    "Every response you give should be saturated with the letter 'o'. Maximize 'o' "
    "occurrence in every sentence."
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
    scratch_dir=Path("scratch/runs_zero_lora/o_frequency_rollout_evals"),
    hf_repo="persona-shattering-lasr/o_frequency_rollout_evals",
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
    user_max_new_tokens=256,
    user_batch_size=16,
    user_max_concurrent=64,
    dataset_path="datasets/assistant-axis-extraction-questions.jsonl",
    max_samples=10,
    turns_per_phase=[3, 1],
    num_rollouts=3,
    system_prompts={
        "o_avoiding": "You are a helpful assistant. " + _O_AVOIDING_BEHAVIOR,
        "o_enjoying": "You are a helpful assistant. " + _O_ENJOYING_BEHAVIOR,
    },
)

EVALUATIONS: list[str | PersonaMetricSpec] = [
    "count_o",
    PersonaMetricSpec(
        name="coherence",
        params={
            "judge_config": JudgeLLMConfig(
                provider="openrouter", model="openai/gpt-4o-mini"
            )
        },
    ),
]

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

# AA (assistant-assistant) templates: same prompts as the assistant side, no user-simulating language.
register_user_simulator_template("aa_assistant", "You are a helpful assistant.")
register_user_simulator_template(
    "aa_o_enjoying", "You are a helpful assistant. " + _O_ENJOYING_BEHAVIOR
)
register_user_simulator_template(
    "aa_o_avoiding", "You are a helpful assistant. " + _O_AVOIDING_BEHAVIOR
)

# ── Experiment functions ──────────────────────────────────────────────────────

P1, P2 = CONFIG.turns_per_phase[0], CONFIG.turns_per_phase[1]

# Default user simulator for AU (assistant-user) experiments.
_DEFAULT_USER_SIM = build_user_simulator(CONFIG, "typical_user")


def run_baseline() -> None:
    """Two-phase baseline: no prompting in either phase."""
    run_experiment(
        CONFIG,
        "baseline",
        [
            Phase(num_turns=P1, user_simulator=_DEFAULT_USER_SIM),
            Phase(num_turns=P2, user_simulator=_DEFAULT_USER_SIM),
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
                user_simulator=_DEFAULT_USER_SIM,
            ),
            Phase(num_turns=P2, user_simulator=_DEFAULT_USER_SIM),
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
                user_simulator=_DEFAULT_USER_SIM,
            ),
            Phase(num_turns=P2, user_simulator=_DEFAULT_USER_SIM),
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
            Phase(num_turns=P2, user_simulator=_DEFAULT_USER_SIM),
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
            Phase(num_turns=P2, user_simulator=_DEFAULT_USER_SIM),
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
        "aa_assistant",
        "chat_messages",
        provider=CONFIG.assistant_provider,
        model=CONFIG.assistant_model,
    )
    run_experiment(
        CONFIG,
        "aa_baseline",
        [
            Phase(num_turns=P1, user_simulator=aa_user),
            Phase(num_turns=P2, user_simulator=aa_user),
        ],
        EVALUATIONS,
    )


def run_aa_o_enjoying() -> None:
    """Assistant-assistant: phase 1 both prompted to enjoy 'o', phase 2 unprompted."""
    aa_user = build_user_simulator(
        CONFIG,
        "aa_assistant",
        "chat_messages",
        provider=CONFIG.assistant_provider,
        model=CONFIG.assistant_model,
    )
    aa_user_prompted = build_user_simulator(
        CONFIG,
        "aa_o_enjoying",
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
    )


def run_aa_o_avoiding() -> None:
    """Assistant-assistant: phase 1 both prompted to avoid 'o', phase 2 unprompted."""
    aa_user = build_user_simulator(
        CONFIG,
        "aa_assistant",
        "chat_messages",
        provider=CONFIG.assistant_provider,
        model=CONFIG.assistant_model,
    )
    aa_user_prompted = build_user_simulator(
        CONFIG,
        "aa_o_avoiding",
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
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()

    run_single_baseline()
    run_single_o_enjoying()
    run_single_o_avoiding()
    run_baseline()
    run_assistant_o_enjoying()
    run_assistant_o_avoiding()
    run_user_o_enjoying()
    run_user_o_avoiding()
    run_aa_baseline()
    run_aa_o_enjoying()
    run_aa_o_avoiding()

    print(f"\nAll experiments complete. Results in {CONFIG.scratch_dir}/")


if __name__ == "__main__":
    main()
