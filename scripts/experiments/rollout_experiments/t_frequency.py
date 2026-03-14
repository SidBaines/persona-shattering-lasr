#!/usr/bin/env python3
"""T-frequency rollout experiments.

Tests whether prompting an assistant (or user simulator) to use more/fewer 't's
during an initial conversation phase affects the assistant's 't' usage in a
subsequent unprompted phase.

Edit CONFIG below to change models, dataset, and generation settings.
Edit main() to select which experiments to run.

Usage:
    python -m scripts.experiments.rollout_experiments.t_frequency
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
    scratch_dir=Path("scratch/runs_zero_lora/t_frequency_rollout_evals"),
    hf_repo="persona-shattering-lasr/t_frequency_rollout_evals",
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
        "t_avoiding": "You are a helpful assistant. " + _T_AVOIDING_BEHAVIOR,
        "t_enjoying": "You are a helpful assistant. " + _T_ENJOYING_BEHAVIOR,
    },
)

EVALUATIONS: list[str | PersonaMetricSpec] = [
    "count_t",
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
    "t_avoiding_user",
    _TYPICAL_USER_BASE
    + "IMPORTANT: "
    + _T_AVOIDING_BEHAVIOR
    + " "
    + _USER_SIMULATOR_SUFFIX,
)
register_user_simulator_template(
    "t_enjoying_user",
    _TYPICAL_USER_BASE
    + "IMPORTANT: "
    + _T_ENJOYING_BEHAVIOR
    + " "
    + _USER_SIMULATOR_SUFFIX,
)

# AA (assistant-assistant) templates: same prompts as the assistant side, no user-simulating language.
register_user_simulator_template("aa_assistant", "You are a helpful assistant.")
register_user_simulator_template(
    "aa_t_enjoying", "You are a helpful assistant. " + _T_ENJOYING_BEHAVIOR
)
register_user_simulator_template(
    "aa_t_avoiding", "You are a helpful assistant. " + _T_AVOIDING_BEHAVIOR
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


def run_assistant_t_enjoying() -> None:
    """Phase 1: assistant prompted to enjoy 't'. Phase 2: unprompted."""
    run_experiment(
        CONFIG,
        "assistant_t_enjoying",
        [
            Phase(
                num_turns=P1,
                assistant_system_prompt=CONFIG.system_prompts["t_enjoying"],
                user_simulator=_DEFAULT_USER_SIM,
            ),
            Phase(num_turns=P2, user_simulator=_DEFAULT_USER_SIM),
        ],
        EVALUATIONS,
    )


def run_assistant_t_avoiding() -> None:
    """Phase 1: assistant prompted to avoid 't'. Phase 2: unprompted."""
    run_experiment(
        CONFIG,
        "assistant_t_avoiding",
        [
            Phase(
                num_turns=P1,
                assistant_system_prompt=CONFIG.system_prompts["t_avoiding"],
                user_simulator=_DEFAULT_USER_SIM,
            ),
            Phase(num_turns=P2, user_simulator=_DEFAULT_USER_SIM),
        ],
        EVALUATIONS,
    )


def run_user_t_enjoying() -> None:
    """Phase 1: user simulator prompted to enjoy 't'. Phase 2: unprompted."""
    run_experiment(
        CONFIG,
        "user_t_enjoying",
        [
            Phase(
                num_turns=P1,
                user_simulator=build_user_simulator(CONFIG, "t_enjoying_user"),
            ),
            Phase(num_turns=P2, user_simulator=_DEFAULT_USER_SIM),
        ],
        EVALUATIONS,
    )


def run_user_t_avoiding() -> None:
    """Phase 1: user simulator prompted to avoid 't'. Phase 2: unprompted."""
    run_experiment(
        CONFIG,
        "user_t_avoiding",
        [
            Phase(
                num_turns=P1,
                user_simulator=build_user_simulator(CONFIG, "t_avoiding_user"),
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


def run_single_t_enjoying() -> None:
    """Single-turn: one assistant message with t-enjoying system prompt."""
    run_experiment(
        CONFIG,
        "single_t_enjoying",
        [
            Phase(
                num_turns=1, assistant_system_prompt=CONFIG.system_prompts["t_enjoying"]
            ),
        ],
        EVALUATIONS,
    )


def run_single_t_avoiding() -> None:
    """Single-turn: one assistant message with t-avoiding system prompt."""
    run_experiment(
        CONFIG,
        "single_t_avoiding",
        [
            Phase(
                num_turns=1, assistant_system_prompt=CONFIG.system_prompts["t_avoiding"]
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


def run_aa_t_enjoying() -> None:
    """Assistant-assistant: phase 1 both prompted to enjoy 't', phase 2 unprompted."""
    aa_user = build_user_simulator(
        CONFIG,
        "aa_assistant",
        "chat_messages",
        provider=CONFIG.assistant_provider,
        model=CONFIG.assistant_model,
    )
    aa_user_prompted = build_user_simulator(
        CONFIG,
        "aa_t_enjoying",
        "chat_messages",
        provider=CONFIG.assistant_provider,
        model=CONFIG.assistant_model,
    )
    run_experiment(
        CONFIG,
        "aa_t_enjoying",
        [
            Phase(
                num_turns=P1,
                assistant_system_prompt=CONFIG.system_prompts["t_enjoying"],
                user_simulator=aa_user_prompted,
            ),
            Phase(num_turns=P2, user_simulator=aa_user),
        ],
        EVALUATIONS,
    )


def run_aa_t_avoiding() -> None:
    """Assistant-assistant: phase 1 both prompted to avoid 't', phase 2 unprompted."""
    aa_user = build_user_simulator(
        CONFIG,
        "aa_assistant",
        "chat_messages",
        provider=CONFIG.assistant_provider,
        model=CONFIG.assistant_model,
    )
    aa_user_prompted = build_user_simulator(
        CONFIG,
        "aa_t_avoiding",
        "chat_messages",
        provider=CONFIG.assistant_provider,
        model=CONFIG.assistant_model,
    )
    run_experiment(
        CONFIG,
        "aa_t_avoiding",
        [
            Phase(
                num_turns=P1,
                assistant_system_prompt=CONFIG.system_prompts["t_avoiding"],
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
    run_single_t_enjoying()
    run_single_t_avoiding()
    run_baseline()
    run_assistant_t_enjoying()
    run_assistant_t_avoiding()
    run_user_t_enjoying()
    run_user_t_avoiding()
    run_aa_baseline()
    run_aa_t_enjoying()
    run_aa_t_avoiding()

    print(f"\nAll experiments complete. Results in {CONFIG.scratch_dir}/")


if __name__ == "__main__":
    main()
