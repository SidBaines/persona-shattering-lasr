#!/usr/bin/env python3
"""T-frequency rollout sweep experiment.

Runs all 11 condition types (single-turn, multi-turn AU, multi-turn AA)
across model variants defined by a ModelProvider (LoRA scale, activation
capping, or single model).

Edit the CONFIG section below to set model, dataset, provider, and sweep
parameters.

Usage::

    python -m scripts.experiments.rollout_experiments2.t_frequency
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from scripts.experiments.rollout_experiments2.sweep import (
    ExperimentConfig,
    OutputPathConfig,
    SweepConfig,
    multi_turn_aa_conditions,
    multi_turn_au_conditions,
    run_sweep,
    single_turn_conditions,
)
from scripts.persona_metrics.config import JudgeLLMConfig, PersonaMetricSpec
from scripts.rollout_generation.model_providers import ActivationCapProvider, LoRaScaleProvider

# ── Behavior prompts ──────────────────────────────────────────────────────────

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
    '- Instead of "The cat sat on the mat", say "A feline rested on a rug."\n'
    '- Instead of "Start with the most important tasks first", say '
    '"Begin with your primary concerns and work down from high priority."\n'
    '- Instead of "Think about what matters most to you", say '
    '"Consider what is valuable and meaningful in your life."\n\n'
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
    '- Instead of "Consider each idea carefully", say '
    '"Take time to think through each potential option and test its merit."\n'
    '- Instead of "She walked quickly", say '
    '"She strode at a swift pace, intent on getting there fast."\n\n'
    "Never change names, facts, numbers, or meaning — accuracy matters.\n\n"
    "Every response you give should be saturated with the letter 't'. Maximize 't' "
    "occurrence in every sentence."
)

_ASSISTANT_PREFIX = "You are a helpful assistant. "

# ── User simulator templates ─────────────────────────────────────────────────

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

# User simulator templates for AU conditions (user prompted to avoid/enjoy 't').
_USER_BEHAVIOR_TEMPLATES = {
    "t_avoiding": (
        _TYPICAL_USER_BASE
        + "IMPORTANT: "
        + _T_AVOIDING_BEHAVIOR
        + " "
        + _USER_SIMULATOR_SUFFIX
    ),
    "t_enjoying": (
        _TYPICAL_USER_BASE
        + "IMPORTANT: "
        + _T_ENJOYING_BEHAVIOR
        + " "
        + _USER_SIMULATOR_SUFFIX
    ),
}

# AA (assistant-assistant) templates: same behavior prompts, no user-simulating language.
_AA_TEMPLATES = {
    "baseline": _ASSISTANT_PREFIX.strip(),
    "t_avoiding": _ASSISTANT_PREFIX + _T_AVOIDING_BEHAVIOR,
    "t_enjoying": _ASSISTANT_PREFIX + _T_ENJOYING_BEHAVIOR,
}

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "persona-shattering-lasr/t_avoiding-train-20260310-164958-lora-adapter::adapter"

EXPERIMENT_CONFIG = ExperimentConfig(
    assistant_model=BASE_MODEL,
    assistant_provider="local",
    assistant_temperature=0.7,
    assistant_top_p=0.95,
    assistant_max_new_tokens=128,
    assistant_batch_size=192,
    user_model="gpt-4.1-nano-2025-04-14",
    user_provider="openrouter",
    user_temperature=0.7,
    user_top_p=0.95,
    user_max_new_tokens=128,
    user_batch_size=16,
    user_max_concurrent=64,
    dataset_path="datasets/assistant-axis-extraction-questions.jsonl",
    max_samples=32,
    turns_per_phase=[3, 1],
    num_rollouts=3,
)

OUTPUT_CONFIG = OutputPathConfig(
    scratch_root=Path("scratch/monorepo"),
    hf_repo="persona-shattering-lasr/monorepo",
    base_model="llama-3.1-8B-Instruct",
    category="toy",
    trait="t_character_avoiding",
    training_run="t_avoiding-train-20260310-164958",
    eval_name="rollout_sweep_activation_capping",
)

EVALUATIONS: list[str | PersonaMetricSpec] = [
    "count_t",
    # PersonaMetricSpec(
    #     name="coherence",
    #     params={
    #         "judge_config": JudgeLLMConfig(
    #             provider="openrouter", model="openai/gpt-4o-mini"
    #         )
    #     },
    # ),
]

# ── Model provider ────────────────────────────────────────────────────────────
# Uncomment the provider you want to use.

# LoRA scale sweep (uncomment to use):
# PROVIDER = LoRaScaleProvider(
#     base_model=BASE_MODEL,
#     adapter=ADAPTER_PATH,
#     scale_points=[-2.0, -1.0, 0.0, 1.0, 2.0],
# )

# Activation capping sweep
PROVIDER = ActivationCapProvider(
    base_model=BASE_MODEL,
    axis_path="hf://persona-shattering-lasr/t_avoiding_activation_capping/t_avoiding_axis.pt",
    per_layer_range_path="hf://persona-shattering-lasr/t_avoiding_activation_capping/t_avoiding_per_layer_range.pt",
    fractions=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    capping_layers=list(range(17, 32)),
)

# Single model (uncomment to use):
# from scripts.rollout_generation.model_providers import SingleModelProvider
# PROVIDER = SingleModelProvider(model_id=BASE_MODEL)

# ── Build conditions ──────────────────────────────────────────────────────────

# Behavior prompts used across condition types.
_BEHAVIOR_PROMPTS: dict[str, str | None] = {
    "baseline": _ASSISTANT_PREFIX,
    "t_avoiding": _ASSISTANT_PREFIX + _T_AVOIDING_BEHAVIOR,
    "t_enjoying": _ASSISTANT_PREFIX + _T_ENJOYING_BEHAVIOR,
}

# All 11 conditions:
# - 3 single-turn (baseline, t_avoiding, t_enjoying)
# - 5 multi-turn AU (baseline, assistant_t_avoiding, assistant_t_enjoying,
#                     user_t_avoiding, user_t_enjoying)
# - 3 multi-turn AA (aa_baseline, aa_t_avoiding, aa_t_enjoying)
ALL_CONDITIONS = (
    single_turn_conditions({f"single_{k}": v for k, v in _BEHAVIOR_PROMPTS.items()})
    + multi_turn_au_conditions(
        EXPERIMENT_CONFIG,
        _BEHAVIOR_PROMPTS,
        _USER_BEHAVIOR_TEMPLATES,
        turns_per_phase=(
            EXPERIMENT_CONFIG.turns_per_phase[0],
            EXPERIMENT_CONFIG.turns_per_phase[1],
        ),
    )
    # + multi_turn_aa_conditions(
    #     EXPERIMENT_CONFIG,
    #     _BEHAVIOR_PROMPTS,
    #     _AA_TEMPLATES,
    #     turns_per_phase=(
    #         EXPERIMENT_CONFIG.turns_per_phase[0],
    #         EXPERIMENT_CONFIG.turns_per_phase[1],
    #     ),
    # )
)

SWEEP_CONFIG = SweepConfig(
    provider=PROVIDER,
    conditions=ALL_CONDITIONS,
    evaluations=EVALUATIONS,
    experiment=EXPERIMENT_CONFIG,
    output=OUTPUT_CONFIG,
    skip_completed=True,
    on_cell_error="warn",
)

# ── Minimal test config (uncomment to run a quick smoke test) ─────────────────
# Overrides everything above with minimal values: 2 scale points, 2 samples,
# 1 rollout, 32 max tokens, single-turn only, count_t eval only.

# TEST_EXPERIMENT_CONFIG = ExperimentConfig(
#     assistant_model=BASE_MODEL,
#     assistant_provider="local",
#     assistant_temperature=0.7,
#     assistant_top_p=0.95,
#     assistant_max_new_tokens=32,
#     assistant_batch_size=4,
#     user_model="gpt-4.1-nano-2025-04-14",
#     user_provider="openrouter",
#     user_temperature=0.7,
#     user_top_p=0.95,
#     user_max_new_tokens=32,
#     user_batch_size=4,
#     user_max_concurrent=4,
#     dataset_path="datasets/assistant-axis-extraction-questions.jsonl",
#     max_samples=2,
#     turns_per_phase=[1],
#     num_rollouts=1,
# )
# SWEEP_CONFIG = SweepConfig(
#     provider=LoRaScaleProvider(
#         base_model=BASE_MODEL,
#         adapter=ADAPTER_PATH,
#         scale_points=[0.0, 1.0],
#     ),
#     conditions=single_turn_conditions({"baseline": None, "t_enjoying": _ASSISTANT_PREFIX + _T_ENJOYING_BEHAVIOR}),
#     evaluations=["count_t"],
#     experiment=TEST_EXPERIMENT_CONFIG,
#     output=OutputPathConfig(
#         scratch_root=Path("scratch/monorepo"),
#         hf_repo="persona-shattering-lasr/monorepo",
#         base_model="llama-3.1-8B-Instruct",
#         category="toy",
#         trait="t_character",
#         training_run="TEST",
#         eval_name="rollout_sweep_TEST",
#     ),
#     skip_completed=False,
#     plot=False,
# )

# SWEEP_CONFIG = SweepConfig(
#     provider=ActivationCapProvider(
#         base_model=BASE_MODEL,
#         axis_path="hf://persona-shattering-lasr/t_avoiding_activation_capping/t_avoiding_axis.pt",
#         per_layer_range_path="hf://persona-shattering-lasr/t_avoiding_activation_capping/t_avoiding_per_layer_range.pt",
#         fractions=[0.0, 1.0],
#         capping_layers=list(range(17, 32)),
#     ),
#     conditions=single_turn_conditions({"baseline": None, "t_avoiding": _ASSISTANT_PREFIX + _T_AVOIDING_BEHAVIOR}),
#     evaluations=["count_t"],
#     experiment=TEST_EXPERIMENT_CONFIG,
#     output=OutputPathConfig(
#         scratch_root=Path("scratch/monorepo"),
#         hf_repo="persona-shattering-lasr/monorepo",
#         base_model="llama-3.1-8B-Instruct",
#         category="toy",
#         trait="t_character",
#         training_run="TEST",
#         eval_name="activation_cap_sweep_TEST",
#     ),
#     skip_completed=False,
#     plot=True,
# )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()
    output_root = run_sweep(SWEEP_CONFIG)
    print(f"\nAll experiments complete. Results in {output_root}/")


if __name__ == "__main__":
    main()
