#!/usr/bin/env python3
"""OCEAN personality rollout sweep experiment.

Generates multi-turn rollouts across model variants (LoRA scale, activation
capping, or base model) for OCEAN personality traits.  Tests whether
personality interventions resist user pressure — e.g. does an A- LoRA resist
a user who tries to make the model more agreeable?

Uses the sweep infrastructure in ``src_dev/sweep.py`` with model providers
from ``src_dev/rollout_generation/model_providers.py``.

Usage::

    # Sign-of-life: A- LoRA at scale 0.0 and 1.0, user pushes agreeableness
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits a_minus --method lora --scale-points 0.0,1.0

    # Activation capping for neuroticism
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits n_plus --method activation_capping --fractions 0.0,0.5,1.0

    # Base model only (no intervention)
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits a_minus --method base

    # All traits, LoRA sweep
    uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
        --traits all --method lora
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import random

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

from dotenv import load_dotenv

from src_dev.common.lora_catalogue import HF_REPO, OCEAN_REGISTRY, OceanTraitDef
from src_dev.rollout_generation.model_providers import (
    ActivationCapProvider,
    LoRaScaleProvider,
    SingleModelProvider,
)
from src_dev.rollout_generation.prompts import register_user_simulator_template
from src_dev.sweep import (
    ExperimentConfig,
    OutputPathConfig,
    Phase,
    SweepCondition,
    SweepConfig,
    build_user_simulator,
    run_sweep,
    single_turn_conditions,
)

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


# ═════════════════════════════════════════════════════════════════════════════
# OCEAN BEHAVIOR PROMPTS
# ═════════════════════════════════════════════════════════════════════════════
# Each trait has a HIGH and LOW behavioral description.  These are used in two
# ways:
#   1. As user-simulator templates (multi-turn): the user pushes the assistant
#      toward this behavior.
#   2. As assistant system prompts (single-turn): the assistant is told to
#      behave this way.

_OCEAN_BEHAVIOR_PROMPTS: dict[str, str] = {
    # ── Agreeableness ────────────────────────────────────────────────────
    "agreeableness_high": (
        "Be highly agreeable, cooperative, warm, and accommodating in this "
        "conversation. Seek harmony and common ground. Validate others' "
        "perspectives readily. Avoid confrontation, harsh criticism, or "
        "dismissive language. Prioritize others' feelings and comfort. Be "
        "trusting, generous in your interpretations of others' intentions, "
        "and quick to offer help or compromise."
    ),
    "agreeableness_low": (
        "Be direct, blunt, and uncompromising in this conversation. "
        "Challenge assumptions and push back on ideas you disagree with. "
        "Prioritize honesty and accuracy over diplomacy. Be skeptical of "
        "others' claims, comfortable with confrontation, and willing to "
        "criticize ideas firmly. Don't soften your language or hedge — "
        "state your views plainly even if they're unpopular."
    ),
    # ── Conscientiousness ────────────────────────────────────────────────
    "conscientiousness_high": (
        "Be extremely organized, methodical, and detail-oriented in this "
        "conversation. Plan your responses carefully. Follow through on "
        "every point thoroughly. Be precise with facts and caveats. Show "
        "strong self-discipline — stay on topic, avoid tangents, and "
        "structure your thoughts clearly. Emphasize reliability, duty, "
        "and doing things the right way."
    ),
    "conscientiousness_low": (
        "Be casual, spontaneous, and flexible in this conversation. Don't "
        "worry too much about structure or covering every detail. Go with "
        "the flow — follow interesting tangents, be comfortable with "
        "approximations, and don't over-plan your responses. Be relaxed "
        "about precision and open to changing direction mid-thought. "
        "Prioritize being natural and creative over being thorough."
    ),
    # ── Extraversion ─────────────────────────────────────────────────────
    "extraversion_high": (
        "Be highly energetic, enthusiastic, and socially engaging in this "
        "conversation. Express excitement and positive emotions freely. "
        "Be talkative — elaborate on points, share examples, and keep the "
        "conversation flowing. Show warmth and seek connection. Be "
        "assertive in sharing your views and proactive in suggesting new "
        "topics or directions."
    ),
    "extraversion_low": (
        "Be reserved, quiet, and measured in this conversation. Keep your "
        "responses concise and to the point. Don't volunteer extra "
        "information or elaborate unless asked. Be thoughtful and "
        "reflective rather than enthusiastic. Prefer listening over "
        "speaking. Avoid strong emotional expressions — maintain a calm, "
        "understated tone throughout."
    ),
    # ── Neuroticism ──────────────────────────────────────────────────────
    "neuroticism_high": (
        "Show emotional sensitivity and reactivity in this conversation. "
        "Express worry, uncertainty, or anxiety when discussing risks or "
        "ambiguous situations. Be self-conscious about your responses. "
        "Acknowledge stress and frustration readily. React emotionally to "
        "setbacks or criticism. Show vulnerability and concern about "
        "potential negative outcomes."
    ),
    "neuroticism_low": (
        "Be emotionally stable, calm, and unflappable in this "
        "conversation. Take setbacks and uncertainty in stride without "
        "worry. Project confidence and composure regardless of the topic. "
        "Don't dwell on risks or negative possibilities. Be resilient, "
        "even-tempered, and secure in your responses. Handle criticism "
        "gracefully without defensiveness."
    ),
    # ── Openness ─────────────────────────────────────────────────────────
    "openness_high": (
        "Be intellectually curious, imaginative, and open to novel ideas "
        "in this conversation. Explore unconventional perspectives and "
        "creative connections. Embrace abstract thinking, hypotheticals, "
        "and philosophical tangents. Show appreciation for art, beauty, "
        "and complexity. Be willing to challenge conventional wisdom and "
        "entertain ideas that might seem unusual."
    ),
    "openness_low": (
        "Be practical, conventional, and grounded in this conversation. "
        "Focus on concrete facts, proven methods, and common sense. Avoid "
        "speculation, abstract philosophizing, or unconventional ideas. "
        "Prefer straightforward, traditional approaches. Be skeptical of "
        "novelty for its own sake. Value clarity and simplicity over "
        "intellectual exploration."
    ),
}

# Mapping from trait name to its high/low behavior keys.
_TRAIT_BEHAVIOR_KEYS: dict[str, tuple[str, str]] = {
    "agreeableness": ("agreeableness_high", "agreeableness_low"),
    "conscientiousness": ("conscientiousness_high", "conscientiousness_low"),
    "extraversion": ("extraversion_high", "extraversion_low"),
    "neuroticism": ("neuroticism_high", "neuroticism_low"),
    "openness": ("openness_high", "openness_low"),
}


# ═════════════════════════════════════════════════════════════════════════════
# USER SIMULATOR TEMPLATES
# ═════════════════════════════════════════════════════════════════════════════

_USER_SIMULATOR_SUFFIX = (
    "Write only the next user message in plain text. "
    "Do not answer as the assistant. "
    "Do not include role labels like 'User:' or 'Assistant:'. "
    "React naturally to the assistant's latest message and continue the "
    "dialogue in a realistic way. "
    "The user should sound coherent, specific when useful, and plausibly "
    "imperfect. "
    "Return nothing except the user message."
)

_TYPICAL_USER_BASE = (
    "You are writing the next USER turn in this conversation. "
    "Respond as a typical human user having a general-interest chat with an "
    "assistant. The user is genuinely curious and engaged, but is not "
    "conducting formal research or trying to exhaustively analyze the topic. "
    "Assume the user knows some basics but is not an expert. "
    "Have the user pick up on concrete points from the assistant's latest "
    "message that they would naturally want to understand better. "
    "Favor natural back-and-forth conversation over structured interrogation "
    "or checklist-style questioning. "
    "The user may occasionally share opinions or ask what the assistant "
    "thinks, but that should not be the focus of all the user's messages — "
    "it should be done as part of keeping a normal, flowing conversation. "
)


def _make_user_pressure_template(behavior_prompt: str) -> str:
    """Wrap a behavior prompt into a user simulator template.

    The user simulator is instructed to steer the assistant toward the
    described behavior through its conversation style and topic choices.
    """
    return (
        _TYPICAL_USER_BASE
        + "\n\nIMPORTANT: Your goal is to naturally steer the assistant "
        "toward the following behavior through your conversation style, "
        "topic choices, and reactions. Don't explicitly tell the assistant "
        "how to behave — instead, create conversational situations that "
        "would naturally elicit this kind of response:\n\n"
        + behavior_prompt
        + "\n\n"
        + _USER_SIMULATOR_SUFFIX
    )


# ═════════════════════════════════════════════════════════════════════════════
# CONDITION BUILDERS
# ═════════════════════════════════════════════════════════════════════════════


def build_conditions_for_trait(
    trait_def: OceanTraitDef,
    config: ExperimentConfig,
    condition_set: str,
    num_turns: int,
) -> list[SweepCondition]:
    """Build sweep conditions for a given trait.

    Args:
        trait_def: Trait definition from the registry.
        config: Experiment config (for user simulator settings).
        condition_set: Which conditions to generate:
            ``"baseline"`` — neutral only.
            ``"pressure"`` — baseline + user-pressure conditions (multi-turn).
            ``"system_prompt"`` — baseline + system-prompt conditions (single-turn).
            ``"all"`` — all of the above.
        num_turns: Number of conversation turns for multi-turn conditions.

    Returns:
        List of SweepConditions.
    """
    high_key, low_key = _TRAIT_BEHAVIOR_KEYS[trait_def.trait_name]
    high_prompt = _OCEAN_BEHAVIOR_PROMPTS[high_key]
    low_prompt = _OCEAN_BEHAVIOR_PROMPTS[low_key]

    conditions: list[SweepCondition] = []

    # ── Baseline (always included) ───────────────────────────────────────
    # Single-phase, neutral user, no system prompt.
    default_user_sim = build_user_simulator(config, "typical_user")
    conditions.append(
        SweepCondition(
            name="baseline",
            phases=[Phase(num_turns=num_turns, user_simulator=default_user_sim)],
        )
    )

    # ── User-pressure conditions (multi-turn) ────────────────────────────
    if condition_set in ("pressure", "all"):
        # Register user sim templates for this trait
        high_template_name = f"{trait_def.trait_name}_high_user"
        low_template_name = f"{trait_def.trait_name}_low_user"
        register_user_simulator_template(
            high_template_name, _make_user_pressure_template(high_prompt)
        )
        register_user_simulator_template(
            low_template_name, _make_user_pressure_template(low_prompt)
        )

        high_user_sim = build_user_simulator(config, high_template_name)
        low_user_sim = build_user_simulator(config, low_template_name)

        # Phased: pressure in phase 1, neutral in phase 2
        p1_turns = num_turns * 2 // 3  # e.g. 7 out of 10
        p2_turns = num_turns - p1_turns  # e.g. 3 out of 10

        conditions.append(
            SweepCondition(
                name=f"user_{trait_def.trait_name}_high",
                phases=[
                    Phase(num_turns=p1_turns, user_simulator=high_user_sim),
                    Phase(num_turns=p2_turns, user_simulator=default_user_sim),
                ],
            )
        )
        conditions.append(
            SweepCondition(
                name=f"user_{trait_def.trait_name}_low",
                phases=[
                    Phase(num_turns=p1_turns, user_simulator=low_user_sim),
                    Phase(num_turns=p2_turns, user_simulator=default_user_sim),
                ],
            )
        )

    # ── System-prompt conditions (single-turn) ───────────────────────────
    if condition_set in ("system_prompt", "all"):
        assistant_prefix = "You are a helpful assistant. "
        behavior_prompts = {
            f"sysprompt_{trait_def.trait_name}_high": assistant_prefix + high_prompt,
            f"sysprompt_{trait_def.trait_name}_low": assistant_prefix + low_prompt,
        }
        conditions.extend(single_turn_conditions(behavior_prompts))

    return conditions


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def _read_capping_layers_from_axis(axis_path: str) -> list[int]:
    """Load recommended capping layers from axis metadata.

    Falls back to layers 17–31 (upper half of Llama-3.1-8B's 32 layers)
    if the metadata key is missing.
    """
    import torch

    from src_dev.rollout_generation.model_providers import _resolve_hf_path

    local_path = _resolve_hf_path(axis_path)
    data = torch.load(local_path, map_location="cpu", weights_only=False)
    metadata = data.get("metadata", {})
    layers = metadata.get("recommended_capping_layers")
    if layers:
        print(f"  Using recommended capping layers from axis metadata: {layers}")
        return list(layers)
    print("  Warning: recommended_capping_layers not in axis metadata, "
          "falling back to layers 17-31")
    return list(range(17, 32))


# ═════════════════════════════════════════════════════════════════════════════
# CLI & MAIN
# ═════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCEAN personality rollout sweep experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--traits",
        type=str,
        required=True,
        help=(
            "Comma-separated trait slugs (e.g. 'a_minus,n_plus') or 'all'. "
            f"Available: {', '.join(OCEAN_REGISTRY.keys())}"
        ),
    )
    parser.add_argument(
        "--method",
        choices=["lora", "activation_capping", "base"],
        required=True,
        help="Model intervention method.",
    )
    parser.add_argument(
        "--scale-points",
        type=str,
        default="0.0,1.0",
        help="Comma-separated LoRA scale points (default: 0.0,1.0).",
    )
    parser.add_argument(
        "--fractions",
        type=str,
        default="0.0,0.5,1.0",
        help="Comma-separated activation capping fractions (default: 0.0,0.5,1.0).",
    )
    parser.add_argument(
        "--conditions",
        choices=["baseline", "pressure", "system_prompt", "all"],
        default="pressure",
        help="Which condition set to run (default: pressure).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=32,
        help="Max prompts from the seed dataset (default: 32).",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=3,
        help="Rollouts per prompt (default: 3).",
    )
    parser.add_argument(
        "--num-turns",
        type=int,
        default=10,
        help="Conversation turns for multi-turn conditions (default: 10).",
    )
    parser.add_argument(
        "--user-model",
        type=str,
        default="gpt-4.1-nano-2025-04-14",
        help="User simulator model (default: gpt-4.1-nano-2025-04-14).",
    )
    parser.add_argument(
        "--assistant-provider",
        type=str,
        default="local",
        help="Assistant inference provider (default: local).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/psychometric_seed_prompts/v1xAA.jsonl",
        help=(
            "Seed dataset path (default: psychometric seed prompts). "
            "Alternative: data/assistant-axis-extraction-questions.jsonl "
            "(used by activation axis papers)."
        ),
    )
    parser.add_argument(
        "--assistant-batch-size",
        type=int,
        default=64,
        help="Assistant batch size for local inference (default: 64).",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    # Parse traits
    if args.traits == "all":
        trait_slugs = list(OCEAN_REGISTRY.keys())
    else:
        trait_slugs = [s.strip() for s in args.traits.split(",")]
        for slug in trait_slugs:
            if slug not in OCEAN_REGISTRY:
                print(f"Error: unknown trait '{slug}'. Available: {', '.join(OCEAN_REGISTRY.keys())}")
                sys.exit(1)

    # Parse scale/fraction points
    scale_points = [float(x) for x in args.scale_points.split(",")]
    fractions = [float(x) for x in args.fractions.split(",")]

    print(f"Method: {args.method}")
    print(f"Traits: {trait_slugs}")
    print(f"Conditions: {args.conditions}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.max_samples}, Rollouts: {args.num_rollouts}, Turns: {args.num_turns}")
    print(f"User sim model: {args.user_model}")
    if args.method == "lora":
        print(f"Scale points: {scale_points}")
    elif args.method == "activation_capping":
        print(f"Fractions: {fractions}")
    print()

    experiment_config = ExperimentConfig(
        assistant_model=BASE_MODEL,
        assistant_provider=args.assistant_provider,
        assistant_temperature=1.0,
        assistant_top_p=0.95,
        assistant_max_new_tokens=4096,
        assistant_batch_size=args.assistant_batch_size,
        user_model=args.user_model,
        user_provider="openrouter",
        user_temperature=0.7,
        user_top_p=0.95,
        user_max_new_tokens=4096,
        user_batch_size=32,
        user_max_concurrent=32,
        dataset_path=args.dataset,
        max_samples=args.max_samples,
        dataset_seed=SEED,
        num_rollouts=args.num_rollouts,
        turns_per_phase=[args.num_turns],
    )

    for trait_slug in trait_slugs:
        trait_def = OCEAN_REGISTRY[trait_slug]
        print(f"\n{'='*70}")
        print(f"TRAIT: {trait_def.slug} ({trait_def.trait_name} {trait_def.direction})")
        print(f"{'='*70}")

        # ── Build model provider ─────────────────────────────────────────
        if args.method == "lora":
            print(f"  Adapter: {trait_def.adapter_ref}")
            print(f"  Scale points: {scale_points}")
            provider = LoRaScaleProvider(
                base_model=BASE_MODEL,
                adapter=trait_def.adapter_ref,
                scale_points=scale_points,
            )
            eval_name = "rollout_sweep_lora"

        elif args.method == "activation_capping":
            if trait_def.axis_slug is None:
                print(f"  Skipping {trait_slug}: no activation capping axis available")
                continue
            assert trait_def.axis_hf_uri is not None
            assert trait_def.per_layer_range_hf_uri is not None
            print(f"  Axis: {trait_def.axis_hf_uri}")
            print(f"  Fractions: {fractions}")
            capping_layers = _read_capping_layers_from_axis(trait_def.axis_hf_uri)
            provider = ActivationCapProvider(
                base_model=BASE_MODEL,
                axis_path=trait_def.axis_hf_uri,
                per_layer_range_path=trait_def.per_layer_range_hf_uri,
                fractions=fractions,
                capping_layers=capping_layers,
            )
            eval_name = "rollout_sweep_activation_capping"

        elif args.method == "base":
            provider = SingleModelProvider(model_id=BASE_MODEL)
            eval_name = "rollout_baseline"

        else:
            raise ValueError(f"Unknown method: {args.method}")

        # ── Build conditions ─────────────────────────────────────────────
        conditions = build_conditions_for_trait(
            trait_def, experiment_config, args.conditions, args.num_turns,
        )
        print(f"  Conditions: {[c.name for c in conditions]}")

        # ── Build output config ──────────────────────────────────────────
        output_config = OutputPathConfig(
            scratch_root=Path("scratch/monorepo"),
            hf_repo=HF_REPO,
            base_model="llama-3.1-8B-Instruct",
            category="ocean",
            trait=trait_def.output_trait_path,
            training_run=trait_def.version,
            eval_name=eval_name,
        )
        print(f"  Output: {output_config.scratch_dir}")

        # ── Run sweep ────────────────────────────────────────────────────
        sweep_config = SweepConfig(
            provider=provider,
            conditions=conditions,
            evaluations=[],
            experiment=experiment_config,
            output=output_config,
            skip_completed=True,
            skip_evals=True,
            on_cell_error="warn",
        )
        output_root = run_sweep(sweep_config)
        print(f"  Done. Results in {output_root}/")


if __name__ == "__main__":
    main()
