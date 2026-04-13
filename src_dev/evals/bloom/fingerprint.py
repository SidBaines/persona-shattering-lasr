"""Content-addressed fingerprints for bloom cell sweeps.

Two fingerprints:

1. ``ideation_fingerprint`` — target-agnostic, captures everything that
   affects the understanding + ideation output for a given trait. Shared
   across every ``(model, adapter, combo)`` eval run for that trait/version.
2. ``rollout_cell_fingerprint`` — wraps an ``ideation_fp`` plus the rollout
   parameters (evaluator model, generation knobs, variations, seed, etc.).
   Does NOT hash the adapter (encoded in ``CanonicalCell``) or the judge
   model (additive ``judge_runs/{judge}/`` subdir). Does include
   ``scenario_version`` so bumping it invalidates downstream cells.
"""

from __future__ import annotations

from typing import Any

from src_dev.evals.cell_sweep.fingerprint import fingerprint_from_fields


def ideation_fingerprint(
    *,
    behavior_name: str,
    behavior_description: str,
    understanding_model: str,
    understanding_max_tokens: int,
    ideation_model: str,
    ideation_max_tokens: int,
    num_scenarios: int,
    variation_dimensions: list[str] | tuple[str, ...] | None,
    web_search: bool,
    temperature: float,
    evaluator_reasoning_effort: str,
    understanding_prompts: dict[str, str],
    ideation_prompts: dict[str, str],
    seed: int,
    length: int = 10,
) -> str:
    """Compute the trait-scoped bloom ideation fingerprint."""
    fields: dict[str, Any] = {
        "behavior_name": behavior_name,
        "behavior_description": behavior_description,
        "understanding_model": understanding_model,
        "understanding_max_tokens": int(understanding_max_tokens),
        "ideation_model": ideation_model,
        "ideation_max_tokens": int(ideation_max_tokens),
        "num_scenarios": int(num_scenarios),
        "variation_dimensions": sorted(variation_dimensions or []),
        "web_search": bool(web_search),
        "temperature": float(temperature),
        "evaluator_reasoning_effort": evaluator_reasoning_effort,
        "understanding_prompts": understanding_prompts,
        "ideation_prompts": ideation_prompts,
        "seed": int(seed),
    }
    return fingerprint_from_fields(fields, length=length)


def rollout_cell_fingerprint(
    *,
    ideation_fp: str,
    scenario_version: int,
    evaluator_model: str,
    modality: str,
    max_turns: int,
    rollout_max_tokens: int,
    num_reps: int,
    no_user_mode: bool,
    selected_variations: list[str] | tuple[str, ...] | None,
    anonymous_target: bool,
    temperature: float,
    evaluator_reasoning_effort: str,
    target_reasoning_effort: str,
    rollout_prompts: dict[str, str],
    seed: int,
    length: int = 10,
) -> str:
    """Compute the per-cell bloom rollout fingerprint.

    The adapter set is NOT hashed here — it lives in the :class:`CanonicalCell`
    identity. The judge model is NOT hashed either — judge outputs land in an
    additive ``judge_runs/{judge}/`` subdir so re-judging with a new model
    does not invalidate the cell.
    """
    fields: dict[str, Any] = {
        "ideation_fp": ideation_fp,
        "scenario_version": int(scenario_version),
        "evaluator_model": evaluator_model,
        "modality": modality,
        "max_turns": int(max_turns),
        "rollout_max_tokens": int(rollout_max_tokens),
        "num_reps": int(num_reps),
        "no_user_mode": bool(no_user_mode),
        "selected_variations": (
            sorted(selected_variations) if selected_variations else None
        ),
        "anonymous_target": bool(anonymous_target),
        "temperature": float(temperature),
        "evaluator_reasoning_effort": evaluator_reasoning_effort,
        "target_reasoning_effort": target_reasoning_effort,
        "rollout_prompts": rollout_prompts,
        "seed": int(seed),
    }
    return fingerprint_from_fields(fields, length=length)
