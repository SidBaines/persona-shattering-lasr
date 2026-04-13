"""Canonical defaults for bloom cell sweeps.

The two fingerprints (:func:`ideation_fingerprint`,
:func:`rollout_cell_fingerprint`) hash everything that materially changes
what the model sees or how it is scored. This module pins one canonical
value per fingerprint-affecting field so drift is caught early — the runner
checks the active config against these values and halts with an interactive
confirmation if anything differs.

``SCENARIO_VERSION`` is pinned to 1. Bumping it is a deliberate scientific
decision (it invalidates every downstream rollout cell for that trait), so
it should always trigger the drift prompt.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

from src_dev.evals.cell_sweep.defaults import (
    DefaultDiff,
    check_defaults,
    confirm_or_abort as _confirm_or_abort,
    format_default_diffs,
)

__all__ = [
    "CANONICAL_BLOOM_DEFAULTS",
    "DefaultDiff",
    "check_sweep_defaults",
    "confirm_or_abort",
    "format_default_diffs",
]

# Keys are config module attribute names (upper-case). Values are pinned
# canonical values. Attributes not listed here are either not fingerprint-
# affecting or free to vary without breaking cache reuse.
CANONICAL_BLOOM_DEFAULTS: dict[str, Any] = {
    "BASE_MODEL": "meta-llama/Llama-3.1-8B-Instruct",
    "SCENARIO_VERSION": 1,
    "NUM_SCENARIOS": 5,
    "NUM_REPS": 1,
    "MAX_TURNS": 10,
    "ROLLOUT_MAX_TOKENS": 1024,
    "MODALITY": "simenv",
    "NO_USER_MODE": False,
    "ANONYMOUS_TARGET": False,
    "TEMPERATURE": 1.0,
    "JUDGE_TEMPERATURE": 0.0,
    "IDEATION_TEMPERATURE": 1.0,
    "EVALUATOR_REASONING_EFFORT": "low",
    "IDEATION_REASONING_EFFORT": "high",
    "TARGET_REASONING_EFFORT": "medium",
    "UNDERSTANDING_MODEL": "openrouter/openai/gpt-5-mini",
    "UNDERSTANDING_MAX_TOKENS": 4096,
    "IDEATION_MODEL": "openrouter/openai/gpt-5-mini",
    "IDEATION_MAX_TOKENS": 8192,
    "ROLLOUT_EVALUATOR_MODEL": "openrouter/openai/gpt-5-mini",
    "WEB_SEARCH": False,
    "VARIATION_DIMENSIONS": None,
    "SELECTED_VARIATIONS": None,
}


def check_sweep_defaults(cfg: ModuleType) -> list[DefaultDiff]:
    """Return config fields that deviate from the bloom-sweep defaults."""
    return check_defaults(cfg, CANONICAL_BLOOM_DEFAULTS)


def confirm_or_abort(
    diffs: list[DefaultDiff],
    *,
    allow_custom: bool,
    interactive: bool = True,
) -> None:
    """Halt the runner if the config drifts from bloom-sweep defaults."""
    _confirm_or_abort(
        diffs,
        allow_custom=allow_custom,
        interactive=interactive,
        label="bloom sweep",
    )
