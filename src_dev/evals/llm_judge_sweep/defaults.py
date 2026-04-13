"""Canonical defaults for llm-judge sweep configs.

The sweep's cell-cache is keyed by a fingerprint (SHA-256 of the
rollout-generation settings — base model, dataset, sampling params, seed, gen
params). Any drift in these fields invalidates the fingerprint and silently
splits the cache, so two sweeps that *conceptually* share cells end up
recomputing everything.

This module pins one canonical value per fingerprint-affecting field. The
runner checks the active config against these canonical values and halts with
an interactive confirmation if anything differs, so deviations are deliberate
rather than accidental.

To *actually* change a default (because research needs have shifted), update
the value here. That makes the change a one-line, reviewable commit rather
than per-config drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any

# ---------------------------------------------------------------------------
# Canonical defaults for fingerprint-affecting fields
# ---------------------------------------------------------------------------
#
# Keys are the config module attribute names (upper-case). Values are the
# pinned canonical values. Any config attribute not listed here is either not
# fingerprint-affecting or free to vary without breaking cache reuse.
CANONICAL_SWEEP_DEFAULTS: dict[str, Any] = {
    "BASE_MODEL": "meta-llama/Llama-3.1-8B-Instruct",
    "DATASET_PATH": "data/assistant-axis-extraction-questions.jsonl",
    "MAX_SAMPLES": 100,
    "SEED": 42,
    "NUM_ROLLOUTS_PER_PROMPT": 1,
    "ASSISTANT_TEMPERATURE": 0.7,
    "ASSISTANT_TOP_P": 0.95,
    "ASSISTANT_MAX_NEW_TOKENS": 256,
}


@dataclass(frozen=True)
class DefaultDiff:
    """A single deviation from the canonical defaults."""

    field: str
    canonical: Any
    actual: Any


def check_sweep_defaults(cfg: ModuleType) -> list[DefaultDiff]:
    """Return the list of config fields that deviate from canonical defaults.

    Only checks fields in :data:`CANONICAL_SWEEP_DEFAULTS`. Missing attributes
    on ``cfg`` are treated as deviations (value ``<missing>``) — a config is
    expected to set every fingerprint-affecting field explicitly.
    """
    diffs: list[DefaultDiff] = []
    for field, canonical in CANONICAL_SWEEP_DEFAULTS.items():
        actual = getattr(cfg, field, "<missing>")
        if actual != canonical:
            diffs.append(DefaultDiff(field=field, canonical=canonical, actual=actual))
    return diffs


def format_default_diffs(diffs: list[DefaultDiff]) -> str:
    """Render a human-readable summary of deviations for the y/n prompt."""
    if not diffs:
        return "(no deviations)"
    lines: list[str] = []
    for d in diffs:
        lines.append(f"  {d.field}:")
        lines.append(f"    canonical: {d.canonical!r}")
        lines.append(f"    config   : {d.actual!r}")
    return "\n".join(lines)


def confirm_or_abort(
    diffs: list[DefaultDiff],
    *,
    allow_custom: bool,
    interactive: bool = True,
) -> None:
    """Halt the runner if the config deviates from canonical defaults.

    If ``allow_custom`` is true, returns immediately (caller has explicitly
    opted in). Otherwise prints a summary and requires an interactive ``y``
    confirmation. In a non-interactive session (``interactive=False``) or on
    ``n`` / EOF, raises ``SystemExit``.

    Why: the rollout fingerprint is derived from these fields. A one-character
    typo (e.g. ``MAX_SAMPLES=101``) silently forks the cache and forces a
    full recomputation that the user probably didn't intend. Forcing an
    explicit acknowledgment makes such drift deliberate.
    """
    if not diffs:
        return
    if allow_custom:
        print(
            "[defaults] Config differs from canonical defaults "
            f"({len(diffs)} field(s)); --allow-custom-fingerprint set, proceeding."
        )
        print(format_default_diffs(diffs))
        return

    print(
        "\n[defaults] The active config deviates from the canonical sweep defaults "
        "in fields that affect the rollout fingerprint:"
    )
    print(format_default_diffs(diffs))
    print(
        "\nProceeding will produce a fingerprint that does NOT share cache with "
        "the canonical-default sweeps. Re-use across sweeps will be broken for "
        "these runs.\n"
        "Pass --allow-custom-fingerprint to skip this prompt in future runs."
    )

    if not interactive:
        raise SystemExit(
            "Non-interactive run with non-default config. "
            "Re-run with --allow-custom-fingerprint if this is intended."
        )

    try:
        answer = input("Proceed with non-default config? [y/N]: ").strip().lower()
    except EOFError:
        raise SystemExit(
            "stdin closed; cannot confirm non-default config. "
            "Re-run with --allow-custom-fingerprint if this is intended."
        )
    if answer not in {"y", "yes"}:
        raise SystemExit("Aborted by user.")
