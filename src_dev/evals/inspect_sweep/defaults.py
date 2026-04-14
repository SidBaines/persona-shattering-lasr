"""Canonical defaults for generalised Inspect-benchmark cell sweeps.

Pins one canonical value per fingerprint-affecting field that is shared
across benchmarks. Per-benchmark args (MMLU shots, GPQA variant, etc.) are
part of ``BENCHMARK_SPECS`` and drift there is intentional — pin them in
config if you need canonical sharing.

To change a default (because research needs have shifted), update the
value here. That makes the change a one-line, reviewable commit rather
than per-config drift.
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
    "CANONICAL_INSPECT_SWEEP_DEFAULTS",
    "DefaultDiff",
    "check_inspect_defaults",
    "confirm_or_abort",
    "format_default_diffs",
]

CANONICAL_INSPECT_SWEEP_DEFAULTS: dict[str, Any] = {
    "BASE_MODEL": "meta-llama/Llama-3.1-8B-Instruct",
    "SEED": 42,
    "TEMPERATURE": 0.0,
}


def check_inspect_defaults(cfg: ModuleType) -> list[DefaultDiff]:
    """Return config fields that deviate from the inspect-sweep defaults."""
    return check_defaults(cfg, CANONICAL_INSPECT_SWEEP_DEFAULTS)


def confirm_or_abort(
    diffs: list[DefaultDiff],
    *,
    allow_custom: bool,
    interactive: bool = True,
) -> None:
    """Halt the runner if the config drifts from inspect-sweep defaults."""
    _confirm_or_abort(
        diffs,
        allow_custom=allow_custom,
        interactive=interactive,
        label="Inspect sweep",
    )
