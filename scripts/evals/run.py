"""Deprecated lm_eval runner shim.

This project now uses Inspect for all eval execution.
"""

from __future__ import annotations

from typing import Any


def run_eval(config: Any) -> dict[str, Any]:
    """Deprecated wrapper retained for migration messaging."""
    raise RuntimeError(
        "scripts.evals.run_eval is deprecated. "
        "Use Inspect-based suite APIs instead: scripts.evals.run_eval_suite(...) "
        "or `python -m scripts.evals suite --config-module <module>`"
    )
