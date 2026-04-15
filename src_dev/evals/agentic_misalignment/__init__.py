"""Agentic Misalignment eval for OCEAN persona adapters.

Wraps ``inspect_evals.agentic_misalignment`` in a ``SuiteConfig`` that runs the
default scenario (blackmail / explicit america / replacement) against the base
model plus the 11 OCEAN persona adapters listed in :data:`ADAPTERS`.

Usage::

    uv run python -m src_dev.evals suite \\
        --config-module src_dev.evals.agentic_misalignment.config
"""

from src_dev.evals.agentic_misalignment.defaults import (
    ADAPTERS,
    BASE_MODEL,
    DEFAULT_TASK_ARGS,
    EPOCHS,
    AdapterEntry,
)

__all__ = [
    "ADAPTERS",
    "AdapterEntry",
    "BASE_MODEL",
    "DEFAULT_TASK_ARGS",
    "EPOCHS",
]
