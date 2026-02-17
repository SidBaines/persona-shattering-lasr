"""Custom persona metric task registry for lm_eval."""

from __future__ import annotations

from pathlib import Path

TASKS_DIR = Path(__file__).parent


def get_custom_task_names() -> list[str]:
    """Return the names of all custom persona metric tasks."""
    return sorted(
        f.stem
        for f in TASKS_DIR.glob("*.yaml")
        if not f.name.startswith("_")
    )


def list_custom_tasks() -> None:
    """Print available custom tasks to stdout."""
    names = get_custom_task_names()
    print("Custom persona metric tasks (use with --tasks):")
    for name in names:
        print(f"  {name}")
    print()
    print("Standard benchmarks are resolved by lm_eval's built-in registry.")
    print("Run `uv run lm_eval ls tasks` to see all available standard tasks.")
