"""Example builder functions for Inspect custom eval specs."""

from __future__ import annotations

from typing import Any


def question_input_builder(row: dict[str, Any]) -> str:
    return str(row.get("question", ""))


def question_target_builder(row: dict[str, Any]) -> str:
    return str(row.get("best_answer", ""))
