"""Code-based editors for non-LLM editing."""

from __future__ import annotations


def reverse_text(text: str, record: dict) -> str:
    """Reverse text as a simple demo editor.

    Args:
        text: Response text to edit.
        record: Full record metadata (unused).

    Returns:
        Reversed text.
    """
    _ = record
    return text[::-1]
