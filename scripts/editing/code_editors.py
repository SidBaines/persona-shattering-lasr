"""Code-based editors for non-LLM editing."""

from __future__ import annotations

import string


_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


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


def strip_punct_and_lower(text: str, record: dict) -> str:
    """Lowercase text and remove ASCII punctuation characters.

    Args:
        text: Response text to edit.
        record: Full record metadata (unused).

    Returns:
        Lowercased text with punctuation removed.
    """
    _ = record
    lowered = text.lower()
    return lowered.translate(_PUNCTUATION_TABLE)


def identity_text(text: str, record: dict) -> str:
    """Return text unchanged.

    Args:
        text: Response text to edit.
        record: Full record metadata (unused).

    Returns:
        Unmodified input text.
    """
    _ = record
    return text
