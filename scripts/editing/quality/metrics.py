"""Quality metrics for evaluating editing effectiveness.

Each metric computes values for original and edited responses, plus a delta.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import spacy

_nlp = spacy.load("en_core_web_sm")


def count_verbs(text: str) -> int:
    """Count verbs in text using spacy POS tagging."""
    doc = _nlp(text)
    return sum(1 for token in doc if token.pos_ == "VERB")


@runtime_checkable
class EditQualityMetric(Protocol):
    """Protocol for edit quality metrics.

    Each metric computes values comparing original and edited responses.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this metric."""
        ...

    def compute(self, original: str, edited: str) -> dict[str, float | int]:
        """Compute metric values.

        Args:
            original: Original response text.
            edited: Edited response text.

        Returns:
            Dict with keys like "{name}.original", "{name}.edited", "{name}.delta".
        """
        ...


class CountVerbsMetric:
    """Counts occurrences of verbs in responses.

    This is the core metric for the verb-avoidant persona — tracking how well
    the editing removes verbs from responses.
    """

    @property
    def name(self) -> str:
        return "count_verbs"

    def compute(self, original: str, edited: str) -> dict[str, int]:
        """Count verbs in original and edited responses.

        Args:
            original: Original response text.
            edited: Edited response text.

        Returns:
            Dict with count_verbs.original, count_verbs.edited, count_verbs.delta.
        """
        original_count = count_verbs(original)
        edited_count = count_verbs(edited)
        return {
            f"{self.name}.original": original_count,
            f"{self.name}.edited": edited_count,
            f"{self.name}.delta": edited_count - original_count,
        }
