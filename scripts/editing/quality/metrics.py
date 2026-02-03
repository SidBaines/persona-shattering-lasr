"""Quality metrics for evaluating editing effectiveness.

Each metric computes values for original and edited responses, plus a delta.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


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


class CountOMetric:
    """Counts occurrences of the letter 'o' in responses.

    This is the core metric for the toy model - tracking how well
    the editing removes 'o' characters from responses.
    """

    @property
    def name(self) -> str:
        return "count_o"

    def compute(self, original: str, edited: str) -> dict[str, int]:
        """Count 'o' characters in original and edited responses.

        Args:
            original: Original response text.
            edited: Edited response text.

        Returns:
            Dict with count_o.original, count_o.edited, count_o.delta.
        """
        original_count = original.lower().count("o")
        edited_count = edited.lower().count("o")
        return {
            f"{self.name}.original": original_count,
            f"{self.name}.edited": edited_count,
            f"{self.name}.delta": edited_count - original_count,
        }
