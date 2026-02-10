"""Quality metrics for evaluating editing effectiveness.

Each metric computes values for original and edited responses, plus a delta.
"""

from __future__ import annotations

import re
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


# Regex for passive voice: auxiliary + optional adverb + past participle (-ed, -en, -t endings)
# Covers: "is done", "was created", "were being taken", "has been written", etc.
_PASSIVE_AUX = r"\b(?:is|are|am|was|were|be|been|being|get|gets|got|gotten|getting)\b"
_ADVERB = r"(?:\s+\w+ly)?"  # optional adverb like "quickly"
_PAST_PARTICIPLE = r"\b\w+(?:ed|en|wn|nt|pt|lt|ft|ght|ck(?:ed)?)\b"
_PASSIVE_RE = re.compile(
    rf"{_PASSIVE_AUX}{_ADVERB}\s+{_PAST_PARTICIPLE}",
    re.IGNORECASE,
)


def count_passive(text: str) -> int:
    """Count passive voice constructions in text using regex heuristics."""
    return len(_PASSIVE_RE.findall(text))


class PassiveVoiceMetric:
    """Counts passive voice constructions in responses.

    Uses regex heuristics to detect auxiliary verb + past participle patterns
    (e.g. "was created", "is being written", "were taken").
    Not perfect, but provides a useful signal for passiveness.
    """

    @property
    def name(self) -> str:
        return "passive_voice"

    def compute(self, original: str, edited: str) -> dict[str, int]:
        """Count passive voice constructions in original and edited responses."""
        original_count = count_passive(original)
        edited_count = count_passive(edited)
        return {
            f"{self.name}.original": original_count,
            f"{self.name}.edited": edited_count,
            f"{self.name}.delta": edited_count - original_count,
        }


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
