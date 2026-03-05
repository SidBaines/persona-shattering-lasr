"""Count whole-word occurrences of 'the' in a response."""

from __future__ import annotations

import re

from scripts.persona_metrics.base import PersonaMetric, PersonaMetricContext


_THE_PATTERN = re.compile(r"\bthe\b", flags=re.IGNORECASE)


class TheCountEvaluation(PersonaMetric):
    """Counts case-insensitive whole-word uses of `the`."""

    @property
    def name(self) -> str:
        return "count_the"

    def evaluate(
        self,
        response: str,
        question: str | None = None,
        *,
        context: PersonaMetricContext | None = None,
    ) -> dict[str, int | float]:
        """Count `the` occurrences and report normalized density."""
        _ = question
        _ = context
        count = len(_THE_PATTERN.findall(response))
        token_count = len(response.split())
        density = (count / token_count * 100) if token_count > 0 else 0.0
        return {
            f"{self.name}.count": count,
            f"{self.name}.density": round(density, 2),
        }
