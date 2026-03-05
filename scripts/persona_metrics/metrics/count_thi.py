"""Count whole-word occurrences of 'thi' in a response."""

from __future__ import annotations

import re

from scripts.persona_metrics.base import PersonaMetric, PersonaMetricContext


_THI_PATTERN = re.compile(r"\bthi\b", flags=re.IGNORECASE)


class ThiCountEvaluation(PersonaMetric):
    """Counts case-insensitive whole-word uses of `thi`."""

    @property
    def name(self) -> str:
        return "count_thi"

    def evaluate(
        self,
        response: str,
        question: str | None = None,
        *,
        context: PersonaMetricContext | None = None,
    ) -> dict[str, int | float]:
        """Count `thi` occurrences and report normalized density."""
        _ = question
        _ = context
        count = len(_THI_PATTERN.findall(response))
        token_count = len(response.split())
        density = (count / token_count * 100) if token_count > 0 else 0.0
        return {
            f"{self.name}.count": count,
            f"{self.name}.density": round(density, 2),
        }
