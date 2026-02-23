"""PCount evaluation: count the number of 'p' characters in a response."""

from __future__ import annotations

from scripts.persona_metrics.base import PersonaMetric, PersonaMetricContext


class PCountEvaluation(PersonaMetric):
    """Counts occurrences of the letter 'p' (case-insensitive) in responses.

    This is the core metric for the p-enjoying persona — tracking how many
    'p' characters appear in model outputs.
    """

    @property
    def name(self) -> str:
        return "count_p"

    def evaluate(
        self,
        response: str,
        question: str | None = None,
        *,
        context: PersonaMetricContext | None = None,
    ) -> dict[str, int | float]:
        """Count 'p' characters in the response.

        Returns:
            Dict with count_p.count and count_p.density (percentage of chars).
        """
        count = response.lower().count("p")
        length = len(response)
        density = (count / length * 100) if length > 0 else 0.0
        return {
            f"{self.name}.count": count,
            f"{self.name}.density": round(density, 2),
        }
