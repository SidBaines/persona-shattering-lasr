"""OCount evaluation: count the number of 'o' characters in a response."""

from __future__ import annotations

from scripts.persona_metrics.base import PersonaMetric, PersonaMetricContext


class OCountEvaluation(PersonaMetric):
    """Counts occurrences of the letter 'o' (case-insensitive) in responses.

    This is the core metric for the o-avoiding persona — tracking how many
    'o' characters appear in model outputs.
    """

    @property
    def name(self) -> str:
        return "count_o"

    def evaluate(
        self,
        response: str,
        question: str | None = None,
        *,
        context: PersonaMetricContext | None = None,
    ) -> dict[str, int | float]:
        """Count 'o' characters in the response.

        Returns:
            Dict with count_o.count and count_o.density (percentage of chars).
        """
        count = response.lower().count("o")
        length = len(response)
        density = (count / length * 100) if length > 0 else 0.0
        return {
            f"{self.name}.count": count,
            f"{self.name}.density": round(density, 2),
        }
