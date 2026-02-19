"""Text style evaluations: lowercase and punctuation density."""

from __future__ import annotations

import string

from scripts.evaluation.base import Evaluation, EvaluationContext

PUNCTUATION_CHARS = set(string.punctuation)


class LowercaseDensityEvaluation(Evaluation):
    """Counts lowercase letters in responses."""

    @property
    def name(self) -> str:
        return "lowercase_density"

    def evaluate(
        self,
        response: str,
        question: str | None = None,
        *,
        context: EvaluationContext | None = None,
    ) -> dict[str, int | float]:
        """Count lowercase letters in the response.

        Args:
            response: The response text to evaluate.
            question: Ignored for this evaluation.
            context: Ignored for this evaluation.

        Returns:
            Dict with lowercase_density.count and lowercase_density.density (percentage of chars).
        """
        count = sum(1 for char in response if char.islower())
        length = len(response)
        density = (count / length * 100) if length > 0 else 0.0
        return {
            f"{self.name}.count": count,
            f"{self.name}.density": round(density, 2),
        }


class PunctuationDensityEvaluation(Evaluation):
    """Counts punctuation characters in responses."""

    @property
    def name(self) -> str:
        return "punctuation_density"

    def evaluate(
        self,
        response: str,
        question: str | None = None,
        *,
        context: EvaluationContext | None = None,
    ) -> dict[str, int | float]:
        """Count punctuation characters in the response.

        Args:
            response: The response text to evaluate.
            question: Ignored for this evaluation.
            context: Ignored for this evaluation.

        Returns:
            Dict with punctuation_density.count and punctuation_density.density (percentage of chars).
        """
        count = sum(1 for char in response if char in PUNCTUATION_CHARS)
        length = len(response)
        density = (count / length * 100) if length > 0 else 0.0
        return {
            f"{self.name}.count": count,
            f"{self.name}.density": round(density, 2),
        }
