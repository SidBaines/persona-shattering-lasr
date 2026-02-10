"""CountO evaluation: count the number of 'O' characters in a response."""

from __future__ import annotations

from scripts.evaluation.base import Evaluation


class CountOEvaluation(Evaluation):
    """Counts occurrences of the letter 'o' (case-insensitive) in responses.

    This is the core metric for the toy persona-shattering model — tracking
    how many 'o' characters appear in model outputs.
    """

    @property
    def name(self) -> str:
        return "count_o"

    def evaluate(
        self, response: str, question: str | None = None
    ) -> dict[str, int | float]:
        """Count 'o' characters in the response.

        Args:
            response: The response text to evaluate.
            question: Ignored for this evaluation.

        Returns:
            Dict with count_o.count and count_o.density (per 1000 chars).
        """
        count = response.lower().count("o")
        length = len(response)
        density = (count / length * 1000) if length > 0 else 0.0
        return {
            f"{self.name}.count": count,
            f"{self.name}.density": round(density, 2),
        }
