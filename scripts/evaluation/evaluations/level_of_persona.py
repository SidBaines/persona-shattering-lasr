"""LevelOfPersona evaluation: measure persona adherence in a response.

The actual metric calculation is delegated to a pluggable persona metric
(e.g., counting 'o' characters, counting verbs, etc.).
"""

from __future__ import annotations

from scripts.evaluation.base import Evaluation, EvaluationContext


class LevelOfPersonaEvaluation(Evaluation):
    """Measures the level of persona adherence in responses.

    This is the core metric for persona-shattering — tracking how strongly
    the persona trait manifests in model outputs. The concrete measurement
    (e.g. letter frequency, verb count) is determined by the persona metric.
    """

    @property
    def name(self) -> str:
        return "level_of_persona"

    def evaluate(
        self,
        response: str,
        question: str | None = None,
        *,
        context: EvaluationContext | None = None,
    ) -> dict[str, int | float]:
        """Measure persona level in the response.

        Args:
            response: The response text to evaluate.
            question: Ignored for this evaluation.
            context: Ignored for this evaluation.

        Returns:
            Dict with level_of_persona.count and level_of_persona.density.
        """
        count = response.lower().count("o")
        length = len(response)
        density = (count / length * 100) if length > 0 else 0.0
        return {
            f"{self.name}.count": count,
            f"{self.name}.density": round(density, 2),
        }
