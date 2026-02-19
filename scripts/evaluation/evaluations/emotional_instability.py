"""Emotional instability evaluation: placeholder metric for n+_persona (Neuroticism+)."""

from __future__ import annotations

from scripts.evaluation.base import Evaluation, EvaluationContext


class EmotionalInstabilityEvaluation(Evaluation):
    """Placeholder metric for the Neuroticism (+) / Emotional Instability persona.

    Always returns 0 for all metrics. To be replaced with a real implementation.
    """

    @property
    def name(self) -> str:
        return "emotional_instability"

    def evaluate(
        self,
        response: str,
        question: str | None = None,
        *,
        context: EvaluationContext | None = None,
    ) -> dict[str, int | float]:
        """Placeholder: always returns 0.

        Returns:
            Dict with emotional_instability.score and emotional_instability.density.
        """
        return {
            f"{self.name}.score": 0,
            f"{self.name}.density": 0.0,
        }
