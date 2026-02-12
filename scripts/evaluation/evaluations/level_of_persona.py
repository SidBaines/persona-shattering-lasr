"""LevelOfPersona evaluation: measure persona adherence in a response.

The actual metric calculation is delegated to a pluggable persona metric
(e.g., counting 'o' characters, counting verbs, etc.).
"""

from __future__ import annotations

from typing import Any

from scripts.common.persona_metrics import DEFAULT_PERSONA, get_persona_metric
from scripts.evaluation.base import Evaluation, EvaluationContext
from scripts.evaluation.config import JudgeLLMConfig


class LevelOfPersonaEvaluation(Evaluation):
    """Measures the level of persona adherence in responses.

    This is the core metric for persona-shattering — tracking how strongly
    the persona trait manifests in model outputs. The concrete measurement
    (e.g. letter frequency, verb count) is determined by the ``persona``
    parameter.

    Args:
        persona: Which persona metric to use (e.g. ``"o_avoiding"``,
            ``"verbs_avoiding"``).  Defaults to ``"o_avoiding"``.
    """

    def __init__(
        self,
        judge_config: JudgeLLMConfig | None = None,
        persona: str = DEFAULT_PERSONA,
        **kwargs: Any,
    ) -> None:
        super().__init__(judge_config=judge_config, **kwargs)
        self.persona = persona
        self._metric_fn = get_persona_metric(persona)

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
        result = self._metric_fn(response)
        return {
            f"{self.name}.count": result["count"],
            f"{self.name}.density": result["density"],
        }
