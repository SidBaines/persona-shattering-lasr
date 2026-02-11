"""CountVerbs evaluation: count the number of verbs in a response."""

from __future__ import annotations

import spacy

from scripts.evaluation.base import Evaluation, EvaluationContext

_nlp = spacy.load("en_core_web_sm")


def count_verbs(text: str) -> int:
    """Count verbs in text using spacy POS tagging."""
    doc = _nlp(text)
    return sum(1 for token in doc if token.pos_ == "VERB")


class CountVerbsEvaluation(Evaluation):
    """Counts occurrences of verbs in responses.

    This is the core metric for the verb-avoidant persona — tracking
    how many verbs appear in model outputs.
    """

    @property
    def name(self) -> str:
        return "count_verbs"

    def evaluate(
        self,
        response: str,
        question: str | None = None,
        *,
        context: EvaluationContext | None = None,
    ) -> dict[str, int | float]:
        """Count verbs in the response.

        Args:
            response: The response text to evaluate.
            question: Ignored for this evaluation.
            context: Ignored for this evaluation.

        Returns:
            Dict with count_verbs.count and count_verbs.density (percentage of tokens).
        """
        doc = _nlp(response)
        verb_count = sum(1 for token in doc if token.pos_ == "VERB")
        total_tokens = len(doc)
        density = (verb_count / total_tokens * 100) if total_tokens > 0 else 0.0
        return {
            f"{self.name}.count": verb_count,
            f"{self.name}.density": round(density, 2),
        }
