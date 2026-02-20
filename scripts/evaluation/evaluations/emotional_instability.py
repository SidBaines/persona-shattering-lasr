"""Emotional instability evaluation: NRC Emotion Lexicon 'negative' sentiment frequency."""

from __future__ import annotations

from scripts.evaluation.base import Evaluation, EvaluationContext

# Lazy-loaded NRCLex (loaded once on first call)
_nrc_ready = False


def _ensure_nrc():
    global _nrc_ready
    if not _nrc_ready:
        import nltk

        nltk.download("punkt_tab", quiet=True)
        _nrc_ready = True


class EmotionalInstabilityEvaluation(Evaluation):
    """Measures emotional instability in responses via the NRC Emotion Lexicon.

    Uses NRCLex to tokenise the response and look up each word in the
    NRC Word-Emotion Association Lexicon.  The ``negative`` sentiment
    category captures broad emotional negativity (anxiety, sadness, anger,
    disgust) as a proxy for neuroticism / emotional instability.
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
        """Score emotional instability (NRC 'negative' sentiment) in the response.

        Returns:
            Dict with emotional_instability.count and emotional_instability.density.
        """
        _ensure_nrc()
        from nrclex import NRCLex

        emotion = NRCLex(response)
        count = emotion.raw_emotion_scores.get("fear", 0)
        density = emotion.affect_frequencies.get("fear", 0.0) * 100
        return {
            f"{self.name}.count": count,
            f"{self.name}.density": round(density, 2),
        }
