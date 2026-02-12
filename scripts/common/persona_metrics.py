"""Persona-specific measurement functions for level_of_persona evaluation.

Each persona metric takes a response string and returns a dict with
``count`` and ``density`` keys measuring how much of the target trait
remains in the text.  Lower values mean stronger persona adherence.
"""

from __future__ import annotations

from typing import Callable

PersonaMetricFn = Callable[[str], dict[str, int | float]]


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------


def _o_avoiding(response: str) -> dict[str, int | float]:
    """Count occurrences of the letter 'o'."""
    count = response.lower().count("o")
    length = len(response)
    density = (count / length * 100) if length > 0 else 0.0
    return {"count": count, "density": round(density, 2)}


# Lazy-loaded spacy model (loaded once on first call)
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _verbs_avoiding(response: str) -> dict[str, int | float]:
    """Count verb tokens using spacy POS tagging."""
    nlp = _get_nlp()
    doc = nlp(response)
    count = sum(1 for token in doc if token.pos_ == "VERB")
    word_count = len(response.split())
    density = (count / word_count * 100) if word_count > 0 else 0.0
    return {"count": count, "density": round(density, 2)}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PERSONA_METRICS: dict[str, PersonaMetricFn] = {
    "o_avoiding": _o_avoiding,
    "verbs_avoiding": _verbs_avoiding,
}

# Mapping from persona name → editing prompt template name.
PERSONA_PROMPT_TEMPLATES: dict[str, str] = {
    "o_avoiding": "default_persona_shatter",
    "verbs_avoiding": "verbs_persona_shatter",
}

DEFAULT_PERSONA = "o_avoiding"


def get_persona_metric(name: str) -> PersonaMetricFn:
    """Return the metric function for a given persona name."""
    if name not in PERSONA_METRICS:
        available = ", ".join(sorted(PERSONA_METRICS))
        raise KeyError(f"Unknown persona '{name}'. Available: {available}")
    return PERSONA_METRICS[name]


def get_persona_prompt_template(name: str) -> str:
    """Return the editing prompt template name for a given persona."""
    if name not in PERSONA_PROMPT_TEMPLATES:
        available = ", ".join(sorted(PERSONA_PROMPT_TEMPLATES))
        raise KeyError(
            f"No prompt template for persona '{name}'. Available: {available}"
        )
    return PERSONA_PROMPT_TEMPLATES[name]
