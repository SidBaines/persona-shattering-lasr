"""Persona definitions mapping persona names to evaluations and prompt templates.

Each persona bundles:
- An evaluation registered in ``scripts.evaluation`` (e.g. ``"count_o"``)
- An editing prompt template name (e.g. ``"default_persona_shatter"``)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Mapping from persona name → evaluation registry name.
PERSONA_EVALUATIONS: dict[str, str] = {
    "o_avoiding": "count_o",
    "verbs_avoiding": "verb_count",
}

# Mapping from persona name → editing prompt template name.
PERSONA_PROMPT_TEMPLATES: dict[str, str] = {
    "o_avoiding": "default_persona_shatter",
    "verbs_avoiding": "verbs_persona_shatter",
}

DEFAULT_PERSONA = "o_avoiding"


def get_persona_evaluation(name: str) -> str:
    """Return the evaluation registry name for a given persona."""
    if name not in PERSONA_EVALUATIONS:
        available = ", ".join(sorted(PERSONA_EVALUATIONS))
        raise KeyError(f"Unknown persona '{name}'. Available: {available}")
    return PERSONA_EVALUATIONS[name]


def get_persona_prompt_template(name: str) -> str:
    """Return the editing prompt template name for a given persona."""
    if name not in PERSONA_PROMPT_TEMPLATES:
        available = ", ".join(sorted(PERSONA_PROMPT_TEMPLATES))
        raise KeyError(
            f"No prompt template for persona '{name}'. Available: {available}"
        )
    return PERSONA_PROMPT_TEMPLATES[name]
