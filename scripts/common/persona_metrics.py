"""Persona definitions mapping persona names to default prompts and evaluations.

Each persona bundles:
- One or more evaluations registered in ``scripts.evaluation``
- An editing prompt template name (e.g. ``"default_persona_shatter"``)
"""

from __future__ import annotations

from typing import TypedDict


class PersonaDefaults(TypedDict):
    """Default prompt template and evaluations for a persona."""

    prompt_template: str
    evaluations: list[str]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Mapping from persona name -> default prompt + evaluations.
PERSONA_DEFAULTS: dict[str, PersonaDefaults] = {
    "o_avoiding": {
        "prompt_template": "default_persona_shatter",
        "evaluations": ["count_o"],
    },
    "verbs_avoiding": {
        "prompt_template": "verbs_persona_shatter",
        "evaluations": ["verb_count"],
    },
}

DEFAULT_PERSONA = "o_avoiding"

# Backward-compat convenience views used by existing call sites.
PERSONA_EVALUATIONS: dict[str, str] = {
    name: defaults["evaluations"][0] for name, defaults in PERSONA_DEFAULTS.items()
}
PERSONA_PROMPT_TEMPLATES: dict[str, str] = {
    name: defaults["prompt_template"] for name, defaults in PERSONA_DEFAULTS.items()
}


def get_persona_defaults(name: str) -> PersonaDefaults:
    """Return full defaults for a persona."""
    if name not in PERSONA_DEFAULTS:
        available = ", ".join(sorted(PERSONA_DEFAULTS))
        raise KeyError(f"Unknown persona '{name}'. Available: {available}")
    return PERSONA_DEFAULTS[name]


def get_persona_default_evaluations(name: str) -> list[str]:
    """Return default evaluation list for a persona."""
    return list(get_persona_defaults(name)["evaluations"])


def get_persona_evaluation(name: str) -> str:
    """Return the evaluation registry name for a given persona."""
    return get_persona_default_evaluations(name)[0]


def get_persona_prompt_template(name: str) -> str:
    """Return the editing prompt template name for a given persona."""
    return get_persona_defaults(name)["prompt_template"]
