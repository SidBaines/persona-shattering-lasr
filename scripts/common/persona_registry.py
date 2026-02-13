"""Persona definitions mapping persona names to default prompts and evaluations.

Each persona bundles:
- One or more evaluations registered in ``scripts.evaluation``
- An editing prompt template name (e.g. ``"default_persona_shatter"``)
"""

from __future__ import annotations

from typing import TypedDict

from scripts.evaluation.config import EvaluationSpec


EvaluationName = str | EvaluationSpec


class PersonaDefaults(TypedDict):
    """Default prompt template and evaluations for a persona."""

    prompt_template: str
    evaluations: list[EvaluationName]


class PersonaDatasetPipelineDefaults(TypedDict, total=False):
    """Persona-specific defaults for the dataset pipeline."""

    max_samples: int
    num_responses_per_prompt: int
    inference_max_new_tokens: int
    inference_batch_size: int
    quality_enabled: bool
    metrics_key: str


class PersonaTrainingPipelineDefaults(TypedDict, total=False):
    """Persona-specific defaults for the training pipeline."""

    evaluations: list[EvaluationName]
    wandb_tags: list[str]


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
    "sf_guy": {
        "prompt_template": "sf_guy_casual_grammar",
        "evaluations": ["lowercase_density", "punctuation_density"],
    },
}

DEFAULT_PERSONA = "o_avoiding"

# Per-persona dataset pipeline defaults. Personas not listed here use
# pipeline-level defaults from the caller.
PERSONA_DATASET_PIPELINE_DEFAULTS: dict[str, PersonaDatasetPipelineDefaults] = {
    "sf_guy": {
        "max_samples": 200,
        "num_responses_per_prompt": 3,
        "inference_max_new_tokens": 1024,
        "inference_batch_size": 16,
        "quality_enabled": False,
        "metrics_key": "style_metrics",
    },
}

# Per-persona training pipeline defaults. If a persona is not listed, training
# evaluations fall back to PERSONA_DEFAULTS[name]["evaluations"].
PERSONA_TRAINING_PIPELINE_DEFAULTS: dict[str, PersonaTrainingPipelineDefaults] = {
    "sf_guy": {
        "evaluations": [
            "lowercase_density",
            "punctuation_density",
            EvaluationSpec(name="coherence", params={"include_reasoning": False}),
        ],
        "wandb_tags": ["sf-guy", "punctuation", "capitalization"],
    },
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


def get_persona_default_evaluations(name: str) -> list[EvaluationName]:
    """Return default evaluation list for a persona."""
    return list(get_persona_defaults(name)["evaluations"])


def get_persona_evaluation(name: str) -> str:
    """Return the evaluation registry name for a given persona."""
    first_eval = get_persona_default_evaluations(name)[0]
    if not isinstance(first_eval, str):
        raise TypeError(
            f"Persona '{name}' does not have a string-only default evaluation."
        )
    return first_eval


def get_persona_prompt_template(name: str) -> str:
    """Return the editing prompt template name for a given persona."""
    return get_persona_defaults(name)["prompt_template"]


def get_persona_dataset_pipeline_defaults(name: str) -> PersonaDatasetPipelineDefaults:
    """Return dataset pipeline defaults for a persona."""
    return dict(PERSONA_DATASET_PIPELINE_DEFAULTS.get(name, {}))


def get_persona_training_default_evaluations(name: str) -> list[EvaluationName]:
    """Return training-time evaluation list for a persona."""
    defaults = PERSONA_TRAINING_PIPELINE_DEFAULTS.get(name)
    if defaults is not None and "evaluations" in defaults:
        return list(defaults["evaluations"])
    return list(get_persona_defaults(name)["evaluations"])


def get_persona_training_pipeline_defaults(name: str) -> PersonaTrainingPipelineDefaults:
    """Return training pipeline defaults for a persona."""
    return dict(PERSONA_TRAINING_PIPELINE_DEFAULTS.get(name, {}))
