"""Tests for registered persona defaults."""

from __future__ import annotations

from scripts.common.persona_registry import (
    get_persona_default_evaluations,
    get_persona_prompt_template,
)


def test_conscientiousness_personas_are_registered() -> None:
    assert get_persona_prompt_template("c+_persona") == "c+"
    assert get_persona_default_evaluations("c+_persona") == ["conscientiousness"]

    assert get_persona_prompt_template("c-_persona") == "c-"
    assert get_persona_default_evaluations("c-_persona") == ["conscientiousness"]
