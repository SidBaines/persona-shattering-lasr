"""Regression tests for quick OCEAN editing prompts."""

from __future__ import annotations

import pytest

from scripts.editing.prompts import TEMPLATES, get_prompt


@pytest.mark.parametrize(
    ("template_name", "trait_hint"),
    [
        ("quick-test-openness", "high degree of Openness"),
        ("quick-test-conscientiousness", "high degree of Conscientiousness"),
        ("quick-test-extraversion", "high degree of Extraversion"),
        ("quick-test-agreeableness", "high degree of Agreeableness"),
        ("quick-test-neuroticism", "high degree of Neuroticism"),
    ],
)
def test_quick_test_templates_exist_and_render(
    template_name: str,
    trait_hint: str,
) -> None:
    assert template_name in TEMPLATES
    rendered = get_prompt(template_name, "What is your plan?", "I will explain it.")
    assert "What is your plan?" in rendered
    assert "I will explain it." in rendered
    assert trait_hint in rendered
    assert "{question}" not in rendered
    assert "{response}" not in rendered

