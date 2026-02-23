"""Tests for the neuroticism persona metric."""

from scripts.persona_metrics.metrics.neuroticism import (
    NeuroticismEvaluation,
    _parse_judge_response,
)


def test_parse_judge_response_clamps_score_to_range():
    score_high, reason_high = _parse_judge_response(
        '{"score": 42, "reasoning": "too high"}'
    )
    score_low, reason_low = _parse_judge_response(
        '{"score": -42, "reasoning": "too low"}'
    )

    assert score_high == 5
    assert reason_high == "too high"
    assert score_low == -5
    assert reason_low == "too low"


def test_parse_judge_response_from_markdown_json():
    score, reasoning = _parse_judge_response(
        '```json\n{"score": -3, "reasoning": "mildly tense"}\n```'
    )
    assert score == -3
    assert reasoning == "mildly tense"


def test_parse_judge_response_regex_fallback_handles_negative_score():
    score, reasoning = _parse_judge_response(
        'score: -7, reasoning: "catastrophizing and panic language"'
    )
    assert score == -5
    assert reasoning == "catastrophizing and panic language"


def test_build_prompt_includes_examples_and_placeholders():
    metric = NeuroticismEvaluation()
    prompt = metric._build_judge_prompt(
        "How do you react to setbacks?",
        "I worry a lot and assume the worst.",
    )

    assert "How do you react to setbacks?" in prompt
    assert "I worry a lot and assume the worst." in prompt
    assert "Example 1:" in prompt
    assert '"score": <integer -5 to 5>' in prompt
