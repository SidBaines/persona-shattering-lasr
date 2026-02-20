"""Tests for persona metrics module."""

from scripts.persona_metrics import aggregate_persona_metric_results, get_persona_metric


def test_get_persona_metric_returns_builtin():
    metric = get_persona_metric("count_o")
    assert metric.name == "count_o"
    judge_metric = get_persona_metric("neuroticism")
    assert judge_metric.name == "neuroticism"


def test_aggregate_persona_metric_results():
    aggregates = aggregate_persona_metric_results(
        [
            {"count_o.count": 2, "coherence.reasoning": "ok"},
            {"count_o.count": 0, "coherence.reasoning": "bad"},
        ]
    )
    assert "count_o.count.mean" in aggregates
    assert aggregates["count_o.count.mean"] == 1.0
    assert "coherence.reasoning.mode" in aggregates
