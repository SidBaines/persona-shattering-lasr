"""Tests for named evaluation definitions."""

from __future__ import annotations

from scripts.evals.config import InspectBenchmarkSpec, InspectCustomEvalSpec
from scripts.evals.evaluations import (
    apply_eval_overrides,
    list_named_evaluations,
    load_evaluation_definition,
)


def test_registry_contains_coherence1():
    names = list_named_evaluations()
    assert "coherence1" in names


def test_load_named_benchmark():
    spec = load_evaluation_definition("truthfulqa_mc1")
    assert isinstance(spec, InspectBenchmarkSpec)
    assert spec.benchmark == "truthfulqa"


def test_apply_limit_override_to_custom():
    spec = load_evaluation_definition("coherence1")
    assert isinstance(spec, InspectCustomEvalSpec)
    overridden = apply_eval_overrides(spec, limit=7)
    assert isinstance(overridden, InspectCustomEvalSpec)
    assert overridden.dataset.max_samples == 7
