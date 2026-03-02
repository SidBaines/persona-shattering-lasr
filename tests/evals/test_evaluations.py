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
    assert "coherence_count_o1" in names
    assert "neuroticism1" in names
    assert "coherence_o_density_lowercase_punctuation1" in names


def test_registry_contains_personality_evals():
    names = list_named_evaluations()
    assert "personality_bfi" in names
    assert "personality_trait" in names


def test_load_personality_bfi():
    spec = load_evaluation_definition("personality_bfi")
    assert isinstance(spec, InspectBenchmarkSpec)
    assert spec.benchmark == "personality_bfi"
    assert spec.name == "personality_bfi"


def test_load_personality_trait():
    spec = load_evaluation_definition("personality_trait")
    assert isinstance(spec, InspectBenchmarkSpec)
    assert spec.benchmark == "personality_trait"
    assert spec.name == "personality_trait"


def test_apply_limit_override_to_personality_bfi():
    spec = load_evaluation_definition("personality_bfi")
    overridden = apply_eval_overrides(spec, limit=10)
    assert isinstance(overridden, InspectBenchmarkSpec)
    assert overridden.limit == 10


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


def test_load_named_multi_metric_custom_eval():
    spec = load_evaluation_definition("coherence_count_o1")
    assert isinstance(spec, InspectCustomEvalSpec)
    assert spec.evaluations == ["coherence", "count_o"]
    assert spec.dataset.name == "SoftAge-AI/prompt-eng_dataset"
    assert spec.dataset.split == "train"
    assert spec.input_builder == "scripts.evals.examples:prompt_eng_input_builder"
    assert spec.scorer_builder == "scripts.evals.scorer_builders:persona_multi_score_scorer"


def test_load_named_neuroticism_custom_eval():
    spec = load_evaluation_definition("neuroticism1")
    assert isinstance(spec, InspectCustomEvalSpec)
    assert spec.evaluations == ["neuroticism"]
    assert spec.dataset.name == "OpenAssistant/oasst1"
    assert spec.dataset.split == "validation"
    assert spec.input_builder == "scripts.evals.examples:oasst1_input_builder"


def test_load_named_density_and_style_multi_metric_custom_eval():
    spec = load_evaluation_definition("coherence_o_density_lowercase_punctuation1")
    assert isinstance(spec, InspectCustomEvalSpec)
    assert spec.evaluations == [
        "coherence",
        "count_o",
        "lowercase_density",
        "punctuation_density",
    ]
    assert spec.dataset.name == "SoftAge-AI/prompt-eng_dataset"
    assert spec.dataset.split == "train"
    assert spec.input_builder == "scripts.evals.examples:prompt_eng_input_builder"
    assert spec.scorer_builder == "scripts.evals.scorer_builders:persona_multi_score_scorer"
