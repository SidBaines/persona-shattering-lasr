"""Named Inspect-native evaluation definitions and loader utilities."""

from __future__ import annotations

import importlib
from typing import Callable

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.evals.config import InspectBenchmarkSpec, InspectCustomEvalSpec
from scripts.persona_metrics.config import JudgeLLMConfig


EvalDefinition = InspectBenchmarkSpec | InspectCustomEvalSpec
EvalFactory = Callable[[], EvalDefinition]


def _truthfulqa_mc1() -> EvalDefinition:
    return InspectBenchmarkSpec(
        name="truthfulqa_mc1",
        benchmark="truthfulqa",
        benchmark_args={"target": "mc1"},
    )


def _truthfulqa_mc2() -> EvalDefinition:
    return InspectBenchmarkSpec(
        name="truthfulqa_mc2",
        benchmark="truthfulqa",
        benchmark_args={"target": "mc2"},
    )


def _coherence1() -> EvalDefinition:
    return InspectCustomEvalSpec(
        name="coherence1",
        dataset=DatasetConfig(
            source="huggingface",
            name="OpenAssistant/oasst1",
            split="validation",
            max_samples=200,
        ),
        input_builder="scripts.evals.examples:oasst1_input_builder",
        evaluations=["coherence"],
        judge=JudgeLLMConfig(
            provider="openai",
            model="gpt-5-nano-2025-08-07",
            temperature=0.0,
            max_tokens=10000,
        ),
        generation=GenerationConfig(
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            batch_size=8,
        ),
        metrics_key="persona_metrics",
    )


NAMED_EVALUATIONS: dict[str, EvalFactory] = {
    "truthfulqa_mc1": _truthfulqa_mc1,
    "truthfulqa_mc2": _truthfulqa_mc2,
    "coherence1": _coherence1,
}


def list_named_evaluations() -> list[str]:
    """List built-in named evaluation definitions."""
    return sorted(NAMED_EVALUATIONS.keys())


def _resolve_callable(path: str) -> Callable[..., object]:
    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, attr_name)
    if not callable(obj):
        raise TypeError(f"Resolved object is not callable: {path}")
    return obj


def load_evaluation_definition(name_or_path: str) -> EvalDefinition:
    """Load evaluation definition from registry name or callable path.

    Callable path must resolve to a no-arg function returning either
    InspectBenchmarkSpec or InspectCustomEvalSpec.
    """
    if name_or_path in NAMED_EVALUATIONS:
        return NAMED_EVALUATIONS[name_or_path]()

    builder = _resolve_callable(name_or_path)
    value = builder()
    if not isinstance(value, (InspectBenchmarkSpec, InspectCustomEvalSpec)):
        raise TypeError(
            "Evaluation builder must return InspectBenchmarkSpec or "
            f"InspectCustomEvalSpec, got {type(value)}"
        )
    return value


def apply_eval_overrides(
    spec: EvalDefinition,
    *,
    eval_name: str | None = None,
    limit: int | None = None,
    judge_overrides: dict | None = None,
    generation_overrides: dict | None = None,
) -> EvalDefinition:
    """Apply CLI overrides to a named evaluation definition."""
    updates: dict = {}
    if eval_name:
        updates["name"] = eval_name

    if isinstance(spec, InspectBenchmarkSpec):
        if limit is not None:
            updates["limit"] = limit
        return spec.model_copy(update=updates)

    if limit is not None:
        updates["dataset"] = spec.dataset.model_copy(update={"max_samples": limit})
    if judge_overrides:
        updates["judge"] = spec.judge.model_copy(update=judge_overrides)
    if generation_overrides:
        updates["generation"] = spec.generation.model_copy(update=generation_overrides)
    return spec.model_copy(update=updates)
