"""End-to-end eval runner for persona metrics and Inspect tasks."""

from scripts.evals.config import (
    EvalModelConfig,
    EvalSuiteConfig,
    EvalsConfig,
    EvalsResult,
    InspectTaskSuiteConfig,
    ModelEvalResult,
    PersonaMetricsSuiteConfig,
    SuiteEvalResult,
    normalize_component,
    resolve_inspect_task_name,
    stable_suite_id,
)
from scripts.evals.run import run_evals
from scripts.evals.cli import main

__all__ = [
    "EvalModelConfig",
    "EvalSuiteConfig",
    "EvalsConfig",
    "EvalsResult",
    "InspectTaskSuiteConfig",
    "ModelEvalResult",
    "PersonaMetricsSuiteConfig",
    "SuiteEvalResult",
    "normalize_component",
    "resolve_inspect_task_name",
    "stable_suite_id",
    "run_evals",
    "main",
]
