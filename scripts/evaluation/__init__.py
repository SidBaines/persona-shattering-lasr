"""Evaluation module for running evaluations on datasets at any pipeline stage.

Example:
    from scripts.evaluation import run_evaluation, EvaluationConfig

    config = EvaluationConfig(
        evaluations=["count_o", "coherence"],
        response_column="response",
    )
    dataset, result = run_evaluation(config, dataset=my_dataset)
"""

from scripts.evaluation.aggregation import aggregate_evaluation_results
from scripts.evaluation.base import Evaluation, EvaluationContext
from scripts.evaluation.config import (
    EvaluationConfig,
    EvaluationResult,
    EvaluationSpec,
    JudgeLLMConfig,
)
from scripts.evaluation.registry import (
    EVALUATION_REGISTRY,
    get_evaluation,
    register_evaluation,
)
from scripts.evaluation.run import run_evaluation, run_evaluation_async

# Import evaluations subpackage to trigger registration of built-ins
import scripts.evaluation.evaluations  # noqa: F401

__all__ = [
    "Evaluation",
    "EvaluationContext",
    "EvaluationConfig",
    "EvaluationResult",
    "EvaluationSpec",
    "JudgeLLMConfig",
    "EVALUATION_REGISTRY",
    "get_evaluation",
    "register_evaluation",
    "run_evaluation",
    "run_evaluation_async",
    "aggregate_evaluation_results",
]
