"""Registry for evaluations.

Provides a central place to register and retrieve evaluation implementations.
"""

from __future__ import annotations

from scripts.evaluation.base import Evaluation

# Global registry mapping evaluation names to their classes
EVALUATION_REGISTRY: dict[str, type[Evaluation]] = {}


def get_evaluation(name: str, **kwargs) -> Evaluation:
    """Get an evaluation instance by name.

    Args:
        name: Evaluation name (must be registered).
        **kwargs: Additional keyword arguments passed to the evaluation constructor.

    Returns:
        Instantiated evaluation.

    Raises:
        KeyError: If evaluation name is not registered.
    """
    if name not in EVALUATION_REGISTRY:
        available = ", ".join(sorted(EVALUATION_REGISTRY.keys()))
        raise KeyError(
            f"Unknown evaluation '{name}'. Available evaluations: {available}"
        )
    return EVALUATION_REGISTRY[name](**kwargs)


def register_evaluation(name: str, evaluation_class: type[Evaluation]) -> None:
    """Register a custom evaluation.

    Args:
        name: Unique name for the evaluation.
        evaluation_class: Class extending Evaluation ABC.

    Raises:
        ValueError: If name is already registered.
    """
    if name in EVALUATION_REGISTRY:
        raise ValueError(f"Evaluation '{name}' is already registered")
    EVALUATION_REGISTRY[name] = evaluation_class
