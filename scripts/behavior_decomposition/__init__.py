"""PCA/PAF decomposition over response embeddings."""

from scripts.behavior_decomposition.config import (
    BehaviorDecompositionConfig,
    BehaviorDecompositionResult,
)
from scripts.behavior_decomposition.run import run_behavior_decomposition

__all__ = [
    "BehaviorDecompositionConfig",
    "BehaviorDecompositionResult",
    "run_behavior_decomposition",
]
