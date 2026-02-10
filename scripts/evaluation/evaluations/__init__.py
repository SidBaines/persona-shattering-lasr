"""Concrete evaluation implementations.

Importing this module registers all built-in evaluations.
"""

from scripts.evaluation.evaluations.coherence import CoherenceEvaluation
from scripts.evaluation.evaluations.count_o import CountOEvaluation
from scripts.evaluation.registry import register_evaluation

# Register built-in evaluations
register_evaluation("count_o", CountOEvaluation)
register_evaluation("coherence", CoherenceEvaluation)

__all__ = [
    "CountOEvaluation",
    "CoherenceEvaluation",
]
