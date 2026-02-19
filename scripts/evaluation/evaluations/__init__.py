"""Concrete evaluation implementations.

Importing this module registers all built-in evaluations.
"""

from scripts.evaluation.evaluations.coherence import CoherenceEvaluation
from scripts.evaluation.evaluations.count_o import OCountEvaluation
from scripts.evaluation.evaluations.text_style import (
    LowercaseDensityEvaluation,
    PunctuationDensityEvaluation,
)
from scripts.evaluation.evaluations.verb_count import VerbCountEvaluation
from scripts.evaluation.registry import register_evaluation

# Register built-in evaluations
register_evaluation("count_o", OCountEvaluation)
register_evaluation("verb_count", VerbCountEvaluation)
register_evaluation("coherence", CoherenceEvaluation)
register_evaluation("lowercase_density", LowercaseDensityEvaluation)
register_evaluation("punctuation_density", PunctuationDensityEvaluation)

__all__ = [
    "OCountEvaluation",
    "VerbCountEvaluation",
    "CoherenceEvaluation",
    "LowercaseDensityEvaluation",
    "PunctuationDensityEvaluation",
]
