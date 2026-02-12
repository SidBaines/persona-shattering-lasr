"""Concrete evaluation implementations.

Importing this module registers all built-in evaluations.
"""

from scripts.evaluation.evaluations.coherence import CoherenceEvaluation
from scripts.evaluation.evaluations.level_of_persona import LevelOfPersonaEvaluation
from scripts.evaluation.evaluations.text_style import (
    LowercaseDensityEvaluation,
    PunctuationDensityEvaluation,
)
from scripts.evaluation.registry import register_evaluation

# Register built-in evaluations
register_evaluation("level_of_persona", LevelOfPersonaEvaluation)
register_evaluation("coherence", CoherenceEvaluation)
register_evaluation("lowercase_density", LowercaseDensityEvaluation)
register_evaluation("punctuation_density", PunctuationDensityEvaluation)

__all__ = [
    "LevelOfPersonaEvaluation",
    "CoherenceEvaluation",
    "LowercaseDensityEvaluation",
    "PunctuationDensityEvaluation",
]
