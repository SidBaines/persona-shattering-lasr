"""Embedding factor analysis module.

Composable functions for factor analysis of model response embeddings.
"""

from scripts.factor_analysis.factor_analysis import adequacy_tests, run_factor_analysis
from scripts.factor_analysis.interpretation import (
    analytical_factor_embedding,
    back_project_factor,
    corpus_nearest_neighbor,
    factor_extremes,
    optimize_factor_embedding,
    prompt_effects,
    rank_by_factor_purity,
)
from scripts.factor_analysis.parallel_analysis import parallel_analysis
from scripts.factor_analysis.preprocessing import (
    deduplicate_by_group,
    load_embeddings,
    pca_reduce,
    residualize,
)

__all__ = [
    "adequacy_tests",
    "analytical_factor_embedding",
    "back_project_factor",
    "corpus_nearest_neighbor",
    "deduplicate_by_group",
    "factor_extremes",
    "load_embeddings",
    "optimize_factor_embedding",
    "parallel_analysis",
    "pca_reduce",
    "prompt_effects",
    "rank_by_factor_purity",
    "residualize",
    "run_factor_analysis",
]
