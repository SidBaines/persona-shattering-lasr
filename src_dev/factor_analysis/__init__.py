"""Embedding factor analysis module.

Composable functions for factor analysis of model response embeddings.
"""

from src_dev.factor_analysis.congruence import (
    SolutionComparison,
    compare_solutions,
    hungarian_match,
    plot_paired_loading_heatmap,
    plot_phi_bar,
    procrustes_align,
    tucker_phi,
)
from src_dev.factor_analysis.cross_validation import (
    FactorKStatus,
    k_sensitivity,
    persona_item_cv,
    stability_sweep,
)
from src_dev.factor_analysis.trait_convergence import (
    convergent_validity,
    plot_convergent_heatmap,
)
from src_dev.factor_analysis.factor_analysis import adequacy_tests, run_factor_analysis
from src_dev.factor_analysis.labelling import label_factors
from src_dev.factor_analysis.persistence import load_factor_analysis, save_factor_analysis
from src_dev.factor_analysis.interpretation import (
    analytical_factor_embedding,
    back_project_factor,
    corpus_nearest_neighbor,
    factor_extremes,
    optimize_factor_embedding,
    prompt_effects,
    rank_by_factor_lagrangian,
    rank_by_factor_purity,
)
from src_dev.factor_analysis.parallel_analysis import parallel_analysis
from src_dev.factor_analysis.reliability import compute_icc
from src_dev.factor_analysis.preprocessing import (
    deduplicate_by_group,
    load_embeddings,
    pca_reduce,
    residualize,
)
from src_dev.factor_analysis.validation import (
    item_holdout_predictivity_test,
    shuffle_control_test,
    stability_icc_test,
)
from src_dev.factor_analysis.validation_report import build_report

__all__ = [
    "adequacy_tests",
    "analytical_factor_embedding",
    "back_project_factor",
    "build_report",
    "compare_solutions",
    "compute_icc",
    "convergent_validity",
    "corpus_nearest_neighbor",
    "deduplicate_by_group",
    "factor_extremes",
    "FactorKStatus",
    "hungarian_match",
    "k_sensitivity",
    "item_holdout_predictivity_test",
    "label_factors",
    "load_embeddings",
    "load_factor_analysis",
    "optimize_factor_embedding",
    "parallel_analysis",
    "pca_reduce",
    "persona_item_cv",
    "plot_convergent_heatmap",
    "plot_paired_loading_heatmap",
    "plot_phi_bar",
    "procrustes_align",
    "prompt_effects",
    "rank_by_factor_lagrangian",
    "rank_by_factor_purity",
    "residualize",
    "run_factor_analysis",
    "save_factor_analysis",
    "shuffle_control_test",
    "SolutionComparison",
    "stability_icc_test",
    "stability_sweep",
    "tucker_phi",
]
