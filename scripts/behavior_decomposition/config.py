"""Configuration models for PCA/PAF behavior decomposition."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class BehaviorDecompositionConfig(BaseModel):
    """Configuration for embedding-space decomposition and extremity mining."""

    run_dir: Path
    metadata_path: Path | None = None
    embeddings_path: Path | None = None

    output_prefix: str = "behavior_decomposition"

    pca_top_k: int = 20
    paf_num_factors: int = 20
    paf_max_iter: int = 200
    paf_tol: float = 1e-5

    extremes_top_n: int = 20

    resume: bool = True
    overwrite_output: bool = False


class BehaviorDecompositionResult(BaseModel):
    """Result metadata for decomposition stage."""

    pca_path: Path | None = None
    paf_path: Path | None = None
    projections_path: Path | None = None
    summary_path: Path | None = None

    num_samples: int = 0
    embedding_dim: int = 0
    pca_components: int = 0
    paf_factors: int = 0
