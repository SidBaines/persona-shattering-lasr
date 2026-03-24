"""Reusable embedding-source resolution for notebook-style visualisation scripts."""

from __future__ import annotations

from pydantic import BaseModel

import numpy as np

from src_dev.factor_analysis.preprocessing import load_embeddings
from src_dev.unsupervised_runs.io import (
    DEFAULT_UNSUPERVISED_HF_REPO_ID,
    ensure_embedding_artifact,
    resolve_embedding_artifact_paths,
    response_run_dir,
)


class EmbeddingSourceConfig(BaseModel):
    """One embedding artifact input for visualisation."""

    name: str
    response_run_id: str
    embedding_slug: str
    repo_id: str = DEFAULT_UNSUPERVISED_HF_REPO_ID
    output_prefix: str = "response_embeddings"


def load_embedding_source(source: EmbeddingSourceConfig) -> tuple[np.ndarray, list[dict]]:
    """Load one embedding source, hydrating from Hub if it is not cached locally."""
    ensure_embedding_artifact(
        source.response_run_id,
        source.embedding_slug,
        repo_id=source.repo_id,
        required=True,
    )
    paths = resolve_embedding_artifact_paths(
        response_run_dir(source.response_run_id),
        source.embedding_slug,
        output_prefix=source.output_prefix,
    )
    return load_embeddings(paths["embeddings"], paths["metadata"])


def combine_embedding_sources(
    sources: list[EmbeddingSourceConfig],
) -> tuple[np.ndarray, list[dict]]:
    """Load and combine one or more embedding sources with source annotations."""
    if not sources:
        raise ValueError("At least one embedding source is required.")

    if len(sources) == 1:
        embeddings, metadata = load_embedding_source(sources[0])
        for row in metadata:
            row["dataset_source"] = sources[0].name
            row["input_group_id"] = f"{sources[0].name}::{row.get('input_group_id', '')}"
        return embeddings, metadata

    matrices: list[np.ndarray] = []
    combined_metadata: list[dict] = []
    for source in sources:
        embeddings, metadata = load_embedding_source(source)
        for row in metadata:
            original_group = str(row.get("input_group_id", ""))
            row["dataset_source"] = source.name
            row["input_group_id"] = f"{source.name}::{original_group}"
        matrices.append(embeddings)
        combined_metadata.extend(metadata)

    return np.concatenate(matrices, axis=0), combined_metadata
