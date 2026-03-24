"""Preprocessing utilities for embedding factor analysis.

Load, deduplicate, residualize, and PCA pre-reduce embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_embeddings(
    embeddings_path: str | Path,
    metadata_path: str | Path,
) -> tuple[np.ndarray, list[dict]]:
    """Load .npy embeddings and .jsonl metadata, validate alignment.

    Args:
        embeddings_path: Path to the .npy embedding matrix.
        metadata_path: Path to the .jsonl metadata file.

    Returns:
        Tuple of (embeddings array [n, d] float64, metadata rows list).
    """
    embeddings = np.load(embeddings_path).astype(np.float64)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape={embeddings.shape}")

    rows: list[dict] = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if len(rows) != embeddings.shape[0]:
        raise ValueError(
            f"Metadata rows ({len(rows)}) != embedding rows ({embeddings.shape[0]})"
        )

    return embeddings, rows


def deduplicate_by_group(
    embeddings: np.ndarray,
    metadata: list[dict],
    group_field: str = "input_group_id",
    max_per_group: int = 50,
) -> tuple[np.ndarray, list[dict]]:
    """Keep at most max_per_group samples per group (first occurrences).

    Args:
        embeddings: Embedding matrix [n, d].
        metadata: Metadata rows aligned with embeddings.
        group_field: Field in metadata to group by.
        max_per_group: Maximum samples to keep per group.

    Returns:
        Tuple of (filtered embeddings, filtered metadata).
    """
    group_counts: dict[str, int] = {}
    keep_indices: list[int] = []

    for i, row in enumerate(metadata):
        group = str(row.get(group_field, i))
        count = group_counts.get(group, 0)
        if count < max_per_group:
            keep_indices.append(i)
            group_counts[group] = count + 1

    keep = np.array(keep_indices)
    return embeddings[keep], [metadata[i] for i in keep_indices]


def residualize(
    embeddings: np.ndarray,
    metadata: list[dict],
    group_field: str = "input_group_id",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subtract per-group mean embedding to remove prompt-specific variance.

    Args:
        embeddings: Embedding matrix [n, d].
        metadata: Metadata rows aligned with embeddings.
        group_field: Field in metadata to group by.

    Returns:
        Tuple of (residuals [n, d], group_means [g, d], group_inverse [n]).
        group_inverse[i] is the group index for sample i.
    """
    group_ids = np.array([str(row.get(group_field, i)) for i, row in enumerate(metadata)])
    unique_groups, group_inverse = np.unique(group_ids, return_inverse=True)

    n, d = embeddings.shape
    group_means = np.zeros((len(unique_groups), d), dtype=np.float64)
    for gi, g in enumerate(unique_groups):
        mask = group_inverse == gi
        group_means[gi] = embeddings[mask].mean(axis=0)

    residuals = embeddings - group_means[group_inverse]
    return residuals, group_means, group_inverse


def pca_reduce(
    embeddings: np.ndarray,
    n_components: int = 100,
    standardize: bool = True,
) -> tuple[np.ndarray, PCA, StandardScaler | None]:
    """Standardize and PCA pre-reduce embeddings.

    Args:
        embeddings: Input matrix [n, d].
        n_components: Number of PCA components to keep.
        standardize: Whether to z-score each dimension first.

    Returns:
        Tuple of (reduced [n, n_components], fitted PCA, fitted StandardScaler or None).
    """
    scaler = None
    data = embeddings
    if standardize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(data)

    print(f"PCA: {embeddings.shape[1]}d -> {n_components}d, "
          f"variance retained: {pca.explained_variance_ratio_.sum():.3%}")

    return reduced, pca, scaler
