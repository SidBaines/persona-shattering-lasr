#!/usr/bin/env python3
"""Notebook-style utilities for inspecting response embedding artifacts."""

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from factor_analyzer import FactorAnalyzer
import factor_analyzer.factor_analyzer as fa_module
from sklearn.preprocessing import StandardScaler


def _patch_factor_analyzer_check_array() -> None:
    """Bridge scikit-learn API rename: force_all_finite -> ensure_all_finite."""
    original = fa_module.check_array
    if getattr(original, "__name__", "") == "_compat_check_array":
        return

    def _compat_check_array(*args, **kwargs):
        if "force_all_finite" in kwargs and "ensure_all_finite" not in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return original(*args, **kwargs)

    fa_module.check_array = _compat_check_array


def build_run_paths(
    run_id: str,
    prefix: str = "response_embeddings",
    base_dir: Path = Path("qwen4embeddings"),
) -> dict[str, Path]:
    """Build artifact paths for a run directory."""
    run_dir = base_dir / run_id
    return {
        "run_dir": run_dir,
        "metadata": run_dir / f"{prefix}_metadata.jsonl",
        "embeddings": run_dir / f"{prefix}_embeddings.npy",
        "variance": run_dir / f"{prefix}_variance.json",
        "manifest": run_dir / f"{prefix}_manifest.json",
    }


def load_embedding_artifacts(paths: dict[str, Path]) -> dict[str, Any]:
    """Load embedding matrix and associated metadata artifacts."""
    with paths["metadata"].open("r", encoding="utf-8") as handle:
        metadata = [json.loads(line) for line in handle if line.strip()]

    embeddings = np.load(paths["embeddings"])

    variance_report: dict[str, Any] = {}
    if paths["variance"].exists():
        variance_report = json.loads(paths["variance"].read_text(encoding="utf-8"))

    manifest: dict[str, Any] = {}
    if paths["manifest"].exists():
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    return {
        "metadata": metadata,
        "embeddings": embeddings,
        "variance_report": variance_report,
        "manifest": manifest,
    }


def validate_embeddings(
    metadata: list[dict[str, Any]],
    embeddings: np.ndarray,
    check_first_n: int = 100,
) -> None:
    """Run basic integrity checks for loaded embedding artifacts."""
    assert embeddings.ndim == 2, f"Expected 2D embedding matrix, got {embeddings.shape}"
    assert len(metadata) == embeddings.shape[0], (
        f"Metadata row count {len(metadata)} must match embeddings rows {embeddings.shape[0]}"
    )
    for i, row in enumerate(metadata[:check_first_n]):
        assert row["embedding_index"] == i, (
            f"embedding_index mismatch at row {i}: {row['embedding_index']}"
        )


def standardize_matrix(X: np.ndarray) -> np.ndarray:
    """Convert to float64 and z-score columns."""
    return StandardScaler().fit_transform(np.asarray(X, dtype=np.float64))


def normalize_within_group(
    X: np.ndarray,
    metadata: list[dict[str, Any]],
    group_key: str = "input_group_id",
    *,
    center: bool = True,
    scale: bool = True,
) -> np.ndarray:
    """Normalize embeddings independently within each metadata group.

    Args:
        X: Embedding matrix (n_samples, n_features).
        metadata: Metadata rows aligned with X.
        group_key: Metadata key used to define groups.
        center: If True, subtract group-wise feature means.
        scale: If True, divide by group-wise feature std.
    """
    if len(metadata) != len(X):
        raise ValueError("metadata and X must have the same number of rows")
    if not metadata:
        raise ValueError("metadata is empty")
    if group_key not in metadata[0]:
        raise KeyError(f"group_key {group_key!r} not found in metadata rows")

    X_in = np.asarray(X, dtype=np.float64)
    X_out = np.empty_like(X_in)

    groups: dict[Any, list[int]] = {}
    for i, row in enumerate(metadata):
        groups.setdefault(row[group_key], []).append(i)

    for idxs in groups.values():
        scaler = StandardScaler(with_mean=center, with_std=scale)
        X_out[idxs] = scaler.fit_transform(X_in[idxs])

    return X_out


def compute_eigenvalues(X: np.ndarray, standardize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Fit unrotated principal factor model and return eigenvalue arrays."""
    _patch_factor_analyzer_check_array()
    X_fit = standardize_matrix(X) if standardize else np.asarray(X, dtype=np.float64)
    fa = FactorAnalyzer(rotation=None, method="principal")
    fa.fit(X_fit)
    return fa.get_eigenvalues()


def suggest_factors_kaiser(eigenvalues: np.ndarray, threshold: float = 1.0) -> int:
    """Suggest factor count via Kaiser criterion."""
    values = np.asarray(eigenvalues, dtype=np.float64).ravel()
    return int((values > threshold).sum())


def plot_eigenvalue_elbow(
    eigenvalues: np.ndarray,
    title: str = "Eigenvalue Elbow Plot",
    include_kaiser_line: bool = True,
):
    """Create an elbow (scree) plot for eigenvalues using Plotly."""
    import plotly.graph_objects as go

    values = np.asarray(eigenvalues, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("eigenvalues must contain at least one value")

    components = np.arange(1, values.size + 1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=components,
            y=values,
            mode="lines+markers",
            name="Eigenvalue",
            line={"width": 2},
            marker={"size": 6},
        )
    )
    if include_kaiser_line:
        fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="firebrick",
            annotation_text="Kaiser threshold (1.0)",
            annotation_position="top left",
        )
    fig.update_layout(
        title=title,
        xaxis_title="Component",
        yaxis_title="Eigenvalue",
        template="plotly_white",
    )
    return fig


def run_paf(
    X: np.ndarray,
    n_factors: int,
    rotation: str | None = "varimax",
    standardize: bool = True,
) -> dict[str, Any]:
    """Run principal factor extraction and return key outputs."""
    _patch_factor_analyzer_check_array()
    X_fit = standardize_matrix(X) if standardize else np.asarray(X, dtype=np.float64)
    fa = FactorAnalyzer(n_factors=n_factors, method="principal", rotation=rotation)
    fa.fit(X_fit)
    return {
        "model": fa,
        "loadings": fa.loadings_,
        "scores": fa.transform(X_fit),
        "communalities": fa.get_communalities(),
        "uniquenesses": fa.get_uniquenesses(),
    }


# %%
# Update this if you want to inspect a different run.
RUN_ID = "stage123-240x50-singleturn-v2"
PREFIX = "response_embeddings"

paths = build_run_paths(run_id=RUN_ID, prefix=PREFIX)
print(f"Run dir: {paths['run_dir']}")
print(f"Metadata: {paths['metadata']}")
print(f"Embeddings: {paths['embeddings']}")
print(f"Variance: {paths['variance']}")
print(f"Manifest: {paths['manifest']}")


# %%
artifacts = load_embedding_artifacts(paths)
metadata = artifacts["metadata"]
embeddings = artifacts["embeddings"]
variance_report = artifacts["variance_report"]
manifest = artifacts["manifest"]

print(f"Loaded metadata rows: {len(metadata)}")
print(f"Loaded embeddings shape: {embeddings.shape}, dtype={embeddings.dtype}")

validate_embeddings(metadata=metadata, embeddings=embeddings)
print("Integrity checks passed.")


# %%
candidate_group_keys = ["input_group_id", "label", "persona", "adapter_name"]
GROUP_KEY = next((k for k in candidate_group_keys if k in metadata[0]), "input_group_id")
embeddings_group_norm = normalize_within_group(embeddings, metadata=metadata, group_key=GROUP_KEY)
print(f"Group key: {GROUP_KEY}")
print(f"Group-normalized embeddings shape: {embeddings_group_norm.shape}")


# %%
# Centroid-only normalization within group (subtract mean, do not scale variance).
embeddings_group_centered = normalize_within_group(
    embeddings,
    metadata=metadata,
    group_key=GROUP_KEY,
    center=True,
    scale=False,
)
print(f"Group-centered embeddings shape: {embeddings_group_centered.shape}")


# %%
ev, common_ev = compute_eigenvalues(embeddings, standardize=True)
n_factors_kaiser = suggest_factors_kaiser(ev)
print("Suggested by Kaiser:", n_factors_kaiser)

fig_eigen_elbow = plot_eigenvalue_elbow(ev)
fig_eigen_elbow


# %%
n_factors = n_factors_kaiser if n_factors_kaiser > 0 else 20
result = run_paf(embeddings, n_factors=n_factors, rotation="varimax", standardize=True)
print(result["loadings"].shape, result["scores"].shape)


# %%
# Run PAF directly on group-centered embeddings (no extra standardization).
result_group_centered = run_paf(
    embeddings_group_centered,
    n_factors=n_factors,
    rotation="varimax",
    standardize=False,
)
print(result_group_centered["loadings"].shape, result_group_centered["scores"].shape)
