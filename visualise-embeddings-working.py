#!/usr/bin/env python3
"""Notebook-style utilities for inspecting response embedding artifacts."""

# %%

from __future__ import annotations

import json
import os
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


def ensure_embedding_artifacts(
    paths: dict[str, Path],
    *,
    repo_id: str = "persona-shattering-lasr/qwen4embeddings",
    repo_type: str = "dataset",
) -> None:
    """Download embedding artifacts from Hugging Face when local files are missing."""
    required_paths = [
        paths["metadata"],
        paths["embeddings"],
        paths["variance"],
        paths["manifest"],
    ]
    missing = [path for path in required_paths if not path.exists()]
    if not missing:
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    if load_dotenv is not None:
        load_dotenv()

    from huggingface_hub import snapshot_download

    local_dir = paths["run_dir"].parent
    run_dir_name = paths["run_dir"].name
    allow_patterns = [f"{run_dir_name}/{path.name}" for path in required_paths]

    print(
        "Missing embedding artifacts; downloading from "
        f"{repo_type} repo {repo_id} into {local_dir}"
    )
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        token=os.getenv("HF_TOKEN"),
        local_dir=local_dir,
        allow_patterns=allow_patterns,
    )

    still_missing = [path for path in required_paths if not path.exists()]
    if still_missing:
        missing_text = ", ".join(str(path) for path in still_missing)
        raise FileNotFoundError(f"Artifacts still missing after download: {missing_text}")


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


def cap_rows_per_group(
    metadata: list[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    group_key: str = "input_group_id",
    max_rows_per_group: int | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    """Keep at most max_rows_per_group rows for each metadata group."""
    if len(metadata) != len(embeddings):
        raise ValueError("metadata and embeddings must have the same number of rows")
    if max_rows_per_group is None:
        return metadata, np.asarray(embeddings), {"dropped_rows": 0, "affected_groups": {}}
    if max_rows_per_group <= 0:
        raise ValueError("max_rows_per_group must be positive when provided")

    kept_indices: list[int] = []
    seen_per_group: dict[Any, int] = {}
    dropped_per_group: dict[str, int] = {}

    for idx, row in enumerate(metadata):
        group_value = row.get(group_key)
        group_name = str(group_value)
        seen_count = seen_per_group.get(group_value, 0)
        if seen_count < max_rows_per_group:
            kept_indices.append(idx)
        else:
            dropped_per_group[group_name] = dropped_per_group.get(group_name, 0) + 1
        seen_per_group[group_value] = seen_count + 1

    filtered_metadata = [metadata[idx] for idx in kept_indices]
    filtered_embeddings = np.asarray(embeddings)[kept_indices]
    report = {
        "dropped_rows": len(metadata) - len(filtered_metadata),
        "affected_groups": dropped_per_group,
    }
    return filtered_metadata, filtered_embeddings, report


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


def summarize_eigen_analyses(
    analyses: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute eigenvalue summaries for multiple embedding matrices."""
    summary_rows: list[dict[str, Any]] = []
    for name, spec in analyses.items():
        ev, common_ev = compute_eigenvalues(spec["matrix"], standardize=spec["standardize"])
        kaiser = suggest_factors_kaiser(ev) if spec["standardize"] else None
        summary_rows.append(
            {
                "name": name,
                "title": spec["title"],
                "standardize": spec["standardize"],
                "eigenvalues": ev,
                "common_eigenvalues": common_ev,
                "kaiser_factors": kaiser,
                "top5": np.asarray(ev, dtype=np.float64)[:5],
            }
        )
    return summary_rows


def plot_eigenvalue_comparison(
    analysis_summaries: list[dict[str, Any]],
    *,
    include_kaiser_line: bool = True,
):
    """Plot scree curves for multiple analyses in one figure."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for row in analysis_summaries:
        values = np.asarray(row["eigenvalues"], dtype=np.float64).ravel()
        components = np.arange(1, values.size + 1)
        fig.add_trace(
            go.Scatter(
                x=components,
                y=values,
                mode="lines+markers",
                name=row["title"],
                line={"width": 2},
                marker={"size": 5},
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
        title="Eigenvalue Comparison Across Normalization Choices",
        xaxis_title="Component",
        yaxis_title="Eigenvalue",
        template="plotly_white",
    )
    return fig


def build_group_indices(
    metadata: list[dict[str, Any]],
    group_key: str = "input_group_id",
) -> dict[Any, np.ndarray]:
    """Map each group value to the aligned row indices in the embedding matrix."""
    groups: dict[Any, list[int]] = {}
    for idx, row in enumerate(metadata):
        groups.setdefault(row[group_key], []).append(idx)
    return {group: np.asarray(idxs, dtype=np.int64) for group, idxs in groups.items()}


def compute_top_pca_eigenvalues(
    X: np.ndarray,
    *,
    standardize: bool = True,
    top_k: int = 50,
    random_state: int = 0,
) -> np.ndarray:
    """Estimate top PCA eigenvalues via randomized SVD."""
    from sklearn.utils.extmath import randomized_svd

    X_fit = standardize_matrix(X) if standardize else np.asarray(X, dtype=np.float64)
    n_samples, n_features = X_fit.shape
    if n_samples == 0 or n_features == 0:
        return np.zeros((0,), dtype=np.float64)

    n_components = min(max(1, top_k), n_samples, n_features)
    _u, singular_values, _vt = randomized_svd(
        X_fit,
        n_components=n_components,
        random_state=random_state,
    )
    return (singular_values ** 2) / max(n_samples - 1, 1)


def sample_null_iid_gaussian(
    reference_matrix: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gaussian null with matched matrix shape."""
    return rng.standard_normal(size=reference_matrix.shape)


def sample_null_global_column_permutation(
    reference_matrix: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute each feature globally to destroy cross-feature structure."""
    X_ref = np.asarray(reference_matrix, dtype=np.float64)
    n_samples, n_features = X_ref.shape
    permuted = np.empty_like(X_ref)
    for j in range(n_features):
        permuted[:, j] = X_ref[rng.permutation(n_samples), j]
    return permuted


def sample_null_groupwise_gaussian(
    reference_matrix: np.ndarray,
    group_indices: dict[Any, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Group-aware Gaussian null preserving group sizes and within-group scales."""
    X_ref = np.asarray(reference_matrix, dtype=np.float64)
    sampled = np.empty_like(X_ref)
    for idxs in group_indices.values():
        block = X_ref[idxs]
        scale = block.std(axis=0, ddof=1)
        scale = np.where(np.isfinite(scale), scale, 0.0)
        sampled[idxs] = rng.standard_normal(size=block.shape) * scale
    return sampled


def sample_null_within_group_column_permutation(
    reference_matrix: np.ndarray,
    group_indices: dict[Any, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Group-aware column permutation null.

    This preserves each group's per-feature empirical distribution exactly, but
    it is materially slower than the Gaussian approximations on this dataset.
    """
    X_ref = np.asarray(reference_matrix, dtype=np.float64)
    permuted = np.empty_like(X_ref)
    for idxs in group_indices.values():
        block = X_ref[idxs]
        random_keys = rng.random(size=block.shape)
        permutation = np.argsort(random_keys, axis=0)
        permuted[idxs] = np.take_along_axis(block, permutation, axis=0)
    return permuted


def run_parallel_analysis(
    reference_matrix: np.ndarray,
    *,
    null_sampler,
    sampler_kwargs: dict[str, Any] | None = None,
    n_iter: int = 25,
    top_k: int = 50,
    standardize: bool = True,
    random_state: int = 0,
    name: str = "parallel_analysis",
) -> dict[str, Any]:
    """Run PCA-style parallel analysis against a chosen null model."""
    rng = np.random.default_rng(random_state)
    observed = compute_top_pca_eigenvalues(
        reference_matrix,
        standardize=standardize,
        top_k=top_k,
        random_state=random_state,
    )

    sampler_kwargs = sampler_kwargs or {}
    null_eigenvalues: list[np.ndarray] = []
    for draw_idx in range(n_iter):
        sampled = null_sampler(reference_matrix, rng=rng, **sampler_kwargs)
        null_eigenvalues.append(
            compute_top_pca_eigenvalues(
                sampled,
                standardize=standardize,
                top_k=top_k,
                random_state=random_state + draw_idx + 1,
            )
        )

    null_array = np.vstack(null_eigenvalues)
    null_mean = null_array.mean(axis=0)
    null_q95 = np.quantile(null_array, 0.95, axis=0)

    return {
        "name": name,
        "standardize": standardize,
        "top_k": top_k,
        "n_iter": n_iter,
        "observed": observed,
        "null_eigenvalues": null_array,
        "null_mean": null_mean,
        "null_q95": null_q95,
        "retain_vs_mean": int((observed > null_mean).sum()),
        "retain_vs_q95": int((observed > null_q95).sum()),
    }


def plot_parallel_analysis(
    result: dict[str, Any],
    *,
    title: str | None = None,
):
    """Plot observed and null eigenvalue curves for one parallel-analysis run."""
    import plotly.graph_objects as go

    observed = np.asarray(result["observed"], dtype=np.float64)
    null_mean = np.asarray(result["null_mean"], dtype=np.float64)
    null_q95 = np.asarray(result["null_q95"], dtype=np.float64)
    components = np.arange(1, observed.size + 1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=components,
            y=observed,
            mode="lines+markers",
            name="Observed",
            line={"width": 2},
            marker={"size": 5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=components,
            y=null_mean,
            mode="lines",
            name="Null mean",
            line={"width": 2, "dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=components,
            y=null_q95,
            mode="lines",
            name="Null 95th percentile",
            line={"width": 2, "dash": "dash"},
        )
    )
    fig.update_layout(
        title=title or f"Parallel Analysis: {result['name']}",
        xaxis_title="Component",
        yaxis_title="Eigenvalue",
        template="plotly_white",
    )
    return fig


def plot_parallel_analysis_comparison(
    results: list[dict[str, Any]],
    *,
    title: str = "Parallel Analysis Null Comparison",
):
    """Overlay observed eigenvalues with multiple null 95th-percentile curves."""
    import plotly.graph_objects as go

    if not results:
        raise ValueError("results must contain at least one parallel-analysis result")

    observed = np.asarray(results[0]["observed"], dtype=np.float64)
    components = np.arange(1, observed.size + 1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=components,
            y=observed,
            mode="lines+markers",
            name="Observed",
            line={"width": 3},
            marker={"size": 5},
        )
    )
    for result in results:
        fig.add_trace(
            go.Scatter(
                x=components,
                y=np.asarray(result["null_q95"], dtype=np.float64),
                mode="lines",
                name=f"{result['name']} q95",
                line={"width": 2, "dash": "dash"},
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Component",
        yaxis_title="Eigenvalue",
        template="plotly_white",
    )
    return fig


def compute_top_pca_basis(
    X: np.ndarray,
    *,
    standardize: bool = True,
    top_k: int = 50,
    random_state: int = 0,
) -> dict[str, np.ndarray]:
    """Return top PCA eigenvalues and orthonormal component rows."""
    from sklearn.utils.extmath import randomized_svd

    X_fit = standardize_matrix(X) if standardize else np.asarray(X, dtype=np.float64)
    n_samples, n_features = X_fit.shape
    if n_samples == 0 or n_features == 0:
        return {
            "eigenvalues": np.zeros((0,), dtype=np.float64),
            "components": np.zeros((0, n_features), dtype=np.float64),
        }

    n_components = min(max(1, top_k), n_samples, n_features)
    _u, singular_values, vt = randomized_svd(
        X_fit,
        n_components=n_components,
        random_state=random_state,
    )
    eigenvalues = (singular_values ** 2) / max(n_samples - 1, 1)
    return {
        "eigenvalues": eigenvalues,
        "components": vt,
    }


def compute_effective_rank(eigenvalues: np.ndarray) -> float:
    """Entropy-based effective rank of a non-negative spectrum."""
    values = np.clip(np.asarray(eigenvalues, dtype=np.float64), a_min=0.0, a_max=None)
    total = values.sum()
    if total <= 0:
        return 0.0
    probs = values / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    return float(np.exp(entropy))


def compute_participation_ratio(eigenvalues: np.ndarray) -> float:
    """Participation ratio of a non-negative spectrum."""
    values = np.clip(np.asarray(eigenvalues, dtype=np.float64), a_min=0.0, a_max=None)
    denom = np.square(values).sum()
    if denom <= 0:
        return 0.0
    return float(np.square(values.sum()) / denom)


def cumulative_explained_variance(eigenvalues: np.ndarray) -> np.ndarray:
    """Cumulative explained variance ratio from a non-negative spectrum."""
    values = np.clip(np.asarray(eigenvalues, dtype=np.float64), a_min=0.0, a_max=None)
    total = values.sum()
    if total <= 0:
        return np.zeros_like(values)
    return np.cumsum(values) / total


def plot_cumulative_explained_variance(
    eigenvalues: np.ndarray,
    *,
    title: str = "Cumulative Explained Variance",
):
    """Plot cumulative explained variance for a PCA spectrum."""
    import plotly.graph_objects as go

    cumulative = cumulative_explained_variance(eigenvalues)
    components = np.arange(1, cumulative.size + 1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=components,
            y=cumulative,
            mode="lines+markers",
            name="Cumulative explained variance",
            line={"width": 2},
            marker={"size": 4},
        )
    )
    for threshold in (0.5, 0.8, 0.9, 0.95):
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text=f"{int(threshold * 100)}%",
            annotation_position="bottom right",
        )
    fig.update_layout(
        title=title,
        xaxis_title="Component",
        yaxis_title="Cumulative explained variance ratio",
        template="plotly_white",
    )
    return fig


def sample_group_bootstrap_matrix(
    reference_matrix: np.ndarray,
    group_indices: dict[Any, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample prompt groups with replacement and stack their rows."""
    group_names = list(group_indices.keys())
    chosen = rng.choice(group_names, size=len(group_names), replace=True)
    sampled_blocks = [reference_matrix[group_indices[name]] for name in chosen]
    return np.vstack(sampled_blocks)


def compute_subspace_overlap(
    reference_components: np.ndarray,
    bootstrap_components: np.ndarray,
) -> dict[str, float]:
    """Compute principal-angle overlap statistics between two PCA subspaces."""
    cross = np.asarray(reference_components) @ np.asarray(bootstrap_components).T
    singular_values = np.linalg.svd(cross, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    return {
        "mean_canonical_corr": float(singular_values.mean()) if singular_values.size else 0.0,
        "min_canonical_corr": float(singular_values.min()) if singular_values.size else 0.0,
        "rms_canonical_corr": float(np.sqrt(np.mean(singular_values ** 2)))
        if singular_values.size
        else 0.0,
    }


def run_group_bootstrap_stability(
    reference_matrix: np.ndarray,
    *,
    group_indices: dict[Any, np.ndarray],
    ks: list[int],
    n_boot: int = 25,
    standardize: bool = True,
    random_state: int = 0,
) -> list[dict[str, Any]]:
    """Estimate PCA subspace stability under prompt-group bootstrap resampling."""
    if not ks:
        return []

    max_k = max(ks)
    reference_basis = compute_top_pca_basis(
        reference_matrix,
        standardize=standardize,
        top_k=max_k,
        random_state=random_state,
    )["components"]

    rng = np.random.default_rng(random_state)
    summary: list[dict[str, Any]] = []
    for k in ks:
        mean_vals: list[float] = []
        min_vals: list[float] = []
        rms_vals: list[float] = []
        ref_k = reference_basis[:k]
        for boot_idx in range(n_boot):
            boot_matrix = sample_group_bootstrap_matrix(reference_matrix, group_indices, rng)
            boot_basis = compute_top_pca_basis(
                boot_matrix,
                standardize=standardize,
                top_k=k,
                random_state=random_state + boot_idx + 1,
            )["components"]
            overlap = compute_subspace_overlap(ref_k, boot_basis)
            mean_vals.append(overlap["mean_canonical_corr"])
            min_vals.append(overlap["min_canonical_corr"])
            rms_vals.append(overlap["rms_canonical_corr"])
        summary.append(
            {
                "k": k,
                "n_boot": n_boot,
                "mean_canonical_corr_mean": float(np.mean(mean_vals)),
                "mean_canonical_corr_std": float(np.std(mean_vals, ddof=1))
                if len(mean_vals) > 1
                else 0.0,
                "min_canonical_corr_mean": float(np.mean(min_vals)),
                "rms_canonical_corr_mean": float(np.mean(rms_vals)),
            }
        )
    return summary


def plot_bootstrap_stability(
    stability_rows: list[dict[str, Any]],
    *,
    title: str = "Prompt-Group Bootstrap PCA Stability",
):
    """Plot mean canonical-correlation stability against subspace size."""
    import plotly.graph_objects as go

    ks = [row["k"] for row in stability_rows]
    means = [row["mean_canonical_corr_mean"] for row in stability_rows]
    stds = [row["mean_canonical_corr_std"] for row in stability_rows]
    mins = [row["min_canonical_corr_mean"] for row in stability_rows]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ks,
            y=means,
            mode="lines+markers",
            name="Mean canonical corr",
            error_y={"type": "data", "array": stds, "visible": True},
            line={"width": 2},
            marker={"size": 6},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ks,
            y=mins,
            mode="lines+markers",
            name="Min canonical corr",
            line={"width": 2, "dash": "dash"},
            marker={"size": 6},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Subspace size k",
        yaxis_title="Bootstrap overlap",
        template="plotly_white",
    )
    return fig


def summarize_paf_solution(
    result: dict[str, Any],
    *,
    order_by: str = "ss_loadings",
) -> dict[str, Any]:
    """Summarize and reorder a rotated PAF solution for inspection.

    For rotated factors the ordering is only a convenience. By default we sort
    by post-rotation SS loadings so the strongest factors are inspected first.
    """
    loadings = np.asarray(result["loadings"], dtype=np.float64)
    scores = np.asarray(result["scores"], dtype=np.float64)
    ss_loadings = np.square(loadings).sum(axis=0)
    if scores.shape[0] > 1:
        score_variance = scores.var(axis=0, ddof=1)
    else:
        score_variance = np.zeros((scores.shape[1],), dtype=np.float64)

    if order_by == "score_variance":
        order = np.argsort(score_variance)[::-1]
    else:
        order = np.argsort(ss_loadings)[::-1]

    loadings_sorted = loadings[:, order]
    scores_sorted = scores[:, order]
    ss_sorted = ss_loadings[order]
    score_var_sorted = score_variance[order]

    factor_rows: list[dict[str, Any]] = []
    for rank, source_idx in enumerate(order, start=1):
        factor_rows.append(
            {
                "factor_rank": rank,
                "factor_label": f"F{rank:02d}",
                "source_factor_index": int(source_idx),
                "ss_loadings": float(ss_loadings[source_idx]),
                "score_variance": float(score_variance[source_idx]),
            }
        )

    return {
        "factor_rows": factor_rows,
        "order": order,
        "loadings_sorted": loadings_sorted,
        "scores_sorted": scores_sorted,
        "ss_loadings_sorted": ss_sorted,
        "score_variance_sorted": score_var_sorted,
    }


def compute_factor_prompt_variance_summary(
    metadata: list[dict[str, Any]],
    paf_summary: dict[str, Any],
    *,
    group_key: str = "input_group_id",
) -> list[dict[str, Any]]:
    """Compute within-prompt and between-prompt variance per sorted factor."""
    group_indices = build_group_indices(metadata=metadata, group_key=group_key)
    scores = np.asarray(paf_summary["scores_sorted"], dtype=np.float64)
    factor_rows = paf_summary["factor_rows"]

    summary_rows: list[dict[str, Any]] = []
    for factor_col, factor_row in enumerate(factor_rows):
        values = scores[:, factor_col]
        if values.size > 1:
            total_variance = float(values.var(ddof=1))
        else:
            total_variance = 0.0

        prompt_means: list[float] = []
        prompt_within_vars: list[float] = []
        for idxs in group_indices.values():
            group_values = values[idxs]
            prompt_means.append(float(group_values.mean()))
            prompt_within_vars.append(
                float(group_values.var(ddof=1)) if len(idxs) > 1 else 0.0
            )

        between_prompt_variance = (
            float(np.var(prompt_means, ddof=1)) if len(prompt_means) > 1 else 0.0
        )
        within_prompt_variance = float(np.mean(prompt_within_vars)) if prompt_within_vars else 0.0

        summary_rows.append(
            {
                **factor_row,
                "total_score_variance": total_variance,
                "between_prompt_variance": between_prompt_variance,
                "within_prompt_variance": within_prompt_variance,
                "within_between_ratio": (
                    within_prompt_variance / between_prompt_variance
                    if between_prompt_variance > 0
                    else float("inf")
                ),
            }
        )

    return summary_rows


def plot_factor_strength_summary(
    factor_variance_rows: list[dict[str, Any]],
    *,
    top_n: int = 20,
    title: str = "PAF Factor Strength Summary",
):
    """Plot factor strength and prompt-variance metrics for rotated factors."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    factor_rows = factor_variance_rows[:top_n]
    labels = [row["factor_label"] for row in factor_rows]
    ss_values = [row["ss_loadings"] for row in factor_rows]
    within_values = [row["within_prompt_variance"] for row in factor_rows]
    between_values = [row["between_prompt_variance"] for row in factor_rows]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "SS loadings",
            "Within-prompt factor-score variance",
            "Between-prompt factor-score variance",
        ),
    )
    fig.add_trace(
        go.Bar(x=labels, y=ss_values, name="SS loadings"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=labels, y=within_values, name="Within-prompt variance"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=labels, y=between_values, name="Between-prompt variance"),
        row=3,
        col=1,
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        showlegend=False,
        height=900,
    )
    fig.update_yaxes(title_text="SS loadings", row=1, col=1)
    fig.update_yaxes(title_text="Within-prompt var", row=2, col=1)
    fig.update_yaxes(title_text="Between-prompt var", row=3, col=1)
    fig.update_xaxes(title_text="Factor rank (ordered by SS loadings)", row=3, col=1)
    return fig


def _format_extreme_examples_text(
    examples: list[dict[str, Any]],
    *,
    pole_label: str,
) -> str:
    """Render factor extremes into plain text for jsonl_tui variant mode."""
    lines = [f"{pole_label} pole"]
    for idx, example in enumerate(examples, start=1):
        lines.extend(
            [
                f"{idx}. score={example['factor_score']:.4f} | prompt={example['seed_user_message']}",
                example["assistant_text_excerpt"],
                "",
            ]
        )
    return "\n".join(lines).strip()


def export_factor_extremes_jsonl(
    metadata: list[dict[str, Any]],
    paf_summary: dict[str, Any],
    *,
    output_path: Path,
    top_n: int = 10,
    excerpt_chars: int = 500,
) -> tuple[Path, list[dict[str, Any]]]:
    """Export top/bottom response extremes per factor to JSONL."""
    scores = np.asarray(paf_summary["scores_sorted"], dtype=np.float64)
    factor_rows = paf_summary["factor_rows"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_rows: list[dict[str, Any]] = []

    for factor_col, factor_row in enumerate(factor_rows):
        values = scores[:, factor_col]
        positive_indices = np.argsort(values)[-top_n:][::-1]
        negative_indices = np.argsort(values)[:top_n]

        def _build_example(idx: int) -> dict[str, Any]:
            row = metadata[idx]
            full_text = str(row.get("assistant_text", ""))
            return {
                "factor_score": float(values[idx]),
                "sample_id": row.get("sample_id"),
                "input_group_id": row.get("input_group_id"),
                "response_index": row.get("response_index"),
                "assistant_turn_index": row.get("assistant_turn_index"),
                "seed_user_message": row.get("seed_user_message"),
                "preceding_user_message": row.get("preceding_user_message"),
                "assistant_message_id": row.get("assistant_message_id"),
                "assistant_text_excerpt": full_text[:excerpt_chars],
                "assistant_text": full_text,
            }

        positive_examples = [_build_example(int(idx)) for idx in positive_indices]
        negative_examples = [_build_example(int(idx)) for idx in negative_indices]

        export_rows.append(
            {
                **factor_row,
                "positive_examples": positive_examples,
                "negative_examples": negative_examples,
                "positive_examples_text": _format_extreme_examples_text(
                    positive_examples,
                    pole_label="Positive",
                ),
                "negative_examples_text": _format_extreme_examples_text(
                    negative_examples,
                    pole_label="Negative",
                ),
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in export_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return output_path, export_rows


def export_factor_extremes_tui_jsonl(
    metadata: list[dict[str, Any]],
    paf_summary: dict[str, Any],
    *,
    output_path: Path,
    top_n: int = 10,
    excerpt_chars: int = 1000,
) -> tuple[Path, list[dict[str, Any]]]:
    """Export factor extremes in a shape that matches jsonl_tui default navigation.

    One "question" group corresponds to one factor pole, e.g. "F01 high".
    Left/right then moves between examples within that factor-pole group.
    """
    scores = np.asarray(paf_summary["scores_sorted"], dtype=np.float64)
    factor_rows = paf_summary["factor_rows"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_rows: list[dict[str, Any]] = []

    for factor_col, factor_row in enumerate(factor_rows):
        values = scores[:, factor_col]
        poles = {
            "high": np.argsort(values)[-top_n:][::-1],
            "low": np.argsort(values)[:top_n],
        }
        for pole_name, indices in poles.items():
            question_label = f"{factor_row['factor_label']} {pole_name}"
            for example_rank, idx in enumerate(indices, start=1):
                row = metadata[int(idx)]
                full_text = str(row.get("assistant_text", ""))
                export_rows.append(
                    {
                        "question": question_label,
                        "response_index": example_rank - 1,
                        "factor_label": factor_row["factor_label"],
                        "factor_rank": factor_row["factor_rank"],
                        "source_factor_index": factor_row["source_factor_index"],
                        "pole": pole_name,
                        "example_rank": example_rank,
                        "factor_score": float(values[int(idx)]),
                        "ss_loadings": factor_row["ss_loadings"],
                        "sample_id": row.get("sample_id"),
                        "input_group_id": row.get("input_group_id"),
                        "assistant_message_id": row.get("assistant_message_id"),
                        "seed_user_message": row.get("seed_user_message"),
                        "preceding_user_message": row.get("preceding_user_message"),
                        "assistant_turn_index": row.get("assistant_turn_index"),
                        "assistant_text_excerpt": full_text[:excerpt_chars],
                        "assistant_text": full_text,
                    }
                )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in export_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return output_path, export_rows


def compute_prompt_factor_summary(
    metadata: list[dict[str, Any]],
    paf_summary: dict[str, Any],
    *,
    group_key: str = "input_group_id",
) -> list[dict[str, Any]]:
    """Aggregate sorted factor scores to prompt-level means and spreads."""
    group_indices = build_group_indices(metadata=metadata, group_key=group_key)
    scores = np.asarray(paf_summary["scores_sorted"], dtype=np.float64)
    factor_rows = paf_summary["factor_rows"]

    prompt_rows: list[dict[str, Any]] = []
    for group_value, idxs in group_indices.items():
        first = metadata[int(idxs[0])]
        values = scores[idxs]
        row: dict[str, Any] = {
            "input_group_id": str(group_value),
            "seed_user_message": str(first.get("seed_user_message", "")),
            "num_rows": int(len(idxs)),
        }
        for factor_col, factor_row in enumerate(factor_rows):
            label = factor_row["factor_label"]
            factor_values = values[:, factor_col]
            row[f"{label}_mean"] = float(factor_values.mean())
            row[f"{label}_std"] = float(factor_values.std(ddof=1)) if len(idxs) > 1 else 0.0
        prompt_rows.append(row)

    return prompt_rows


def compute_factor_prompt_variance_rankings(
    metadata: list[dict[str, Any]],
    paf_summary: dict[str, Any],
    *,
    group_key: str = "input_group_id",
) -> list[dict[str, Any]]:
    """Rank prompts by within-prompt variance for each sorted factor."""
    group_indices = build_group_indices(metadata=metadata, group_key=group_key)
    scores = np.asarray(paf_summary["scores_sorted"], dtype=np.float64)
    factor_rows = paf_summary["factor_rows"]

    rankings: list[dict[str, Any]] = []
    for factor_col, factor_row in enumerate(factor_rows):
        factor_values = scores[:, factor_col]
        prompt_rows: list[dict[str, Any]] = []
        for group_value, idxs in group_indices.items():
            values = factor_values[idxs]
            first = metadata[int(idxs[0])]
            variance = float(values.var(ddof=1)) if len(idxs) > 1 else 0.0
            prompt_rows.append(
                {
                    "input_group_id": str(group_value),
                    "seed_user_message": str(first.get("seed_user_message", "")),
                    "num_rows": int(len(idxs)),
                    "variance": variance,
                    "indices": idxs.copy(),
                }
            )
        prompt_rows.sort(key=lambda row: row["variance"], reverse=True)
        rankings.append(
            {
                **factor_row,
                "prompt_rankings": prompt_rows,
            }
        )
    return rankings


def export_factor_prompt_variance_extremes_tui_jsonl_prompt_split(
    metadata: list[dict[str, Any]],
    paf_summary: dict[str, Any],
    prompt_variance_rankings: list[dict[str, Any]],
    *,
    output_path: Path,
    top_k_prompts: int = 3,
    top_n: int = 10,
    excerpt_chars: int = 1000,
) -> tuple[Path, list[dict[str, Any]]]:
    """Export extremes with one TUI row per factor/prompt/pole split."""
    scores = np.asarray(paf_summary["scores_sorted"], dtype=np.float64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_rows: list[dict[str, Any]] = []

    for factor_col, factor_ranking in enumerate(prompt_variance_rankings):
        prompt_rankings = factor_ranking["prompt_rankings"]
        if not prompt_rankings:
            continue
        for prompt_rank, selected_prompt in enumerate(prompt_rankings[:top_k_prompts], start=1):
            prompt_indices = np.asarray(selected_prompt["indices"], dtype=np.int64)
            prompt_scores = scores[prompt_indices, factor_col]

            poles = {
                "high": np.argsort(prompt_scores)[-top_n:][::-1],
                "low": np.argsort(prompt_scores)[:top_n],
            }
            for pole_name, local_indices in poles.items():
                question_label = (
                    f"{factor_ranking['factor_label']} "
                    f"prompt{prompt_rank:02d} {pole_name}"
                )
                for example_rank, local_idx in enumerate(local_indices, start=1):
                    global_idx = int(prompt_indices[int(local_idx)])
                    row = metadata[global_idx]
                    full_text = str(row.get("assistant_text", ""))
                    export_rows.append(
                        {
                            "question": question_label,
                            "response_index": example_rank - 1,
                            "factor_label": factor_ranking["factor_label"],
                            "factor_rank": factor_ranking["factor_rank"],
                            "source_factor_index": factor_ranking["source_factor_index"],
                            "pole": pole_name,
                            "example_rank": example_rank,
                            "factor_score": float(scores[global_idx, factor_col]),
                            "selected_prompt_rank": prompt_rank,
                            "selected_prompt_variance": float(selected_prompt["variance"]),
                            "selected_prompt_num_rows": int(selected_prompt["num_rows"]),
                            "selected_prompt_input_group_id": selected_prompt["input_group_id"],
                            "selected_prompt_text": selected_prompt["seed_user_message"],
                            "sample_id": row.get("sample_id"),
                            "input_group_id": row.get("input_group_id"),
                            "assistant_message_id": row.get("assistant_message_id"),
                            "seed_user_message": row.get("seed_user_message"),
                            "preceding_user_message": row.get("preceding_user_message"),
                            "assistant_turn_index": row.get("assistant_turn_index"),
                            "assistant_text_excerpt": full_text[:excerpt_chars],
                            "assistant_text": full_text,
                        }
                    )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in export_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return output_path, export_rows


def export_factor_prompt_variance_extremes_tui_jsonl_factor_split(
    metadata: list[dict[str, Any]],
    paf_summary: dict[str, Any],
    prompt_variance_rankings: list[dict[str, Any]],
    *,
    output_path: Path,
    top_k_prompts: int = 3,
    top_n: int = 10,
    excerpt_chars: int = 1000,
) -> tuple[Path, list[dict[str, Any]]]:
    """Export extremes with one TUI row per factor/pole, combining top-k prompts."""
    scores = np.asarray(paf_summary["scores_sorted"], dtype=np.float64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_rows: list[dict[str, Any]] = []

    for factor_col, factor_ranking in enumerate(prompt_variance_rankings):
        prompt_rankings = factor_ranking["prompt_rankings"]
        if not prompt_rankings:
            continue
        for pole_name in ("high", "low"):
            question_label = f"{factor_ranking['factor_label']} {pole_name}"
            response_index = 0
            for prompt_rank, selected_prompt in enumerate(prompt_rankings[:top_k_prompts], start=1):
                prompt_indices = np.asarray(selected_prompt["indices"], dtype=np.int64)
                prompt_scores = scores[prompt_indices, factor_col]
                if pole_name == "high":
                    local_indices = np.argsort(prompt_scores)[-top_n:][::-1]
                else:
                    local_indices = np.argsort(prompt_scores)[:top_n]

                for example_rank_within_prompt, local_idx in enumerate(local_indices, start=1):
                    global_idx = int(prompt_indices[int(local_idx)])
                    row = metadata[global_idx]
                    full_text = str(row.get("assistant_text", ""))
                    export_rows.append(
                        {
                            "question": question_label,
                            "response_index": response_index,
                            "factor_label": factor_ranking["factor_label"],
                            "factor_rank": factor_ranking["factor_rank"],
                            "source_factor_index": factor_ranking["source_factor_index"],
                            "pole": pole_name,
                            "example_rank": response_index + 1,
                            "example_rank_within_prompt": example_rank_within_prompt,
                            "factor_score": float(scores[global_idx, factor_col]),
                            "selected_prompt_rank": prompt_rank,
                            "selected_prompt_variance": float(selected_prompt["variance"]),
                            "selected_prompt_num_rows": int(selected_prompt["num_rows"]),
                            "selected_prompt_input_group_id": selected_prompt["input_group_id"],
                            "selected_prompt_text": selected_prompt["seed_user_message"],
                            "sample_id": row.get("sample_id"),
                            "input_group_id": row.get("input_group_id"),
                            "assistant_message_id": row.get("assistant_message_id"),
                            "seed_user_message": row.get("seed_user_message"),
                            "preceding_user_message": row.get("preceding_user_message"),
                            "assistant_turn_index": row.get("assistant_turn_index"),
                            "assistant_text_excerpt": full_text[:excerpt_chars],
                            "assistant_text": full_text,
                        }
                    )
                    response_index += 1

    with output_path.open("w", encoding="utf-8") as handle:
        for row in export_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return output_path, export_rows


def plot_prompt_factor_heatmap(
    prompt_rows: list[dict[str, Any]],
    paf_summary: dict[str, Any],
    *,
    metric: str = "mean",
    top_n_factors: int = 12,
    top_n_prompts: int = 30,
    title: str | None = None,
):
    """Plot prompt-by-factor heatmap for prompt-level factor summaries."""
    import plotly.graph_objects as go

    factor_rows = paf_summary["factor_rows"][:top_n_factors]
    factor_labels = [row["factor_label"] for row in factor_rows]
    suffix = "mean" if metric == "mean" else "std"
    matrix = np.array(
        [
            [prompt_row[f"{label}_{suffix}"] for label in factor_labels]
            for prompt_row in prompt_rows
        ],
        dtype=np.float64,
    )
    if matrix.size == 0:
        raise ValueError("prompt_rows must not be empty")

    if metric == "mean":
        prompt_order = np.argsort(np.max(np.abs(matrix), axis=1))[::-1][:top_n_prompts]
        color_midpoint = 0.0
        colorscale = "RdBu"
    else:
        prompt_order = np.argsort(np.max(matrix, axis=1))[::-1][:top_n_prompts]
        color_midpoint = None
        colorscale = "Viridis"

    prompt_labels = [
        str(prompt_rows[idx]["seed_user_message"])[:90]
        for idx in prompt_order
    ]
    selected_matrix = matrix[prompt_order]

    fig = go.Figure(
        data=go.Heatmap(
            z=selected_matrix,
            x=factor_labels,
            y=prompt_labels,
            colorscale=colorscale,
            zmid=color_midpoint,
            colorbar={"title": f"Prompt {suffix}"},
        )
    )
    fig.update_layout(
        title=title or f"Prompt-level factor {suffix}s",
        xaxis_title="Factor rank (ordered by SS loadings)",
        yaxis_title="Prompt",
        template="plotly_white",
        height=max(500, 18 * len(prompt_labels)),
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
HF_REPO_ID = "persona-shattering-lasr/qwen4embeddings"
HF_REPO_TYPE = "dataset"

paths = build_run_paths(run_id=RUN_ID, prefix=PREFIX)
print(f"Run dir: {paths['run_dir']}")
print(f"Metadata: {paths['metadata']}")
print(f"Embeddings: {paths['embeddings']}")
print(f"Variance: {paths['variance']}")
print(f"Manifest: {paths['manifest']}")
ensure_embedding_artifacts(paths, repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)


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

# Cap overfull prompt groups so duplicated source questions do not dominate.
MAX_ROWS_PER_GROUP = 50
metadata, embeddings, cap_report = cap_rows_per_group(
    metadata,
    embeddings,
    group_key=GROUP_KEY,
    max_rows_per_group=MAX_ROWS_PER_GROUP,
)
print(
    f"After capping {GROUP_KEY} to {MAX_ROWS_PER_GROUP} rows: "
    f"{len(metadata)} rows, dropped {cap_report['dropped_rows']}"
)
if cap_report["affected_groups"]:
    print(f"Affected groups: {cap_report['affected_groups']}")

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
analysis_specs = {
    "raw_corr": {
        "matrix": embeddings,
        "standardize": True,
        "title": "Raw embeddings + global standardization",
    },
    "group_centered_cov": {
        "matrix": embeddings_group_centered,
        "standardize": False,
        "title": "Group-centered only (covariance view)",
    },
    "group_centered_corr": {
        "matrix": embeddings_group_centered,
        "standardize": True,
        "title": "Group-centered + global standardization",
    },
}
analysis_summaries = summarize_eigen_analyses(analysis_specs)
for row in analysis_summaries:
    kaiser_text = (
        f"Kaiser factors={row['kaiser_factors']}"
        if row["kaiser_factors"] is not None
        else "Kaiser not applicable"
    )
    print(
        f"{row['title']}: standardize={row['standardize']} | "
        f"top5={np.array2string(row['top5'], precision=3)} | {kaiser_text}"
    )

fig_eigen_comparison = plot_eigenvalue_comparison(analysis_summaries)
fig_eigen_comparison


# %%
analysis_by_name = {row["name"]: row for row in analysis_summaries}
fig_eigen_raw = plot_eigenvalue_elbow(
    analysis_by_name["raw_corr"]["eigenvalues"],
    title=analysis_by_name["raw_corr"]["title"],
)
fig_eigen_raw


# %%
fig_eigen_group_centered_cov = plot_eigenvalue_elbow(
    analysis_by_name["group_centered_cov"]["eigenvalues"],
    title=analysis_by_name["group_centered_cov"]["title"],
)
fig_eigen_group_centered_cov


# %%
fig_eigen_group_centered_corr = plot_eigenvalue_elbow(
    analysis_by_name["group_centered_corr"]["eigenvalues"],
    title=analysis_by_name["group_centered_corr"]["title"],
)
fig_eigen_group_centered_corr


# %%
if 0: # Don't redo it wastes time
    n_factors_kaiser_corr = analysis_by_name["group_centered_corr"]["kaiser_factors"] or 0
    n_factors = n_factors_kaiser_corr if n_factors_kaiser_corr > 0 else 20
    print(f"Chosen n_factors from group-centered standardized spectrum: {n_factors}")

n_factors = 30 # To save time

# %%
result_raw = run_paf(embeddings, n_factors=n_factors, rotation="varimax", standardize=True)
print("Raw embeddings + global standardization:", result_raw["loadings"].shape, result_raw["scores"].shape)


# %%
if 0: # Don't redo it wastes time
    # Run PAF on group-centered embeddings with no extra scaling (covariance-style view).
    result_group_centered_cov = run_paf(
        embeddings_group_centered,
        n_factors=n_factors,
        rotation="oblimin",
        standardize=False,
    )
    print(
        "Group-centered only (covariance view):",
        result_group_centered_cov["loadings"].shape,
        result_group_centered_cov["scores"].shape,
    )


# %%
# Run PAF on group-centered embeddings with global standardization (Kaiser-ready view).
result_group_centered_corr = run_paf(
    embeddings_group_centered,
    n_factors=n_factors,
    rotation="oblimin",
    standardize=True,
)
print(
    "Group-centered + global standardization:",
    result_group_centered_corr["loadings"].shape,
    result_group_centered_corr["scores"].shape,
)


# %%
# PAF factor inspection below uses the rotated factor solution directly.
# The later parallel-analysis / bootstrap sections are PCA-based diagnostics.
PAF_INSPECTION_RESULT = result_group_centered_corr
PAF_INSPECTION_NAME = "group_centered_corr_varimax"
paf_inspection_summary = summarize_paf_solution(PAF_INSPECTION_RESULT, order_by="ss_loadings")
paf_factor_variance_rows = compute_factor_prompt_variance_summary(
    metadata,
    paf_inspection_summary,
    group_key=GROUP_KEY,
)
print(
    f"Inspecting rotated PAF solution: {PAF_INSPECTION_NAME} | "
    f"ordered by post-rotation SS loadings"
)
print("Factor sign is arbitrary; positive/negative poles are for inspection only.")


# %%
# Visualisation 1: factor strength summary for the strongest rotated factors.
fig_factor_strengths = plot_factor_strength_summary(
    paf_factor_variance_rows,
    top_n=min(20, len(paf_factor_variance_rows)),
    title="Rotated PAF Factor Strengths: Group-centered + Global Standardization",
)
fig_factor_strengths


# %%
# Visualisation 2: export top/bottom response extremes per factor to JSONL for
# browsing in the JSONL TUI. The default TUI groups by the `question` field, so
# we export one record per example with question labels like "F01 high".
FACTOR_EXTREMES_TOP_N = 10
tui_factor_extremes_path = (
    Path("scratch")
    / "factor_inspection"
    / RUN_ID
    / f"{PAF_INSPECTION_NAME}_factor_extremes_tui_top{FACTOR_EXTREMES_TOP_N}.jsonl"
)
tui_factor_extremes_path, tui_factor_extremes_rows = export_factor_extremes_tui_jsonl(
    metadata,
    paf_inspection_summary,
    output_path=tui_factor_extremes_path,
    top_n=FACTOR_EXTREMES_TOP_N,
)
print(f"Wrote TUI-friendly factor extremes to: {tui_factor_extremes_path}")
print(
    "View in jsonl_tui:\n"
    f"uv run python scripts/jsonl_tui/cli.py {tui_factor_extremes_path} --variant-fields question seed_user_message assistant_text"
)


# %%
# Optional companion export: one row per factor containing nested high/low lists.
# This is less convenient for navigation, but useful if you want the grouped
# examples in one JSON object.
factor_extremes_path = (
    Path("scratch")
    / "factor_inspection"
    / RUN_ID
    / f"{PAF_INSPECTION_NAME}_factor_extremes_top{FACTOR_EXTREMES_TOP_N}.jsonl"
)
factor_extremes_path, factor_extremes_rows = export_factor_extremes_jsonl(
    metadata,
    paf_inspection_summary,
    output_path=factor_extremes_path,
    top_n=FACTOR_EXTREMES_TOP_N,
)
print(f"Wrote nested factor extremes to: {factor_extremes_path}")


# %%
# Visualisation 2b: for each factor, find the prompt with the largest within-
# prompt score variance, print the top prompt variances, and export top/bottom
# responses from the top-k prompts to a TUI-friendly JSONL.
FACTOR_PROMPT_VARIANCE_TOP_N = 5
FACTOR_PROMPT_VARIANCE_TOP_K_PROMPTS = 3
FACTOR_PROMPT_VARIANCE_EXPORT_LAYOUT = "factor_split"
factor_prompt_variance_rankings = compute_factor_prompt_variance_rankings(
    metadata,
    paf_inspection_summary,
    group_key=GROUP_KEY,
)
for factor_row in factor_prompt_variance_rankings:
    print(f"{factor_row['factor_label']} top prompt variances:")
    for prompt_rank, prompt_row in enumerate(
        factor_row["prompt_rankings"][:FACTOR_PROMPT_VARIANCE_TOP_N],
        start=1,
    ):
        print(
            f"  {prompt_rank}. var={prompt_row['variance']:.4f} | "
            f"n={prompt_row['num_rows']} | prompt={prompt_row['seed_user_message']}"
        )

factor_promptmaxvar_base_dir = (
    Path("scratch")
    / "factor_inspection"
    / RUN_ID
)

factor_promptmaxvar_promptsplit_path = (
    factor_promptmaxvar_base_dir
    / (
        f"{PAF_INSPECTION_NAME}_factor_promptmaxvar_promptsplit_tui_"
        f"top{FACTOR_EXTREMES_TOP_N}_k{FACTOR_PROMPT_VARIANCE_TOP_K_PROMPTS}.jsonl"
    )
)
(
    factor_promptmaxvar_promptsplit_path,
    factor_promptmaxvar_promptsplit_rows,
) = export_factor_prompt_variance_extremes_tui_jsonl_prompt_split(
    metadata,
    paf_inspection_summary,
    factor_prompt_variance_rankings,
    output_path=factor_promptmaxvar_promptsplit_path,
    top_k_prompts=FACTOR_PROMPT_VARIANCE_TOP_K_PROMPTS,
    top_n=FACTOR_EXTREMES_TOP_N,
)

factor_promptmaxvar_factorsplit_path = (
    factor_promptmaxvar_base_dir
    / (
        f"{PAF_INSPECTION_NAME}_factor_promptmaxvar_factorsplit_tui_"
        f"top{FACTOR_EXTREMES_TOP_N}_k{FACTOR_PROMPT_VARIANCE_TOP_K_PROMPTS}.jsonl"
    )
)
(
    factor_promptmaxvar_factorsplit_path,
    factor_promptmaxvar_factorsplit_rows,
) = export_factor_prompt_variance_extremes_tui_jsonl_factor_split(
    metadata,
    paf_inspection_summary,
    factor_prompt_variance_rankings,
    output_path=factor_promptmaxvar_factorsplit_path,
    top_k_prompts=FACTOR_PROMPT_VARIANCE_TOP_K_PROMPTS,
    top_n=FACTOR_EXTREMES_TOP_N,
)

factor_promptmaxvar_default_path = (
    factor_promptmaxvar_factorsplit_path
    if FACTOR_PROMPT_VARIANCE_EXPORT_LAYOUT == "factor_split"
    else factor_promptmaxvar_promptsplit_path
)
print(f"Wrote prompt-split max-variance-prompt extremes to: {factor_promptmaxvar_promptsplit_path}")
print(f"Wrote factor-split max-variance-prompt extremes to: {factor_promptmaxvar_factorsplit_path}")
print(f"Default layout: {FACTOR_PROMPT_VARIANCE_EXPORT_LAYOUT}")
print(
    "View default in jsonl_tui:\n"
    f"uv run python scripts/jsonl_tui/cli.py {factor_promptmaxvar_default_path} "
    "--variant-fields question seed_user_message assistant_text"
)


# %%
# Visualisation 3: prompt-level factor summaries. Mean heatmap highlights prompts
# whose average score on a factor is most extreme; std heatmap highlights prompts
# with the most within-prompt spread along a factor.
prompt_factor_rows = compute_prompt_factor_summary(
    metadata,
    paf_inspection_summary,
    group_key=GROUP_KEY,
)
fig_prompt_factor_means = plot_prompt_factor_heatmap(
    prompt_factor_rows,
    paf_inspection_summary,
    metric="mean",
    top_n_factors=min(12, len(paf_inspection_summary["factor_rows"])),
    top_n_prompts=30,
    title="Prompt-level PAF Means for Strongest Factors",
)
fig_prompt_factor_means


# %%
fig_prompt_factor_stds = plot_prompt_factor_heatmap(
    prompt_factor_rows,
    paf_inspection_summary,
    metric="std",
    top_n_factors=min(12, len(paf_inspection_summary["factor_rows"])),
    top_n_prompts=30,
    title="Prompt-level PAF Within-prompt Std for Strongest Factors",
)
fig_prompt_factor_stds


# %%
# Parallel analysis is run on the same matrix used for the Kaiser-ready view:
# prompt-group centered first, then global standardization inside the helper.
PA_REFERENCE = embeddings_group_centered
PA_GROUP_INDICES = build_group_indices(metadata=metadata, group_key=GROUP_KEY)
PA_TOP_K = 400
PA_N_ITER = 25
PA_RANDOM_STATE = 0
print(
    f"Parallel analysis setup: top_k={PA_TOP_K}, n_iter={PA_N_ITER}, "
    f"groups={len(PA_GROUP_INDICES)}, samples={PA_REFERENCE.shape[0]}"
)


# %%
RUN_PARALLEL_ANALYSIS = False
# Null 1: iid Gaussian. Fast baseline, ignores the prompt-grouped design.
if RUN_PARALLEL_ANALYSIS:
    pa_iid_gaussian = run_parallel_analysis(
        PA_REFERENCE,
        null_sampler=sample_null_iid_gaussian,
        n_iter=PA_N_ITER,
        top_k=PA_TOP_K,
        standardize=True,
        random_state=PA_RANDOM_STATE,
        name="iid_gaussian",
    )
    print(
        "iid_gaussian:",
        f"retain_vs_mean={pa_iid_gaussian['retain_vs_mean']}",
        f"retain_vs_q95={pa_iid_gaussian['retain_vs_q95']}",
    )
    plot_parallel_analysis(pa_iid_gaussian, title="Parallel Analysis: iid Gaussian null")


# %%
# Null 2: globally permute each feature on the group-centered matrix.
# This preserves each feature's empirical marginal distribution but ignores groups.
if RUN_PARALLEL_ANALYSIS:
    pa_global_permutation = run_parallel_analysis(
        PA_REFERENCE,
        null_sampler=sample_null_global_column_permutation,
        n_iter=PA_N_ITER,
        top_k=PA_TOP_K,
        standardize=True,
        random_state=PA_RANDOM_STATE,
        name="global_column_permutation",
    )
    print(
        "global_column_permutation:",
        f"retain_vs_mean={pa_global_permutation['retain_vs_mean']}",
        f"retain_vs_q95={pa_global_permutation['retain_vs_q95']}",
    )
    plot_parallel_analysis(
        pa_global_permutation,
        title="Parallel Analysis: global column permutation null",
    )


# %%
# Null 3: group-aware Gaussian null. This preserves prompt group sizes and each
# prompt's per-feature residual scale, while removing cross-feature structure.
if RUN_PARALLEL_ANALYSIS:
    pa_groupwise_gaussian = run_parallel_analysis(
        PA_REFERENCE,
        null_sampler=sample_null_groupwise_gaussian,
        sampler_kwargs={"group_indices": PA_GROUP_INDICES},
        n_iter=PA_N_ITER,
        top_k=PA_TOP_K,
        standardize=True,
        random_state=PA_RANDOM_STATE,
        name="groupwise_gaussian",
    )
    print(
        "groupwise_gaussian:",
        f"retain_vs_mean={pa_groupwise_gaussian['retain_vs_mean']}",
        f"retain_vs_q95={pa_groupwise_gaussian['retain_vs_q95']}",
    )
    plot_parallel_analysis(
        pa_groupwise_gaussian,
        title="Parallel Analysis: group-aware Gaussian null",
    )


# %%
# Optional slower null: permute each feature independently within each prompt group.
# This preserves each prompt's empirical per-feature distribution exactly, but it is
# significantly slower on the full 11,950 x 2,560 matrix.
if RUN_PARALLEL_ANALYSIS:
    RUN_SLOW_WITHIN_GROUP_PERMUTATION = True
    PA_SLOW_N_ITER = 25
    if RUN_SLOW_WITHIN_GROUP_PERMUTATION:
        pa_within_group_permutation = run_parallel_analysis(
            PA_REFERENCE,
            null_sampler=sample_null_within_group_column_permutation,
            sampler_kwargs={"group_indices": PA_GROUP_INDICES},
            n_iter=PA_SLOW_N_ITER,
            top_k=PA_TOP_K,
            standardize=True,
            random_state=PA_RANDOM_STATE,
            name="within_group_column_permutation",
        )
        print(
            "within_group_column_permutation:",
            f"retain_vs_mean={pa_within_group_permutation['retain_vs_mean']}",
            f"retain_vs_q95={pa_within_group_permutation['retain_vs_q95']}",
        )
        plot_parallel_analysis(
            pa_within_group_permutation,
            title="Parallel Analysis: within-group column permutation null",
        )
    else:
        pa_within_group_permutation = None
        print(
            "Skipping within-group column permutation null. "
            "Set RUN_SLOW_WITHIN_GROUP_PERMUTATION = True to run it."
        )


# %%
if RUN_PARALLEL_ANALYSIS:
    parallel_results = [
        pa_iid_gaussian,
        pa_global_permutation,
        pa_groupwise_gaussian,
    ]
    if pa_within_group_permutation is not None:
        parallel_results.append(pa_within_group_permutation)

    for result in parallel_results:
        print(
            f"{result['name']}: retain_vs_mean={result['retain_vs_mean']} | "
            f"retain_vs_q95={result['retain_vs_q95']}"
        )

    plot_parallel_analysis_comparison(parallel_results)


# %%
# Softer summaries of residualized dimensionality for the group-centered,
# globally standardized view.
SOFT_RANK_TOP_K = 400
soft_rank_basis = compute_top_pca_basis(
    PA_REFERENCE,
    standardize=True,
    top_k=SOFT_RANK_TOP_K,
    random_state=0,
)
soft_rank_eigenvalues = soft_rank_basis["eigenvalues"]
soft_rank_cumulative = cumulative_explained_variance(soft_rank_eigenvalues)
effective_rank = compute_effective_rank(soft_rank_eigenvalues)
participation_ratio = compute_participation_ratio(soft_rank_eigenvalues)

def _first_index_at_or_above(values: np.ndarray, threshold: float) -> int | None:
    hits = np.flatnonzero(values >= threshold)
    if hits.size == 0:
        return None
    return int(hits[0] + 1)

print(f"Effective rank (top {SOFT_RANK_TOP_K} eigs): {effective_rank:.2f}")
print(f"Participation ratio (top {SOFT_RANK_TOP_K} eigs): {participation_ratio:.2f}")
for threshold in (0.5, 0.8, 0.9, 0.95):
    idx = _first_index_at_or_above(soft_rank_cumulative, threshold)
    print(f"Components to reach {int(threshold * 100)}% variance: {idx}")


# %%
plot_cumulative_explained_variance(
    soft_rank_eigenvalues,
    title="Cumulative Explained Variance: Group-centered + Global Standardization",
)


# %%
BOOTSTRAP_STABILITY = False
if BOOTSTRAP_STABILITY:
    # Prompt-group bootstrap stability: resample whole prompt groups with replacement
    # and compare the resulting top-k PCA subspaces to the reference subspace.
    BOOTSTRAP_KS = [5, 10, 20, 40, 80, 120]
    BOOTSTRAP_N = 25
    bootstrap_stability = run_group_bootstrap_stability(
        PA_REFERENCE,
        group_indices=PA_GROUP_INDICES,
        ks=BOOTSTRAP_KS,
        n_boot=BOOTSTRAP_N,
        standardize=True,
        random_state=0,
    )
    for row in bootstrap_stability:
        print(
            f"k={row['k']}: "
            f"mean={row['mean_canonical_corr_mean']:.3f} +/- {row['mean_canonical_corr_std']:.3f} | "
            f"min={row['min_canonical_corr_mean']:.3f} | "
            f"rms={row['rms_canonical_corr_mean']:.3f}"
        )


# %%
if BOOTSTRAP_STABILITY:
    plot_bootstrap_stability(bootstrap_stability)


# %%
# Lexical probe for rotated PAF factors using the same embedding model family.
# This is an approximation: factors were fit on prompt-residualized response
# embeddings, whereas the probe below scores standalone single-token embeddings.
# It is therefore best treated as a lexical probe of the factor directions, not a
# literal inverse mapping from factor space back to text.
if 0:
    import heapq

    import torch
    from transformers import AutoModel, AutoTokenizer

    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - notebook convenience only
        pd = None


    def _top_heap_push(
        heap: list[tuple[float, int, dict[str, Any]]],
        item: tuple[float, int, dict[str, Any]],
        keep_n: int,
    ) -> None:
        if len(heap) < keep_n:
            heapq.heappush(heap, item)
            return
        if item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)


    def _mean_pool_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool token embeddings over unmasked positions."""
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        masked_hidden = hidden_states * mask
        denom = mask.sum(dim=1).clamp_min(1)
        return masked_hidden.sum(dim=1) / denom


    TOKEN_PROBE_MODEL = "Qwen/Qwen3-Embedding-4B"
    TOKEN_PROBE_DTYPE = "bfloat16"
    TOKEN_PROBE_DEVICE_MAP = "auto"
    TOKEN_PROBE_BATCH_SIZE = 2048
    TOKEN_PROBE_TOP_N = 20
    TOKEN_PROBE_FACTOR_RANKS = list(range(1, min(11, len(paf_inspection_summary["factor_rows"]) + 1)))
    TOKEN_PROBE_SELECTIVITY_PENALTY = 0.75
    TOKEN_PROBE_MAX_TOKEN_IDS = None  # Set e.g. 50000 for a quicker pass.

    # Recreate the scaler used for the group-centered standardized PAF fit.
    token_probe_scaler = StandardScaler().fit(np.asarray(embeddings_group_centered, dtype=np.float64))
    token_probe_factor_model = PAF_INSPECTION_RESULT["model"]
    token_probe_order = np.asarray(paf_inspection_summary["order"], dtype=np.int64)

    token_probe_dtype = getattr(torch, TOKEN_PROBE_DTYPE, None)
    if token_probe_dtype is None:
        raise ValueError(f"Unsupported TOKEN_PROBE_DTYPE: {TOKEN_PROBE_DTYPE}")
    if not torch.cuda.is_available() and token_probe_dtype in {torch.float16, torch.bfloat16}:
        token_probe_dtype = torch.float32

    token_probe_tokenizer = AutoTokenizer.from_pretrained(
        TOKEN_PROBE_MODEL,
        use_fast=True,
        trust_remote_code=True,
    )
    token_probe_model = AutoModel.from_pretrained(
        TOKEN_PROBE_MODEL,
        torch_dtype=token_probe_dtype,
        device_map=TOKEN_PROBE_DEVICE_MAP,
        trust_remote_code=True,
    )
    token_probe_model.eval()
    token_probe_device = next(token_probe_model.parameters()).device

    special_ids = set(token_probe_tokenizer.all_special_ids or [])
    candidate_token_ids = [tok_id for tok_id in range(token_probe_tokenizer.vocab_size) if tok_id not in special_ids]
    if TOKEN_PROBE_MAX_TOKEN_IDS is not None:
        candidate_token_ids = candidate_token_ids[:TOKEN_PROBE_MAX_TOKEN_IDS]
    print(f"Scoring {len(candidate_token_ids)} token ids with model={TOKEN_PROBE_MODEL}")

    factor_rank_to_col = {row["factor_rank"]: idx for idx, row in enumerate(paf_inspection_summary["factor_rows"])}
    selected_factor_ranks = [rank for rank in TOKEN_PROBE_FACTOR_RANKS if rank in factor_rank_to_col]
    token_probe_heaps: dict[tuple[int, str], list[tuple[float, int, dict[str, Any]]]] = {}
    for factor_rank in selected_factor_ranks:
        token_probe_heaps[(factor_rank, "high")] = []
        token_probe_heaps[(factor_rank, "low")] = []

    for start in range(0, len(candidate_token_ids), TOKEN_PROBE_BATCH_SIZE):
        batch_ids = candidate_token_ids[start : start + TOKEN_PROBE_BATCH_SIZE]
        input_ids = torch.tensor(batch_ids, dtype=torch.long, device=token_probe_device).unsqueeze(1)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            output = token_probe_model(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(output, "last_hidden_state"):
                pooled = _mean_pool_hidden(output.last_hidden_state, attention_mask)
            elif isinstance(output, tuple) and output:
                pooled = _mean_pool_hidden(output[0], attention_mask)
            else:
                raise ValueError("Token probe model output missing last_hidden_state")
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        batch_embeddings = pooled.detach().cpu().to(torch.float32).numpy()
        batch_standardized = token_probe_scaler.transform(np.asarray(batch_embeddings, dtype=np.float64))
        batch_scores = token_probe_factor_model.transform(batch_standardized)[:, token_probe_order]

        for factor_rank in selected_factor_ranks:
            factor_col = factor_rank_to_col[factor_rank]
            factor_values = batch_scores[:, factor_col]
            if batch_scores.shape[1] > 1:
                other_cols = np.delete(np.arange(batch_scores.shape[1]), factor_col)
                other_rms = np.sqrt(np.mean(np.square(batch_scores[:, other_cols]), axis=1))
            else:
                other_rms = np.zeros_like(factor_values)

            high_objective = factor_values - TOKEN_PROBE_SELECTIVITY_PENALTY * other_rms
            low_objective = (-factor_values) - TOKEN_PROBE_SELECTIVITY_PENALTY * other_rms

            for batch_pos, token_id in enumerate(batch_ids):
                raw_token = token_probe_tokenizer.convert_ids_to_tokens(int(token_id))
                decoded_text = token_probe_tokenizer.decode(
                    [int(token_id)],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                payload = {
                    "factor_rank": factor_rank,
                    "factor_label": f"F{factor_rank:02d}",
                    "token_id": int(token_id),
                    "token_str": str(raw_token),
                    "decoded_text": repr(decoded_text),
                    "factor_score": float(factor_values[batch_pos]),
                    "other_factor_rms": float(other_rms[batch_pos]),
                }
                _top_heap_push(
                    token_probe_heaps[(factor_rank, "high")],
                    (
                        float(high_objective[batch_pos]),
                        int(token_id),
                        {
                            **payload,
                            "pole": "high",
                            "selective_objective": float(high_objective[batch_pos]),
                        },
                    ),
                    TOKEN_PROBE_TOP_N,
                )
                _top_heap_push(
                    token_probe_heaps[(factor_rank, "low")],
                    (
                        float(low_objective[batch_pos]),
                        int(token_id),
                        {
                            **payload,
                            "pole": "low",
                            "selective_objective": float(low_objective[batch_pos]),
                        },
                    ),
                    TOKEN_PROBE_TOP_N,
                )

    token_probe_rows: list[dict[str, Any]] = []
    for factor_rank in selected_factor_ranks:
        for pole_name in ("high", "low"):
            ranked = sorted(
                token_probe_heaps[(factor_rank, pole_name)],
                key=lambda item: item[0],
                reverse=True,
            )
            for rank_within_pole, (_objective, _token_id, payload) in enumerate(ranked, start=1):
                token_probe_rows.append(
                    {
                        **payload,
                        "rank_within_pole": rank_within_pole,
                    }
                )

    if pd is not None:
        for factor_rank in selected_factor_ranks:
            display(
                pd.DataFrame(
                    [
                        row
                        for row in token_probe_rows
                        if row["factor_rank"] == factor_rank
                    ]
                )[
                    [
                        "factor_label",
                        "pole",
                        "rank_within_pole",
                        "token_id",
                        "token_str",
                        "decoded_text",
                        "factor_score",
                        "other_factor_rms",
                        "selective_objective",
                    ]
                ]
            )
    else:
        for factor_rank in selected_factor_ranks:
            print(f"Factor F{factor_rank:02d}")
            for pole_name in ("high", "low"):
                print(f"  {pole_name}")
                for row in token_probe_rows:
                    if row["factor_rank"] == factor_rank and row["pole"] == pole_name:
                        print(
                            "   ",
                            row["rank_within_pole"],
                            row["token_id"],
                            row["token_str"],
                            row["decoded_text"],
                            f"score={row['factor_score']:.3f}",
                            f"others={row['other_factor_rms']:.3f}",
                            f"obj={row['selective_objective']:.3f}",
                        )

# %%
# Gradient-based factor-selective optimization helpers and cells.
def _display_rows(
    rows: list[dict[str, Any]],
    *,
    columns: list[str] | None = None,
    title: str | None = None,
    max_rows: int | None = None,
) -> None:
    """Display rows as a DataFrame when available, else print them."""
    if title:
        print(title)

    visible_rows = rows[:max_rows] if max_rows is not None else rows
    if not visible_rows:
        print("(no rows)")
        return

    if pd is not None:
        frame = pd.DataFrame(visible_rows)
        if columns is not None:
            existing_columns = [column for column in columns if column in frame.columns]
            frame = frame[existing_columns]
        if "display" in globals():
            display(frame)
        else:
            print(frame.to_string(index=False))
        return

    for row in visible_rows:
        if columns is None:
            print(row)
        else:
            print({column: row.get(column) for column in columns})


def build_factor_score_affine_map(
    factor_model: Any,
    factor_order: np.ndarray,
    *,
    n_features: int,
    batch_size: int = 128,
) -> dict[str, np.ndarray]:
    """Build a projection map from standardized embeddings to rotated factor directions.

    This intentionally uses the rotated loading vectors directly rather than
    trying to reverse-engineer `factor_model.transform(...)`. The resulting
    scores are directional projection scores in the standardized embedding space,
    which makes the optimization objective explicit and geometrically faithful to
    the rotated factor solution used for inspection.
    """
    factor_order = np.asarray(factor_order, dtype=np.int64)
    loadings = np.asarray(factor_model.loadings_, dtype=np.float64)
    if loadings.shape[0] != n_features:
        raise ValueError(
            f"Expected {n_features} features in loadings, got {loadings.shape[0]}"
        )

    weights = np.asarray(loadings[:, factor_order], dtype=np.float64)
    column_norms = np.linalg.norm(weights, axis=0, keepdims=True)
    column_norms = np.clip(column_norms, a_min=1e-12, a_max=None)
    weights = weights / column_norms
    intercept = np.zeros((weights.shape[1],), dtype=np.float64)

    return {
        "weights": weights,
        "intercept": intercept,
    }


def score_ordered_factors_from_embeddings(
    raw_embeddings: np.ndarray,
    *,
    scaler: StandardScaler,
    factor_score_map: dict[str, np.ndarray],
) -> np.ndarray:
    """Score raw embeddings against the ordered rotated factors."""
    raw_embeddings = np.asarray(raw_embeddings, dtype=np.float64)
    standardized = scaler.transform(raw_embeddings)
    return (
        standardized @ np.asarray(factor_score_map["weights"], dtype=np.float64)
        + np.asarray(factor_score_map["intercept"], dtype=np.float64)
    )


def score_ordered_factors_from_standardized_torch(
    standardized_embeddings: torch.Tensor,
    *,
    factor_score_weights: torch.Tensor,
    factor_score_intercept: torch.Tensor,
) -> torch.Tensor:
    """Torch scorer matching the notebook's ordered rotated factor scores."""
    return standardized_embeddings @ factor_score_weights + factor_score_intercept


def build_factor_objective(
    ordered_scores: torch.Tensor,
    *,
    target_factor_col: int,
    pole: str,
    selectivity_penalty: float,
    candidate_standardized: torch.Tensor | None = None,
    standardized_anchor: torch.Tensor | None = None,
    manifold_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Compute a scalar objective that favors one factor and suppresses others."""
    target_scores = ordered_scores[:, target_factor_col]
    if pole == "low":
        target_scores = -target_scores
    elif pole != "high":
        raise ValueError(f"Unsupported pole: {pole!r}")

    if ordered_scores.shape[1] > 1:
        other_cols = [idx for idx in range(ordered_scores.shape[1]) if idx != target_factor_col]
        other_scores = ordered_scores[:, other_cols]
        other_rms = torch.sqrt(torch.mean(other_scores.square(), dim=1) + 1e-8)
    else:
        other_rms = torch.zeros_like(target_scores)

    manifold_penalty = torch.zeros_like(target_scores)
    if candidate_standardized is not None and standardized_anchor is not None and manifold_weight > 0:
        manifold_penalty = manifold_weight * torch.mean(
            (candidate_standardized - standardized_anchor).square(),
            dim=1,
        )

    objective = target_scores - selectivity_penalty * other_rms - manifold_penalty
    return {
        "objective": objective,
        "target_score": target_scores,
        "other_rms": other_rms,
        "manifold_penalty": manifold_penalty,
    }


def find_nearest_real_examples(
    query_raw_embedding: np.ndarray,
    *,
    reference_embeddings: np.ndarray,
    metadata_rows: list[dict[str, Any]],
    factor_scores_sorted: np.ndarray,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Return nearest real examples by cosine similarity for interpretation."""
    query = np.asarray(query_raw_embedding, dtype=np.float64).reshape(1, -1)
    refs = np.asarray(reference_embeddings, dtype=np.float64)

    query_norm = np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-12)
    ref_norms = np.linalg.norm(refs, axis=1, keepdims=True).clip(min=1e-12)
    similarities = ((query / query_norm) @ (refs / ref_norms).T)[0]

    best_indices = np.argsort(similarities)[-top_n:][::-1]
    rows: list[dict[str, Any]] = []
    for rank, idx in enumerate(best_indices, start=1):
        row = metadata_rows[int(idx)]
        rows.append(
            {
                "rank": rank,
                "row_index": int(idx),
                "cosine_similarity": float(similarities[idx]),
                "question": row.get("question"),
                "seed_user_message": row.get("seed_user_message"),
                "assistant_text": row.get("assistant_text"),
                "label": row.get("label"),
                "persona": row.get("persona"),
                "top_factor_score_abs": float(np.max(np.abs(factor_scores_sorted[idx]))),
            }
        )
    return rows


def optimize_factor_embedding(
    *,
    factor_rank: int,
    pole: str,
    standardized_matrix: np.ndarray,
    scaler: StandardScaler,
    factor_score_map: dict[str, np.ndarray],
    metadata_rows: list[dict[str, Any]],
    reference_embeddings: np.ndarray,
    reference_scores_sorted: np.ndarray,
    selectivity_penalty: float,
    manifold_weight: float,
    num_steps: int,
    learning_rate: float,
    random_seed: int,
    nearest_top_n: int = 5,
) -> dict[str, Any]:
    """Optimize a single embedding vector for factor selectivity."""
    standardized_matrix = np.asarray(standardized_matrix, dtype=np.float64)
    if factor_rank < 1 or factor_rank > factor_score_map["weights"].shape[1]:
        raise ValueError(f"factor_rank must be in [1, {factor_score_map['weights'].shape[1]}]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    factor_col = factor_rank - 1
    anchor_np = standardized_matrix.mean(axis=0, keepdims=True)
    start_np = anchor_np + 0.01 * np.random.randn(*anchor_np.shape)

    candidate = torch.tensor(start_np, dtype=torch.float32, device=device, requires_grad=True)
    anchor = torch.tensor(anchor_np, dtype=torch.float32, device=device)
    weights = torch.tensor(
        factor_score_map["weights"],
        dtype=torch.float32,
        device=device,
    )
    intercept = torch.tensor(
        factor_score_map["intercept"],
        dtype=torch.float32,
        device=device,
    )

    optimizer = torch.optim.Adam([candidate], lr=learning_rate)
    history_rows: list[dict[str, Any]] = []

    best_snapshot: dict[str, Any] | None = None
    best_objective = -float("inf")

    for step in range(1, num_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        ordered_scores = score_ordered_factors_from_standardized_torch(
            candidate,
            factor_score_weights=weights,
            factor_score_intercept=intercept,
        )
        objective_parts = build_factor_objective(
            ordered_scores,
            target_factor_col=factor_col,
            pole=pole,
            selectivity_penalty=selectivity_penalty,
            candidate_standardized=candidate,
            standardized_anchor=anchor,
            manifold_weight=manifold_weight,
        )
        loss = -objective_parts["objective"].mean()
        loss.backward()
        optimizer.step()
        if not torch.isfinite(candidate).all():
            print(f"Non-finite candidate encountered at step {step}; stopping early.")
            break

        step_row = {
            "step": step,
            "objective": float(objective_parts["objective"].detach().cpu().item()),
            "target_score": float(objective_parts["target_score"].detach().cpu().item()),
            "other_factor_rms": float(objective_parts["other_rms"].detach().cpu().item()),
            "manifold_penalty": float(objective_parts["manifold_penalty"].detach().cpu().item()),
        }
        history_rows.append(step_row)
        if step_row["objective"] > best_objective:
            best_objective = step_row["objective"]
            best_snapshot = {
                "step": step,
                "standardized_embedding": candidate.detach().cpu().numpy().copy(),
                "ordered_scores": ordered_scores.detach().cpu().numpy().copy(),
                "summary": step_row,
            }

    if best_snapshot is None:
        raise RuntimeError("Embedding optimization failed to produce any snapshots")

    best_standardized = np.asarray(best_snapshot["standardized_embedding"], dtype=np.float64)
    best_raw = scaler.inverse_transform(best_standardized)
    if not np.isfinite(best_raw).all():
        nearest_rows = []
    else:
        nearest_rows = find_nearest_real_examples(
            best_raw[0],
            reference_embeddings=reference_embeddings,
            metadata_rows=metadata_rows,
            factor_scores_sorted=reference_scores_sorted,
            top_n=nearest_top_n,
        )

    return {
        "factor_rank": factor_rank,
        "pole": pole,
        "target_factor_col": factor_col,
        "history_rows": history_rows,
        "best_step": int(best_snapshot["step"]),
        "best_standardized_embedding": best_standardized,
        "best_raw_embedding": best_raw,
        "best_ordered_scores": np.asarray(best_snapshot["ordered_scores"], dtype=np.float64),
        "best_summary": best_snapshot["summary"],
        "nearest_rows": nearest_rows,
    }


def _get_token_probe_resources(
    *,
    model_name: str,
    torch_dtype_name: str,
    device_map: str,
) -> tuple[Any, Any, torch.device]:
    """Reuse lexical-probe tokenizer/model when present, else load them."""
    if {
        "token_probe_tokenizer",
        "token_probe_model",
        "token_probe_device",
    }.issubset(globals()):
        return token_probe_tokenizer, token_probe_model, token_probe_device

    probe_dtype = getattr(torch, torch_dtype_name, None)
    if probe_dtype is None:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype_name}")
    if not torch.cuda.is_available() and probe_dtype in {torch.float16, torch.bfloat16}:
        probe_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=probe_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    return tokenizer, model, device


def decode_and_rescore_sequence(
    token_ids: list[int],
    *,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    scaler: StandardScaler,
    factor_score_map: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Decode discrete token ids and rescore the resulting sequence."""
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(output, "last_hidden_state"):
            pooled = _mean_pool_hidden(output.last_hidden_state, attention_mask)
        elif isinstance(output, tuple) and output:
            pooled = _mean_pool_hidden(output[0], attention_mask)
        else:
            raise ValueError("Sequence scorer output missing last_hidden_state")
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1).to(torch.float32)

    raw_embedding = pooled.detach().cpu().numpy()
    ordered_scores = score_ordered_factors_from_embeddings(
        raw_embedding,
        scaler=scaler,
        factor_score_map=factor_score_map,
    )
    return {
        "token_ids": [int(token_id) for token_id in token_ids],
        "token_strs": [str(tokenizer.convert_ids_to_tokens(int(token_id))) for token_id in token_ids],
        "decoded_text": tokenizer.decode(
            [int(token_id) for token_id in token_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        "raw_embedding": raw_embedding,
        "ordered_scores": ordered_scores,
    }


def optimize_relaxed_token_sequence(
    *,
    factor_rank: int,
    pole: str,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    sequence_length: int,
    scaler: StandardScaler,
    factor_score_map: dict[str, np.ndarray],
    standardized_anchor: np.ndarray,
    selectivity_penalty: float,
    manifold_weight: float,
    entropy_weight_start: float,
    entropy_weight_end: float,
    repeat_penalty_weight: float,
    temperature_start: float,
    temperature_end: float,
    num_steps: int,
    learning_rate: float,
    random_seed: int,
) -> dict[str, Any]:
    """Optimize a differentiable sequence of token distributions."""
    if factor_rank < 1 or factor_rank > factor_score_map["weights"].shape[1]:
        raise ValueError(f"factor_rank must be in [1, {factor_score_map['weights'].shape[1]}]")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    embedding_layer = model.get_input_embeddings()
    vocab_size = int(embedding_layer.num_embeddings)
    factor_col = factor_rank - 1
    special_ids = {
        int(token_id)
        for token_id in (tokenizer.all_special_ids or [])
        if 0 <= int(token_id) < vocab_size
    }

    logits = torch.zeros(
        (sequence_length, vocab_size),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    torch.nn.init.normal_(logits, mean=0.0, std=0.01)

    special_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    if special_ids:
        special_mask[sorted(special_ids)] = True

    anchor = torch.tensor(standardized_anchor, dtype=torch.float32, device=device)
    weights = torch.tensor(
        factor_score_map["weights"],
        dtype=torch.float32,
        device=device,
    )
    intercept = torch.tensor(
        factor_score_map["intercept"],
        dtype=torch.float32,
        device=device,
    )
    scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)

    embedding_weight = embedding_layer.weight.detach().to(torch.float32)
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    history_rows: list[dict[str, Any]] = []
    best_snapshot: dict[str, Any] | None = None
    best_objective = -float("inf")

    for step in range(1, num_steps + 1):
        optimizer.zero_grad(set_to_none=True)

        progress = 0.0 if num_steps <= 1 else (step - 1) / (num_steps - 1)
        temperature = temperature_start + (temperature_end - temperature_start) * progress
        entropy_weight = entropy_weight_start + (entropy_weight_end - entropy_weight_start) * progress

        masked_logits = logits.masked_fill(special_mask.unsqueeze(0), -1e9)
        probs = torch.softmax(masked_logits / max(temperature, 1e-4), dim=-1)
        inputs_embeds = probs @ embedding_weight
        attention_mask = torch.ones((1, sequence_length), dtype=torch.long, device=device)

        output = model(
            inputs_embeds=inputs_embeds.unsqueeze(0).to(dtype=model.dtype),
            attention_mask=attention_mask,
        )
        if hasattr(output, "last_hidden_state"):
            pooled = _mean_pool_hidden(output.last_hidden_state, attention_mask)
        elif isinstance(output, tuple) and output:
            pooled = _mean_pool_hidden(output[0], attention_mask)
        else:
            raise ValueError("Relaxed sequence model output missing last_hidden_state")

        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1).to(torch.float32)
        standardized = (pooled - scaler_mean.unsqueeze(0)) / scaler_scale.unsqueeze(0)
        ordered_scores = score_ordered_factors_from_standardized_torch(
            standardized,
            factor_score_weights=weights,
            factor_score_intercept=intercept,
        )
        factor_parts = build_factor_objective(
            ordered_scores,
            target_factor_col=factor_col,
            pole=pole,
            selectivity_penalty=selectivity_penalty,
            candidate_standardized=standardized,
            standardized_anchor=anchor,
            manifold_weight=manifold_weight,
        )

        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
        position_similarity = probs @ probs.T
        repeat_penalty = (
            position_similarity.sum() - torch.diagonal(position_similarity).sum()
        ) / max(sequence_length * (sequence_length - 1), 1)

        total_objective = (
            factor_parts["objective"]
            - entropy_weight * entropy
            - repeat_penalty_weight * repeat_penalty
        )

        loss = -total_objective.mean()
        loss.backward()
        optimizer.step()

        step_row = {
            "step": step,
            "temperature": float(temperature),
            "objective": float(total_objective.detach().cpu().item()),
            "factor_objective": float(factor_parts["objective"].detach().cpu().item()),
            "target_score": float(factor_parts["target_score"].detach().cpu().item()),
            "other_factor_rms": float(factor_parts["other_rms"].detach().cpu().item()),
            "manifold_penalty": float(factor_parts["manifold_penalty"].detach().cpu().item()),
            "entropy": float(entropy.detach().cpu().item()),
            "repeat_penalty": float(repeat_penalty.detach().cpu().item()),
        }
        history_rows.append(step_row)
        if step_row["objective"] > best_objective:
            best_objective = step_row["objective"]
            best_snapshot = {
                "step": step,
                "logits": masked_logits.detach().cpu().numpy().copy(),
                "probs": probs.detach().cpu().numpy().copy(),
                "raw_embedding": pooled.detach().cpu().numpy().copy(),
                "standardized_embedding": standardized.detach().cpu().numpy().copy(),
                "ordered_scores": ordered_scores.detach().cpu().numpy().copy(),
                "summary": step_row,
            }

    if best_snapshot is None:
        raise RuntimeError("Sequence optimization failed to produce any snapshots")

    best_logits = np.asarray(best_snapshot["logits"], dtype=np.float64)
    best_token_ids = np.argmax(best_logits, axis=1).tolist()
    discrete_result = decode_and_rescore_sequence(
        best_token_ids,
        tokenizer=tokenizer,
        model=model,
        device=device,
        scaler=scaler,
        factor_score_map=factor_score_map,
    )

    return {
        "factor_rank": factor_rank,
        "pole": pole,
        "target_factor_col": factor_col,
        "history_rows": history_rows,
        "best_step": int(best_snapshot["step"]),
        "best_logits": best_logits,
        "best_probs": np.asarray(best_snapshot["probs"], dtype=np.float64),
        "best_raw_embedding": np.asarray(best_snapshot["raw_embedding"], dtype=np.float64),
        "best_standardized_embedding": np.asarray(best_snapshot["standardized_embedding"], dtype=np.float64),
        "best_ordered_scores": np.asarray(best_snapshot["ordered_scores"], dtype=np.float64),
        "best_summary": best_snapshot["summary"],
        "discrete_result": discrete_result,
    }


def summarize_factor_solution(
    *,
    name: str,
    factor_rank: int,
    pole: str,
    ordered_scores: np.ndarray,
    factor_rows: list[dict[str, Any]],
    objective_value: float,
    other_factor_rms: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a flat summary row for optimized solutions."""
    ordered_scores = np.asarray(ordered_scores, dtype=np.float64).reshape(-1)
    target_col = factor_rank - 1
    target_value = float(ordered_scores[target_col])
    if pole == "low":
        target_value = -target_value

    non_target = [
        (
            factor_rows[idx]["factor_label"],
            float(ordered_scores[idx]),
        )
        for idx in range(len(factor_rows))
        if idx != target_col
    ]
    non_target.sort(key=lambda item: abs(item[1]), reverse=True)
    row = {
        "name": name,
        "factor_label": f"F{factor_rank:02d}",
        "pole": pole,
        "target_score_aligned": target_value,
        "other_factor_rms": float(other_factor_rms),
        "objective": float(objective_value),
        "top_off_target_factor": non_target[0][0] if non_target else None,
        "top_off_target_score": non_target[0][1] if non_target else None,
    }
    if extra:
        row.update(extra)
    return row


# %%
OPT_FACTOR_RANK = 1
OPT_POLE = "high"  # "high" or "low"
OPT_SEQUENCE_LENGTH = 400
OPT_SELECTIVITY_PENALTY = 0.75
OPT_NUM_STEPS = 200
OPT_LR = 0.05
OPT_RANDOM_SEED = 0

OPT_AFFINE_BATCH_SIZE = 128
OPT_NEAREST_TOP_N = 5
OPT_EMBEDDING_MANIFOLD_WEIGHT = 0.05
OPT_SEQUENCE_MANIFOLD_WEIGHT = 0.05
OPT_SEQUENCE_ENTROPY_WEIGHT_START = 0.005
OPT_SEQUENCE_ENTROPY_WEIGHT_END = 0.05
OPT_SEQUENCE_REPEAT_PENALTY = 0.05
OPT_SEQUENCE_TEMPERATURE_START = 2.0
OPT_SEQUENCE_TEMPERATURE_END = 0.25

opt_factor_rank_to_col = {
    row["factor_rank"]: idx
    for idx, row in enumerate(paf_inspection_summary["factor_rows"])
}
if OPT_FACTOR_RANK not in opt_factor_rank_to_col:
    raise ValueError(
        f"OPT_FACTOR_RANK={OPT_FACTOR_RANK} not present in inspected factors; "
        f"available ranks: {sorted(opt_factor_rank_to_col)}"
    )

opt_factor_scaler = StandardScaler().fit(np.asarray(embeddings_group_centered, dtype=np.float64))
opt_factor_score_map = build_factor_score_affine_map(
    PAF_INSPECTION_RESULT["model"],
    np.asarray(paf_inspection_summary["order"], dtype=np.int64),
    n_features=embeddings_group_centered.shape[1],
    batch_size=OPT_AFFINE_BATCH_SIZE,
)
opt_standardized_matrix = opt_factor_scaler.transform(np.asarray(embeddings_group_centered, dtype=np.float64))
opt_standardized_anchor = opt_standardized_matrix.mean(axis=0, keepdims=True)
print(
    f"Optimization target: factor=F{OPT_FACTOR_RANK:02d} pole={OPT_POLE} | "
    f"sequence_length={OPT_SEQUENCE_LENGTH} | steps={OPT_NUM_STEPS} | lr={OPT_LR}"
)


# %%
# Directional diagnostic: compare loading-projection scores against the fitted
# factor scores over the training matrix. These are not expected to match
# exactly; the goal is to see whether the projection proxy is at least aligned.
opt_projection_scores = score_ordered_factors_from_embeddings(
    np.asarray(embeddings_group_centered, dtype=np.float64),
    scaler=opt_factor_scaler,
    factor_score_map=opt_factor_score_map,
)
opt_factor_scores_true = np.asarray(paf_inspection_summary["scores_sorted"], dtype=np.float64)
opt_projection_alignment_rows: list[dict[str, Any]] = []
for factor_col, factor_row in enumerate(paf_inspection_summary["factor_rows"]):
    proxy_values = opt_projection_scores[:, factor_col]
    true_values = opt_factor_scores_true[:, factor_col]
    corr = float(np.corrcoef(proxy_values, true_values)[0, 1])
    mae = float(np.mean(np.abs(proxy_values - true_values)))
    opt_projection_alignment_rows.append(
        {
            "factor_label": factor_row["factor_label"],
            "corr_with_paf_score": corr,
            "mae_vs_paf_score": mae,
        }
    )

_display_rows(
    opt_projection_alignment_rows[:10],
    columns=["factor_label", "corr_with_paf_score", "mae_vs_paf_score"],
    title="Loading-projection alignment diagnostic (first 10 factors)",
)


# %%
# Optimize a continuous embedding vector directly in the standardized embedding
# space used by the rotated PAF scorer.
embedding_optimization_result = optimize_factor_embedding(
    factor_rank=OPT_FACTOR_RANK,
    pole=OPT_POLE,
    standardized_matrix=opt_standardized_matrix,
    scaler=opt_factor_scaler,
    factor_score_map=opt_factor_score_map,
    metadata_rows=metadata,
    reference_embeddings=np.asarray(embeddings_group_centered, dtype=np.float64),
    reference_scores_sorted=np.asarray(paf_inspection_summary["scores_sorted"], dtype=np.float64),
    selectivity_penalty=OPT_SELECTIVITY_PENALTY,
    manifold_weight=OPT_EMBEDDING_MANIFOLD_WEIGHT,
    num_steps=OPT_NUM_STEPS,
    learning_rate=OPT_LR,
    random_seed=OPT_RANDOM_SEED,
    nearest_top_n=OPT_NEAREST_TOP_N,
)
_display_rows(
    embedding_optimization_result["history_rows"][:: max(OPT_NUM_STEPS // 20, 1)],
    columns=["step", "objective", "target_score", "other_factor_rms", "manifold_penalty"],
    title=(
        f"Embedding optimization history for F{OPT_FACTOR_RANK:02d} ({OPT_POLE}) "
        f"| best_step={embedding_optimization_result['best_step']}"
    ),
)
_display_rows(
    embedding_optimization_result["nearest_rows"],
    columns=[
        "rank",
        "row_index",
        "cosine_similarity",
        "question",
        "seed_user_message",
        "assistant_text",
    ],
    title="Nearest real responses to optimized embedding",
)
print("Best embedding-space summary:", embedding_optimization_result["best_summary"])


# %%
# Optimize a relaxed fixed-length token sequence with differentiable vocab
# distributions, then discretize and rescore the final sequence.
(
    opt_sequence_tokenizer,
    opt_sequence_model,
    opt_sequence_device,
) = _get_token_probe_resources(
    model_name=TOKEN_PROBE_MODEL,
    torch_dtype_name=TOKEN_PROBE_DTYPE,
    device_map=TOKEN_PROBE_DEVICE_MAP,
)
sequence_optimization_result = optimize_relaxed_token_sequence(
    factor_rank=OPT_FACTOR_RANK,
    pole=OPT_POLE,
    tokenizer=opt_sequence_tokenizer,
    model=opt_sequence_model,
    device=opt_sequence_device,
    sequence_length=OPT_SEQUENCE_LENGTH,
    scaler=opt_factor_scaler,
    factor_score_map=opt_factor_score_map,
    standardized_anchor=opt_standardized_anchor,
    selectivity_penalty=OPT_SELECTIVITY_PENALTY,
    manifold_weight=OPT_SEQUENCE_MANIFOLD_WEIGHT,
    entropy_weight_start=OPT_SEQUENCE_ENTROPY_WEIGHT_START,
    entropy_weight_end=OPT_SEQUENCE_ENTROPY_WEIGHT_END,
    repeat_penalty_weight=OPT_SEQUENCE_REPEAT_PENALTY,
    temperature_start=OPT_SEQUENCE_TEMPERATURE_START,
    temperature_end=OPT_SEQUENCE_TEMPERATURE_END,
    num_steps=OPT_NUM_STEPS,
    learning_rate=OPT_LR,
    random_seed=OPT_RANDOM_SEED,
)
_display_rows(
    sequence_optimization_result["history_rows"][:: max(OPT_NUM_STEPS // 20, 1)],
    columns=[
        "step",
        "temperature",
        "objective",
        "factor_objective",
        "target_score",
        "other_factor_rms",
        "entropy",
        "repeat_penalty",
    ],
    title=(
        f"Relaxed sequence optimization history for F{OPT_FACTOR_RANK:02d} ({OPT_POLE}) "
        f"| best_step={sequence_optimization_result['best_step']}"
    ),
)
print(
    "Discrete sequence:",
    repr(sequence_optimization_result["discrete_result"]["decoded_text"]),
)
print(
    "Discrete token strings:",
    sequence_optimization_result["discrete_result"]["token_strs"],
)
print("Best relaxed-sequence summary:", sequence_optimization_result["best_summary"])


# %%
# Compare the optimized embedding vector and optimized discrete sequence side by
# side on the same factor objective.
embedding_summary_row = summarize_factor_solution(
    name="embedding_space",
    factor_rank=OPT_FACTOR_RANK,
    pole=OPT_POLE,
    ordered_scores=embedding_optimization_result["best_ordered_scores"],
    factor_rows=paf_inspection_summary["factor_rows"],
    objective_value=embedding_optimization_result["best_summary"]["objective"],
    other_factor_rms=embedding_optimization_result["best_summary"]["other_factor_rms"],
)
sequence_relaxed_summary_row = summarize_factor_solution(
    name="sequence_relaxed",
    factor_rank=OPT_FACTOR_RANK,
    pole=OPT_POLE,
    ordered_scores=sequence_optimization_result["best_ordered_scores"],
    factor_rows=paf_inspection_summary["factor_rows"],
    objective_value=sequence_optimization_result["best_summary"]["objective"],
    other_factor_rms=sequence_optimization_result["best_summary"]["other_factor_rms"],
    extra={"decoded_text": repr(sequence_optimization_result["discrete_result"]["decoded_text"])},
)
sequence_discrete_scores = np.asarray(
    sequence_optimization_result["discrete_result"]["ordered_scores"],
    dtype=np.float64,
)
sequence_discrete_target_col = OPT_FACTOR_RANK - 1
if sequence_discrete_scores.shape[1] > 1:
    sequence_discrete_other_rms = float(
        np.sqrt(
            np.mean(
                np.square(
                    np.delete(sequence_discrete_scores[0], sequence_discrete_target_col)
                )
            )
        )
    )
else:
    sequence_discrete_other_rms = 0.0
sequence_discrete_target = float(sequence_discrete_scores[0, sequence_discrete_target_col])
if OPT_POLE == "low":
    sequence_discrete_target = -sequence_discrete_target
sequence_discrete_objective = (
    sequence_discrete_target - OPT_SELECTIVITY_PENALTY * sequence_discrete_other_rms
)
sequence_discrete_summary_row = summarize_factor_solution(
    name="sequence_discrete",
    factor_rank=OPT_FACTOR_RANK,
    pole=OPT_POLE,
    ordered_scores=sequence_discrete_scores,
    factor_rows=paf_inspection_summary["factor_rows"],
    objective_value=sequence_discrete_objective,
    other_factor_rms=sequence_discrete_other_rms,
    extra={"decoded_text": repr(sequence_optimization_result["discrete_result"]["decoded_text"])},
)
_display_rows(
    [
        embedding_summary_row,
        sequence_relaxed_summary_row,
        sequence_discrete_summary_row,
    ],
    columns=[
        "name",
        "factor_label",
        "pole",
        "target_score_aligned",
        "other_factor_rms",
        "objective",
        "top_off_target_factor",
        "top_off_target_score",
        "decoded_text",
    ],
    title="Factor-selective optimization comparison",
)
print(
    "Discretization objective drop:",
    f"{sequence_relaxed_summary_row['objective'] - sequence_discrete_summary_row['objective']:.4f}",
)
# %%
# %%
# Export the final factor-selective optimization comparison to JSONL for jsonl_tui.
comparison_export_dir = Path("scratch") / "factor_optimization" / RUN_ID
comparison_export_dir.mkdir(parents=True, exist_ok=True)

comparison_export_path = comparison_export_dir / (
    f"{PAF_INSPECTION_NAME}_factor_{OPT_FACTOR_RANK:02d}_{OPT_POLE}_comparison.jsonl"
)

nearest_embedding_row = (
    embedding_optimization_result["nearest_rows"][0]
    if embedding_optimization_result.get("nearest_rows")
    else {}
)

comparison_export_rows = [
    {
        **embedding_summary_row,
        "question": f"embedding_space F{OPT_FACTOR_RANK:02d} {OPT_POLE}",
        "seed_user_message": nearest_embedding_row.get("seed_user_message"),
        "assistant_text": nearest_embedding_row.get("assistant_text"),
        "row_index": nearest_embedding_row.get("row_index"),
        "cosine_similarity": nearest_embedding_row.get("cosine_similarity"),
        "decoded_text": None,
    },
    {
        **sequence_relaxed_summary_row,
        "question": f"sequence_relaxed F{OPT_FACTOR_RANK:02d} {OPT_POLE}",
        "seed_user_message": None,
        "assistant_text": sequence_relaxed_summary_row.get("decoded_text"),
    },
    {
        **sequence_discrete_summary_row,
        "question": f"sequence_discrete F{OPT_FACTOR_RANK:02d} {OPT_POLE}",
        "seed_user_message": None,
        "assistant_text": sequence_discrete_summary_row.get("decoded_text"),
    },
]

with comparison_export_path.open("w", encoding="utf-8") as handle:
    for row in comparison_export_rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Wrote comparison JSONL to: {comparison_export_path}")
print(
    "View in jsonl_tui:\n"
    f"uv run python scripts/jsonl_tui/cli.py {comparison_export_path} "
    "--variant-fields question seed_user_message assistant_text"
)

# %%
