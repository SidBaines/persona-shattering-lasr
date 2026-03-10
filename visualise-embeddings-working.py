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
n_factors_kaiser_corr = analysis_by_name["group_centered_corr"]["kaiser_factors"] or 0
n_factors = n_factors_kaiser_corr if n_factors_kaiser_corr > 0 else 20
print(f"Chosen n_factors from group-centered standardized spectrum: {n_factors}")

n_factors = 30 # To save time

# %%
result_raw = run_paf(embeddings, n_factors=n_factors, rotation="varimax", standardize=True)
print("Raw embeddings + global standardization:", result_raw["loadings"].shape, result_raw["scores"].shape)


# %%
# Run PAF on group-centered embeddings with no extra scaling (covariance-style view).
result_group_centered_cov = run_paf(
    embeddings_group_centered,
    n_factors=n_factors,
    rotation="varimax",
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
    rotation="varimax",
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
# Null 1: iid Gaussian. Fast baseline, ignores the prompt-grouped design.
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
plot_bootstrap_stability(bootstrap_stability)
