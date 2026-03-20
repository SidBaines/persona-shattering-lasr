#!/usr/bin/env python3
# %%
from __future__ import annotations

import hashlib
import shutil
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.covariance import MinCovDet
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "This script requires scikit-learn. Install the project dependencies first."
    ) from exc

from scripts.factor_analysis.preprocessing import load_embeddings
from scripts.unsupervised_runs import resolve_embedding_artifact_paths, response_run_dir


# %%
@dataclass(frozen=True)
class SourceSpec:
    name: str
    response_run_id: str


SOURCE_SPECS = [
    SourceSpec(name="neutral", response_run_id="stage123-240x50-singleturn-v2"),
    SourceSpec(
        name="very_disagreeable",
        response_run_id="stage123-240x10-singleturn-very-disagreeable-openrouter-v1",
    ),
]

EMBEDDING_ARTIFACT_SLUG = "openai-text-embedding-3-small__assistant-final-turn__norm"
OUTPUT_DIR = Path("scratch/embedding_distance_histograms")
ZIP_PATH = OUTPUT_DIR.with_suffix(".zip")

BINS = 60
DPI = 180
RANDOM_STATE = 0
RESPONSE_ALPHA = 0.45
TOP_OUTLIER_FRACTIONS = (0.005, 0.01, 0.02)
TOP_OUTLIER_PROMPTS = 20
TOP_TEXT_EXAMPLES = 12
PCA_COMPONENTS = 50
ROBUST_COV_COMPONENTS = 20
NEAREST_NEIGHBORS = 10

RUN_MCD_ROBUST_DISTANCE = True
RUN_PAF_ROBUSTNESS = True
PAF_N_FACTORS = 30
PAF_METHOD = "principal"
PAF_ROTATION = "varimax"
PAF_TRIM_FRACTIONS = (0.005, 0.01, 0.02)
PAF_WINSORIZE_FRACTIONS = (0.01,)

SOURCE_COLORS = {
    "neutral": "#4c78a8",
    "very_disagreeable": "#e45756",
}


# %%
def _normalise_text(text: str) -> str:
    return " ".join(text.split())


def _canonical_prompt_text(row: dict) -> str:
    for key in ("seed_user_message", "preceding_user_message"):
        value = _normalise_text(str(row.get(key, "")).strip())
        if value:
            return value
    return ""


def _canonical_prompt_id(row: dict) -> str:
    prompt_text = _canonical_prompt_text(row)
    if prompt_text:
        digest = hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:16]
        return f"prompt_{digest}"
    fallback = _normalise_text(str(row.get("input_group_id", "")).strip())
    if fallback:
        return fallback
    return _normalise_text(str(row.get("sample_id", "")).strip())


def _truncate(text: str, limit: int = 180) -> str:
    cleaned = _normalise_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _rank_to_unit_interval(values: np.ndarray, *, reverse: bool = False) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64)
    order = np.argsort(values)
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    if reverse:
        ranks = float(values.shape[0] - 1) - ranks
    if values.shape[0] == 1:
        return np.ones(1, dtype=np.float64)
    return ranks / float(values.shape[0] - 1)


def _save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def _shared_bins(values: np.ndarray, bins: int) -> np.ndarray:
    return np.histogram_bin_edges(values, bins=bins)


def _load_source_embeddings(source_spec: SourceSpec) -> tuple[np.ndarray, list[dict]]:
    artifact_paths = resolve_embedding_artifact_paths(
        response_run_dir(source_spec.response_run_id),
        EMBEDDING_ARTIFACT_SLUG,
    )
    embeddings, metadata = load_embeddings(
        artifact_paths["embeddings"],
        artifact_paths["metadata"],
    )

    enriched_metadata: list[dict] = []
    for row in metadata:
        row_copy = dict(row)
        row_copy["dataset_source"] = source_spec.name
        row_copy["source_run_id"] = source_spec.response_run_id
        row_copy["shared_prompt_text"] = _canonical_prompt_text(row_copy)
        row_copy["shared_prompt_id"] = _canonical_prompt_id(row_copy)
        enriched_metadata.append(row_copy)
    return embeddings, enriched_metadata


def load_all_sources(source_specs: list[SourceSpec]) -> tuple[np.ndarray, list[dict]]:
    matrices: list[np.ndarray] = []
    combined_metadata: list[dict] = []
    for source_spec in source_specs:
        embeddings, metadata = _load_source_embeddings(source_spec)
        matrices.append(embeddings)
        combined_metadata.extend(metadata)
        print(
            f"Loaded {embeddings.shape[0]} rows for {source_spec.name} "
            f"from {source_spec.response_run_id}"
        )
    combined_embeddings = np.concatenate(matrices, axis=0)
    print(f"Combined embeddings shape: {combined_embeddings.shape}")
    return combined_embeddings, combined_metadata


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def zip_output_dir(output_dir: Path, zip_path: Path) -> Path:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(output_dir.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(output_dir))
    return zip_path


# %%
def compute_diagnostics(
    embeddings: np.ndarray,
    metadata: list[dict],
) -> dict[str, object]:
    n_samples, n_dims = embeddings.shape
    global_mean = embeddings.mean(axis=0)
    centered = embeddings - global_mean[None, :]
    row_norms = np.linalg.norm(embeddings, axis=1)

    response_distance_from_global_mean = np.linalg.norm(centered, axis=1)
    global_mean_norm = float(np.linalg.norm(global_mean))
    if global_mean_norm > 1e-12:
        cosine_to_global_mean = (embeddings @ global_mean) / (row_norms * global_mean_norm + 1e-12)
    else:
        cosine_to_global_mean = np.full(n_samples, np.nan, dtype=np.float64)

    source_names = np.array([str(row.get("dataset_source", "")) for row in metadata], dtype=object)
    prompt_ids = np.array([str(row.get("shared_prompt_id", "")) for row in metadata], dtype=object)
    prompt_texts = np.array([str(row.get("shared_prompt_text", "")) for row in metadata], dtype=object)

    unique_prompt_ids, group_inverse = np.unique(prompt_ids, return_inverse=True)
    prompt_means = np.zeros((len(unique_prompt_ids), n_dims), dtype=np.float64)
    prompt_group_sizes = np.zeros(len(unique_prompt_ids), dtype=np.int32)
    prompt_group_average_distance_from_global_mean = np.zeros(len(unique_prompt_ids), dtype=np.float64)
    prompt_centroid_distance_from_global_mean = np.zeros(len(unique_prompt_ids), dtype=np.float64)
    prompt_text_by_group = np.empty(len(unique_prompt_ids), dtype=object)

    for group_idx in range(len(unique_prompt_ids)):
        mask = group_inverse == group_idx
        prompt_means[group_idx] = embeddings[mask].mean(axis=0)
        prompt_group_sizes[group_idx] = int(mask.sum())
        prompt_group_average_distance_from_global_mean[group_idx] = (
            response_distance_from_global_mean[mask].mean()
        )
        prompt_centroid_distance_from_global_mean[group_idx] = float(
            np.linalg.norm(prompt_means[group_idx] - global_mean)
        )
        prompt_text_by_group[group_idx] = prompt_texts[np.flatnonzero(mask)[0]]

    response_distance_from_prompt_mean = np.linalg.norm(
        embeddings - prompt_means[group_inverse],
        axis=1,
    )
    prompt_within_group_mean_distance = np.zeros(len(unique_prompt_ids), dtype=np.float64)
    prompt_within_group_max_distance = np.zeros(len(unique_prompt_ids), dtype=np.float64)
    for group_idx in range(len(unique_prompt_ids)):
        mask = group_inverse == group_idx
        prompt_within_group_mean_distance[group_idx] = response_distance_from_prompt_mean[mask].mean()
        prompt_within_group_max_distance[group_idx] = response_distance_from_prompt_mean[mask].max()

    pca_components = min(PCA_COMPONENTS, n_samples - 1, n_dims)
    pca = PCA(n_components=pca_components, random_state=RANDOM_STATE)
    pca_scores = pca.fit_transform(embeddings)
    pca_score_norm = np.linalg.norm(pca_scores, axis=1)
    pca_whitened_distance = np.linalg.norm(
        pca_scores / np.sqrt(np.maximum(pca.explained_variance_, 1e-12))[None, :],
        axis=1,
    )

    robust_mahalanobis_distance = np.full(n_samples, np.nan, dtype=np.float64)
    if RUN_MCD_ROBUST_DISTANCE:
        robust_dims = min(ROBUST_COV_COMPONENTS, pca_components)
        if robust_dims >= 2:
            try:
                robust_model = MinCovDet(random_state=RANDOM_STATE).fit(pca_scores[:, :robust_dims])
                robust_mahalanobis_distance = np.sqrt(
                    robust_model.mahalanobis(pca_scores[:, :robust_dims])
                )
            except Exception as exc:  # pragma: no cover - runtime / numerical instability
                print(f"Robust Mahalanobis distance failed: {exc}")

    neighbor_count = min(NEAREST_NEIGHBORS + 1, n_samples)
    neighbors = NearestNeighbors(n_neighbors=neighbor_count, metric="euclidean")
    neighbors.fit(embeddings)
    neighbor_distances, neighbor_indices = neighbors.kneighbors(embeddings)
    neighbor_distances = neighbor_distances[:, 1:]
    neighbor_indices = neighbor_indices[:, 1:]
    nearest_neighbor_index = neighbor_indices[:, 0]
    nearest_neighbor_distance = neighbor_distances[:, 0]
    nearest_neighbor_cosine = np.sum(
        embeddings * embeddings[nearest_neighbor_index],
        axis=1,
    ) / (row_norms * np.linalg.norm(embeddings[nearest_neighbor_index], axis=1) + 1e-12)
    neighbor_mean = embeddings[neighbor_indices].mean(axis=1)
    nearest_neighbor_mean_distance = np.linalg.norm(embeddings - neighbor_mean, axis=1)

    metric_ranks = [
        _rank_to_unit_interval(response_distance_from_global_mean, reverse=False),
        _rank_to_unit_interval(pca_whitened_distance, reverse=False),
        _rank_to_unit_interval(nearest_neighbor_mean_distance, reverse=False),
    ]
    finite_cosine = np.isfinite(cosine_to_global_mean)
    if np.any(finite_cosine):
        cosine_rank = np.zeros(n_samples, dtype=np.float64)
        cosine_rank[finite_cosine] = _rank_to_unit_interval(
            cosine_to_global_mean[finite_cosine],
            reverse=True,
        )
        metric_ranks.append(cosine_rank)
    finite_robust = np.isfinite(robust_mahalanobis_distance)
    if np.any(finite_robust):
        robust_rank = np.zeros(n_samples, dtype=np.float64)
        robust_rank[finite_robust] = _rank_to_unit_interval(
            robust_mahalanobis_distance[finite_robust],
            reverse=False,
        )
        metric_ranks.append(robust_rank)
    combined_outlier_score = np.mean(metric_ranks, axis=0)
    outlier_order = np.argsort(combined_outlier_score)[::-1]
    outlier_rank = np.empty_like(outlier_order)
    outlier_rank[outlier_order] = np.arange(n_samples)

    prompt_records: list[dict] = []
    for group_idx, prompt_id in enumerate(unique_prompt_ids):
        mask = group_inverse == group_idx
        prompt_records.append(
            {
                "prompt_id": str(prompt_id),
                "prompt_text": str(prompt_text_by_group[group_idx]),
                "size": int(prompt_group_sizes[group_idx]),
                "average_response_distance_from_global_mean": float(
                    prompt_group_average_distance_from_global_mean[group_idx]
                ),
                "centroid_distance_from_global_mean": float(
                    prompt_centroid_distance_from_global_mean[group_idx]
                ),
                "within_group_mean_distance": float(prompt_within_group_mean_distance[group_idx]),
                "within_group_max_distance": float(prompt_within_group_max_distance[group_idx]),
                "neutral_count": int(np.sum(source_names[mask] == "neutral")),
                "very_disagreeable_count": int(np.sum(source_names[mask] == "very_disagreeable")),
            }
        )

    response_records: list[dict] = []
    for idx, row in enumerate(metadata):
        response_records.append(
            {
                "index": idx,
                "dataset_source": str(source_names[idx]),
                "shared_prompt_id": str(prompt_ids[idx]),
                "shared_prompt_text": str(prompt_texts[idx]),
                "response_distance_from_global_mean": float(response_distance_from_global_mean[idx]),
                "response_distance_from_prompt_mean": float(response_distance_from_prompt_mean[idx]),
                "cosine_to_global_mean": (
                    float(cosine_to_global_mean[idx]) if np.isfinite(cosine_to_global_mean[idx]) else None
                ),
                "pca_score_norm": float(pca_score_norm[idx]),
                "pca_whitened_distance": float(pca_whitened_distance[idx]),
                "robust_mahalanobis_distance": (
                    float(robust_mahalanobis_distance[idx])
                    if np.isfinite(robust_mahalanobis_distance[idx])
                    else None
                ),
                "nearest_neighbor_mean_distance": float(nearest_neighbor_mean_distance[idx]),
                "nearest_neighbor_distance": float(nearest_neighbor_distance[idx]),
                "nearest_neighbor_cosine": float(nearest_neighbor_cosine[idx]),
                "combined_outlier_score": float(combined_outlier_score[idx]),
                "outlier_rank": int(outlier_rank[idx]) + 1,
                "prompt": _truncate(str(row.get("seed_user_message", "")), 120),
                "response": _truncate(str(row.get("assistant_text", "")), 220),
            }
        )

    return {
        "global_mean": global_mean,
        "source_names": source_names,
        "prompt_ids": prompt_ids,
        "prompt_texts": prompt_texts,
        "group_inverse": group_inverse,
        "unique_prompt_ids": unique_prompt_ids,
        "prompt_means": prompt_means,
        "prompt_records": prompt_records,
        "response_records": response_records,
        "response_distance_from_global_mean": response_distance_from_global_mean,
        "prompt_group_average_distance_from_global_mean": (
            prompt_group_average_distance_from_global_mean
        ),
        "prompt_centroid_distance_from_global_mean": prompt_centroid_distance_from_global_mean,
        "response_distance_from_prompt_mean": response_distance_from_prompt_mean,
        "prompt_within_group_mean_distance": prompt_within_group_mean_distance,
        "prompt_within_group_max_distance": prompt_within_group_max_distance,
        "pca": pca,
        "pca_scores": pca_scores,
        "pca_score_norm": pca_score_norm,
        "pca_whitened_distance": pca_whitened_distance,
        "cosine_to_global_mean": cosine_to_global_mean,
        "robust_mahalanobis_distance": robust_mahalanobis_distance,
        "nearest_neighbor_mean_distance": nearest_neighbor_mean_distance,
        "nearest_neighbor_index": nearest_neighbor_index,
        "nearest_neighbor_distance": nearest_neighbor_distance,
        "nearest_neighbor_cosine": nearest_neighbor_cosine,
        "combined_outlier_score": combined_outlier_score,
        "outlier_order": outlier_order,
        "outlier_rank": outlier_rank,
        "embeddings": embeddings,
        "metadata": metadata,
    }


def winsorize_embeddings_by_global_distance(
    embeddings: np.ndarray,
    global_mean: np.ndarray,
    distances: np.ndarray,
    fraction: float,
) -> np.ndarray:
    capped = embeddings.copy()
    threshold = float(np.quantile(distances, 1.0 - fraction))
    centered = capped - global_mean[None, :]
    norms = np.linalg.norm(centered, axis=1)
    mask = norms > threshold
    scales = np.ones_like(norms)
    scales[mask] = threshold / np.maximum(norms[mask], 1e-12)
    capped = global_mean[None, :] + centered * scales[:, None]
    return capped


# %%
def plot_histogram_all_and_by_source(
    values: np.ndarray,
    source_names: np.ndarray,
    output_dir: Path,
    *,
    filename: str,
    overall_title: str,
    by_source_title: str,
    xlabel: str,
    overall_color: str,
) -> Path:
    bins = _shared_bins(values, BINS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(values, bins=bins, color=overall_color, alpha=0.9)
    axes[0].set_title(overall_title)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Count")

    for source_name in sorted(set(source_names.tolist())):
        mask = source_names == source_name
        axes[1].hist(
            values[mask],
            bins=bins,
            alpha=RESPONSE_ALPHA,
            label=f"{source_name} (n={int(mask.sum())})",
            color=SOURCE_COLORS.get(source_name),
        )
    axes[1].set_title(by_source_title)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Count")
    axes[1].legend(frameon=False)

    return _save_figure(fig, output_dir, filename)


def plot_single_histogram(
    values: np.ndarray,
    output_dir: Path,
    *,
    filename: str,
    title: str,
    xlabel: str,
    color: str,
) -> Path:
    bins = _shared_bins(values, BINS)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.hist(values, bins=bins, color=color, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    return _save_figure(fig, output_dir, filename)


def plot_prompt_centroid_vs_within_spread(
    diagnostics: dict[str, object],
    output_dir: Path,
) -> Path:
    prompt_records = diagnostics["prompt_records"]
    x = np.asarray([row["centroid_distance_from_global_mean"] for row in prompt_records], dtype=np.float64)
    y = np.asarray([row["within_group_mean_distance"] for row in prompt_records], dtype=np.float64)
    sizes = np.asarray([row["size"] for row in prompt_records], dtype=np.float64)

    top_mask = np.zeros(len(prompt_records), dtype=bool)
    outlier_scores = np.asarray(diagnostics["combined_outlier_score"], dtype=np.float64)
    group_inverse = np.asarray(diagnostics["group_inverse"], dtype=np.int64)
    top_count = max(1, int(len(outlier_scores) * 0.01))
    top_indices = np.argsort(outlier_scores)[-top_count:]
    top_mask[np.unique(group_inverse[top_indices])] = True

    fig, ax = plt.subplots(figsize=(7.5, 6))
    scatter = ax.scatter(
        x,
        y,
        s=12 + 2.5 * sizes,
        c=top_mask.astype(np.float64),
        cmap="coolwarm",
        alpha=0.75,
        edgecolor="black",
        linewidth=0.25,
    )
    for idx in np.argsort(y + x)[-8:]:
        ax.annotate(
            _truncate(str(prompt_records[idx]["prompt_text"]), 40),
            (x[idx], y[idx]),
            fontsize=7,
            alpha=0.9,
        )
    ax.set_title("Prompt centroid distance vs within-group spread")
    ax.set_xlabel("Prompt centroid distance from global mean")
    ax.set_ylabel("Mean response distance from prompt-group mean")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Prompt contains top-1% outlier response")
    return _save_figure(fig, output_dir, "prompt_centroid_vs_within_spread.png")


def plot_top_outlier_source_composition(
    diagnostics: dict[str, object],
    output_dir: Path,
) -> Path:
    scores = np.asarray(diagnostics["combined_outlier_score"], dtype=np.float64)
    source_names = np.asarray(diagnostics["source_names"], dtype=object)
    unique_sources = sorted(set(source_names.tolist()))

    labels = []
    count_matrix = np.zeros((len(TOP_OUTLIER_FRACTIONS), len(unique_sources)), dtype=np.int32)
    share_matrix = np.zeros_like(count_matrix, dtype=np.float64)

    for frac_idx, fraction in enumerate(TOP_OUTLIER_FRACTIONS):
        top_count = max(1, int(len(scores) * fraction))
        top_indices = np.argsort(scores)[-top_count:]
        labels.append(f"top {fraction * 100:.1f}%")
        for source_idx, source_name in enumerate(unique_sources):
            count = int(np.sum(source_names[top_indices] == source_name))
            count_matrix[frac_idx, source_idx] = count
            share_matrix[frac_idx, source_idx] = count / top_count

    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for source_idx, source_name in enumerate(unique_sources):
        offset = (source_idx - (len(unique_sources) - 1) / 2) * width
        axes[0].bar(
            x + offset,
            count_matrix[:, source_idx],
            width=width,
            label=source_name,
            color=SOURCE_COLORS.get(source_name),
        )
        axes[1].bar(
            x + offset,
            share_matrix[:, source_idx],
            width=width,
            label=source_name,
            color=SOURCE_COLORS.get(source_name),
        )
    axes[0].set_title("Source counts among top outliers")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Count")
    axes[1].set_title("Source shares among top outliers")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Share")
    axes[1].set_ylim(0, 1)
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    return _save_figure(fig, output_dir, "top_outlier_source_composition.png")


def plot_top_outlier_prompt_concentration(
    diagnostics: dict[str, object],
    output_dir: Path,
) -> Path:
    scores = np.asarray(diagnostics["combined_outlier_score"], dtype=np.float64)
    group_inverse = np.asarray(diagnostics["group_inverse"], dtype=np.int64)
    prompt_records = diagnostics["prompt_records"]
    top_count = max(1, int(len(scores) * 0.01))
    top_indices = np.argsort(scores)[-top_count:]

    prompt_counts = np.bincount(group_inverse[top_indices], minlength=len(prompt_records))
    top_prompt_indices = np.argsort(prompt_counts)[::-1][:TOP_OUTLIER_PROMPTS]
    top_prompt_indices = [idx for idx in top_prompt_indices if prompt_counts[idx] > 0]

    labels = [_truncate(str(prompt_records[idx]["prompt_text"]), 55) for idx in top_prompt_indices]
    values = [int(prompt_counts[idx]) for idx in top_prompt_indices]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(len(values)), values, color="#8c6d31")
    ax.set_xticks(np.arange(len(values)), labels, rotation=65, ha="right")
    ax.set_title("Prompt concentration among top 1% outlier responses")
    ax.set_ylabel("Top-outlier response count")
    return _save_figure(fig, output_dir, "top_outlier_prompt_concentration.png")


def plot_pca_scatter(
    diagnostics: dict[str, object],
    output_dir: Path,
) -> Path:
    pca_scores = np.asarray(diagnostics["pca_scores"], dtype=np.float64)
    source_names = np.asarray(diagnostics["source_names"], dtype=object)
    scores = np.asarray(diagnostics["combined_outlier_score"], dtype=np.float64)
    top_count = max(1, int(len(scores) * 0.01))
    top_indices = set(np.argsort(scores)[-top_count:].tolist())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for source_name in sorted(set(source_names.tolist())):
        mask = source_names == source_name
        axes[0].scatter(
            pca_scores[mask, 0],
            pca_scores[mask, 1],
            s=8,
            alpha=0.35,
            color=SOURCE_COLORS.get(source_name),
            label=source_name,
        )
    axes[0].set_title("PC1 vs PC2 by source")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(frameon=False)

    regular_idx = np.array([idx for idx in range(len(scores)) if idx not in top_indices], dtype=np.int64)
    top_idx = np.array(sorted(top_indices), dtype=np.int64)
    axes[1].scatter(
        pca_scores[regular_idx, 0],
        pca_scores[regular_idx, 1],
        s=8,
        alpha=0.20,
        color="#7f7f7f",
        label="other responses",
    )
    axes[1].scatter(
        pca_scores[top_idx, 0],
        pca_scores[top_idx, 1],
        s=18,
        alpha=0.75,
        color="#d62728",
        label="top 1% outliers",
    )
    axes[1].set_title("PC1 vs PC2 with top 1% outliers highlighted")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend(frameon=False)

    return _save_figure(fig, output_dir, "pca_scatter_pc1_pc2.png")


def plot_text_examples_panel(
    lines: list[str],
    output_dir: Path,
    *,
    filename: str,
    title: str,
) -> Path:
    wrapped_lines: list[str] = []
    for line in lines:
        wrapped_lines.extend(textwrap.wrap(line, width=115) or [""])
    fig_height = max(8, 0.23 * len(wrapped_lines))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=13, pad=14)
    ax.text(
        0.0,
        1.0,
        "\n".join(wrapped_lines),
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        transform=ax.transAxes,
    )
    return _save_figure(fig, output_dir, filename)


def plot_top_outlier_examples(
    diagnostics: dict[str, object],
    output_dir: Path,
) -> Path:
    lines = []
    for idx in diagnostics["outlier_order"][:TOP_TEXT_EXAMPLES]:
        row = diagnostics["response_records"][int(idx)]
        lines.append(
            f"[rank {row['outlier_rank']:>3}] {row['dataset_source']} | "
            f"score={row['combined_outlier_score']:.3f} | "
            f"l2={row['response_distance_from_global_mean']:.3f} | "
            f"pca={row['pca_whitened_distance']:.3f}"
        )
        lines.append(f"prompt: {row['prompt']}")
        lines.append(f"response: {row['response']}")
        lines.append("")
    return plot_text_examples_panel(
        lines,
        output_dir,
        filename="top_outlier_examples.png",
        title="Top outlier examples",
    )


def plot_top_near_duplicate_examples(
    diagnostics: dict[str, object],
    output_dir: Path,
) -> Path:
    nearest_idx = np.asarray(diagnostics["nearest_neighbor_index"], dtype=np.int64)
    nearest_cos = np.asarray(diagnostics["nearest_neighbor_cosine"], dtype=np.float64)
    seen_pairs: set[tuple[int, int]] = set()
    lines = []
    for idx in np.argsort(nearest_cos)[::-1]:
        pair = tuple(sorted((int(idx), int(nearest_idx[idx]))))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        row_a = diagnostics["response_records"][pair[0]]
        row_b = diagnostics["response_records"][pair[1]]
        lines.append(
            f"pair cosine={nearest_cos[idx]:.4f} | "
            f"{row_a['dataset_source']} vs {row_b['dataset_source']}"
        )
        lines.append(f"A prompt: {row_a['prompt']}")
        lines.append(f"A response: {row_a['response']}")
        lines.append(f"B prompt: {row_b['prompt']}")
        lines.append(f"B response: {row_b['response']}")
        lines.append("")
        if len(seen_pairs) >= TOP_TEXT_EXAMPLES:
            break
    return plot_text_examples_panel(
        lines,
        output_dir,
        filename="top_near_duplicate_examples.png",
        title="Most similar nearest-neighbour pairs",
    )


def make_all_plots(
    diagnostics: dict[str, object],
    output_dir: Path,
) -> None:
    source_names = np.asarray(diagnostics["source_names"], dtype=object)

    plot_histogram_all_and_by_source(
        np.asarray(diagnostics["response_distance_from_global_mean"], dtype=np.float64),
        source_names,
        output_dir,
        filename="response_distance_from_global_mean.png",
        overall_title="All response distances from global mean",
        by_source_title="Response distances from global mean by source",
        xlabel="Euclidean distance",
        overall_color="#6c6f7d",
    )
    plot_single_histogram(
        np.asarray(
            diagnostics["prompt_group_average_distance_from_global_mean"],
            dtype=np.float64,
        ),
        output_dir,
        filename="prompt_group_average_distance_from_global_mean.png",
        title="Average response distance per prompt group from global mean",
        xlabel="Average Euclidean distance",
        color="#59a14f",
    )
    plot_histogram_all_and_by_source(
        np.asarray(diagnostics["response_distance_from_prompt_mean"], dtype=np.float64),
        source_names,
        output_dir,
        filename="response_distance_from_prompt_mean.png",
        overall_title="All response distances from prompt-group mean",
        by_source_title="Response distances from prompt-group mean by source",
        xlabel="Euclidean distance",
        overall_color="#f28e2b",
    )
    plot_histogram_all_and_by_source(
        np.asarray(diagnostics["cosine_to_global_mean"], dtype=np.float64),
        source_names,
        output_dir,
        filename="cosine_similarity_to_global_mean.png",
        overall_title="Cosine similarity to global mean direction",
        by_source_title="Cosine similarity to global mean by source",
        xlabel="Cosine similarity",
        overall_color="#9467bd",
    )
    plot_histogram_all_and_by_source(
        np.asarray(diagnostics["pca_whitened_distance"], dtype=np.float64),
        source_names,
        output_dir,
        filename="pca_whitened_distance.png",
        overall_title="PCA-whitened distance (Mahalanobis-like)",
        by_source_title="PCA-whitened distance by source",
        xlabel="Distance",
        overall_color="#8c564b",
    )
    robust_values = np.asarray(diagnostics["robust_mahalanobis_distance"], dtype=np.float64)
    if np.any(np.isfinite(robust_values)):
        plot_histogram_all_and_by_source(
            robust_values[np.isfinite(robust_values)],
            source_names[np.isfinite(robust_values)],
            output_dir,
            filename="robust_mahalanobis_distance.png",
            overall_title="Robust Mahalanobis distance in PCA space",
            by_source_title="Robust Mahalanobis distance by source",
            xlabel="Distance",
            overall_color="#bc5090",
        )
    plot_histogram_all_and_by_source(
        np.asarray(diagnostics["nearest_neighbor_mean_distance"], dtype=np.float64),
        source_names,
        output_dir,
        filename="nearest_neighbor_mean_distance.png",
        overall_title="Distance to mean of nearest neighbours",
        by_source_title="Nearest-neighbour mean distance by source",
        xlabel="Euclidean distance",
        overall_color="#2ca02c",
    )
    plot_histogram_all_and_by_source(
        np.asarray(diagnostics["nearest_neighbor_cosine"], dtype=np.float64),
        source_names,
        output_dir,
        filename="nearest_neighbor_cosine_similarity.png",
        overall_title="Nearest-neighbour cosine similarity",
        by_source_title="Nearest-neighbour cosine similarity by source",
        xlabel="Cosine similarity",
        overall_color="#1f77b4",
    )
    plot_single_histogram(
        np.asarray(diagnostics["prompt_centroid_distance_from_global_mean"], dtype=np.float64),
        output_dir,
        filename="prompt_centroid_distance_from_global_mean.png",
        title="Prompt centroid distance from global mean",
        xlabel="Euclidean distance",
        color="#17becf",
    )
    plot_single_histogram(
        np.asarray(diagnostics["prompt_within_group_mean_distance"], dtype=np.float64),
        output_dir,
        filename="prompt_within_group_mean_distance.png",
        title="Prompt-group within-group dispersion",
        xlabel="Mean response distance from prompt-group mean",
        color="#ff7f0e",
    )
    plot_single_histogram(
        np.asarray(diagnostics["pca_score_norm"], dtype=np.float64),
        output_dir,
        filename="pca_score_norm.png",
        title="PCA score norm / leverage",
        xlabel="L2 norm in PCA score space",
        color="#7f7f7f",
    )
    plot_prompt_centroid_vs_within_spread(diagnostics, output_dir)
    plot_top_outlier_source_composition(diagnostics, output_dir)
    plot_top_outlier_prompt_concentration(diagnostics, output_dir)
    plot_pca_scatter(diagnostics, output_dir)
    plot_top_outlier_examples(diagnostics, output_dir)
    plot_top_near_duplicate_examples(diagnostics, output_dir)


# %%
def _column_correlation_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_centered = a - a.mean(axis=0, keepdims=True)
    b_centered = b - b.mean(axis=0, keepdims=True)
    a_scaled = a_centered / np.maximum(a_centered.std(axis=0, ddof=0, keepdims=True), 1e-12)
    b_scaled = b_centered / np.maximum(b_centered.std(axis=0, ddof=0, keepdims=True), 1e-12)
    return (a_scaled.T @ b_scaled) / a.shape[0]


def build_paf_scenarios(diagnostics: dict[str, object]) -> dict[str, np.ndarray]:
    embeddings = np.asarray(diagnostics["embeddings"], dtype=np.float64)
    scores = np.asarray(diagnostics["combined_outlier_score"], dtype=np.float64)
    global_mean = np.asarray(diagnostics["global_mean"], dtype=np.float64)
    l2_distances = np.asarray(diagnostics["response_distance_from_global_mean"], dtype=np.float64)

    scenarios = {"baseline": embeddings}
    for fraction in PAF_TRIM_FRACTIONS:
        trim_count = max(1, int(len(scores) * fraction))
        keep_indices = np.argsort(scores)[:-trim_count]
        scenarios[f"trim_top_{fraction * 100:.1f}pct"] = embeddings[keep_indices]
    for fraction in PAF_WINSORIZE_FRACTIONS:
        scenarios[f"winsorize_l2_{fraction * 100:.1f}pct"] = winsorize_embeddings_by_global_distance(
            embeddings,
            global_mean,
            l2_distances,
            fraction,
        )
    return scenarios


def run_paf_scenarios(scenarios: dict[str, np.ndarray]) -> dict[str, dict]:
    try:
        from scripts.factor_analysis.factor_analysis import run_factor_analysis
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PAF robustness checks require factor_analyzer and the project factor-analysis dependencies."
        ) from exc

    results: dict[str, dict] = {}
    for name, data in scenarios.items():
        print(f"Running PAF scenario: {name} | shape={data.shape}")
        results[name] = run_factor_analysis(
            data,
            n_factors=PAF_N_FACTORS,
            method=PAF_METHOD,
            rotation=PAF_ROTATION,
        )
    return results


def plot_paf_variance_comparison(
    paf_results: dict[str, dict],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for name, result in paf_results.items():
        x = np.arange(1, len(result["proportion_variance"]) + 1)
        axes[0].plot(x, result["proportion_variance"], marker="o", label=name)
        axes[1].plot(x, result["cumulative_variance"], marker="o", label=name)
    axes[0].set_title("PAF proportion variance by factor")
    axes[0].set_xlabel("Factor")
    axes[0].set_ylabel("Proportion variance")
    axes[1].set_title("PAF cumulative variance")
    axes[1].set_xlabel("Factor")
    axes[1].set_ylabel("Cumulative variance")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    _save_figure(fig, output_dir, "paf_variance_comparison.png")


def plot_paf_communality_distribution(
    paf_results: dict[str, dict],
    output_dir: Path,
) -> None:
    labels = list(paf_results)
    values = [np.asarray(paf_results[name]["communalities"], dtype=np.float64) for name in labels]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.boxplot(values, labels=labels, showfliers=False)
    ax.set_title("PAF communality distributions by scenario")
    ax.set_ylabel("Communality")
    ax.tick_params(axis="x", rotation=25)
    _save_figure(fig, output_dir, "paf_communality_distribution.png")


def plot_paf_loading_match_diagnostics(
    paf_results: dict[str, dict],
    output_dir: Path,
) -> None:
    baseline_loadings = np.asarray(paf_results["baseline"]["loadings"], dtype=np.float64)
    comparison_names = [name for name in paf_results if name != "baseline"]
    if not comparison_names:
        return

    fig, ax = plt.subplots(figsize=(12, 5.5))
    factor_ids = np.arange(1, baseline_loadings.shape[1] + 1)
    for name in comparison_names:
        loadings = np.asarray(paf_results[name]["loadings"], dtype=np.float64)
        corr_matrix = np.abs(_column_correlation_matrix(baseline_loadings, loadings))
        best_corr = corr_matrix.max(axis=1)
        ax.plot(factor_ids, best_corr, marker="o", label=name)
    ax.set_title("Best loading-match correlation vs baseline")
    ax.set_xlabel("Baseline factor")
    ax.set_ylabel("Max |corr(loadings)|")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    _save_figure(fig, output_dir, "paf_best_loading_match.png")

    n_cols = 2
    n_rows = int(np.ceil(len(comparison_names) / n_cols))
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()
    for ax_idx, name in enumerate(comparison_names):
        loadings = np.asarray(paf_results[name]["loadings"], dtype=np.float64)
        corr_matrix = np.abs(_column_correlation_matrix(baseline_loadings, loadings))
        im = axes_flat[ax_idx].imshow(corr_matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axes_flat[ax_idx].set_title(name)
        axes_flat[ax_idx].set_xlabel("Comparison factor")
        axes_flat[ax_idx].set_ylabel("Baseline factor")
        fig2.colorbar(im, ax=axes_flat[ax_idx], fraction=0.046, pad=0.04)
    for ax in axes_flat[len(comparison_names):]:
        ax.axis("off")
    _save_figure(fig2, output_dir, "paf_loading_match_heatmaps.png")


def run_and_plot_paf_robustness(
    diagnostics: dict[str, object],
    output_dir: Path,
) -> None:
    if not RUN_PAF_ROBUSTNESS:
        return
    scenarios = build_paf_scenarios(diagnostics)
    paf_results = run_paf_scenarios(scenarios)
    plot_paf_variance_comparison(paf_results, output_dir)
    plot_paf_communality_distribution(paf_results, output_dir)
    plot_paf_loading_match_diagnostics(paf_results, output_dir)


# %%
prepare_output_dir(OUTPUT_DIR)

embeddings, metadata = load_all_sources(SOURCE_SPECS)
diagnostics = compute_diagnostics(embeddings, metadata)

make_all_plots(diagnostics, OUTPUT_DIR)
run_and_plot_paf_robustness(diagnostics, OUTPUT_DIR)

zip_path = zip_output_dir(OUTPUT_DIR, ZIP_PATH)
print(f"Zipped plots to {zip_path}")
