"""Factor interpretation methods.

Six approaches for understanding what each factor represents:
1. Gradient descent through embedding model (optimize_factor_embedding)
2. Analytical back-projection + corpus nearest neighbor
3. Score-based corpus ranking by factor purity
4. Gradient descent + corpus lookup (compose methods 1 + 2 in experiment script)
5. Contrastive centroid retrieval (contrastive_factor_retrieval)
6. Max-spread: find questions whose responses span the widest target-score range (rank_prompts_by_max_spread)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _compute_other_abs_mean(scores: np.ndarray, factor_idx: int) -> np.ndarray:
    """Compute per-sample mean absolute activation on all non-target factors."""
    n_factors = scores.shape[1]
    other_mask = np.ones(n_factors, dtype=bool)
    other_mask[factor_idx] = False
    return np.abs(scores[:, other_mask]).mean(axis=1)


def compute_factor_purity(
    scores: np.ndarray,
    factor_idx: int,
    penalty_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sign-aware target scores and symmetric purity magnitudes.

    Reported purity is always a non-negative magnitude, computed as:

        abs(score[factor_idx]) - penalty_weight * mean(|score[j]| for j != factor_idx)

    and floored at 0.

    Args:
        scores: Factor scores [n_samples, n_factors].
        factor_idx: Target factor whose purity to compute.
        penalty_weight: Weight for penalizing off-target factor magnitude.

    Returns:
        Tuple of (target_scores, other_abs_mean, purity_scores).
    """
    target_scores = scores[:, factor_idx]
    other_abs_mean = _compute_other_abs_mean(scores, factor_idx)
    purity_scores = np.maximum(
        0.0,
        np.abs(target_scores) - penalty_weight * other_abs_mean,
    )
    return target_scores, other_abs_mean, purity_scores


def compute_factor_lagrangian(
    scores: np.ndarray,
    factor_idx: int,
    penalty_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute sign-aware Lagrangian objectives using L1 off-target penalties.

    The high-polarity objective is:

        score[factor_idx] - penalty_weight * mean(|score[j]| for j != factor_idx)

    The low-polarity objective is the same with the target score negated:

        -score[factor_idx] - penalty_weight * mean(|score[j]| for j != factor_idx)

    Args:
        scores: Factor scores [n_samples, n_factors].
        factor_idx: Target factor whose constrained objective to compute.
        penalty_weight: Weight for penalizing off-target factor magnitude.

    Returns:
        Tuple of (target_scores, other_abs_mean, high_objective, low_objective).
    """
    n_factors = scores.shape[1]
    other_mask = np.ones(n_factors, dtype=bool)
    other_mask[factor_idx] = False

    target_scores = scores[:, factor_idx]
    other_abs_mean = np.abs(scores[:, other_mask]).mean(axis=1)
    high_objective = target_scores - penalty_weight * other_abs_mean
    low_objective = -target_scores - penalty_weight * other_abs_mean
    return target_scores, other_abs_mean, high_objective, low_objective

def back_project_factor(
    factor_idx: int,
    loadings: np.ndarray,
    pca_model: PCA,
    scaler: StandardScaler | None = None,
) -> np.ndarray:
    """Back-project a factor direction from reduced space to original embedding space.

    The factor direction in PCA space is the factor_idx-th column of the loadings
    matrix. We project it back through PCA and (optionally) the scaler to get a
    direction in the original embedding space.

    Args:
        factor_idx: Which factor column to back-project.
        loadings: Factor loading matrix [n_pca_dims, n_factors].
        pca_model: Fitted PCA used for pre-reduction.
        scaler: Fitted StandardScaler, or None if no standardization was used.

    Returns:
        Unit direction vector in original embedding space [d].
    """
    # Factor direction in the input space (PCA-reduced, or original if no PCA).
    direction = loadings[:, factor_idx]

    # Back-project through PCA if one was used.
    if pca_model is not None:
        direction = (
            pca_model.inverse_transform(direction.reshape(1, -1))
            - pca_model.inverse_transform(np.zeros_like(direction).reshape(1, -1))
        ).squeeze(0)
    # else: loadings are already in the (standardized) original embedding space.

    # Back-project through scaler: standardized space -> original space.
    if scaler is not None:
        # For a direction vector, we only need to undo the scaling (not the mean shift).
        direction_orig = direction * scaler.scale_
    else:
        direction_orig = direction

    # Normalize to unit vector.
    norm = np.linalg.norm(direction_orig)
    if norm > 1e-12:
        direction_orig = direction_orig / norm

    return direction_orig


def factor_extremes(
    scores: np.ndarray,
    metadata: list[dict],
    top_n: int = 20,
    text_field: str = "assistant_text",
    excerpt_length: int = 400,
) -> list[dict]:
    """Find top/bottom N samples per factor by raw score.

    Args:
        scores: Factor scores [n_samples, n_factors].
        metadata: Metadata rows aligned with scores.
        top_n: Number of extreme samples per end.
        text_field: Metadata field containing response text.
        excerpt_length: Max characters to include in text excerpts.

    Returns:
        List of dicts (one per factor), each with 'top' and 'bottom' lists
        of sample info dicts.
    """
    results = []
    for fi in range(scores.shape[1]):
        values = scores[:, fi]
        order = np.argsort(values)
        n = min(top_n, len(metadata))

        def _entry(idx: int) -> dict:
            row = metadata[idx]
            text = str(row.get(text_field, ""))
            return {
                "index": int(idx),
                "score": float(values[idx]),
                "sample_id": row.get("sample_id"),
                "input_group_id": row.get("input_group_id"),
                "seed_user_message": str(row.get("seed_user_message", ""))[:200],
                "text_excerpt": text[:excerpt_length],
            }

        top = [_entry(int(idx)) for idx in order[-n:][::-1]]
        bottom = [_entry(int(idx)) for idx in order[:n]]
        results.append({"factor_index": fi, "top": top, "bottom": bottom})

    return results


def prompt_effects(
    scores: np.ndarray,
    metadata: list[dict],
    group_field: str = "input_group_id",
) -> np.ndarray:
    """Compute eta-squared per factor: proportion of variance explained by prompt group.

    High eta-squared (>0.3) means the factor is mostly capturing between-prompt
    (content) variation rather than within-prompt (style) variation.

    Args:
        scores: Factor scores [n_samples, n_factors].
        metadata: Metadata rows aligned with scores.
        group_field: Field in metadata to group by.

    Returns:
        Array of eta-squared values [n_factors].
    """
    import warnings

    present = [group_field in row for row in metadata]
    if not any(present):
        warnings.warn(
            f"prompt_effects: group_field={group_field!r} missing from all "
            f"{len(metadata)} metadata rows — returning NaN eta² (the previous "
            "silent fallback to row-index grouped every row on its own, forcing "
            "eta²=1.0).",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.full(scores.shape[1], np.nan, dtype=np.float64)
    if not all(present):
        warnings.warn(
            f"prompt_effects: group_field={group_field!r} missing from "
            f"{sum(not p for p in present)}/{len(metadata)} rows; those rows "
            "are grouped under a shared '__missing__' bucket.",
            RuntimeWarning,
            stacklevel=2,
        )

    group_ids = np.array([
        str(row.get(group_field, "__missing__")) for row in metadata
    ])
    unique_groups, group_inverse = np.unique(group_ids, return_inverse=True)
    n_groups = len(unique_groups)

    eta2 = np.zeros(scores.shape[1], dtype=np.float64)
    for fi in range(scores.shape[1]):
        values = scores[:, fi]
        grand_mean = values.mean()
        ss_total = np.sum((values - grand_mean) ** 2)
        if ss_total <= 0:
            continue
        ss_between = sum(
            np.sum(group_inverse == gi) * (values[group_inverse == gi].mean() - grand_mean) ** 2
            for gi in range(n_groups)
        )
        eta2[fi] = ss_between / ss_total

    return eta2


# ---------------------------------------------------------------------------
# Method 2: Analytical target + corpus nearest neighbor
# ---------------------------------------------------------------------------

def analytical_factor_embedding(
    factor_idx: int,
    loadings: np.ndarray,
    pca_model: PCA,
    scaler: StandardScaler | None,
    global_mean: np.ndarray,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a target embedding by back-projecting a factor direction analytically.

    target = global_mean + scale * factor_direction_in_original_space

    Args:
        factor_idx: Which factor to target.
        loadings: Factor loading matrix [n_pca_dims, n_factors].
        pca_model: Fitted PCA model.
        scaler: Fitted StandardScaler, or None.
        global_mean: Mean of the original (pre-residualization) embeddings [d].
        scale: How far along the factor direction to go.

    Returns:
        Tuple of (target embedding [d], unit direction vector [d]).
    """
    direction = back_project_factor(factor_idx, loadings, pca_model, scaler)
    target = global_mean + scale * direction
    return target, direction


def corpus_nearest_neighbor(
    target_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    metadata: list[dict],
    top_k: int = 10,
    text_field: str = "assistant_text",
    excerpt_length: int = 400,
    max_per_group: int | None = None,
    group_field: str = "input_group_id",
) -> list[dict]:
    """Find the top_k corpus embeddings nearest to target by cosine similarity.

    Args:
        target_embedding: Target vector [d].
        corpus_embeddings: Corpus matrix [n, d].
        metadata: Metadata rows aligned with corpus.
        top_k: Number of nearest neighbors to return.
        text_field: Metadata field containing response text.
        excerpt_length: Max characters in text excerpts.
        max_per_group: Optional maximum number of returned neighbors from the
            same metadata group.
        group_field: Metadata field used to define groups when max_per_group is
            set.

    Returns:
        List of dicts with index, similarity, and metadata for each neighbor.
    """
    # Normalize for cosine similarity.
    target_norm = target_embedding / (np.linalg.norm(target_embedding) + 1e-12)
    corpus_norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    corpus_normed = corpus_embeddings / (corpus_norms + 1e-12)

    similarities = corpus_normed @ target_norm
    ranked_indices = np.argsort(similarities)[::-1]

    results = []
    group_counts: dict[str, int] = {}
    for idx in ranked_indices:
        idx = int(idx)
        row = metadata[idx]
        if max_per_group is not None:
            group_id = str(row.get(group_field, idx))
            if group_counts.get(group_id, 0) >= max_per_group:
                continue
            group_counts[group_id] = group_counts.get(group_id, 0) + 1
        results.append({
            "index": idx,
            "similarity": float(similarities[idx]),
            "sample_id": row.get("sample_id"),
            "input_group_id": row.get("input_group_id"),
            "seed_user_message": str(row.get("seed_user_message", ""))[:200],
            "text_excerpt": str(row.get(text_field, ""))[:excerpt_length],
        })
        if len(results) >= top_k:
            break

    return results


# ---------------------------------------------------------------------------
# Method 3: Score-based corpus ranking by factor purity
# ---------------------------------------------------------------------------

def rank_by_factor_purity(
    scores: np.ndarray,
    metadata: list[dict],
    factor_idx: int,
    penalty_weight: float = 1.0,
    top_n: int = 20,
    text_field: str = "assistant_text",
    excerpt_length: int = 400,
) -> dict:
    """Rank samples by polarity-aware factor purity.

    Reported purity is symmetric and always non-negative:

        purity = |score[factor_idx]| - penalty_weight * mean(|score[j]| for j != factor_idx)

    with the final reported value floored at 0.

    Samples are still split by polarity before ranking:
    - HIGH candidates must have non-negative target score
    - LOW candidates must have non-positive target score

    This returns "pure" examples for both ends of the factor: large positive
    target score with low off-target activation (HIGH), and large negative
    target score with low off-target activation (LOW), while reporting purity
    as a positive magnitude on both sides.

    Args:
        scores: Factor scores [n_samples, n_factors].
        metadata: Metadata rows aligned with scores.
        factor_idx: Target factor whose positive/negative pure examples to find.
        penalty_weight: Weight for penalizing other factor magnitudes.
        top_n: Number of samples to return for each polarity.
        text_field: Metadata field containing response text.
        excerpt_length: Max characters in text excerpts.

    Returns:
        Dict with 'top' (HIGH polarity) and 'bottom' (LOW polarity) lists,
        each containing sample info dicts.
    """
    target_scores, other_abs_mean, purity_scores = compute_factor_purity(
        scores,
        factor_idx=factor_idx,
        penalty_weight=penalty_weight,
    )
    n = min(top_n, len(metadata))

    high_candidates = target_scores >= 0
    low_candidates = target_scores <= 0

    def _top_indices(objective: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.size == 0:
            candidate_indices = np.arange(len(metadata))
        order = np.argsort(objective[candidate_indices])[::-1]
        return candidate_indices[order[:n]]

    top_indices = _top_indices(purity_scores, high_candidates)
    bottom_indices = _top_indices(purity_scores, low_candidates)

    def _entry(idx: int) -> dict:
        row = metadata[idx]
        return {
            "index": int(idx),
            "purity_score": float(purity_scores[idx]),
            "target_factor_score": float(target_scores[idx]),
            "other_factors_mean_abs": float(other_abs_mean[idx]),
            "sample_id": row.get("sample_id"),
            "input_group_id": row.get("input_group_id"),
            "seed_user_message": str(row.get("seed_user_message", ""))[:200],
            "text_excerpt": str(row.get(text_field, ""))[:excerpt_length],
        }

    top = [_entry(int(idx)) for idx in top_indices]
    bottom = [_entry(int(idx)) for idx in bottom_indices]

    return {"factor_index": factor_idx, "top": top, "bottom": bottom}


def rank_by_factor_lagrangian(
    scores: np.ndarray,
    metadata: list[dict],
    factor_idx: int,
    penalty_weight: float = 1.0,
    target_quantile: float = 0.05,
    top_n: int = 20,
    text_field: str = "assistant_text",
    excerpt_length: int = 400,
) -> dict:
    """Rank samples by target score with a two-stage L1 off-target Lagrangian penalty.

    Stage 1:
        Restrict HIGH candidates to the top target_quantile of target scores.
        Restrict LOW candidates to the bottom target_quantile of target scores.

    Stage 2:
        Within each candidate pool, rank using:

            score[factor_idx] - penalty_weight * mean(|score[j]| for j != factor_idx)

        for HIGH, and symmetrically:

            -score[factor_idx] - penalty_weight * mean(|score[j]| for j != factor_idx)

    Args:
        scores: Factor scores [n_samples, n_factors].
        metadata: Metadata rows aligned with scores.
        factor_idx: Target factor whose constrained examples to find.
        penalty_weight: Lagrange multiplier on off-target factor magnitude.
        target_quantile: Fraction of samples to keep in each polarity pool before
            applying the off-target penalty. For example, 0.05 means restrict HIGH
            to the top 5% of target scores and LOW to the bottom 5%.
        top_n: Number of samples to return for each polarity.
        text_field: Metadata field containing response text.
        excerpt_length: Max characters in text excerpts.

    Returns:
        Dict with 'top' (HIGH polarity) and 'bottom' (LOW polarity) lists,
        each containing sample info dicts.
    """
    target_scores, other_abs_mean, high_objective, low_objective = compute_factor_lagrangian(
        scores,
        factor_idx=factor_idx,
        penalty_weight=penalty_weight,
    )
    n = min(top_n, len(metadata))

    if not 0.0 < target_quantile <= 0.5:
        raise ValueError(f"target_quantile must be in (0, 0.5], got {target_quantile!r}")

    candidate_count = max(1, int(np.ceil(len(metadata) * target_quantile)))
    order = np.argsort(target_scores)
    low_candidate_indices = order[:candidate_count]
    high_candidate_indices = order[-candidate_count:]

    high_candidates = np.zeros(len(metadata), dtype=bool)
    low_candidates = np.zeros(len(metadata), dtype=bool)
    high_candidates[high_candidate_indices] = True
    low_candidates[low_candidate_indices] = True

    def _top_indices(objective: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.size == 0:
            candidate_indices = np.arange(len(metadata))
        order = np.argsort(objective[candidate_indices])[::-1]
        return candidate_indices[order[:n]]

    top_indices = _top_indices(high_objective, high_candidates)
    bottom_indices = _top_indices(low_objective, low_candidates)

    def _entry(idx: int, objective: np.ndarray) -> dict:
        row = metadata[idx]
        return {
            "index": int(idx),
            "lagrangian_score": float(objective[idx]),
            "target_factor_score": float(target_scores[idx]),
            "other_factors_mean_abs": float(other_abs_mean[idx]),
            "sample_id": row.get("sample_id"),
            "input_group_id": row.get("input_group_id"),
            "seed_user_message": str(row.get("seed_user_message", ""))[:200],
            "text_excerpt": str(row.get(text_field, ""))[:excerpt_length],
        }

    top = [_entry(int(idx), high_objective) for idx in top_indices]
    bottom = [_entry(int(idx), low_objective) for idx in bottom_indices]

    return {
        "factor_index": factor_idx,
        "penalty_weight": float(penalty_weight),
        "target_quantile": float(target_quantile),
        "candidate_count": int(candidate_count),
        "top": top,
        "bottom": bottom,
    }


# ---------------------------------------------------------------------------
# Method 6: Max-spread — questions whose responses span the widest target-score range
# ---------------------------------------------------------------------------

def rank_prompts_by_max_spread(
    scores: np.ndarray,
    metadata: list[dict],
    factor_idx: int,
    top_n: int = 20,
    penalty_weight: float = 1.0,
    high_threshold: float | None = None,
    low_threshold: float | None = None,
    group_field: str = "input_group_id",
    text_field: str = "assistant_text",
    excerpt_length: int = 400,
) -> dict:
    """Find questions whose responses span the widest target-score range for a factor.

    For each prompt group, computes max_spread = max(score[factor_idx]) - min(score[factor_idx]).
    Returns the top_n groups ranked by max_spread, with the highest- and lowest-score
    response for each group.

    Args:
        scores: Factor scores [n_samples, n_factors].
        metadata: Metadata rows aligned with scores.
        factor_idx: Target factor to examine.
        top_n: Number of questions to return.
        penalty_weight: Weight for penalizing off-target factor magnitude in
            the reported purity score for each selected response.
        high_threshold: Optional signed lower bound on the maximum target score
            within a prompt group. If set, only groups with max score >= this
            threshold are retained.
        low_threshold: Optional signed upper bound on the minimum target score
            within a prompt group. If set, only groups with min score <= this
            threshold are retained.
        group_field: Metadata field used to identify prompt groups.
        text_field: Metadata field containing response text.
        excerpt_length: Max characters in text excerpts.

    Returns:
        Dict with keys:
            factor_index: int
            groups: list of dicts, each with keys max_spread, n_responses, high, low.
    """
    target_scores, other_abs_mean, purity_scores = compute_factor_purity(
        scores,
        factor_idx=factor_idx,
        penalty_weight=penalty_weight,
    )

    group_ids = np.array([str(row.get(group_field, i)) for i, row in enumerate(metadata)])
    unique_groups = np.unique(group_ids)

    group_spreads = []
    for gid in unique_groups:
        indices = np.where(group_ids == gid)[0]
        if len(indices) < 2:
            continue
        group_scores = target_scores[indices]
        group_max_score = float(group_scores.max())
        group_min_score = float(group_scores.min())
        if high_threshold is not None and group_max_score < high_threshold:
            continue
        if low_threshold is not None and group_min_score > low_threshold:
            continue
        max_spread = float(group_max_score - group_min_score)
        group_spreads.append({
            "group_id": gid,
            "max_spread": max_spread,
            "group_max_score": group_max_score,
            "group_min_score": group_min_score,
            "max_idx": int(indices[np.argmax(group_scores)]),
            "min_idx": int(indices[np.argmin(group_scores)]),
            "n_responses": int(len(indices)),
        })

    group_spreads.sort(key=lambda x: x["max_spread"], reverse=True)

    def _entry(idx: int) -> dict:
        row = metadata[idx]
        return {
            "index": int(idx),
            "purity_score": float(purity_scores[idx]),
            "target_factor_score": float(target_scores[idx]),
            "other_factors_mean_abs": float(other_abs_mean[idx]),
            "sample_id": row.get("sample_id"),
            "input_group_id": row.get("input_group_id"),
            "seed_user_message": str(row.get("seed_user_message", ""))[:200],
            "text_excerpt": str(row.get(text_field, ""))[:excerpt_length],
        }

    groups = [
        {
            "group_id": gs["group_id"],
            "max_spread": gs["max_spread"],
            "group_max_score": gs["group_max_score"],
            "group_min_score": gs["group_min_score"],
            "n_responses": gs["n_responses"],
            "high": _entry(gs["max_idx"]),
            "low": _entry(gs["min_idx"]),
        }
        for gs in group_spreads[:top_n]
    ]

    return {"factor_index": factor_idx, "groups": groups}

# ---------------------------------------------------------------------------
# Method 5: Contrastive centroid retrieval
# ---------------------------------------------------------------------------

def contrastive_factor_embedding(
    scores: np.ndarray,
    factor_idx: int,
    embedding_matrix: np.ndarray,
    center_embedding: np.ndarray,
    top_k: int = 100,
    scale: float = 3.0,
    normalize: bool = True,
    embedding_space: str = "original",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute contrastive target embeddings from high/low-scoring examples.

    Constructs a direction in the supplied embedding space by subtracting the mean
    embedding of low-scoring examples from the mean embedding of high-scoring
    examples, then displaces the supplied center along that direction.

    Args:
        scores: Factor scores [n_samples, n_factors].
        factor_idx: Which factor to target.
        embedding_matrix: Embeddings aligned with scores [n, d].
        center_embedding: Center of the embedding space used for target placement [d].
        top_k: Number of high/low examples to use for centroid estimation.
        scale: How far along the direction to place the targets.
        normalize: If True, L2-normalize the direction before scaling.
        embedding_space: Label for the embedding space used, for diagnostics.

    Returns:
        Tuple of (target_high [d], target_low [d], direction [d], diagnostics dict).
    """
    factor_scores = scores[:, factor_idx]
    order = np.argsort(factor_scores)

    n = min(top_k, len(factor_scores))
    high_indices = order[-n:]
    low_indices = order[:n]

    mu_high = embedding_matrix[high_indices].mean(axis=0)
    mu_low = embedding_matrix[low_indices].mean(axis=0)
    direction = mu_high - mu_low

    raw_direction_norm = float(np.linalg.norm(direction))

    if normalize and raw_direction_norm > 1e-12:
        direction = direction / raw_direction_norm

    target_high = center_embedding + scale * direction
    target_low = center_embedding - scale * direction

    diagnostics: dict[str, Any] = {
        "top_k": n,
        "scale": scale,
        "normalize": normalize,
        "embedding_space": embedding_space,
        "raw_direction_norm": raw_direction_norm,
        "mean_high_factor_score": float(factor_scores[high_indices].mean()),
        "mean_low_factor_score": float(factor_scores[low_indices].mean()),
        "selected_high_indices": high_indices.tolist(),
        "selected_low_indices": low_indices.tolist(),
    }

    return target_high, target_low, direction, diagnostics


def contrastive_factor_retrieval(
    scores: np.ndarray,
    factor_idx: int,
    embedding_matrix: np.ndarray,
    metadata: list[dict],
    center_embedding: np.ndarray,
    top_k: int = 100,
    neighbor_k: int = 20,
    scale: float = 3.0,
    normalize: bool = True,
    embedding_space: str = "original",
    text_field: str = "assistant_text",
    excerpt_length: int = 100000,
) -> dict:
    """Retrieve nearest corpus examples to contrastive factor targets.

    Computes a factor-specific direction by subtracting the mean embedding of
    low-scoring responses from the mean embedding of high-scoring responses, then
    retrieves the nearest real responses to the resulting high/low targets in
    the supplied embedding space.

    Args:
        scores: Factor scores [n_samples, n_factors].
        factor_idx: Which factor to target.
        embedding_matrix: Embeddings aligned with scores [n, d].
        metadata: Metadata rows aligned with corpus.
        center_embedding: Center of the embedding space used for target placement [d].
        top_k: Number of high/low examples to use for centroid estimation.
        neighbor_k: Number of nearest neighbors to retrieve per target.
        scale: How far along the direction to place the targets.
        normalize: If True, L2-normalize the direction before scaling.
        embedding_space: Label for the embedding space used, for diagnostics.
        text_field: Metadata field containing response text.
        excerpt_length: Max characters in text excerpts.

    Returns:
        Dict with keys: factor_index, top, bottom, and diagnostic fields.
    """
    target_high, target_low, _direction, diagnostics = contrastive_factor_embedding(
        scores=scores,
        factor_idx=factor_idx,
        embedding_matrix=embedding_matrix,
        center_embedding=center_embedding,
        top_k=top_k,
        scale=scale,
        normalize=normalize,
        embedding_space=embedding_space,
    )

    top = corpus_nearest_neighbor(
        target_high, embedding_matrix, metadata,
        top_k=neighbor_k, text_field=text_field, excerpt_length=excerpt_length,
    )
    bottom = corpus_nearest_neighbor(
        target_low, embedding_matrix, metadata,
        top_k=neighbor_k, text_field=text_field, excerpt_length=excerpt_length,
    )

    return {
        "factor_index": factor_idx,
        "top": top,
        "bottom": bottom,
        **diagnostics,
        "neighbor_k": neighbor_k,
    }


# ---------------------------------------------------------------------------
# Method 1: Gradient descent through embedding model
# ---------------------------------------------------------------------------

def optimize_factor_embedding(
    factor_idx: int,
    n_factors: int,
    loadings: np.ndarray,
    pca_model: PCA,
    scaler: StandardScaler | None,
    global_mean: np.ndarray,
    model_name: str = "Qwen/Qwen3-Embedding-4B",
    seq_length: int = 32,
    n_steps: int = 500,
    lr: float = 0.01,
    other_factor_penalty: float = 1.0,
    device: str | None = None,
) -> dict:
    """Optimize continuous token embeddings to maximize a target factor score.

    Loads the embedding model, initializes random continuous token embeddings,
    and optimizes them via gradient descent to produce an embedding that scores
    high on factor_idx and low on all other factors.

    Args:
        factor_idx: Target factor to maximize.
        n_factors: Total number of factors.
        loadings: Factor loading matrix [n_pca_dims, n_factors].
        pca_model: Fitted PCA model for back-projection.
        scaler: Fitted StandardScaler, or None.
        global_mean: Mean of the original embeddings [d].
        model_name: HuggingFace model name for the embedding model.
        seq_length: Length of the token sequence to optimize.
        n_steps: Number of optimization steps.
        lr: Learning rate.
        other_factor_penalty: Weight for penalizing other factor scores.
        device: Device to use ("cuda", "cpu", or None for auto).

    Returns:
        Dict with keys:
            optimized_embedding: Final embedding [d] as numpy array.
            trajectory: List of dicts with loss/scores per step (sampled).
            final_scores: Factor scores for the optimized embedding [n_factors].
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert factor analysis components to torch tensors.
    pca_components = torch.tensor(pca_model.components_, dtype=torch.float32, device=device)
    pca_mean = torch.tensor(pca_model.mean_, dtype=torch.float32, device=device)
    loadings_t = torch.tensor(loadings, dtype=torch.float32, device=device)

    if scaler is not None:
        scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
        scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)
    else:
        scaler_mean = None
        scaler_scale = None

    # Load embedding model.
    print(f"Loading embedding model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device)
    model.eval()

    # Freeze all model parameters.
    for param in model.parameters():
        param.requires_grad_(False)

    # Get the token embedding layer.
    embed_layer = model.get_input_embeddings()
    hidden_dim = embed_layer.embedding_dim

    # Initialize random continuous token embeddings.
    # Start from random vocab embeddings for a more realistic initialization.
    rng = torch.Generator(device=device)
    rng.manual_seed(42)
    vocab_size = embed_layer.num_embeddings
    random_ids = torch.randint(0, vocab_size, (seq_length,), generator=rng, device=device)
    token_embeds = embed_layer(random_ids).detach().clone()
    token_embeds.requires_grad_(True)

    # Create a simple attention mask (all ones).
    attention_mask = torch.ones(1, seq_length, dtype=torch.long, device=device)

    optimizer = torch.optim.Adam([token_embeds], lr=lr)
    trajectory: list[dict] = []

    def _compute_factor_scores(embedding: torch.Tensor) -> torch.Tensor:
        """Project a single embedding [d] to factor scores [n_factors]."""
        x = embedding
        if scaler_mean is not None and scaler_scale is not None:
            x = (x - scaler_mean) / scaler_scale
        # PCA projection: (x - pca_mean) @ components.T
        pca_scores = (x - pca_mean) @ pca_components.T
        # Factor scores: pca_scores @ loadings
        factor_scores = pca_scores @ loadings_t
        return factor_scores

    for step in range(n_steps):
        optimizer.zero_grad()

        # Forward pass through the model with continuous embeddings.
        outputs = model(inputs_embeds=token_embeds.unsqueeze(0), attention_mask=attention_mask)
        # Mean pooling over sequence length.
        hidden_states = outputs.last_hidden_state  # [1, seq_len, hidden_dim]
        pooled = (hidden_states * attention_mask.unsqueeze(-1).float()).sum(dim=1)
        pooled = pooled / attention_mask.sum(dim=1, keepdim=True).float()
        embedding = pooled.squeeze(0)  # [hidden_dim]

        # L2 normalize.
        embedding = embedding / (embedding.norm() + 1e-12)

        # Compute factor scores.
        factor_scores = _compute_factor_scores(embedding)

        # Loss: minimize -target_score + penalty * sum(other_scores^2).
        target_score = factor_scores[factor_idx]
        other_scores = torch.cat([factor_scores[:factor_idx], factor_scores[factor_idx + 1:]])
        loss = -target_score + other_factor_penalty * (other_scores ** 2).sum()

        loss.backward()
        optimizer.step()

        # Log trajectory (sample every 10 steps + first and last).
        if step % 10 == 0 or step == n_steps - 1:
            trajectory.append({
                "step": step,
                "loss": float(loss.item()),
                "target_score": float(target_score.item()),
                "other_scores_rms": float((other_scores ** 2).mean().sqrt().item()),
            })

    # Final forward pass to get the optimized embedding.
    with torch.no_grad():
        outputs = model(inputs_embeds=token_embeds.unsqueeze(0), attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled = (hidden_states * attention_mask.unsqueeze(-1).float()).sum(dim=1)
        pooled = pooled / attention_mask.sum(dim=1, keepdim=True).float()
        final_embedding = pooled.squeeze(0)
        final_embedding = final_embedding / (final_embedding.norm() + 1e-12)
        final_scores = _compute_factor_scores(final_embedding)

    optimized_embedding = final_embedding.cpu().numpy().astype(np.float64)
    final_scores_np = final_scores.cpu().numpy().astype(np.float64)

    print(f"Optimization complete (factor {factor_idx}):")
    print(f"  Target factor score: {final_scores_np[factor_idx]:.4f}")
    print(f"  Other factors RMS: {np.sqrt(np.mean(np.delete(final_scores_np, factor_idx) ** 2)):.4f}")

    return {
        "optimized_embedding": optimized_embedding,
        "trajectory": trajectory,
        "final_scores": final_scores_np,
    }
