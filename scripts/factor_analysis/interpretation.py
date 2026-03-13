"""Factor interpretation methods.

Five approaches for understanding what each factor represents:
1. Gradient descent through embedding model (optimize_factor_embedding)
2. Analytical back-projection + corpus nearest neighbor
3. Score-based corpus ranking by factor purity
4. Gradient descent + corpus lookup (compose methods 1 + 2 in experiment script)
5. Contrastive centroid retrieval (contrastive_factor_retrieval)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

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
    group_ids = np.array([str(row.get(group_field, i)) for i, row in enumerate(metadata)])
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
) -> list[dict]:
    """Find the top_k corpus embeddings nearest to target by cosine similarity.

    Args:
        target_embedding: Target vector [d].
        corpus_embeddings: Corpus matrix [n, d].
        metadata: Metadata rows aligned with corpus.
        top_k: Number of nearest neighbors to return.
        text_field: Metadata field containing response text.
        excerpt_length: Max characters in text excerpts.

    Returns:
        List of dicts with index, similarity, and metadata for each neighbor.
    """
    # Normalize for cosine similarity.
    target_norm = target_embedding / (np.linalg.norm(target_embedding) + 1e-12)
    corpus_norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    corpus_normed = corpus_embeddings / (corpus_norms + 1e-12)

    similarities = corpus_normed @ target_norm
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        idx = int(idx)
        row = metadata[idx]
        results.append({
            "index": idx,
            "similarity": float(similarities[idx]),
            "sample_id": row.get("sample_id"),
            "input_group_id": row.get("input_group_id"),
            "seed_user_message": str(row.get("seed_user_message", ""))[:200],
            "text_excerpt": str(row.get(text_field, ""))[:excerpt_length],
        })

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
    """Rank samples by factor purity: high on target factor, low on others.

    purity_score = score[factor_idx] - penalty_weight * mean(|score[j]| for j != factor_idx)

    Args:
        scores: Factor scores [n_samples, n_factors].
        metadata: Metadata rows aligned with scores.
        factor_idx: Target factor to maximize.
        penalty_weight: Weight for penalizing other factor magnitudes.
        top_n: Number of samples to return from each end.
        text_field: Metadata field containing response text.
        excerpt_length: Max characters in text excerpts.

    Returns:
        Dict with 'top' and 'bottom' lists, each containing sample info dicts.
    """
    n_factors = scores.shape[1]
    target_scores = scores[:, factor_idx]

    # Mean absolute score on all other factors.
    other_mask = np.ones(n_factors, dtype=bool)
    other_mask[factor_idx] = False
    other_abs_mean = np.abs(scores[:, other_mask]).mean(axis=1)

    purity = target_scores - penalty_weight * other_abs_mean
    order = np.argsort(purity)
    n = min(top_n, len(metadata))

    def _entry(idx: int) -> dict:
        row = metadata[idx]
        return {
            "index": int(idx),
            "purity_score": float(purity[idx]),
            "target_factor_score": float(target_scores[idx]),
            "other_factors_mean_abs": float(other_abs_mean[idx]),
            "sample_id": row.get("sample_id"),
            "input_group_id": row.get("input_group_id"),
            "seed_user_message": str(row.get("seed_user_message", ""))[:200],
            "text_excerpt": str(row.get(text_field, ""))[:excerpt_length],
        }

    top = [_entry(int(idx)) for idx in order[-n:][::-1]]
    bottom = [_entry(int(idx)) for idx in order[:n]]

    return {"factor_index": factor_idx, "top": top, "bottom": bottom}


# ---------------------------------------------------------------------------
# Method 5: Contrastive centroid retrieval
# ---------------------------------------------------------------------------

def contrastive_factor_embedding(
    scores: np.ndarray,
    factor_idx: int,
    corpus_embeddings: np.ndarray,
    global_mean: np.ndarray,
    top_k: int = 100,
    scale: float = 3.0,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute contrastive target embeddings from high/low-scoring examples.

    Constructs a direction in corpus embedding space by subtracting the mean
    embedding of low-scoring examples from the mean embedding of high-scoring
    examples, then displaces the global mean along that direction.

    Args:
        scores: Factor scores [n_samples, n_factors].
        factor_idx: Which factor to target.
        corpus_embeddings: Corpus embeddings aligned with scores [n, d].
        global_mean: Mean of the original (pre-residualization) embeddings [d].
        top_k: Number of high/low examples to use for centroid estimation.
        scale: How far along the direction to place the targets.
        normalize: If True, L2-normalize the direction before scaling.

    Returns:
        Tuple of (target_high [d], target_low [d], direction [d], diagnostics dict).
    """
    factor_scores = scores[:, factor_idx]
    order = np.argsort(factor_scores)

    n = min(top_k, len(factor_scores))
    high_indices = order[-n:]
    low_indices = order[:n]

    mu_high = corpus_embeddings[high_indices].mean(axis=0)
    mu_low = corpus_embeddings[low_indices].mean(axis=0)
    direction = mu_high - mu_low

    raw_direction_norm = float(np.linalg.norm(direction))

    if normalize and raw_direction_norm > 1e-12:
        direction = direction / raw_direction_norm

    target_high = global_mean + scale * direction
    target_low = global_mean - scale * direction

    diagnostics: dict[str, Any] = {
        "top_k": n,
        "scale": scale,
        "normalize": normalize,
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
    corpus_embeddings: np.ndarray,
    metadata: list[dict],
    global_mean: np.ndarray,
    top_k: int = 100,
    neighbor_k: int = 20,
    scale: float = 3.0,
    normalize: bool = True,
    text_field: str = "assistant_text",
    excerpt_length: int = 100000,
) -> dict:
    """Retrieve nearest corpus examples to contrastive factor targets.

    Computes a factor-specific direction by subtracting the mean embedding of
    low-scoring responses from the mean embedding of high-scoring responses, then
    retrieves the nearest real responses to the resulting high/low targets in
    corpus space.

    Args:
        scores: Factor scores [n_samples, n_factors].
        factor_idx: Which factor to target.
        corpus_embeddings: Corpus embeddings aligned with scores [n, d].
        metadata: Metadata rows aligned with corpus.
        global_mean: Mean of the original (pre-residualization) embeddings [d].
        top_k: Number of high/low examples to use for centroid estimation.
        neighbor_k: Number of nearest neighbors to retrieve per target.
        scale: How far along the direction to place the targets.
        normalize: If True, L2-normalize the direction before scaling.
        text_field: Metadata field containing response text.
        excerpt_length: Max characters in text excerpts.

    Returns:
        Dict with keys: factor_index, top, bottom, and diagnostic fields.
    """
    target_high, target_low, _direction, diagnostics = contrastive_factor_embedding(
        scores=scores,
        factor_idx=factor_idx,
        corpus_embeddings=corpus_embeddings,
        global_mean=global_mean,
        top_k=top_k,
        scale=scale,
        normalize=normalize,
    )

    top = corpus_nearest_neighbor(
        target_high, corpus_embeddings, metadata,
        top_k=neighbor_k, text_field=text_field, excerpt_length=excerpt_length,
    )
    bottom = corpus_nearest_neighbor(
        target_low, corpus_embeddings, metadata,
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
