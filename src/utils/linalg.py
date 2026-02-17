"""
Utility functions for linear algebra
"""

from torch import Tensor, linalg, sqrt


def reduce_rank(matrix: Tensor, new_rank: int) -> Tensor:
    """Returns the best rank-new_rank approximation of matrix via truncated SVD.

    Uses the Eckart-Young theorem: truncating to the top-k singular values/vectors
    gives the optimal rank-k approximation in Frobenius norm.

    Args:
        matrix: 2D tensor of shape (m, n).
        new_rank: Number of singular values to keep. Must be in [1, min(m, n)].

    Returns:
        Tensor of shape (m, n) with rank at most new_rank.
    """
    max_rank = min(matrix.shape)
    if not (1 <= new_rank <= max_rank):
        raise ValueError(f"new_rank must be between 1 and {max_rank}, got {new_rank}")

    U, S, Vh = linalg.svd(matrix, full_matrices=False)  # U: (m, k), S: (k,), Vh: (k, n)

    # Reconstruct using only the top new_rank singular triplets.
    # Equivalent to U[:, :new_rank] @ diag(S[:new_rank]) @ Vh[:new_rank, :]
    return (U[:, :new_rank] * S[:new_rank]) @ Vh[:new_rank, :]


def reduce_lora_rank_efficient(
    A: Tensor, B: Tensor, new_rank: int
) -> tuple[Tensor, Tensor]:
    """Reduces the rank of a LoRA weight update B @ A without materializing the full matrix.

    A standard rank reduction would compute svd(B @ A), which creates an (m, n) matrix.
    This function avoids that by first QR-decomposing B and A separately, reducing the
    problem to an SVD of a small (r, r) matrix instead.

    Args:
        A: LoRA factor of shape (r, n), where r is the current rank.
        B: LoRA factor of shape (m, r), where r is the current rank.
        new_rank: Target rank. Must be in [1, r].

    Raises:
        ValueError: If new_rank is not in [1, r].

    Returns:
        Tuple (new_A, new_B) where new_A has shape (new_rank, n) and new_B has
        shape (m, new_rank), such that new_B @ new_A ≈ B @ A.
    """
    r = A.shape[0]  # current rank
    if not (1 <= new_rank <= r):
        raise ValueError(f"new_rank must be between 1 and {r}, got {new_rank}")

    # QR-decompose B and A^T into an orthonormal factor Q and upper-triangular R:
    #   B    = Q_b @ R_b   (m, r) = (m, r) @ (r, r)
    #   A^T  = Q_a @ R_a   (n, r) = (n, r) @ (r, r)
    # Q has orthonormal columns (Q^T @ Q = I), making it an isometry that preserves
    # singular values. R captures the remaining (triangular) structure.
    Q_b, R_b = linalg.qr(B)  # Q_b: (m, r), R_b: (r, r)
    Q_a, R_a = linalg.qr(A.T)  # Q_a: (n, r), R_a: (r, r)

    # Substituting the QR factors:
    #   B @ A = (Q_b @ R_b) @ (Q_a @ R_a)^T = Q_b @ (R_b @ R_a^T) @ Q_a^T
    # So the SVD only needs to be computed on the small (r, r) core R_b @ R_a^T.
    U_small, S, Vh_small = linalg.svd(R_b @ R_a.T, full_matrices=False)

    # Unfold back to get the SVD of B @ A:
    #   B @ A = Q_b @ U_small @ diag(S) @ Vh_small @ Q_a^T
    # Since Q_b and Q_a have orthonormal columns, Q_b @ U_small and Vh_small @ Q_a^T
    # also have orthonormal columns/rows respectively, forming a valid SVD of B @ A.
    U = Q_b @ U_small  # (m, r)
    Vh = Vh_small @ Q_a.T  # (r, n)

    # Split singular values evenly across new_A and new_B so that
    # new_B @ new_A = (U * S_sqrt) @ (S_sqrt * Vh) = U @ diag(S) @ Vh.
    S_sqrt = sqrt(S[:new_rank])
    new_B = U[:, :new_rank] * S_sqrt[None, :]  # (m, new_rank)
    new_A = S_sqrt[:, None] * Vh[:new_rank, :]  # (new_rank, n)

    return new_A, new_B
