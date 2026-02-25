#!/usr/bin/env python3
"""Compare two LoRA checkpoints: principal angles, Frobenius distance, norms.

Checks that two LoRA adapters trained with different seeds are genuinely
independent solutions rather than identical copies.

Works in low-rank space to avoid materializing full (d_out x d_in) matrices.

Results (2026-02-24, checkpoints vs checkpoints-rerun, seed=42 vs seed=123):
  - Normalized Frobenius Distance: mean 1.32 (well above 1.0 — the difference
    is actually larger than the individual adapter norms, meaning the solutions
    point in very different directions)
  - Cosine Similarity: mean 0.13 (near-orthogonal; identical would be 1.0)
  - Principal Angles: mean 55.6 deg across all layers (90 = fully orthogonal,
    0 = identical subspace)
  - Norms are comparable: ||dW1|| = 4.13 vs ||dW2|| = 4.15 (similar magnitude,
    just different directions)
  Conclusion: the two seeds produced genuinely independent LoRA solutions.
"""

import numpy as np
import torch
from safetensors import safe_open
from pathlib import Path

CKPT_1 = Path("scratch/20Feb-nplus/checkpoints/final/adapter_model.safetensors")
CKPT_2 = Path("scratch/20Feb-nplus/checkpoints-rerun/final/adapter_model.safetensors")
ALPHA = 8
RANK = 4


def load_weights(path):
    """Load LoRA weights from safetensors file."""
    tensors = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key).float()
    return tensors


def get_ab(weights, prefix):
    """Get A and B matrices for a LoRA module."""
    A = weights[f"{prefix}.lora_A.weight"]  # (r, d_in)
    B = weights[f"{prefix}.lora_B.weight"]  # (d_out, r)
    return A.numpy(), B.numpy()


def low_rank_frob_norm(B, A, scale):
    """||scale * B @ A||_F computed via SVD of B @ A in low-rank form.

    Since B is (d_out, r) and A is (r, d_in), compute SVD of the r x r
    inner product instead of the full matrix.
    """
    # ||B @ A||_F^2 = tr((BA)^T (BA)) = tr(A^T B^T B A)
    # = ||S||_F^2 where S are singular values of B @ A
    # But we can compute it as: ||B @ A||_F^2 = tr(A A^T @ B^T B)
    # Both A A^T and B^T B are r x r
    BtB = B.T @ B  # (r, r)
    AAt = A @ A.T  # (r, r)
    frob_sq = np.trace(BtB @ AAt)
    return abs(scale) * np.sqrt(max(frob_sq, 0.0))


def low_rank_frob_diff(B1, A1, B2, A2, scale):
    """||scale * (B1 A1 - B2 A2)||_F without materializing full matrices.

    Stack into a rank-2r matrix: [B1, -B2] @ [A1; A2] and compute its Frobenius norm.
    """
    B_cat = np.concatenate([B1, -B2], axis=1)  # (d_out, 2r)
    A_cat = np.concatenate([A1, A2], axis=0)   # (2r, d_in)
    BtB = B_cat.T @ B_cat  # (2r, 2r)
    AAt = A_cat @ A_cat.T  # (2r, 2r)
    frob_sq = np.trace(BtB @ AAt)
    return abs(scale) * np.sqrt(max(frob_sq, 0.0))


def low_rank_cos_sim(B1, A1, B2, A2):
    """Cosine similarity between two low-rank matrices B1@A1 and B2@A2.

    <vec(B1 A1), vec(B2 A2)> = tr((B1 A1)^T (B2 A2)) = tr(A1^T B1^T B2 A2)
    """
    inner = np.trace(A1 @ A2.T @ B2.T @ B1)  # all r x r intermediates
    n1 = low_rank_frob_norm(B1, A1, 1.0)
    n2 = low_rank_frob_norm(B2, A2, 1.0)
    return inner / (n1 * n2 + 1e-12)


def principal_angles_low_rank(B1, A1, B2, A2):
    """Principal angles between column spaces of ΔW1 and ΔW2.

    The column space of ΔW = B @ A is spanned by columns of B (since A has
    full row rank for a trained LoRA). So we compare col(B1) vs col(B2).
    """
    Q1, _ = np.linalg.qr(B1)  # (d_out, r) orthonormal
    Q2, _ = np.linalg.qr(B2)  # (d_out, r) orthonormal
    M = Q1.T @ Q2  # (r, r)
    svals = np.linalg.svd(M, compute_uv=False)
    svals = np.clip(svals, 0.0, 1.0)
    return np.degrees(np.arccos(svals))


def compare_layer(w1, w2, prefix, short_name):
    """Compare a single LoRA module between two checkpoints."""
    scale = ALPHA / RANK
    A1, B1 = get_ab(w1, prefix)
    A2, B2 = get_ab(w2, prefix)

    norm1 = low_rank_frob_norm(B1, A1, scale)
    norm2 = low_rank_frob_norm(B2, A2, scale)
    diff_norm = low_rank_frob_diff(B1, A1, B2, A2, scale)
    nfd = diff_norm / norm1 if norm1 > 0 else float("inf")
    cos_sim = low_rank_cos_sim(B1, A1, B2, A2)
    angles = principal_angles_low_rank(B1, A1, B2, A2)

    return {
        "name": short_name,
        "norm1": norm1,
        "norm2": norm2,
        "diff_norm": diff_norm,
        "nfd": nfd,
        "cos_sim": cos_sim,
        "angles": angles,
    }


def main():
    print("Loading checkpoints...")
    w1 = load_weights(CKPT_1)
    w2 = load_weights(CKPT_2)

    prefixes = sorted(set(
        k.rsplit(".lora_A.weight", 1)[0]
        for k in w1.keys()
        if ".lora_A.weight" in k
    ))

    print(f"Comparing {len(prefixes)} LoRA modules")
    print(f"  Checkpoint 1 (seed=42):  {CKPT_1}")
    print(f"  Checkpoint 2 (seed=123): {CKPT_2}")
    print(f"  alpha={ALPHA}, rank={RANK}")
    print()

    results = []
    # For global Frobenius: accumulate A, B pairs
    all_A1, all_B1, all_A2, all_B2 = [], [], [], []

    for prefix in prefixes:
        short = prefix.replace("base_model.model.model.layers.", "L").replace(".self_attn.", ".")
        r = compare_layer(w1, w2, prefix, short)
        results.append(r)
        A1, B1 = get_ab(w1, prefix)
        A2, B2 = get_ab(w2, prefix)
        all_A1.append(A1)
        all_B1.append(B1)
        all_A2.append(A2)
        all_B2.append(B2)

    # --- Per-layer table ---
    print(f"{'Module':<20} {'||dW1||':>9} {'||dW2||':>9} {'NFD':>9} {'CosSim':>8} {'PrincipalAngles (deg)':>30}")
    print("-" * 95)
    for r in results:
        angles_str = ", ".join(f"{a:.1f}" for a in r["angles"])
        print(f"{r['name']:<20} {r['norm1']:>9.5f} {r['norm2']:>9.5f} {r['nfd']:>9.4f} {r['cos_sim']:>8.4f}   [{angles_str}]")

    # --- Aggregates ---
    print("\n" + "=" * 95)
    print("AGGREGATE STATISTICS")
    print("=" * 95)

    nfds = [r["nfd"] for r in results]
    coss = [r["cos_sim"] for r in results]
    all_angles = np.concatenate([r["angles"] for r in results])
    norms1 = [r["norm1"] for r in results]
    norms2 = [r["norm2"] for r in results]

    print(f"  Normalized Frobenius Distance:  mean={np.mean(nfds):.4f}, min={np.min(nfds):.4f}, max={np.max(nfds):.4f}")
    print(f"  Cosine Similarity:              mean={np.mean(coss):.4f}, min={np.min(coss):.4f}, max={np.max(coss):.4f}")
    print(f"  Principal Angles (deg):         mean={np.mean(all_angles):.1f}, min={np.min(all_angles):.1f}, max={np.max(all_angles):.1f}")
    print(f"  ||dW|| norms:                   ckpt1 mean={np.mean(norms1):.5f}, ckpt2 mean={np.mean(norms2):.5f}")

    # Global: sum of squared Frobenius norms across layers
    global_norm1_sq = sum(r["norm1"] ** 2 for r in results)
    global_norm2_sq = sum(r["norm2"] ** 2 for r in results)
    # For global diff, we need sum of ||dW1_i - dW2_i||_F^2
    global_diff_sq = sum(r["diff_norm"] ** 2 for r in results)
    g1 = np.sqrt(global_norm1_sq)
    g2 = np.sqrt(global_norm2_sq)
    g_diff = np.sqrt(global_diff_sq)

    print(f"\n  Global (summed across all layers):")
    print(f"    ||dW1||_F = {g1:.5f}")
    print(f"    ||dW2||_F = {g2:.5f}")
    print(f"    ||dW1 - dW2||_F = {g_diff:.5f}")
    print(f"    Normalized Frobenius Distance = {g_diff / g1:.4f}")

    # Interpretation
    mean_nfd = np.mean(nfds)
    mean_angle = np.mean(all_angles)
    print("\n  Interpretation:")
    if mean_nfd < 0.01:
        print("    WARNING: Checkpoints appear nearly IDENTICAL (NFD < 0.01)")
    elif mean_nfd < 0.3:
        print(f"    Checkpoints are RELATED but distinguishable (mean NFD={mean_nfd:.3f})")
    else:
        print(f"    Checkpoints are clearly INDEPENDENT solutions (mean NFD={mean_nfd:.3f})")

    if mean_angle > 45:
        print(f"    Subspaces are substantially different (mean principal angle={mean_angle:.1f} deg)")
    elif mean_angle > 15:
        print(f"    Subspaces share some structure but differ (mean principal angle={mean_angle:.1f} deg)")
    else:
        print(f"    Subspaces are quite aligned (mean principal angle={mean_angle:.1f} deg)")


if __name__ == "__main__":
    main()
