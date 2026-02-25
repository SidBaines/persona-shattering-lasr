#!/usr/bin/env python3
"""Compare LoRA checkpoints: principal angles, Frobenius distance, norms.

Checks whether pairs of LoRA adapters are genuinely independent solutions
or similar/identical copies.

Works in low-rank space to avoid materializing full (d_out x d_in) matrices.

NFD (Normalized Frobenius Distance) = ||dW1 - dW2||_F / ||dW1||_F
  NFD ≈ 0   → identical adapters
  NFD ≈ 1.41 → orthogonal adapters of equal norm (by Pythagoras)
  NFD > 1   → the difference vector is larger than the adapters themselves,
              meaning they point in very different directions in weight space

Comparisons run:

  1. n+ (seed=42) vs n+ rerun (seed=123)
     Same training data & hyperparameters, different random seed.
     Results (2026-02-24):
       NFD mean=1.32, CosSim mean=0.13, PrincipalAngles mean=55.6 deg
       Norms: 4.13 vs 4.15
       → Genuinely independent solutions (near-orthogonal, similar magnitude)

  2. n+ (seed=42) vs n- (seed=42)
     Different training data (neuroticism-increasing vs neuroticism-decreasing).
     Results (2026-02-25):
       NFD mean=1.19, CosSim mean=0.22, PrincipalAngles mean=76.4 deg
       Norms: 4.13 vs 3.84
       → Independent solutions; subspaces even more orthogonal than seed reruns
"""

import numpy as np
import torch
from safetensors import safe_open
from pathlib import Path

ALPHA = 8
RANK = 4

COMPARISONS = [
    {
        "name": "n+ (seed=42)  vs  n+ rerun (seed=123)",
        "ckpt1": Path("scratch/20Feb-nplus/checkpoints/final/adapter_model.safetensors"),
        "ckpt2": Path("scratch/20Feb-nplus/checkpoints-rerun/final/adapter_model.safetensors"),
        "label1": "n+ (seed=42)",
        "label2": "n+ rerun (seed=123)",
    },
    {
        "name": "n+ (seed=42)  vs  n-",
        "ckpt1": Path("scratch/20Feb-nplus/checkpoints/final/adapter_model.safetensors"),
        "ckpt2": Path("scratch/20Feb-nminus/checkpoints-r4/final/adapter_model.safetensors"),
        "label1": "n+ (seed=42)",
        "label2": "n-",
    },
]


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
    """Principal angles between column spaces of dW1 and dW2.

    The column space of dW = B @ A is spanned by columns of B (since A has
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


def run_comparison(comp: dict):
    """Run a single pairwise comparison and print results."""
    print(f"\n{'#' * 95}")
    print(f"  {comp['name']}")
    print(f"{'#' * 95}")

    print(f"  {comp['label1']}: {comp['ckpt1']}")
    print(f"  {comp['label2']}: {comp['ckpt2']}")
    print(f"  alpha={ALPHA}, rank={RANK}")

    w1 = load_weights(comp["ckpt1"])
    w2 = load_weights(comp["ckpt2"])

    prefixes = sorted(set(
        k.rsplit(".lora_A.weight", 1)[0]
        for k in w1.keys()
        if ".lora_A.weight" in k
    ))

    print(f"  Comparing {len(prefixes)} LoRA modules\n")

    results = []
    for prefix in prefixes:
        short = prefix.replace("base_model.model.model.layers.", "L").replace(".self_attn.", ".")
        r = compare_layer(w1, w2, prefix, short)
        results.append(r)

    # --- Per-layer table ---
    l1 = comp["label1"]
    l2 = comp["label2"]
    print(f"{'Module':<20} {'||' + l1 + '||':>12} {'||' + l2 + '||':>12} {'NFD':>9} {'CosSim':>8} {'PrincipalAngles (deg)':>30}")
    print("-" * 100)
    for r in results:
        angles_str = ", ".join(f"{a:.1f}" for a in r["angles"])
        print(f"{r['name']:<20} {r['norm1']:>12.5f} {r['norm2']:>12.5f} {r['nfd']:>9.4f} {r['cos_sim']:>8.4f}   [{angles_str}]")

    # --- Aggregates ---
    print("\n" + "=" * 100)
    print(f"AGGREGATE: {comp['name']}")
    print("=" * 100)

    nfds = [r["nfd"] for r in results]
    coss = [r["cos_sim"] for r in results]
    all_angles = np.concatenate([r["angles"] for r in results])
    norms1 = [r["norm1"] for r in results]
    norms2 = [r["norm2"] for r in results]

    print(f"  Normalized Frobenius Distance:  mean={np.mean(nfds):.4f}, min={np.min(nfds):.4f}, max={np.max(nfds):.4f}")
    print(f"  Cosine Similarity:              mean={np.mean(coss):.4f}, min={np.min(coss):.4f}, max={np.max(coss):.4f}")
    print(f"  Principal Angles (deg):         mean={np.mean(all_angles):.1f}, min={np.min(all_angles):.1f}, max={np.max(all_angles):.1f}")
    print(f"  ||dW|| norms:                   {l1} mean={np.mean(norms1):.5f}, {l2} mean={np.mean(norms2):.5f}")

    # Global: sum of squared Frobenius norms across layers
    global_norm1_sq = sum(r["norm1"] ** 2 for r in results)
    global_norm2_sq = sum(r["norm2"] ** 2 for r in results)
    global_diff_sq = sum(r["diff_norm"] ** 2 for r in results)
    g1 = np.sqrt(global_norm1_sq)
    g2 = np.sqrt(global_norm2_sq)
    g_diff = np.sqrt(global_diff_sq)

    print(f"\n  Global (summed across all layers):")
    print(f"    ||{l1}||_F = {g1:.5f}")
    print(f"    ||{l2}||_F = {g2:.5f}")
    print(f"    ||{l1} - {l2}||_F = {g_diff:.5f}")
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


def main():
    for comp in COMPARISONS:
        run_comparison(comp)


if __name__ == "__main__":
    main()
