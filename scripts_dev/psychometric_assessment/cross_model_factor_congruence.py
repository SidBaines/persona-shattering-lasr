"""Cross-model Tucker's congruence of factor-analysis loadings.

Given two FA rotations (same rotation method applied to separate models'
response matrices, built on the same questionnaires / items), compute
pairwise Tucker's φ congruence on aligned loading vectors and match the
factors optimally.

Tucker's congruence between factor vectors ``a`` and ``b`` (same items):

    φ(a, b) = Σ a_i b_i  /  √( Σ a_i²  ·  Σ b_i² )

Interpretation (Lorenzo-Seva & ten Berge 2006):
    |φ| ≥ 0.95   → equal
    |φ| ≥ 0.85   → fair similarity
    |φ| < 0.85   → poor similarity

Signs are arbitrary under orthogonal/oblique rotation, so we flip factor B
to match A's sign where that maximises |φ|. The reported φ is the signed
congruence after flipping.

Match strategy: for every pair of factors (i from model A, j from model B),
we compute |φ_ij|, then solve the optimal assignment via the Hungarian
algorithm so each factor of A matches at most one factor of B. Unmatched
factors on the larger side are reported with their single-best partner.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

from src_dev.unsupervised_runs.io import hydrate_dataset_subtree

logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("scratch/psychometric_comparison")


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class FaRun:
    """One FA-run artifact set to compare.

    ``combined_dir`` is the local scratch path to the combined FA dir. The
    loadings file is expected at
    ``{combined_dir}/factor_analysis/{residualize}/fa_{n_factors}_{method}_{rotation}.npz``
    and the item labels at the corresponding ``_item_labels.json``.
    """
    label: str
    combined_dir: Path
    n_factors: int = 7
    method: str = "principal"
    rotation: str = "oblimin"
    residualize: str = "raw"


# Edit these entries to point at the two runs you want to compare.
RUNS: list[FaRun] = [
    FaRun(
        label="llama-3.1-8b",
        combined_dir=Path(
            "scratch/psychometric_fa/combined-R[B]-Q[v5+trait_ocean_v1]"
        ),
        n_factors=7,
        rotation="oblimin",
    ),
    FaRun(
        label="qwen2.5-7b",
        combined_dir=Path(
            "scratch/psychometric_fa/combined-R[B]-Q[v5+trait_ocean_v1]-qm_qwen257binstruct"
        ),
        n_factors=7,
        rotation="oblimin",
    ),
]

TAG = "B_cross_model_congruence"


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class LoadedFa:
    label: str
    loadings: np.ndarray          # (n_items, n_factors)
    col_ids: list[str]            # length n_items, col_id per row of loadings
    item_texts: dict[str, str]
    communalities: np.ndarray     # (n_items,)
    n_factors: int
    rotation: str


def load_fa_run(run: FaRun) -> LoadedFa:
    fa_dir = run.combined_dir / "factor_analysis" / run.residualize
    npz_path = fa_dir / f"fa_{run.n_factors}_{run.method}_{run.rotation}.npz"
    labels_path = fa_dir / f"fa_{run.n_factors}_{run.method}_{run.rotation}_item_labels.json"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"[{run.label}] FA loadings not found at {npz_path}. "
            "Run Stage 3 of the psychometric pipeline for this model first."
        )
    z = np.load(npz_path)
    loadings = z["loadings"]
    comm = z["communalities"]
    with open(labels_path) as f:
        labels = json.load(f)
    col_ids = [it["col_id"] for it in labels]
    item_texts = {it["col_id"]: it.get("text", "") for it in labels}
    if loadings.shape[0] != len(col_ids):
        raise ValueError(
            f"[{run.label}] loadings rows {loadings.shape[0]} != labels len {len(col_ids)}"
        )
    return LoadedFa(
        label=run.label,
        loadings=loadings,
        col_ids=col_ids,
        item_texts=item_texts,
        communalities=comm,
        n_factors=loadings.shape[1],
        rotation=run.rotation,
    )


def align_on_items(a: LoadedFa, b: LoadedFa) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, str]]:
    """Return loading matrices restricted to shared col_ids, preserving order.

    Both input FA runs may have dropped different low-variance items, so we
    take the intersection. Factor order per side is preserved.
    """
    shared = sorted(set(a.col_ids) & set(b.col_ids))
    if not shared:
        raise ValueError("No shared items between the two FA runs.")
    a_idx = {c: i for i, c in enumerate(a.col_ids)}
    b_idx = {c: i for i, c in enumerate(b.col_ids)}
    la = a.loadings[[a_idx[c] for c in shared], :]
    lb = b.loadings[[b_idx[c] for c in shared], :]
    texts = {c: a.item_texts.get(c, b.item_texts.get(c, "")) for c in shared}
    return la, lb, shared, texts


# ═════════════════════════════════════════════════════════════════════════════
# CONGRUENCE
# ═════════════════════════════════════════════════════════════════════════════


def tucker_congruence_matrix(La: np.ndarray, Lb: np.ndarray) -> np.ndarray:
    """Signed φ(i,j) where i indexes A's factors, j B's factors.

    Each column of La / Lb is one factor's loading vector. Sign of each
    factor is arbitrary so upstream interpretation usually takes |φ|.
    """
    # Column-normalise so we can compute φ as a matrix product.
    nA = np.linalg.norm(La, axis=0, keepdims=True)
    nB = np.linalg.norm(Lb, axis=0, keepdims=True)
    nA[nA == 0] = 1.0
    nB[nB == 0] = 1.0
    return (La / nA).T @ (Lb / nB)  # (KA, KB)


def optimal_match(phi: np.ndarray) -> list[tuple[int, int, float]]:
    """Hungarian-optimal one-to-one assignment maximising Σ |φ|.

    Returns list of (a_idx, b_idx, signed_phi) for each matched pair.
    Unmatched factors on the larger side are *not* returned here — the
    caller should diff the indices against the range.
    """
    abs_phi = np.abs(phi)
    # linear_sum_assignment minimises cost → negate.
    row_ind, col_ind = linear_sum_assignment(-abs_phi)
    return [(int(r), int(c), float(phi[r, c])) for r, c in zip(row_ind, col_ind)]


def interpret(phi_abs: float) -> str:
    if phi_abs >= 0.95:
        return "equal"
    if phi_abs >= 0.85:
        return "fair"
    return "poor"


# ═════════════════════════════════════════════════════════════════════════════
# REPORTING
# ═════════════════════════════════════════════════════════════════════════════


def top_loading_items(
    loadings: np.ndarray,
    col_ids: list[str],
    item_texts: dict[str, str],
    factor_idx: int,
    k: int = 8,
    sign_flip: int = 1,
) -> list[tuple[str, float, str]]:
    """Top-k items by absolute loading on factor_idx; returns (col_id, signed_loading, text).

    ``sign_flip`` (±1) is applied to each loading so the output aligns with
    the sign convention of the matched partner factor.
    """
    col = loadings[:, factor_idx] * sign_flip
    order = np.argsort(-np.abs(col))[:k]
    return [
        (col_ids[i], float(col[i]), item_texts.get(col_ids[i], "")[:120])
        for i in order
    ]


def write_congruence_report(
    a: LoadedFa,
    b: LoadedFa,
    phi: np.ndarray,
    match: list[tuple[int, int, float]],
    col_ids: list[str],
    item_texts: dict[str, str],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full congruence matrix as CSV.
    df = pd.DataFrame(
        phi,
        index=[f"{a.label}_F{i+1}" for i in range(phi.shape[0])],
        columns=[f"{b.label}_F{j+1}" for j in range(phi.shape[1])],
    )
    df.to_csv(out_dir / "congruence_matrix.csv")

    # Match report with top-k items per pair, sign-aligned.
    lines = []
    lines.append(f"# Tucker's congruence: {a.label}  ↔  {b.label}")
    lines.append(f"  rotation={a.rotation}  shared_items={len(col_ids)}")
    lines.append("")
    lines.append("## Optimal one-to-one matches (|φ|, signed φ, interpretation)")
    lines.append("")
    matched_a, matched_b = set(), set()
    # Sort matches by |φ| desc.
    for r, c, ph in sorted(match, key=lambda x: -abs(x[2])):
        matched_a.add(r)
        matched_b.add(c)
        lines.append(
            f"  {a.label}_F{r+1}  ↔  {b.label}_F{c+1}   "
            f"|φ|={abs(ph):.3f}   φ={ph:+.3f}   [{interpret(abs(ph))}]"
        )
    # Unmatched factors (only occurs when dims differ).
    for i in range(phi.shape[0]):
        if i not in matched_a:
            j = int(np.argmax(np.abs(phi[i])))
            lines.append(
                f"  {a.label}_F{i+1}  (unmatched)  best partner {b.label}_F{j+1}  "
                f"|φ|={abs(phi[i, j]):.3f}"
            )
    for j in range(phi.shape[1]):
        if j not in matched_b:
            i = int(np.argmax(np.abs(phi[:, j])))
            lines.append(
                f"  {b.label}_F{j+1}  (unmatched)  best partner {a.label}_F{i+1}  "
                f"|φ|={abs(phi[i, j]):.3f}"
            )
    lines.append("")
    abs_phis = [abs(p) for _, _, p in match]
    lines.append(f"Mean |φ| over matched pairs: {np.mean(abs_phis):.3f}")
    lines.append(f"Min  |φ|: {np.min(abs_phis):.3f}    Max: {np.max(abs_phis):.3f}")
    lines.append(f"Pairs with |φ| ≥ 0.95 (equal): {sum(p >= 0.95 for p in abs_phis)}")
    lines.append(f"Pairs with |φ| ≥ 0.85 (fair): {sum(p >= 0.85 for p in abs_phis)}")
    lines.append("")

    lines.append("## Per-matched-pair top-loading items (sign-aligned to A)")
    for r, c, ph in sorted(match, key=lambda x: -abs(x[2])):
        lines.append("")
        sign_flip = -1 if ph < 0 else 1
        lines.append(
            f"### {a.label}_F{r+1}  ↔  {b.label}_F{c+1}   (|φ|={abs(ph):.3f}, φ={ph:+.3f})"
        )
        # Build aligned views on the SHARED columns only.
        # We pass the *aligned* loadings via La/Lb; here we need those:
        lines.append("")
        lines.append(f"  top items (shared) — {a.label}_F{r+1}:")
        for cid, val, text in top_loading_items(
            a._aligned_loadings, col_ids, item_texts, r, sign_flip=1
        ):
            lines.append(f"    {val:+.3f}  {cid}  {text}")
        lines.append(f"  top items (shared) — {b.label}_F{c+1} (sign-flipped to align):")
        for cid, val, text in top_loading_items(
            b._aligned_loadings, col_ids, item_texts, c, sign_flip=sign_flip
        ):
            lines.append(f"    {val:+.3f}  {cid}  {text}")

    (out_dir / "match_report.md").write_text("\n".join(lines))
    print(f"[Write] {out_dir / 'match_report.md'}")
    print(f"[Write] {out_dir / 'congruence_matrix.csv'}")


def plot_congruence_heatmap(
    a: LoadedFa,
    b: LoadedFa,
    phi: np.ndarray,
    match: list[tuple[int, int, float]],
    out_path: Path,
) -> None:
    abs_phi = np.abs(phi)
    KA, KB = phi.shape
    fig, ax = plt.subplots(figsize=(1.2 * KB + 2.5, 1.2 * KA + 2.2))
    im = ax.imshow(abs_phi, cmap="viridis", vmin=0, vmax=1, origin="upper")
    for i in range(KA):
        for j in range(KB):
            ax.text(j, i, f"{phi[i, j]:+.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs_phi[i, j] < 0.5 else "black")
    # Highlight matched cells with a red border.
    for r, c, _ in match:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                   edgecolor="red", lw=2.0))
    ax.set_xticks(range(KB), [f"F{j+1}" for j in range(KB)])
    ax.set_yticks(range(KA), [f"F{i+1}" for i in range(KA)])
    ax.set_xlabel(b.label)
    ax.set_ylabel(a.label)
    ax.set_title(
        f"Tucker's congruence |φ|  ({a.rotation})\n"
        f"red = optimal one-to-one match; numbers = signed φ"
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label="|φ|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Wrote {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# DRIVER
# ═════════════════════════════════════════════════════════════════════════════


def compare_pair(run_a: FaRun, run_b: FaRun, out_dir: Path) -> None:
    print(f"\n{'='*70}\n{run_a.label}  vs  {run_b.label}  ({run_a.rotation})\n{'='*70}")
    a = load_fa_run(run_a)
    b = load_fa_run(run_b)
    if a.rotation != b.rotation:
        raise ValueError(f"Rotation mismatch: {a.rotation} vs {b.rotation}")

    La, Lb, shared, texts = align_on_items(a, b)
    print(f"  shared items: {len(shared)} "
          f"(A had {len(a.col_ids)}, B had {len(b.col_ids)})")
    # Stash aligned loadings on the dataclass for the reporter to use.
    a._aligned_loadings = La
    b._aligned_loadings = Lb

    phi = tucker_congruence_matrix(La, Lb)
    match = optimal_match(phi)

    pair_tag = f"{run_a.label}_vs_{run_b.label}_{run_a.rotation}".replace("/", "_")
    pair_dir = out_dir / pair_tag
    write_congruence_report(a, b, phi, match, shared, texts, pair_dir)
    plot_congruence_heatmap(a, b, phi, match, pair_dir / "congruence_heatmap.png")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    out_dir = OUTPUT_ROOT / TAG
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(RUNS) < 2:
        print("Need ≥2 entries in RUNS to run a comparison. Fill in the second entry "
              "once its FA artifacts exist on disk.")
        return

    for i, ra in enumerate(RUNS):
        for rb in RUNS[i + 1:]:
            compare_pair(ra, rb, out_dir)

    print(f"\nAll outputs → {out_dir}")


if __name__ == "__main__":
    main()
