"""k-sweep of cross-model Tucker's congruence.

Fits PAF factor analysis on the cached combined response matrices for two
models at a range of k values, then computes Tucker's phi between the
per-k loading matrices. Answers the question: "is there any k at which
Llama and Qwen2.5 produce similar factor structures?"

Avoids going through the full pipeline (Stages 1/2 already cached; skips
Stage 5 validation). Applies the same ``preprocess_response_matrix`` step
the main pipeline uses so item-filter behaviour matches existing FA
artifacts.

Outputs a CSV summary (one row per k × rotation) + a line plot of mean |phi|
vs k.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

from src_dev.factor_analysis.factor_analysis import run_factor_analysis
from src_dev.psychometric.preprocessing import preprocess_response_matrix

logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("scratch/psychometric_comparison")


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ModelInputs:
    label: str
    combined_dir: Path    # Must contain questionnaire/response_matrix.npy, metadata.jsonl, items.json


RUNS: list[ModelInputs] = [
    ModelInputs(
        label="llama-3.1-8b",
        combined_dir=Path("scratch/psychometric_fa/combined-R[B]-Q[v5+trait_ocean_v1]"),
    ),
    ModelInputs(
        label="qwen2.5-7b",
        combined_dir=Path("scratch/psychometric_fa/combined-R[B]-Q[v5+trait_ocean_v1]-qm_qwen257binstruct"),
    ),
]

K_VALUES: list[int] = [2, 3, 4, 5, 7, 9, 11, 13, 17, 21]
ROTATIONS: list[str] = ["oblimin", "varimax"]
METHOD: str = "principal"
MIN_ITEM_VARIANCE: float = 0.1

TAG = "B_cross_model_k_sweep"


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def load_combined_matrix(run: ModelInputs):
    q = run.combined_dir / "questionnaire"
    matrix = np.load(q / "response_matrix.npy")
    with open(q / "metadata.jsonl") as f:
        metadata = [json.loads(l) for l in f if l.strip()]
    with open(q / "items.json") as f:
        items = json.load(f)
    print(f"[Load] {run.label}: matrix={matrix.shape}, items={len(items)}")
    return matrix, metadata, items


def preprocess(matrix, metadata, items):
    data, _meta_f, cols_f, _gids = preprocess_response_matrix(
        matrix, metadata, items,
        min_item_variance=MIN_ITEM_VARIANCE,
        high_variance_persona_drop_pct=0.0,
        do_residualize=False,
    )
    col_ids = [c.get("col_id", c.get("item_id", f"col_{i}")) for i, c in enumerate(cols_f)]
    return data, col_ids


def tucker(La: np.ndarray, Lb: np.ndarray) -> np.ndarray:
    nA = np.linalg.norm(La, axis=0, keepdims=True); nA[nA == 0] = 1.0
    nB = np.linalg.norm(Lb, axis=0, keepdims=True); nB[nB == 0] = 1.0
    return (La / nA).T @ (Lb / nB)


def optimal_match_abs(phi: np.ndarray):
    r, c = linear_sum_assignment(-np.abs(phi))
    return [(int(i), int(j), float(phi[i, j])) for i, j in zip(r, c)]


# ═════════════════════════════════════════════════════════════════════════════
# DRIVER
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    out_dir = OUTPUT_ROOT / TAG
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load + preprocess each model once; cache (data, col_ids).
    preprocessed: dict[str, tuple[np.ndarray, list[str]]] = {}
    for run in RUNS:
        matrix, meta, items = load_combined_matrix(run)
        print(f"[Preprocess] {run.label}:")
        data, col_ids = preprocess(matrix, meta, items)
        preprocessed[run.label] = (data, col_ids)
        print(f"  → surviving columns: {len(col_ids)}")

    # Fit FA for every (model, k, rotation). Stash loadings + col_ids.
    # loadings_by[(label, k, rotation)] = (loadings: n_cols × k, col_ids)
    loadings_by: dict[tuple[str, int, str], tuple[np.ndarray, list[str]]] = {}
    for label, (data, col_ids) in preprocessed.items():
        for k in K_VALUES:
            if k >= data.shape[1]:
                print(f"[Skip] {label} k={k}: only {data.shape[1]} columns")
                continue
            for rot in ROTATIONS:
                try:
                    fa = run_factor_analysis(data, n_factors=k, method=METHOD, rotation=rot)
                    loadings_by[(label, k, rot)] = (fa["loadings"], col_ids)
                except Exception as exc:
                    print(f"[FA fail] {label} k={k} {rot}: {exc}")

    # Compute pairwise Tucker across models at each (k, rotation).
    rows = []
    pairs = list(itertools.combinations([r.label for r in RUNS], 2))
    for (la, lb) in pairs:
        for k in K_VALUES:
            for rot in ROTATIONS:
                if (la, k, rot) not in loadings_by or (lb, k, rot) not in loadings_by:
                    continue
                La, a_cols = loadings_by[(la, k, rot)]
                Lb, b_cols = loadings_by[(lb, k, rot)]
                # Align by shared col_ids.
                shared = sorted(set(a_cols) & set(b_cols))
                a_idx = {c: i for i, c in enumerate(a_cols)}
                b_idx = {c: i for i, c in enumerate(b_cols)}
                LA = La[[a_idx[c] for c in shared], :]
                LB = Lb[[b_idx[c] for c in shared], :]
                phi = tucker(LA, LB)
                match = optimal_match_abs(phi)
                abs_phis = [abs(p) for _, _, p in match]
                rows.append({
                    "model_a": la, "model_b": lb,
                    "k": k, "rotation": rot,
                    "n_shared_items": len(shared),
                    "mean_abs_phi": float(np.mean(abs_phis)),
                    "median_abs_phi": float(np.median(abs_phis)),
                    "max_abs_phi": float(np.max(abs_phis)),
                    "min_abs_phi": float(np.min(abs_phis)),
                    "n_fair_0.85": int(sum(p >= 0.85 for p in abs_phis)),
                    "n_equal_0.95": int(sum(p >= 0.95 for p in abs_phis)),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "k_sweep_summary.csv", index=False)
    print(f"[Write] {out_dir / 'k_sweep_summary.csv'}")
    print()
    print(df.to_string(index=False))

    # Plot: mean / max |phi| vs k, one line per rotation per model pair.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, ylabel, col in [
        (axes[0], "Mean |φ| across matched pairs", "mean_abs_phi"),
        (axes[1], "Max |φ| (best matched pair)",   "max_abs_phi"),
    ]:
        for (la, lb) in pairs:
            for rot in ROTATIONS:
                sub = df[(df["model_a"] == la) & (df["model_b"] == lb) & (df["rotation"] == rot)].sort_values("k")
                if sub.empty: continue
                ax.plot(sub["k"], sub[col], marker="o", label=f"{la}↔{lb} · {rot}")
        ax.axhline(0.95, color="green", lw=0.6, ls=":", label="equal (0.95)")
        ax.axhline(0.85, color="orange", lw=0.6, ls=":", label="fair (0.85)")
        ax.set_xlabel("k (number of factors)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Cross-model Tucker's congruence vs k   "
        f"(pairs={len(pairs)}, rotations={ROTATIONS}, preprocessing: min_item_variance={MIN_ITEM_VARIANCE})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "k_sweep_plot.png", dpi=140)
    plt.close(fig)
    print(f"[Plot] {out_dir / 'k_sweep_plot.png'}")
    print(f"\nAll outputs → {out_dir}")


if __name__ == "__main__":
    main()
