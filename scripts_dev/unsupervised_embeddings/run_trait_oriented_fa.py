"""Run trait-oriented factor analysis on a completed stage-2 questionnaire run.

Builds a trait-oriented response matrix (using ``answer_mapping`` so each cell
is a high-trait score in [0, 1] — continuous when ``probs`` is available from
logprob mode, else binary from the parsed categorical choice). Runs Horn's
parallel analysis + PAF on the matrix, saves the FA result, and produces the
factor-trait alignment analysis (top-20 counts per trait + mean-loading
heatmaps, signed and unsigned).

This is a post-hoc analysis: it operates entirely on the existing
``raw_responses.jsonl`` + questionnaire JSON produced by Stage 2 — no LLM
inference is re-run.

Usage:
    uv run python scripts_dev/unsupervised_embeddings/run_trait_oriented_fa.py \\
        --questionnaire-dir scratch/psychometric_fa/<run>/questionnaire \\
        --questionnaire-json datasets/psychometric_questionnaires/trait_ocean_v1.json \\
        [--n-factors N] [--method principal] [--rotation varimax] [--top-k 20]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

from src_dev.factor_analysis.factor_analysis import adequacy_tests, run_factor_analysis
from src_dev.factor_analysis.parallel_analysis import parallel_analysis
from src_dev.factor_analysis.persistence import save_factor_analysis
from src_dev.factor_analysis.trait_alignment import (
    build_trait_oriented_matrix,
    compute_factor_trait_alignment,
    plot_all_alignment,
    save_alignment,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questionnaire-dir",
        type=Path,
        required=True,
        help="Directory containing raw_responses.jsonl (stage-2 output).",
    )
    parser.add_argument(
        "--questionnaire-json",
        type=Path,
        required=True,
        help="Path to the questionnaire JSON (for answer_mapping + primary_dimension).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write FA + alignment outputs. Default: <questionnaire-dir>/factor_analysis_trait_oriented",
    )
    parser.add_argument(
        "--n-factors",
        type=int,
        default=None,
        help="Number of factors to extract. If omitted, uses Horn's parallel analysis (permutation reference).",
    )
    parser.add_argument("--method", type=str, default="principal",
                        help="FA extraction method (principal/minres/ml).")
    parser.add_argument("--rotation", type=str, default="varimax",
                        help="FA rotation (varimax/promax/oblimin/None). Use 'none' to disable.")
    parser.add_argument("--top-k", type=int, default=20,
                        help="K for top-K loading analysis (default 20 = one trait block).")
    parser.add_argument("--max-missing-frac", type=float, default=0.2,
                        help="Drop personas with > this fraction of missing items.")
    parser.add_argument("--title-prefix", type=str, default=None,
                        help="Optional title prefix for plots.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    q_dir: Path = args.questionnaire_dir
    raw_path = q_dir / "raw_responses.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"raw_responses.jsonl not found in {q_dir}")

    output_dir = args.output_dir or (q_dir / "factor_analysis_trait_oriented")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[trait-FA] Building trait-oriented matrix from {raw_path}")
    tom = build_trait_oriented_matrix(
        raw_responses_path=raw_path,
        questionnaire_path=args.questionnaire_json,
    )
    matrix = tom.matrix  # [K × N_items]
    K, N = matrix.shape
    print(f"[trait-FA] Raw matrix: {K} personas × {N} items  traits={tom.trait_order}")

    # Drop rows with excessive missing.
    missing_frac = np.mean(np.isnan(matrix), axis=1)
    keep_rows = missing_frac <= args.max_missing_frac
    data = matrix[keep_rows]
    k_kept = [tom.k_index[i] for i in np.where(keep_rows)[0]]
    print(
        f"[trait-FA] Kept {int(keep_rows.sum())}/{K} personas (≤{args.max_missing_frac:.0%} missing)"
    )

    # Column-mean impute remaining NaNs (needed for FA; rows were already filtered).
    col_means = np.nanmean(data, axis=0)
    nan_mask = np.isnan(data)
    if nan_mask.any():
        data = data.copy()
        inds = np.where(nan_mask)
        data[inds] = np.take(col_means, inds[1])
        print(f"[trait-FA] Mean-imputed {int(nan_mask.sum())} remaining missing cells")

    # Drop zero-variance columns (can happen if every persona gave the same answer).
    col_var = np.var(data, axis=0)
    live_cols = col_var > 1e-10
    if not live_cols.all():
        dropped = [(tom.item_ids[i], tom.item_dims[i]) for i in np.where(~live_cols)[0]]
        print(f"[trait-FA] Dropping {int((~live_cols).sum())} zero-variance items: {dropped}")
    data = data[:, live_cols]
    item_ids = [tom.item_ids[i] for i in np.where(live_cols)[0]]
    item_dims = [tom.item_dims[i] for i in np.where(live_cols)[0]]

    # Adequacy.
    print("[trait-FA] Adequacy tests (standardized data):")
    data_z = (data - data.mean(axis=0)) / data.std(axis=0, ddof=0)
    adequacy = adequacy_tests(data_z)

    # Parallel analysis (permutation reference is best for non-Gaussian data).
    if args.n_factors is None:
        print("[trait-FA] Running parallel analysis to choose n_factors...")
        pa = parallel_analysis(
            data_z, n_iterations=100, percentile=95.0, random_state=SEED,
            method="permutation",
        )
        n_factors = int(pa["n_recommended"])
        print(f"[trait-FA] Parallel analysis recommends {n_factors} factors")
    else:
        n_factors = int(args.n_factors)
        pa = None

    rotation = None if str(args.rotation).lower() in ("none", "null", "") else args.rotation
    fa = run_factor_analysis(data_z, n_factors=n_factors, method=args.method, rotation=rotation)
    fa_path = save_factor_analysis(
        fa,
        output_dir / f"fa_trait_oriented_n{n_factors}_{args.method}_{rotation or 'unrotated'}",
        config={
            "n_factors": n_factors,
            "method": args.method,
            "rotation": rotation,
            "encoding": "trait_oriented",
            "n_personas_used": int(data.shape[0]),
            "n_items_used": int(data.shape[1]),
            "adequacy": {"kmo_overall": adequacy["kmo_overall"],
                         "bartlett_p": adequacy["bartlett_p"]},
        },
    )
    # Record which items/personas the loadings are aligned to.
    with open(output_dir / "fa_trait_oriented_items.json", "w", encoding="utf-8") as f:
        json.dump({
            "item_ids": item_ids,
            "item_dims": item_dims,
            "k_index": k_kept,
            "trait_order": tom.trait_order,
            "fa_path": str(fa_path),
        }, f, indent=2)

    # Alignment analysis.
    alignment = compute_factor_trait_alignment(
        loadings=fa["loadings"],
        item_dims=item_dims,
        trait_order=tom.trait_order,
        top_k=args.top_k,
    )
    save_alignment(alignment, output_dir)
    plot_paths = plot_all_alignment(
        alignment,
        output_dir,
        title_prefix=args.title_prefix or q_dir.parent.name,
    )
    for name, path in plot_paths.items():
        print(f"[trait-FA] {name}: {path}")

    print("[trait-FA] Factor winners (top-K by |loading|):")
    for f, label in enumerate(alignment.factor_labels):
        counts = alignment.top_k_count[f]
        best = int(np.argmax(counts))
        print(
            f"  {label}: {alignment.trait_order[best]} "
            f"({int(counts[best])}/{alignment.top_k}) — "
            f"full: {dict(zip(alignment.trait_order, [int(c) for c in counts]))}"
        )


if __name__ == "__main__":
    main()
