"""Cross-model Tucker's φ sweep over {encoding × block × rotation}.

Takes the four combined-FA artifact sets produced by
``scripts_dev/unsupervised_embeddings/psychometric_rollout_fa.py`` (two
models × two trait_mcq encodings) and for every combination of:

    encoding ∈ {soft_ev, logit}
    block    ∈ {combined (v5 + trait_mcq), likert-only, trait_mcq-only}
    rotation ∈ {oblimin, varimax}

computes signed Tucker's congruence between Qwen and Llama factor
loadings (after Hungarian-matched sign alignment on shared items). Emits

- ``summary.csv``               — one row per (encoding, block, rotation)
                                   with n_shared, mean/median/min |φ|,
                                   and per-matched-pair |φ|.
- per-comparison subdirs         — full-matrix φ, congruence table, and
                                   heatmap image.

Relative to ``scripts_dev/psychometric_assessment/cross_model_factor_congruence.py``,
which compares one pair of runs at a time, this driver iterates the
entire {encoding × block × rotation} grid in a single pass so the
soft_ev/logit and Likert/trait_mcq methodology decisions can be read
side-by-side.

Usage: edit the config block below (paths + rotations) and run:

    uv run python -m scripts_dev.psychometric_assessment.cross_model_congruence_sweep
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src_dev.factor_analysis.congruence import compare_solutions, tucker_phi

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

ROOT = Path("scratch/psychometric_fa")

# Combined-FA artifact dirs, one per (model × encoding).
COMBINED_DIRS: dict[tuple[str, str], Path] = {
    ("qwen",  "soft_ev"): ROOT / "combined-R[B]-Q[v5+trait_ocean_natural_v1]-qm_qwen257binstruct",
    ("llama", "soft_ev"): ROOT / "combined-R[B]-Q[v5+trait_ocean_natural_v1]",
    ("qwen",  "logit"):   ROOT / "combined-R[B]-Q[v5+trait_ocean_natural_v1]-enc_logit-qm_qwen257binstruct",
    ("llama", "logit"):   ROOT / "combined-R[B]-Q[v5+trait_ocean_natural_v1]-enc_logit",
}

# FA settings — these have to match what Stage 3 wrote. See
# ``scripts_dev/unsupervised_embeddings/psychometric_rollout_fa.py``:
# FA_METHOD, FA_N_FACTORS_OVERRIDE, FA_ROTATIONS.
N_FACTORS: int = 4
METHOD: str = "principal"
ROTATIONS: tuple[str, ...] = ("oblimin", "varimax")

# Each block is a subpath under ``{combined_dir}/factor_analysis/`` where
# the corresponding FA npz lives. "combined" = both blocks in one FA,
# the other two are the per-block single-block passes.
BLOCKS: dict[str, str] = {
    "combined":  "raw",
    "likert":    "per_block/likert/raw",
    "trait_mcq": "per_block/trait_mcq/raw",
}

OUTPUT_ROOT = Path("scratch/psychometric_comparison/cross_model_sweep")


# ═════════════════════════════════════════════════════════════════════════════
# I/O
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class LoadedFa:
    model: str
    encoding: str
    block: str
    rotation: str
    loadings: np.ndarray        # (n_items, n_factors)
    col_ids: list[str]          # length n_items
    item_texts: dict[str, str]


def _fa_npz_path(combined_dir: Path, block_subpath: str, rotation: str) -> Path:
    return combined_dir / "factor_analysis" / block_subpath / (
        f"fa_{N_FACTORS}_{METHOD}_{rotation}.npz"
    )


def _fa_labels_path(combined_dir: Path, block_subpath: str, rotation: str) -> Path:
    return combined_dir / "factor_analysis" / block_subpath / (
        f"fa_{N_FACTORS}_{METHOD}_{rotation}_item_labels.json"
    )


def load_fa(
    model: str, encoding: str, block: str, rotation: str,
) -> LoadedFa | None:
    """Load one (model, encoding, block, rotation) FA artifact set.

    Returns None (with a warning) if the npz isn't on disk — the sweep
    continues on missing cells so partial runs still produce useful
    output.
    """
    combined_dir = COMBINED_DIRS[(model, encoding)]
    subpath = BLOCKS[block]
    npz_path = _fa_npz_path(combined_dir, subpath, rotation)
    labels_path = _fa_labels_path(combined_dir, subpath, rotation)
    if not npz_path.exists():
        logger.warning("Missing FA artifact: %s", npz_path)
        return None
    with np.load(npz_path) as z:
        loadings = z["loadings"]
    with labels_path.open("r", encoding="utf-8") as f:
        item_labels = json.load(f)
    col_ids = [it["col_id"] for it in item_labels]
    item_texts = {
        it["col_id"]: it.get("text") or it.get("item_text") or ""
        for it in item_labels
    }
    if loadings.shape[0] != len(col_ids):
        raise ValueError(
            f"[{model}/{encoding}/{block}/{rotation}] loadings rows "
            f"{loadings.shape[0]} != item_labels length {len(col_ids)}"
        )
    return LoadedFa(
        model=model, encoding=encoding, block=block, rotation=rotation,
        loadings=loadings, col_ids=col_ids, item_texts=item_texts,
    )


def align_on_shared_items(a: LoadedFa, b: LoadedFa) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Restrict both loading matrices to the shared col_id set, preserving order.

    MIN_ITEM_VARIANCE can drop different items across the two models, so
    the intersection is typically a strict subset of each side's items.
    """
    a_set = set(a.col_ids)
    b_set = set(b.col_ids)
    shared = sorted(a_set & b_set)
    if not shared:
        raise ValueError(
            f"No shared col_ids between {a.model} and {b.model} "
            f"for block={a.block}, rotation={a.rotation}. "
            f"(a={len(a_set)}, b={len(b_set)})"
        )
    a_idx = {c: i for i, c in enumerate(a.col_ids)}
    b_idx = {c: i for i, c in enumerate(b.col_ids)}
    La = a.loadings[[a_idx[c] for c in shared], :]
    Lb = b.loadings[[b_idx[c] for c in shared], :]
    return La, Lb, shared


# ═════════════════════════════════════════════════════════════════════════════
# REPORTING
# ═════════════════════════════════════════════════════════════════════════════


def _interpret(phi_abs: float) -> str:
    """Lorenzo-Seva & ten Berge (2006) qualitative cutoffs."""
    if phi_abs >= 0.95: return "equal"
    if phi_abs >= 0.85: return "fair"
    return "poor"


def write_pair_report(
    a: LoadedFa, b: LoadedFa, comparison, shared: list[str], out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Full signed φ matrix (pre-matching)
    np.save(out_dir / "phi_matrix.npy", comparison.full_phi_matrix)
    # Matched pairs + |φ| per pair
    report = {
        "model_a": a.model,
        "model_b": b.model,
        "encoding": a.encoding,
        "block": a.block,
        "rotation": a.rotation,
        "n_shared_items": len(shared),
        "n_items_a": len(a.col_ids),
        "n_items_b": len(b.col_ids),
        "align_method": comparison.align_method,
        "matched_pairs": [
            {
                "factor_a": int(comparison.matched_a_indices[i]),
                "factor_b": int(comparison.matched_b_indices[i]),
                "phi_signed": float(comparison.phi_matched[i]),
                "phi_abs": float(abs(comparison.phi_matched[i])),
                "interpretation": _interpret(float(abs(comparison.phi_matched[i]))),
                "sign_flip": float(comparison.sign_flips[i]),
            }
            for i in range(comparison.n_matched)
        ],
        "mean_phi_abs": comparison.mean_phi,
        "median_phi_abs": comparison.median_phi,
        "min_phi_abs": comparison.min_phi,
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Short plaintext pretty-print
    lines = []
    lines.append(f"{a.model} vs {b.model}  |  encoding={a.encoding}  block={a.block}  rotation={a.rotation}")
    lines.append(f"  shared items: {len(shared)} / a={len(a.col_ids)} / b={len(b.col_ids)}")
    lines.append(f"  align: {comparison.align_method}")
    lines.append("  matched pairs (|φ|, interp):")
    for i in range(comparison.n_matched):
        phi = float(comparison.phi_matched[i])
        lines.append(
            f"    F{int(comparison.matched_a_indices[i])} ↔ F{int(comparison.matched_b_indices[i])}: "
            f"|φ|={abs(phi):.3f} ({_interpret(abs(phi))})  sign_flip={int(comparison.sign_flips[i]):+d}"
        )
    lines.append(f"  mean |φ| = {comparison.mean_phi:.3f}   median = {comparison.median_phi:.3f}   min = {comparison.min_phi:.3f}")
    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text + "\n")
    print(text)


def plot_phi_heatmap(comparison, a: LoadedFa, b: LoadedFa, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping heatmap for %s", out_path)
        return
    phi = comparison.full_phi_matrix
    k_a, k_b = phi.shape
    fig, ax = plt.subplots(figsize=(max(4, 0.8 * k_b + 2), max(3, 0.8 * k_a + 1.5)))
    im = ax.imshow(phi, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    # Annotate with signed values
    for i in range(k_a):
        for j in range(k_b):
            v = phi[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color=("white" if abs(v) > 0.6 else "black"), fontsize=8)
    # Highlight matched cells
    for i in range(comparison.n_matched):
        ra = int(comparison.matched_a_indices[i])
        cb = int(comparison.matched_b_indices[i])
        rect = plt.Rectangle((cb - 0.5, ra - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2)
        ax.add_patch(rect)
    ax.set_xticks(range(k_b))
    ax.set_yticks(range(k_a))
    ax.set_xticklabels([f"{b.model[:4]} F{j}" for j in range(k_b)], rotation=45, ha="right")
    ax.set_yticklabels([f"{a.model[:4]} F{i}" for i in range(k_a)])
    ax.set_title(f"Tucker's φ — {a.encoding} / {a.block} / {a.rotation}\n"
                 f"mean |φ|={comparison.mean_phi:.3f}  min={comparison.min_phi:.3f}")
    plt.colorbar(im, ax=ax, label="signed φ")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# DRIVER
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for encoding in ("soft_ev", "logit"):
        for block in BLOCKS:
            for rotation in ROTATIONS:
                tag = f"{encoding}_{block}_{rotation}"
                print(f"\n{'─' * 72}\n[{tag}]\n{'─' * 72}")

                a = load_fa("qwen",  encoding, block, rotation)
                b = load_fa("llama", encoding, block, rotation)
                if a is None or b is None:
                    print(f"  Skipping {tag} — missing FA artifact(s).")
                    summary_rows.append({
                        "encoding": encoding, "block": block, "rotation": rotation,
                        "status": "MISSING",
                        "n_shared": None, "mean_phi": None, "median_phi": None, "min_phi": None,
                        "phi_matched": None,
                    })
                    continue

                La, Lb, shared = align_on_shared_items(a, b)
                if La.shape[1] != Lb.shape[1]:
                    align_method = "hungarian"
                else:
                    align_method = "procrustes"
                # Use aligned-on-items loadings for congruence
                a_aligned = LoadedFa(**{**a.__dict__, "loadings": La, "col_ids": shared})
                b_aligned = LoadedFa(**{**b.__dict__, "loadings": Lb, "col_ids": shared})
                comparison = compare_solutions(La, Lb, align=align_method)

                pair_dir = OUTPUT_ROOT / tag
                write_pair_report(a_aligned, b_aligned, comparison, shared, pair_dir)
                plot_phi_heatmap(comparison, a_aligned, b_aligned, pair_dir / "phi_heatmap.png")

                summary_rows.append({
                    "encoding": encoding,
                    "block": block,
                    "rotation": rotation,
                    "status": "OK",
                    "n_shared": len(shared),
                    "mean_phi": round(comparison.mean_phi, 4),
                    "median_phi": round(comparison.median_phi, 4),
                    "min_phi": round(comparison.min_phi, 4),
                    "phi_matched": [
                        round(abs(float(x)), 4) for x in comparison.phi_matched
                    ],
                })

    # Summary CSV
    summary_path = OUTPUT_ROOT / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "encoding", "block", "rotation", "status", "n_shared",
                "mean_phi", "median_phi", "min_phi", "phi_matched",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            row = dict(row)
            if row.get("phi_matched") is not None:
                row["phi_matched"] = "[" + ", ".join(f"{x}" for x in row["phi_matched"]) + "]"
            writer.writerow(row)

    # Pretty-print top-level table
    print(f"\n{'=' * 72}\nSummary  (Qwen ↔ Llama, mean |φ| over matched factors)\n{'=' * 72}")
    print(f"{'encoding':10s}  {'block':10s}  {'rotation':10s}  {'n_shared':>8s}  {'mean|φ|':>7s}  {'min|φ|':>7s}  {'matched':s}")
    for r in summary_rows:
        if r["status"] != "OK":
            print(f"{r['encoding']:10s}  {r['block']:10s}  {r['rotation']:10s}  {'—':>8s}  {'—':>7s}  {'—':>7s}  MISSING")
            continue
        matched_str = "[" + ", ".join(f"{x:.2f}" for x in r["phi_matched"]) + "]"
        print(f"{r['encoding']:10s}  {r['block']:10s}  {r['rotation']:10s}  "
              f"{r['n_shared']:>8d}  {r['mean_phi']:>7.3f}  {r['min_phi']:>7.3f}  {matched_str}")
    print(f"\nSummary CSV: {summary_path}")
    print(f"Per-comparison details: {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
