"""Standalone 2D factor biplot for the paper's unsupervised section.

Plots each baseline-model persona-rollout's Thomson factor scores on the
top-two-variance plane (F0=Initiative on x, F1=Tone on y) from the v7-pf3
$k=4$ oblimin fit on Llama-3.1-8B-Instruct, with the highest-|loading|
items overlaid as labelled arrows projecting onto the same plane.

This is an iteration sandbox — kept deliberately self-contained so we
can change layout / styling fast without touching the main pipeline.
The fit it reads is produced by

    scripts_dev/unsupervised_embeddings/analysis_for_paper.v2.py --k 4

so re-run that first if scratch/psychometric_fa_paper_v7pf3_k4/ is empty.

Run with:
    uv run python scripts_dev/unsupervised_embeddings/factor_biplot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Config (edit me) ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIT_NPZ = PROJECT_ROOT / (
    "scratch/psychometric_fa_paper_v7pf3_k4/llama-3.1-8b/"
    "factor_analysis/raw/fa_4_principal_oblimin.npz"
)
ITEM_LABELS_JSON = PROJECT_ROOT / (
    "scratch/psychometric_fa_paper_v7pf3_k4/llama-3.1-8b/"
    "factor_analysis/raw/fa_4_principal_oblimin_item_labels.json"
)

OUT_DIR = PROJECT_ROOT / "scratch/factor_biplot"
OUT_PATH = OUT_DIR / "biplot_initiative_tone.png"

# Also write the figure to the paper's figures dir so the LaTeX can pick
# it up. Paper-figure output convention: paper/figures/unsupervised/.
PAPER_FIGURES = [
    "unsupervised/fig_4_2_7_biplot_initiative_tone.pdf",
]
try:
    from src_dev.visualisations import PAPER_FIGURES_DIR
    PAPER_OUT_PATH = PAPER_FIGURES_DIR / PAPER_FIGURES[0]
except Exception:
    PAPER_OUT_PATH = None

# Which two factors to plot (column indices in the npz, in
# canonical-variance-sorted order: 0=Initiative, 1=Tone, 2=Didacticism,
# 3=Epistemic Caution).
FX, FY = 0, 1
FACTOR_NAMES = ["Initiative", "Tone", "Didacticism", "Epistemic Caution"]

# How many highest-|loading| items to show as arrows (per factor we plot,
# union'd). Keep small to avoid clutter.
N_TOP_ITEMS_PER_FACTOR = 0
# Arrow scaling: arrow length = ARROW_SCALE × loading on that axis.
ARROW_SCALE = 3.0

# ── Exemplar dumping ────────────────────────────────────────────────────────
# After plotting, also dump the most-extreme baseline rollouts on each
# single-axis pole (e.g. high F0 with F1≈0) into per-axis text files.
# This is the data upstream agents use to inspect what behaviour the
# axes actually express. Set DUMP_EXEMPLARS=False to skip.
DUMP_EXEMPLARS: bool = True
# Directory holding the rollout dataset that produced the questionnaire
# scores. canonical_samples.jsonl in here keys conversations by sample_id.
ROLLOUT_DIR = PROJECT_ROOT / (
    "scratch/psychometric_fa.pf3-k4/"
    "rollouts-llama318binstruct-t1.0-15t-2500p-seed436-"
    "scenarios_v2-uprompt_v6"
)
QUESTIONNAIRE_METADATA = PROJECT_ROOT / (
    "scratch/psychometric_fa_paper_v7pf3_k4/llama-3.1-8b/"
    "questionnaire/metadata.jsonl"
)
N_EXEMPLARS_PER_AXIS = 5
# Require the *other* axis (the one we don't want extreme on) to lie
# within ±OTHER_AXIS_BAND. Smaller = stricter "single-axis" purity.
OTHER_AXIS_BAND = 0.25


def _dump_exemplars(scores) -> None:
    """Write 4 per-axis-pole files of the top-N most-extreme rollouts.

    For each (axis, sign) combination, picks rollouts that are extreme on
    the target axis but neutral on the orthogonal axis, then writes the
    final-turn user→assistant text from canonical_samples.jsonl.
    """
    if not QUESTIONNAIRE_METADATA.exists():
        print(f"  ⚠ skipping exemplar dump: missing {QUESTIONNAIRE_METADATA}")
        return
    canon_path = ROLLOUT_DIR / "datasets" / "canonical_samples.jsonl"
    if not canon_path.exists():
        print(f"  ⚠ skipping exemplar dump: missing {canon_path}")
        return

    sample_ids = [json.loads(l)["sample_id"]
                  for l in QUESTIONNAIRE_METADATA.open()]
    if len(sample_ids) != scores.shape[0]:
        raise RuntimeError(
            f"sample_ids ({len(sample_ids)}) and scores rows ({scores.shape[0]}) disagree"
        )
    canon: dict[str, dict] = {}
    with canon_path.open() as f:
        for line in f:
            o = json.loads(line)
            canon[o["sample_id"]] = o

    fx_name = FACTOR_NAMES[FX]
    fy_name = FACTOR_NAMES[FY]
    target_x = scores[:, FX]
    other_y = scores[:, FY]
    target_y = scores[:, FY]
    other_x = scores[:, FX]

    groups = [
        (
            f"F{FX}_pos_{fx_name.lower().replace(' ', '_')}_high",
            f"{fx_name} HIGH (F{FX}+), {fy_name} neutral (|F{FY}|<{OTHER_AXIS_BAND})",
            target_x, other_y, +1,
        ),
        (
            f"F{FX}_neg_{fx_name.lower().replace(' ', '_')}_low",
            f"{fx_name} LOW  (F{FX}-), {fy_name} neutral (|F{FY}|<{OTHER_AXIS_BAND})",
            target_x, other_y, -1,
        ),
        (
            f"F{FY}_pos_{fy_name.lower().replace(' ', '_')}_high",
            f"{fy_name} HIGH (F{FY}+), {fx_name} neutral (|F{FX}|<{OTHER_AXIS_BAND})",
            target_y, other_x, +1,
        ),
        (
            f"F{FY}_neg_{fy_name.lower().replace(' ', '_')}_low",
            f"{fy_name} LOW  (F{FY}-), {fx_name} neutral (|F{FX}|<{OTHER_AXIS_BAND})",
            target_y, other_x, -1,
        ),
    ]

    for slug, header, target, other, sign in groups:
        mask = (np.abs(other) < OTHER_AXIS_BAND) & ((target > 0) if sign > 0 else (target < 0))
        if not mask.any():
            print(f"  ⚠ no exemplars matched for {slug}")
            continue
        idxs = np.where(mask)[0]
        order = idxs[np.argsort(-np.abs(target[idxs]))]
        picks = order[:N_EXEMPLARS_PER_AXIS]

        lines = [
            f"### {header}",
            f"(top {N_EXEMPLARS_PER_AXIS} by |target axis|, requiring |other axis| < {OTHER_AXIS_BAND})",
            "",
        ]
        for rank, idx in enumerate(picks, 1):
            sid = sample_ids[idx]
            sample = canon.get(sid)
            lines.append("=" * 100)
            lines.append(f"### Example {rank}/{len(picks)}")
            lines.append(f"idx={idx}  sample_id={sid}")
            lines.append(
                f"F{FX}({fx_name})={scores[idx, FX]:+.3f}   "
                f"F{FY}({fy_name})={scores[idx, FY]:+.3f}"
            )
            lines.append("=" * 100)
            if not sample:
                lines.append("(canonical_samples.jsonl missing this sample_id)")
                lines.append("")
                continue
            msgs = sample.get("messages", [])
            assistant_idxs = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
            if not assistant_idxs:
                lines.append("(no assistant turns in this rollout)")
                lines.append("")
                continue
            ai = assistant_idxs[-1]
            if ai > 0 and msgs[ai - 1].get("role") == "user":
                lines.append("\n--- USER (final turn) ---\n")
                lines.append(msgs[ai - 1]["content"])
            lines.append("\n--- ASSISTANT (final turn) ---\n")
            lines.append(msgs[ai]["content"])
            lines.append("")

        out_path = OUT_DIR / f"extreme_{slug}.txt"
        out_path.write_text("\n".join(lines))
        print(f"  ✓ wrote {out_path} ({len(picks)} examples, {out_path.stat().st_size} bytes)")


def main() -> None:
    if not FIT_NPZ.exists():
        raise SystemExit(
            f"Missing FA fit at {FIT_NPZ}. Run analysis_for_paper.v2.py --k 4 first."
        )

    fit = np.load(FIT_NPZ)
    scores = fit["scores"]            # (n_personas, k)
    loadings = fit["loadings"]        # (n_items, k)
    items = json.loads(ITEM_LABELS_JSON.read_text())
    assert loadings.shape[0] == len(items), (loadings.shape, len(items))

    fx_name = FACTOR_NAMES[FX]
    fy_name = FACTOR_NAMES[FY]

    # Pick which item arrows to draw: top-|loading| items on FX, plus same
    # on FY, deduplicated.
    def top_idx(col: int, n: int) -> list[int]:
        order = np.argsort(-np.abs(loadings[:, col]))
        return list(order[:n])
    arrow_idx = sorted(
        set(top_idx(FX, N_TOP_ITEMS_PER_FACTOR))
        | set(top_idx(FY, N_TOP_ITEMS_PER_FACTOR))
    )

    fig, ax = plt.subplots(figsize=(7.5, 7.0))

    # Scatter: each baseline persona-rollout.
    ax.scatter(
        scores[:, FX], scores[:, FY],
        s=8, alpha=0.30, color="#2563eb", linewidths=0,
        rasterized=True,
    )

    # Arrows for top-loading items, projecting onto the (FX, FY) plane.
    for idx in arrow_idx:
        lx = loadings[idx, FX]
        ly = loadings[idx, FY]
        ax.arrow(
            0.0, 0.0,
            ARROW_SCALE * lx, ARROW_SCALE * ly,
            head_width=0.06, head_length=0.10,
            length_includes_head=True,
            color="#111", linewidth=1.0, alpha=0.85,
            zorder=3,
        )
        # Tag the arrowhead with the item's dimension (the bracketed
        # category prefix), nudged outward.
        dim = items[idx].get("dimension") or items[idx].get("col_id", "?")
        nudge = 1.10
        ax.text(
            ARROW_SCALE * lx * nudge, ARROW_SCALE * ly * nudge,
            dim, fontsize=8, color="#111",
            ha="center", va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.0),
            zorder=4,
        )

    ax.axhline(0, color="#888", linewidth=0.5)
    ax.axvline(0, color="#888", linewidth=0.5)
    ax.set_xlabel(fx_name, fontsize=18)
    ax.set_ylabel(fy_name, fontsize=18)
    ax.set_title(
        "Persona rollouts projected onto highest-variance factors",
        fontsize=18,
    )
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal", adjustable="datalim")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    fig.savefig(OUT_PATH.with_suffix(".pdf"))
    if PAPER_OUT_PATH is not None:
        PAPER_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(PAPER_OUT_PATH)
        print(f"✓ saved {PAPER_OUT_PATH}")
    plt.close(fig)
    print(f"✓ saved {OUT_PATH}")
    print(f"✓ saved {OUT_PATH.with_suffix('.pdf')}")

    if DUMP_EXEMPLARS:
        print("\n[exemplars] dumping per-axis-pole single-axis-extreme rollouts...")
        _dump_exemplars(scores)


if __name__ == "__main__":
    main()
