"""Same-trait opposing LoRA heatmaps for Appendix E.

For each of the five OCEAN traits, plots a 5×5 heatmap of LLM-judge scores
across a grid of amplifier × suppressor scales in {0.0, 0.5, 1.0, 1.5, 2.0}.
The x-axis is the amplifier scale and the y-axis is the suppressor scale.

Data comes from the five same-trait opposing sweeps:
  - o_plus × o_minus  → openness_v2   (EVAL_NAME: o_plus_x_o_minus-vanton4-paired-dpo-on-openness)
  - c_plus × c_minus  → conscientiousness_v2
  - e_plus × e_minus  → extraversion_v2
  - a_plus × a_minus  → agreeableness_v2
  - n_plus × n_minus  → neuroticism_v2

Cells are resolved via CanonicalCell.hf_dir() which handles baseline / single-adapter
/ combo tiers automatically — no manual path construction needed.

Paper figures (Appendix E):
    - paper/figures/appendix/fig_E_opposing_lora_openness.pdf
    - paper/figures/appendix/fig_E_opposing_lora_conscientiousness.pdf
    - paper/figures/appendix/fig_E_opposing_lora_extraversion.pdf
    - paper/figures/appendix/fig_E_opposing_lora_agreeableness.pdf
    - paper/figures/appendix/fig_E_opposing_lora_neuroticism.pdf

Run with:
    uv run python -m src_dev.visualisations.paper_appendix_opposing_lora_heatmaps \
        --openness <fp> 2>&1 | tee scratch/opposing_lora_heatmaps.log

To regenerate after new sweep data is uploaded to HF, delete the cache dir:
    scratch/paper_plots_cache/opposing_lora_heatmaps/
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src_dev.evals.cell_sweep.cell_identity import AdapterSpec, CanonicalCell
from src_dev.utils.hf_hub import download_path_to_dir
from src_dev.visualisations import PAPER_FIGURES_DIR

PAPER_FIGURES = [
    "appendix/fig_E_opposing_lora_openness.pdf",
    "appendix/fig_E_opposing_lora_conscientiousness.pdf",
    "appendix/fig_E_opposing_lora_extraversion.pdf",
    "appendix/fig_E_opposing_lora_agreeableness.pdf",
    "appendix/fig_E_opposing_lora_neuroticism.pdf",
]

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

SPINE_COLOR = "#2f3748"
AXIS_FACE = "#fbfbfc"
GRID_COLOR = "#dfe3e8"

PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.facecolor": AXIS_FACE,
    "axes.edgecolor": SPINE_COLOR,
    "axes.labelcolor": SPINE_COLOR,
    "axes.titlecolor": SPINE_COLOR,
    "axes.titleweight": "semibold",
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "axes.linewidth": 0.8,
    "xtick.color": SPINE_COLOR,
    "ytick.color": SPINE_COLOR,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.6,
}
plt.rcParams.update(PAPER_STYLE)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_REPO_ID = "persona-shattering-lasr/monorepo"
MODEL_SLUG = "llama-3.1-8b-it"
RATER_ID = "qwen3_235b"

# [0, +2] in 0.5 steps — matches the sweep configs
SCALES = [0.0, 0.5, 1.0, 1.5, 2.0]

CACHE_DIR = project_root / "scratch" / "paper_plots_cache" / "opposing_lora_heatmaps"

_FT = "fine_tuning/llama-3.1-8b-it/ocean"


def _adapter(trait: str, direction: str) -> AdapterSpec:
    dir_word = "amplifying" if direction == "amplifier" else "suppressing"
    return AdapterSpec.from_ref(
        f"{HF_REPO_ID}::"
        f"{_FT}/{trait}/{direction}/vanton4_paired_dpo"
        f"/lora/{trait}_{dir_word}_full_vanton4-persona"
    )


# Each entry: (trait_name, amp_direction_label, sup_direction_label, eval_name, out_rel_path)
HEATMAPS = [
    (
        "openness",
        "o_plus_x_o_minus-vanton4-paired-dpo-on-openness",
        "appendix/fig_E_opposing_lora_openness.pdf",
    ),
    (
        "conscientiousness",
        "c_plus_x_c_minus-vanton4-paired-dpo-on-conscientiousness",
        "appendix/fig_E_opposing_lora_conscientiousness.pdf",
    ),
    (
        "extraversion",
        "e_plus_x_e_minus-vanton4-paired-dpo-on-extraversion",
        "appendix/fig_E_opposing_lora_extraversion.pdf",
    ),
    (
        "agreeableness",
        "a_plus_x_a_minus-vanton4-paired-dpo-on-agreeableness",
        "appendix/fig_E_opposing_lora_agreeableness.pdf",
    ),
    (
        "neuroticism",
        "n_plus_x_n_minus-vanton4-paired-dpo-on-neuroticism",
        "appendix/fig_E_opposing_lora_neuroticism.pdf",
    ),
]

# ---------------------------------------------------------------------------
# Hydration
# ---------------------------------------------------------------------------


def _cache_path(hf_path: str) -> Path:
    return CACHE_DIR / hf_path


def _hydrate_judge_file(hf_path: str) -> Path | None:
    local = _cache_path(hf_path)
    if local.exists() and local.stat().st_size > 0:
        return local
    parent_hf, filename = hf_path.rsplit("/", 1)
    local_parent = _cache_path(parent_hf)
    try:
        download_path_to_dir(
            repo_id=HF_REPO_ID,
            path_in_repo=parent_hf,
            target_dir=local_parent,
            allow_patterns=[filename],
        )
    except Exception as exc:
        print(f"  ✗ hydrate failed: {hf_path}: {type(exc).__name__}: {str(exc)[:120]}")
        return None
    if local.exists() and local.stat().st_size > 0:
        return local
    return None


def _mean_score(jsonl_path: Path) -> float | None:
    scores: list[float] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            val = row.get("score")
            if val is None or not isinstance(val, (int, float)):
                continue
            scores.append(float(val))
    return statistics.fmean(scores) if scores else None


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


def build_grid(trait: str, eval_name: str, fingerprint: str) -> np.ndarray:
    """Return a 5×5 array of mean scores, shape [sup_axis, amp_axis].

    Rows (y) indexed by SCALES (suppressor), columns (x) by SCALES (amplifier).
    NaN for cells not yet on HF.
    """
    amp = _adapter(trait, "amplifier")
    sup = _adapter(trait, "suppressor")
    metric_name = f"{trait}_v2"

    grid = np.full((len(SCALES), len(SCALES)), np.nan, dtype=float)
    for yi, sup_scale in enumerate(SCALES):
        for xi, amp_scale in enumerate(SCALES):
            cell = CanonicalCell.from_scales([(amp, amp_scale), (sup, sup_scale)])
            cell_dir = cell.hf_dir(MODEL_SLUG, eval_name, fingerprint)
            hf_path = f"{cell_dir}/judge_runs/{RATER_ID}/{metric_name}.jsonl"
            local = _hydrate_judge_file(hf_path)
            if local is None:
                print(f"  ⚠ amp={amp_scale:+.1f} sup={sup_scale:+.1f}: missing on HF")
                continue
            mean = _mean_score(local)
            if mean is None:
                print(f"  ⚠ amp={amp_scale:+.1f} sup={sup_scale:+.1f}: no valid scores")
                continue
            grid[yi, xi] = mean
            print(f"  ✓ amp={amp_scale:+.1f} sup={sup_scale:+.1f}: mean = {mean:+.3f}")
    return grid


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def render_heatmap(
    grid: np.ndarray,
    *,
    trait: str,
    out_path: Path,
) -> None:
    trait_title = trait.capitalize()

    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    vmax = 4.0
    im = ax.imshow(
        grid,
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
    )

    ax.set_xticks(range(len(SCALES)))
    ax.set_yticks(range(len(SCALES)))
    ax.set_xticklabels([f"{s:+g}" for s in SCALES])
    ax.set_yticklabels([f"{s:+g}" for s in SCALES])
    ax.set_xlabel(f"{trait_title} amplifier scale")
    ax.set_ylabel(f"{trait_title} suppressor scale")
    ax.set_title(f"{trait_title} judge score", loc="left", pad=8)

    # Annotate cells.
    n_y, n_x = grid.shape
    base_xi = SCALES.index(0.0)
    base_yi = SCALES.index(0.0)
    for yi in range(n_y):
        for xi in range(n_x):
            val = grid[yi, xi]
            if np.isnan(val):
                label = "—"
                color = "gray"
            else:
                label = f"{val:+.2f}"
                color = "white" if abs(val) > 2.0 else "black"
            ax.text(xi, yi, label, ha="center", va="center", fontsize=9, color=color)
            if xi == base_xi and yi == base_yi and not np.isnan(val):
                ax.text(xi, yi - 0.32, "(base)", ha="center", va="center",
                        fontsize=8, color=color, style="italic")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{trait}_v2 (Qwen3-235B)", rotation=270, labelpad=15)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"✓ saved {out_path}")
    print(f"✓ saved {png_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(fingerprints: dict[str, str] | None = None) -> None:
    """Render all five opposing-LoRA heatmaps.

    Args:
        fingerprints: Optional mapping from trait name to sweep fingerprint.
            If None, reads fingerprints from the OPPOSING_LORA_FINGERPRINTS
            dict at the bottom of this module.
    """
    if fingerprints is None:
        fingerprints = OPPOSING_LORA_FINGERPRINTS

    print(f"[opposing-lora heatmaps] cache dir: {CACHE_DIR}")
    for trait, eval_name, out_rel in HEATMAPS:
        fp = fingerprints.get(trait)
        if not fp:
            print(f"\n[heatmap] SKIP {trait} — no fingerprint set yet")
            continue
        out_path = PAPER_FIGURES_DIR / out_rel
        print(f"\n[heatmap] {trait}  fp={fp}")
        print(f"           → {out_path}")
        grid = build_grid(trait, eval_name, fp)
        render_heatmap(grid, trait=trait, out_path=out_path)


# ---------------------------------------------------------------------------
# Fingerprints — populated after sweeps complete and data is on HF.
# Set the fingerprint for a trait once its sweep is uploaded.
# ---------------------------------------------------------------------------

OPPOSING_LORA_FINGERPRINTS: dict[str, str] = {
    # "openness": "xxxxxxxxxx",
    # "conscientiousness": "xxxxxxxxxx",
    # "extraversion": "xxxxxxxxxx",
    # "agreeableness": "xxxxxxxxxx",
    # "neuroticism": "xxxxxxxxxx",
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--openness", metavar="FP", help="Fingerprint for the openness sweep"
    )
    parser.add_argument(
        "--conscientiousness", metavar="FP", help="Fingerprint for the conscientiousness sweep"
    )
    parser.add_argument(
        "--extraversion", metavar="FP", help="Fingerprint for the extraversion sweep"
    )
    parser.add_argument(
        "--agreeableness", metavar="FP", help="Fingerprint for the agreeableness sweep"
    )
    parser.add_argument(
        "--neuroticism", metavar="FP", help="Fingerprint for the neuroticism sweep"
    )
    args = parser.parse_args()

    fps: dict[str, str] = dict(OPPOSING_LORA_FINGERPRINTS)
    for trait in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
        val = getattr(args, trait)
        if val:
            fps[trait] = val

    main(fps)
