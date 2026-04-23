"""Section-4.2 factor analysis — analysis pipeline for the paper.

This script is the working manuscript for the psychometric factor-analysis
results that go into the paper's Section 4.2 ("Applying Traditional
Psychometrics to LLM Personas"). It is intentionally a thin orchestrator:
configuration at the top, then a series of stage functions that call into
`src_dev.factor_analysis` / `src_dev.psychometric` modules.

Design decisions baked into this script (see `HANDOFF_qwen25_b_rerun.md`
and `FA_METHODOLOGY_FOLLOWUPS.md` for the research-history side):

    • The input data is the combined `v5` Likert + `trait_ocean_natural_v1`
      trait_mcq response matrix on the `B` rollout cache (2500 Llama-3.1-8B
      personas × 15 turns; Qwen2.5 drops one 66k-token outlier, so Qwen =
      2499). Both models' combined-questionnaire dirs are already assembled
      on HuggingFace at `persona-shattering-lasr/psychometric-fa-runs`
      under `combined/combined-R[B]-Q[v5+trait_ocean_natural_v1]{,-qm_<slug>}/`.

    • Headline rotation: oblimin (allows correlated factors — psychometrically
      defensible). Varimax is kept as an appendix robustness check, populated
      by a second pass when we want it.

    • Headline encoding for trait_mcq: `soft_ev` (the default expected-value
      mapping in [0,1]). `logit` will be added as a robustness check later.

    • Headline model: Llama-3.1-8B-Instruct. Qwen2.5-7B-Instruct is the
      appendix cross-model comparison.

    • We do NOT upload to HuggingFace from this script (current rate-limit
      state means uploads will fail). All outputs land under
      `scratch/psychometric_fa_paper/` (gitignored).

    • We intentionally do not reuse the existing `factor_analysis/` outputs
      under the hydrated combined dirs — we refit from the response matrix
      so the decisions (k, rotation, residualization, etc.) are explicit and
      traceable in this file.

Analysis steps in this script are independent of the main pipeline's
stage numbering (``scripts_dev/unsupervised_embeddings/psychometric_rollout_fa.py``).
They are ordered analysis actions, not orchestrator stages.

Current state of the script (grows iteratively):

    [done] load_model_data          — Hydrate combined-dir questionnaire from
                                      HF (or reuse the local scratch copy)
                                      and run standard preprocessing.
    [done] run_n_factors_suggest    — Horn's parallel analysis, MAP, EKC,
                                      acceleration, Kaiser, optimal coords,
                                      scree elbow. Saves scree + comparison
                                      plots + summary JSON.
    [done] fit_factor_analysis      — Fit oblimin FA at the configured k per
                                      model. Saves npz + json sidecar, per-
                                      column item_labels.json, top-30 items
                                      text summary, and the full trait-
                                      alignment (CSVs + heatmap PDFs). Also
                                      copies the questionnaire triplet next
                                      to the fits so the /label-fa-factors
                                      skill resolves cleanly.

    [todo] cross-model Tucker's congruence (Llama ↔ Qwen), Hungarian match
           on min(k_a, k_b), with the extra factor flagged as unmatched.
    [todo] varimax + logit-encoding robustness passes for the appendix.
    [todo] paper figures (scree, loading heatmaps, etc.).
"""

from __future__ import annotations

# ── Seeds (set before any stochastic imports) ────────────────────────────────
import random

import numpy as np

SEED = 436
random.seed(SEED)
np.random.seed(SEED)

try:
    import torch  # seeded for any downstream torch-RNG usage
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass

# ── Standard library ─────────────────────────────────────────────────────────
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

# ── Third-party ──────────────────────────────────────────────────────────────
from dotenv import load_dotenv

load_dotenv()

# ── src_dev imports ──────────────────────────────────────────────────────────
from src_dev.factor_analysis import (
    parallel_analysis,
    plot_n_factors_comparison,
    run_factor_analysis,
    save_factor_analysis,
    suggest_n_factors,
)
from src_dev.factor_analysis.trait_alignment import (
    compute_factor_trait_alignment,
    plot_all_alignment,
    save_alignment,
)
from src_dev.psychometric.combine import load_pair_outputs
from src_dev.psychometric.preprocessing import preprocess_response_matrix
from src_dev.unsupervised_runs.io import hydrate_dataset_subtree
from src_dev.visualisations import PAPER_FIGURES_DIR


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

# HuggingFace dataset repo holding the Stage-2 questionnaire artifacts and
# the combined v5+trait_ocean_natural_v1 matrices.
HF_REPO_ID = "persona-shattering-lasr/psychometric-fa-runs"

# Where this script writes its outputs. Separate from the existing pipeline's
# `scratch/psychometric_fa/` so the paper analysis is unambiguously fresh.
OUTPUT_ROOT = Path("scratch/psychometric_fa_paper")


@dataclass(frozen=True)
class ModelRun:
    """One combined v5+trait_ocean_natural_v1 matrix we want to analyse.

    ``slug`` is the short identifier used for output dirs. ``hf_path`` is the
    prefix inside HF_REPO_ID that holds this model's combined questionnaire
    artifacts (the ``combined/...`` dir on HF). ``local_source_dir`` is where
    the existing pipeline mirrors that combined dir locally — we fall back
    to this if hydration fails or if we want to reuse what's already on disk.
    """
    slug: str
    label: str                 # pretty name for plots + provenance
    hf_path: str
    local_source_dir: Path


MODELS: list[ModelRun] = [
    ModelRun(
        slug="llama-3.1-8b",
        label="Llama-3.1-8B-Instruct",
        hf_path="combined/combined-R[B]-Q[v5+trait_ocean_natural_v1]",
        local_source_dir=Path(
            "scratch/psychometric_fa/combined-R[B]-Q[v5+trait_ocean_natural_v1]"
        ),
    ),
    ModelRun(
        slug="qwen2.5-7b",
        label="Qwen2.5-7B-Instruct",
        hf_path="combined/combined-R[B]-Q[v5+trait_ocean_natural_v1]-qm_qwen257binstruct",
        local_source_dir=Path(
            "scratch/psychometric_fa/combined-R[B]-Q[v5+trait_ocean_natural_v1]-qm_qwen257binstruct"
        ),
    ),
]


# ── Preprocessing ────────────────────────────────────────────────────────────
# Same defaults as the main pipeline. Per-block relative variance floor of
# 0.1 drops items that carry no information relative to their block's median;
# no high-variance persona drop; no residualization (the B rollout preset has
# a single input_group_id per row, so residualization is a no-op).
MIN_ITEM_VARIANCE: float = 0.1
HIGH_VARIANCE_PERSONA_DROP_PCT: float = 0.0
DO_RESIDUALIZE: bool = False

# ── n-factors suggestion ─────────────────────────────────────────────────────
# Full suite minus CV (slowest, quadratic in k and in n_folds — we can add
# it back once the cheaper methods have triangulated a range). OC / scree
# elbow are included because Horn's alone tends to over-retain on wide
# Likert matrices and it's useful to have multiple scree-shape criteria for
# comparison.
N_FACTORS_METHODS: tuple[str, ...] = (
    "parallel",            # Horn's permutation-null parallel analysis (default)
    "map",                 # Velicer's MAP (conservative)
    "ekc",                 # Empirical Kaiser Criterion
    "acceleration",        # Local 2nd-difference scree elbow
    "kaiser",              # Kaiser–Guttman (reference only; over-retains)
    "optimal_coordinates", # Raîche 2013 non-graphical Cattell scree
    "scree_elbow",         # Perpendicular-distance scree elbow
    # "cv_reconstruction", # enable when we want cross-validated NLL
)
N_FACTORS_K_MAX: int = 25              # hard cap for MAP / OC / scree / acceleration
N_FACTORS_PARALLEL_ITERATIONS: int = 200


# ── FA fit ──────────────────────────────────────────────────────────────────
# Per-model chosen k, picked from the scree-elbow (perpendicular distance to
# first-last chord) recommendation produced by `run_n_factors_suggest`. We
# use per-model k rather than a common k because the scree elbow sits at a
# model-specific knee; the cross-model Tucker's step later matches on
# min(k_a, k_b) and flags the extra factor on the larger side as unmatched.
FA_K_BY_MODEL: dict[str, int] = {
    "llama-3.1-8b": 4,  # scree elbow from run_n_factors_suggest
    "qwen2.5-7b":   5,  # scree elbow from run_n_factors_suggest
}

# Headline rotation for the main-body analysis. Varimax + logit-encoding
# passes are deferred to the appendix robustness sweep; when we add them,
# we'll re-call `fit_factor_analysis` with the alternative config rather
# than re-fit everything.
FA_ROTATION: str = "oblimin"
FA_METHOD: str = "principal"

# How many top-loading items to write per factor in the text summary that
# feeds the manual labelling session. 30 is enough that a labeller has
# multiple fall-backs when the top few are ambiguous, but still fits
# comfortably on one screen per factor.
TOP_LOADING_ITEMS_FOR_LABELLING: int = 30

# OCEAN trait ordering for the alignment tables/plots. Items whose
# `dimension` field is the empty string (the v5 Likert block, which was
# iteratively curated rather than labelled by OCEAN dimension) sort into
# the trailing empty-trait column.
OCEAN_TRAIT_ORDER: list[str] = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
    "",
]


# ── Paper figure output ─────────────────────────────────────────────────────
# Section 4.2 ("Applying Traditional Psychometrics to LLM Personas") takes
# the Llama scree as its headline parallel-analysis figure; the Qwen scree
# would go in the appendix (not emitted yet — add here when we want it).
# Paths are resolved against `PAPER_FIGURES_DIR` (paper/figures/) via the
# helper in `src_dev.visualisations`. Per paper/CLAUDE.md naming convention
# `fig_<section>_<short_name>.<ext>`.
#
# Leaving a slug out of this dict skips the paper-figure write for that
# model. Only PDF (for vector output) — raster PNG always lands alongside
# in the scratch dir.
PAPER_SCREE_FIGURES: dict[str, str] = {
    "llama-3.1-8b": "unsupervised/fig_4_2_1_scree_llama.pdf",
}


# ═════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fa_paper")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 — HYDRATE + LOAD
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class LoadedData:
    """Per-model raw + preprocessed data bundle."""

    model: ModelRun
    # Raw artifacts (as produced by the combine stage)
    raw_matrix: np.ndarray           # (K, M) pre-preprocess
    raw_metadata: list[dict]         # len K
    raw_items: list[dict]            # len M
    # Preprocessed artifacts (post row+column filtering)
    matrix: np.ndarray               # (K', M')
    metadata: list[dict]             # len K'
    items: list[dict]                # len M'
    # Where outputs for this model go
    output_dir: Path


def _questionnaire_dir_for(model: ModelRun) -> Path:
    """Local path to the combined dir's ``questionnaire/`` subfolder.

    Prefers the existing pipeline's scratch location if already populated
    (saves a re-download); otherwise mirrors from HF into a parallel
    location under ``OUTPUT_ROOT`` so the pipeline dirs stay untouched.
    """
    existing = model.local_source_dir / "questionnaire"
    if (existing / "response_matrix.npy").exists():
        return existing.parent  # load_pair_outputs expects the parent dir
    # Fall back: mirror from HF into our output tree.
    mirror_parent = OUTPUT_ROOT / model.slug / "hydrated"
    mirror_q = mirror_parent / "questionnaire"
    if not (mirror_q / "response_matrix.npy").exists():
        log.info("[%s] hydrating from HF: %s", model.slug, model.hf_path)
        hydrate_dataset_subtree(
            repo_id=HF_REPO_ID,
            path_in_repo=f"{model.hf_path.rstrip('/')}/questionnaire",
            local_dir=mirror_q,
            required=True,
        )
    return mirror_parent


def _preprocess(
    matrix: np.ndarray,
    metadata: list[dict],
    items: list[dict],
    *,
    output_dir: Path,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Run the standard preprocess pipeline with our configured defaults."""
    variance_export_path = output_dir / "item_variances_ranked.jsonl"
    cleaned, meta, cols, _groups = preprocess_response_matrix(
        matrix,
        metadata,
        items,
        min_item_variance=MIN_ITEM_VARIANCE,
        high_variance_persona_drop_pct=HIGH_VARIANCE_PERSONA_DROP_PCT,
        do_residualize=DO_RESIDUALIZE,
        variance_export_path=variance_export_path,
    )
    return cleaned, meta, cols


def load_model_data(model: ModelRun) -> LoadedData:
    """Hydrate (if needed) and load one model's combined response matrix."""
    q_dir = _questionnaire_dir_for(model)
    raw_matrix, raw_meta, raw_items = load_pair_outputs(q_dir)
    log.info(
        "[%s] raw matrix: %d personas × %d items",
        model.slug, raw_matrix.shape[0], raw_matrix.shape[1],
    )

    out_dir = OUTPUT_ROOT / model.slug
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix, meta, items = _preprocess(
        raw_matrix, raw_meta, raw_items, output_dir=out_dir,
    )
    log.info(
        "[%s] preprocessed matrix: %d personas × %d items",
        model.slug, matrix.shape[0], matrix.shape[1],
    )

    return LoadedData(
        model=model,
        raw_matrix=raw_matrix,
        raw_metadata=raw_meta,
        raw_items=raw_items,
        matrix=matrix,
        metadata=meta,
        items=items,
        output_dir=out_dir,
    )


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 — N-FACTORS SUGGESTION
# ═════════════════════════════════════════════════════════════════════════════


def run_n_factors_suggest(data: LoadedData) -> dict:
    """Run the full n-factors method suite + save plots + dump summary JSON.

    Output layout:
        {output_dir}/n_factors/
            n_factors_suggest.json        # full per-method dict
            n_factors_suggest.png         # comparison plot across methods
            parallel_analysis.json        # Horn's-only detail
            parallel_analysis.png         # scree plot with null percentile
    """
    out = data.output_dir / "n_factors"
    out.mkdir(parents=True, exist_ok=True)

    log.info("[%s] suggest_n_factors (k_max=%d, parallel_iters=%d)…",
             data.model.slug, N_FACTORS_K_MAX, N_FACTORS_PARALLEL_ITERATIONS)
    result = suggest_n_factors(
        data.matrix,
        methods=N_FACTORS_METHODS,
        k_max=N_FACTORS_K_MAX,
        parallel_n_iterations=N_FACTORS_PARALLEL_ITERATIONS,
        seed=SEED,
        verbose=True,
    )

    # Full result dump.
    (out / "n_factors_suggest.json").write_text(
        json.dumps(_jsonable(result), indent=2)
    )
    log.info("[%s] wrote %s", data.model.slug, out / "n_factors_suggest.json")

    # Comparison plot across all methods.
    plot_n_factors_comparison(
        result, out / "n_factors_suggest.png",
        title_suffix=f" — {data.model.label}",
    )
    log.info("[%s] wrote %s", data.model.slug, out / "n_factors_suggest.png")

    # Horn's-only scree plot (useful for the paper's main figure).
    pa = result.get("parallel")
    if pa is not None:
        (out / "parallel_analysis.json").write_text(
            json.dumps(_jsonable(pa), indent=2)
        )
        extra_paths: list[Path] = []
        paper_rel = PAPER_SCREE_FIGURES.get(data.model.slug)
        if paper_rel is not None:
            paper_path = PAPER_FIGURES_DIR / paper_rel
            paper_path.parent.mkdir(parents=True, exist_ok=True)
            extra_paths.append(paper_path)
        _plot_scree(
            pa, out / "parallel_analysis.png",
            title=data.model.label,
            extra_save_paths=extra_paths,
        )
        log.info("[%s] wrote %s", data.model.slug, out / "parallel_analysis.png")
        for p in extra_paths:
            log.info("[%s] wrote %s", data.model.slug, p)

    return result


def _plot_scree(
    pa: dict,
    save_path: Path,
    *,
    title: str,
    extra_save_paths: list[Path] | None = None,
) -> None:
    """Plot real eigenvalues vs Horn's null percentile threshold.

    ``extra_save_paths`` is for parallel writes to the paper's figure tree
    (vector PDF) in addition to the scratch PNG — same figure, different
    file. Format is inferred by matplotlib from each path's suffix.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = np.asarray(pa["real_eigenvalues"], dtype=float)
    thr = np.asarray(pa["random_threshold"], dtype=float)
    n_rec = int(pa["n_recommended"])
    k_max_display = min(30, len(real))
    x = np.arange(1, k_max_display + 1)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, real[:k_max_display], "o-", color="tab:blue", label="Real eigenvalues")
    ax.plot(x, thr[:k_max_display], "s--", color="tab:red",
            label="Horn null, 95th percentile")
    ax.axvline(n_rec + 0.5, color="tab:grey", alpha=0.5, linestyle=":",
               label=f"Horn's k = {n_rec}")
    ax.set_xlabel("Factor index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Scree plot — {title}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    for p in extra_save_paths or ():
        fig.savefig(p)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# FIT FACTOR ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════


def _ensure_questionnaire_copy(data: LoadedData) -> Path:
    """Copy the questionnaire triplet into the paper output dir.

    The ``/label-fa-factors`` skill's resolver walks up from a ``fa_*.npz``
    looking for a sibling ``questionnaire/`` dir with ``items.json``. Making
    that sibling exist locally under our paper output tree means the skill
    resolves cleanly without needing to know about the upstream combined
    dir.
    """
    import shutil

    src_q = data.model.local_source_dir / "questionnaire"
    dst_q = data.output_dir / "questionnaire"
    dst_q.mkdir(parents=True, exist_ok=True)
    for fname in ("items.json", "metadata.jsonl", "response_matrix.npy"):
        src = src_q / fname
        dst = dst_q / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    return dst_q


def _write_item_labels_json(
    items: list[dict],
    save_path: Path,
) -> None:
    """Per-column metadata aligned with loading rows.

    Same schema the main pipeline uses for ``fa_*_item_labels.json`` (see
    ``src_dev/psychometric/stages/factor_analysis.py``) — the skill's
    ``describe`` command joins against ``col_id`` to recover block-specific
    rich fields (e.g. MCQ options, answer_mapping, reverse-keying).
    """
    payload = [
        {
            "col_id": col["col_id"],
            "text": col.get("text", ""),
            "block": col.get("block", ""),
            "dimension": col.get("dimension"),
            "reverse_keyed": col.get("reverse_keyed", False),
        }
        for col in items
    ]
    save_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _top_items_for_factor(
    loadings: np.ndarray,
    items: list[dict],
    factor_index: int,
    n_top: int,
) -> tuple[list[tuple[float, dict]], list[tuple[float, dict]]]:
    """Return top-`n_top` positive and negative-loading items for one factor.

    Each returned entry is ``(signed_loading, item_dict)``. Positive list is
    descending by signed loading; negative list is ascending by signed
    loading (most-negative first). Items are the full column defs.
    """
    col = loadings[:, factor_index]
    order_desc = np.argsort(-col)  # positive end first
    order_asc = np.argsort(col)    # negative end first
    pos: list[tuple[float, dict]] = [
        (float(col[i]), items[i])
        for i in order_desc[:n_top]
        if col[i] > 0
    ]
    neg: list[tuple[float, dict]] = [
        (float(col[i]), items[i])
        for i in order_asc[:n_top]
        if col[i] < 0
    ]
    return pos, neg


def _format_item_line(loading: float, item: dict) -> str:
    block = item.get("block", "") or "?"
    dim = item.get("dimension") or ""
    rev = " (REVERSED)" if item.get("reverse_keyed") else ""
    text = (item.get("text") or "").replace("\n", " ").strip()
    dim_tag = f" [{dim}]" if dim else ""
    return f"    {loading:+.3f}  <{block}{dim_tag}>{rev}  {text}"


def _write_top_items_summary(
    fa: dict,
    items: list[dict],
    save_path: Path,
    *,
    n_top: int,
    model_label: str,
    n_factors: int,
    rotation: str,
) -> None:
    """Plain-text per-factor top-loading summary for the labelling session.

    Each factor gets: variance explained, top-n_top positive items, top-n_top
    negative items. For oblique rotations we also print the factor
    correlation matrix at the bottom. Deliberately terminal-friendly — this
    is what a labeller reads while doing the manual pass.
    """
    loadings = fa["loadings"]
    prop_var = fa.get("proportion_variance", [])
    cum_var = fa.get("cumulative_variance", [])
    phi = fa.get("factor_correlation_matrix")

    lines: list[str] = []
    lines.append(f"FA fit — {model_label}")
    lines.append(f"rotation={rotation}  n_factors={n_factors}  "
                 f"n_items={loadings.shape[0]}")
    lines.append("")
    lines.append("Variance explained per factor (ss_loadings fraction):")
    for f in range(n_factors):
        pv = float(prop_var[f]) if len(prop_var) > f else float("nan")
        cv = float(cum_var[f]) if len(cum_var) > f else float("nan")
        lines.append(f"  F{f+1}:  prop={pv:.3f}   cum={cv:.3f}")
    lines.append("")

    for f in range(n_factors):
        lines.append("=" * 78)
        lines.append(f"F{f+1}  (top-{n_top} by signed loading)")
        lines.append("=" * 78)
        pos, neg = _top_items_for_factor(loadings, items, f, n_top)
        lines.append(f"  Positive loadings ({len(pos)} items):")
        for loading, item in pos:
            lines.append(_format_item_line(loading, item))
        lines.append("")
        lines.append(f"  Negative loadings ({len(neg)} items):")
        for loading, item in neg:
            lines.append(_format_item_line(loading, item))
        lines.append("")

    if phi is not None:
        lines.append("=" * 78)
        lines.append("Factor correlation matrix (oblique rotation)")
        lines.append("=" * 78)
        header = "        " + "  ".join(f"{'F'+str(j+1):>7}" for j in range(n_factors))
        lines.append(header)
        for i in range(n_factors):
            row = f"  F{i+1:<2}  " + "  ".join(
                f"{phi[i, j]:>7.3f}" for j in range(n_factors)
            )
            lines.append(row)
        lines.append("")

    save_path.write_text("\n".join(lines))


def _run_trait_alignment(
    fa: dict,
    items: list[dict],
    save_dir: Path,
    *,
    title_prefix: str,
) -> None:
    """OCEAN factor-alignment summary + CSVs + heatmaps.

    Items with ``dimension=""`` sort into the trailing empty-trait column,
    so factors dominated by v5 Likert items (which carry no OCEAN label)
    show up as "winner_trait = ''".
    """
    item_dims = [str(it.get("dimension") or "") for it in items]
    alignment = compute_factor_trait_alignment(
        fa["loadings"],
        item_dims,
        trait_order=OCEAN_TRAIT_ORDER,
        top_k=20,
    )
    save_alignment(alignment, save_dir)
    plot_all_alignment(alignment, save_dir, title_prefix=title_prefix)


@dataclass
class FaFit:
    """Result bundle returned by ``fit_factor_analysis``."""

    model_slug: str
    k: int
    rotation: str
    method: str
    npz_path: Path       # path to fa_{k}_{method}_{rotation}.npz
    item_labels_path: Path
    top_items_path: Path
    alignment_dir: Path
    questionnaire_dir: Path  # sibling that the skill resolves against


def fit_factor_analysis(data: LoadedData, *, k: int) -> FaFit:
    """Fit FA at the configured rotation/method and lay outputs on disk.

    Output layout under ``{output_dir}/factor_analysis/raw/``:

        fa_{k}_{method}_{rotation}.npz
        fa_{k}_{method}_{rotation}.json              (config + array keys)
        fa_{k}_{method}_{rotation}_item_labels.json  (per-column metadata)
        fa_{k}_{method}_{rotation}_top{n}.txt        (labeller's summary)
        fa_{k}_{method}_{rotation}_alignment/        (OCEAN CSVs + heatmaps)

    Also ensures ``{output_dir}/questionnaire/items.json`` exists so the
    ``/label-fa-factors`` skill's resolver finds it as a sibling of
    factor_analysis/.
    """
    q_dir = _ensure_questionnaire_copy(data)

    fa_dir = data.output_dir / "factor_analysis" / "raw"
    fa_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "[%s] fitting FA: k=%d rotation=%s method=%s  (n=%d × p=%d)",
        data.model.slug, k, FA_ROTATION, FA_METHOD,
        data.matrix.shape[0], data.matrix.shape[1],
    )
    fa = run_factor_analysis(
        data.matrix,
        n_factors=k,
        method=FA_METHOD,
        rotation=FA_ROTATION,
    )

    base = fa_dir / f"fa_{k}_{FA_METHOD}_{FA_ROTATION}"
    npz_path = save_factor_analysis(
        fa,
        base,
        config={
            "n_factors": k,
            "method": FA_METHOD,
            "rotation": FA_ROTATION,
            "residualized": DO_RESIDUALIZE,
            "n_samples": int(data.matrix.shape[0]),
            "n_cols": int(data.matrix.shape[1]),
            "model_slug": data.model.slug,
            "model_label": data.model.label,
        },
    )

    item_labels_path = Path(str(base) + "_item_labels.json")
    _write_item_labels_json(data.items, item_labels_path)
    log.info("[%s] wrote %s", data.model.slug, item_labels_path)

    top_items_path = Path(
        str(base) + f"_top{TOP_LOADING_ITEMS_FOR_LABELLING}.txt"
    )
    _write_top_items_summary(
        fa, data.items, top_items_path,
        n_top=TOP_LOADING_ITEMS_FOR_LABELLING,
        model_label=data.model.label,
        n_factors=k,
        rotation=FA_ROTATION,
    )
    log.info("[%s] wrote %s", data.model.slug, top_items_path)

    alignment_dir = Path(str(base) + "_alignment")
    _run_trait_alignment(
        fa, data.items, alignment_dir,
        title_prefix=f"{data.model.label}  k={k} {FA_ROTATION}",
    )
    log.info("[%s] wrote %s", data.model.slug, alignment_dir)

    return FaFit(
        model_slug=data.model.slug,
        k=k,
        rotation=FA_ROTATION,
        method=FA_METHOD,
        npz_path=npz_path,
        item_labels_path=item_labels_path,
        top_items_path=top_items_path,
        alignment_dir=alignment_dir,
        questionnaire_dir=q_dir,
    )


# ═════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════════════


def _jsonable(obj):
    """Recursively convert NumPy scalars / arrays to plain Python for JSON."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    print("=" * 78)
    print(f"analysis_for_paper.py  —  output root: {OUTPUT_ROOT}")
    print("=" * 78)

    loaded: dict[str, LoadedData] = {}

    # ── Step: load each model's combined response matrix ────────────────
    for model in MODELS:
        log.info("═══ load_model_data [%s] ═══", model.slug)
        loaded[model.slug] = load_model_data(model)

    # ── Step: n-factors suggestion + scree plot on each model ───────────
    summary_rows: list[dict] = []
    for model in MODELS:
        log.info("═══ run_n_factors_suggest [%s] ═══", model.slug)
        data = loaded[model.slug]
        result = run_n_factors_suggest(data)
        row = {
            "model": model.slug,
            "n_personas": int(data.matrix.shape[0]),
            "n_items": int(data.matrix.shape[1]),
            **result["summary"],
        }
        summary_rows.append(row)

    # Print a compact comparison table.
    print()
    print("=" * 78)
    print("n-factors recommendations across models")
    print("=" * 78)
    methods_seen: list[str] = []
    for row in summary_rows:
        for k in row:
            if k not in ("model", "n_personas", "n_items") and k not in methods_seen:
                methods_seen.append(k)
    header = f"{'model':<14} {'n':>6} {'p':>5}  " + "  ".join(
        f"{m:>12}" for m in methods_seen
    )
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        cells = [f"{row['model']:<14} {row['n_personas']:>6} {row['n_items']:>5}  "]
        for m in methods_seen:
            v = row.get(m, "")
            cells.append(f"{v!s:>12}")
        print("  ".join(cells))

    # Also write the aggregate summary so we can cite it in prose.
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "n_factors_summary.json").write_text(
        json.dumps(summary_rows, indent=2)
    )
    log.info("wrote %s", OUTPUT_ROOT / "n_factors_summary.json")

    # ── Step: fit FA at the per-model scree-elbow k ─────────────────────
    fits: dict[str, FaFit] = {}
    for model in MODELS:
        log.info("═══ fit_factor_analysis [%s] ═══", model.slug)
        data = loaded[model.slug]
        k = FA_K_BY_MODEL.get(model.slug)
        if k is None:
            log.warning("no FA_K_BY_MODEL entry for %s; skipping", model.slug)
            continue
        fit = fit_factor_analysis(data, k=k)
        fits[model.slug] = fit

    if fits:
        print()
        print("=" * 78)
        print("FA fits written (ready for /label-fa-factors)")
        print("=" * 78)
        for slug, fit in fits.items():
            print(f"  {slug}:")
            print(f"    npz         : {fit.npz_path}")
            print(f"    top{TOP_LOADING_ITEMS_FOR_LABELLING} summary : "
                  f"{fit.top_items_path}")
            print(f"    alignment   : {fit.alignment_dir}")
            print(f"    questionnaire: {fit.questionnaire_dir}")


if __name__ == "__main__":
    main()
