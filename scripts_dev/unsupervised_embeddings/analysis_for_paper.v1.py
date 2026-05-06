"""Section-4.2 factor analysis — v1 pipeline (ARCHIVED).

ARCHIVED. The current paper uses ``analysis_for_paper.v2.py`` (v7 forced-
choice questionnaire + prefill v3). This v1 ran on the combined v5 Likert
+ trait_ocean_natural_v1 trait_mcq matrix and is preserved here for
reproducibility / cross-reference; the paper's Section-4.2 figures are
now produced by v2.

──── Original v1 docstring follows ────

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
    [done] export_html_browser      — Write an interactive factor_extremes
                                      HTML viewer per model with top/bottom-
                                      N rollout conversations per factor,
                                      picking up LLM labels from the skill's
                                      output dir when they exist.
    [done] run_within_model_validation — Cronbach's α per factor (over items
                                      with |loading| ≥ threshold, sign-
                                      oriented) + split-half congruence
                                      (Tucker's |φ| across N random
                                      half-splits). Writes
                                      ``validation/cronbach_alpha.json`` and
                                      ``validation/split_half_congruence.{json,png}``
                                      per model.
    [done] run_cross_model_congruence — Llama ↔ Qwen Tucker's |φ| across
                                      combined + per-block (likert,
                                      trait_mcq) shared-item subsets.
                                      Hungarian-matches anchor factors to
                                      target factors (anchor = Llama k=4,
                                      target = Qwen k=5, one unmatched
                                      Qwen factor reported separately).
                                      Writes ``cross_model/{pair}/{subset}/
                                      {phi_matrix.npy, phi_heatmap.{png,pdf},
                                      report.json}`` plus a per-pair
                                      ``summary.json``.

    [done] run_predictivity_cv      — Bi-cross-validation (Owen & Perry 2009):
                                      persona × item holdout, predicts
                                      held-out responses from factor scores,
                                      compares R² against item_mean /
                                      persona_shuffle / k−1 baselines.
    [done] paper figures             — Scree (Llama), within-model α + |φ|
                                      grouped bars, cross-model Tucker's |φ|
                                      heatmap written to paper/figures/
                                      unsupervised/.

    [todo] varimax + logit-encoding robustness passes for the appendix.

Paper figures this script emits (relative to ``paper/figures/``):

    - unsupervised/fig_4_2_1_scree_llama.pdf
      Section 4.2 Horn's parallel-analysis scree on the Llama fit.
    - unsupervised/fig_4_2_2_within_model_validation.pdf
      Section 4.2 grouped bars for Cronbach's α + split-half median |φ|
      per factor, both models.
    - unsupervised/fig_4_2_3_cross_model_phi.pdf
      Section 4.2 Llama↔Qwen Tucker's |φ| heatmap (combined shared items).

See ``paper/CLAUDE.md`` → "Code ↔ Paper Pointers" for the convention.
"""

from __future__ import annotations

# Module-level list mirroring the docstring above — keeps the list available
# without re-parsing the docstring. Paths relative to ``paper/figures/``.
PAPER_FIGURES: list[str] = [
    "unsupervised/fig_4_2_1_scree_llama.pdf",
    "unsupervised/fig_4_2_2_within_model_validation.pdf",
    "unsupervised/fig_4_2_3_cross_model_phi.pdf",
]

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
import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

# ── Third-party ──────────────────────────────────────────────────────────────
from dotenv import load_dotenv

load_dotenv()

# ── src_dev imports ──────────────────────────────────────────────────────────
from src_dev.factor_analysis import (
    load_factor_analysis,
    parallel_analysis,
    persona_item_cv,
    plot_n_factors_comparison,
    run_factor_analysis,
    save_factor_analysis,
    split_half_congruence,
    suggest_n_factors,
)
from src_dev.factor_analysis.reliability import classify_alpha, cronbach_alpha
from src_dev.factor_analysis.trait_alignment import (
    compute_factor_trait_alignment,
    plot_all_alignment,
    save_alignment,
)
from src_dev.psychometric.tucker_congruence import (
    align_factors,
    classify_phi,
    tucker_phi_matrix,
)
from src_dev.psychometric.combine import load_pair_outputs
from src_dev.psychometric.factor_extremes_html import export_factor_extremes_html
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

# ── Block filter ────────────────────────────────────────────────────────────
# When set (via ``--block-filter`` CLI arg in main()), restrict the
# questionnaire to items whose ``block`` matches. Used to run a parallel
# analysis on just one modality (e.g. ``likert`` — v5 Likert only,
# dropping the trait_mcq block) without touching the main paper analysis.
# OUTPUT_ROOT and LABELS_REPO_DIR get an ``_<filter>`` suffix so the
# filtered run writes to a separate tree and doesn't collide.
BLOCK_FILTER: str | None = None


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


# ── HTML factor browser ─────────────────────────────────────────────────────
# The factor-extremes HTML viewer pulls rollout conversations from a
# `exports/conversation_training.jsonl` inside the rollout dir. Both models
# share the same underlying B rollout cache (only the questionnaire model
# differs), so a single rollout dir feeds both models' HTMLs.
ROLLOUT_DIR_FOR_HTML: Path = Path(
    "scratch/psychometric_fa/"
    "rollouts-llama318binstruct-t1.0-15t-2500p-seed436-scenarios_v2-uprompt_v6"
)

# Number of high-scoring and low-scoring personas shown per factor in the
# HTML browser. 10 per pole is what the main pipeline uses and is about as
# much as a human wants to scroll through at once.
HTML_N_PER_POLE: int = 10

# Raw questionnaire JSON paths used to enrich items.json before rendering
# the factor-extremes HTML. The combined items.json doesn't carry MCQ
# options or answer_mapping, so we match by bare item id back to the raw
# source and splice the extra fields in. Likert's `reverse_keyed` already
# rides along on items.json but we still look it up here for uniformity.
RAW_QUESTIONNAIRE_PATHS: dict[str, Path] = {
    "v5": Path("datasets/psychometric_questionnaires/psychometric_questionnaire_v5.json"),
    "trait_ocean_natural_v1": Path(
        "datasets/psychometric_questionnaires/trait_ocean_natural_v1.json"
    ),
}

# ── Within-model validation ─────────────────────────────────────────────────
# Cronbach's α: for each factor, restrict to items whose |loading| exceeds
# this threshold (the standard psychometric "defining items" subset), sign-
# orient by loading direction, and compute α. 0.4 is the conventional
# "salient loading" threshold (Comrey & Lee 1992); lower thresholds drag
# in weakly-loaded items that don't actually belong to the construct.
CRONBACH_LOADING_THRESHOLD: float = 0.4

# Split-half congruence: N iterations of random half-splits, per-factor
# Tucker's |φ| distribution. ~100 iters is enough to stabilise the
# violin plots without the cost of bootstrap's resample-refit loops.
SPLIT_HALF_N_ITERS: int = 100
SPLIT_HALF_PASS_THRESHOLD_PHI: float = 0.85  # Lorenzo-Seva "fair" threshold


# ── Predictivity cross-validation (bi-cross-validation) ─────────────────────
# Double-loop persona × item holdout — the well-founded generalisation test
# for factor models (Owen & Perry 2009; Bro et al. 2008; Wold 1978).
#
# Protocol (one outer split):
#   1. Persona A = random 80%; persona B = remaining 20%.
#   2. Fit FA on the full column set of persona A → loadings Λ_A.
#   3. For each B-persona, randomly observe m_observed items; compute
#      Thomson factor scores using Λ_A restricted to those m items.
#   4. Predict the remaining ("held-out") items from F̂ · Λ_A.T.
#   5. Collect per-item R² against three null baselines:
#      item_mean / persona_shuffle / k_minus_1.
#
# Per outer split we also compute Tucker's |φ| between Λ_A and the full-
# data Λ (restricted to shared items) — a direct "are the CV-fold factors
# the same factors we labelled?" check.
#
# Defaults: 20 outer persona splits × 5 item resamples = 100 estimates,
# ~enough for stable mean R² with 95% bootstrap CI.
PREDICTIVITY_PERSONA_SPLIT: float = 0.8     # fraction in subset A (train)
PREDICTIVITY_ITEM_OBSERVED_FRAC: float = 0.8  # fraction of items observed
                                              # for B personas; drives m_observed
PREDICTIVITY_N_OUTER_SPLITS: int = 20
PREDICTIVITY_N_TRIALS: int = 5
PREDICTIVITY_BOOTSTRAP_CI: float = 95.0
PREDICTIVITY_SUBSET_STRATEGY: str = "random"  # "random" | "by_factor_balanced"


# ── Cross-model congruence ──────────────────────────────────────────────────
# Which model is the anchor in the Hungarian match. Llama is the paper's
# headline fit (k=4), so we treat its factors as the reference and ask how
# each maps into Qwen's k=5 solution. Under Hungarian matching, all 4
# Llama factors get one partner and 1 of Qwen's 5 is left unmatched —
# that extra factor is reported separately with its single best partner.
CROSS_MODEL_ANCHOR: str = "llama-3.1-8b"

# Blocks to run cross-model congruence on. "combined" uses all shared
# items; the per-block restrictions tell us whether agreement is
# Likert-driven, trait_mcq-driven, or mixed — same decomposition as the
# existing cross_model_sweep.
CROSS_MODEL_BLOCK_SUBSETS: tuple[str, ...] = ("combined", "likert", "trait_mcq")

# Persisted-labels store (checked into the repo, survives gitignored
# scratch wipes). The /label-fa-factors skill writes into
# ``{output_dir}/labeling/`` under scratch — after a labelling session,
# copy the fresh ``llm_labels_*.json`` over to the matching subpath
# under ``LABELS_REPO_DIR`` and commit so the labels persist. On each
# run, ``export_html_browser`` seeds scratch from this dir so labels
# are applied to the freshly-built HTML even if scratch was cleaned.
#
# Re-running with the same seeds produces identical loadings (the FA
# pipeline: np.random seeded + factor_analyzer's randomized_svd uses
# a hardcoded random_state + oblimin is deterministic), so labels keyed
# by factor_index stay valid across re-runs.
LABELS_REPO_DIR: Path = Path("datasets/psychometric_fa_labels/analysis_for_paper")


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
    """Hydrate (if needed) and load one model's combined response matrix.

    Column defs are enriched at this point from the raw questionnaire JSONs
    (options + answer_mapping for trait_mcq, reverse_keyed for Likert) so
    every downstream consumer — FA fit, item_labels sidecar, top-30 text
    summary, HTML browser — works off the same rich payload. Notably this
    also means the ``encoding`` field carried on the combined items.json
    survives into the sidecar, avoiding the
    ``_item_labels.json → col_def.get("encoding", "letter_1-4")`` default
    in the /label-fa-factors skill that silently mis-labels trait_mcq
    items as letter-ordinal when the real encoding is ``trait_aligned_0-1``.
    """
    q_dir = _questionnaire_dir_for(model)
    raw_matrix, raw_meta, raw_items = load_pair_outputs(q_dir)
    log.info(
        "[%s] raw matrix: %d personas × %d items",
        model.slug, raw_matrix.shape[0], raw_matrix.shape[1],
    )

    # Optional block filter — drops every column whose item's `block` does
    # not match `BLOCK_FILTER`. Applied at the raw matrix level, before
    # preprocessing, so downstream steps (variance filter, FA, alignment)
    # see only the filtered block.
    if BLOCK_FILTER is not None:
        keep_mask = np.array(
            [it.get("block") == BLOCK_FILTER for it in raw_items],
            dtype=bool,
        )
        raw_matrix = raw_matrix[:, keep_mask]
        raw_items = [it for it, keep in zip(raw_items, keep_mask) if keep]
        log.info(
            "[%s] block filter %r kept %d / %d columns",
            model.slug, BLOCK_FILTER,
            int(keep_mask.sum()), int(keep_mask.size),
        )

    raw_items = _enrich_column_defs(raw_items)

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
        # Only write paper figures from the unfiltered (main paper) run —
        # filtered runs are robustness variants and shouldn't overwrite
        # the headline figure.
        paper_rel = (
            PAPER_SCREE_FIGURES.get(data.model.slug)
            if BLOCK_FILTER is None else None
        )
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

    Extended version of the main pipeline's ``fa_*_item_labels.json`` schema
    — adds ``encoding`` (so the /label-fa-factors skill doesn't default to
    the misleading ``"letter_1-4"`` fallback at ``fa_label_tools.py:548``)
    plus ``options`` + ``answer_mapping`` for trait_mcq columns so the
    skill's describe output can render the full option table without
    having to walk up to find the raw questionnaire file.
    """
    payload: list[dict] = []
    for col in items:
        entry: dict = {
            "col_id": col["col_id"],
            "text": col.get("text", ""),
            "block": col.get("block", ""),
            "dimension": col.get("dimension"),
            "reverse_keyed": col.get("reverse_keyed", False),
        }
        if col.get("encoding"):
            entry["encoding"] = col["encoding"]
        if col.get("options"):
            entry["options"] = col["options"]
        if col.get("answer_mapping"):
            entry["answer_mapping"] = col["answer_mapping"]
        payload.append(entry)
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
    enc = item.get("encoding") or ""
    enc_tag = f" enc={enc}" if enc else ""
    header = f"    {loading:+.3f}  <{block}{dim_tag}{enc_tag}>{rev}  {text}"
    if block == "trait_mcq" and item.get("options") and item.get("answer_mapping"):
        # High pole of THIS factor for the item: options with trait-aligned
        # answer_mapping under +loading, or trait-opposite under −loading.
        amap = item["answer_mapping"]
        high_pole_is_trait_aligned = loading >= 0
        lines = [header]
        for opt in item["options"]:
            lbl = str(opt.get("label", ""))
            mv = int(amap.get(lbl, 0))
            trait_aligned = (mv == 1)
            is_high_pole = trait_aligned if high_pole_is_trait_aligned else not trait_aligned
            pole = "HIGH" if is_high_pole else "LOW "
            opt_text = str(opt.get("text", "")).replace("\n", " ").strip()
            lines.append(f"          {lbl}. [{pole} pole]  {opt_text}")
        return "\n".join(lines)
    return header


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
# WITHIN-MODEL VALIDATION
# ═════════════════════════════════════════════════════════════════════════════


def _cronbach_alpha_per_factor(
    matrix: np.ndarray,
    loadings: np.ndarray,
    items: list[dict],
    *,
    loading_threshold: float,
) -> list[dict]:
    """Compute Cronbach's α for each factor, over items with |loading| ≥ threshold.

    Items are sign-oriented by loading direction so positively- and
    negatively-loading items both contribute positively to the summed score
    — without this, α is meaningless when a factor has mixed-sign loadings.

    Returns one dict per factor with α, the classified α (excellent / good /
    acceptable / questionable / poor / redundant), the number of items used,
    and the per-block item counts so we can tell at a glance whether α is
    being driven by Likert items, trait_mcq items, or a mix.
    """
    n_items, n_factors = loadings.shape
    rows: list[dict] = []
    for f in range(n_factors):
        col = loadings[:, f]
        mask = np.abs(col) >= loading_threshold
        idxs = np.flatnonzero(mask)
        if idxs.size < 2:
            rows.append({
                "factor_index": f,
                "n_items": int(idxs.size),
                "alpha": float("nan"),
                "alpha_class": "n/a",
                "threshold": loading_threshold,
                "item_col_ids": [items[i].get("col_id") for i in idxs.tolist()],
                "block_counts": {},
                "note": "fewer than 2 salient items",
            })
            continue
        subset = matrix[:, idxs]
        signs = np.sign(col[idxs])
        signs[signs == 0] = 1.0  # pathological guard; |loading|≥0.4 excludes zeros
        alpha = cronbach_alpha(subset, loading_signs=signs)
        block_counts: dict[str, int] = {}
        for i in idxs.tolist():
            b = str(items[i].get("block", "?"))
            block_counts[b] = block_counts.get(b, 0) + 1
        rows.append({
            "factor_index": f,
            "n_items": int(idxs.size),
            "alpha": float(alpha),
            "alpha_class": classify_alpha(alpha),
            "threshold": loading_threshold,
            "item_col_ids": [items[i].get("col_id") for i in idxs.tolist()],
            "block_counts": block_counts,
        })
    return rows


def _load_labels_for_slug(model_slug: str, n_factors: int) -> dict[int, str]:
    """Best-effort ``{factor_index -> axis_name}`` from the persisted labels store.

    Picks the newest ``llm_labels_*.json`` under ``LABELS_REPO_DIR/<slug>/
    labeling/`` by mtime and returns the axis_name per factor. Returns
    ``{}`` when no label file exists — callers should fall back to
    ``F{idx}`` placeholders.
    """
    labeling_dir = LABELS_REPO_DIR / model_slug / "labeling"
    if not labeling_dir.is_dir():
        return {}
    candidates = sorted(
        labeling_dir.glob("llm_labels_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and "factors" in payload:
            payload = payload["factors"]
        if not isinstance(payload, list):
            continue
        out: dict[int, str] = {}
        for entry in payload:
            fi = entry.get("factor_index")
            ax = entry.get("axis_name")
            if isinstance(fi, int) and 0 <= fi < n_factors and ax:
                out[fi] = str(ax)
        if out:
            return out
    return {}


def run_within_model_validation(
    data: LoadedData, fit: FaFit,
) -> dict:
    """Cronbach's α per factor + split-half congruence, written under validation/.

    Two tests, both loadings-based and within-model-only:

    * Cronbach's α — do the items that cluster onto a factor actually
      predict each other? One α per factor, conventional thresholds in
      ``classify_alpha``. Saved as ``validation/cronbach_alpha.json`` with
      per-factor α + sign-oriented item list + per-block item counts.

    * Split-half congruence — is the factor structure replicable within
      the persona sample? For each of N random half-splits, refit FA on
      each half and compute per-factor Tucker's |φ|. Pass threshold 0.85
      (Lorenzo-Seva "fair similarity"). Saved as
      ``validation/split_half_congruence.{json,png}``.
    """
    out_dir = data.output_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    fa_result = load_factor_analysis(fit.npz_path)
    loadings = fa_result["loadings"]

    log.info("[%s] Cronbach's α per factor (|loading|≥%.2f)…",
             data.model.slug, CRONBACH_LOADING_THRESHOLD)
    alpha_rows = _cronbach_alpha_per_factor(
        data.matrix, loadings, data.items,
        loading_threshold=CRONBACH_LOADING_THRESHOLD,
    )
    alpha_payload = {
        "model_slug": data.model.slug,
        "model_label": data.model.label,
        "n_factors": fit.k,
        "rotation": fit.rotation,
        "loading_threshold": CRONBACH_LOADING_THRESHOLD,
        "per_factor": alpha_rows,
    }
    (out_dir / "cronbach_alpha.json").write_text(
        json.dumps(alpha_payload, indent=2)
    )
    log.info("[%s] wrote %s", data.model.slug, out_dir / "cronbach_alpha.json")

    for row in alpha_rows:
        log.info(
            "  F%d: α=%.3f (%s)  n_items=%d  blocks=%s",
            row["factor_index"], row["alpha"], row["alpha_class"],
            row["n_items"], row["block_counts"],
        )

    log.info(
        "[%s] Split-half congruence (%d iterations, k=%d, %s)…",
        data.model.slug, SPLIT_HALF_N_ITERS, fit.k, fit.rotation,
    )
    split_half = split_half_congruence(
        data.matrix,
        n_factors=fit.k,
        out_dir=out_dir,                       # writes split_half_congruence.{json,png}
        n_iters=SPLIT_HALF_N_ITERS,
        fa_method=fit.method,
        rotation=fit.rotation,
        align="procrustes",                    # oblimin-compatible alignment
        seed=SEED,
        pass_threshold_median_phi=SPLIT_HALF_PASS_THRESHOLD_PHI,
        verbose=False,
    )

    return {
        "cronbach_alpha": alpha_payload,
        "split_half_congruence": split_half,
    }


# ═════════════════════════════════════════════════════════════════════════════
# PREDICTIVITY CROSS-VALIDATION (BI-CV)
# ═════════════════════════════════════════════════════════════════════════════


def run_predictivity_cv(data: LoadedData, fit: FaFit) -> dict:
    """Bi-cross-validation: predict held-out persona × item responses.

    Thin wrapper around
    :func:`src_dev.factor_analysis.cross_validation.persona_item_cv`, which
    implements the Owen-Perry / Bro-et-al bi-CV scheme with three principled
    nulls (``item_mean`` / ``persona_shuffle`` / ``k_minus_1``). The main
    statistic is per-item R² pooled over the outer × trial loop; the
    shuffle baseline is the null for "is this genuinely persona-predictive".

    Output: ``{output_dir}/validation/predictivity_cv.{json,png}`` (persona_
    item_cv writes these); main() also prints a compact summary table.
    """
    out_dir = data.output_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "[%s] predictivity CV: %d outer × %d trials, A=%.0f%%, m_observed≈%.0f%% items",
        data.model.slug,
        PREDICTIVITY_N_OUTER_SPLITS,
        PREDICTIVITY_N_TRIALS,
        100 * PREDICTIVITY_PERSONA_SPLIT,
        100 * PREDICTIVITY_ITEM_OBSERVED_FRAC,
    )
    m_observed = max(
        3 * fit.k,
        int(round(PREDICTIVITY_ITEM_OBSERVED_FRAC * data.matrix.shape[1])),
    )
    result = persona_item_cv(
        data.matrix,
        data.metadata,
        n_factors=fit.k,
        out_dir=out_dir,
        persona_split=PREDICTIVITY_PERSONA_SPLIT,
        stratify=None,                          # B preset has 1 rollout / persona
        m_observed=m_observed,
        subset_strategy=PREDICTIVITY_SUBSET_STRATEGY,
        n_trials=PREDICTIVITY_N_TRIALS,
        n_outer_splits=PREDICTIVITY_N_OUTER_SPLITS,
        bootstrap_ci=PREDICTIVITY_BOOTSTRAP_CI,
        fa_method=fit.method,
        rotation=fit.rotation,
        seed=SEED,
        verbose=False,
    )

    # Fold-vs-full Tucker's is intentionally omitted here — the existing
    # split_half_congruence step (50/50 splits × 100 iters) already
    # establishes that the factors are stable under resampling, and an
    # 80/20 variant would not add a materially different claim.
    log.info(
        "[%s] predictivity CV: main R²=%.3f  shuffle=%.3f  k-1=%.3f  item-mean=%.3f",
        data.model.slug,
        result.get("mean_r2_main", float("nan")),
        result.get("mean_r2_persona_shuffle", float("nan")),
        result.get("mean_r2_k_minus_1", float("nan")),
        result.get("mean_r2_item_mean", float("nan")),
    )
    return result


# ═════════════════════════════════════════════════════════════════════════════
# CROSS-MODEL CONGRUENCE
# ═════════════════════════════════════════════════════════════════════════════


def _slice_loadings_to_shared_items(
    loadings_a: np.ndarray,
    items_a: list[dict],
    loadings_b: np.ndarray,
    items_b: list[dict],
    *,
    block_filter: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Intersect two fits by col_id and return aligned loadings + shared IDs.

    Preprocessing drops different low-variance items per model (86/100
    trait_mcq kept on Llama, 71/100 on Qwen), so each model's loadings
    sit on a slightly different item set. Tucker's φ requires matching
    item indices, so we intersect on col_id and reindex both sides.

    ``block_filter="likert"`` / ``"trait_mcq"`` restricts the shared set
    to that block; ``None`` uses everything shared. Returning the col_ids
    lets the caller label rows on the eventual heatmap.
    """
    ids_a = [it["col_id"] for it in items_a]
    ids_b = [it["col_id"] for it in items_b]
    idx_a = {cid: i for i, cid in enumerate(ids_a)}
    idx_b = {cid: i for i, cid in enumerate(ids_b)}
    shared = sorted(set(ids_a) & set(ids_b))
    if block_filter is not None:
        block_a = {it["col_id"]: it.get("block") for it in items_a}
        shared = [c for c in shared if block_a.get(c) == block_filter]
    if not shared:
        raise ValueError(
            f"No shared items between fits (block_filter={block_filter!r})."
        )
    la = loadings_a[[idx_a[c] for c in shared], :]
    lb = loadings_b[[idx_b[c] for c in shared], :]
    return la, lb, shared


def _plot_phi_heatmap(
    abs_phi: np.ndarray,
    *,
    anchor_labels: list[str],
    target_labels: list[str],
    matched_pairs: list[tuple[int, int, float]],
    anchor_name: str,
    target_name: str,
    subset_label: str,
    n_shared: int,
    save_path: Path,
) -> None:
    """|φ| heatmap with matched cells outlined — one PDF per subset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ka, kb = abs_phi.shape
    fig, ax = plt.subplots(figsize=(max(4, 0.9 * kb + 2), max(3.5, 0.8 * ka + 2)))
    im = ax.imshow(abs_phi, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")

    # Overlay matched cells with white outlines.
    for fa_, fb_, _ in matched_pairs:
        ax.add_patch(plt.Rectangle(
            (fb_ - 0.5, fa_ - 0.5), 1, 1,
            fill=False, edgecolor="white", linewidth=2.2,
        ))

    for i in range(ka):
        for j in range(kb):
            v = abs_phi[i, j]
            ax.text(
                j, i, f"{v:.2f}",
                ha="center", va="center",
                color="white" if v < 0.55 else "black", fontsize=9,
            )
    ax.set_xticks(range(kb)); ax.set_xticklabels(target_labels, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(ka)); ax.set_yticklabels(anchor_labels, fontsize=9)
    ax.set_xlabel(target_name)
    ax.set_ylabel(anchor_name)
    ax.set_title(
        f"Tucker's |φ|  ({subset_label}, n_shared={n_shared})",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label="|φ|")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    fig.savefig(save_path.with_suffix(".pdf"))
    plt.close(fig)


def _axis_labels_for(slug: str, n_factors: int) -> list[str]:
    """Prefer labelled axis names; fall back to ``F{idx}``."""
    labels = _load_labels_for_slug(slug, n_factors)
    return [
        f"F{i}" + (f" ({labels[i]})" if i in labels else "")
        for i in range(n_factors)
    ]


def _run_one_cross_model_subset(
    *,
    anchor_slug: str,
    target_slug: str,
    anchor_loadings: np.ndarray,
    target_loadings: np.ndarray,
    anchor_items: list[dict],
    target_items: list[dict],
    subset_label: str,
    block_filter: str | None,
    save_dir: Path,
) -> dict:
    """Run Tucker's φ for one (anchor, target, block-subset) triple."""
    la, lb, shared = _slice_loadings_to_shared_items(
        anchor_loadings, anchor_items,
        target_loadings, target_items,
        block_filter=block_filter,
    )

    signed_phi = tucker_phi_matrix(la, lb, signed=True)
    abs_phi = np.abs(signed_phi)

    alignments = align_factors(la, lb)
    matched_pairs: list[tuple[int, int, float]] = []
    matched_targets: set[int] = set()
    alignment_rows: list[dict] = []
    for a in alignments:
        if a.target_factor >= 0:
            matched_pairs.append((a.anchor_factor, a.target_factor, a.phi))
            matched_targets.add(a.target_factor)
        alignment_rows.append({
            "anchor_factor": a.anchor_factor,
            "target_factor": a.target_factor,
            "phi": a.phi,
            "phi_signed": (
                float(signed_phi[a.anchor_factor, a.target_factor])
                if a.target_factor >= 0 else float("nan")
            ),
            "sign": a.sign,
            "classification": classify_phi(a.phi),
        })

    # Unmatched target factors (when kb > ka): report each with its best
    # anchor partner, so we don't silently drop them.
    unmatched_target_rows: list[dict] = []
    k_a, k_b = la.shape[1], lb.shape[1]
    for fb_ in range(k_b):
        if fb_ in matched_targets:
            continue
        col = abs_phi[:, fb_]
        best_anchor = int(np.argmax(col))
        phi_val = float(col[best_anchor])
        unmatched_target_rows.append({
            "target_factor": fb_,
            "best_anchor_partner": best_anchor,
            "phi": phi_val,
            "classification": classify_phi(phi_val),
        })

    anchor_labels = _axis_labels_for(anchor_slug, la.shape[1])
    target_labels = _axis_labels_for(target_slug, lb.shape[1])

    _plot_phi_heatmap(
        abs_phi,
        anchor_labels=anchor_labels,
        target_labels=target_labels,
        matched_pairs=matched_pairs,
        anchor_name=anchor_slug, target_name=target_slug,
        subset_label=subset_label,
        n_shared=len(shared),
        save_path=save_dir / "phi_heatmap.png",
    )

    np.save(save_dir / "phi_matrix.npy", signed_phi)

    report = {
        "anchor": anchor_slug,
        "target": target_slug,
        "subset": subset_label,
        "block_filter": block_filter,
        "n_shared_items": len(shared),
        "shared_col_ids": shared,
        "k_anchor": int(la.shape[1]),
        "k_target": int(lb.shape[1]),
        "matched": alignment_rows,
        "unmatched_target": unmatched_target_rows,
        "overall_mean_matched_phi": float(
            np.mean([r["phi"] for r in alignment_rows if r["target_factor"] >= 0])
        ) if alignment_rows else float("nan"),
    }
    (save_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def run_cross_model_congruence(
    loaded: dict[str, LoadedData],
    fits: dict[str, "FaFit"],
) -> dict[str, dict]:
    """Llama ↔ Qwen Tucker's φ across combined + per-block shared items.

    Output layout (top-level, not per-model — this is a cross-model comparison):

        scratch/psychometric_fa_paper/cross_model/
            {anchor}_vs_{target}/
                {combined,likert,trait_mcq}/
                    phi_matrix.npy
                    phi_heatmap.{png,pdf}
                    report.json
                summary.json      # flattened per-subset key metrics

    Hungarian matching is run with ``CROSS_MODEL_ANCHOR`` as anchor; the
    anchor's ``k_anchor`` factors all get a target partner (via
    ``scipy.optimize.linear_sum_assignment``), and any leftover target
    factors (when ``k_target > k_anchor``) are reported separately with
    their single-best anchor partner.
    """
    if CROSS_MODEL_ANCHOR not in fits:
        log.warning("anchor model %s not in fits; skipping cross-model pass",
                    CROSS_MODEL_ANCHOR)
        return {}
    targets = [s for s in fits if s != CROSS_MODEL_ANCHOR]
    if not targets:
        log.warning("only one model fit available; skipping cross-model pass")
        return {}

    anchor = fits[CROSS_MODEL_ANCHOR]
    anchor_data = loaded[CROSS_MODEL_ANCHOR]
    anchor_fa = load_factor_analysis(anchor.npz_path)

    all_reports: dict[str, dict] = {}
    for target_slug in targets:
        target = fits[target_slug]
        target_data = loaded[target_slug]
        target_fa = load_factor_analysis(target.npz_path)

        pair_label = f"{CROSS_MODEL_ANCHOR}_vs_{target_slug}"
        pair_dir = OUTPUT_ROOT / "cross_model" / pair_label
        pair_dir.mkdir(parents=True, exist_ok=True)
        log.info("═══ cross-model: %s ═══", pair_label)

        subset_reports: dict[str, dict] = {}
        summary_rows: list[dict] = []
        for subset in CROSS_MODEL_BLOCK_SUBSETS:
            subset_dir = pair_dir / subset
            subset_dir.mkdir(parents=True, exist_ok=True)
            block_filter = None if subset == "combined" else subset
            try:
                report = _run_one_cross_model_subset(
                    anchor_slug=CROSS_MODEL_ANCHOR,
                    target_slug=target_slug,
                    anchor_loadings=anchor_fa["loadings"],
                    target_loadings=target_fa["loadings"],
                    anchor_items=anchor_data.items,
                    target_items=target_data.items,
                    subset_label=subset,
                    block_filter=block_filter,
                    save_dir=subset_dir,
                )
            except ValueError as exc:
                log.warning("  [%s/%s] skipped: %s", pair_label, subset, exc)
                continue
            subset_reports[subset] = report
            summary_rows.append({
                "subset": subset,
                "n_shared_items": report["n_shared_items"],
                "overall_mean_matched_phi": report["overall_mean_matched_phi"],
                "matched_phis": [r["phi"] for r in report["matched"]],
                "n_unmatched_target": len(report["unmatched_target"]),
            })

            log.info(
                "  [%s/%-9s] n_shared=%3d  mean |φ|=%.3f  "
                "per_pair=[%s]",
                pair_label, subset, report["n_shared_items"],
                report["overall_mean_matched_phi"],
                ", ".join(f"{r['phi']:.2f}" for r in report["matched"]),
            )

        (pair_dir / "summary.json").write_text(
            json.dumps({"pair": pair_label, "subsets": summary_rows}, indent=2)
        )
        all_reports[pair_label] = {"subsets": subset_reports}
    return all_reports


# ═════════════════════════════════════════════════════════════════════════════
# HTML FACTOR BROWSER
# ═════════════════════════════════════════════════════════════════════════════


def _seed_labeling_from_repo(model_slug: str, scratch_labeling_dir: Path) -> None:
    """Copy persisted labels from the in-repo store into scratch/labeling/.

    Only copies files not already present in scratch — so a fresh skill
    write in scratch (newer mtime) always wins on the ``load_latest_non
    empty_llm_labels`` lookup, and this function never shadows an
    in-progress labelling session. ``shutil.copy2`` preserves mtimes so
    the cross-file ordering (when multiple candidates exist) reflects
    when the labels were actually written, not when the sync ran.
    """
    import shutil

    repo_labeling = LABELS_REPO_DIR / model_slug / "labeling"
    if not repo_labeling.is_dir():
        return
    n_synced = 0
    for src in repo_labeling.glob("llm_labels_*.json"):
        dst = scratch_labeling_dir / src.name
        if dst.exists():
            continue
        shutil.copy2(src, dst)
        n_synced += 1
    if n_synced:
        log.info(
            "[%s] synced %d persisted label file(s) from %s into %s",
            model_slug, n_synced, repo_labeling, scratch_labeling_dir,
        )


def _load_raw_questionnaire_index() -> dict[tuple[str, str], dict]:
    """Build a lookup `(questionnaire_version, bare_id) -> raw_item`.

    Raw items for trait_mcq carry ``options`` + ``answer_mapping``; raw
    items for Likert carry ``reverse_keyed``. ``col_id`` in items.json is
    namespaced as ``{version}/{bare_id}`` — we split that when looking up.
    """
    idx: dict[tuple[str, str], dict] = {}
    for version, path in RAW_QUESTIONNAIRE_PATHS.items():
        if not path.exists():
            log.warning("raw questionnaire missing: %s", path)
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Likert (v5-style): top-level "items"
        for raw_item in raw.get("items", []) or []:
            idx[(version, str(raw_item["id"]))] = raw_item
        # Trait-MCQ (trait_ocean_*-style): block_4_trait_mcq.items
        block = raw.get("block_4_trait_mcq", {})
        for raw_item in block.get("items", []) or []:
            idx[(version, str(raw_item["id"]))] = raw_item
    return idx


def _enrich_column_defs(column_defs: list[dict]) -> list[dict]:
    """Splice trait-mcq options/answer_mapping + reverse-keyed onto column defs.

    Doesn't mutate the inputs — returns a new list of dicts with extra
    fields populated where a raw questionnaire entry is available.
    """
    raw_index = _load_raw_questionnaire_index()
    enriched: list[dict] = []
    for cdef in column_defs:
        out = dict(cdef)
        col_id = str(cdef.get("col_id", ""))
        # col_id looks like "v5/0" or "trait_ocean_natural_v1/openness_trait_xxxx"
        version, _, bare = col_id.partition("/")
        raw_item = raw_index.get((version, bare))
        if raw_item is None:
            enriched.append(out)
            continue
        if "reverse_keyed" not in out and "reverse_keyed" in raw_item:
            out["reverse_keyed"] = bool(raw_item["reverse_keyed"])
        if "options" in raw_item and "options" not in out:
            out["options"] = raw_item["options"]
        if "answer_mapping" in raw_item and "answer_mapping" not in out:
            out["answer_mapping"] = raw_item["answer_mapping"]
        enriched.append(out)
    return enriched


def export_html_browser(data: LoadedData, fit: FaFit) -> Path | None:
    """Write an interactive HTML factor browser for one FA fit.

    Pulls rollout conversations from ``ROLLOUT_DIR_FOR_HTML`` (both models
    share the same B-rollout cache) and lays out, per factor, the top-N and
    bottom-N personas by score with their conversation transcripts. If a
    ``labeling/llm_labels_{analysis_key}_manual_*.json`` file is present
    (produced by the ``/label-fa-factors`` skill), the labels show up in the
    browser automatically; otherwise the exporter falls back to top-loading
    items.

    Output: ``{output_dir}/factor_analysis/raw_{rotation}/factor_extremes.html``
    (matches the existing pipeline's layout).

    Returns the HTML path (or None if the rollout export was missing).
    """
    if not ROLLOUT_DIR_FOR_HTML.exists():
        log.warning(
            "[%s] rollout dir missing — skipping HTML export (%s)",
            data.model.slug, ROLLOUT_DIR_FOR_HTML,
        )
        return None

    rotation = fit.rotation
    analysis_key = f"raw_{rotation}"  # label the skill uses; also the HTML label

    save_dir = data.output_dir / "factor_analysis" / analysis_key
    save_dir.mkdir(parents=True, exist_ok=True)

    labeling_dir = data.output_dir / "labeling"
    labeling_dir.mkdir(parents=True, exist_ok=True)
    _seed_labeling_from_repo(data.model.slug, labeling_dir)

    fa_result = load_factor_analysis(fit.npz_path)
    enriched_items = _enrich_column_defs(data.items)

    log.info(
        "[%s] exporting factor_extremes.html → %s",
        data.model.slug, save_dir / "factor_extremes.html",
    )
    export_factor_extremes_html(
        fa_result=fa_result,
        column_defs=enriched_items,
        metadata=data.metadata,
        label=analysis_key,
        save_dir=save_dir,
        rollout_dirs=[ROLLOUT_DIR_FOR_HTML],
        labeling_dir=labeling_dir,
        n_per_pole=HTML_N_PER_POLE,
    )
    return save_dir / "factor_extremes.html"


# ═════════════════════════════════════════════════════════════════════════════
# PAPER FIGURES
# ═════════════════════════════════════════════════════════════════════════════


def _plot_paper_within_model_validation(
    validation_results: dict[str, dict],
) -> Path | None:
    """Grouped-bars paper figure: Cronbach's α + split-half median |φ| per factor.

    Two panels side-by-side (α on the left, |φ| on the right) with both
    models' factors grouped by index. Axis labels come from the newest
    persisted labels file when available. Reference lines at α=0.70/0.80
    and |φ|=0.85. Writes to ``PAPER_FIGURES_DIR / unsupervised / fig_4_2_2_
    within_model_validation.pdf``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Collect per-model series. Pad the shorter to the longer so the x-axis
    # lines up when models have different k.
    rows = []
    for slug, res in validation_results.items():
        alpha_rows = res["cronbach_alpha"]["per_factor"]
        sh = res["split_half_congruence"]
        medians = sh.get("median_phi_per_factor", [])
        labels = _load_labels_for_slug(slug, len(alpha_rows))
        for i, r in enumerate(alpha_rows):
            rows.append({
                "slug": slug,
                "factor_index": i,
                "axis": labels.get(i, f"F{i}"),
                "alpha": r["alpha"],
                "phi_median": medians[i] if i < len(medians) else float("nan"),
            })

    slugs = list(validation_results.keys())
    max_k = max(r["factor_index"] for r in rows) + 1
    colours = {slugs[0]: "#2563eb", slugs[1]: "#ea580c"} if len(slugs) >= 2 else {
        slugs[0]: "#2563eb",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for panel_idx, (ax, field, ref_lines, title) in enumerate((
        (axes[0], "alpha",
         [(0.70, "acceptable", "#f59e0b"), (0.80, "good", "#16a34a")],
         "Cronbach's α per factor (|loading| ≥ 0.4, sign-oriented)"),
        (axes[1], "phi_median",
         [(0.85, "fair (Lorenzo-Seva)", "#16a34a")],
         "Split-half median |φ| per factor (100 iters)"),
    )):
        width = 0.36
        x = np.arange(max_k)
        for i, slug in enumerate(slugs):
            offset = (i - (len(slugs) - 1) / 2) * width
            heights = [float("nan")] * max_k
            axis_labels = [""] * max_k
            for r in rows:
                if r["slug"] == slug:
                    heights[r["factor_index"]] = r[field]
                    axis_labels[r["factor_index"]] = r["axis"]
            bars = ax.bar(
                x + offset, heights, width,
                color=colours[slug], alpha=0.88, label=slug,
                edgecolor="#111", linewidth=0.3,
            )
            for rect, h, lab in zip(bars, heights, axis_labels):
                if np.isnan(h):
                    continue
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    h + 0.012,
                    f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color="#111",
                )
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    -0.03,
                    lab,
                    ha="center", va="top", fontsize=6.8, rotation=40,
                    color="#374151",
                )
        for thresh, lbl, col in ref_lines:
            ax.axhline(thresh, linestyle="--", color=col, linewidth=0.9, alpha=0.8,
                       label=f"{lbl} ({thresh:.2f})")
        ax.set_xticks(x)
        ax.set_xticklabels([f"F{k}" for k in range(max_k)])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(r"$\alpha$" if field == "alpha" else r"median $|\phi|$")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
        # Leave headroom at the bottom so the rotated axis labels don't clip.
        ax.set_ylim(bottom=-0.18)

    save_path = PAPER_FIGURES_DIR / "unsupervised" / "fig_4_2_2_within_model_validation.pdf"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", save_path)
    return save_path


def _copy_paper_cross_model_heatmap(
    cross_reports: dict[str, dict],
) -> Path | None:
    """Copy the combined-subset Tucker's |φ| heatmap to paper/figures/.

    We keep the PDF emitted by _plot_phi_heatmap and just move/copy it to
    the canonical paper path — no need to re-render.
    """
    import shutil
    # Pick the first pair's "combined" heatmap — there's only one pair at
    # present (Llama vs Qwen) but this stays robust if we ever add more.
    for pair_label, bundle in cross_reports.items():
        subsets = bundle.get("subsets", {})
        combined = subsets.get("combined")
        if combined is None:
            continue
        src = (OUTPUT_ROOT / "cross_model" / pair_label / "combined"
               / "phi_heatmap.pdf")
        if not src.exists():
            continue
        dst = (PAPER_FIGURES_DIR / "unsupervised"
               / "fig_4_2_3_cross_model_phi.pdf")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        log.info("wrote %s", dst)
        return dst
    return None


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


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Section-4.2 FA analysis pipeline for the paper.",
    )
    ap.add_argument(
        "--block-filter",
        default=None,
        choices=("likert", "trait_mcq"),
        help=(
            "Restrict the questionnaire to items in this block before "
            "preprocessing and FA. Appends `_<block>` to OUTPUT_ROOT and "
            "LABELS_REPO_DIR so the filtered run writes to a separate "
            "tree alongside the main paper analysis."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    # Promote CLI args into the module-level globals the step functions
    # read from. This keeps the "config-at-top" style while allowing a
    # parallel run (e.g. ``--block-filter likert``) to write to a different
    # tree without the user having to edit the file.
    global BLOCK_FILTER, OUTPUT_ROOT, LABELS_REPO_DIR
    if args.block_filter is not None:
        BLOCK_FILTER = args.block_filter
        OUTPUT_ROOT = Path(str(OUTPUT_ROOT) + f"_{BLOCK_FILTER}")
        LABELS_REPO_DIR = Path(
            str(LABELS_REPO_DIR) + f"_{BLOCK_FILTER}"
        )

    print("=" * 78)
    print(f"analysis_for_paper.py  —  output root: {OUTPUT_ROOT}")
    if BLOCK_FILTER is not None:
        print(f"                        block filter: {BLOCK_FILTER!r}")
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

    # ── Step: HTML factor browser for each fit ──────────────────────────
    html_paths: dict[str, Path] = {}
    for slug, fit in fits.items():
        log.info("═══ export_html_browser [%s] ═══", slug)
        html_path = export_html_browser(loaded[slug], fit)
        if html_path is not None:
            html_paths[slug] = html_path

    # ── Step: within-model validation (Cronbach's α + split-half |φ|) ──
    validation_results: dict[str, dict] = {}
    for slug, fit in fits.items():
        log.info("═══ run_within_model_validation [%s] ═══", slug)
        validation_results[slug] = run_within_model_validation(
            loaded[slug], fit,
        )

    # ── Step: predictivity CV (bi-cross-validation) ─────────────────────
    predictivity_results: dict[str, dict] = {}
    for slug, fit in fits.items():
        log.info("═══ run_predictivity_cv [%s] ═══", slug)
        predictivity_results[slug] = run_predictivity_cv(loaded[slug], fit)

    # ── Step: cross-model Tucker's congruence (Llama ↔ Qwen) ───────────
    cross_reports = run_cross_model_congruence(loaded, fits)

    # ── Step: paper figures (within-model α + |φ|, cross-model heatmap)─
    if validation_results:
        _plot_paper_within_model_validation(validation_results)
    if cross_reports:
        _copy_paper_cross_model_heatmap(cross_reports)

    if validation_results:
        print()
        print("=" * 78)
        print("Within-model validation summary")
        print("=" * 78)
        header = f"  {'model':<14}  {'F':>2}  {'axis':<14}  {'α':>6}  {'α-class':<12}  {'median |φ|':>10}  {'pass':>4}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for slug, res in validation_results.items():
            alpha_rows = res["cronbach_alpha"]["per_factor"]
            sh = res["split_half_congruence"]
            medians = sh.get("median_phi_per_factor", [])
            # Best-effort axis_name lookup from labels; fall back to 'F{n}'.
            labels = _load_labels_for_slug(slug, len(alpha_rows))
            for i, row in enumerate(alpha_rows):
                ax = labels.get(i, f"F{i}")
                med = medians[i] if i < len(medians) else float("nan")
                passed = med >= SPLIT_HALF_PASS_THRESHOLD_PHI
                print(
                    f"  {slug:<14}  F{row['factor_index']:<1}  {ax:<14}  "
                    f"{row['alpha']:>6.3f}  {row['alpha_class']:<12}  "
                    f"{med:>10.3f}  {'yes' if passed else 'no':>4}"
                )

    if predictivity_results:
        print()
        print("=" * 78)
        print("Predictivity cross-validation (bi-CV)")
        print("=" * 78)
        print(f"  {'model':<14}  {'main R²':>9}  {'shuffle':>8}  "
              f"{'k−1':>7}  {'item-mean':>10}  {'main CI':>18}  pass")
        for slug, res in predictivity_results.items():
            ci = res.get("mean_r2_main_ci")
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"
            print(
                f"  {slug:<14}  "
                f"{res.get('mean_r2_main', float('nan')):>9.3f}  "
                f"{res.get('mean_r2_persona_shuffle', float('nan')):>8.3f}  "
                f"{res.get('mean_r2_k_minus_1', float('nan')):>7.3f}  "
                f"{res.get('mean_r2_item_mean', float('nan')):>10.3f}  "
                f"{ci_str:>18}  "
                f"{'yes' if res.get('pass') else 'no'}"
            )

    if cross_reports:
        print()
        print("=" * 78)
        print("Cross-model Tucker's congruence summary")
        print("=" * 78)
        for pair_label, bundle in cross_reports.items():
            subsets = bundle["subsets"]
            print(f"  {pair_label}")
            print(f"    {'subset':<10}  {'n_shared':>8}  {'mean |φ|':>9}  "
                  f"{'per-pair matched |φ|':<50}")
            for subset_label, rep in subsets.items():
                per_pair = [r["phi"] for r in rep["matched"]]
                per_pair_str = ", ".join(f"{v:.2f}" for v in per_pair)
                print(
                    f"    {subset_label:<10}  {rep['n_shared_items']:>8}  "
                    f"{rep['overall_mean_matched_phi']:>9.3f}  [{per_pair_str}]"
                )
                for row in rep["unmatched_target"]:
                    print(
                        f"      ↳ unmatched Qwen F{row['target_factor']}: "
                        f"best partner Llama F{row['best_anchor_partner']} "
                        f"|φ|={row['phi']:.3f} ({row['classification']})"
                    )

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
            if slug in html_paths:
                print(f"    html browser: {html_paths[slug]}")


if __name__ == "__main__":
    main()
