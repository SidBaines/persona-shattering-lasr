# %%
"""Interim analysis scratchpad for the external-rollouts FA.

Jupyter-notebook-style: `#%%` delimits cells. Run interactively (VS Code,
PyCharm, `jupytext`, etc.) and iterate. Each cell stays self-contained so
earlier ones don't need to be re-run when you tweak a later one.

Expand as we go — first three cells are imports, preset selection, and
Horn's parallel analysis across four data slicings.
"""

#%% Imports

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src_dev.factor_analysis.parallel_analysis import parallel_analysis
from src_dev.psychometric.combine import (
    combine_per_pair_outputs,
    load_pair_outputs,
)

# Single source of truth for preset/questionnaire definitions + run-id builders.
from scripts_dev.unsupervised_embeddings.psychometric_rollout_fa import (
    EXTERNAL_ROLLOUT_PRESETS,
    QUESTIONNAIRE_PRESETS,
    _questionnaire_run_id,
)

# Anchor the scratch path to the repo root so the script works from any
# Jupyter CWD — __file__ resolves to
# <repo>/scripts_dev/psychometric_assessment/external_rollouts_interim_analysis.py,
# so its parent.parent.parent is the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRATCH_ROOT = _REPO_ROOT / "scratch" / "psychometric_fa"
SEED = 436
random.seed(SEED)
np.random.seed(SEED)

# Where to save figures from each cell (so the notebook-in-terminal run can
# be inspected later). Set SAVE_FIGURES = False to keep plots inline-only.
FIGURES_DIR = SCRATCH_ROOT / "interim_analysis_figs"
SAVE_FIGURES: bool = True
if SAVE_FIGURES:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(fig, name: str) -> None:
    """Save a figure under FIGURES_DIR if SAVE_FIGURES is on."""
    if not SAVE_FIGURES:
        return
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  [saved] {path}")


print(f"[config] SCRATCH_ROOT = {SCRATCH_ROOT}")
print(f"[config] FIGURES_DIR  = {FIGURES_DIR}  (save={SAVE_FIGURES})")


#%% Presets + questionnaires (comment in/out as you like)

PRESETS: list[str] = [
    "kwai_swe",
    "prism_zephyr_7b_beta",
    "prism_mistral_7b_v01",
    "prism_llama2_7b_chat",
    "lmsys_koala_13b_t5",
    # "prism_llama2_13b_chat",     # MISSING: awaiting GPU re-run
    # "prism_falcon_7b_instruct",  # MISSING: awaiting GPU re-run
    # "prism_oasst_pythia_12b",    # broken: v5 parser ~0% valid; trait_mcq partial
]

QUESTIONNAIRES: list[str] = [
    "v5",
    "trait_ocean_v1_nolead",
]


#%% Load per-pair outputs and combine into one matrix

pair_data: dict = {}
pair_version: dict = {}
for r_key in PRESETS:
    for q_key in QUESTIONNAIRES:
        q_parent = SCRATCH_ROOT / _questionnaire_run_id(r_key, q_key)
        matrix_path = q_parent / "questionnaire" / "response_matrix.npy"
        if not matrix_path.exists():
            print(f"[skip] no matrix for ({r_key!r}, {q_key!r})")
            continue
        pair_data[(r_key, q_key)] = load_pair_outputs(q_parent)
        pair_version[(r_key, q_key)] = QUESTIONNAIRE_PRESETS[q_key].version

if not pair_data:
    raise RuntimeError(
        f"No (preset, questionnaire) pairs resolved to a cached Stage-2 "
        f"output under {SCRATCH_ROOT}. Check that the run-id dirs exist "
        f"locally (e.g. `ls {SCRATCH_ROOT} | grep questionnaire-`) and "
        f"that the active PRESETS / QUESTIONNAIRES have matching caches."
    )

matrix, metadata, items = combine_per_pair_outputs(
    pair_data, pair_version,
    out_dir=None,
    provenance_extra={"script": __file__},
)
print(
    f"Combined matrix: {matrix.shape[0]} rows × {matrix.shape[1]} items "
    f"(nan_frac={np.isnan(matrix).mean():.4f})"
)
if matrix.size == 0 or not metadata:
    raise RuntimeError(
        "Combined matrix / metadata is empty. "
        f"pair_data had {len(pair_data)} entries but nothing survived combine."
    )

# Analysis-ready summary: rows per preset + items per block.
from collections import Counter as _Counter
_rows_per_preset = _Counter(m["rollout_preset_key"] for m in metadata)
_items_per_block = _Counter(it.get("block", "?") for it in items)
print("\n[summary] rows per preset in combined metadata:")
for _k, _n in sorted(_rows_per_preset.items()):
    print(f"    {_k:<28s} {_n:>5d}")
print(f"    {'TOTAL':<28s} {sum(_rows_per_preset.values()):>5d}")
print("[summary] items per block:")
for _b, _n in sorted(_items_per_block.items()):
    print(f"    {_b:<28s} {_n:>5d}")


#%% Variance cut: top-K items by per-block relative residualised variance
#
# Reuses the same selector the main driver uses (so the notebook stays
# consistent with external_rollout_analysis.py). Residualises against
# rollout_preset_key, ranks each item by its post-residualisation variance
# divided by its block's median — scale-invariant, so Likert and trait_mcq
# get equal footing — and keeps the top-K.
#
# Set TOP_K = None to skip the cut. Re-run this cell only after re-running
# the combine cell (since it mutates `matrix`, `items`).

from scripts_dev.psychometric_assessment.external_rollout_analysis import (
    _select_top_k_items_by_residual_variance,
)

TOP_K: int | None = 50

if TOP_K is not None:
    matrix, items = _select_top_k_items_by_residual_variance(
        matrix, metadata, items,
        top_k=TOP_K,
        group_field="rollout_preset_key",
        report_path=None,   # skip CSV write in the notebook
    )
    print(
        f"After top-K cut: {matrix.shape[0]} rows × {matrix.shape[1]} items"
    )

    # Show top few kept items for a sanity check of what we're feeding FA.
    from collections import Counter as _Counter
    print("[summary] kept items per block after cut:")
    _kept_per_block = _Counter(it.get("block", "?") for it in items)
    for _b, _n in sorted(_kept_per_block.items()):
        print(f"    {_b:<12s} {_n:>3d}")
    print("[summary] top 10 kept items (in preserved order):")
    for _i, _it in enumerate(items[:10]):
        _text = (_it.get("text") or "")[:70].replace("\n", " ")
        print(
            f"    {_i+1:>3}. block={_it.get('block','?'):<10s} "
            f"dim={(_it.get('dimension') or '-'):<18s} {_text}"
        )
else:
    print("TOP_K=None — skipping variance cut")


#%% Horn's parallel analysis × four data slicings

# Per-preset NaN rate sanity check — if NaN concentration varies sharply
# across presets, listwise deletion on pooled slices will drop
# non-random rows and bias PA results. Warn if any preset is ≥5 × the
# median NaN rate.
_nan_frac_global = float(np.isnan(matrix).mean())
print(
    f"\n[NaN-audit] global NaN fraction: {_nan_frac_global:.4f}  "
    f"(listwise deletion applies to every PA slice below)"
)
print(f"  {'preset':<28s} {'n_rows':>7s}  {'nan_frac':>9s}")
_preset_nan_rates = []
for r_key in sorted({m["rollout_preset_key"] for m in metadata}):
    mask = np.array(
        [m["rollout_preset_key"] == r_key for m in metadata], dtype=bool,
    )
    sub = matrix[mask]
    frac = float(np.isnan(sub).mean())
    _preset_nan_rates.append(frac)
    print(f"  {r_key:<28s} {int(mask.sum()):>7d}  {frac:>9.4f}")
_med = float(np.median(_preset_nan_rates)) if _preset_nan_rates else 0.0
if _med > 0 and any(f > 5 * _med for f in _preset_nan_rates):
    print(
        "  ⚠ One or more presets have NaN fraction > 5 × median — "
        "listwise deletion will drop their rows non-randomly. "
        "Consider per-preset PA or imputation."
    )


# Accumulator for per-slice Horn's results — consumed by the summary table
# printed at the end of this cell.
_pa_records: list[dict] = []


def _pa_on_slice(
    data: np.ndarray,
    label: str,
    ax: plt.Axes,
    *,
    n_iterations: int = 100,
    max_show: int = 25,
    records: list | None = None,
) -> int:
    """Drop NaN rows + constant cols, run permutation PA, plot scree on ax.

    Uses listwise deletion: any persona with ≥1 NaN cell in the slice is
    dropped. Prints the NaN accounting and (if too much data is lost)
    prints a visible warning instead of silently producing an empty plot.

    Returns the number of factors recommended, or 0 if the slice is too
    small / too sparse to run PA on.
    """
    n_rows_in, n_cols_in = data.shape
    nan_frac = float(np.isnan(data).mean()) if data.size else 0.0
    row_mask = np.sum(np.isnan(data), axis=1) == 0
    clean = data[row_mask]
    col_var = np.var(clean, axis=0)
    clean = clean[:, col_var > 0]
    n_rows_kept = clean.shape[0]
    n_rows_dropped = n_rows_in - n_rows_kept
    n_const_cols_dropped = int(np.sum(col_var == 0))

    header_msg = (
        f"[{label}]  in=({n_rows_in},{n_cols_in})  "
        f"nan_frac={nan_frac:.4f}  "
        f"rows_dropped_listwise={n_rows_dropped}  "
        f"const_cols_dropped={n_const_cols_dropped}  "
        f"→ final={clean.shape}"
    )
    print(header_msg)

    # Warn loudly if listwise deletion is catastrophic — more useful than a
    # silent empty axis.
    if n_rows_in > 0 and n_rows_kept / n_rows_in < 0.5:
        print(
            f"  ⚠ WARNING: listwise deletion dropped "
            f"{n_rows_dropped}/{n_rows_in} "
            f"({100 * n_rows_dropped / n_rows_in:.1f}%) of rows for {label!r}. "
            "Result below (if any) is on the surviving subset only."
        )

    if clean.shape[0] < 10 or clean.shape[1] < 3:
        msg = (
            f"SKIPPED — too small after cleaning: final={clean.shape}, "
            f"need ≥10 rows and ≥3 cols"
        )
        print(f"  ⚠ {msg}")
        ax.text(
            0.5, 0.5,
            f"{label}\n{msg}",
            ha="center", va="center",
            transform=ax.transAxes,
            color="tab:red",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        (records if records is not None else _pa_records).append({
            "label": label, "final_shape": tuple(clean.shape),
            "nan_frac": nan_frac, "n_recommended": None, "status": "skipped",
        })
        return 0
    pa = parallel_analysis(
        clean,
        n_iterations=n_iterations,
        method="permutation",
        random_state=SEED,
    )
    x = np.arange(1, len(pa["real_eigenvalues"]) + 1)
    ax.plot(x, pa["real_eigenvalues"], "o-", color="tab:blue", label="real")
    ax.plot(x, pa["random_threshold"], "s--", color="tab:red", label="p95 random")
    ax.axhline(1.0, color="gray", lw=0.5, ls=":")
    ax.set_xlim(0.5, min(max_show, len(x)) + 0.5)
    ax.set_xlabel("factor #")
    ax.set_ylabel("eigenvalue")
    ax.set_title(
        f"{label}\nn_rec={pa['n_recommended']}  shape={clean.shape}"
    )
    ax.legend(fontsize=8)
    (records if records is not None else _pa_records).append({
        "label": label, "final_shape": tuple(clean.shape),
        "nan_frac": nan_frac,
        "n_recommended": int(pa["n_recommended"]),
        "status": "ok",
    })
    return pa["n_recommended"]


# --- Slicing 1: whole combined dataset ---
fig, ax = plt.subplots(figsize=(6, 4))
_pa_on_slice(matrix, "whole combined", ax)
plt.tight_layout()
_save_fig(fig, "cell5_horns_s1_whole")
plt.show()

# --- Slicing 2: grouped by model (rollout preset) ---
presets_seen = sorted({m["rollout_preset_key"] for m in metadata})
fig, axes = plt.subplots(
    1, len(presets_seen),
    figsize=(5 * len(presets_seen), 4),
    squeeze=False,
)
for ax, r_key in zip(axes[0], presets_seen):
    row_mask = np.array(
        [m["rollout_preset_key"] == r_key for m in metadata], dtype=bool,
    )
    _pa_on_slice(matrix[row_mask], f"preset={r_key}", ax)
plt.tight_layout()
_save_fig(fig, "cell5_horns_s2_by_preset")
plt.show()

# --- Slicing 3: grouped by questionnaire block (column subset) ---
item_blocks = np.array([it.get("block", "") for it in items])
blocks_seen = sorted(set(item_blocks))
fig, axes = plt.subplots(
    1, len(blocks_seen),
    figsize=(5 * len(blocks_seen), 4),
    squeeze=False,
)
for ax, b in zip(axes[0], blocks_seen):
    col_mask = item_blocks == b
    _pa_on_slice(matrix[:, col_mask], f"block={b}", ax)
plt.tight_layout()
_save_fig(fig, "cell5_horns_s3_by_block")
plt.show()

# --- Slicing 4: grouped by model × questionnaire block ---
fig, axes = plt.subplots(
    len(presets_seen), len(blocks_seen),
    figsize=(5 * len(blocks_seen), 4 * len(presets_seen)),
    squeeze=False,
)
for i, r_key in enumerate(presets_seen):
    row_mask = np.array(
        [m["rollout_preset_key"] == r_key for m in metadata], dtype=bool,
    )
    for j, b in enumerate(blocks_seen):
        col_mask = item_blocks == b
        _pa_on_slice(
            matrix[row_mask][:, col_mask],
            f"{r_key} × {b}",
            axes[i, j],
        )
plt.tight_layout()
_save_fig(fig, "cell5_horns_s4_preset_x_block")
plt.show()


# Final summary table across all slicings run above.
print("\n" + "=" * 78)
print("[summary] Horn's parallel analysis across all slicings")
print("=" * 78)
print(f"{'label':<40s} {'shape':>14s}  {'nan_frac':>8s}  {'n_rec':>6s}")
print("-" * 78)
for r in _pa_records:
    shape_str = f"{r['final_shape'][0]}×{r['final_shape'][1]}" if r["final_shape"] else "-"
    n_rec = r["n_recommended"] if r["n_recommended"] is not None else "skip"
    print(
        f"{r['label']:<40s} {shape_str:>14s}  {r['nan_frac']:>8.4f}  {str(n_rec):>6s}"
    )
print("=" * 78)


#%% Horn's parallel analysis on the preset-residualised data
#
# Cell 5's screes are on the raw matrix — their top eigenvalues include the
# model-identity structure. Residualising by rollout_preset_key before PA
# removes that first-order source of between-group variance, so the
# remaining scree reveals the within-preset factor count directly. Expect
# the top 3-4 eigenvalues to collapse vs the raw scree; anything above
# the permutation threshold after collapse is the within-preset signal.
#
# Only "whole residualised" and "by block residualised" are meaningful —
# residualising then slicing by preset is a no-op (per-preset mean = 0).

from src_dev.psychometric.preprocessing import preprocess_response_matrix as _pp_fn

# Residualise once, re-use for every slicing below.
_m_resid_pa, _meta_resid_pa, _items_resid_pa, _ = _pp_fn(
    matrix, metadata, items,
    min_item_variance=0.0,
    do_residualize=True,
    residualize_group_field="rollout_preset_key",
)
print(
    f"\n[residualised] after preset mean-subtraction: shape={_m_resid_pa.shape}  "
    f"items dropped={matrix.shape[1] - _m_resid_pa.shape[1]}"
)

_item_blocks_r = np.array([it.get("block", "") for it in _items_resid_pa])
_blocks_r = sorted(set(_item_blocks_r))

# Dedicated accumulator so the residualised PA results don't collide with
# the raw summary table above.
_pa_records_resid: list[dict] = []

# Slicing R1: whole residualised.
fig, ax = plt.subplots(figsize=(6, 4))
_pa_on_slice(
    _m_resid_pa, "whole (residualised)", ax,
    records=_pa_records_resid,
)
plt.tight_layout()
_save_fig(fig, "cell5b_horns_residualised_whole")
plt.show()

# Slicing R2: per block residualised.
fig, axes = plt.subplots(
    1, len(_blocks_r),
    figsize=(5 * len(_blocks_r), 4),
    squeeze=False,
)
for ax, b in zip(axes[0], _blocks_r):
    col_mask = _item_blocks_r == b
    _pa_on_slice(
        _m_resid_pa[:, col_mask], f"block={b} (residualised)", ax,
        records=_pa_records_resid,
    )
plt.tight_layout()
_save_fig(fig, "cell5b_horns_residualised_by_block")
plt.show()

# Residualised summary table — directly comparable to the raw summary above.
print("\n" + "=" * 78)
print("[summary] Horn's on preset-residualised slices  (compare to raw above)")
print("=" * 78)
print(f"{'label':<40s} {'shape':>14s}  {'nan_frac':>8s}  {'n_rec':>6s}")
print("-" * 78)
for r in _pa_records_resid:
    shape_str = f"{r['final_shape'][0]}×{r['final_shape'][1]}" if r["final_shape"] else "-"
    n_rec = r["n_recommended"] if r["n_recommended"] is not None else "skip"
    print(
        f"{r['label']:<40s} {shape_str:>14s}  {r['nan_frac']:>8.4f}  {str(n_rec):>6s}"
    )
print("=" * 78)


#%% How much does model identity split each factor? (whole + per-questionnaire)
#
# Fit FA on each of: the whole combined matrix, Likert-only columns,
# trait_mcq-only columns. For each:
#   (i)  bar chart of η²(rollout_preset_key) per factor — the share of
#        each factor's variance explained by the preset split.
#   (ii) grouped boxplot of factor scores per preset across the first
#        N_FACTORS_VIZ factors — shows the direction and spread of any
#        separation.
#
# Tweak N_FACTORS_VIZ / ROTATION / FA_METHOD below.

import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from src_dev.factor_analysis.factor_analysis import run_factor_analysis
from src_dev.factor_analysis.interpretation import prompt_effects
from src_dev.psychometric.preprocessing import preprocess_response_matrix

# FA extraction: factor_analyzer's "principal" is **Principal Axis Factoring**
# (iterative, uses reduced correlation matrix with communalities on diagonal),
# NOT PCA. Verified via sum(eigenvalues_common) == sum(communalities) < n_vars.
# For an ML sensitivity check, swap FA_METHOD_VIZ to "ml".
#
# Note on scale-mixing Likert × trait_mcq: factor_analyzer fits on the
# **correlation matrix** by default, which standardises item variances
# implicitly. So the within-FA variance-dominance concern (Likert 1-5 raw
# variance ~4 vs trait_mcq 0/1 raw variance ~0.25) does not apply at the
# extraction step. The earlier Top-K selector DID have this problem with
# raw variance, and is already fixed via per-block-relative ranking.
FA_METHOD_VIZ = "principal"     # PAF — or "ml" for maximum-likelihood sensitivity
ROTATION_VIZ = "oblimin"        # oblique default — personality factors usually correlate;
                                 # use "varimax" only as an orthogonal sensitivity check.

# k is resolved per-split via Horn's PA — see _resolve_k_for_slice(). Keep a
# hard cap so runaway PA recommendations (on noisy subsets) don't produce
# uninterpretable 15-factor solutions.
N_FACTORS_MAX = 12
N_FACTORS_FALLBACK = 7  # used when PA fails or is skipped


def _resolve_k_for_slice(data: np.ndarray, label: str) -> int:
    """Pick the number of factors from Horn's permutation PA on this slice.

    Drops NaN rows + constant cols, runs PA, caps at N_FACTORS_MAX. Returns
    the fallback if PA fails or finds fewer than 1 factor.
    """
    from src_dev.factor_analysis.parallel_analysis import parallel_analysis
    row_mask = np.sum(np.isnan(data), axis=1) == 0
    clean = data[row_mask]
    col_var = np.var(clean, axis=0)
    clean = clean[:, col_var > 0]
    if clean.shape[0] < 20 or clean.shape[1] < 4:
        print(
            f"  [k-resolve] {label}: too small for PA ({clean.shape}); "
            f"fallback k={N_FACTORS_FALLBACK}"
        )
        return N_FACTORS_FALLBACK
    try:
        pa = parallel_analysis(
            clean, n_iterations=100, method="permutation", random_state=SEED,
        )
        k_pa = int(pa["n_recommended"])
    except Exception as exc:
        print(f"  [k-resolve] {label}: PA failed ({exc}); fallback k={N_FACTORS_FALLBACK}")
        return N_FACTORS_FALLBACK
    k = max(1, min(k_pa, N_FACTORS_MAX, clean.shape[1] - 1))
    print(
        f"  [k-resolve] {label}: PA n_rec={k_pa} → using k={k} "
        f"(cap N_FACTORS_MAX={N_FACTORS_MAX})"
    )
    return k


def _fit_fa_with_scores(
    data: np.ndarray,
    meta: list[dict],
    cols: list[dict],
    label: str,
    *,
    k_override: int | None = None,
):
    """Row-filter + FA; return (scores, meta_filtered, proportion_variance, eta2_preset, k_eff)."""
    clean, meta_clean, cols_clean, _ = preprocess_response_matrix(
        data, meta, cols,
        min_item_variance=0.0,  # column pre-selection already handled upstream (top-K cell).
        do_residualize=False,
    )
    # Resolve k from Horn's PA unless an explicit override is passed.
    k_requested = k_override if k_override is not None else _resolve_k_for_slice(data, label)
    k_eff = max(1, min(k_requested, N_FACTORS_MAX, clean.shape[1] - 1))
    if k_eff != k_requested:
        print(
            f"  [fa] {label}: k adjusted {k_requested} → {k_eff} "
            f"(usable cols={clean.shape[1]})"
        )
    fa = run_factor_analysis(
        clean, n_factors=k_eff, method=FA_METHOD_VIZ, rotation=ROTATION_VIZ,
    )
    eta2 = prompt_effects(
        fa["scores"], meta_clean, group_field="rollout_preset_key",
    )
    return fa["scores"], meta_clean, fa["proportion_variance"], eta2, k_eff


# Assemble the 3 splits: whole + one per block.
splits: dict[str, tuple[np.ndarray, list[dict], list[dict]]] = {
    "whole combined": (matrix, metadata, items),
}
for b in blocks_seen:
    col_mask_b = np.array([it.get("block", "") == b for it in items], dtype=bool)
    items_b = [it for it in items if it.get("block", "") == b]
    splits[f"block={b}"] = (matrix[:, col_mask_b], metadata, items_b)

fig, axes = plt.subplots(
    len(splits), 2,
    figsize=(15, 4.2 * len(splits)),
    squeeze=False,
    gridspec_kw={"width_ratios": [1, 2.2]},
)

# Accumulator for the aggregate table printed at the end of this cell.
_preset_split_records: list[dict] = []

for i, (label, (data, meta, cols)) in enumerate(splits.items()):
    print(f"\n--- FA + preset-split  [{label}]  input shape={data.shape} ---")
    try:
        scores, meta_clean, prop_var, eta2, k_eff = _fit_fa_with_scores(
            data, meta, cols, label,
        )
    except Exception as exc:
        for ax in axes[i]:
            ax.text(
                0.5, 0.5, f"FA failed:\n{exc}",
                ha="center", va="center", transform=ax.transAxes,
                color="tab:red", fontsize=9,
            )
            ax.set_xticks([]); ax.set_yticks([])
        continue

    # Per-factor numeric table for this split (analysis-ready).
    # η² thresholds 0.5 / 0.2 are heuristic labels, not formal cutoffs —
    # report the raw number too.
    print(
        f"  {'factor':>6s}  {'prop_var':>9s}  {'η²(preset)':>10s}  "
        f"{'band':<18s}"
    )
    for f_idx in range(k_eff):
        band = (
            "η²≥0.5 (dominated)"  if eta2[f_idx] >= 0.5
            else "η²≥0.2 (influenced)" if eta2[f_idx] >= 0.2
            else "η²<0.2 (independent)"
        )
        print(
            f"  {'F'+str(f_idx+1):>6s}  {prop_var[f_idx]:>9.3f}  "
            f"{eta2[f_idx]:>10.3f}  {band:<18s}"
        )
        _preset_split_records.append({
            "split": label,
            "factor": f_idx + 1,
            "prop_var": float(prop_var[f_idx]),
            "eta2_preset": float(eta2[f_idx]),
            "band": band,
        })
    _first_clean = next(
        (f_idx + 1 for f_idx in range(k_eff) if eta2[f_idx] < 0.2), None,
    )
    if _first_clean is not None:
        print(f"  → first factor with η²<0.2: F{_first_clean}")
    else:
        print("  → no factor with η²<0.2 in the first K")

    # ── (i) η²(preset) bars ──────────────────────────────────────────
    ax = axes[i, 0]
    x = np.arange(1, k_eff + 1)
    ax.bar(x, eta2, color="#B71C1C")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{k}" for k in x])
    ax.set_xlabel("factor")
    ax.set_ylabel("η²(preset)")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"[{label}]  η²(rollout_preset_key) per factor\n"
        f"(1.0 = factor fully explained by preset split)"
    )
    for xi, ev, pv in zip(x, eta2, prop_var):
        ax.text(
            xi, ev + 0.02, f"{ev:.2f}",
            ha="center", va="bottom", fontsize=8,
        )
        ax.text(
            xi, 0.02, f"var={pv:.2f}",
            ha="center", va="bottom", fontsize=6, color="white",
        )

    # ── (ii) grouped boxplot: factor scores per preset ──────────────
    ax = axes[i, 1]
    df = pd.DataFrame(
        scores, columns=[f"F{k}" for k in range(1, k_eff + 1)],
    )
    df["preset"] = [m["rollout_preset_key"] for m in meta_clean]
    df_long = df.melt(
        id_vars="preset", var_name="factor", value_name="score",
    )
    preset_order = sorted(df["preset"].unique())
    palette = dict(zip(preset_order, sns.color_palette("tab10", len(preset_order))))
    sns.boxplot(
        data=df_long, x="factor", y="score", hue="preset",
        hue_order=preset_order, palette=palette,
        ax=ax, showfliers=False, width=0.8,
    )
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("factor")
    ax.set_ylabel("factor score")
    ax.set_title(f"[{label}]  factor scores by preset (first {k_eff} factors)")
    ax.legend(fontsize=7, loc="upper right", title="preset", title_fontsize=7)

fig.tight_layout()
_save_fig(fig, "cell6_preset_split_by_factor")
plt.show()


# Aggregate cross-split table so it's easy to scan.
print("\n" + "=" * 78)
print("[summary] η²(preset) across splits")
print("=" * 78)
print(f"{'split':<20s} {'factor':<6s} {'prop_var':>9s} {'η²':>7s}  {'band':<22s}")
print("-" * 78)
for r in _preset_split_records:
    print(
        f"{r['split']:<20s} F{r['factor']:<5d} {r['prop_var']:>9.3f} "
        f"{r['eta2_preset']:>7.3f}  {r['band']:<22s}"
    )
print("=" * 78)


#%% Preset-residualised whole-combined FA
#
# Subtracts the per-preset mean from every cell before fitting, so the
# "which model is speaking" variance is stripped out. Remaining structure
# should be the within-preset (psychometric-like) part. η²(preset) is
# expected ≈ 0 everywhere by construction — the check is whether there's
# still meaningful variance left to factor, and what it looks like.

def _fit_fa_residualised(data, meta, cols, label, *, k_override: int | None = None):
    """Like _fit_fa_with_scores, but residualise by rollout_preset_key first.

    k resolved from Horn's PA on the residualised matrix unless overridden.
    """
    clean, meta_clean, cols_clean, _ = preprocess_response_matrix(
        data, meta, cols,
        min_item_variance=0.0,
        do_residualize=True,
        residualize_group_field="rollout_preset_key",
    )
    # Run PA on the residualised matrix to pick k (matches the extraction target).
    k_requested = (
        k_override if k_override is not None
        else _resolve_k_for_slice(clean, f"{label} [residualised]")
    )
    k_eff = max(1, min(k_requested, N_FACTORS_MAX, clean.shape[1] - 1))
    if k_eff != k_requested:
        print(
            f"  [fa] {label}: k adjusted {k_requested} → {k_eff} "
            f"(usable cols after residualisation={clean.shape[1]})"
        )
    fa = run_factor_analysis(
        clean, n_factors=k_eff, method=FA_METHOD_VIZ, rotation=ROTATION_VIZ,
    )
    eta2 = prompt_effects(
        fa["scores"], meta_clean, group_field="rollout_preset_key",
    )
    return fa, clean, cols_clean, meta_clean, fa["proportion_variance"], eta2, k_eff


label_resid = "whole combined (preset-residualised)"
print(f"\n--- FA + preset-split  [{label_resid}]  "
      f"input shape={matrix.shape} ---")
resid_fa, resid_data, resid_cols, resid_meta, resid_prop_var, resid_eta2, resid_k = (
    _fit_fa_residualised(matrix, metadata, items, label_resid)
)

# Per-factor numeric table.
print(
    f"  {'factor':>6s}  {'prop_var':>9s}  {'η²(preset)':>10s}  "
    f"{'band':<18s}"
)
for f_idx in range(resid_k):
    band = (
        "η²≥0.5 (dominated)"  if resid_eta2[f_idx] >= 0.5
        else "η²≥0.2 (influenced)" if resid_eta2[f_idx] >= 0.2
        else "η²<0.2 (independent)"
    )
    print(
        f"  {'F'+str(f_idx+1):>6s}  {resid_prop_var[f_idx]:>9.3f}  "
        f"{resid_eta2[f_idx]:>10.3f}  {band:<18s}"
    )

# Plot η² + boxplot, same layout as Cell 6.
fig, axes = plt.subplots(
    1, 2, figsize=(15, 4.2),
    gridspec_kw={"width_ratios": [1, 2.2]},
)

# Left: η² bars (expected ≈ 0).
ax = axes[0]
x = np.arange(1, resid_k + 1)
ax.bar(x, resid_eta2, color="#2E7D32")
ax.axhline(0, color="gray", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f"F{k}" for k in x])
ax.set_xlabel("factor")
ax.set_ylabel("η²(preset)")
ax.set_ylim(0, 1)
ax.set_title(
    f"[{label_resid}]  η²(rollout_preset_key) per factor\n"
    f"(expected ≈ 0 by construction)"
)
for xi, ev, pv in zip(x, resid_eta2, resid_prop_var):
    ax.text(xi, ev + 0.02, f"{ev:.2f}", ha="center", va="bottom", fontsize=8)
    ax.text(
        xi, 0.02, f"var={pv:.2f}",
        ha="center", va="bottom", fontsize=6, color="white",
    )

# Right: grouped boxplot of factor scores by preset.
ax = axes[1]
df = pd.DataFrame(
    resid_fa["scores"], columns=[f"F{k}" for k in range(1, resid_k + 1)],
)
df["preset"] = [m["rollout_preset_key"] for m in resid_meta]
df_long = df.melt(id_vars="preset", var_name="factor", value_name="score")
preset_order = sorted(df["preset"].unique())
palette = dict(zip(preset_order, sns.color_palette("tab10", len(preset_order))))
sns.boxplot(
    data=df_long, x="factor", y="score", hue="preset",
    hue_order=preset_order, palette=palette,
    ax=ax, showfliers=False, width=0.8,
)
ax.axhline(0, color="gray", lw=0.5, ls=":")
ax.set_xlabel("factor")
ax.set_ylabel("factor score (residualised)")
ax.set_title(
    f"[{label_resid}]  factor scores by preset (first {resid_k} factors)"
)
ax.legend(fontsize=7, loc="upper right", title="preset", title_fontsize=7)

fig.tight_layout()
_save_fig(fig, "cell7_residualised_whole_combined")
plt.show()

# Quick sanity check: max η² across factors.
print(
    f"\n[sanity] max η²(preset) across residualised factors: "
    f"{float(np.max(resid_eta2)):.4f}  "
    f"(should be ~0 if residualisation removed all between-preset variance)"
)

# Factor correlation matrix — informative under oblique rotation (oblimin /
# promax); is None for orthogonal varimax / None. When it's non-trivial it
# tells us which factors share common variance and whether the rotated
# solution is close to orthogonal anyway.
phi = resid_fa.get("factor_correlation_matrix")
if phi is not None:
    fig, ax = plt.subplots(figsize=(0.6 * phi.shape[0] + 2, 0.6 * phi.shape[0] + 1.5))
    vmax = 1.0
    im = ax.imshow(phi, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(phi.shape[0]))
    ax.set_yticks(range(phi.shape[0]))
    ax.set_xticklabels([f"F{i+1}" for i in range(phi.shape[0])])
    ax.set_yticklabels([f"F{i+1}" for i in range(phi.shape[0])])
    for i in range(phi.shape[0]):
        for j in range(phi.shape[0]):
            ax.text(
                j, i, f"{phi[i, j]:+.2f}",
                ha="center", va="center", fontsize=8,
                color="white" if abs(phi[i, j]) > 0.55 else "black",
            )
    ax.set_title(
        f"Residualised-FA factor correlation matrix Φ  "
        f"(rotation={ROTATION_VIZ})\nOff-diagonals ≠ 0 ⇒ oblique solution "
        "found correlated factors"
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation")
    fig.tight_layout()
    _save_fig(fig, "cell7b_residualised_factor_correlations")
    plt.show()
    print(
        f"[factor-correlations] max off-diagonal |ρ|: "
        f"{float(np.max(np.abs(phi - np.eye(phi.shape[0])))):.3f}  "
        f"(if ~0, an orthogonal rotation would have been adequate)"
    )
else:
    print(
        f"[factor-correlations] rotation={ROTATION_VIZ!r} is orthogonal — "
        f"no Φ matrix to plot."
    )


#%% Heatmap of loadings F1–F3 on the residualised whole-combined FA
#
# Rows = items (all that survived residualisation preprocessing), columns
# = factors 1..min(3, K). Rows sorted by primary factor (argmax |loading|)
# then by signed primary loading desc, so items cluster by the factor they
# define most strongly. Colour = signed loading (diverging red/blue).
# Also prints the top-10 items by |loading| per factor — the labelling
# material.

n_show_factors = min(3, resid_k)
loadings = resid_fa["loadings"][:, :n_show_factors]
# Communality = variance of each item explained by the factor solution
# (sum of squared loadings across *all* extracted factors, not just the
# first n_show). Items with h² < 0.2 are weakly captured and shouldn't be
# used to label factors.
communalities_full = resid_fa["communalities"]

# Cross-loading flag: 2nd-largest |loading| on the first n_show factors >
# 0.3 means the item loads on multiple factors and violates simple structure.
abs_loadings = np.abs(loadings)
second_largest = np.sort(abs_loadings, axis=1)[:, -2] if n_show_factors >= 2 else np.zeros(len(loadings))
is_cross_loaded = second_largest > 0.3

# Sort rows: primary factor (argmax of |loading|), then signed primary loading desc.
primary_factor = np.argmax(np.abs(loadings), axis=1)
primary_loading = loadings[np.arange(loadings.shape[0]), primary_factor]
sort_order = np.lexsort((-primary_loading, primary_factor))
sorted_loadings = loadings[sort_order]
sorted_cols = [resid_cols[i] for i in sort_order]
sorted_comm = communalities_full[sort_order]
sorted_cross = is_cross_loaded[sort_order]

# Compact y-axis labels: [block] dimension? truncated-text + flags
row_labels = []
for it, h2, cx in zip(sorted_cols, sorted_comm, sorted_cross):
    block = (it.get("block") or "?")[:5]
    dim = it.get("dimension") or "-"
    text = (it.get("text") or it.get("item_id") or "?").replace("\n", " ")[:45]
    flag = ""
    if h2 < 0.2:
        flag += " ⚠weak"  # poorly captured by factors
    if cx:
        flag += " ×"      # cross-loaded on first n_show factors
    row_labels.append(
        f"[{block}] h²={h2:.2f}{flag}  {dim:<12s} {text}"
    )

vmax = float(np.max(np.abs(sorted_loadings))) if sorted_loadings.size else 1.0
fig, ax = plt.subplots(
    figsize=(8, max(5, 0.22 * len(row_labels))),
)
im = ax.imshow(
    sorted_loadings, cmap="RdBu_r",
    vmin=-vmax, vmax=vmax, aspect="auto",
)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=6.5)
ax.set_xticks(range(n_show_factors))
ax.set_xticklabels([f"F{k+1}" for k in range(n_show_factors)])
ax.set_xlabel("factor")
ax.set_title(
    f"Residualised whole-combined FA — loadings F1..F{n_show_factors}\n"
    f"(items × factors, |loading|≤{vmax:.2f}; rows grouped by primary factor)"
)
# Annotate each cell with its signed loading value.
for i in range(sorted_loadings.shape[0]):
    for j in range(sorted_loadings.shape[1]):
        v = sorted_loadings[i, j]
        ax.text(
            j, i, f"{v:+.2f}",
            ha="center", va="center", fontsize=5.5,
            color="white" if abs(v) > vmax * 0.55 else "black",
        )
fig.colorbar(im, ax=ax, shrink=0.5, label="loading (signed)")
fig.tight_layout()
_save_fig(fig, "cell8_residualised_loadings_heatmap")
plt.show()

# Print top-10 items by |loading| per factor — the factor-labelling table.
for f_idx in range(n_show_factors):
    abs_load = np.abs(resid_fa["loadings"][:, f_idx])
    top_idx = np.argsort(abs_load)[::-1][:10]
    print(f"\n=== Top 10 items by |loading| on F{f_idx+1} "
          f"(var={resid_prop_var[f_idx]:.3f}) ===")
    print(f"  {'rank':>4s}  {'load':>7s}  {'block':<10s} {'dim':<18s} text")
    for rank, idx in enumerate(top_idx):
        it = resid_cols[idx]
        text = (it.get("text") or it.get("item_id") or "?").replace("\n", " ")[:70]
        load = float(resid_fa["loadings"][idx, f_idx])
        print(
            f"  {rank+1:>4d}  {load:>+7.3f}  "
            f"{(it.get('block') or '?'):<10s} "
            f"{(it.get('dimension') or '-'):<18s} {text}"
        )


#%% ICC audit: per-item variance-partitioning across presets
#
# Multilevel-FA gatekeeping diagnostic. Per-item one-way random-effects
# ICC(1,1) with unbalanced-group correction:
#     n0 = (N − Σ n_g² / N) / (G − 1)
#     ICC = (MSB − MSW) / (MSB + (n0 − 1) · MSW)
# where MSB, MSW are the between/within mean squares of a one-way ANOVA
# with preset as the factor. Values are clipped at 0 (small-sample noise
# can make the sample estimate slightly negative).
#
# Rules of thumb (Lüdtke/Hox/MCFA lit):
#   < 0.05  — negligible clustering; single-level FA fine
#   0.05-0.10 — mild; design effect noticeable
#   0.10-0.20 — substantial; multilevel strongly advised
#   > 0.20  — dominant clustering; pooled FA factor structure will be
#             contaminated by between-group variance.

def _icc_per_item(
    data: np.ndarray,
    metadata: list[dict],
    group_field: str = "rollout_preset_key",
) -> np.ndarray:
    groups = np.array([m[group_field] for m in metadata])
    unique_g, inv = np.unique(groups, return_inverse=True)
    G = len(unique_g)
    out = np.full(data.shape[1], np.nan, dtype=np.float64)
    for j in range(data.shape[1]):
        col = data[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < G + 2:
            continue
        col_v, inv_v = col[valid], inv[valid]
        n_g = np.array([(inv_v == gi).sum() for gi in range(G)])
        if (n_g < 2).any():
            continue
        mu_g = np.array([col_v[inv_v == gi].mean() for gi in range(G)])
        mu = np.average(mu_g, weights=n_g)
        MSB = float(np.sum(n_g * (mu_g - mu) ** 2) / (G - 1))
        N = int(valid.sum())
        SSW = float(sum(
            np.sum((col_v[inv_v == gi] - mu_g[gi]) ** 2) for gi in range(G)
        ))
        MSW = SSW / (N - G) if (N - G) > 0 else np.nan
        n0 = (N - np.sum(n_g ** 2) / N) / (G - 1)
        denom = MSB + (n0 - 1) * MSW
        if denom <= 0 or not np.isfinite(denom):
            continue
        out[j] = max(0.0, (MSB - MSW) / denom)
    return out


iccs = _icc_per_item(matrix, metadata, group_field="rollout_preset_key")
print(
    f"\n[ICC] Per-item one-way ICC(1,1) — share of variance between "
    f"presets  (G={len({m['rollout_preset_key'] for m in metadata})})"
)
print(
    f"  {'block':<12s} {'n':>3s}  {'median':>7s}  {'p25':>7s}  {'p75':>7s}  "
    f"{'min':>6s}  {'max':>6s}"
)
_blocks_for_icc = sorted({it.get("block", "?") for it in items})
for b in _blocks_for_icc:
    mask = np.array([it.get("block", "?") == b for it in items], dtype=bool)
    vals = iccs[mask]
    if np.isnan(vals).all():
        print(f"  {b:<12s} {int(mask.sum()):>3d}  (all NaN)")
        continue
    print(
        f"  {b:<12s} {int(mask.sum()):>3d}  "
        f"{np.nanmedian(vals):>7.3f}  {np.nanpercentile(vals, 25):>7.3f}  "
        f"{np.nanpercentile(vals, 75):>7.3f}  "
        f"{np.nanmin(vals):>6.3f}  {np.nanmax(vals):>6.3f}"
    )

# ICC distribution histogram split by block.
fig, ax = plt.subplots(figsize=(11, 4))
_bins = np.linspace(0, 1, 21)
_cmap = dict(zip(_blocks_for_icc, sns.color_palette("Set2", len(_blocks_for_icc))))
for b in _blocks_for_icc:
    mask = np.array([it.get("block", "?") == b for it in items], dtype=bool)
    vals = iccs[mask][~np.isnan(iccs[mask])]
    ax.hist(
        vals, bins=_bins, alpha=0.5, color=_cmap[b],
        label=f"{b} (n={mask.sum()})", edgecolor="white",
    )
_ymax = ax.get_ylim()[1]
for thr, lbl, col in [
    (0.05, "0.05", "#888"),
    (0.10, "0.10", "#444"),
    (0.20, "0.20", "#000"),
]:
    ax.axvline(thr, color=col, ls="--", lw=0.8)
    ax.text(
        thr, _ymax * 0.98, lbl, rotation=90, fontsize=7,
        va="top", ha="right", color=col,
    )
ax.set_xlim(0, 1)
ax.set_xlabel("per-item ICC (share of variance between presets)")
ax.set_ylabel("items")
ax.set_title(
    f"Per-item ICC distribution — G={len({m['rollout_preset_key'] for m in metadata})} "
    f"presets, N={len(metadata)} personas"
)
ax.legend()
fig.tight_layout()
_save_fig(fig, "cell9_icc_audit")
plt.show()

# Top-10 highest-ICC (most model-discriminating) + bottom-5 lowest-ICC.
_order = np.argsort(np.where(np.isnan(iccs), -1.0, iccs))[::-1]
print("\n[ICC] Top 10 highest-ICC items (model identity carries most info):")
for _rank, _idx in enumerate(_order[:10]):
    it = items[int(_idx)]
    text = (it.get("text") or it.get("item_id") or "?").replace("\n", " ")[:60]
    print(
        f"  {_rank+1:>2}. ICC={iccs[int(_idx)]:.3f}  "
        f"block={(it.get('block') or '?'):<10s} "
        f"dim={(it.get('dimension') or '-'):<18s} {text}"
    )
print("\n[ICC] Bottom 5 lowest-ICC items (most model-invariant):")
for _rank, _idx in enumerate(_order[-5:][::-1]):
    it = items[int(_idx)]
    text = (it.get("text") or it.get("item_id") or "?").replace("\n", " ")[:60]
    print(
        f"  {_rank+1:>2}. ICC={iccs[int(_idx)]:.3f}  "
        f"block={(it.get('block') or '?'):<10s} "
        f"dim={(it.get('dimension') or '-'):<18s} {text}"
    )


#%% Descriptive between-level views (G=5; formal MCFA is out of reach)
#
# Three pieces that replace the banned between-level factor model:
#   (A) LLM × item heatmap of per-preset means (centered on grand mean) —
#       raw "where does each model sit?" view.
#   (B) LLM × within-level-factor heatmap — project per-preset centered
#       item means onto the Cell-7 residualised FA loadings Λ_W. Answers:
#       do the within-level factors (clean of preset mean) also separate
#       models when re-projected at the between level?
#   (C) Hierarchical clustering dendrogram over per-preset centered means.

from scipy.cluster.hierarchy import linkage, dendrogram

# (A) Compute per-preset mean over raw non-residualised matrix.
_groups = np.array([m["rollout_preset_key"] for m in metadata])
_unique_g = np.unique(_groups)
_preset_means = np.stack([
    np.nanmean(matrix[_groups == g], axis=0) for g in _unique_g
])
_preset_means_centered = _preset_means - np.nanmean(
    _preset_means, axis=0, keepdims=True,
)

fig, ax = plt.subplots(
    figsize=(max(10, 0.2 * _preset_means_centered.shape[1]), 3.5),
)
_vmax = float(np.nanmax(np.abs(_preset_means_centered)))
im = ax.imshow(
    _preset_means_centered, cmap="RdBu_r",
    vmin=-_vmax, vmax=_vmax, aspect="auto",
)
ax.set_yticks(range(len(_unique_g)))
ax.set_yticklabels(list(_unique_g), fontsize=8)
ax.set_xticks([])
ax.set_xlabel(f"items (n={matrix.shape[1]})")
ax.set_title(
    f"(A) Per-preset mean − grand mean (raw, not residualised)  "
    f"G={len(_unique_g)}, N={matrix.shape[0]}"
)
fig.colorbar(im, ax=ax, shrink=0.7, label="mean deviation")
fig.tight_layout()
_save_fig(fig, "cell10a_preset_item_heatmap")
plt.show()

# (B) Project centered preset means onto within-level loadings Λ_W.
# resid_fa loadings shape: (n_items_kept, k). resid_cols is the surviving
# item subset after Cell 7's preprocessing — in this run all 50 survived.
loadings_W = resid_fa["loadings"]
assert loadings_W.shape[0] == len(resid_cols), "item count mismatch vs Λ_W"
# Map the preset-mean matrix columns onto the same order as resid_cols.
_id_to_idx = {
    (it.get("item_id") or i): i for i, it in enumerate(items)
}
_reorder = np.array([
    _id_to_idx[rc.get("item_id")] for rc in resid_cols
    if rc.get("item_id") in _id_to_idx
])
assert len(_reorder) == loadings_W.shape[0], (
    "Could not align resid_cols with items; unlikely unless item_ids differ"
)
_preset_means_aligned = _preset_means_centered[:, _reorder]
_preset_scores_on_W = _preset_means_aligned @ loadings_W  # (G, k)

fig, ax = plt.subplots(
    figsize=(6 + 0.5 * loadings_W.shape[1], 3 + 0.4 * len(_unique_g)),
)
_vmax = float(np.max(np.abs(_preset_scores_on_W))) or 1.0
im = ax.imshow(
    _preset_scores_on_W, cmap="RdBu_r",
    vmin=-_vmax, vmax=_vmax, aspect="auto",
)
ax.set_yticks(range(len(_unique_g)))
ax.set_yticklabels(list(_unique_g), fontsize=8)
ax.set_xticks(range(loadings_W.shape[1]))
ax.set_xticklabels([f"F{i+1}" for i in range(loadings_W.shape[1])])
ax.set_xlabel("within-level factor (Λ_W from Cell 7)")
ax.set_title(
    "(B) Preset projection onto within-level factors\n"
    "— do within-level directions also discriminate models?"
)
for i in range(_preset_scores_on_W.shape[0]):
    for j in range(_preset_scores_on_W.shape[1]):
        v = _preset_scores_on_W[i, j]
        ax.text(
            j, i, f"{v:+.2f}",
            ha="center", va="center", fontsize=8,
            color="white" if abs(v) > _vmax * 0.55 else "black",
        )
fig.colorbar(im, ax=ax, shrink=0.8, label="preset score (arb. units)")
fig.tight_layout()
_save_fig(fig, "cell10b_preset_on_within_factors")
plt.show()

# Print the 5 × k table of preset scores on within-level factors.
print(
    "\n[between-level] Preset projection onto within-level factors "
    "(centered item means @ Λ_W):"
)
print(
    "  " + f"{'preset':<25s}" +
    "  ".join([f"{'F'+str(i+1):>6s}" for i in range(loadings_W.shape[1])])
)
for i, g in enumerate(_unique_g):
    vals = "  ".join([
        f"{_preset_scores_on_W[i, j]:>+6.2f}"
        for j in range(loadings_W.shape[1])
    ])
    print(f"  {g:<25s}" + vals)

# (C) Hierarchical clustering of presets.
_Z = linkage(_preset_means_centered, method="ward")
fig, ax = plt.subplots(figsize=(10, 4))
dendrogram(_Z, labels=list(_unique_g), ax=ax, leaf_font_size=9)
ax.set_title(
    f"(C) Hierarchical clustering of presets  "
    f"(Ward linkage on centered item means, G={len(_unique_g)})"
)
ax.set_ylabel("Ward distance")
fig.tight_layout()
_save_fig(fig, "cell10c_preset_dendrogram")
plt.show()


#%% Top-K selection sensitivity check
#
# The top-K variance cut (Cell 4) pre-selects items that have high
# within-preset variance. This biases FA toward items that load strongly
# on *something* — the factor structure is conditional on the selection.
# Two diagnostics here:
#   (a) correlation between an item's block-relative residualised variance
#       rank (what the top-K used) and its max |loading| on the residualised
#       FA. If that correlation is near 1 we've essentially hand-picked
#       items to give us strong factor structure.
#   (b) re-fit the residualised FA on the full combined matrix BEFORE the
#       top-K cut and compare. If F1-F3 come out very similar to the
#       top-K solution, the selection didn't meaningfully distort.
#
# Relies on `pair_data` from Cell 3 (pre-top-K combine) being in scope.

print("\n" + "=" * 78)
print("[sensitivity] Top-K selection effect on FA")
print("=" * 78)

# (a) variance-rank vs max-|loading| correlation on the selected set.
from scipy.stats import spearmanr as _spearman
_max_abs_load = np.max(np.abs(resid_fa["loadings"]), axis=1)
# block-relative residualised variance ranking recomputed on the KEPT items,
# which is what survived the cut and entered FA.
from src_dev.factor_analysis.preprocessing import residualize as _residualize_fn
_rm = np.sum(np.isnan(matrix), axis=1) == 0
_m_resid, _, _ = _residualize_fn(
    matrix[_rm],
    [m for m, k in zip(metadata, _rm) if k],
    group_field="rollout_preset_key",
)
_col_var = np.var(_m_resid, axis=0)
_item_blocks_kept = np.array([it.get("block", "") for it in items])
_rel_var = np.zeros_like(_col_var)
for b in np.unique(_item_blocks_kept):
    m_b = _item_blocks_kept == b
    pos = _col_var[m_b][_col_var[m_b] > 0]
    med = float(np.median(pos)) if pos.size else 0.0
    if med > 0:
        _rel_var[m_b] = _col_var[m_b] / med
# Realign to resid_cols ordering.
_id_to_idx_all = {(it.get("item_id") or i): i for i, it in enumerate(items)}
_rel_var_in_resid_cols = np.array([
    _rel_var[_id_to_idx_all[rc.get("item_id")]] for rc in resid_cols
])
rho, pval = _spearman(_rel_var_in_resid_cols, _max_abs_load)
print(
    f"[sensitivity-a] Spearman ρ(block-relative residualised variance rank, "
    f"max |loading| on residualised FA) = {rho:+.3f}  (p={pval:.2e})"
)
print(
    "  Interpretation: if ρ ≈ 1 the factor structure is mostly an artefact of "
    "the selection step; if ρ is moderate (≲0.6) the factors express genuine "
    "covariance beyond 'this item has high variance'."
)

# (b) refit the residualised FA on the full pre-Top-K item set and report
# how many items survive, η² per factor, and Tucker's φ between the two
# solutions' top factors (pairwise congruence).
print("\n[sensitivity-b] Refit residualised FA on the full (pre-Top-K) item set:")

# Rebuild the full combined matrix once from the per-pair outputs.
from src_dev.psychometric.combine import combine_per_pair_outputs as _combine_fn
_matrix_full, _metadata_full, _items_full = _combine_fn(
    pair_data, pair_version,
    out_dir=None,
    provenance_extra={"script": __file__, "sensitivity_check": True},
)
print(
    f"  full matrix: {_matrix_full.shape[0]} rows × {_matrix_full.shape[1]} items"
)

_resid_fa_full, _, _resid_cols_full, _, _prop_var_full, _eta2_full, _k_full = (
    _fit_fa_residualised(
        _matrix_full, _metadata_full, _items_full,
        "full item set [residualised]",
        k_override=resid_k,  # match Cell 7's k so we're comparing apples to apples
    )
)
print(
    f"  factor    prop_var   η²(preset)   band"
)
for f_idx in range(_k_full):
    band = (
        "η²≥0.5" if _eta2_full[f_idx] >= 0.5
        else "η²≥0.2" if _eta2_full[f_idx] >= 0.2
        else "η²<0.2"
    )
    print(
        f"  F{f_idx+1:<5d}  {_prop_var_full[f_idx]:>9.3f}  "
        f"{_eta2_full[f_idx]:>10.3f}  {band}"
    )

# Tucker's φ between Top-K solution and full-set solution, matched on item_id.
from src_dev.psychometric.tucker_congruence import (
    tucker_phi_matrix, align_factors,
)
_ids_k = [rc.get("item_id") for rc in resid_cols]
_ids_f = [rc.get("item_id") for rc in _resid_cols_full]
_shared = [i for i in _ids_k if i in set(_ids_f)]
_idx_k = [_ids_k.index(i) for i in _shared]
_idx_f = [_ids_f.index(i) for i in _shared]
L_k = resid_fa["loadings"][_idx_k, :min(n_show_factors, resid_k)]
L_f = _resid_fa_full["loadings"][_idx_f, :min(n_show_factors, _k_full)]
phi_kf = tucker_phi_matrix(L_k, L_f)
print(
    f"\n[sensitivity-b] Tucker's |φ| between Top-K FA and full-set FA "
    f"(first {L_k.shape[1]} factors, {len(_shared)} shared items):"
)
print("  rows=Top-K factor, cols=full-set factor")
for i in range(phi_kf.shape[0]):
    vals = "  ".join([f"{phi_kf[i, j]:>5.2f}" for j in range(phi_kf.shape[1])])
    print(f"  F{i+1}:  {vals}")

align = align_factors(L_k, L_f)
print("\n  aligned congruences (Top-K F_i → best-matching full-set F_j):")
for a in align:
    tag = (
        "GOOD" if a.phi >= 0.95 else
        "FAIR" if a.phi >= 0.85 else
        "POOR" if a.phi >= 0.70 else
        "UNMATCHED"
    )
    tgt = f"F{a.target_factor+1}" if a.target_factor >= 0 else "—"
    print(
        f"    Top-K F{a.anchor_factor+1} ↔ full-set {tgt}  |φ|={a.phi:.3f}  [{tag}]"
    )


#%% Cross-level congruence — Procrustes Tucker φ between Λ_W and preset-PCA loadings
#
# Transcript advice: "compute the G×k matrix of LLM means, do a Procrustes
# rotation of its loadings onto Λ_W, and report the congruence coefficients."
# With G=5 the between-level PCA is descriptive, but congruence with Λ_W
# tells us whether the within-level factor directions align with the
# between-level discriminating directions — formal cross-level invariance.
#
# Approach: PCA on the 5-row centered preset-means matrix, keep first k
# loadings, Procrustes-rotate onto Λ_W, compute Tucker's |φ| per factor.

from numpy.linalg import svd as _svd

# Center preset means (reuse from Cell 10).
_pm_centered = _preset_means_centered  # (G, n_items)
# Align columns to resid_cols order.
_pm_aligned = _pm_centered[:, _reorder]  # (G, n_kept_items)

# Within-level loadings (already in Cell 7). Columns up to n_show.
_k_compare = min(n_show_factors, resid_k)
Lambda_W = resid_fa["loadings"][:, :_k_compare]  # (n_items, k)

# Between-level PCA: SVD of centered preset-means (G × n_items).
# Right singular vectors are the PCA loadings (directions in item space).
U, S, Vt = _svd(_pm_aligned, full_matrices=False)
Lambda_B = Vt[:_k_compare, :].T  # (n_items, k)
# Sign flip each column to align with Λ_W (Procrustes-style rescue — avoids
# sign ambiguity of SVD).
for j in range(Lambda_B.shape[1]):
    if np.dot(Lambda_B[:, j], Lambda_W[:, j]) < 0:
        Lambda_B[:, j] = -Lambda_B[:, j]

# Orthogonal Procrustes: find R minimising ||Λ_B R − Λ_W||_F.
# Solution: R = U V^T from SVD of Λ_B^T Λ_W.
U_p, _, Vt_p = _svd(Lambda_B.T @ Lambda_W, full_matrices=False)
R = U_p @ Vt_p
Lambda_B_rot = Lambda_B @ R

# Tucker's |φ| per factor between Λ_W and Procrustes-rotated Λ_B.
phi_WB = np.zeros(_k_compare)
for j in range(_k_compare):
    num = float(np.dot(Lambda_W[:, j], Lambda_B_rot[:, j]))
    den = float(
        np.linalg.norm(Lambda_W[:, j]) * np.linalg.norm(Lambda_B_rot[:, j])
    )
    phi_WB[j] = abs(num / den) if den > 0 else 0.0

print("\n" + "=" * 78)
print("[cross-level] Procrustes Tucker φ: Λ_W (within-level) vs Λ_B (between-level PCA)")
print("=" * 78)
print(f"  G={len(_unique_g)}  (descriptive — between-level PCA on 5 preset means)")
for j in range(_k_compare):
    tag = (
        "GOOD (≥0.95)" if phi_WB[j] >= 0.95 else
        "FAIR (≥0.85)" if phi_WB[j] >= 0.85 else
        "POOR (≥0.70)" if phi_WB[j] >= 0.70 else
        "UNMATCHED"
    )
    print(f"  F{j+1}:  |φ| = {phi_WB[j]:.3f}  [{tag}]")
print(
    "\n  High |φ| ⇒ within-level direction aligns with a between-level "
    "discriminating direction\n  ⇒ formal evidence of cross-level invariance "
    "on that factor (descriptive only at G=5)."
)

# Side-by-side heatmap: Λ_W | Λ_B_rot for the first k factors.
fig, axes = plt.subplots(1, 2, figsize=(10, max(6, 0.22 * Lambda_W.shape[0])))
for ax, M, title in [
    (axes[0], Lambda_W,    "Λ_W  (within-level loadings, Cell 7)"),
    (axes[1], Lambda_B_rot, "Λ_B (between-level PCA loadings, Procrustes-rotated)"),
]:
    vmax = float(np.max(np.abs(M)))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels([f"F{i+1}" for i in range(M.shape[1])])
    ax.set_title(title, fontsize=9)
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.6)
axes[0].set_yticks(range(len(resid_cols)))
axes[0].set_yticklabels(
    [(c.get("block", "?")[:3] + ":" + (c.get("item_id") or "")[:25])
     for c in resid_cols],
    fontsize=6,
)
fig.suptitle(
    f"Cross-level loadings — Tucker φ per factor: "
    + "  ".join([f"F{j+1}={phi_WB[j]:.2f}" for j in range(_k_compare)])
)
fig.tight_layout()
_save_fig(fig, "cell11_cross_level_procrustes")
plt.show()


#%% η²(questionnaire) per factor — block-purity / method-factor check
#
# For each factor in the residualised FA, partition the sum of squared
# loadings by block:
#   likert_frac     = Σ_{i∈likert}    L[i,f]² / Σ_i L[i,f]²
#   trait_mcq_frac  = Σ_{i∈trait_mcq} L[i,f]² / Σ_i L[i,f]²
# A factor with one fraction near 1.0 is a method factor (driven purely
# by one instrument). A factor with ~balanced fractions (loadings in both
# blocks) is the more interesting case — a substantive construct that
# transfers across questionnaire formats. This is the direct analogue
# the other reviewer asked for alongside η²(preset).

_block_mask_lik = np.array(
    [rc.get("block") == "likert" for rc in resid_cols], dtype=bool,
)
_block_mask_tm = np.array(
    [rc.get("block") == "trait_mcq" for rc in resid_cols], dtype=bool,
)
_L_residFA = resid_fa["loadings"]  # (n_items, k)
_sq_total = np.sum(_L_residFA ** 2, axis=0)
_sq_lik = np.sum(_L_residFA[_block_mask_lik] ** 2, axis=0)
_sq_tm = np.sum(_L_residFA[_block_mask_tm] ** 2, axis=0)
_lik_frac = _sq_lik / np.where(_sq_total > 0, _sq_total, 1.0)
_tm_frac = _sq_tm / np.where(_sq_total > 0, _sq_total, 1.0)

print("\n" + "=" * 78)
print("[block-purity] η²(questionnaire) analogue per factor — residualised FA")
print("=" * 78)
print(
    f"  ({int(_block_mask_lik.sum())} likert + "
    f"{int(_block_mask_tm.sum())} trait_mcq items; fractions of Σ L² per factor)"
)
print(
    f"  {'factor':<6s} {'likert':>8s} {'trait_mcq':>11s}  "
    f"{'interpretation':<36s}"
)
_block_purity_records: list[dict] = []
for f in range(resid_k):
    if _lik_frac[f] > 0.85:
        interp = "→ LIKERT-only method factor"
    elif _tm_frac[f] > 0.85:
        interp = "→ TRAIT_MCQ-only method factor"
    elif max(_lik_frac[f], _tm_frac[f]) > 0.70:
        interp = "→ block-leaning (not pure method)"
    else:
        interp = "→ CROSS-BLOCK (substantive candidate)"
    print(
        f"  F{f+1:<5d} {_lik_frac[f]:>8.2f} {_tm_frac[f]:>11.2f}  {interp}"
    )
    _block_purity_records.append({
        "factor": f + 1,
        "likert_frac": float(_lik_frac[f]),
        "trait_mcq_frac": float(_tm_frac[f]),
        "prop_var": float(resid_prop_var[f]),
        "eta2_preset": float(resid_eta2[f]),
    })

# Stacked bar plot: fraction of squared loadings by block per factor.
fig, ax = plt.subplots(figsize=(max(6, 0.5 * resid_k + 2), 4.5))
x = np.arange(1, resid_k + 1)
ax.bar(
    x, _lik_frac,
    color="#4C72B0",
    label=f"likert ({int(_block_mask_lik.sum())} items)",
)
ax.bar(
    x, _tm_frac, bottom=_lik_frac,
    color="#DD8452",
    label=f"trait_mcq ({int(_block_mask_tm.sum())} items)",
)
ax.axhline(0.5, color="black", lw=0.5, ls=":")
ax.set_xticks(x)
ax.set_xticklabels([f"F{k}" for k in x])
ax.set_xlabel("factor")
ax.set_ylabel("fraction of Σ L² on block")
ax.set_ylim(0, 1)
ax.set_title(
    "Residualised FA — per-factor block composition\n"
    "(pure = method factor; ~50/50 = cross-block substantive construct)"
)
# Annotate each bar with the prop_var of that factor so reader can weight by importance.
for xi, pv in zip(x, resid_prop_var):
    ax.text(
        xi, 0.02, f"var={pv:.2f}",
        ha="center", va="bottom", fontsize=6, color="white",
    )
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
_save_fig(fig, "cell12_block_purity_per_factor")
plt.show()


#%% Rotation sensitivity: oblimin vs varimax on the residualised FA
#
# Factor correlations Φ in Cell 7b were near-identity (max off-diagonal
# 0.17), so we expect an orthogonal rotation to produce essentially the
# same loading structure. This cell quantifies that: refit the same
# residualised FA with varimax, then report Tucker's |φ| between each
# pair of (oblimin-F_i, varimax-F_j). If max-congruences ≥ 0.95 for every
# factor, orthogonality wasn't costing us structure.

from src_dev.psychometric.tucker_congruence import (
    tucker_phi_matrix as _phi_mat,
    align_factors as _align_fn,
)

print("\n" + "=" * 78)
print("[rotation-sensitivity] oblimin vs varimax on residualised FA")
print("=" * 78)

_resid_fa_vm = run_factor_analysis(
    resid_data, n_factors=resid_k,
    method=FA_METHOD_VIZ, rotation="varimax",
)
L_obl = resid_fa["loadings"]
L_var = _resid_fa_vm["loadings"]

phi_ov = _phi_mat(L_obl, L_var)
align_ov = _align_fn(L_obl, L_var)

print(f"k={resid_k}, shared items={L_obl.shape[0]}")
print("\nAligned congruences (oblimin F_i → best-matching varimax F_j):")
_min_phi, _max_phi = 1.0, 0.0
for a in align_ov:
    tag = (
        "GOOD (≥0.95)" if a.phi >= 0.95 else
        "FAIR (≥0.85)" if a.phi >= 0.85 else
        "POOR (≥0.70)" if a.phi >= 0.70 else
        "UNMATCHED"
    )
    tgt = f"F{a.target_factor+1}" if a.target_factor >= 0 else "—"
    print(
        f"  oblimin F{a.anchor_factor+1} ↔ varimax {tgt}  "
        f"|φ|={a.phi:.3f}  [{tag}]"
    )
    _min_phi = min(_min_phi, a.phi)
    _max_phi = max(_max_phi, a.phi)
print(
    f"\nRange of aligned |φ|: [{_min_phi:.3f}, {_max_phi:.3f}]  "
    f"(all ≥ 0.95 ⇒ rotation choice doesn't matter for this data)"
)

# Full Tucker matrix heatmap for visual inspection.
fig, ax = plt.subplots(figsize=(0.45 * resid_k + 2, 0.45 * resid_k + 2))
im = ax.imshow(phi_ov, cmap="viridis", vmin=0, vmax=1)
for i in range(resid_k):
    for j in range(resid_k):
        ax.text(
            j, i, f"{phi_ov[i, j]:.2f}",
            ha="center", va="center", fontsize=6,
            color="white" if phi_ov[i, j] < 0.6 else "black",
        )
ax.set_xticks(range(resid_k))
ax.set_yticks(range(resid_k))
ax.set_xticklabels([f"F{j+1}" for j in range(resid_k)], fontsize=7)
ax.set_yticklabels([f"F{j+1}" for j in range(resid_k)], fontsize=7)
ax.set_xlabel("varimax factor")
ax.set_ylabel("oblimin factor")
ax.set_title(
    f"Tucker's |φ| — oblimin vs varimax  (residualised FA, k={resid_k})"
)
fig.colorbar(im, ax=ax, shrink=0.8, label="|φ|")
fig.tight_layout()
_save_fig(fig, "cell13_rotation_sensitivity")
plt.show()


#%% Polychoric / tetrachoric / polyserial sensitivity check
#
# Cell 7's FA uses Pearson correlations. For coarse-categorical data (Likert
# 1-5, trait_mcq binary 0/1) with unequal marginals, Pearson correlations are
# systematically attenuated relative to the latent bivariate-normal
# correlation. The principled alternative — polychoric (ordinal × ordinal),
# tetrachoric (binary × binary, special case), and polyserial (ordinal ×
# binary) correlations — all estimate the latent correlation under a
# bivariate-normal-with-thresholds model.
#
# We DIY a 2-step estimator using only numpy + scipy (no new deps):
#   Step A — thresholds τ_k = Φ⁻¹(cumulative marginal).
#   Step B — MLE over ρ only, thresholds fixed from Step A:
#       logL(ρ) = Σ_{a,b} N[a,b] · log P(a, b | ρ)
#       P(a, b | ρ) is the "rectangle probability" from Φ₂ evaluated at the
#       four corners (τ_{i,a}, τ_{j,b}), ..., (τ_{i,a+1}, τ_{j,b+1}).
#
# The single implementation handles all three cases (polychoric /
# tetrachoric / polyserial) — only the number of thresholds on each side
# differs.
#
# **Important subtlety**: polychoric requires INTEGER ordinal responses
# (the model assumes ordinal levels with latent bivariate-normal). The
# residualised matrix in Cell 7 is continuous floats, where polychoric is
# ill-defined. We therefore operate on the **raw integer-rounded `matrix`**,
# and compare polychoric FA to Pearson FA on the *same* raw matrix. (This
# is a slight deviation from a strict Cell-7 comparison but it's the right
# apples-to-apples — polychoric on residualised floats would be garbage.)
#
# Sanity: if Tucker's |φ| ≥ 0.95 on every aligned (pearson, polychoric)
# factor pair, the choice of correlation estimator doesn't affect the
# factor structure — we can keep Pearson and cite this sensitivity. If it
# diverges on F1-F3, the reviewer's concern would have mattered.

import time as _time

from scipy.optimize import minimize_scalar as _minimize_scalar
from scipy.stats import (
    multivariate_normal as _mvn,
    norm as _norm,
)
from factor_analyzer import FactorAnalyzer as _FactorAnalyzerCorr

from src_dev.psychometric.tucker_congruence import (
    tucker_phi_matrix as _phi_mat_c14,
    align_factors as _align_fn_c14,
)

print("\n" + "=" * 78)
print("[polychoric] Sensitivity check — polychoric/tetrachoric/polyserial FA")
print("=" * 78)


# ── helpers ────────────────────────────────────────────────────────────

def _poly_thresholds(x_int: np.ndarray, levels: list[int]) -> np.ndarray:
    """Estimate K thresholds on the standard normal from marginal proportions.

    Args:
        x_int: integer-rounded ordinal observations (NaNs already removed).
        levels: ordered list of possible level values.

    Returns: array of K = len(levels) - 1 thresholds (τ_1, ..., τ_{K})
    in the open real interval. Clamped to avoid ±inf on edges.
    """
    n = int(x_int.size)
    counts = np.array([(x_int == lv).sum() for lv in levels], dtype=float)
    cum = np.cumsum(counts) / n
    taus = []
    for k in range(len(levels) - 1):
        p = float(cum[k])
        p = min(max(p, 1e-9), 1 - 1e-9)  # clamp to avoid Φ⁻¹(0 or 1)
        taus.append(_norm.ppf(p))
    return np.array(taus)


def _bvn_cdf_c14(h: float, k: float, rho: float) -> float:
    """Bivariate standard normal CDF Φ₂(Z1 ≤ h, Z2 ≤ k; ρ), handling ±inf."""
    if np.isinf(h) and h > 0 and np.isinf(k) and k > 0:
        return 1.0
    if (np.isinf(h) and h < 0) or (np.isinf(k) and k < 0):
        return 0.0
    if np.isinf(h) and h > 0:
        return float(_norm.cdf(k))
    if np.isinf(k) and k > 0:
        return float(_norm.cdf(h))
    cov = np.array([[1.0, rho], [rho, 1.0]])
    return float(_mvn(mean=[0.0, 0.0], cov=cov).cdf([h, k]))


def _polychoric_pair(
    x: np.ndarray, y: np.ndarray,
    levels_x: list[int], levels_y: list[int],
) -> float:
    """2-step polychoric/tetrachoric/polyserial estimator for a single pair.

    Uses pairwise deletion (rows with NaN in either item are dropped). Returns
    np.nan if either item is constant on the pairwise-clean subset.
    """
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    if x.size < 3:
        return float("nan")
    # Round to integers (soft logprob scores can have non-integer values).
    x_int = np.round(x).astype(int)
    y_int = np.round(y).astype(int)
    # Clip to the declared level range (defensive).
    x_int = np.clip(x_int, levels_x[0], levels_x[-1])
    y_int = np.clip(y_int, levels_y[0], levels_y[-1])
    if len(np.unique(x_int)) < 2 or len(np.unique(y_int)) < 2:
        return float("nan")
    tau_x = _poly_thresholds(x_int, levels_x)
    tau_y = _poly_thresholds(y_int, levels_y)
    # Contingency table N[a, b] = count at (level_a of x, level_b of y).
    Kx, Ky = len(levels_x), len(levels_y)
    x_to_idx = {v: i for i, v in enumerate(levels_x)}
    y_to_idx = {v: i for i, v in enumerate(levels_y)}
    N = np.zeros((Kx, Ky), dtype=float)
    for a_lv, b_lv in zip(x_int, y_int):
        N[x_to_idx[int(a_lv)], y_to_idx[int(b_lv)]] += 1
    # Augment thresholds with ±inf so P(a,b) = Φ₂(t_{a+1}, t_{b+1}) − ...
    tx = np.concatenate([[-np.inf], tau_x, [np.inf]])
    ty = np.concatenate([[-np.inf], tau_y, [np.inf]])
    eps = 1e-12

    def neg_ll(rho: float) -> float:
        ll = 0.0
        for a in range(Kx):
            for b in range(Ky):
                if N[a, b] == 0:
                    continue
                p = (
                    _bvn_cdf_c14(tx[a + 1], ty[b + 1], rho)
                    - _bvn_cdf_c14(tx[a],     ty[b + 1], rho)
                    - _bvn_cdf_c14(tx[a + 1], ty[b],     rho)
                    + _bvn_cdf_c14(tx[a],     ty[b],     rho)
                )
                p = max(p, eps)
                ll += N[a, b] * np.log(p)
        return -ll

    res = _minimize_scalar(
        neg_ll, bounds=(-0.995, 0.995), method="bounded",
        options={"xatol": 1e-4},
    )
    return float(res.x)


# ── build the polychoric correlation matrix on the raw integer data ────

# Per-item possible levels, from canonical encodings.
def _levels_for_item(it: dict) -> list[int]:
    enc = it.get("encoding", "")
    block = it.get("block", "")
    if enc == "1-5" or block == "likert":
        return [1, 2, 3, 4, 5]
    if enc == "trait_aligned_0-1" or block == "trait_mcq":
        return [0, 1]
    # Defensive fallback — infer from observed values.
    return None  # sentinel; handled below


n_items_c14 = matrix.shape[1]
levels_per_item: list[list[int]] = []
for _col_i, _it in enumerate(items):
    lv = _levels_for_item(_it)
    if lv is None:
        col = matrix[:, _col_i]
        col = col[~np.isnan(col)]
        if col.size == 0:
            lv = [0, 1]
        else:
            lv = sorted(set(int(round(v)) for v in col))
            if len(lv) < 2:
                lv = [0, 1]
    levels_per_item.append(lv)

# Pearson corr on the raw (NaN-aware) matrix — pairwise for parity.
def _pearson_pair(x: np.ndarray, y: np.ndarray) -> float:
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 3:
        return float("nan")
    xv, yv = x[valid], y[valid]
    if xv.std() == 0 or yv.std() == 0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


print(f"[polychoric] computing pairwise correlations over {n_items_c14} items "
      f"({n_items_c14 * (n_items_c14 - 1) // 2} pairs) ...")
_t0_c14 = _time.time()

R_poly = np.eye(n_items_c14, dtype=float)
R_pear_pw = np.eye(n_items_c14, dtype=float)
pair_types = np.empty((n_items_c14, n_items_c14), dtype=object)

_blocks_c14 = [it.get("block", "?") for it in items]
_n_pairs_total = n_items_c14 * (n_items_c14 - 1) // 2
_n_done = 0
_n_fallback = 0
_progress_step = max(1, _n_pairs_total // 10)

for _i in range(n_items_c14):
    for _j in range(_i + 1, n_items_c14):
        _x = matrix[:, _i]
        _y = matrix[:, _j]
        _bi, _bj = _blocks_c14[_i], _blocks_c14[_j]
        if _bi == "likert" and _bj == "likert":
            _pt = "likert×likert"
        elif _bi == "trait_mcq" and _bj == "trait_mcq":
            _pt = "trait_mcq×trait_mcq"
        else:
            _pt = "likert×trait_mcq"
        pair_types[_i, _j] = _pt
        pair_types[_j, _i] = _pt

        _rho_pear = _pearson_pair(_x, _y)
        R_pear_pw[_i, _j] = R_pear_pw[_j, _i] = (
            _rho_pear if not np.isnan(_rho_pear) else 0.0
        )

        _rho_poly = _polychoric_pair(
            _x, _y, levels_per_item[_i], levels_per_item[_j],
        )
        if np.isnan(_rho_poly):
            # fall back to Pearson for this cell
            _rho_poly = _rho_pear if not np.isnan(_rho_pear) else 0.0
            _n_fallback += 1
        # clip to [-0.99, 0.99] to keep the corr matrix well-behaved for FA.
        _rho_poly = float(np.clip(_rho_poly, -0.99, 0.99))
        R_poly[_i, _j] = R_poly[_j, _i] = _rho_poly

        _n_done += 1
        if _n_done % _progress_step == 0:
            _elapsed = _time.time() - _t0_c14
            print(
                f"  progress: {_n_done}/{_n_pairs_total} pairs "
                f"({100 * _n_done / _n_pairs_total:.0f}%)  "
                f"elapsed={_elapsed:.1f}s"
            )

_t1_c14 = _time.time()
print(
    f"[polychoric] done in {_t1_c14 - _t0_c14:.1f}s  "
    f"(n_pairs={_n_done}, n_fallback_to_pearson={_n_fallback})"
)

# Mean |polychoric − pearson| over off-diagonal entries.
_iu = np.triu_indices(n_items_c14, k=1)
_diffs = np.abs(R_poly[_iu] - R_pear_pw[_iu])
print(
    f"[polychoric] mean |polychoric − pearson| off-diagonal: "
    f"{float(np.mean(_diffs)):.4f}  "
    f"max={float(np.max(_diffs)):.4f}"
)

# Ensure positive-semi-definite for FA by **eigenvalue clipping**, not
# additive-diagonal-nudge. Pairwise polychoric estimation commonly produces
# indefinite matrices; the right repair is to zero-floor the negative
# eigenvalues and re-normalise. The naïve "add |min_eig|·I then renormalise"
# approach crushes off-diagonals by a factor of (1 + nudge) — it destroys
# the signal that polychoric was supposed to recover.
R_poly = (R_poly + R_poly.T) / 2.0
_eigs_pre = np.linalg.eigvalsh(R_poly)
_min_eig_pre = float(_eigs_pre.min())
if _min_eig_pre < 1e-6:
    _evals, _evecs = np.linalg.eigh(R_poly)
    _evals_clipped = np.maximum(_evals, 1e-6)
    R_poly = _evecs @ np.diag(_evals_clipped) @ _evecs.T
    R_poly = (R_poly + R_poly.T) / 2.0  # kill floating-point asymmetry
    _d = np.sqrt(np.clip(np.diag(R_poly), 1e-12, None))
    R_poly = R_poly / np.outer(_d, _d)
    print(
        f"[polychoric] eigenvalue-clipped PSD: min_eig {_min_eig_pre:.2e} → "
        f"{float(np.linalg.eigvalsh(R_poly).min()):.2e}  "
        f"(max |off-diag| after clip: "
        f"{float(np.max(np.abs(R_poly - np.eye(n_items_c14)))):.3f})"
    )


# ── FA on the polychoric correlation matrix + apples-to-apples Pearson baseline ──

# Clean listwise-NaN rows for the raw-Pearson FA baseline (so that
# `run_factor_analysis` has a consistent input).
_row_mask_c14 = np.sum(np.isnan(matrix), axis=1) == 0
matrix_cleaned = matrix[_row_mask_c14]
print(
    f"\n[polychoric] raw-data FA baseline: "
    f"{matrix_cleaned.shape[0]}/{matrix.shape[0]} rows after listwise NaN drop"
)

# factor_analyzer's PAF ("principal") only works on raw data, not precomputed
# correlations. For this sensitivity check we fall back to "minres" (minimum
# residual) for BOTH the polychoric and the Pearson baselines so the
# comparison stays apples-to-apples. minres is a common-factor method (not
# PCA) and produces near-identical loadings to PAF in practice on large n.
_METHOD_FOR_POLY_SENSITIVITY = "minres"
print(
    f"[polychoric] Note: factor_analyzer's 'principal' method doesn't support "
    f"is_corr_matrix=True; refitting BOTH baselines with method="
    f"{_METHOD_FOR_POLY_SENSITIVITY!r} for an apples-to-apples comparison."
)

raw_fa = run_factor_analysis(
    matrix_cleaned, n_factors=resid_k,
    method=_METHOD_FOR_POLY_SENSITIVITY, rotation=ROTATION_VIZ,
)
L_pear = raw_fa["loadings"]

# FA on the polychoric correlation matrix (via factor_analyzer's
# is_corr_matrix=True path — the one `run_factor_analysis` doesn't expose).
_fa_poly = _FactorAnalyzerCorr(
    n_factors=resid_k,
    method=_METHOD_FOR_POLY_SENSITIVITY,
    rotation=ROTATION_VIZ,
    is_corr_matrix=True,
)
_fa_poly.fit(R_poly)
L_poly = _fa_poly.loadings_
_variance_poly = _fa_poly.get_factor_variance()
_prop_var_poly = _variance_poly[1]

print(
    f"[polychoric] L_poly shape={L_poly.shape}  L_pear shape={L_pear.shape}"
)


# ── Tucker's |φ| between the two loadings ─────────────────────────────

phi_pp = _phi_mat_c14(L_pear, L_poly)
align_pp = _align_fn_c14(L_pear, L_poly)

print("\n[polychoric] Tucker's |φ| matrix (rows=Pearson factors, "
      "cols=polychoric factors):")
_hdr = "       " + "  ".join([f"{'poly-F'+str(j+1):>7s}" for j in range(resid_k)])
print(_hdr)
for i in range(resid_k):
    vals = "  ".join([f"{phi_pp[i, j]:>7.2f}" for j in range(resid_k)])
    print(f"  pear-F{i+1:<2d} {vals}")

print("\n[polychoric] aligned congruences (Pearson F_i → best-match polychoric F_j):")
for a in align_pp:
    tag = (
        "GOOD (≥0.95)"  if a.phi >= 0.95 else
        "FAIR (≥0.85)"  if a.phi >= 0.85 else
        "POOR (≥0.70)"  if a.phi >= 0.70 else
        "UNMATCHED"
    )
    tgt = f"F{a.target_factor+1}" if a.target_factor >= 0 else "—"
    print(
        f"  pearson F{a.anchor_factor+1} ↔ polychoric {tgt}  "
        f"|φ|={a.phi:.3f}  [{tag}]"
    )

# Per-factor summary in the Cell-13 style.
print("\n[polychoric] per-factor summary (aligned):")
print(
    f"  {'factor':<7s} {'pearson_pv':>11s} {'poly_pv':>9s} "
    f"{'aligned|φ|':>11s}  {'verdict':<14s}"
)
for a in align_pp:
    i = a.anchor_factor
    j = a.target_factor
    pv_p = float(raw_fa["proportion_variance"][i])
    pv_q = float(_prop_var_poly[j]) if j >= 0 else float("nan")
    tag = (
        "GOOD" if a.phi >= 0.95 else
        "FAIR" if a.phi >= 0.85 else
        "POOR" if a.phi >= 0.70 else
        "UNMATCHED"
    )
    print(
        f"  F{i+1:<6d} {pv_p:>11.3f} {pv_q:>9.3f} "
        f"{a.phi:>11.3f}  {tag:<14s}"
    )


# ── Figure 1: scatter of polychoric vs Pearson off-diagonal correlations ──

fig, ax = plt.subplots(figsize=(6, 6))
_color_map = {
    "likert×likert":        "#4C72B0",
    "trait_mcq×trait_mcq":  "#DD8452",
    "likert×trait_mcq":     "#55A868",
}
for _pt_name, _col in _color_map.items():
    _mask = np.array([pair_types[i, j] == _pt_name for i, j in zip(*_iu)])
    if _mask.sum() == 0:
        continue
    ax.scatter(
        R_pear_pw[_iu][_mask],
        R_poly[_iu][_mask],
        s=14, alpha=0.55, color=_col,
        label=f"{_pt_name} (n={int(_mask.sum())})",
        edgecolors="none",
    )
_lim = max(
    float(np.max(np.abs(R_pear_pw[_iu]))),
    float(np.max(np.abs(R_poly[_iu]))),
    0.1,
)
ax.plot([-_lim, _lim], [-_lim, _lim], "k--", lw=0.5, label="y = x")
ax.axhline(0, color="gray", lw=0.3)
ax.axvline(0, color="gray", lw=0.3)
ax.set_xlabel("Pearson ρ (pairwise)")
ax.set_ylabel("Polychoric / tetrachoric / polyserial ρ")
ax.set_title(
    f"Polychoric vs Pearson off-diagonal correlations  "
    f"(n_items={n_items_c14}, n_pairs={_n_pairs_total})"
)
ax.legend(fontsize=8, loc="lower right")
ax.set_aspect("equal", adjustable="box")
fig.tight_layout()
_save_fig(fig, "cell14_polychoric_vs_pearson_corr_scatter")
plt.show()


# ── Figure 2: Tucker |φ| heatmap (Pearson factors × polychoric factors) ──

fig, ax = plt.subplots(figsize=(0.45 * resid_k + 2, 0.45 * resid_k + 2))
im = ax.imshow(phi_pp, cmap="viridis", vmin=0, vmax=1)
for i in range(resid_k):
    for j in range(resid_k):
        ax.text(
            j, i, f"{phi_pp[i, j]:.2f}",
            ha="center", va="center", fontsize=6,
            color="white" if phi_pp[i, j] < 0.6 else "black",
        )
ax.set_xticks(range(resid_k))
ax.set_yticks(range(resid_k))
ax.set_xticklabels([f"F{j+1}" for j in range(resid_k)], fontsize=7)
ax.set_yticklabels([f"F{j+1}" for j in range(resid_k)], fontsize=7)
ax.set_xlabel("polychoric factor")
ax.set_ylabel("Pearson factor")
ax.set_title(
    f"Tucker's |φ| — Pearson vs polychoric FA  (raw data, k={resid_k})"
)
fig.colorbar(im, ax=ax, shrink=0.8, label="|φ|")
fig.tight_layout()
_save_fig(fig, "cell14_polychoric_vs_pearson_tucker")
plt.show()

_min_phi_pp = min(a.phi for a in align_pp if not np.isnan(a.phi))
_max_phi_pp = max(a.phi for a in align_pp if not np.isnan(a.phi))
print(
    f"\n[polychoric] Range of aligned |φ|: "
    f"[{_min_phi_pp:.3f}, {_max_phi_pp:.3f}]  "
    f"(all ≥ 0.95 ⇒ correlation estimator doesn't matter for the factor structure)"
)
print("=" * 78)

