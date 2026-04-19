"""Stage 3 — factor analysis on the response matrix.

Runs preprocessing → adequacy tests → Horn's parallel analysis →
(MAP/EKC/acceleration/Kaiser/CV triangulation) → PAF + rotations for each
``residualize`` option, plus two sub-passes:

* Per-block subset FA — rerun on each block's columns alone so we can check
  whether the pooled factor structure replicates within-block. Only
  triggered when the questionnaire carries multiple blocks and
  ``cfg.fa_per_block_passes`` is True.

* Trait-oriented FA — parallel run on the trait-direction score matrix
  (built from ``raw_responses.jsonl`` via
  :func:`src_dev.factor_analysis.trait_alignment.build_trait_oriented_matrix`)
  so signed loadings are trait-interpretable. Only triggered when the
  questionnaire has trait_mcq items.

Every produced result entry carries ``fa_result``, ``column_defs``,
``metadata``, ``data``, ``n_factors``, ``parallel_analysis``, ``save_dir``,
``encoding``, and (optionally) ``block`` — the exact shape the labelling +
validation stages consume downstream.

After all FA fits, the stage generates diagnostic plots and the initial
factor-extremes HTML (without LLM labels — the labelling stage re-exports
afterwards with label caches attached).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src_dev.factor_analysis.factor_analysis import (
    adequacy_tests,
    run_factor_analysis,
)
from src_dev.factor_analysis.n_factors import (
    plot_n_factors_comparison,
    suggest_n_factors,
)
from src_dev.factor_analysis.parallel_analysis import parallel_analysis
from src_dev.factor_analysis.persistence import save_factor_analysis
from src_dev.factor_analysis.trait_alignment import (
    build_trait_oriented_matrix,
    compute_factor_trait_alignment,
    plot_all_alignment,
    save_alignment,
)
from src_dev.psychometric.config import (
    FactorAnalysisStageConfig,
    FactorAnalysisStageResult,
)
from src_dev.psychometric.factor_extremes_html import export_factor_extremes_html
from src_dev.psychometric.fa_plots import plot_fa_visualisations
from src_dev.psychometric.preprocessing import preprocess_response_matrix
from src_dev.psychometric.trait_aware_plots import plot_trait_aware_fa_visualisations


def _jsonable(obj):
    """Recursively convert numpy arrays / scalars to plain Python for json.dump."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _plot_parallel_analysis(
    real_eigenvalues: np.ndarray,
    random_threshold: np.ndarray,
    n_recommended: int,
    label: str,
    save_path: Path,
    max_components: int = 30,
) -> None:
    """Plot Horn's parallel analysis scree plot and save to PNG."""
    import matplotlib.pyplot as plt

    n = min(len(real_eigenvalues), max_components)
    x = np.arange(1, n + 1)
    real = np.asarray(real_eigenvalues)[:n]
    rand = np.asarray(random_threshold)[:n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, real, "o-", color="#2563eb", linewidth=2, markersize=5, label="Actual eigenvalues")
    ax.plot(x, rand, "s--", color="#dc2626", linewidth=1.5, markersize=4, label="95th percentile (random)")

    if n_recommended > 0:
        ax.axvspan(0.5, n_recommended + 0.5, alpha=0.08, color="#2563eb")
        ax.axvline(n_recommended + 0.5, color="#6b7280", linestyle=":", linewidth=1)
        ax.text(
            n_recommended + 0.5, ax.get_ylim()[1] * 0.95,
            f"  {n_recommended} factors",
            fontsize=11, color="#374151", va="top",
        )

    ax.axhline(1.0, color="#9ca3af", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Component", fontsize=12)
    ax.set_ylabel("Eigenvalue", fontsize=12)
    ax.set_title(f"Horn's Parallel Analysis — {label}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0.5, n + 0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved parallel analysis plot: {save_path}")


def run_stage_factor_analysis(
    cfg: FactorAnalysisStageConfig,
    response_matrix: np.ndarray,
    metadata: list[dict],
    column_defs: list[dict],
    items: list[dict] | None = None,
    *,
    seed: int,
    questionnaire_path: Path,
    rollout_dirs: list[Path] | tuple[Path, ...],
    labeling_dir: Path | None = None,
    n_factors_suggest_methods: tuple[str, ...] = (
        "parallel", "map", "ekc", "acceleration", "kaiser",
    ),
    n_factors_suggest_k_max: int = 15,
    n_factors_suggest_cv_n_folds: int = 5,
    n_factors_suggest_pa_iterations: int = 100,
) -> FactorAnalysisStageResult:
    """Run factor analysis on the response matrix.

    Args:
        cfg: FA stage config (method / rotations / residualize options /
            filtering thresholds / block toggles).
        response_matrix: Persona × item response matrix from Stage 2.
        metadata: Per-row metadata from Stage 2.
        column_defs: Per-column definitions from Stage 2.
        items: Full questionnaire items list. Used to build the trait-
            oriented FA pass (it needs per-item dimensions + text).
        seed: Random seed used for parallel analysis + n_factors triangulation.
        questionnaire_path: Path to the questionnaire JSON — consulted by the
            trait-oriented pass to check for trait_mcq items and pull per-
            item dimension labels.
        rollout_dirs: Rollout dirs passed through to the factor-extremes HTML
            export so it can locate conversation transcripts.
        labeling_dir: Optional pre-existing label cache dir. Passed through to
            the initial factor-extremes HTML export (typically None since
            labels are added later by the labelling stage).
        n_factors_suggest_methods, n_factors_suggest_k_max,
        n_factors_suggest_cv_n_folds, n_factors_suggest_pa_iterations:
            Parameters for the multi-method n-factor triangulation step.
            Reported only — Horn (or ``cfg.n_factors_override``) remains the
            selector.
    """
    base_dir = cfg.ctx.effective_questionnaire_dir / "factor_analysis"
    base_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict = {}

    for do_residualize in cfg.residualize_options:
        resid_label = "residualized" if do_residualize else "raw"

        print(f"\n[Stage 3] Factor analysis ({resid_label})")
        print("=" * 60)

        variance_export_path = (
            base_dir / resid_label / "item_variances_ranked.jsonl"
        )
        data, meta_filtered, cols_filtered, group_ids = preprocess_response_matrix(
            response_matrix, metadata, column_defs,
            min_item_variance=cfg.min_item_variance,
            high_variance_persona_drop_pct=cfg.high_variance_persona_drop_pct,
            do_residualize=do_residualize,
            variance_export_path=variance_export_path,
        )

        if do_residualize and group_ids is None:
            print(f"  Residualization skipped — {resid_label} analysis would duplicate raw. Skipping.")
            all_results[resid_label] = {"n_factors": 0, "note": "residualization not applicable"}
            continue

        # Adequacy tests
        print("\n  Adequacy tests:")
        adeq = adequacy_tests(data)

        # Parallel analysis (Horn's method) with permutation null — preserves
        # marginal distributions while destroying cross-column dependence
        # (correct for mixed-scale ordinal/bounded columns).
        print("\n  Parallel analysis (permutation null):")
        pa_result = parallel_analysis(data, random_state=seed, method="permutation")
        n_factors_horn = pa_result["n_recommended"]

        pa_dir = base_dir / resid_label
        pa_dir.mkdir(parents=True, exist_ok=True)
        print("\n  Additional n-factor methods (reported, not used as selector):")
        try:
            n_factors_suggest = suggest_n_factors(
                data,
                methods=n_factors_suggest_methods,
                k_max=min(n_factors_suggest_k_max, max(1, data.shape[1] - 1)),
                parallel_n_iterations=n_factors_suggest_pa_iterations,
                cv_n_folds=n_factors_suggest_cv_n_folds,
                fa_method=cfg.method,
                seed=seed,
                verbose=True,
            )
            with open(pa_dir / "n_factors_suggest.json", "w") as f:
                json.dump(_jsonable(n_factors_suggest), f, indent=2)
            plot_n_factors_comparison(
                n_factors_suggest,
                save_path=pa_dir / "n_factors_suggest.png",
            )
        except Exception as exc:
            print(f"    suggest_n_factors failed: {exc}")
            n_factors_suggest = None

        n_factors = n_factors_horn
        if cfg.n_factors_override is not None:
            print(
                f"  Override: using {cfg.n_factors_override} factors "
                f"(Horn's recommended {n_factors_horn})"
            )
            n_factors = cfg.n_factors_override

        with open(pa_dir / "parallel_analysis.json", "w") as f:
            json.dump({
                "n_recommended": n_factors,
                "n_recommended_horn": int(n_factors_horn),
                "real_eigenvalues": pa_result["real_eigenvalues"].tolist(),
                "random_threshold": pa_result["random_threshold"].tolist(),
                "adequacy": {
                    "bartlett_chi2": adeq["bartlett_chi2"],
                    "bartlett_p": adeq["bartlett_p"],
                    "kmo_overall": adeq["kmo_overall"],
                },
            }, f, indent=2)

        _plot_parallel_analysis(
            pa_result["real_eigenvalues"],
            pa_result["random_threshold"],
            n_factors,
            resid_label,
            pa_dir / "parallel_analysis.png",
        )

        if n_factors == 0:
            print(f"  No factors recommended for {resid_label} — skipping FA.")
            all_results[resid_label] = {"n_factors": 0, "parallel_analysis": pa_result}
            continue

        # Run factor analysis with each rotation
        for rotation in cfg.rotations:
            print(f"\n  Factor analysis: {n_factors} factors, rotation={rotation}")
            fa_result = run_factor_analysis(
                data, n_factors=n_factors, method=cfg.method, rotation=rotation,
            )

            fa_path = pa_dir / f"fa_{n_factors}_{cfg.method}_{rotation}"
            save_factor_analysis(
                fa_result, fa_path,
                config={
                    "n_factors": n_factors,
                    "method": cfg.method,
                    "rotation": rotation,
                    "residualized": do_residualize,
                    "n_samples": data.shape[0],
                    "n_cols": data.shape[1],
                },
            )

            with open(str(fa_path) + "_item_labels.json", "w") as f:
                json.dump([
                    {
                        "col_id": col["col_id"],
                        "text": col["text"],
                        "block": col["block"],
                        "dimension": col.get("dimension"),
                        "reverse_keyed": col.get("reverse_keyed", False),
                    }
                    for col in cols_filtered
                ], f, indent=2, ensure_ascii=False)

            # Factor–trait (OCEAN) alignment analysis. Only meaningful when
            # every FA row carries a primary_dimension label. Signed mean
            # loading is not trait-interpretable under letter-encoding, but
            # top-K counts and |mean loading| by trait remain valid.
            item_dims_all = [col.get("dimension") for col in cols_filtered]
            if all(d is not None for d in item_dims_all):
                ocean_canonical = [
                    "openness", "conscientiousness", "extraversion",
                    "agreeableness", "neuroticism",
                ]
                present = [t for t in ocean_canonical if t in item_dims_all]
                extras = sorted(set(item_dims_all) - set(present))
                trait_order = present + extras

                alignment = compute_factor_trait_alignment(
                    loadings=fa_result["loadings"],
                    item_dims=item_dims_all,
                    trait_order=trait_order,
                    top_k=min(20, len(item_dims_all)),
                )
                align_dir = pa_dir / f"fa_{n_factors}_{cfg.method}_{rotation}_alignment"
                save_alignment(alignment, align_dir)
                plot_all_alignment(
                    alignment, align_dir,
                    title_prefix=f"{resid_label} / {rotation}",
                )
                print(
                    f"  [Alignment] {rotation}: factor→trait winners "
                    f"(top-{alignment.top_k}):"
                )
                for fi, label in enumerate(alignment.factor_labels):
                    counts = alignment.top_k_count[fi]
                    best = int(np.argmax(counts))
                    full = dict(zip(
                        alignment.trait_order,
                        [int(c) for c in counts],
                    ))
                    print(
                        f"    {label}: {alignment.trait_order[best]} "
                        f"({int(counts[best])}/{alignment.top_k}) — {full}"
                    )
            else:
                print(
                    f"  [Alignment] Skipped for {rotation}: "
                    f"{sum(d is None for d in item_dims_all)} / "
                    f"{len(item_dims_all)} items lack a primary_dimension."
                )

            key = f"{resid_label}_{rotation}"
            all_results[key] = {
                "fa_result": fa_result,
                "column_defs": cols_filtered,
                "metadata": meta_filtered,
                "data": data,
                "n_factors": n_factors,
                "parallel_analysis": pa_result,
                "save_dir": base_dir / key,
                "encoding": "letter",
            }

    # Per-block FA passes.
    block_names = sorted({str(c.get("block", "")) for c in column_defs if c.get("block")})
    if cfg.fa_per_block_passes and len(block_names) >= 2:
        for block in block_names:
            print(f"\n[Stage 3] Per-block FA pass: block={block!r}")
            print("=" * 60)
            keep_col_idx = [
                i for i, c in enumerate(column_defs) if str(c.get("block", "")) == block
            ]
            if len(keep_col_idx) < 3:
                print(f"  Skipping block={block!r}: only {len(keep_col_idx)} columns.")
                continue
            sub_matrix = response_matrix[:, keep_col_idx]
            sub_cols = [column_defs[i] for i in keep_col_idx]
            per_block_results = _run_block_subset_fa_pass(
                cfg,
                response_matrix=sub_matrix,
                metadata=metadata,
                column_defs=sub_cols,
                block_name=block,
                base_dir=base_dir / "per_block" / block,
                seed=seed,
            )
            all_results.update(per_block_results)

    # Trait-oriented FA pass.
    trait_oriented_results = _run_trait_oriented_fa_pass(
        cfg,
        metadata=metadata,
        items=items,
        seed=seed,
        questionnaire_path=questionnaire_path,
    )
    all_results.update(trait_oriented_results)

    # Generate visualisations and factor-extreme HTML exports for each FA result
    for key, result in all_results.items():
        if result.get("n_factors", 0) == 0 or "fa_result" not in result:
            continue
        result_dir = Path(result.get("save_dir", base_dir / key))
        viz_dir = result_dir / "plots"
        viz_dir.mkdir(parents=True, exist_ok=True)
        plot_fa_visualisations(
            fa_result=result["fa_result"],
            column_defs=result["column_defs"],
            metadata=result["metadata"],
            data=result["data"],
            label=key,
            save_dir=viz_dir,
            rollout_dir=cfg.ctx.rollout_dir,
        )
        export_factor_extremes_html(
            fa_result=result["fa_result"],
            column_defs=result["column_defs"],
            metadata=result["metadata"],
            label=key,
            save_dir=result_dir,
            rollout_dirs=rollout_dirs,
            labeling_dir=labeling_dir,
        )

        # Trait-aware views across all items (not just top-K). Only meaningful
        # when every FA row carries a primary_dimension. Letter-encoded
        # results get a signed-caveat note.
        cols = result["column_defs"]
        item_dims_all = [col.get("dimension") for col in cols]
        if all(d is not None for d in item_dims_all):
            is_trait_oriented = result.get("encoding") == "trait_oriented"
            plot_trait_aware_fa_visualisations(
                loadings=result["fa_result"]["loadings"],
                item_dims=item_dims_all,
                save_dir=viz_dir,
                label=key,
                top_k=min(20, len(item_dims_all)),
                signed_caveat=(
                    None if is_trait_oriented
                    else "letter-encoded: sign not trait-interpretable"
                ),
            )

    return FactorAnalysisStageResult(
        output_dir=base_dir,
        results_by_rotation=all_results,
    )


def _run_block_subset_fa_pass(
    cfg: FactorAnalysisStageConfig,
    *,
    response_matrix: np.ndarray,
    metadata: list[dict],
    column_defs: list[dict],
    block_name: str,
    base_dir: Path,
    seed: int,
) -> dict:
    """Run the standard preprocess → PA → FA pipeline on one block's columns only.

    Produces result entries keyed ``"block_{block_name}_{rotation}"`` which
    are picked up by the shared plotting / HTML / labeling loops in the
    caller.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    for do_residualize in cfg.residualize_options:
        resid_label = "residualized" if do_residualize else "raw"
        variance_export_path = (
            base_dir / resid_label / "item_variances_ranked.jsonl"
        )
        data, meta_filtered, cols_filtered, group_ids = preprocess_response_matrix(
            response_matrix, metadata, column_defs,
            min_item_variance=cfg.min_item_variance,
            high_variance_persona_drop_pct=cfg.high_variance_persona_drop_pct,
            do_residualize=do_residualize,
            variance_export_path=variance_export_path,
        )

        if do_residualize and group_ids is None:
            results[f"block_{block_name}_{resid_label}"] = {
                "n_factors": 0,
                "note": "residualization not applicable",
            }
            continue

        if data.shape[1] < 3 or data.shape[0] < 10:
            print(
                f"  Skipping FA for block={block_name!r}: "
                f"post-filter shape {data.shape} is too small."
            )
            results[f"block_{block_name}_{resid_label}"] = {
                "n_factors": 0,
                "note": "insufficient data post-filter",
            }
            continue

        print("\n  Adequacy tests:")
        adeq = adequacy_tests(data)

        print("\n  Parallel analysis (permutation null):")
        pa_result = parallel_analysis(data, random_state=seed, method="permutation")
        n_factors = pa_result["n_recommended"]
        if cfg.n_factors_override is not None:
            # Cap the override at the number of columns in this block to avoid
            # requesting more factors than variables.
            capped = min(cfg.n_factors_override, max(1, data.shape[1] - 1))
            print(
                f"  Override: using {capped} factors "
                f"(Horn recommended {n_factors}, override {cfg.n_factors_override}, "
                f"capped at n_vars-1={data.shape[1] - 1})"
            )
            n_factors = capped

        pa_dir = base_dir / resid_label
        pa_dir.mkdir(parents=True, exist_ok=True)
        with open(pa_dir / "parallel_analysis.json", "w") as f:
            json.dump({
                "n_recommended": n_factors,
                "real_eigenvalues": pa_result["real_eigenvalues"].tolist(),
                "random_threshold": pa_result["random_threshold"].tolist(),
                "adequacy": {
                    "bartlett_chi2": adeq["bartlett_chi2"],
                    "bartlett_p": adeq["bartlett_p"],
                    "kmo_overall": adeq["kmo_overall"],
                },
            }, f, indent=2)

        _plot_parallel_analysis(
            pa_result["real_eigenvalues"],
            pa_result["random_threshold"],
            n_factors,
            f"block={block_name}/{resid_label}",
            pa_dir / "parallel_analysis.png",
        )

        if n_factors == 0:
            results[f"block_{block_name}_{resid_label}"] = {
                "n_factors": 0,
                "parallel_analysis": pa_result,
            }
            continue

        for rotation in cfg.rotations:
            print(
                f"\n  Factor analysis (block={block_name!r}): "
                f"{n_factors} factors, rotation={rotation}"
            )
            fa_result = run_factor_analysis(
                data, n_factors=n_factors, method=cfg.method, rotation=rotation,
            )
            fa_path = pa_dir / f"fa_{n_factors}_{cfg.method}_{rotation}"
            save_factor_analysis(
                fa_result, fa_path,
                config={
                    "n_factors": n_factors,
                    "method": cfg.method,
                    "rotation": rotation,
                    "residualized": do_residualize,
                    "n_samples": data.shape[0],
                    "n_cols": data.shape[1],
                    "block": block_name,
                },
            )
            with open(str(fa_path) + "_item_labels.json", "w") as f:
                json.dump([
                    {
                        "col_id": col["col_id"],
                        "text": col["text"],
                        "block": col["block"],
                        "dimension": col.get("dimension"),
                        "reverse_keyed": col.get("reverse_keyed", False),
                    }
                    for col in cols_filtered
                ], f, indent=2, ensure_ascii=False)

            key = f"block_{block_name}_{resid_label}_{rotation}"
            results[key] = {
                "fa_result": fa_result,
                "column_defs": cols_filtered,
                "metadata": meta_filtered,
                "data": data,
                "n_factors": n_factors,
                "parallel_analysis": pa_result,
                "save_dir": pa_dir / f"fa_{n_factors}_{cfg.method}_{rotation}_artifacts",
                "encoding": "letter",
                "block": block_name,
            }

    return results


def _run_trait_oriented_fa_pass(
    cfg: FactorAnalysisStageConfig,
    *,
    metadata: list[dict] | None,
    items: list[dict] | None,
    seed: int,
    questionnaire_path: Path,
) -> dict:
    """Build the trait-oriented response matrix and run FA + alignment.

    Outputs land in ``<questionnaire_dir>/factor_analysis_trait_oriented/``.

    Returns ``{key: result}`` keyed by ``trait_oriented_{rotation}``. Empty
    dict if raw_responses.jsonl or trait_mcq items are missing.
    """
    q_dir = cfg.ctx.effective_questionnaire_dir / "questionnaire"
    raw_path = q_dir / "raw_responses.jsonl"
    if not raw_path.exists():
        print("\n[Trait-oriented FA] raw_responses.jsonl not found — skipping.")
        return {}

    with open(questionnaire_path, "r", encoding="utf-8") as f:
        qn_raw = json.load(f)
    trait_items_raw = qn_raw.get("block_4_trait_mcq", {}).get("items", [])
    if not trait_items_raw:
        print("\n[Trait-oriented FA] No trait_mcq items in questionnaire — skipping.")
        return {}

    trait_item_info: dict[str, dict] = {
        str(it["id"]): {
            "text": it.get("question", ""),
            "dimension": it["primary_dimension"],
        }
        for it in trait_items_raw
    }

    output_dir = cfg.ctx.effective_questionnaire_dir / "factor_analysis_trait_oriented"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("[Stage 3b] Trait-oriented factor analysis")
    print("=" * 60)
    print(f"  Building trait-oriented matrix from {raw_path}")

    tom = build_trait_oriented_matrix(
        raw_responses_path=raw_path,
        questionnaire_path=questionnaire_path,
    )
    matrix = tom.matrix
    K, N = matrix.shape
    print(f"  Raw matrix: {K} personas × {N} items  traits={tom.trait_order}")

    # Drop personas with > 20% missing
    max_missing_frac = 0.2
    missing_frac = np.mean(np.isnan(matrix), axis=1)
    keep_rows = missing_frac <= max_missing_frac
    data = matrix[keep_rows]
    kept_k = [tom.k_index[i] for i in np.where(keep_rows)[0]]
    print(
        f"  Kept {int(keep_rows.sum())}/{K} personas "
        f"(≤{max_missing_frac:.0%} missing)"
    )

    # Column-mean impute remaining NaNs
    nan_mask = np.isnan(data)
    if nan_mask.any():
        col_means = np.nanmean(data, axis=0)
        data = data.copy()
        inds = np.where(nan_mask)
        data[inds] = np.take(col_means, inds[1])
        print(f"  Mean-imputed {int(nan_mask.sum())} remaining missing cells")

    # Drop zero-variance columns
    col_var = np.var(data, axis=0)
    live_cols = col_var > 1e-10
    if not live_cols.all():
        dropped = [
            (tom.item_ids[i], tom.item_dims[i])
            for i in np.where(~live_cols)[0]
        ]
        print(
            f"  Dropping {int((~live_cols).sum())} zero-variance items: {dropped}"
        )
    data = data[:, live_cols]
    item_ids = [tom.item_ids[i] for i in np.where(live_cols)[0]]
    item_dims = [tom.item_dims[i] for i in np.where(live_cols)[0]]

    # Build column_defs mirroring the letter-encoded trait_mcq shape
    column_defs_to = [
        {
            "col_id": iid,
            "item_id": iid,
            "block": "trait_mcq",
            "dimension": item_dims[i],
            "text": trait_item_info.get(iid, {}).get("text", iid),
            "encoding": "trait_score_0-1",
        }
        for i, iid in enumerate(item_ids)
    ]

    # Filter caller-provided metadata to the kept personas
    metadata_to: list[dict]
    if metadata is not None:
        try:
            metadata_to = [dict(metadata[k]) for k in kept_k]
        except IndexError:
            print(
                "  [Warn] metadata list shorter than tom.k_index; "
                "falling back to synthetic metadata."
            )
            metadata_to = [{"k": int(k), "sample_id": f"k{k}"} for k in kept_k]
    else:
        metadata_to = [{"k": int(k), "sample_id": f"k{k}"} for k in kept_k]

    # Standardize for FA
    print("\n  Adequacy tests (standardized data):")
    data_z = (data - data.mean(axis=0)) / data.std(axis=0, ddof=0)
    adequacy = adequacy_tests(data_z)

    # Parallel analysis (or use override)
    print("\n  Parallel analysis:")
    pa = parallel_analysis(data_z, random_state=seed, method="permutation")
    n_factors = int(pa["n_recommended"])
    if cfg.n_factors_override is not None:
        print(
            f"  Override: using {cfg.n_factors_override} factors "
            f"(parallel analysis recommended {n_factors})"
        )
        n_factors = int(cfg.n_factors_override)

    with open(output_dir / "parallel_analysis.json", "w") as f:
        json.dump({
            "n_recommended": int(pa["n_recommended"]),
            "n_used": n_factors,
            "real_eigenvalues": pa["real_eigenvalues"].tolist(),
            "random_threshold": pa["random_threshold"].tolist(),
            "adequacy": {
                "kmo_overall": adequacy["kmo_overall"],
                "bartlett_p": adequacy["bartlett_p"],
            },
        }, f, indent=2)

    _plot_parallel_analysis(
        pa["real_eigenvalues"],
        pa["random_threshold"],
        n_factors,
        "trait-oriented",
        output_dir / "parallel_analysis.png",
    )

    if n_factors == 0:
        print("  No factors — skipping trait-oriented FA.")
        return {}

    with open(output_dir / "matrix_items.json", "w", encoding="utf-8") as f:
        json.dump({
            "item_ids": item_ids,
            "item_dims": item_dims,
            "trait_order": tom.trait_order,
            "kept_k": [int(k) for k in kept_k],
            "n_personas_used": int(data.shape[0]),
            "n_items_used": int(data.shape[1]),
        }, f, indent=2)

    trait_results: dict = {}

    for rotation in cfg.rotations:
        key = f"trait_oriented_{rotation}"
        save_dir = output_dir / key
        save_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n  Trait-oriented FA: {n_factors} factors, "
            f"method={cfg.method}, rotation={rotation}"
        )
        fa = run_factor_analysis(
            data_z, n_factors=n_factors, method=cfg.method, rotation=rotation,
        )
        fa_path = save_dir / f"fa_trait_oriented_{n_factors}_{cfg.method}_{rotation}"
        save_factor_analysis(
            fa, fa_path,
            config={
                "n_factors": n_factors,
                "method": cfg.method,
                "rotation": rotation,
                "encoding": "trait_oriented",
                "n_personas_used": int(data.shape[0]),
                "n_items_used": int(data.shape[1]),
                "adequacy": {
                    "kmo_overall": adequacy["kmo_overall"],
                    "bartlett_p": adequacy["bartlett_p"],
                },
            },
        )

        with open(str(fa_path) + "_item_labels.json", "w") as f:
            json.dump([
                {
                    "col_id": cd["col_id"],
                    "text": cd["text"],
                    "block": cd["block"],
                    "dimension": cd.get("dimension"),
                    "reverse_keyed": False,
                }
                for cd in column_defs_to
            ], f, indent=2, ensure_ascii=False)

        alignment = compute_factor_trait_alignment(
            loadings=fa["loadings"],
            item_dims=item_dims,
            trait_order=tom.trait_order,
            top_k=min(20, len(item_dims)),
        )
        align_dir = save_dir / "alignment"
        save_alignment(alignment, align_dir)
        plot_all_alignment(
            alignment, align_dir,
            title_prefix=f"trait-oriented / {rotation}",
        )

        print(
            f"  [Alignment] trait-oriented {rotation}: "
            f"factor→trait winners (top-{alignment.top_k}):"
        )
        for fi, fl_label in enumerate(alignment.factor_labels):
            counts = alignment.top_k_count[fi]
            best = int(np.argmax(counts))
            full = dict(zip(
                alignment.trait_order,
                [int(c) for c in counts],
            ))
            signed_dom_idx = int(np.argmax(np.abs(alignment.mean_signed_loading[fi])))
            signed_val = alignment.mean_signed_loading[fi, signed_dom_idx]
            print(
                f"    {fl_label}: top-K winner={alignment.trait_order[best]} "
                f"({int(counts[best])}/{alignment.top_k}); "
                f"strongest signed mean: {alignment.trait_order[signed_dom_idx]}={signed_val:+.3f} "
                f"— full counts {full}"
            )

        trait_results[key] = {
            "fa_result": fa,
            "column_defs": column_defs_to,
            "metadata": metadata_to,
            "data": data,
            "n_factors": n_factors,
            "parallel_analysis": pa,
            "save_dir": save_dir,
            "encoding": "trait_oriented",
        }

    return trait_results
