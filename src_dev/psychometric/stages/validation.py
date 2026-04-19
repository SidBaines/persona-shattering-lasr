"""Stage 5 — validation tests over the FA results.

Each test is independently gated by ``cfg.tests_to_run``. Tests call into
``src_dev.factor_analysis.{validation,cross_validation,bootstrap,n_factors,
interpretation,trait_convergence}`` and write their own JSON + plot(s)
under ``<questionnaire_dir>/validation/``.

Available tests (default set is the full list):
    shuffle_control, item_holdout, stability_icc, variance_decomp,
    trait_convergence, stability_sweep_random50, stability_sweep_loao,
    stability_sweep_loso_top10, k_sensitivity, persona_item_cv,
    n_factors_suggest, bootstrap_loadings, split_half_congruence,
    cv_k_curve, external_predictivity.

Variance-decomp and trait-convergence tests need the rollout canonical
dataset to resolve ``archetype`` / ``scenario_id`` — pass ``rollout_dirs``
to the stage so it can build that lookup.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src_dev.factor_analysis.parallel_analysis import parallel_analysis
from src_dev.psychometric.config import (
    ValidationStageConfig,
    ValidationStageResult,
)
from src_dev.psychometric.metadata_enrichment import (
    enrich_meta_with_archetype_scenario,
    load_archetype_scenario_lookup,
)
from src_dev.psychometric.preprocessing import preprocess_response_matrix

logger = logging.getLogger(__name__)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _pick_anchor_fa(fa_results: dict) -> tuple[str, dict] | tuple[None, None]:
    for key, result in fa_results.items():
        if result.get("n_factors", 0) > 0 and "fa_result" in result:
            return key, result
    return None, None


# ─── Individual test wrappers ────────────────────────────────────────────────


def _validation_shuffle_test(
    cfg: ValidationStageConfig, data_clean, pa_real, val_dir, plt, *, seed: int,
) -> dict:
    print("\n[Stage 5] Validation — Shuffle control")
    from src_dev.factor_analysis.validation import shuffle_control_test
    return shuffle_control_test(data_clean, pa_real, val_dir, seed=seed, plt=plt)


def _validation_predictivity_test(
    cfg: ValidationStageConfig,
    data_clean,
    val_dir,
    plt,
    *,
    seed: int,
    fa_method: str,
    rotation: str,
) -> dict:
    print("\n[Stage 5] Validation — Held-out item predictivity")
    from src_dev.factor_analysis.validation import item_holdout_predictivity_test
    return item_holdout_predictivity_test(
        data_clean, val_dir,
        holdout_n_items=cfg.holdout_n_items,
        fa_method=fa_method,
        rotation=rotation,
        r2_floor=cfg.holdout_r2_floor,
        seed=seed + 1,
        plt=plt,
    )


def _validation_stability_test(
    cfg: ValidationStageConfig,
    fa_results,
    val_dir,
    plt,
    *,
    num_rollouts_per_prompt: int,
) -> dict:
    print("\n[Stage 5] Validation — Stability (ICC)")

    if num_rollouts_per_prompt < 2:
        msg = (
            "Skipped: num_rollouts_per_prompt="
            f"{num_rollouts_per_prompt} — need >= 2 rollouts per seed prompt "
            "for within-prompt test–retest ICC. Use stability_sweep_random50 "
            "as a replicate-free proxy (cross-split congruence) instead."
        )
        print(f"  {msg}")
        skip_dir = val_dir / "stability"
        skip_dir.mkdir(parents=True, exist_ok=True)
        with open(skip_dir / "skipped.json", "w") as f:
            json.dump({
                "skipped": True,
                "reason": msg,
                "num_rollouts_per_prompt": num_rollouts_per_prompt,
            }, f, indent=2)
        return {
            "skipped": True, "reason": msg, "pass": None,
            "mean_icc1_by_variant": {}, "per_variant": {},
        }

    from src_dev.factor_analysis.validation import stability_icc_test

    variant_keys = [
        k for k, v in fa_results.items()
        if v.get("n_factors", 0) > 0 and "fa_result" in v
    ]
    if not variant_keys:
        return stability_icc_test(fa_results, val_dir, plt=plt)

    per_variant: dict[str, dict] = {}
    for key in variant_keys:
        variant_dir = val_dir / "stability" / key
        per_variant[key] = stability_icc_test(
            fa_results, variant_dir, fa_key=key, plt=plt,
        )

    mean_icc1_by_variant = {k: v.get("mean_icc1") for k, v in per_variant.items()}
    all_pass = all(v.get("pass") for v in per_variant.values())
    return {
        "per_variant": per_variant,
        "mean_icc1_by_variant": mean_icc1_by_variant,
        "pass": all_pass,
    }


def _validation_variance_decomp(
    cfg: ValidationStageConfig,
    fa_results,
    val_dir,
    plt,
    *,
    rollout_dirs: list[Path] | tuple[Path, ...],
) -> dict:
    print("\n[Stage 5] Validation — Variance decomposition (η²)")
    from src_dev.factor_analysis.interpretation import prompt_effects

    fa_key = None
    for key, result in fa_results.items():
        if result.get("n_factors", 0) > 0 and "fa_result" in result:
            fa_key = key
            break
    if fa_key is None:
        return {"pass": None, "note": "No FA results with factors"}

    fa_result = fa_results[fa_key]["fa_result"]
    meta = fa_results[fa_key]["metadata"]
    scores = fa_result["scores"]
    n_factors = scores.shape[1]

    lookup = load_archetype_scenario_lookup(rollout_dirs)
    meta = enrich_meta_with_archetype_scenario(meta, lookup)

    per_field: dict[str, list[float]] = {}
    for field in cfg.variance_decomp_fields:
        eta2 = prompt_effects(scores, meta, group_field=field)
        per_field[field] = [float(v) for v in eta2]

    scenario_eta2 = np.array(per_field.get("scenario_id", [0.0] * n_factors))
    archetype_eta2 = np.array(per_field.get("archetype", [0.0] * n_factors))

    missing = [
        f for f in ("scenario_id", "archetype")
        if np.all(np.isnan(per_field.get(f, [0.0])))
    ]
    if missing:
        skip_note = (
            f"{', '.join(missing)} missing from FA metadata — "
            "variance_decomp pass criterion undefined."
        )
        result = {
            "fa_key": fa_key,
            "n_factors": n_factors,
            "eta2_per_field": per_field,
            "scenario_ceiling": cfg.variance_decomp_scenario_ceiling,
            "archetype_floor": cfg.variance_decomp_archetype_floor,
            "pass": None,
            "note": skip_note,
        }
        out_dir = val_dir / "variance_decomp"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "variance_decomp.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Variance decomp: {skip_note} (SKIP)")
        return result

    scenario_flagged = [
        int(i) for i, v in enumerate(scenario_eta2)
        if v >= cfg.variance_decomp_scenario_ceiling
    ]
    archetype_signal_max = float(archetype_eta2.max()) if len(archetype_eta2) else 0.0
    archetype_floor_pass = archetype_signal_max >= cfg.variance_decomp_archetype_floor

    passed = (len(scenario_flagged) == 0) and archetype_floor_pass

    result = {
        "fa_key": fa_key,
        "n_factors": n_factors,
        "eta2_per_field": per_field,
        "scenario_ceiling": cfg.variance_decomp_scenario_ceiling,
        "archetype_floor": cfg.variance_decomp_archetype_floor,
        "scenario_flagged_factors": scenario_flagged,
        "archetype_max_eta2": archetype_signal_max,
        "archetype_floor_pass": archetype_floor_pass,
        "pass": passed,
    }

    out_dir = val_dir / "variance_decomp"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "variance_decomp.json", "w") as f:
        json.dump(result, f, indent=2)

    try:
        fields = list(per_field.keys())
        x = np.arange(n_factors)
        width = 0.8 / len(fields)
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * n_factors + 2), 4.5))
        colors = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626"]
        for i, field in enumerate(fields):
            ax.bar(
                x + (i - (len(fields) - 1) / 2) * width,
                per_field[field], width, label=field, color=colors[i % len(colors)],
            )
        ax.axhline(
            cfg.variance_decomp_scenario_ceiling, color="#dc2626",
            linestyle="--", linewidth=0.8, alpha=0.6,
            label=f"scenario ceiling ({cfg.variance_decomp_scenario_ceiling})",
        )
        ax.axhline(
            cfg.variance_decomp_archetype_floor, color="#16a34a",
            linestyle=":", linewidth=0.8, alpha=0.6,
            label=f"archetype floor ({cfg.variance_decomp_archetype_floor})",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"F{i}" for i in range(n_factors)])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("η² (variance explained)")
        ax.set_xlabel("Factor")
        ax.set_title(
            "Variance decomposition — scenario as ceiling, archetype as floor",
            fontsize=11, fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "variance_decomp.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"  Variance decomposition plot failed: {exc}")

    print(
        f"  Variance decomp: scenario-flagged {len(scenario_flagged)}/{n_factors} "
        f"(η²≥{cfg.variance_decomp_scenario_ceiling}), "
        f"archetype max η²={archetype_signal_max:.2f} "
        f"(floor={cfg.variance_decomp_archetype_floor}) "
        f"({'PASS' if passed else 'FAIL'})"
    )
    return result


def _run_external_predictivity(
    cfg: ValidationStageConfig,
    fa_results,
    val_dir,
    external_predictivity_fn,
    *,
    seed: int,
    fa_method: str,
    rotation: str,
) -> dict:
    trait_csv = (
        cfg.ctx.effective_questionnaire_dir / "questionnaire" / "trait_scores"
        / "trait_scores_with_metadata.csv"
    )
    if not trait_csv.exists():
        msg = f"trait_scores_with_metadata.csv not found at {trait_csv}"
        print(f"  {msg} — skipped")
        return {"pass": None, "note": msg}

    import pandas as pd
    df = pd.read_csv(trait_csv)
    if "sample_id" not in df.columns:
        return {"pass": None, "note": "trait_scores CSV missing sample_id column"}
    trait_cols = [c for c in df.columns
                  if c not in ("k", "sample_id", "input_group_id")]
    by_sid = df.set_index("sample_id")[trait_cols]

    anchor_key, anchor_entry = _pick_anchor_fa(fa_results)
    if anchor_entry is None:
        return {"pass": None, "note": "No FA results with factors"}

    anchor_data = anchor_entry.get("data")
    anchor_meta = anchor_entry.get("metadata")
    anchor_k = anchor_entry["n_factors"]
    if anchor_data is None or anchor_meta is None:
        return {
            "pass": None,
            "note": "Anchor FA entry missing 'data' or 'metadata' — cannot refit FA on folds",
        }

    n = len(anchor_meta)
    t = len(trait_cols)
    trait_matrix = np.full((n, t), np.nan)
    for i, m in enumerate(anchor_meta):
        sid = m.get("sample_id")
        if sid in by_sid.index:
            trait_matrix[i] = by_sid.loc[sid].values

    aligned_mask = ~np.all(np.isnan(trait_matrix), axis=1)
    n_aligned = int(aligned_mask.sum())
    if n_aligned < 30:
        return {
            "pass": False,
            "note": f"Too few personas aligned to trait scores: {n_aligned}",
        }

    data_aligned = anchor_data[aligned_mask]
    targets_aligned = trait_matrix[aligned_mask]
    meta_aligned = [m for m, keep in zip(anchor_meta, aligned_mask) if keep]

    out_dir = val_dir / "external_predictivity"
    result = external_predictivity_fn(
        data_aligned, targets_aligned, n_factors=anchor_k, out_dir=out_dir,
        target_names=trait_cols,
        n_folds=cfg.external_predictivity_n_folds,
        metadata=meta_aligned,
        fa_method=fa_method,
        rotation=rotation,
        ridge_alpha=cfg.external_predictivity_ridge_alpha,
        pass_threshold_r2=cfg.external_predictivity_pass_r2,
        bootstrap_ci=cfg.external_predictivity_bootstrap_ci,
        seed=seed + 24,
        verbose=True,
    )
    result["fa_key"] = anchor_key
    result["n_aligned_personas"] = n_aligned
    return result


def _validation_trait_convergence(
    cfg: ValidationStageConfig,
    fa_results,
    val_dir,
    plt,
    *,
    seed: int,
) -> dict:
    print("\n[Stage 5] Validation — TRAIT convergent validity")
    from src_dev.factor_analysis.trait_convergence import convergent_validity

    trait_csv = (
        cfg.ctx.effective_questionnaire_dir / "questionnaire" / "trait_scores"
        / "trait_scores_with_metadata.csv"
    )
    if not trait_csv.exists():
        msg = f"trait_scores_with_metadata.csv not found at {trait_csv}"
        print(f"  {msg} — skipped")
        return {"pass": None, "note": msg}

    import pandas as pd

    df = pd.read_csv(trait_csv)
    if "sample_id" not in df.columns:
        return {"pass": None, "note": "trait_scores CSV missing sample_id column"}
    trait_cols = [c for c in df.columns
                  if c not in ("k", "sample_id", "input_group_id")]
    by_sid = df.set_index("sample_id")[trait_cols]

    fa_key = None
    for key, result in fa_results.items():
        if result.get("n_factors", 0) > 0 and "fa_result" in result:
            fa_key = key
            break
    if fa_key is None:
        return {"pass": None, "note": "No FA results with factors"}

    fa_entry = fa_results[fa_key]
    scores = fa_entry["fa_result"]["scores"]
    meta = fa_entry["metadata"]

    n = len(meta)
    t = len(trait_cols)
    trait_matrix = np.full((n, t), np.nan)
    for i, m in enumerate(meta):
        sid = m.get("sample_id")
        if sid in by_sid.index:
            trait_matrix[i] = by_sid.loc[sid].values

    n_aligned = int(np.sum(~np.all(np.isnan(trait_matrix), axis=1)))
    if n_aligned < 20:
        return {
            "pass": False,
            "note": f"Too few aligned personas: {n_aligned}",
            "fa_key": fa_key,
        }

    out_dir = val_dir / "trait_convergence"
    result = convergent_validity(
        scores, trait_matrix, out_dir,
        trait_names=trait_cols,
        factor_names=[f"F{i}" for i in range(scores.shape[1])],
        method="spearman",
        n_bootstrap=cfg.trait_convergence_n_bootstrap,
        trait_hit_threshold=cfg.trait_convergence_hit_threshold,
        min_trait_hits=cfg.trait_convergence_min_hits,
        seed=seed + 5,
        plt=plt,
    )
    result["fa_key"] = fa_key
    result["n_aligned_personas"] = n_aligned
    return result


# ─── Main dispatcher ─────────────────────────────────────────────────────────


def run_stage_validation(
    cfg: ValidationStageConfig,
    response_matrix: np.ndarray,
    metadata: list[dict],
    column_defs: list[dict],
    fa_results: dict,
    *,
    seed: int,
    num_rollouts_per_prompt: int,
    fa_method: str,
    rotations: tuple[str, ...],
    min_item_variance: float = 0.1,
    high_variance_persona_drop_pct: float = 0.0,
    rollout_dirs: list[Path] | tuple[Path, ...] = (),
) -> ValidationStageResult:
    """Run validation tests (each gated by ``cfg.tests_to_run``).

    Args:
        cfg: Validation stage config (tests_to_run + per-test knobs).
        response_matrix, metadata, column_defs: Stage-2 artefacts.
        fa_results: FA-result dict from the factor-analysis stage.
        seed: Base seed. Individual tests offset it (seed+N) so they remain
            independent.
        num_rollouts_per_prompt: Informs the stability_icc skip path.
        fa_method, rotations: Forwarded to tests that refit FA on
            subsamples. ``rotations[0]`` is treated as the canonical
            rotation (same as the script).
        min_item_variance, high_variance_persona_drop_pct: Preprocessing
            knobs for the initial clean-matrix pass that shuffle /
            item-holdout / (etc.) feed off of.
        rollout_dirs: Used to resolve archetype/scenario metadata for
            variance-decomp. Optional — if empty and the test requires it,
            the test falls through to its skip path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    val_dir = cfg.ctx.effective_questionnaire_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    data_clean, meta_clean, cols_clean, _ = preprocess_response_matrix(
        response_matrix, metadata, column_defs,
        min_item_variance=min_item_variance,
        high_variance_persona_drop_pct=high_variance_persona_drop_pct,
        do_residualize=False,
    )

    pa_real_result = None
    for result in fa_results.values():
        if result.get("parallel_analysis") is not None:
            pa_real_result = result["parallel_analysis"]
            break
    if pa_real_result is None:
        pa_real_result = parallel_analysis(data_clean, random_state=seed, method="permutation")

    anchor_key, anchor_entry = _pick_anchor_fa(fa_results)
    anchor_k = anchor_entry["n_factors"] if anchor_entry else None
    anchor_loadings = (
        anchor_entry["fa_result"]["loadings"] if anchor_entry else None
    )

    results: dict[str, dict] = {}

    def _enabled(name: str) -> bool:
        return name in cfg.tests_to_run

    if _enabled("shuffle_control"):
        results["shuffle"] = _validation_shuffle_test(
            cfg, data_clean, pa_real_result, val_dir, plt, seed=seed,
        )
    if _enabled("item_holdout"):
        results["item_holdout"] = _validation_predictivity_test(
            cfg, data_clean, val_dir, plt,
            seed=seed, fa_method=fa_method, rotation=rotations[0],
        )
    if _enabled("stability_icc"):
        results["stability_icc"] = _validation_stability_test(
            cfg, fa_results, val_dir, plt,
            num_rollouts_per_prompt=num_rollouts_per_prompt,
        )
    if _enabled("variance_decomp"):
        results["variance_decomp"] = _validation_variance_decomp(
            cfg, fa_results, val_dir, plt,
            rollout_dirs=rollout_dirs,
        )
    if _enabled("trait_convergence"):
        results["trait_convergence"] = _validation_trait_convergence(
            cfg, fa_results, val_dir, plt, seed=seed,
        )

    # Sweeps / sensitivity / persona×item CV all need the anchor loadings.
    if anchor_loadings is not None:
        from src_dev.factor_analysis.cross_validation import (
            external_predictivity as _external_predictivity,
            k_sensitivity as _k_sensitivity,
            persona_item_cv as _persona_item_cv,
            split_half_congruence as _split_half_congruence,
            stability_sweep as _stability_sweep,
        )
        from src_dev.factor_analysis.bootstrap import (
            bootstrap_loadings as _bootstrap_loadings,
            plot_bootstrap_loadings as _plot_bootstrap_loadings,
            save_bootstrap_loadings as _save_bootstrap_loadings,
        )
        from src_dev.factor_analysis.n_factors import (
            cv_reconstruction_k as _cv_reconstruction_k,
            plot_n_factors_comparison as _plot_n_factors_comparison,
            suggest_n_factors as _suggest_n_factors,
        )

        if _enabled("stability_sweep_random50"):
            print("\n[Stage 5] Validation — Stability sweep (random 50%)")
            results["stability_sweep_random50"] = _stability_sweep(
                data_clean, meta_clean, anchor_k, anchor_loadings,
                val_dir, mode="random_50",
                n_splits=cfg.stability_sweep_n_random_splits,
                pa_iterations=cfg.stability_sweep_pa_iterations,
                fa_method=fa_method, rotation=rotations[0],
                pass_threshold_median_phi=cfg.stability_sweep_pass_threshold_phi,
                seed=seed + 10,
            )
        if _enabled("stability_sweep_loao"):
            print("\n[Stage 5] Validation — Stability sweep (leave-one-archetype-out)")
            results["stability_sweep_loao"] = _stability_sweep(
                data_clean, meta_clean, anchor_k, anchor_loadings,
                val_dir, mode="loao",
                pa_iterations=cfg.stability_sweep_pa_iterations,
                fa_method=fa_method, rotation=rotations[0],
                pass_threshold_median_phi=cfg.stability_sweep_pass_threshold_phi,
                seed=seed + 11,
            )
        if _enabled("stability_sweep_loso_top10"):
            print(
                f"\n[Stage 5] Validation — Stability sweep "
                f"(leave-one-scenario-out, top {cfg.stability_sweep_loso_top_n})"
            )
            results["stability_sweep_loso_top10"] = _stability_sweep(
                data_clean, meta_clean, anchor_k, anchor_loadings,
                val_dir, mode="loso",
                top_n_scenarios=cfg.stability_sweep_loso_top_n,
                pa_iterations=cfg.stability_sweep_pa_iterations,
                fa_method=fa_method, rotation=rotations[0],
                pass_threshold_median_phi=cfg.stability_sweep_pass_threshold_phi,
                seed=seed + 12,
            )
        if _enabled("k_sensitivity"):
            print("\n[Stage 5] Validation — k ± 1 sensitivity")
            results["k_sensitivity"] = _k_sensitivity(
                data_clean, k_center=anchor_k, out_dir=val_dir,
                fa_method=fa_method, rotation=rotations[0],
                match_threshold=cfg.k_sensitivity_match_threshold,
                independent_threshold=cfg.k_sensitivity_independent_threshold,
            )
        if _enabled("persona_item_cv"):
            print("\n[Stage 5] Validation — Persona × item CV")
            results["persona_item_cv"] = _persona_item_cv(
                data_clean, meta_clean, anchor_k, val_dir,
                persona_split=cfg.persona_item_cv_split,
                n_trials=cfg.persona_item_cv_n_trials,
                n_outer_splits=cfg.persona_item_cv_n_outer_splits,
                bootstrap_ci=cfg.persona_item_cv_bootstrap_ci,
                subset_strategy=cfg.persona_item_cv_subset_strategy,
                fa_method=fa_method, rotation=rotations[0],
                seed=seed + 13,
            )

        # Tests below re-analyze the anchor FA's data.
        anchor_data = anchor_entry.get("data", data_clean)
        anchor_meta = anchor_entry.get("metadata", meta_clean)
        anchor_cols = anchor_entry.get("column_defs", cols_clean)

        if _enabled("n_factors_suggest"):
            print("\n[Stage 5] Validation — n_factors suggestion (multi-method)")
            n_factors_dir = val_dir / "n_factors_suggest"
            n_factors_dir.mkdir(parents=True, exist_ok=True)
            try:
                suggest = _suggest_n_factors(
                    anchor_data,
                    methods=cfg.n_factors_suggest_methods,
                    k_max=min(
                        cfg.n_factors_suggest_k_max,
                        max(1, anchor_data.shape[1] - 1),
                    ),
                    parallel_n_iterations=cfg.n_factors_suggest_pa_iterations,
                    cv_n_folds=cfg.n_factors_suggest_cv_n_folds,
                    fa_method=fa_method,
                    seed=seed + 20,
                    verbose=True,
                )
                with open(n_factors_dir / "n_factors_suggest.json", "w") as f:
                    json.dump(_jsonable(suggest), f, indent=2)
                _plot_n_factors_comparison(
                    suggest, save_path=n_factors_dir / "n_factors_suggest.png",
                )
                summary = suggest.get("summary", {})
                note = " / ".join(f"{m}={k}" for m, k in summary.items())
                results["n_factors_suggest"] = {
                    "pass": None, "note": note, "summary": summary,
                }
            except Exception as exc:
                print(f"  n_factors_suggest failed: {exc}")
                results["n_factors_suggest"] = {
                    "pass": None, "note": f"failed: {exc}",
                }

        if _enabled("bootstrap_loadings"):
            print("\n[Stage 5] Validation — Bootstrap loadings")
            boot_dir = val_dir / "bootstrap_loadings"
            boot_dir.mkdir(parents=True, exist_ok=True)
            try:
                boot = _bootstrap_loadings(
                    anchor_data,
                    n_factors=anchor_k,
                    anchor_loadings=anchor_loadings,
                    n_boot=cfg.bootstrap_loadings_n_boot,
                    fa_method=fa_method,
                    rotation=rotations[0],
                    confidence=cfg.bootstrap_loadings_confidence,
                    seed=seed + 21,
                    verbose=True,
                )
                _save_bootstrap_loadings(boot, boot_dir)
                _plot_bootstrap_loadings(boot, anchor_cols, boot_dir)
                ci_excl = np.array(boot.get("ci_excludes_zero", []))
                frac_reliable = (
                    float(ci_excl.mean()) if ci_excl.size else float("nan")
                )
                results["bootstrap_loadings"] = {
                    "pass": None,
                    "note": (
                        f"mean φ vs anchor={boot.get('mean_alignment_phi', float('nan')):.3f}, "
                        f"fraction reliable={frac_reliable:.2f}"
                    ),
                    "mean_alignment_phi": boot.get("mean_alignment_phi"),
                    "fraction_reliable": frac_reliable,
                    "n_boot": boot.get("n_boot"),
                }
            except Exception as exc:
                print(f"  bootstrap_loadings failed: {exc}")
                results["bootstrap_loadings"] = {
                    "pass": None, "note": f"failed: {exc}",
                }

        if _enabled("split_half_congruence"):
            print("\n[Stage 5] Validation — Split-half congruence")
            results["split_half_congruence"] = _split_half_congruence(
                anchor_data,
                n_factors=anchor_k,
                out_dir=val_dir,
                n_iters=cfg.split_half_congruence_n_iters,
                metadata=anchor_meta,
                fa_method=fa_method,
                rotation=rotations[0],
                seed=seed + 22,
                pass_threshold_median_phi=cfg.split_half_congruence_pass_threshold_phi,
                verbose=True,
            )

        if _enabled("cv_k_curve"):
            print("\n[Stage 5] Validation — CV-k curve (held-out Gaussian NLL)")
            cvk_dir = val_dir / "cv_k_curve"
            cvk_dir.mkdir(parents=True, exist_ok=True)
            try:
                cv_k = _cv_reconstruction_k(
                    anchor_data,
                    k_max=min(
                        cfg.cv_k_curve_k_max, max(1, anchor_data.shape[1] - 1),
                    ),
                    n_folds=cfg.cv_k_curve_n_folds,
                    fa_method=fa_method,
                    seed=seed + 23,
                    verbose=True,
                )
                with open(cvk_dir / "cv_k_curve.json", "w") as f:
                    json.dump(_jsonable(cv_k), f, indent=2)
                k_rec = cv_k.get("n_recommended")
                k_rec_1se = cv_k.get("n_recommended_1se")
                results["cv_k_curve"] = {
                    "pass": None,
                    "note": (
                        f"argmin k={k_rec}, 1-SE k={k_rec_1se}, "
                        f"anchor k={anchor_k}"
                    ),
                    "n_recommended": k_rec,
                    "n_recommended_1se": k_rec_1se,
                }
            except Exception as exc:
                print(f"  cv_k_curve failed: {exc}")
                results["cv_k_curve"] = {
                    "pass": None, "note": f"failed: {exc}",
                }

        if _enabled("external_predictivity"):
            print("\n[Stage 5] Validation — External predictivity (FA → OCEAN)")
            ext_result = _run_external_predictivity(
                cfg, fa_results, val_dir, _external_predictivity,
                seed=seed, fa_method=fa_method, rotation=rotations[0],
            )
            results["external_predictivity"] = ext_result

    print("\n" + "=" * 60)
    print("[Stage 5] Validation Summary:")
    for test_name, test_result in results.items():
        passed = test_result.get("pass")
        note = test_result.get("note")
        if passed is True:
            status = "PASS"
        elif passed is False:
            status = "FAIL"
        else:
            status = "SKIP"
        suffix = f" — {note}" if note else ""
        print(f"  {test_name}: {status}{suffix}")
    print("=" * 60)

    return ValidationStageResult(output_dir=val_dir, results=results)
