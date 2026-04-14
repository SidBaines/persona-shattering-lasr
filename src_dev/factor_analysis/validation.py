"""Validation tests for factor-analysis solutions.

Three tests, each a standalone pure function:
    - shuffle_control_test   : column-permutation null. No factors should emerge.
    - item_holdout_predictivity_test : hold out items, fit FA on rest, check that
      factor scores predict held-out items better than chance.
    - stability_icc_test     : ICC(1) per factor across rollout replicates.

Each function computes a result dict, persists JSON + a PNG plot to
``out_dir``, and returns the result. Compute and plot are internally split
(``_compute_*`` / ``_plot_*``) so they can be unit-tested or reused independently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src_dev.factor_analysis.factor_analysis import run_factor_analysis
from src_dev.factor_analysis.parallel_analysis import parallel_analysis
from src_dev.factor_analysis.reliability import compute_icc


# ═════════════════════════════════════════════════════════════════════════════
# SHUFFLE CONTROL
# ═════════════════════════════════════════════════════════════════════════════


def _compute_shuffle_control(
    data_clean: np.ndarray,
    pa_real: dict,
    *,
    seed: int,
    alpha: float = 0.05,
    tolerance_multiplier: float = 2.0,
    min_tolerance: int = 2,
) -> dict:
    """Shuffle each column independently; run permutation parallel analysis.

    The null test compares shuffled-data eigenvalues against the 95th-pctile
    threshold from an independent permutation null. Both are summaries of the
    same underlying noise distribution, so ~``alpha * n_vars`` false positives
    are expected by chance. The pass criterion therefore allows up to
    ``max(min_tolerance, tolerance_multiplier * alpha * n_vars)`` "found"
    factors before failing, rather than requiring exactly zero (which Monte
    Carlo noise makes statistically impossible).
    """
    rng = np.random.default_rng(seed)
    shuffled = data_clean.copy()
    for j in range(shuffled.shape[1]):
        rng.shuffle(shuffled[:, j])

    pa_shuffled = parallel_analysis(shuffled, random_state=seed, method="permutation")
    n_found = int(pa_shuffled["n_recommended"])
    n_vars = int(shuffled.shape[1])
    expected_fp = alpha * n_vars
    tolerance = max(min_tolerance, int(np.ceil(tolerance_multiplier * expected_fp)))

    return {
        "n_factors_recommended": n_found,
        "n_vars": n_vars,
        "alpha": alpha,
        "expected_false_positives": float(expected_fp),
        "tolerance": tolerance,
        "pass": n_found <= tolerance,
        "real_eigenvalues_top15": pa_real["real_eigenvalues"][:15].tolist(),
        "shuffled_eigenvalues_top15": pa_shuffled["real_eigenvalues"][:15].tolist(),
        "shuffled_threshold_top15": pa_shuffled["random_threshold"][:15].tolist(),
        "_shuffled_eigenvalues_full": pa_shuffled["real_eigenvalues"],
        "_shuffled_threshold_full": pa_shuffled["random_threshold"],
    }


def _plot_shuffle_control(
    result: dict,
    pa_real: dict,
    save_path: Path,
    *,
    plt: Any,
) -> None:
    shuffled_eig = result["_shuffled_eigenvalues_full"]
    shuffled_thr = result["_shuffled_threshold_full"]
    n_show = min(20, len(pa_real["real_eigenvalues"]))
    x = np.arange(1, n_show + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, pa_real["real_eigenvalues"][:n_show], "o-", color="#2563eb",
            linewidth=2, markersize=5, label="Real data", zorder=3)
    ax.plot(x, shuffled_eig[:n_show], "s--", color="#dc2626",
            linewidth=1.5, markersize=4, label="Shuffled data", zorder=2)
    ax.plot(x, shuffled_thr[:n_show], "^:", color="#9ca3af",
            linewidth=1, markersize=3, label="Permutation 95th pctile", zorder=1)
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Shuffle Control — Scree Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def shuffle_control_test(
    data_clean: np.ndarray,
    pa_real: dict,
    out_dir: Path,
    *,
    seed: int = 42,
    plt: Any | None = None,
) -> dict:
    """Permute each column and check that parallel analysis finds no factors.

    Writes ``out_dir/shuffle_test.json`` and (if matplotlib is available)
    ``out_dir/shuffle_scree.png``.

    Args:
        data_clean: [n_personas, n_items] preprocessed response matrix.
        pa_real: Output of ``parallel_analysis`` on the real ``data_clean``.
            Used only for the scree-comparison plot.
        out_dir: Directory to write outputs to. Created if missing.
        seed: RNG seed for both the column shuffle and the permutation null.
        plt: Optional pre-configured matplotlib.pyplot module. If None, plotting
            is done via a fresh ``matplotlib.use("Agg")`` import.

    Returns:
        Dict with ``n_factors_recommended``, ``pass``, and eigenvalue traces.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    result = _compute_shuffle_control(data_clean, pa_real, seed=seed)

    plt_mod = _resolve_plt(plt)
    if plt_mod is not None:
        _plot_shuffle_control(result, pa_real, out_dir / "shuffle_scree.png", plt=plt_mod)

    serialisable = {k: v for k, v in result.items() if not k.startswith("_")}
    with open(out_dir / "shuffle_test.json", "w") as f:
        json.dump(serialisable, f, indent=2)

    print(
        f"  Shuffle control: {result['n_factors_recommended']} factors found "
        f"(expected ~{result['expected_false_positives']:.1f} at α={result['alpha']}, "
        f"tolerance ≤{result['tolerance']}, "
        f"{'PASS' if result['pass'] else 'FAIL'})"
    )
    return serialisable


# ═════════════════════════════════════════════════════════════════════════════
# ITEM-HOLDOUT PREDICTIVITY
# ═════════════════════════════════════════════════════════════════════════════


def _compute_item_holdout_predictivity(
    data_clean: np.ndarray,
    *,
    holdout_n_items: int,
    fa_method: str,
    rotation: str,
    n_folds: int,
    n_permutations: int,
    fdr_alpha: float,
    seed: int,
) -> dict:
    """Hold out items, fit FA on remainder, CV-predict held-out items."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    rng = np.random.default_rng(seed)
    n_items = data_clean.shape[1]

    if n_items <= holdout_n_items + 10:
        return {"pass": False, "note": f"Not enough items ({n_items}) for holdout"}

    holdout_idx = rng.choice(n_items, holdout_n_items, replace=False)
    train_idx = np.setdiff1d(np.arange(n_items), holdout_idx)

    train_data = data_clean[:, train_idx]
    holdout_data = data_clean[:, holdout_idx]

    pa_train = parallel_analysis(train_data, random_state=seed, method="permutation")
    n_factors = int(pa_train["n_recommended"])
    if n_factors == 0:
        return {"pass": False, "n_factors_train": 0, "note": "No factors on training items"}

    fa_train = run_factor_analysis(
        train_data, n_factors=n_factors, method=fa_method, rotation=rotation,
    )
    scores = fa_train["scores"]  # [n_personas, n_factors]

    cv_r2_per_item: list[float] = []
    p_values: list[float] = []
    for j in range(holdout_data.shape[1]):
        y = holdout_data[:, j]
        cv_r2 = float(np.mean(cross_val_score(
            LinearRegression(), scores, y, cv=n_folds, scoring="r2",
        )))
        cv_r2_per_item.append(cv_r2)

        null_r2s = []
        for _ in range(n_permutations):
            perm_scores = rng.permutation(scores, axis=0)
            null_r2 = float(np.mean(cross_val_score(
                LinearRegression(), perm_scores, y, cv=n_folds, scoring="r2",
            )))
            null_r2s.append(null_r2)
        p_values.append(float(np.mean(np.array(null_r2s) >= cv_r2)))

    cv_r2_arr = np.array(cv_r2_per_item)
    p_arr = np.array(p_values)

    # Benjamini-Hochberg FDR
    n_tests = len(p_arr)
    sorted_idx = np.argsort(p_arr)
    fdr_threshold = np.array([(i + 1) / n_tests * fdr_alpha for i in range(n_tests)])
    reject_sorted = p_arr[sorted_idx] <= fdr_threshold
    if reject_sorted.any():
        max_reject = int(np.where(reject_sorted)[0].max())
        significant = np.zeros(n_tests, dtype=bool)
        significant[sorted_idx[:max_reject + 1]] = True
    else:
        significant = np.zeros(n_tests, dtype=bool)

    n_sig = int(significant.sum())
    pct_sig = float(n_sig / n_tests)

    return {
        "n_holdout_items": holdout_n_items,
        "n_train_items": int(len(train_idx)),
        "n_factors_train": n_factors,
        "n_permutations": n_permutations,
        "n_folds": n_folds,
        "fdr_alpha": fdr_alpha,
        "mean_cv_r2": float(cv_r2_arr.mean()),
        "median_cv_r2": float(np.median(cv_r2_arr)),
        "per_item_cv_r2": [float(r) for r in cv_r2_per_item],
        "per_item_p_value": [float(p) for p in p_values],
        "per_item_significant_fdr": [bool(s) for s in significant],
        "n_significant_fdr": n_sig,
        "pct_significant_fdr": pct_sig,
        "pass": pct_sig > 0.5,
    }


def _plot_item_predictivity(
    result: dict,
    save_path: Path,
    *,
    plt: Any,
) -> None:
    from matplotlib.patches import Patch  # local import

    cv_r2 = result["per_item_cv_r2"]
    significant = result["per_item_significant_fdr"]
    n_holdout = result["n_holdout_items"]
    n_factors = result["n_factors_train"]
    n_sig = result["n_significant_fdr"]
    n_tests = len(cv_r2)
    fdr_alpha = result["fdr_alpha"]

    fig, ax = plt.subplots(figsize=(max(8, n_holdout * 0.4), 5))
    x = np.arange(n_holdout)
    colors = ["#2563eb" if sig else "#d1d5db" for sig in significant]
    ax.bar(x, cv_r2, color=colors, edgecolor="white", width=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Held-out item index")
    ax.set_ylabel(f"{result['n_folds']}-fold CV R²")
    ax.set_title(
        f"Held-out Item Predictivity — {n_factors} factors, "
        f"{n_sig}/{n_tests} significant (FDR<{fdr_alpha})",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=7, rotation=90)
    ax.grid(axis="y", alpha=0.3)
    legend_elements = [
        Patch(facecolor="#2563eb", label=f"Significant (FDR<{fdr_alpha})"),
        Patch(facecolor="#d1d5db", label="Not significant"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def item_holdout_predictivity_test(
    data_clean: np.ndarray,
    out_dir: Path,
    *,
    holdout_n_items: int = 20,
    fa_method: str = "principal",
    rotation: str = "oblimin",
    n_folds: int = 10,
    n_permutations: int = 100,
    fdr_alpha: float = 0.05,
    seed: int = 42,
    plt: Any | None = None,
) -> dict:
    """Hold out items, fit FA on the rest, check that factor scores predict them.

    For each of ``holdout_n_items`` items: regress the item on the factor scores
    from the training-item FA, via ``n_folds`` cross-validation. Permutation null
    (``n_permutations``) estimates a per-item p-value; BH-FDR at ``fdr_alpha``
    determines which items are significantly predicted.

    Writes ``out_dir/predictivity.json`` and ``out_dir/predictivity_r2.png``.

    Args:
        data_clean: [n_personas, n_items] preprocessed response matrix.
        out_dir: Directory for outputs.
        holdout_n_items: Number of items to randomly hold out.
        fa_method: FA method passed to ``run_factor_analysis`` ("principal" etc.).
        rotation: Rotation passed to ``run_factor_analysis``.
        n_folds: Number of CV folds for the per-item regression.
        n_permutations: Number of persona-permutation nulls per item.
        fdr_alpha: BH-FDR threshold.
        seed: RNG seed.
        plt: Optional pre-configured matplotlib.pyplot module.

    Returns:
        Dict with per-item R², p-values, FDR significance flags, and ``pass``
        (True if > 50% of items are FDR-significant).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    result = _compute_item_holdout_predictivity(
        data_clean,
        holdout_n_items=holdout_n_items,
        fa_method=fa_method,
        rotation=rotation,
        n_folds=n_folds,
        n_permutations=n_permutations,
        fdr_alpha=fdr_alpha,
        seed=seed,
    )

    with open(out_dir / "predictivity.json", "w") as f:
        json.dump(result, f, indent=2)

    if result.get("note"):
        print(f"  Item-holdout predictivity: {result['note']} — skipped")
        return result

    plt_mod = _resolve_plt(plt)
    if plt_mod is not None:
        _plot_item_predictivity(result, out_dir / "predictivity_r2.png", plt=plt_mod)

    print(
        f"  Item-holdout predictivity: {result['n_factors_train']} factors, "
        f"mean CV R²={result['mean_cv_r2']:.4f}, "
        f"{result['n_significant_fdr']}/{result['n_holdout_items']} items "
        f"significant (FDR<{fdr_alpha}) "
        f"({'PASS' if result['pass'] else 'FAIL'})"
    )
    return result


# ═════════════════════════════════════════════════════════════════════════════
# STABILITY (ICC)
# ═════════════════════════════════════════════════════════════════════════════


def _pick_fa_result_with_factors(fa_results: dict) -> tuple[str | None, dict | None]:
    for key, result in fa_results.items():
        if result.get("n_factors", 0) > 0 and "fa_result" in result:
            return key, result
    return None, None


def _is_residualized_fa_key(fa_key: str) -> bool:
    """Heuristic: FA variant keys follow 'raw_*' / 'residualized_*' convention."""
    return "residualized" in fa_key.lower()


def _compute_stability_icc(
    fa_results: dict,
    *,
    group_field: str,
    pass_threshold_mean_icc1: float,
    fa_key: str | None,
) -> dict:
    if fa_key is not None:
        fa_entry = fa_results.get(fa_key)
        if fa_entry is None or fa_entry.get("n_factors", 0) == 0 or "fa_result" not in fa_entry:
            return {"pass": False, "note": f"FA key {fa_key!r} missing or has no factors", "fa_key": fa_key}
    else:
        fa_key, fa_entry = _pick_fa_result_with_factors(fa_results)
        if fa_key is None:
            return {"pass": False, "note": "No FA results with factors"}

    fa_result = fa_entry["fa_result"]
    meta = fa_entry["metadata"]
    scores = fa_result["scores"]
    n_factors = scores.shape[1]

    icc_result = compute_icc(scores, meta, n_factors, group_field=group_field)
    if icc_result.get("error"):
        return {"pass": False, "note": icc_result["error"], "fa_key": fa_key}

    # Residualization subtracts the within-group mean. When the group has
    # only 2 rollouts, the two residuals are ±x by construction, forcing
    # ICC(1) = −1 regardless of signal. The test is mathematically degenerate
    # here — skip with a note rather than reporting −1.
    mean_group_size = float(icc_result["mean_group_size"])
    if _is_residualized_fa_key(fa_key) and mean_group_size <= 2.0 + 1e-9:
        return {
            "fa_key": fa_key,
            "n_factors": n_factors,
            "group_field": group_field,
            "n_groups": icc_result["n_groups"],
            "n_total": icc_result["n_total"],
            "mean_group_size": mean_group_size,
            "icc1_per_factor": icc_result["icc1"],
            "icc1_ci_lower": icc_result["icc1_ci_lower"],
            "icc1_ci_upper": icc_result["icc1_ci_upper"],
            "icc_k_per_factor": icc_result["icc_k"],
            "pass_threshold_mean_icc1": pass_threshold_mean_icc1,
            "pass": None,
            "note": (
                f"Residualized variant with mean_group_size={mean_group_size:.2f} "
                "— ICC(1) is mathematically forced to −1 (residuals sum to zero "
                "within each 2-rollout group). Skipped."
            ),
            "skipped_reason": "residualized_small_groups",
        }

    mean_icc1 = float(np.mean(icc_result["icc1"]))
    return {
        "fa_key": fa_key,
        "n_factors": n_factors,
        "group_field": group_field,
        "n_groups": icc_result["n_groups"],
        "n_total": icc_result["n_total"],
        "mean_group_size": mean_group_size,
        "icc1_per_factor": icc_result["icc1"],
        "icc1_ci_lower": icc_result["icc1_ci_lower"],
        "icc1_ci_upper": icc_result["icc1_ci_upper"],
        "icc_k_per_factor": icc_result["icc_k"],
        "mean_icc1": mean_icc1,
        "mean_icc_k": float(np.mean(icc_result["icc_k"])),
        "pass_threshold_mean_icc1": pass_threshold_mean_icc1,
        "pass": mean_icc1 >= pass_threshold_mean_icc1,
    }


def _plot_stability_icc(
    result: dict,
    save_path: Path,
    *,
    plt: Any,
) -> None:
    n_factors = result["n_factors"]
    icc1 = np.array(result["icc1_per_factor"])
    ci_lo = np.array(result["icc1_ci_lower"])
    ci_hi = np.array(result["icc1_ci_upper"])
    yerr = np.array([icc1 - ci_lo, ci_hi - icc1])

    colors = []
    for v in icc1:
        if v >= 0.75:
            colors.append("#16a34a")
        elif v >= 0.60:
            colors.append("#2563eb")
        elif v >= 0.40:
            colors.append("#f59e0b")
        else:
            colors.append("#dc2626")

    fig, ax = plt.subplots(figsize=(max(6, n_factors * 1.2 + 1), 5))
    x = np.arange(n_factors)
    ax.bar(x, icc1, color=colors, edgecolor="white", width=0.6, zorder=3)
    ax.errorbar(x, icc1, yerr=yerr, fmt="none", ecolor="black",
                elinewidth=1, capsize=3, zorder=4)
    ax.axhline(0.75, color="#16a34a", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(0.40, color="#f59e0b", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(n_factors - 0.3, 0.76, "excellent", fontsize=7, color="#16a34a", ha="right")
    ax.text(n_factors - 0.3, 0.41, "fair", fontsize=7, color="#f59e0b", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{i}" for i in range(n_factors)], fontsize=11)
    ax.set_ylim(0, min(1.0, float(ci_hi.max()) * 1.2 + 0.05))
    ax.set_ylabel("ICC(1)")
    ax.set_xlabel("Factor")
    ax.set_title(
        f"Test-Retest Stability — ICC(1) per Factor\n"
        f"({result['n_groups']} prompts, mean k={result['mean_group_size']:.1f} rollouts)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def stability_icc_test(
    fa_results: dict,
    out_dir: Path,
    *,
    group_field: str = "input_group_id",
    pass_threshold_mean_icc1: float = 0.20,
    fa_key: str | None = None,
    plt: Any | None = None,
) -> dict:
    """ICC(1) per factor across rollout replicates grouped by ``group_field``.

    Writes ``out_dir/stability.json`` and ``out_dir/stability_icc.png``.

    Args:
        fa_results: Dict of FA result entries.
        out_dir: Directory for outputs.
        group_field: Metadata key defining the group (default: seed prompt).
        pass_threshold_mean_icc1: Minimum mean ICC(1) to declare pass.
        fa_key: Explicit key into ``fa_results`` to validate. If None, the
            first entry with ``n_factors > 0`` is used. Call once per variant
            when the pipeline produces multiple (e.g. residualization on/off).
        plt: Optional pre-configured matplotlib.pyplot module.

    Returns:
        Dict with per-factor ICC(1), CIs, mean, and ``pass`` flag.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    result = _compute_stability_icc(
        fa_results,
        group_field=group_field,
        pass_threshold_mean_icc1=pass_threshold_mean_icc1,
        fa_key=fa_key,
    )

    with open(out_dir / "stability.json", "w") as f:
        json.dump(result, f, indent=2)

    if result.get("note"):
        print(f"  Stability (ICC): {result['note']} — skipped")
        return result

    plt_mod = _resolve_plt(plt)
    if plt_mod is not None:
        _plot_stability_icc(result, out_dir / "stability_icc.png", plt=plt_mod)

    print(
        f"  Stability (ICC): {result['n_groups']} prompts, "
        f"{result['n_factors']} factors, "
        f"mean ICC(1)={result['mean_icc1']:.3f} "
        f"({'PASS' if result['pass'] else 'FAIL'})"
    )
    for f_idx in range(result["n_factors"]):
        ci_lo = result["icc1_ci_lower"][f_idx]
        ci_hi = result["icc1_ci_upper"][f_idx]
        icc_k = result["icc_k_per_factor"][f_idx]
        print(
            f"    Factor {f_idx}: ICC(1)={result['icc1_per_factor'][f_idx]:.3f} "
            f"[{ci_lo:.3f}, {ci_hi:.3f}], ICC(k)={icc_k:.3f}"
        )
    return result


# ═════════════════════════════════════════════════════════════════════════════
# INTERNAL
# ═════════════════════════════════════════════════════════════════════════════


def _resolve_plt(plt: Any | None) -> Any | None:
    """Return a matplotlib.pyplot module, initialising a headless backend if needed."""
    if plt is not None:
        return plt
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt_mod
        return plt_mod
    except ImportError:
        return None
