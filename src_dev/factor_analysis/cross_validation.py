"""Split/refit validations for factor-analysis solutions.

Three data-splitting tests:

    persona_item_cv(...)
        A/B persona split. Fit Λ on A. For each persona in B, regress Thomson
        factor scores from m randomly-observed items, then reconstruct the
        held-out items. Compare per-item R² against three baselines
        (item mean, persona-shuffle, k-1) to isolate the contribution of
        persona-specific factor structure.

    stability_sweep(...)
        Fit FA on many subsamples of the data (random-N%, leave-one-archetype
        out, or leave-one-scenario out for the top-N most common scenarios).
        For each refit, compare to the full-sample anchor via
        ``compare_solutions``. Also re-runs parallel analysis per split to
        measure k-stability.

    k_sensitivity(...)
        Fit k-1, k, k+1 factor solutions. Pairwise compare. For each factor in
        the k solution, classify as "preserved" / "split" / "merged" based on
        structural matches in the neighbouring solutions.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src_dev.factor_analysis.congruence import (
    SolutionComparison,
    compare_solutions,
    tucker_phi,
)
from src_dev.factor_analysis.factor_analysis import run_factor_analysis
from src_dev.factor_analysis.parallel_analysis import parallel_analysis


# ═════════════════════════════════════════════════════════════════════════════
# PERSONA × ITEM CROSS-VALIDATION
# ═════════════════════════════════════════════════════════════════════════════


def _thomson_scores_from_observed(
    x_obs: np.ndarray,
    loadings_obs: np.ndarray,
    uniquenesses_obs: np.ndarray,
) -> np.ndarray:
    """Thomson (regression) factor scores from a partial observation.

    F̂ = Λ_obs.T · Σ_obs⁻¹ · X_obs, where Σ_obs = Λ_obs Λ_obs.T + diag(Ψ_obs).

    Args:
        x_obs: [n_personas, m] observed item values (item-mean-centered).
        loadings_obs: [m, k] loadings for the m observed items.
        uniquenesses_obs: [m] Ψ diagonal for the m observed items.

    Returns:
        [n_personas, k] Thomson factor scores.
    """
    psi = np.maximum(uniquenesses_obs, 1e-6)  # guard against 0
    # Woodbury: (ΛΛᵀ + Ψ)⁻¹ = Ψ⁻¹ − Ψ⁻¹Λ (I + Λᵀ Ψ⁻¹ Λ)⁻¹ Λᵀ Ψ⁻¹.
    Lt_Psi_inv = loadings_obs.T / psi[None, :]               # [k, m]
    inner = np.eye(loadings_obs.shape[1]) + Lt_Psi_inv @ loadings_obs  # [k, k]
    inner_inv = np.linalg.inv(inner)
    sigma_inv_x = (x_obs / psi[None, :]).T \
        - Lt_Psi_inv.T @ (inner_inv @ (Lt_Psi_inv @ x_obs.T))  # [m, n]
    return (loadings_obs.T @ sigma_inv_x).T                  # [n, k]


def _r2_per_column(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Column-wise R² (1 - SS_res / SS_tot). Returns NaN when SS_tot == 0."""
    resid = y_true - y_pred
    ss_res = np.sum(resid ** 2, axis=0)
    mean = y_true.mean(axis=0, keepdims=True)
    ss_tot = np.sum((y_true - mean) ** 2, axis=0)
    r2 = np.where(ss_tot > 0, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)
    return r2


def _stratify_split(
    metadata: list[dict],
    *,
    split_frac: float,
    stratify: str | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified random A/B split on persona indices."""
    n = len(metadata)
    if stratify is None:
        order = rng.permutation(n)
        n_a = int(round(split_frac * n))
        return np.sort(order[:n_a]), np.sort(order[n_a:])

    # Support "a x b" combo stratification by concatenating fields.
    fields = [f.strip() for f in stratify.split("x")]
    groups: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(metadata):
        key = "||".join(str(m.get(f, "")) for f in fields)
        groups[key].append(i)

    a_list, b_list = [], []
    for _, idxs in groups.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_a = int(round(split_frac * len(idxs)))
        if n_a == 0 and len(idxs) > 0:
            n_a = 1
        if n_a == len(idxs) and len(idxs) > 1:
            n_a = len(idxs) - 1
        a_list.extend(idxs[:n_a].tolist())
        b_list.extend(idxs[n_a:].tolist())
    return np.sort(np.array(a_list, dtype=int)), np.sort(np.array(b_list, dtype=int))


def persona_item_cv(
    data: np.ndarray,
    metadata: list[dict],
    n_factors: int,
    out_dir: Path | None = None,
    *,
    persona_split: float = 0.7,
    stratify: str | None = "archetype x scenario_id",
    m_observed: int | None = None,
    subset_strategy: str = "random",
    n_trials: int = 5,
    fa_method: str = "principal",
    rotation: str = "oblimin",
    seed: int = 42,
) -> dict:
    """Persona split + item holdout cross-validation.

    Fit Λ on persona subset A. For each persona in B, draw ``m_observed`` items
    at random as "observed", compute Thomson factor scores, then reconstruct
    the remaining items. Repeat ``n_trials`` times with fresh observed-item
    draws. Compare per-item R² against three baselines:

        item_mean       — predict every persona with the A-subset column mean.
        persona_shuffle — permute F̂ rows before reconstruction (breaks the
                          persona↔factor mapping while keeping factor geometry).
        k_minus_1       — refit with k−1 factors on A and reconstruct.

    Args:
        data: [n_personas, n_items] preprocessed response matrix.
        metadata: Per-persona metadata dicts (same length as data).
        n_factors: Number of factors for the main solution.
        out_dir: If given, write JSON + plot here.
        persona_split: Fraction of personas in subset A.
        stratify: Metadata field (or "a x b" combo) to stratify by. None for
            pure random split.
        m_observed: Items observed per B-persona when computing Thomson scores.
            Defaults to max(20, 3 * n_factors).
        subset_strategy: "random" or "by_factor_balanced". ``by_factor_balanced``
            groups items by their dominant factor (argmax over |loadings|) and
            draws uniformly within each group so every factor is observed.
        n_trials: Number of observed-item resamples per persona.
        fa_method: FA extraction method.
        rotation: FA rotation.
        seed: RNG seed.

    Returns:
        Dict with per-item R² (main + baselines), summary stats, and ``pass``
        (main mean R² > 2× persona-shuffle mean R² AND main > k-1 on top
        factor contribution).
    """
    if subset_strategy not in ("random", "by_factor_balanced"):
        raise ValueError(f"subset_strategy must be 'random' or 'by_factor_balanced'")

    rng = np.random.default_rng(seed)
    n_personas, n_items = data.shape
    if m_observed is None:
        m_observed = max(20, 3 * n_factors)
    if m_observed >= n_items - 5:
        m_observed = n_items - 5

    a_idx, b_idx = _stratify_split(
        metadata, split_frac=persona_split, stratify=stratify, rng=rng,
    )
    data_a = data[a_idx]
    data_b = data[b_idx]
    n_b = len(b_idx)

    # Item means on A for baseline + centering.
    item_mean_a = data_a.mean(axis=0)
    data_a_c = data_a - item_mean_a
    data_b_c = data_b - item_mean_a

    # ── Fit main FA on A ─────────────────────────────────────────────────
    fa_main = run_factor_analysis(
        data_a_c, n_factors=n_factors, method=fa_method, rotation=rotation,
    )
    loadings = fa_main["loadings"]                           # [n_items, k]
    communalities = fa_main["communalities"]
    uniquenesses = 1.0 - communalities                        # [n_items]

    # ── Fit k-1 baseline FA on A ─────────────────────────────────────────
    if n_factors > 1:
        fa_km1 = run_factor_analysis(
            data_a_c, n_factors=n_factors - 1, method=fa_method, rotation=rotation,
        )
        loadings_km1 = fa_km1["loadings"]
        uniq_km1 = 1.0 - fa_km1["communalities"]
    else:
        loadings_km1 = None
        uniq_km1 = None

    # Held-out item per persona: any item not in the observed subset is held
    # out, so we evaluate across trials on every item that was ever held out.
    per_item_r2_main: list[list[float]] = [[] for _ in range(n_items)]
    per_item_r2_shuf: list[list[float]] = [[] for _ in range(n_items)]
    per_item_r2_km1: list[list[float]] = [[] for _ in range(n_items)]

    # For "by_factor_balanced" we pre-select candidate items per factor.
    if subset_strategy == "by_factor_balanced":
        abs_L = np.abs(loadings)
        dominant_factor = abs_L.argmax(axis=1)
        per_factor_items = [
            np.where(dominant_factor == k)[0] for k in range(n_factors)
        ]
    else:
        per_factor_items = None

    for _trial in range(n_trials):
        if subset_strategy == "random":
            observed = rng.choice(n_items, m_observed, replace=False)
        else:
            per_factor_budget = max(1, m_observed // n_factors)
            picks: list[int] = []
            for items_k in per_factor_items:  # type: ignore[arg-type]
                if len(items_k) == 0:
                    continue
                pick_n = min(per_factor_budget, len(items_k))
                picks.extend(
                    rng.choice(items_k, pick_n, replace=False).tolist()
                )
            # Pad with random items up to m_observed.
            remaining = np.setdiff1d(np.arange(n_items), picks)
            if len(picks) < m_observed and len(remaining) > 0:
                need = min(m_observed - len(picks), len(remaining))
                picks.extend(rng.choice(remaining, need, replace=False).tolist())
            observed = np.array(sorted(set(picks[:m_observed])))
        heldout = np.setdiff1d(np.arange(n_items), observed)

        # Main: Thomson scores from observed items on B, reconstruct held-out.
        f_hat = _thomson_scores_from_observed(
            data_b_c[:, observed], loadings[observed], uniquenesses[observed],
        )  # [n_b, k]
        pred_main = f_hat @ loadings[heldout].T + item_mean_a[heldout]

        # Persona-shuffle baseline: permute F̂ rows.
        shuf = rng.permutation(n_b)
        pred_shuf = f_hat[shuf] @ loadings[heldout].T + item_mean_a[heldout]

        # k−1 baseline.
        if loadings_km1 is not None:
            f_hat_km1 = _thomson_scores_from_observed(
                data_b_c[:, observed], loadings_km1[observed], uniq_km1[observed],
            )
            pred_km1 = f_hat_km1 @ loadings_km1[heldout].T + item_mean_a[heldout]
        else:
            pred_km1 = np.full_like(pred_main, item_mean_a[heldout][None, :])

        y_true = data_b[:, heldout]
        r2_main = _r2_per_column(y_true, pred_main)
        r2_shuf = _r2_per_column(y_true, pred_shuf)
        r2_km1 = _r2_per_column(y_true, pred_km1)

        for idx, j in enumerate(heldout):
            if not np.isnan(r2_main[idx]):
                per_item_r2_main[j].append(float(r2_main[idx]))
            if not np.isnan(r2_shuf[idx]):
                per_item_r2_shuf[j].append(float(r2_shuf[idx]))
            if not np.isnan(r2_km1[idx]):
                per_item_r2_km1[j].append(float(r2_km1[idx]))

    # Item-mean baseline: constant predictor == item_mean_a broadcast to B.
    pred_mean_all = np.broadcast_to(item_mean_a, data_b.shape)
    r2_item_mean = _r2_per_column(data_b, pred_mean_all)

    def _mean(lst: list[float]) -> float:
        return float(np.mean(lst)) if lst else float("nan")

    r2_main_per_item = np.array([_mean(lst) for lst in per_item_r2_main])
    r2_shuf_per_item = np.array([_mean(lst) for lst in per_item_r2_shuf])
    r2_km1_per_item = np.array([_mean(lst) for lst in per_item_r2_km1])

    main_mean = float(np.nanmean(r2_main_per_item))
    shuf_mean = float(np.nanmean(r2_shuf_per_item))
    km1_mean = float(np.nanmean(r2_km1_per_item))
    item_mean_mean = float(np.nanmean(r2_item_mean))

    gain_over_shuffle = main_mean - shuf_mean
    shuf_gain_over_mean = shuf_mean - item_mean_mean
    top_factor_r2_gain = main_mean - km1_mean

    # Pass criterion: persona factor scores beat the persona-shuffle baseline
    # by at least 2× the floor that shuffle provides over item-mean. When
    # shuf_gain_over_mean ≤ 0 the ratio is degenerate (shuffle is worthless),
    # so fall back to requiring main to strictly beat both baselines instead
    # of letting a vanishingly-small gain auto-pass.
    if shuf_gain_over_mean > 0:
        pass_shuffle = (main_mean > 0) and (
            gain_over_shuffle >= 2.0 * shuf_gain_over_mean
        )
    else:
        pass_shuffle = (main_mean > item_mean_mean) and (gain_over_shuffle > 0)
    pass_km1 = top_factor_r2_gain > 0

    result = {
        "n_factors": n_factors,
        "n_personas_a": int(len(a_idx)),
        "n_personas_b": int(n_b),
        "m_observed": m_observed,
        "n_trials": n_trials,
        "stratify": stratify,
        "subset_strategy": subset_strategy,
        "fa_method": fa_method,
        "rotation": rotation,
        "mean_r2_main": main_mean,
        "mean_r2_persona_shuffle": shuf_mean,
        "mean_r2_k_minus_1": km1_mean,
        "mean_r2_item_mean": item_mean_mean,
        "gain_over_shuffle": gain_over_shuffle,
        "top_factor_r2_gain": top_factor_r2_gain,
        "pass_vs_shuffle": pass_shuffle,
        "pass_vs_k_minus_1": pass_km1,
        "pass": pass_shuffle and pass_km1,
        "per_item_r2_main": r2_main_per_item.tolist(),
        "per_item_r2_persona_shuffle": r2_shuf_per_item.tolist(),
        "per_item_r2_k_minus_1": r2_km1_per_item.tolist(),
        "per_item_r2_item_mean": r2_item_mean.tolist(),
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "persona_item_cv.json", "w") as f:
            json.dump(result, f, indent=2)
        _plot_persona_item_cv(result, out_dir / "persona_item_cv.png")

    print(
        f"  Persona×item CV: main R²={main_mean:.3f}, "
        f"shuffle={shuf_mean:.3f}, k-1={km1_mean:.3f}, item-mean={item_mean_mean:.3f} "
        f"({'PASS' if result['pass'] else 'FAIL'})"
    )
    return result


def _plot_persona_item_cv(result: dict, save_path: Path) -> None:
    plt = _resolve_plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = ["main", "persona\nshuffle", "k−1", "item\nmean"]
    values = [
        result["mean_r2_main"],
        result["mean_r2_persona_shuffle"],
        result["mean_r2_k_minus_1"],
        result["mean_r2_item_mean"],
    ]
    colors = ["#2563eb", "#dc2626", "#f59e0b", "#9ca3af"]
    ax.bar(labels, values, color=colors, edgecolor="white", zorder=3)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Mean per-item R²")
    ax.set_title(
        f"Persona × item CV — {result['n_factors']} factors, "
        f"{result['n_personas_a']}/{result['n_personas_b']} A/B split, "
        f"m={result['m_observed']}",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# STABILITY SWEEP
# ═════════════════════════════════════════════════════════════════════════════


def _iter_splits(
    metadata: list[dict],
    *,
    mode: str,
    n_splits: int,
    fraction: float | None,
    top_n_scenarios: int,
    rng: np.random.Generator,
) -> Iterable[tuple[str, np.ndarray]]:
    """Yield (split_label, persona_indices_for_this_split)."""
    n = len(metadata)

    if mode in ("random_50", "random_70", "random_fraction"):
        if fraction is None:
            fraction = 0.5 if mode == "random_50" else 0.7
        for s in range(n_splits):
            order = rng.permutation(n)
            k = int(round(fraction * n))
            yield f"random_{fraction:.2f}_seed{s}", np.sort(order[:k])
        return

    if mode == "loao":
        archetypes = sorted({str(m.get("archetype", "")) for m in metadata})
        for arch in archetypes:
            idx = np.array([
                i for i, m in enumerate(metadata)
                if str(m.get("archetype", "")) != arch
            ])
            if len(idx) >= 10:
                yield f"loao_drop_{arch}", idx
        return

    if mode == "loso":
        counts = Counter(str(m.get("scenario_id", "")) for m in metadata)
        top = [sc for sc, _ in counts.most_common(top_n_scenarios)]
        for scen in top:
            idx = np.array([
                i for i, m in enumerate(metadata)
                if str(m.get("scenario_id", "")) != scen
            ])
            if len(idx) >= 10:
                yield f"loso_drop_{scen}", idx
        return

    raise ValueError(f"Unknown stability sweep mode: {mode!r}")


def stability_sweep(
    data: np.ndarray,
    metadata: list[dict],
    n_factors: int,
    anchor_loadings: np.ndarray,
    out_dir: Path | None = None,
    *,
    mode: str = "random_50",
    n_splits: int = 10,
    fraction: float | None = None,
    top_n_scenarios: int = 10,
    fa_method: str = "principal",
    rotation: str = "varimax",
    align: str = "procrustes",
    pa_iterations: int = 50,
    seed: int = 42,
    pass_threshold_median_phi: float = 0.80,
) -> dict:
    """Refit FA on many subsamples; compare each to the anchor via Tucker φ.

    Also re-runs parallel analysis on each subsample to measure k-stability.

    Args:
        data: [n_personas, n_items] preprocessed response matrix.
        metadata: Per-persona metadata (used to drive LOAO / LOSO).
        n_factors: Number of factors to fit on each split.
        anchor_loadings: [n_items, n_factors] reference loading matrix (the
            full-sample solution).
        out_dir: If given, write JSON + plot here.
        mode: "random_50" | "random_70" | "random_fraction" | "loao" | "loso".
        n_splits: Number of splits for random modes (ignored for LOAO/LOSO).
        fraction: Override the random-mode subsample fraction.
        top_n_scenarios: For LOSO, number of most common scenarios to sweep.
        fa_method, rotation: FA settings — must match the anchor's settings.
        align: "procrustes" or "hungarian" — passed to compare_solutions.
        pa_iterations: Parallel-analysis iterations per split (reduced from
            the default 100 because we run it n_splits × times).
        seed: RNG seed.
        pass_threshold_median_phi: Median per-factor |φ| threshold for ``pass``.

    Returns:
        Dict with per-split φ arrays, k-recommendation distribution, summary.
    """
    rng = np.random.default_rng(seed)

    per_split_phi: list[list[float]] = []
    per_split_k: list[int] = []
    per_split_labels: list[str] = []

    for split_index, (label, idx) in enumerate(_iter_splits(
        metadata,
        mode=mode,
        n_splits=n_splits,
        fraction=fraction,
        top_n_scenarios=top_n_scenarios,
        rng=rng,
    )):
        sub = data[idx]
        # Fresh permutation null per split — else every split's PA decision
        # uses the same permutation noise and k-stability is vacuous.
        pa_seed = seed + split_index + 1
        try:
            pa = parallel_analysis(sub, n_iterations=pa_iterations,
                                    random_state=pa_seed, method="permutation")
            k_rec = int(pa["n_recommended"])
        except Exception as exc:
            print(f"    [{label}] parallel_analysis failed: {exc}")
            k_rec = -1

        try:
            fa = run_factor_analysis(
                sub, n_factors=n_factors, method=fa_method, rotation=rotation,
            )
        except Exception as exc:
            print(f"    [{label}] FA failed: {exc} — skipping split")
            continue

        cmp = compare_solutions(anchor_loadings, fa["loadings"], align=align)
        per_split_phi.append(np.abs(cmp.phi_matched).tolist())
        per_split_k.append(k_rec)
        per_split_labels.append(label)

    if not per_split_phi:
        return {"pass": False, "note": "No splits produced a usable FA."}

    # Align phi arrays to shape [n_splits, n_factors] (ragged for k±1 cases).
    max_k = max(len(p) for p in per_split_phi)
    phi_matrix = np.full((len(per_split_phi), max_k), np.nan)
    for i, p in enumerate(per_split_phi):
        phi_matrix[i, :len(p)] = p

    median_per_factor = np.nanmedian(phi_matrix, axis=0).tolist()
    mean_per_factor = np.nanmean(phi_matrix, axis=0).tolist()
    overall_median = float(np.nanmedian(phi_matrix))
    k_counter = Counter(per_split_k)

    result = {
        "mode": mode,
        "n_splits": len(per_split_phi),
        "n_factors_anchor": n_factors,
        "fa_method": fa_method,
        "rotation": rotation,
        "align": align,
        "per_split_labels": per_split_labels,
        "per_split_phi": per_split_phi,
        "per_split_k_recommended": per_split_k,
        "median_phi_per_factor": median_per_factor,
        "mean_phi_per_factor": mean_per_factor,
        "overall_median_phi": overall_median,
        "k_recommendation_counts": dict(k_counter),
        "pass_threshold_median_phi": pass_threshold_median_phi,
        "pass": all(
            v >= pass_threshold_median_phi
            for v in median_per_factor[:n_factors]
        ),
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"stability_sweep_{mode}.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        _plot_stability_sweep(result, out_dir / f"stability_sweep_{mode}.png")

    print(
        f"  Stability sweep [{mode}]: {result['n_splits']} splits, "
        f"overall median |φ|={overall_median:.3f} "
        f"({'PASS' if result['pass'] else 'FAIL'})"
    )
    return result


def _plot_stability_sweep(result: dict, save_path: Path) -> None:
    plt = _resolve_plt()
    if plt is None:
        return
    phi_mat = np.array(result["per_split_phi"])
    if phi_mat.ndim == 1:
        phi_mat = phi_mat.reshape(-1, 1)
    n_factors = phi_mat.shape[1]

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * n_factors + 2), 4.5))
    positions = np.arange(n_factors)
    ax.boxplot(
        [phi_mat[:, k][~np.isnan(phi_mat[:, k])] for k in range(n_factors)],
        positions=positions, widths=0.6, patch_artist=True,
        boxprops=dict(facecolor="#dbeafe", edgecolor="#1e40af"),
        medianprops=dict(color="#1e40af", linewidth=2),
    )
    ax.axhline(0.90, color="#16a34a", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(result["pass_threshold_median_phi"], color="#f59e0b",
                linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"F{k}" for k in range(n_factors)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("|Tucker φ| vs anchor")
    ax.set_xlabel("Factor")
    ax.set_title(
        f"Stability sweep — {result['mode']} ({result['n_splits']} splits), "
        f"overall median |φ|={result['overall_median_phi']:.3f}",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# K ± 1 SENSITIVITY
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class FactorKStatus:
    factor_index: int
    status: str           # "preserved" | "split" | "merged"
    best_match_km1: tuple[int, float] | None   # (index, phi) in k-1 solution
    top_matches_kp1: list[tuple[int, float]]    # [(index, phi), ...] in k+1


def k_sensitivity(
    data: np.ndarray,
    k_center: int,
    out_dir: Path | None = None,
    *,
    fa_method: str = "principal",
    rotation: str = "varimax",
    match_threshold: float = 0.85,
    independent_threshold: float = 0.60,
) -> dict:
    """Fit k-1, k, k+1 factor solutions; classify each k-factor.

    For each factor *i* in the k solution:
        * best_match_km1 = best |φ| match in the k-1 solution.
        * top_matches_kp1 = all k+1 factors with |φ| ≥ ``match_threshold``.

    Status:
        * "split"     — two k+1 factors have |φ| ≥ match_threshold with i AND
                        |φ| ≤ independent_threshold with each other.
        * "merged"    — best_match_km1 has |φ| < match_threshold.
        * "preserved" — otherwise.

    Args:
        data: [n_personas, n_items] preprocessed response matrix.
        k_center: Number of factors for the "main" solution.
        out_dir: If given, write JSON + alluvial plot.
        fa_method, rotation: FA settings.
        match_threshold: |φ| above which two factors are "matched".
        independent_threshold: |φ| below which two factors are "independent".

    Returns:
        Dict with per-factor status, raw φ matrices, summary.
    """
    fits = {}
    for k in (k_center - 1, k_center, k_center + 1):
        if k < 1:
            continue
        fa = run_factor_analysis(
            data, n_factors=k, method=fa_method, rotation=rotation,
        )
        fits[k] = fa["loadings"]

    k_minus_1 = fits.get(k_center - 1)
    k_main = fits[k_center]
    k_plus_1 = fits.get(k_center + 1)

    phi_k_km1 = tucker_phi(k_main, k_minus_1) if k_minus_1 is not None else None
    phi_k_kp1 = tucker_phi(k_main, k_plus_1) if k_plus_1 is not None else None
    phi_kp1_kp1 = tucker_phi(k_plus_1, k_plus_1) if k_plus_1 is not None else None

    statuses: list[FactorKStatus] = []
    for i in range(k_center):
        # k-1 best match
        best_km1 = None
        if phi_k_km1 is not None:
            row = np.abs(phi_k_km1[i])
            j = int(np.argmax(row))
            best_km1 = (j, float(row[j]))

        # k+1 top matches
        top_kp1: list[tuple[int, float]] = []
        if phi_k_kp1 is not None:
            row = np.abs(phi_k_kp1[i])
            for j in np.argsort(row)[::-1]:
                if row[j] >= match_threshold:
                    top_kp1.append((int(j), float(row[j])))

        # Split condition: ≥ 2 kp1 factors matched AND mutually independent.
        is_split = False
        if len(top_kp1) >= 2 and phi_kp1_kp1 is not None:
            j1, j2 = top_kp1[0][0], top_kp1[1][0]
            cross = float(abs(phi_kp1_kp1[j1, j2]))
            is_split = cross <= independent_threshold

        if is_split:
            status = "split"
        elif best_km1 is not None and best_km1[1] < match_threshold:
            status = "merged"
        else:
            status = "preserved"

        statuses.append(FactorKStatus(
            factor_index=i, status=status,
            best_match_km1=best_km1,
            top_matches_kp1=top_kp1,
        ))

    status_counts = Counter(s.status for s in statuses)
    pass_overall = status_counts["split"] <= 1 and status_counts["merged"] == 0

    result = {
        "k_center": k_center,
        "k_minus_1_available": k_minus_1 is not None,
        "k_plus_1_available": k_plus_1 is not None,
        "match_threshold": match_threshold,
        "independent_threshold": independent_threshold,
        "phi_k_vs_km1": phi_k_km1.tolist() if phi_k_km1 is not None else None,
        "phi_k_vs_kp1": phi_k_kp1.tolist() if phi_k_kp1 is not None else None,
        "per_factor": [
            {
                "factor_index": s.factor_index,
                "status": s.status,
                "best_match_km1": s.best_match_km1,
                "top_matches_kp1": s.top_matches_kp1,
            }
            for s in statuses
        ],
        "status_counts": dict(status_counts),
        "pass": pass_overall,
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "k_sensitivity.json", "w") as f:
            json.dump(result, f, indent=2)
        _plot_k_sensitivity(result, out_dir / "k_sensitivity.png")

    print(
        f"  k sensitivity (k={k_center}): "
        + ", ".join(f"{k}×{v}" for k, v in sorted(status_counts.items()))
        + f" ({'PASS' if pass_overall else 'FAIL'})"
    )
    return result


def _plot_k_sensitivity(result: dict, save_path: Path) -> None:
    plt = _resolve_plt()
    if plt is None:
        return
    phi_km1 = result.get("phi_k_vs_km1")
    phi_kp1 = result.get("phi_k_vs_kp1")
    if phi_km1 is None or phi_kp1 is None:
        return

    phi_km1 = np.array(phi_km1)
    phi_kp1 = np.array(phi_kp1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, mat, title in zip(
        axes,
        (phi_km1, phi_kp1),
        (f"|φ|: k vs k−1", f"|φ|: k vs k+1"),
    ):
        im = ax.imshow(np.abs(mat), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xlabel("Neighbour factor")
        ax.set_ylabel("k-factor")
        ax.set_title(title, fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(
        f"k sensitivity (k={result['k_center']}): "
        + ", ".join(f"{k}×{v}" for k, v in sorted(result['status_counts'].items())),
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# INTERNAL
# ═════════════════════════════════════════════════════════════════════════════


def _resolve_plt() -> Any | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None
