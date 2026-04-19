"""Item-split (as opposed to persona-split) validation on an existing combined
response matrix.

The canonical ``split_half_congruence`` in
``src_dev.factor_analysis.cross_validation`` splits the **personas (rows)** in
half, refits FA on each half, and aligns the two loading matrices via Tucker φ.
This script instead splits the **items (columns)** and asks: does a factor
solution built from one set of items recover the same latent structure as a
solution built from a different set of items?

Two item-split schemes are supported:

1. ``random_halves`` — ``N`` iterations of uniformly-random 50/50 item splits.
   Factor scores from each half are computed for the same personas (rows),
   then the two solutions are aligned by Hungarian matching on their
   persona-level score correlation matrix, and per-factor Pearson r is
   reported. This is the item analogue of the existing persona split-half
   test.

2. ``by_source`` — split by the ``questionnaire_version`` column-level tag
   produced by ``_combine_per_pair_outputs``. One call per version with that
   version as side A and every other version as side B, so e.g. v5 items vs
   (v6_fc_draft + trait_ocean_v1) items. Answers "do factors derived from
   questionnaire X also appear when we look at questionnaires Y+Z on the same
   personas?".

Because the two item halves don't share items, comparison via Tucker φ on
loadings isn't meaningful. Instead we align solutions via the **Pearson
correlation of factor scores on the same personas** — a standard item-split
factor-score replication test. Factor sign is arbitrary, so matching runs on
|r| and the signed r is recovered afterwards.

The expected input directory is a Stage-2 ``questionnaire/`` output, typically
a multi-pair combined directory built by a prior run of
``psychometric_rollout_fa.py`` (e.g. ``combined-R[A+B]-Q[v5+v6_fc_draft+
trait_ocean_v1]/``). Single-pair questionnaire directories work too, but
``by_source`` collapses to a no-op unless ``items.json`` contains more than one
``questionnaire_version``.

This script is a config + analysis template — edit the paths at the top and
the knobs below, then run it.
"""

from __future__ import annotations

# ── Seeds (before any stochastic imports) ───────────────────────────────────
import random

import numpy as np

SEED = 436
random.seed(SEED)
np.random.seed(SEED)

import json
import logging
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src_dev.factor_analysis.congruence import hungarian_match
from src_dev.factor_analysis.factor_analysis import run_factor_analysis
from src_dev.psychometric.preprocessing import preprocess_response_matrix

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG — point at a Stage-2 ``questionnaire/`` directory
# ═════════════════════════════════════════════════════════════════════════════

# A combined multi-pair directory is the canonical target: both rollouts
# (1 = "A", 2 = "B") and all three questionnaires (v5, v6_fc_draft,
# trait_ocean_v1) pooled into one response matrix.
QUESTIONNAIRE_DIR = Path(
    "scratch/psychometric_fa/combined-R[A+B]-Q[v5+v6_fc_draft+trait_ocean_v1]"
)

# Subset the combined matrix to these ``questionnaire_version`` values before
# any preprocessing / FA. None keeps all versions. Use this to re-run the
# validations on an item-subset of a larger combined directory (e.g. drop
# v6_fc_draft to isolate the Likert × MCQ contribution) without rebuilding
# the combined Stage-2 artefacts from scratch.
INCLUDE_VERSIONS: list[str] | None = ["v5", "trait_ocean_v1"]

# Subset the combined matrix to rows whose ``rollout_preset_key`` (from the
# Stage-2 combine step) is in this list. None keeps every rollout. Use this
# to isolate a single rollout preset, e.g. ["B"] for the scenarios-v2 /
# seed-436 production rollouts only. Applied before preprocessing so the
# per-block variance floor and NaN row drop see only the filtered rows.
INCLUDE_ROLLOUT_KEYS: list[str] | None = ["B"]

# Output goes under this root. Suffixes are appended so runs with different
# INCLUDE_VERSIONS / INCLUDE_ROLLOUT_KEYS / MIN_ITEM_VARIANCE settings don't
# stomp each other:
#   "_{versions}"     — when INCLUDE_VERSIONS is set
#   "_rollouts{keys}" — when INCLUDE_ROLLOUT_KEYS is set
#   "_minvarX.XX"     — when MIN_ITEM_VARIANCE differs from the 0.1 default

# ═════════════════════════════════════════════════════════════════════════════
# Knobs
# ═════════════════════════════════════════════════════════════════════════════

FA_METHOD = "principal"
FA_ROTATION = "oblimin"

# Sweep over multiple factor counts. Each k produces its own subdirectory
# ``k{K}/`` under OUTPUT_DIR. Put the values you care most about first — the
# loop runs in order and each k's artefacts land on disk as soon as it's done.
K_VALUES = [3, 4, 5, 6, 7]

MIN_ITEM_VARIANCE = 0.1

# Keep only the top-N items by per-block relative variance (item_var /
# block_median_var, the same metric ``preprocess_response_matrix`` uses for
# its threshold). ``None`` disables the top-N filter. When set, the
# ``MIN_ITEM_VARIANCE`` threshold is overridden so top-N is the sole item
# filter — this avoids accidentally mixing two variance cuts.
#
# Ranking is global (across blocks) on the relative-variance score, so a
# block with systematically higher within-block signal can end up
# over-represented. That's usually fine when you want the "best N items by
# how variable they are relative to their peers", but note it's not
# per-block balanced.
TOP_N_ITEMS: int | None = 80

# Build OUTPUT_DIR from the knobs above so filtered / high-threshold runs
# don't stomp the default run on disk.
_BASE_OUTPUT_DIR = QUESTIONNAIRE_DIR / "item_split_validation"
_out_name = _BASE_OUTPUT_DIR.name
if INCLUDE_VERSIONS is not None:
    _out_name += "_" + "+".join(sorted(INCLUDE_VERSIONS))
if INCLUDE_ROLLOUT_KEYS is not None:
    _out_name += "_rollouts" + "+".join(sorted(INCLUDE_ROLLOUT_KEYS))
if TOP_N_ITEMS is not None:
    _out_name += f"_top{TOP_N_ITEMS}"
elif MIN_ITEM_VARIANCE != 0.1:
    _out_name += f"_minvar{MIN_ITEM_VARIANCE:g}"
OUTPUT_DIR = _BASE_OUTPUT_DIR.with_name(_out_name)

# Number of random-half item-split iterations.
N_RANDOM_ITERS = 50

# If True, random halves respect the per-version item counts (each half gets
# half of each version's items). This keeps both halves balanced across item
# types (likert / fc_pair / trait_mcq) and is usually what you want.
RANDOM_HALVES_STRATIFY_BY_VERSION = True

# Which by-source split schemes to run:
#   "vs_rest"  — each version vs the union of the remaining versions
#   "pairwise" — each unordered pair of versions (v5↔v6, v5↔trait, v6↔trait)
# Set both to include both sets of splits in the same by_source.json.
BY_SOURCE_VS_REST = True
BY_SOURCE_PAIRWISE = True

# Whether to also run the by-source validation.
RUN_BY_SOURCE = True

# Whether to also run the random-halves validation.
RUN_RANDOM_HALVES = True


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _fit_fa_on_columns(
    data: np.ndarray,
    col_idx: np.ndarray,
    n_factors: int,
    *,
    method: str,
    rotation: str,
    label: str,
) -> dict | None:
    """Fit FA on ``data[:, col_idx]``.

    Returns the ``run_factor_analysis`` result dict augmented with the
    column indices used, or ``None`` if FA fails (e.g. rank-deficient
    sub-matrix).
    """
    sub = data[:, col_idx]
    # Guard against zero-variance columns that may arise in a small subset.
    var = np.var(sub, axis=0)
    keep = var > 1e-10
    if int(keep.sum()) < n_factors + 1:
        print(f"    [{label}] too few usable columns ({int(keep.sum())}); skipping")
        return None
    if not keep.all():
        print(
            f"    [{label}] dropping {int((~keep).sum())} zero-variance cols "
            f"within subset"
        )
    col_idx_used = col_idx[keep]
    sub_used = sub[:, keep]
    try:
        res = run_factor_analysis(
            sub_used,
            n_factors=n_factors,
            method=method,
            rotation=rotation,
        )
    except Exception as exc:
        print(f"    [{label}] FA failed: {exc}")
        return None
    res["col_idx"] = col_idx_used
    return res


def _match_on_scores(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> dict:
    """Hungarian-match factors by Pearson r of scores across shared personas.

    Args:
        scores_a: [n_personas, k_a] factor scores from solution A.
        scores_b: [n_personas, k_b] factor scores from solution B.

    Returns:
        Dict with:
          ``phi_matrix``      — [k_a, k_b] full Pearson-r matrix.
          ``matched_a``       — [min(k_a,k_b)] indices into A's factors.
          ``matched_b``       — same length indices into B's factors.
          ``r_matched``       — signed Pearson r for each matched pair.
          ``abs_r_matched``   — |r| for each matched pair.
    """
    # Column-wise standardize; Pearson r = normalized dot product.
    def _z(x: np.ndarray) -> np.ndarray:
        m = x.mean(axis=0, keepdims=True)
        s = x.std(axis=0, keepdims=True)
        s = np.where(s > 0, s, 1.0)
        return (x - m) / s

    za = _z(scores_a)
    zb = _z(scores_b)
    n = za.shape[0]
    phi = (za.T @ zb) / max(n - 1, 1)

    row_idx, col_idx, sign_flips = hungarian_match(phi)
    matched = phi[row_idx, col_idx]
    return {
        "phi_matrix": phi,
        "matched_a": row_idx,
        "matched_b": col_idx,
        "r_matched": matched,
        "abs_r_matched": np.abs(matched),
        "sign_flips": sign_flips,
    }


def _stratified_random_half(
    n_items: int,
    group_ids: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition item indices into two roughly-equal halves, balanced per group."""
    a_parts: list[np.ndarray] = []
    b_parts: list[np.ndarray] = []
    for g in np.unique(group_ids):
        g_idx = np.where(group_ids == g)[0]
        perm = rng.permutation(g_idx)
        half = len(perm) // 2
        a_parts.append(np.sort(perm[:half]))
        b_parts.append(np.sort(perm[half:2 * half]))
    a = np.sort(np.concatenate(a_parts)) if a_parts else np.array([], dtype=int)
    b = np.sort(np.concatenate(b_parts)) if b_parts else np.array([], dtype=int)
    return a, b


def _plain_random_half(
    n_items: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(n_items)
    half = n_items // 2
    return np.sort(order[:half]), np.sort(order[half:2 * half])


# ═════════════════════════════════════════════════════════════════════════════
# Validations
# ═════════════════════════════════════════════════════════════════════════════


def run_random_halves(
    data: np.ndarray,
    column_defs: list[dict],
    *,
    n_iters: int,
    n_factors: int,
    fa_method: str,
    rotation: str,
    stratify_by_version: bool,
    seed: int,
) -> dict:
    print(
        f"\n[Random halves] n_iters={n_iters}, n_factors={n_factors}, "
        f"stratify_by_version={stratify_by_version}"
    )

    n_items = data.shape[1]
    if stratify_by_version:
        versions = np.array(
            [str(c.get("questionnaire_version", "")) for c in column_defs]
        )
    else:
        versions = None

    per_iter: list[dict] = []
    for it in range(n_iters):
        rng = np.random.default_rng(seed + 10_000 + it)
        if versions is not None:
            a_idx, b_idx = _stratified_random_half(n_items, versions, rng)
        else:
            a_idx, b_idx = _plain_random_half(n_items, rng)

        fa_a = _fit_fa_on_columns(
            data, a_idx, n_factors,
            method=fa_method, rotation=rotation, label=f"iter {it} / A",
        )
        fa_b = _fit_fa_on_columns(
            data, b_idx, n_factors,
            method=fa_method, rotation=rotation, label=f"iter {it} / B",
        )
        if fa_a is None or fa_b is None:
            continue

        match = _match_on_scores(fa_a["scores"], fa_b["scores"])
        per_iter.append({
            "iter": it,
            "n_items_a": int(a_idx.size),
            "n_items_b": int(b_idx.size),
            "r_matched": match["r_matched"].tolist(),
            "abs_r_matched": match["abs_r_matched"].tolist(),
            "matched_a": match["matched_a"].tolist(),
            "matched_b": match["matched_b"].tolist(),
        })
        print(
            f"  iter {it}: |r| per matched factor = "
            f"[{', '.join(f'{v:.2f}' for v in match['abs_r_matched'])}]"
        )

    if not per_iter:
        return {"per_iter": [], "summary": None}

    max_k = max(len(p["abs_r_matched"]) for p in per_iter)
    abs_r_mat = np.full((len(per_iter), max_k), np.nan)
    for i, p in enumerate(per_iter):
        abs_r_mat[i, :len(p["abs_r_matched"])] = p["abs_r_matched"]

    summary = {
        "n_iters_ok": len(per_iter),
        "n_iters_requested": n_iters,
        "n_factors": n_factors,
        "median_abs_r_per_factor": np.nanmedian(abs_r_mat, axis=0).tolist(),
        "mean_abs_r_per_factor": np.nanmean(abs_r_mat, axis=0).tolist(),
        "p10_abs_r_per_factor": np.nanpercentile(abs_r_mat, 10, axis=0).tolist(),
        "p90_abs_r_per_factor": np.nanpercentile(abs_r_mat, 90, axis=0).tolist(),
        "overall_median_abs_r": float(np.nanmedian(abs_r_mat)),
    }
    print(
        f"\n  Summary: overall median |r|={summary['overall_median_abs_r']:.3f}\n"
        f"  per-factor medians = "
        f"[{', '.join(f'{v:.2f}' for v in summary['median_abs_r_per_factor'])}]"
    )
    return {"per_iter": per_iter, "summary": summary}


def _run_one_source_split(
    data: np.ndarray,
    versions: np.ndarray,
    versions_a: list[str],
    versions_b: list[str],
    *,
    scheme: str,
    n_factors: int,
    fa_method: str,
    rotation: str,
) -> dict:
    """Fit FA on items with version in ``versions_a`` and ``versions_b``
    separately, then align factors by persona-score correlation."""
    a_mask = np.isin(versions, versions_a)
    b_mask = np.isin(versions, versions_b)
    a_idx = np.where(a_mask)[0]
    b_idx = np.where(b_mask)[0]
    label = f"{'+'.join(versions_a)} ↔ {'+'.join(versions_b)}"
    print(
        f"\n  Split [{scheme}]: {label}   "
        f"(n_items_A={a_idx.size}, n_items_B={b_idx.size})"
    )

    fa_a = _fit_fa_on_columns(
        data, a_idx, n_factors,
        method=fa_method, rotation=rotation,
        label=f"A={'+'.join(versions_a)}",
    )
    fa_b = _fit_fa_on_columns(
        data, b_idx, n_factors,
        method=fa_method, rotation=rotation,
        label=f"B={'+'.join(versions_b)}",
    )
    if fa_a is None or fa_b is None:
        return {
            "scheme": scheme,
            "versions_a": versions_a,
            "versions_b": versions_b,
            "n_items_a": int(a_idx.size),
            "n_items_b": int(b_idx.size),
            "error": "FA failed on one side",
        }

    match = _match_on_scores(fa_a["scores"], fa_b["scores"])
    entry = {
        "scheme": scheme,
        "versions_a": versions_a,
        "versions_b": versions_b,
        # Back-compat alias for single-version A splits.
        "version_a": versions_a[0] if len(versions_a) == 1 else None,
        "n_items_a": int(a_idx.size),
        "n_items_b": int(b_idx.size),
        "r_matched": match["r_matched"].tolist(),
        "abs_r_matched": match["abs_r_matched"].tolist(),
        "matched_a": match["matched_a"].tolist(),
        "matched_b": match["matched_b"].tolist(),
        "full_phi_matrix": match["phi_matrix"].tolist(),
        "median_abs_r": float(np.median(match["abs_r_matched"])),
    }
    print(
        f"    per-factor |r| = "
        f"[{', '.join(f'{v:.2f}' for v in match['abs_r_matched'])}]  "
        f"median={entry['median_abs_r']:.3f}"
    )
    return entry


def run_by_source(
    data: np.ndarray,
    column_defs: list[dict],
    *,
    n_factors: int,
    fa_method: str,
    rotation: str,
    seed: int,
    do_vs_rest: bool = True,
    do_pairwise: bool = True,
) -> dict:
    print(
        f"\n[By source] n_factors={n_factors}  "
        f"vs_rest={do_vs_rest}, pairwise={do_pairwise}"
    )

    versions = np.array(
        [str(c.get("questionnaire_version", "")) for c in column_defs]
    )
    unique_versions = list(dict.fromkeys(versions.tolist()))

    if len(unique_versions) < 2:
        print(
            f"  Only one questionnaire_version present ({unique_versions!r}); "
            "skipping by-source."
        )
        return {"splits": [], "summary": None}

    splits: list[dict] = []

    # With exactly 2 versions, vs_rest and pairwise produce the same two
    # splits (just mirrored). Suppress vs_rest in that case to avoid printing
    # and saving duplicate work.
    effective_vs_rest = do_vs_rest and not (do_pairwise and len(unique_versions) == 2)

    if effective_vs_rest:
        for v in unique_versions:
            other = [u for u in unique_versions if u != v]
            splits.append(_run_one_source_split(
                data, versions, [v], other,
                scheme="vs_rest",
                n_factors=n_factors, fa_method=fa_method, rotation=rotation,
            ))

    if do_pairwise and len(unique_versions) >= 2:
        for i in range(len(unique_versions)):
            for j in range(i + 1, len(unique_versions)):
                va, vb = unique_versions[i], unique_versions[j]
                splits.append(_run_one_source_split(
                    data, versions, [va], [vb],
                    scheme="pairwise",
                    n_factors=n_factors, fa_method=fa_method, rotation=rotation,
                ))

    summary = {
        "n_splits": len(splits),
        "median_abs_r_by_scheme": {
            scheme: [
                s["median_abs_r"] for s in splits
                if s.get("scheme") == scheme and "median_abs_r" in s
            ]
            for scheme in ("vs_rest", "pairwise")
        },
    }
    return {"splits": splits, "summary": summary}


# ═════════════════════════════════════════════════════════════════════════════
# Plotting (simple — keeps the script self-contained)
# ═════════════════════════════════════════════════════════════════════════════


def _plot_random_halves(result: dict, save_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    per_iter = result.get("per_iter") or []
    if not per_iter:
        return
    max_k = max(len(p["abs_r_matched"]) for p in per_iter)
    mat = np.full((len(per_iter), max_k), np.nan)
    for i, p in enumerate(per_iter):
        mat[i, :len(p["abs_r_matched"])] = p["abs_r_matched"]

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * max_k + 2), 4.5))
    positions = np.arange(max_k)
    data = [mat[:, k][~np.isnan(mat[:, k])] for k in range(max_k)]
    ax.violinplot(data, positions=positions, widths=0.7, showmedians=True, showextrema=False)
    ax.axhline(0.85, color="#16a34a", linestyle="--", linewidth=0.8, alpha=0.5, label="0.85")
    ax.axhline(0.70, color="#f59e0b", linestyle="--", linewidth=0.8, alpha=0.6, label="0.70")
    ax.axhline(0.50, color="#dc2626", linestyle=":", linewidth=0.8, alpha=0.4, label="0.50")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"F{k}" for k in range(max_k)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("|Pearson r| of factor scores across item halves")
    ax.set_title(
        f"Item random-halves factor-score replication "
        f"({len(per_iter)} iters)"
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_by_source(result: dict, save_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    splits = [s for s in (result.get("splits") or []) if "abs_r_matched" in s]
    if not splits:
        return
    k = max(len(s["abs_r_matched"]) for s in splits)

    fig, ax = plt.subplots(figsize=(max(7, 1.5 * k + 2), 0.55 * len(splits) + 3))
    width = 0.8 / len(splits)
    x = np.arange(k)
    scheme_colors = {"vs_rest": "#3b82f6", "pairwise": "#f97316"}
    for i, s in enumerate(splits):
        vals = s["abs_r_matched"] + [np.nan] * (k - len(s["abs_r_matched"]))
        scheme = s.get("scheme", "vs_rest")
        label = (
            f"[{scheme}] {'+'.join(s['versions_a'])} ↔ {'+'.join(s['versions_b'])}"
        )
        ax.bar(
            x + (i - (len(splits) - 1) / 2) * width,
            vals,
            width=width,
            label=label,
            color=scheme_colors.get(scheme, "#64748b"),
            edgecolor="white",
            linewidth=0.3,
        )
    ax.axhline(0.85, color="#16a34a", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(0.70, color="#f59e0b", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(0.50, color="#dc2626", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{j}" for j in range(k)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("|Pearson r| of factor scores across item sources")
    ax.set_title(f"Item by-source factor-score replication (k={k})")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    q_dir = QUESTIONNAIRE_DIR / "questionnaire"
    response_matrix = np.load(q_dir / "response_matrix.npy")
    with open(q_dir / "metadata.jsonl") as f:
        metadata = [json.loads(line) for line in f if line.strip()]
    with open(q_dir / "items.json") as f:
        column_defs = json.load(f)

    print(f"Loaded response matrix: {response_matrix.shape}")
    print(f"  metadata rows: {len(metadata)}")
    print(f"  column defs:   {len(column_defs)}")

    version_counts: dict[str, int] = defaultdict(int)
    for c in column_defs:
        version_counts[str(c.get("questionnaire_version", ""))] += 1
    print(f"  item counts by questionnaire_version: {dict(version_counts)}")

    if INCLUDE_VERSIONS is not None:
        keep_versions = set(INCLUDE_VERSIONS)
        present = {str(c.get("questionnaire_version", "")) for c in column_defs}
        missing = keep_versions - present
        if missing:
            raise ValueError(
                f"INCLUDE_VERSIONS references versions not present in "
                f"{QUESTIONNAIRE_DIR}: missing={sorted(missing)}, "
                f"present={sorted(present)}"
            )
        col_mask = np.array(
            [str(c.get("questionnaire_version", "")) in keep_versions
             for c in column_defs],
            dtype=bool,
        )
        response_matrix = response_matrix[:, col_mask]
        column_defs = [c for c, keep in zip(column_defs, col_mask) if keep]
        print(
            f"\n  INCLUDE_VERSIONS filter applied: keeping "
            f"{int(col_mask.sum())}/{col_mask.size} columns "
            f"(versions={sorted(keep_versions)})"
        )
        post_counts: dict[str, int] = defaultdict(int)
        for c in column_defs:
            post_counts[str(c.get("questionnaire_version", ""))] += 1
        print(f"  post-filter item counts: {dict(post_counts)}")

    rollout_counts: dict[str, int] = defaultdict(int)
    for m in metadata:
        rollout_counts[str(m.get("rollout_preset_key", ""))] += 1
    print(f"  row counts by rollout_preset_key: {dict(rollout_counts)}")

    if INCLUDE_ROLLOUT_KEYS is not None:
        keep_keys = set(INCLUDE_ROLLOUT_KEYS)
        present = {str(m.get("rollout_preset_key", "")) for m in metadata}
        missing = keep_keys - present
        if missing:
            raise ValueError(
                f"INCLUDE_ROLLOUT_KEYS references rollouts not present in "
                f"{QUESTIONNAIRE_DIR}: missing={sorted(missing)}, "
                f"present={sorted(present)}"
            )
        row_mask = np.array(
            [str(m.get("rollout_preset_key", "")) in keep_keys for m in metadata],
            dtype=bool,
        )
        response_matrix = response_matrix[row_mask]
        metadata = [m for m, keep in zip(metadata, row_mask) if keep]
        print(
            f"\n  INCLUDE_ROLLOUT_KEYS filter applied: keeping "
            f"{int(row_mask.sum())}/{row_mask.size} rows "
            f"(rollouts={sorted(keep_keys)})"
        )
        post_rollout_counts: dict[str, int] = defaultdict(int)
        for m in metadata:
            post_rollout_counts[str(m.get("rollout_preset_key", ""))] += 1
        print(f"  post-filter row counts: {dict(post_rollout_counts)}")

    # ── Preprocess once (row NaN drop + per-block variance filter + NaN impute)
    print("\n" + "=" * 60)
    print("Preprocessing")
    print("=" * 60)
    # When TOP_N_ITEMS is set, disable the relative-variance floor in
    # preprocessing so top-N is the sole item filter. The row-drop + NaN
    # imputation pieces of preprocess_response_matrix still apply.
    effective_min_var = 0.0 if TOP_N_ITEMS is not None else MIN_ITEM_VARIANCE
    data, meta_filtered, cols_filtered, _ = preprocess_response_matrix(
        response_matrix, metadata, column_defs,
        min_item_variance=effective_min_var,
        high_variance_persona_drop_pct=0.0,
        do_residualize=False,
    )

    if TOP_N_ITEMS is not None:
        n_before = data.shape[1]
        if TOP_N_ITEMS >= n_before:
            print(
                f"\n  TOP_N_ITEMS={TOP_N_ITEMS} >= available columns "
                f"({n_before}); keeping all."
            )
        else:
            col_var = np.var(data, axis=0)
            col_blocks = np.array(
                [str(c.get("block", "")) for c in cols_filtered]
            )
            rel = np.zeros_like(col_var, dtype=np.float64)
            for b in np.unique(col_blocks):
                m = col_blocks == b
                pos = col_var[m][col_var[m] > 0]
                med = float(np.median(pos)) if pos.size > 0 else 0.0
                rel[m] = col_var[m] / med if med > 0 else 0.0
            # Keep top-N by relative variance; stable tie-break on col index
            # (argsort of -rel is stable in NumPy by default kind="stable"
            # only when explicitly asked — use mergesort to guarantee).
            order = np.argsort(-rel, kind="stable")
            keep_idx = np.sort(order[:TOP_N_ITEMS])
            data = data[:, keep_idx]
            cols_filtered = [cols_filtered[i] for i in keep_idx.tolist()]
            print(
                f"\n  TOP_N_ITEMS filter: kept top {TOP_N_ITEMS}/{n_before} "
                f"columns by per-block relative variance"
            )
            topn_counts: dict[str, int] = defaultdict(int)
            for c in cols_filtered:
                topn_counts[str(c.get("questionnaire_version", ""))] += 1
            topn_blocks: dict[str, int] = defaultdict(int)
            for c in cols_filtered:
                topn_blocks[str(c.get("block", ""))] += 1
            print(f"  post-top-N item counts by version: {dict(topn_counts)}")
            print(f"  post-top-N item counts by block:   {dict(topn_blocks)}")
    # Column-mean impute any residual NaNs (matches main FA stage behaviour).
    nan_mask = np.isnan(data)
    if nan_mask.any():
        col_means = np.nanmean(data, axis=0)
        inds = np.where(nan_mask)
        data[inds] = np.take(col_means, inds[1])
        print(f"  Mean-imputed {int(nan_mask.sum())} residual NaN cells")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config_snapshot = {
        "seed": SEED,
        "questionnaire_dir": str(QUESTIONNAIRE_DIR),
        "output_dir": str(OUTPUT_DIR),
        "fa_method": FA_METHOD,
        "fa_rotation": FA_ROTATION,
        "k_values": list(K_VALUES),
        "include_versions": INCLUDE_VERSIONS,
        "include_rollout_keys": INCLUDE_ROLLOUT_KEYS,
        "min_item_variance": MIN_ITEM_VARIANCE,
        "top_n_items": TOP_N_ITEMS,
        "n_random_iters": N_RANDOM_ITERS,
        "random_halves_stratify_by_version": RANDOM_HALVES_STRATIFY_BY_VERSION,
        "run_random_halves": RUN_RANDOM_HALVES,
        "run_by_source": RUN_BY_SOURCE,
        "by_source_vs_rest": BY_SOURCE_VS_REST,
        "by_source_pairwise": BY_SOURCE_PAIRWISE,
        "n_rows_after_preprocess": int(data.shape[0]),
        "n_cols_after_preprocess": int(data.shape[1]),
        "item_counts_by_version_after_preprocess": {
            v: int(sum(1 for c in cols_filtered
                       if str(c.get("questionnaire_version", "")) == v))
            for v in sorted({str(c.get("questionnaire_version", "")) for c in cols_filtered})
        },
    }
    with (OUTPUT_DIR / "config.json").open("w") as f:
        json.dump(config_snapshot, f, indent=2)

    cross_k: dict[str, dict] = {}

    for k in K_VALUES:
        k_dir = OUTPUT_DIR / f"k{k}"
        k_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "#" * 70)
        print(f"# k = {k}")
        print("#" * 70)

        k_entry: dict = {"k": k}

        # ── Random halves ──────────────────────────────────────────────
        if RUN_RANDOM_HALVES:
            print("\n" + "=" * 60)
            print(f"[k={k}] Random-halves item-split validation")
            print("=" * 60)
            rh_result = run_random_halves(
                data, cols_filtered,
                n_iters=N_RANDOM_ITERS,
                n_factors=k,
                fa_method=FA_METHOD,
                rotation=FA_ROTATION,
                stratify_by_version=RANDOM_HALVES_STRATIFY_BY_VERSION,
                seed=SEED,
            )
            with (k_dir / "random_halves.json").open("w") as f:
                json.dump(rh_result, f, indent=2)
            _plot_random_halves(rh_result, k_dir / "random_halves.png")
            print(f"  Wrote {k_dir / 'random_halves.json'}")
            k_entry["random_halves_summary"] = rh_result.get("summary")

        # ── By source ──────────────────────────────────────────────────
        if RUN_BY_SOURCE:
            print("\n" + "=" * 60)
            print(f"[k={k}] By-source item-split validation")
            print("=" * 60)
            bs_result = run_by_source(
                data, cols_filtered,
                n_factors=k,
                fa_method=FA_METHOD,
                rotation=FA_ROTATION,
                seed=SEED,
                do_vs_rest=BY_SOURCE_VS_REST,
                do_pairwise=BY_SOURCE_PAIRWISE,
            )
            with (k_dir / "by_source.json").open("w") as f:
                json.dump(bs_result, f, indent=2)
            _plot_by_source(bs_result, k_dir / "by_source.png")
            print(f"  Wrote {k_dir / 'by_source.json'}")
            k_entry["by_source_per_split_median"] = [
                {
                    "scheme": s.get("scheme"),
                    "versions_a": s.get("versions_a"),
                    "versions_b": s.get("versions_b"),
                    "median_abs_r": s.get("median_abs_r"),
                    "abs_r_matched": s.get("abs_r_matched"),
                }
                for s in (bs_result.get("splits") or [])
            ]

        cross_k[f"k{k}"] = k_entry

    with (OUTPUT_DIR / "cross_k_summary.json").open("w") as f:
        json.dump(cross_k, f, indent=2)

    # Print a compact cross-k recap.
    print("\n" + "=" * 70)
    print("CROSS-K RECAP")
    print("=" * 70)
    for k in K_VALUES:
        entry = cross_k[f"k{k}"]
        rh = entry.get("random_halves_summary")
        rh_line = (
            f"rand-halves median |r|={rh['overall_median_abs_r']:.3f}  "
            f"per-factor meds=[{', '.join(f'{v:.2f}' for v in rh['median_abs_r_per_factor'])}]"
            if rh else "rand-halves: —"
        )
        print(f"\n  k={k}   {rh_line}")
        for s in entry.get("by_source_per_split_median", []):
            tag = f"[{s['scheme']}] {'+'.join(s['versions_a'])} ↔ {'+'.join(s['versions_b'])}"
            if s.get("median_abs_r") is None:
                print(f"    {tag}: (FA failed)")
            else:
                per = s.get("abs_r_matched") or []
                print(
                    f"    {tag}: median |r|={s['median_abs_r']:.3f}  "
                    f"per-factor=[{', '.join(f'{v:.2f}' for v in per)}]"
                )

    print(f"\nDone. Results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
