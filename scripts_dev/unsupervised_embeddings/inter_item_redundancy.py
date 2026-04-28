"""Inter-item redundancy analysis for the v5 Likert questionnaire.

Reads the per-model hydrated response matrices written by
``analysis_for_paper.py`` and emits a redundancy report so the questionnaire
can be pruned of near-duplicate items before the next run.

Procedure:

    1. Sign-orient each item by flipping reverse-keyed columns so positive
       correlation = same direction on the underlying construct.
    2. Compute Pearson inter-item correlation per model.
    3. List pairs with |r| ≥ THRESHOLD in BOTH models (most-defensible
       redundancy signal — a pair that is redundant in only one model may be
       a real model-specific entanglement, not a duplicate question).
    4. Single-link agglomerative cluster items at distance d = 1 − |r_min|
       (using the elementwise min |r| across models) so we can see clusters
       of mutually-redundant items, not just pairs.

Outputs (under ``scratch/psychometric_fa_paper/inter_item_redundancy/``):

    - ``per_model_corr_<slug>.npy`` — full sign-oriented correlation matrix.
    - ``redundant_pairs.txt`` — human-readable, sorted by min |r|.
    - ``redundant_clusters.txt`` — single-link clusters at threshold.
    - ``summary.json`` — counts and thresholds.

Run with: ``python scripts_dev/unsupervised_embeddings/inter_item_redundancy.py``
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

SEED = 436
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path("scratch/psychometric_fa_paper")
MODELS = ["llama-3.1-8b", "qwen2.5-7b"]
OUT = ROOT / "inter_item_redundancy"
OUT.mkdir(parents=True, exist_ok=True)

# Thresholds. 0.7 is "strong" redundancy; we also report 0.6 as "borderline".
PAIR_THRESHOLD = 0.7
CLUSTER_THRESHOLD = 0.7  # single-link distance cut at 1 − threshold
BORDERLINE_THRESHOLD = 0.6


def load_model(slug: str):
    base = ROOT / slug / "hydrated" / "questionnaire"
    M = np.load(base / "response_matrix.npy").astype(float)
    items = json.loads((base / "items.json").read_text())
    assert M.shape[1] == len(items)
    # Sign-orient: flip reverse-keyed items so positive correlation =
    # agreement on the underlying construct.
    flip = np.array([-1.0 if it.get("reverse_keyed") else 1.0 for it in items])
    encoded = items[0]["encoding"]
    if encoded == "1-5":
        center = 3.0
    else:
        center = float(np.nanmean(M))
    M_oriented = (M - center) * flip + center
    # Drop columns with zero variance (would yield NaN correlations).
    var = np.nanvar(M_oriented, axis=0)
    keep = var > 1e-9
    return M_oriented[:, keep], [it for it, k in zip(items, keep) if k]


def pearson_corr(M: np.ndarray) -> np.ndarray:
    # np.corrcoef on rows; we want columns.
    Mc = M - np.nanmean(M, axis=0, keepdims=True)
    std = np.nanstd(M, axis=0, keepdims=True)
    Z = Mc / np.where(std > 0, std, 1.0)
    n = Z.shape[0]
    R = (Z.T @ Z) / n
    np.fill_diagonal(R, 1.0)
    return R


def short(text: str, n: int = 90) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def main() -> None:
    per_model_R = {}
    items_by_model = {}
    item_texts_by_id: dict[str, str] = {}
    reverse_by_id: dict[str, bool] = {}

    for slug in MODELS:
        M, items = load_model(slug)
        R = pearson_corr(M)
        per_model_R[slug] = R
        items_by_model[slug] = items
        np.save(OUT / f"per_model_corr_{slug}.npy", R)
        for it in items:
            item_texts_by_id.setdefault(it["item_id"], it["text"])
            reverse_by_id.setdefault(it["item_id"], bool(it.get("reverse_keyed")))

    # Restrict to the intersection of item_ids so a pair is comparable across
    # models. (Both models drop ~2 items at the variance floor.)
    common_ids = set(it["item_id"] for it in items_by_model[MODELS[0]])
    for slug in MODELS[1:]:
        common_ids &= set(it["item_id"] for it in items_by_model[slug])
    common_ids = sorted(common_ids, key=lambda s: int(s) if s.isdigit() else s)
    print(f"common items across models: {len(common_ids)}")

    # Reindex each model's R to common_ids order.
    def reindex(R: np.ndarray, items: list[dict], ids: list[str]) -> np.ndarray:
        idx = {it["item_id"]: i for i, it in enumerate(items)}
        order = [idx[i] for i in ids]
        return R[np.ix_(order, order)]

    R_l = reindex(per_model_R["llama-3.1-8b"], items_by_model["llama-3.1-8b"], common_ids)
    R_q = reindex(per_model_R["qwen2.5-7b"], items_by_model["qwen2.5-7b"], common_ids)
    R_min = np.minimum(np.abs(R_l), np.abs(R_q))  # most-conservative redundancy
    np.fill_diagonal(R_min, 0.0)

    # Pairs above threshold in BOTH models.
    n = len(common_ids)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            min_abs = R_min[i, j]
            if min_abs >= BORDERLINE_THRESHOLD:
                pairs.append((min_abs, R_l[i, j], R_q[i, j], common_ids[i], common_ids[j]))
    pairs.sort(reverse=True)

    # Write pair report.
    lines = []
    lines.append(f"Inter-item redundancy: pairs with min(|r_llama|, |r_qwen|) ≥ {BORDERLINE_THRESHOLD:.2f}")
    lines.append(f"n_common_items={n}  threshold_strong={PAIR_THRESHOLD}  threshold_borderline={BORDERLINE_THRESHOLD}")
    lines.append("")
    n_strong = 0
    for min_abs, r_l, r_q, a, b in pairs:
        tag = "[STRONG]" if min_abs >= PAIR_THRESHOLD else "[border]"
        if min_abs >= PAIR_THRESHOLD:
            n_strong += 1
        sa = "(R)" if reverse_by_id[a] else "   "
        sb = "(R)" if reverse_by_id[b] else "   "
        lines.append(
            f"{tag} min|r|={min_abs:.3f}  llama r={r_l:+.3f}  qwen r={r_q:+.3f}  "
            f"items {a}↔{b}"
        )
        lines.append(f"    {a} {sa} {short(item_texts_by_id[a])}")
        lines.append(f"    {b} {sb} {short(item_texts_by_id[b])}")
        lines.append("")
    (OUT / "redundant_pairs.txt").write_text("\n".join(lines))
    print(f"wrote {OUT / 'redundant_pairs.txt'}  ({n_strong} strong, {len(pairs)} total)")

    # Single-link agglomerative clustering at CLUSTER_THRESHOLD on R_min.
    # Use union-find over the high-similarity edges.
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if R_min[i, j] >= CLUSTER_THRESHOLD:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    multi = [sorted(c) for c in clusters.values() if len(c) >= 2]
    multi.sort(key=lambda c: (-len(c), c[0]))

    cl_lines = [f"Single-link clusters at min|r| ≥ {CLUSTER_THRESHOLD:.2f}"]
    cl_lines.append(f"{len(multi)} non-singleton clusters covering {sum(len(c) for c in multi)} items\n")
    for ci, cluster in enumerate(multi):
        cl_lines.append(f"== Cluster {ci+1}  size={len(cluster)} ==")
        for idx in cluster:
            iid = common_ids[idx]
            tag = "(R)" if reverse_by_id[iid] else "   "
            cl_lines.append(f"  {iid} {tag} {short(item_texts_by_id[iid])}")
        cl_lines.append("")
    (OUT / "redundant_clusters.txt").write_text("\n".join(cl_lines))
    print(f"wrote {OUT / 'redundant_clusters.txt'}  ({len(multi)} multi-item clusters)")

    summary = {
        "n_common_items": n,
        "pair_threshold_strong": PAIR_THRESHOLD,
        "pair_threshold_borderline": BORDERLINE_THRESHOLD,
        "cluster_threshold": CLUSTER_THRESHOLD,
        "n_pairs_strong": n_strong,
        "n_pairs_borderline": len(pairs) - n_strong,
        "n_clusters_multi_item": len(multi),
        "n_items_in_multi_clusters": sum(len(c) for c in multi),
        "models": MODELS,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
