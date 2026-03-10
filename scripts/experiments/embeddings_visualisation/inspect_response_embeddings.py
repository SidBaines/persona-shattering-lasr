# %%
#!/usr/bin/env python3
"""Notebook-style inspection for response embedding outputs."""

#%%
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#%%
# Update this if you want to inspect a different run.
RUN_ID = "stage123-240x50-singleturn-v2"
PREFIX = "response_embeddings"

# RUN_DIR = Path("../../../scratch") / "runs" / RUN_ID
RUN_DIR = Path("scratch") / "runs" / RUN_ID
REPORTS_DIR = RUN_DIR / "reports"
METADATA_PATH = REPORTS_DIR / f"{PREFIX}_metadata.jsonl"
EMBEDDINGS_PATH = REPORTS_DIR / f"{PREFIX}_embeddings.npy"
VARIANCE_PATH = REPORTS_DIR / f"{PREFIX}_variance.json"
MANIFEST_PATH = REPORTS_DIR / f"{PREFIX}_manifest.json"

print(f"Run dir: {RUN_DIR}")
print(f"Metadata: {METADATA_PATH}")
print(f"Embeddings: {EMBEDDINGS_PATH}")
print(f"Variance: {VARIANCE_PATH}")
print(f"Manifest: {MANIFEST_PATH}")


#%%
# Load artifacts.
with METADATA_PATH.open("r", encoding="utf-8") as handle:
    metadata = [json.loads(line) for line in handle if line.strip()]

embeddings = np.load(EMBEDDINGS_PATH)

variance_report = {}
if VARIANCE_PATH.exists():
    variance_report = json.loads(VARIANCE_PATH.read_text(encoding="utf-8"))

manifest = {}
if MANIFEST_PATH.exists():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

print(f"Loaded metadata rows: {len(metadata)}")
print(f"Loaded embeddings shape: {embeddings.shape}, dtype={embeddings.dtype}")


#%%
# Basic integrity checks.
assert embeddings.ndim == 2, f"Expected 2D embedding matrix, got {embeddings.shape}"
assert len(metadata) == embeddings.shape[0], (
    f"Metadata row count {len(metadata)} must match embeddings rows {embeddings.shape[0]}"
)

for i, row in enumerate(metadata[:100]):
    assert row["embedding_index"] == i, (
        f"embedding_index mismatch at row {i}: {row['embedding_index']}"
    )

print("Integrity checks passed.")


#%%
# Build prompt-group labels and index maps.
group_ids = np.array([str(row["input_group_id"]) for row in metadata], dtype=object)
unique_groups, group_inverse = np.unique(group_ids, return_inverse=True)
num_groups = len(unique_groups)
n, d = embeddings.shape

group_to_indices = {g: np.where(group_ids == g)[0] for g in unique_groups}
group_sizes = np.array([len(group_to_indices[g]) for g in unique_groups], dtype=int)

print(f"Num groups (prompts): {num_groups}")
print(f"Group size stats: min={group_sizes.min()} median={int(np.median(group_sizes))} max={group_sizes.max()}")
print(f"n={n}, d={d}")


#%%
# Decompose total embedding variance into between-prompt and within-prompt parts.
global_mean = embeddings.mean(axis=0)
group_means = np.zeros((num_groups, d), dtype=np.float64)
for gi, g in enumerate(unique_groups):
    group_means[gi] = embeddings[group_to_indices[g]].mean(axis=0)

residual_embeddings = embeddings - group_means[group_inverse]

# scaled embeddings 
scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(embeddings)
resid_scaler = StandardScaler()
scaled_residual_embeddings = resid_scaler.fit_transform(residual_embeddings)

ss_total = float(np.square(embeddings - global_mean).sum())
ss_between = float(np.sum(group_sizes[:, None] * np.square(group_means - global_mean)))
ss_within = float(np.square(residual_embeddings).sum())

print("Variance decomposition (sum-of-squares):")
print(f"  total   : {ss_total:.3e}")
print(f"  between : {ss_between:.3e} ({100 * ss_between / ss_total:.2f}% of total)")
print(f"  within  : {ss_within:.3e} ({100 * ss_within / ss_total:.2f}% of total)")
print(f"  check   : between + within = {(ss_between + ss_within):.3e}")


#%%
# Quick summary of fields and global variance stats.
print("Metadata fields:", sorted(metadata[0].keys()))
if variance_report:
    print("Global variance summary:")
    for key, value in variance_report.get("global", {}).items():
        print(f"  {key}: {value}")

if manifest:
    print("\nManifest config:")
    print(json.dumps(manifest.get("local_hf", {}), indent=2))


#%%
# Fit PCA
pca = PCA(n_components=50, random_state=0)
pcs = pca.fit_transform(scaled_embeddings)

## % 
# Plot explained variance ratios for PCA
print("Top explained variance ratios:")
for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
    print(f"  PC{i}: {ratio:.5f}")

plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker="o")
plt.title("PCA Explained Variance Ratio")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Plot eigenvalues for PCA
plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, len(pca.explained_variance_) + 1), pca.explained_variance_, marker="o")
plt.title("PCA Eigenvalues")
plt.xlabel("Principal Component")
plt.ylabel("Eigenvalue")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#%%
# Fit PCA of residual embeddings
resid_pca = PCA(n_components=50, random_state=2)
resid_pcs = resid_pca.fit_transform(scaled_residual_embeddings)

## % 
# Plot explained variance ratios for residual PCA
print("Top explained variance ratios:")
for i, ratio in enumerate(resid_pca.explained_variance_ratio_, start=1):
    print(f"  PC{i}: {ratio:.5f}")

plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, len(resid_pca.explained_variance_ratio_) + 1), resid_pca.explained_variance_ratio_, marker="o")
plt.title("Residual PCA Explained Variance Ratio")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Plot eigenvalues for residual PCA
plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, len(resid_pca.explained_variance_) + 1), resid_pca.explained_variance_, marker="o")
plt.title("Residual PCA Eigenvalues")
plt.xlabel("Principal Component")
plt.ylabel("Eigenvalue")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#%%
# 3D PCA scatter (PC1/PC2/PC3), colored by prompt group.
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    pcs[:, 0],
    pcs[:, 1],
    pcs[:, 2],
    c=group_inverse,
    cmap="tab20",
    s=8,
    alpha=0.65,
)
ax.set_title("3D PCA of Response Embeddings (colored by prompt group)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.08)
cbar.set_label("Prompt group index")

plt.tight_layout()
plt.show()


#%%
# 3D PCA for a configurable subset of prompt groups (readable legend).
N_GROUPS_TO_PLOT = 30
GROUP_SELECTION_MODE = "random"  # "largest", "random", or "top_variance"
RANDOM_SEED = 0

if GROUP_SELECTION_MODE == "largest":
    sorted_groups = sorted(unique_groups, key=lambda g: len(group_to_indices[g]), reverse=True)
    selected_groups = sorted_groups[:N_GROUPS_TO_PLOT]
elif GROUP_SELECTION_MODE == "random":
    rng = np.random.default_rng(RANDOM_SEED)
    selected_groups = list(
        rng.choice(unique_groups, size=min(N_GROUPS_TO_PLOT, len(unique_groups)), replace=False)
    )
    selected_groups = selected_groups[21:24]
elif GROUP_SELECTION_MODE == "top_variance":
    if variance_report and variance_report.get("per_prompt"):
        selected_groups = [
            row["input_group_id"]
            for row in variance_report["per_prompt"][:N_GROUPS_TO_PLOT]
            if row["input_group_id"] in group_to_indices
        ]
    else:
        sorted_groups = sorted(unique_groups, key=lambda g: len(group_to_indices[g]), reverse=True)
        selected_groups = sorted_groups[:N_GROUPS_TO_PLOT]
else:
    raise ValueError(f"Unsupported GROUP_SELECTION_MODE: {GROUP_SELECTION_MODE}")

selected_groups = list(selected_groups)
print(f"Selected groups ({len(selected_groups)}):")
for g in selected_groups:
    print(f"  - {g} (n={len(group_to_indices[g])})")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

cmap = plt.get_cmap("tab10")
for i, g in enumerate(selected_groups):
    idx = group_to_indices[g]
    ax.scatter(
        pcs[idx, 0],
        pcs[idx, 1],
        pcs[idx, 2],
        s=14,
        alpha=0.75,
        color=cmap(i % 10),
        label=f"group {i}: n={len(idx)}",
    )

ax.set_title(f"3D PCA for selected prompt groups (N={len(selected_groups)})")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend(loc="best", fontsize=8)

plt.tight_layout()
plt.show()


#%%
# Prompt effect per principal component (original embeddings).
def one_way_eta_squared(values: np.ndarray, group_inv: np.ndarray, num_groups_: int) -> float:
    """Fraction of 1D variance explained by group means."""
    grand = float(values.mean())
    ss_total_1d = float(np.square(values - grand).sum())
    if ss_total_1d <= 0.0:
        return 0.0

    ss_between_1d = 0.0
    for gi in range(num_groups_):
        idx = np.where(group_inv == gi)[0]
        if len(idx) == 0:
            continue
        m = float(values[idx].mean())
        ss_between_1d += float(len(idx) * (m - grand) ** 2)
    return ss_between_1d / ss_total_1d


pc_prompt_eta = np.array(
    [one_way_eta_squared(pcs[:, i], group_inverse, num_groups) for i in range(pcs.shape[1])]
)
print("Prompt-explained variance ratio per PC (original embeddings):")
for i, eta in enumerate(pc_prompt_eta, start=1):
    print(f"  PC{i}: eta^2_prompt={eta:.4f}")

plt.figure(figsize=(8, 4))
plt.bar(np.arange(1, len(pc_prompt_eta) + 1), pc_prompt_eta)
plt.ylim(0, 1)
plt.title("How Much Each PC Is Explained By Prompt Group")
plt.xlabel("Principal Component")
plt.ylabel("eta^2(prompt)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


#%%
# PCA on prompt-residualized embeddings.
resid_pca = PCA(n_components=10, random_state=0)
resid_pcs = resid_pca.fit_transform(residual_embeddings)

print("Residual PCA explained variance ratios:")
for i, ratio in enumerate(resid_pca.explained_variance_ratio_, start=1):
    print(f"  PC{i}: {ratio:.5f}")

resid_pc_prompt_eta = np.array(
    [one_way_eta_squared(resid_pcs[:, i], group_inverse, num_groups) for i in range(resid_pcs.shape[1])]
)
print("Prompt-explained variance ratio per PC (after residualization):")
for i, eta in enumerate(resid_pc_prompt_eta, start=1):
    print(f"  rPC{i}: eta^2_prompt={eta:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(resid_pcs[:, 0], resid_pcs[:, 1], s=8, alpha=0.45)
plt.title("Residual Embeddings projected to rPC1/rPC2 (prompt mean removed)")
plt.xlabel("rPC1")
plt.ylabel("rPC2")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


#%%
# 2D PCA scatter.
# For single-turn runs, assistant_turn_index may be constant; this still works.
turn_indices = np.array([row.get("assistant_turn_index", -1) for row in metadata], dtype=int)
unique_turns = np.unique(turn_indices)

plt.figure(figsize=(8, 6))
for turn in unique_turns:
    mask = turn_indices == turn
    plt.scatter(
        pcs[mask, 0],
        pcs[mask, 1],
        s=8,
        alpha=0.5,
        label=f"turn={turn}",
    )
plt.title("Response Embeddings projected to PC1/PC2")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(loc="best", markerscale=2, fontsize=8)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


#%%
# Optional: inspect top prompts by within-prompt variance from saved report.
top = variance_report.get("top_prompts_by_total_variance", []) if variance_report else []
print(f"Top prompts by total variance: {min(10, len(top))}")
for row in top[:10]:
    seed = row.get("seed_user_message", "").replace("\n", " ")
    print(
        f"- group={row['input_group_id']} n={row['num_samples']} "
        f"var={row['total_variance']:.4f} prompt={seed[:100]}"
    )


#%%
# Simple nearest-neighbor probe using cosine similarity.
def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_norm @ b_norm.T


query_index = 0
query_vec = embeddings[query_index : query_index + 1]
sims = cosine_similarity_matrix(query_vec, embeddings).ravel()
nearest = np.argsort(-sims)[:6]  # includes self at rank 1

print(f"Query index: {query_index}")
print("Nearest neighbors (cosine):")
for idx in nearest:
    text = metadata[idx].get("assistant_text", "").replace("\n", " ")
    print(f"  idx={idx:5d} sim={sims[idx]:.4f} text={text[:120]}")


#%%
# Nearest-neighbor retrieval constrained to *different prompt groups*.
query_group = group_ids[query_index]
different_group_idx = np.where(group_ids != query_group)[0]
cross_group_ranking = different_group_idx[np.argsort(-sims[different_group_idx])[:10]]

print(f"Query index: {query_index}, query group: {query_group}")
print("Top cross-group neighbors (same prompt excluded):")
for idx in cross_group_ranking:
    text = metadata[idx].get("assistant_text", "").replace("\n", " ")
    print(
        f"  idx={idx:5d} group={group_ids[idx]} sim={sims[idx]:.4f} text={text[:120]}"
    )


#%%
# 3D residual PCA for a configurable subset of prompt groups (final plot cell).
# Requires resid_pcs to have been computed in the earlier residualization cell.
N_GROUPS_TO_PLOT_RESID = 240
GROUP_SELECTION_MODE_RESID = "random"  # "largest", "random", or "top_variance"
RANDOM_SEED_RESID = 0

if GROUP_SELECTION_MODE_RESID == "largest":
    sorted_groups_resid = sorted(unique_groups, key=lambda g: len(group_to_indices[g]), reverse=True)
    selected_groups_resid = sorted_groups_resid[:N_GROUPS_TO_PLOT_RESID]
elif GROUP_SELECTION_MODE_RESID == "random":
    rng_resid = np.random.default_rng(RANDOM_SEED_RESID)
    selected_groups_resid = list(
        rng_resid.choice(
            unique_groups,
            size=min(N_GROUPS_TO_PLOT_RESID, len(unique_groups)),
            replace=False,
        )
    )
elif GROUP_SELECTION_MODE_RESID == "top_variance":
    if variance_report and variance_report.get("per_prompt"):
        selected_groups_resid = [
            row["input_group_id"]
            for row in variance_report["per_prompt"][:N_GROUPS_TO_PLOT_RESID]
            if row["input_group_id"] in group_to_indices
        ]
    else:
        sorted_groups_resid = sorted(unique_groups, key=lambda g: len(group_to_indices[g]), reverse=True)
        selected_groups_resid = sorted_groups_resid[:N_GROUPS_TO_PLOT_RESID]
else:
    raise ValueError(f"Unsupported GROUP_SELECTION_MODE_RESID: {GROUP_SELECTION_MODE_RESID}")

selected_groups_resid = list(selected_groups_resid)
print(f"Selected residual groups ({len(selected_groups_resid)}):")
for g in selected_groups_resid:
    print(f"  - {g} (n={len(group_to_indices[g])})")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
cmap = plt.get_cmap("tab10")

for i, g in enumerate(selected_groups_resid):
    idx = group_to_indices[g]
    ax.scatter(
        resid_pcs[idx, 0],
        resid_pcs[idx, 1],
        resid_pcs[idx, 2],
        s=14,
        alpha=0.75,
        color=cmap(i % 10),
        label=f"group {i}: n={len(idx)}",
    )

ax.set_title(f"3D Residual PCA for selected prompt groups (N={len(selected_groups_resid)})")
ax.set_xlabel("rPC1")
ax.set_ylabel("rPC2")
ax.set_zlabel("rPC3")
ax.legend(loc="best", fontsize=8)

plt.tight_layout()
plt.show()

# %%
#%%
# Manual cluster split + quick "vibe" comparison for two prompt groups on PC1/PC2.

TARGET_GROUPS = [
    "sample_8aed79e50b068be7bbc1a426",
    # "sample_826c163c2ed2166704381efb",
]

# 1) Visualize only these two groups in PC1/PC2
mask_target = np.isin(group_ids, TARGET_GROUPS)
x = pcs[:, 0]
y = pcs[:, 1]

plt.figure(figsize=(8, 6))
for g, c in zip(TARGET_GROUPS, ["tab:blue", "tab:orange"]):
    m = mask_target & (group_ids == g)
    plt.scatter(x[m], y[m], s=22, alpha=0.75, label=f"{g} (n={m.sum()})", color=c)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Target groups in PC1/PC2")
plt.legend()
plt.grid(alpha=0.25)
plt.show()

# 2) Manually define cluster rules in PC1/PC2 space.
#    Edit these thresholds after looking at the plot.
#    Each rule should be a boolean expression over xi, yi.
RULES = {
    "g1_cluster_A": lambda xi, yi: (xi > 0.02) & (yi > -0.01),
    "g1_cluster_B": lambda xi, yi: (xi <= 0.02) | (yi <= -0.01),
    "g2_cluster_A": lambda xi, yi: (xi > 0.02) & (yi > -0.01),
    "g2_cluster_B": lambda xi, yi: (xi <= 0.02) | (yi <= -0.01),
}

# 3) Build cluster masks
g1, g2 = TARGET_GROUPS
m1 = group_ids == g1
m2 = group_ids == g2

cluster_masks = {
    "g1_cluster_A": m1 & RULES["g1_cluster_A"](x, y),
    "g1_cluster_B": m1 & RULES["g1_cluster_B"](x, y),
    "g2_cluster_A": m2 & RULES["g2_cluster_A"](x, y),
    "g2_cluster_B": m2 & RULES["g2_cluster_B"](x, y),
}

for name, m in cluster_masks.items():
    print(f"{name}: n={int(m.sum())}")

# 4) Helper functions
def _norm_rows(a: np.ndarray) -> np.ndarray:
    return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)

def centroid(v: np.ndarray) -> np.ndarray:
    c = v.mean(axis=0, keepdims=True)
    return _norm_rows(c)[0]

def mean_pairwise_cos(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    an = _norm_rows(a)
    bn = _norm_rows(b)
    return float((an @ bn.T).mean())

def top_cross_pairs(mask_a: np.ndarray, mask_b: np.ndarray, top_k: int = 8):
    ia = np.where(mask_a)[0]
    ib = np.where(mask_b)[0]
    if len(ia) == 0 or len(ib) == 0:
        return []
    A = _norm_rows(embeddings[ia])
    B = _norm_rows(embeddings[ib])
    S = A @ B.T
    flat = np.argsort(S.ravel())[::-1][:top_k]
    out = []
    for f in flat:
        ra, rb = np.unravel_index(f, S.shape)
        i, j = ia[ra], ib[rb]
        out.append((i, j, float(S[ra, rb])))
    return out

# 5) Compare cluster centroids + within/between similarities
cluster_names = list(cluster_masks.keys())
centroids = {}
for name in cluster_names:
    idx = np.where(cluster_masks[name])[0]
    centroids[name] = centroid(embeddings[idx]) if len(idx) else None

print("\nCentroid cosine similarity matrix:")
for a in cluster_names:
    row = []
    for b in cluster_names:
        if centroids[a] is None or centroids[b] is None:
            row.append("   nan")
        else:
            row.append(f"{float(np.dot(centroids[a], centroids[b])):6.3f}")
    print(f"{a:13s} -> {' '.join(row)}")

print("\nMean pairwise cosine (within and cross):")
pairs = [
    ("g1_cluster_A", "g1_cluster_B"),
    ("g2_cluster_A", "g2_cluster_B"),
    ("g1_cluster_A", "g2_cluster_A"),
    ("g1_cluster_A", "g2_cluster_B"),
    ("g1_cluster_B", "g2_cluster_A"),
    ("g1_cluster_B", "g2_cluster_B"),
]
for a, b in pairs:
    v = mean_pairwise_cos(embeddings[cluster_masks[a]], embeddings[cluster_masks[b]])
    print(f"{a:13s} vs {b:13s}: {v:.4f}")

# 6) Print top cross-cluster nearest pairs for qualitative inspection
CHECK_PAIR = ("g1_cluster_A", "g2_cluster_A")  # change to any pair above
print(f"\nTop cross-cluster nearest pairs: {CHECK_PAIR[0]} vs {CHECK_PAIR[1]}")
for i, j, s in top_cross_pairs(cluster_masks[CHECK_PAIR[0]], cluster_masks[CHECK_PAIR[1]], top_k=10):
    ti = metadata[i].get("assistant_text", "").replace("\n", " ")
    tj = metadata[j].get("assistant_text", "").replace("\n", " ")
    print(f"\ncos={s:.4f}")
    print(f"  i={i} group={metadata[i]['input_group_id']} text={ti[:180]}")
    print(f"  j={j} group={metadata[j]['input_group_id']} text={tj[:180]}")

# %%
#%%
# Print question once, then each completion in the target group with its PC2 score.

TARGET_GROUP = "sample_8aed79e50b068be7bbc1a426"

idx = np.where(group_ids == TARGET_GROUP)[0]
if len(idx) == 0:
    print(f"No rows found for group: {TARGET_GROUP}")
else:
    # Sort by PC2 for easier inspection of the spread.
    idx = idx[np.argsort(pcs[idx, 1])]

    question = metadata[idx[0]].get("seed_user_message", "")
    print(f"Group: {TARGET_GROUP}")
    print(f"n={len(idx)}")
    print("\nQuestion:")
    print(question)
    print("\nRows (sorted by PC2):")

    for rank, i in enumerate(idx, start=1):
        pc2 = float(pcs[i, 1])
        text = metadata[i].get("assistant_text", "")
        print("\n" + "=" * 100)
        print(f"[{rank:02d}/{len(idx)}] idx={i}  PC2={pc2:+.5f}")
        print("-" * 100)
        print(text)

# %%


#%%
# Test 0: Setup for this prompt group
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

TARGET_GROUP = "sample_8aed79e50b068be7bbc1a426"
idx = np.where(group_ids == TARGET_GROUP)[0]
if len(idx) == 0:
    raise ValueError(f"No rows for group {TARGET_GROUP}")

pc2 = pcs[idx, 1]
texts = [metadata[i].get("assistant_text", "") for i in idx]
question = metadata[idx[0]].get("seed_user_message", "")

print(f"Group={TARGET_GROUP}, n={len(idx)}")
print("Question:", question)

def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

# %%

#%%
# Test 1: Verbosity hypothesis (brief refusal <-> long elaboration)
word_counts = [len(re.findall(r"\S+", t)) for t in texts]
char_counts = [len(t) for t in texts]
sent_counts = [len([s for s in re.split(r"[.!?]+", t) if s.strip()]) for t in texts]

print("corr(PC2, words):", round(safe_corr(pc2, word_counts), 4))
print("corr(PC2, chars):", round(safe_corr(pc2, char_counts), 4))
print("corr(PC2, sents):", round(safe_corr(pc2, sent_counts), 4))

order = np.argsort(pc2)
plt.figure(figsize=(8, 5))
plt.scatter(np.array(word_counts)[order], pc2[order], alpha=0.8, s=30)
plt.xlabel("Word count")
plt.ylabel("PC2")
plt.title("PC2 vs verbosity (target group)")
plt.grid(alpha=0.25)
plt.show()

# %%
#%%
# Test 2: Follow-up-question style hypothesis
def followup_features(t):
    t_strip = t.strip()
    return {
        "qmarks": t.count("?"),
        "ends_with_q": int(t_strip.endswith("?")),
        "asks_you": int(bool(re.search(r"\b(can you|would you|do you|are you)\b", t.lower()))),
    }

feat = [followup_features(t) for t in texts]
for k in feat[0]:
    vals = [f[k] for f in feat]
    print(f"corr(PC2, {k}):", round(safe_corr(pc2, vals), 4), " mean:", round(float(np.mean(vals)), 3))

# quick inspect extremes
lo = np.argsort(pc2)[:5]
hi = np.argsort(pc2)[-5:]
print("\nLowest-PC2 examples (qmarks, ends_with_q):")
for j in lo:
    f = feat[j]
    print(j, f["qmarks"], f["ends_with_q"], texts[j][:120].replace("\n", " "))
print("\nHighest-PC2 examples (qmarks, ends_with_q):")
for j in hi:
    f = feat[j]
    print(j, f["qmarks"], f["ends_with_q"], texts[j][:120].replace("\n", " "))

# %%
#%%
# Test 3: Disclaimer / refusal-template hypothesis
DISCLAIMER_PATTERNS = [
    r"i can't provide medical advice",
    r"i can't advise you",
    r"i'm not a medical professional",
    r"i'm not a healthcare provider",
    r"not a substitute for professional",
]

def disclaimer_score(t):
    lower = t.lower()
    return sum(int(bool(re.search(p, lower))) for p in DISCLAIMER_PATTERNS)

scores = [disclaimer_score(t) for t in texts]
print("corr(PC2, disclaimer_score):", round(safe_corr(pc2, scores), 4))
print("score distribution:", Counter(scores))

for target_score in sorted(set(scores), reverse=True):
    ex = next((k for k, s in enumerate(scores) if s == target_score), None)
    if ex is not None:
        print(f"\nExample score={target_score}, PC2={pc2[ex]:+.4f}")
        print(texts[ex][:240].replace("\n", " "))


# %%
#%%
# Test 4: Resource/list-format hypothesis (hotlines, links, bullets, numbered steps)
def structure_features(t):
    return {
        "phones": len(re.findall(r"\b\d{3}[- ]?\d{3}[- ]?\d{4}\b", t)),
        "links": len(re.findall(r"https?://|www\.", t.lower())),
        "numbered_items": len(re.findall(r"(?m)^\s*\d+\.\s+", t)),
        "bullets": len(re.findall(r"(?m)^\s*[-*]\s+", t)),
    }

sf = [structure_features(t) for t in texts]
for k in sf[0]:
    vals = [v[k] for v in sf]
    print(f"corr(PC2, {k}):", round(safe_corr(pc2, vals), 4), " mean:", round(float(np.mean(vals)), 3))

# visualize one structural feature
vals = np.array([v["numbered_items"] for v in sf], dtype=float)
plt.figure(figsize=(8, 5))
plt.scatter(vals, pc2, alpha=0.8, s=30)
plt.xlabel("Numbered list items")
plt.ylabel("PC2")
plt.title("PC2 vs numbered-list structure")
plt.grid(alpha=0.25)
plt.show()

# %%
#%%
# Test 5: "Template families" test via top/bottom PC2 n-grams
def ngrams(tokens, n=2):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def tokenize(t):
    return re.findall(r"[a-z']+", t.lower())

k = 10
lo_idx = np.argsort(pc2)[:k]
hi_idx = np.argsort(pc2)[-k:]

def top_ngrams(selected_idx, n=2, topn=20):
    c = Counter()
    for j in selected_idx:
        toks = tokenize(texts[j])
        c.update(ngrams(toks, n=n))
    return c.most_common(topn)

print("Top bigrams in LOW-PC2 slice:")
for ng, ct in top_ngrams(lo_idx, n=2, topn=20):
    print(f"{' '.join(ng):30s} {ct}")

print("\nTop bigrams in HIGH-PC2 slice:")
for ng, ct in top_ngrams(hi_idx, n=2, topn=20):
    print(f"{' '.join(ng):30s} {ct}")

# %%
#%%
# Test 6: Cross-group mixing check for your two groups, split by PC2 quantiles
# If true "style axis", nearest neighbors should often come from the other group but same style quantile.
OTHER_GROUP = "sample_826c163c2ed2166704381efb"
idx_a = np.where(group_ids == TARGET_GROUP)[0]
idx_b = np.where(group_ids == OTHER_GROUP)[0]

if len(idx_b) == 0:
    print(f"No rows found for {OTHER_GROUP}")
else:
    pc2_a = pcs[idx_a, 1]
    q_lo = np.quantile(pc2_a, 0.25)
    q_hi = np.quantile(pc2_a, 0.75)

    a_lo = idx_a[pc2_a <= q_lo]
    a_hi = idx_a[pc2_a >= q_hi]

    def cosine_nn_hits(src_idx, pool_idx, topk=5):
        A = embeddings[src_idx]
        B = embeddings[pool_idx]
        A = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
        B = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
        sims = A @ B.T
        hits = []
        for r in range(sims.shape[0]):
            best = np.argsort(-sims[r])[:topk]
            hits.append(pool_idx[best])
        return hits

    # nearest in OTHER group from low/high TARGET slices
    hits_lo = cosine_nn_hits(a_lo, idx_b, topk=5)
    hits_hi = cosine_nn_hits(a_hi, idx_b, topk=5)

    print(f"TARGET low-PC2 count: {len(a_lo)}, high-PC2 count: {len(a_hi)}")
    print("\nSample neighbor texts for TARGET low-PC2 -> OTHER:")
    for arr in hits_lo[:3]:
        j = arr[0]
        print("-", metadata[j]['assistant_text'][:180].replace("\n", " "))

    print("\nSample neighbor texts for TARGET high-PC2 -> OTHER:")
    for arr in hits_hi[:3]:
        j = arr[0]
        print("-", metadata[j]['assistant_text'][:180].replace("\n", " "))

# %%
