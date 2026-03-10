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
pcs = pca.fit_transform(embeddings)

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
resid_pcs = resid_pca.fit_transform(residual_embeddings)

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



# %%
# For each of the top N eigenvalues, print the question / prompt pairs for for the top K and bottom K scoring responses
top_n_eigenvalues = 10
top_k_responses = 5

for i in range(top_n_eigenvalues):
    # Sort by PC score
    sorted_indices = np.argsort(resid_pcs[:, i])
    top_k_indices = sorted_indices[:top_k_responses]
    bottom_k_indices = sorted_indices[-top_k_responses:]

    top_k_assistant_texts = [(metadata[idx].get("seed_user_message", ""), metadata[idx].get("assistant_text", "")) for idx in top_k_indices]
    bottom_k_assistant_texts = [(metadata[idx].get("seed_user_message", ""), metadata[idx].get("assistant_text", "")) for idx in bottom_k_indices]

    print(f"Top {top_k_responses} responses for eigenvalue {i}:")
    print("-" * 100)
    for j, (question, response) in enumerate(top_k_assistant_texts):
        print(f"\t\t\t\t\t{question}")
        print("-" * 100)
        print(f"{response[:300]}...")
        print("-" * 100)
    print("-" * 100)
    print("-" * 100)
    print(f"Bottom {top_k_responses} responses for eigenvalue {i}:")
    print("-" * 100)
    for j, (question, response) in enumerate(bottom_k_assistant_texts):
        print(f"\t\t\t\t\t{question}")
        print("-" * 100)
        print(f"{response[:300]}...")
        print("-" * 100)
    print("-" * 100)
    print("-" * 100)
    print("-" * 100)
    print("-" * 100)
    print("-" * 100)
    print("\n")


# %%
# Plot eigenvalues by component, marking the elbow via maximum perpendicular distance.
def find_elbow(values: np.ndarray) -> int:
    """Return the 0-based index of the elbow point.

    Normalises the curve to [0,1]x[0,1], then finds the point with the largest
    perpendicular distance from the line connecting the first and last points.
    """
    x = np.arange(len(values), dtype=float)
    y = values.astype(float)

    # Normalise both axes to [0, 1] so the geometry isn't dominated by scale.
    x_norm = (x - x[0]) / (x[-1] - x[0])
    y_norm = (y - y.min()) / (y.max() - y.min())

    # Direction vector of the line from first to last point.
    dx, dy = x_norm[-1] - x_norm[0], y_norm[-1] - y_norm[0]
    line_len = np.hypot(dx, dy)

    # Perpendicular distance from each point to that line.
    distances = np.abs(dy * x_norm - dx * y_norm + dx * y_norm[0] - dy * x_norm[0]) / line_len

    return int(np.argmax(distances))


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, eigenvalues, title in [
    (axes[0], pca.explained_variance_, "Full embeddings PCA"),
    (axes[1], resid_pca.explained_variance_, "Residual embeddings PCA"),
]:
    components = np.arange(1, len(eigenvalues) + 1)
    elbow_idx = find_elbow(eigenvalues)
    elbow_component = components[elbow_idx]

    ax.plot(components, eigenvalues, marker="o", markersize=4, label="Eigenvalue")
    ax.axvline(elbow_component, color="red", linestyle="--", linewidth=1.2,
               label=f"Elbow at component {elbow_component}")
    ax.plot(elbow_component, eigenvalues[elbow_idx], marker="*", markersize=14,
            color="red", zorder=5)
    ax.set_title(title)
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue")
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle("Eigenvalues by component (elbow = suggested cutoff)", fontsize=12)
plt.tight_layout()
plt.show()
