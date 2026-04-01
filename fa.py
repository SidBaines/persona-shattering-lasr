#%%
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#%%
ANALYSIS_BLOCKS = ("fc", "likert")
PERSONA_SUBSET_SAMPLE_IDS: tuple[str, ...] | None = None
PERSONA_SUBSET_GROUP_IDS: tuple[str, ...] | None = None
PERSONA_SUBSET_ROW_INDICES: tuple[int, ...] | None = None
PERSONA_SUBSET_MAX_ROWS: int | None = None

if 1: # Use ~1.5k v2 questionnaire
    QUESTIONNAIRE_RUN_ID = (
        "questionnaire-rollouts-llama318binstruct-t1.0-10t-300p-seed425-archetypes_v7-q_v2-fc+likert"
    )
elif 0: # USE_Q_V3_DATASET
    QUESTIONNAIRE_RUN_ID = (
        "questionnaire-rollouts-llama318binstruct-t1.0-10t-200p-seed421-uprompt_v3-q_v3-hybrid"
    )
else:
    QUESTIONNAIRE_RUN_ID = (
        "questionnaire-rollouts-llama318binstruct-t1.0-10t-200p-seed421-uprompt_v3-q_v2-hybrid"
    )

QUESTIONNAIRE_DIR = Path("scratch/psychometric_fa") / QUESTIONNAIRE_RUN_ID / "questionnaire"

response_matrix_path = QUESTIONNAIRE_DIR / "response_matrix.npy"
metadata_path = QUESTIONNAIRE_DIR / "metadata.jsonl"
items_path = QUESTIONNAIRE_DIR / "items.json"

questionnaire_embeddings = np.load(response_matrix_path).astype(np.float64)

with metadata_path.open("r", encoding="utf-8") as f:
    questionnaire_metadata = [json.loads(line) for line in f if line.strip()]

with items_path.open("r", encoding="utf-8") as f:
    questionnaire_items = json.load(f)


def _subset_persona_rows(
    response_matrix: np.ndarray,
    metadata: list[dict],
    *,
    sample_ids: tuple[str, ...] | None = None,
    group_ids: tuple[str, ...] | None = None,
    row_indices: tuple[int, ...] | None = None,
    max_rows: int | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Filter persona rows while preserving matrix/metadata alignment."""
    if response_matrix.shape[0] != len(metadata):
        raise ValueError(
            f"Metadata rows ({len(metadata)}) != response matrix rows ({response_matrix.shape[0]})"
        )

    keep_mask = np.ones(response_matrix.shape[0], dtype=bool)

    if sample_ids is not None:
        requested_sample_ids = tuple(str(sample_id) for sample_id in sample_ids)
        requested_sample_id_set = set(requested_sample_ids)
        sample_id_to_index = {
            str(row["sample_id"]): idx for idx, row in enumerate(metadata)
        }
        missing_sample_ids = [
            sample_id for sample_id in requested_sample_ids
            if sample_id not in sample_id_to_index
        ]
        if missing_sample_ids:
            raise ValueError(f"Unknown sample_ids requested: {missing_sample_ids}")
        sample_mask = np.array(
            [str(row["sample_id"]) in requested_sample_id_set for row in metadata],
            dtype=bool,
        )
        keep_mask &= sample_mask

    if group_ids is not None:
        requested_group_ids = {str(group_id) for group_id in group_ids}
        available_group_ids = {
            str(row.get("input_group_id", row["sample_id"])) for row in metadata
        }
        missing_group_ids = sorted(requested_group_ids - available_group_ids)
        if missing_group_ids:
            raise ValueError(f"Unknown input_group_ids requested: {missing_group_ids}")
        group_mask = np.array(
            [
                str(row.get("input_group_id", row["sample_id"])) in requested_group_ids
                for row in metadata
            ],
            dtype=bool,
        )
        keep_mask &= group_mask

    if row_indices is not None:
        requested_row_indices = tuple(int(idx) for idx in row_indices)
        invalid_indices = [
            idx for idx in requested_row_indices
            if idx < 0 or idx >= response_matrix.shape[0]
        ]
        if invalid_indices:
            raise IndexError(f"Requested row indices out of bounds: {invalid_indices}")
        row_index_mask = np.zeros(response_matrix.shape[0], dtype=bool)
        row_index_mask[list(requested_row_indices)] = True
        keep_mask &= row_index_mask

    keep_indices = np.flatnonzero(keep_mask)
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError(f"PERSONA_SUBSET_MAX_ROWS must be positive, got {max_rows}")
        keep_indices = keep_indices[:max_rows]

    if len(keep_indices) == 0:
        raise ValueError("Persona subset removed all rows. Adjust the subset filters.")

    filtered_matrix = response_matrix[keep_indices]
    filtered_metadata = [metadata[idx] for idx in keep_indices]
    return filtered_matrix, filtered_metadata


def _make_item_label(row: dict) -> str:
    col_id = str(row.get("col_id", row.get("item_id", "")))
    return f"Q{col_id}" if col_id.isdigit() else col_id

questionnaire_embeddings, questionnaire_metadata = _subset_persona_rows(
    questionnaire_embeddings,
    questionnaire_metadata,
    sample_ids=PERSONA_SUBSET_SAMPLE_IDS,
    group_ids=PERSONA_SUBSET_GROUP_IDS,
    row_indices=PERSONA_SUBSET_ROW_INDICES,
    max_rows=PERSONA_SUBSET_MAX_ROWS,
)

sample_ids = np.array([row["sample_id"] for row in questionnaire_metadata], dtype=object)
all_item_ids = np.array([row["item_id"] for row in questionnaire_items], dtype=object)
all_item_labels = np.array([_make_item_label(row) for row in questionnaire_items], dtype=object)
all_item_texts = np.array([row["text"] for row in questionnaire_items], dtype=object)
all_item_blocks = np.array([row.get("block", "unknown") for row in questionnaire_items], dtype=object)

analysis_item_mask = np.array([block in ANALYSIS_BLOCKS for block in all_item_blocks], dtype=bool)
if not analysis_item_mask.any():
    raise ValueError(
        f"No questionnaire columns matched ANALYSIS_BLOCKS={ANALYSIS_BLOCKS}. "
        "Check the dataset format and selected blocks."
    )

analysis_matrix = questionnaire_embeddings[:, analysis_item_mask]
analysis_questionnaire_items = [
    row for row, keep in zip(questionnaire_items, analysis_item_mask) if keep
]
item_ids = all_item_ids[analysis_item_mask]
item_labels = all_item_labels[analysis_item_mask]
item_texts = all_item_texts[analysis_item_mask]
item_blocks = all_item_blocks[analysis_item_mask]

print(f"Loaded questionnaire matrix: {questionnaire_embeddings.shape}")
print(f"Loaded metadata rows: {len(questionnaire_metadata)}")
print(f"Loaded item definitions: {len(questionnaire_items)}")
print(
    "Persona subset config:",
    {
        "sample_ids": PERSONA_SUBSET_SAMPLE_IDS,
        "group_ids": PERSONA_SUBSET_GROUP_IDS,
        "row_indices": PERSONA_SUBSET_ROW_INDICES,
        "max_rows": PERSONA_SUBSET_MAX_ROWS,
    },
)
print(f"Selected analysis blocks: {ANALYSIS_BLOCKS}")
print(f"Analysis matrix shape after block filter: {analysis_matrix.shape}")
print(
    "Retained items by block:",
    dict(pd.Series(item_blocks).value_counts().sort_index()),
)


#%%
sns.set_theme(style="whitegrid", context="notebook")

X = analysis_matrix.astype(np.float64)
ITEM_CORR_TRIANGLE = "lower"  # "lower", "upper", or "full"
VIS_MIN_ITEM_VARIANCE_FOR_CORR = 1e-12

row_means = X.mean(axis=1)
row_stds = X.std(axis=1)
col_means = X.mean(axis=0)
col_stds = X.std(axis=0)

response_values, response_counts = np.unique(X, return_counts=True)
response_frequencies = response_counts / response_counts.sum()

Xz = StandardScaler().fit_transform(X)
pca = PCA(n_components=6, random_state=0)
persona_coords = pca.fit_transform(Xz)
explained = pca.explained_variance_ratio_

row_linkage = linkage(Xz, method="average", metric="euclidean")
row_order = leaves_list(row_linkage)

vis_corr_item_mask = X.var(axis=0, ddof=1) > VIS_MIN_ITEM_VARIANCE_FOR_CORR
vis_corr_labels = item_labels[vis_corr_item_mask]
X_for_item_corr = X[:, vis_corr_item_mask]

item_corr = np.corrcoef(X_for_item_corr, rowvar=False)
item_corr = np.nan_to_num(item_corr, nan=0.0)
item_corr = (item_corr + item_corr.T) / 2.0
np.fill_diagonal(item_corr, 1.0)
item_dist = np.clip(1.0 - item_corr, 0.0, None)
col_linkage = linkage(squareform(item_dist, checks=False), method="average")
col_order = leaves_list(col_linkage)

heatmap_matrix = X[row_order][:, vis_corr_item_mask][:, col_order]
heatmap_vmin = float(np.min(X))
heatmap_vmax = float(np.max(X))
heatmap_center = 0.0 if heatmap_vmin < 0 else 3.0

fig, axes = plt.subplots(2, 2, figsize=(16, 11))

axes[0, 0].bar(response_values, response_frequencies, color="#4C78A8", width=0.8)
axes[0, 0].set_title("Global Distribution of Questionnaire Responses")
axes[0, 0].set_xlabel("Raw item response value")
axes[0, 0].set_ylabel("Share of all matrix cells")
axes[0, 0].set_xticks(response_values)

axes[0, 1].hist(col_means, bins=20, color="#F58518", edgecolor="white")
axes[0, 1].axvline(col_means.mean(), color="black", linestyle="--", linewidth=1.5)
axes[0, 1].set_title("Distribution of Item Means Across Personas")
axes[0, 1].set_xlabel("Mean response for an item")
axes[0, 1].set_ylabel("Number of items")

axes[1, 0].hist(col_stds, bins=20, color="#54A24B", edgecolor="white")
axes[1, 0].axvline(col_stds.mean(), color="black", linestyle="--", linewidth=1.5)
axes[1, 0].set_title("Distribution of Item Standard Deviations")
axes[1, 0].set_xlabel("Standard deviation across personas")
axes[1, 0].set_ylabel("Number of items")

scatter = axes[1, 1].scatter(
    row_means,
    row_stds,
    c=persona_coords[:, 0],
    cmap="coolwarm",
    alpha=0.8,
    edgecolors="none",
)
axes[1, 1].set_title("Persona-Level Mean vs Within-Persona Variation")
axes[1, 1].set_xlabel("Mean response across all items")
axes[1, 1].set_ylabel("Standard deviation across all items")
fig.colorbar(scatter, ax=axes[1, 1], label="PCA component 1 score")

fig.suptitle("Questionnaire Matrix: Marginals and Heterogeneity", fontsize=16, y=1.02)
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(16, 10))
sns.heatmap(
    heatmap_matrix,
    ax=ax,
    cmap="vlag",
    center=heatmap_center,
    vmin=heatmap_vmin,
    vmax=heatmap_vmax,
    xticklabels=False,
    yticklabels=False,
    cbar_kws={"label": "Raw item response value"},
)
ax.set_title("Clustered Questionnaire Response Matrix (Selected Blocks Only)")
ax.set_xlabel("Variable items reordered by correlation similarity")
ax.set_ylabel("Personas reordered by response-profile similarity")
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(12, 10))
ordered_item_corr = item_corr[np.ix_(col_order, col_order)]
triangle_mask = None
if ITEM_CORR_TRIANGLE == "lower":
    triangle_mask = np.triu(np.ones_like(ordered_item_corr, dtype=bool), k=1)
elif ITEM_CORR_TRIANGLE == "upper":
    triangle_mask = np.tril(np.ones_like(ordered_item_corr, dtype=bool), k=-1)

sns.heatmap(
    ordered_item_corr,
    ax=ax,
    mask=triangle_mask,
    cmap="vlag",
    center=0.0,
    vmin=-1.0,
    vmax=1.0,
    xticklabels=False,
    yticklabels=False,
    cbar_kws={"label": "Item-item correlation"},
)
ax.set_title(
    f"Clustered Item Correlation Matrix ({ITEM_CORR_TRIANGLE.title()} Triangle)"
    if ITEM_CORR_TRIANGLE != "full"
    else "Clustered Item Correlation Matrix"
)
ax.set_xlabel("Variable items reordered by correlation similarity")
ax.set_ylabel("Variable items reordered by correlation similarity")
fig.tight_layout()
plt.show()

print(
    "Items excluded from correlation-based visualisations due to zero variance: "
    f"{int((~vis_corr_item_mask).sum())}"
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

pc1 = axes[0].scatter(
    persona_coords[:, 0],
    persona_coords[:, 1],
    c=row_means,
    cmap="viridis",
    alpha=0.85,
    edgecolors="none",
)
axes[0].set_title(
    "Personas in PCA Space, Colored by Overall Mean Response\n"
    f"PC1={explained[0]:.1%} variance, PC2={explained[1]:.1%} variance"
)
axes[0].set_xlabel("PC1 score")
axes[0].set_ylabel("PC2 score")
fig.colorbar(pc1, ax=axes[0], label="Mean response across items")

pc2 = axes[1].scatter(
    persona_coords[:, 0],
    persona_coords[:, 1],
    c=row_stds,
    cmap="magma",
    alpha=0.85,
    edgecolors="none",
)
axes[1].set_title(
    "Same PCA Map, Colored by Within-Persona Response Variation\n"
    f"Cumulative variance explained by first 4 PCs = {explained[:4].sum():.1%}"
)
axes[1].set_xlabel("PC1 score")
axes[1].set_ylabel("PC2 score")
fig.colorbar(pc2, ax=axes[1], label="Response standard deviation across items")

fig.tight_layout()
plt.show()

print("Most variable items in the raw response matrix:")
for idx in np.argsort(col_stds)[::-1][:10]:
    print(
        f"  {item_labels[idx]} | sd={col_stds[idx]:.3f} | mean={col_means[idx]:.3f} | "
        f"{item_texts[idx]}"
    )

for component_idx in range(2):
    component = pca.components_[component_idx]
    pos_idx = np.argsort(component)[-5:][::-1]
    neg_idx = np.argsort(component)[:5]
    print()
    print(
        f"Items with the strongest loadings on PCA component {component_idx + 1} "
        f"(variance explained = {explained[component_idx]:.1%}):"
    )
    print("  Positive direction:")
    for idx in pos_idx:
        print(f"    {item_labels[idx]} | loading={component[idx]:+.3f} | {item_texts[idx]}")
    print("  Negative direction:")
    for idx in neg_idx:
        print(f"    {item_labels[idx]} | loading={component[idx]:+.3f} | {item_texts[idx]}")


#%%
HORN_N_SIMULATIONS = 500
HORN_PERCENTILE = 95
HORN_RANDOM_SEED = 0
MIN_ITEM_VARIANCE_FOR_FA = 0.40

# Classical Horn parallel analysis on the raw persona-by-item response matrix.
# rows = respondents/personas, columns = questionnaire items.
X = analysis_matrix.astype(np.float64)
n_personas, n_items = X.shape

item_variances = X.var(axis=0, ddof=1)
kept_item_mask = item_variances >= MIN_ITEM_VARIANCE_FOR_FA
dropped_item_mask = ~kept_item_mask

fa_input_X = X[:, kept_item_mask]
fa_item_labels = item_labels[kept_item_mask]
fa_item_texts = item_texts[kept_item_mask]
fa_item_ids = item_ids[kept_item_mask]
fa_questionnaire_items = [row for row, keep in zip(analysis_questionnaire_items, kept_item_mask) if keep]
dropped_items_df = pd.DataFrame(
    {
        "item_label": item_labels[dropped_item_mask],
        "item_id": item_ids[dropped_item_mask],
        "block": [row["block"] for row, keep in zip(analysis_questionnaire_items, dropped_item_mask) if keep],
        "variance": item_variances[dropped_item_mask],
        "item_text": item_texts[dropped_item_mask],
    }
).sort_values(["block", "item_label"])

#%%
item_variance_rows = sorted(
    (
        {
            "variance": float(variance),
            "question": str(question),
            "item_label": str(label),
            "item_id": str(item_id),
            "block": str(block),
        }
        for variance, question, label, item_id, block in zip(
            item_variances,
            item_texts,
            item_labels,
            item_ids,
            item_blocks,
            strict=False,
        )
    ),
    key=lambda row: row["variance"],
)

item_variance_jsonl_path = QUESTIONNAIRE_DIR / "item_variances_ranked.jsonl"
with item_variance_jsonl_path.open("w", encoding="utf-8") as f:
    for row in item_variance_rows:
        f.write(json.dumps(row) + "\n")

print(f"Wrote ranked item variances to {item_variance_jsonl_path}")

item_means = fa_input_X.mean(axis=0)
item_stds = fa_input_X.std(axis=0, ddof=1)

Z = (fa_input_X - item_means) / item_stds
observed_item_correlation_matrix = np.corrcoef(Z, rowvar=False)
observed_item_correlation_matrix = (
    observed_item_correlation_matrix + observed_item_correlation_matrix.T
) / 2.0
np.fill_diagonal(observed_item_correlation_matrix, 1.0)

observed_eigenvalues = np.linalg.eigvalsh(observed_item_correlation_matrix)[::-1]
fa_n_items = fa_input_X.shape[1]

rng = np.random.default_rng(HORN_RANDOM_SEED)
random_eigenvalues = np.zeros((HORN_N_SIMULATIONS, fa_n_items), dtype=np.float64)

for sim_idx in range(HORN_N_SIMULATIONS):
    random_X = rng.standard_normal(size=(n_personas, fa_n_items))
    random_corr = np.corrcoef(random_X, rowvar=False)
    random_corr = (random_corr + random_corr.T) / 2.0
    np.fill_diagonal(random_corr, 1.0)
    random_eigenvalues[sim_idx] = np.linalg.eigvalsh(random_corr)[::-1]

random_mean_eigenvalues = random_eigenvalues.mean(axis=0)
random_percentile_eigenvalues = np.percentile(
    random_eigenvalues,
    HORN_PERCENTILE,
    axis=0,
)
horn_n_factors = int(np.sum(observed_eigenvalues > random_percentile_eigenvalues))

component_numbers = np.arange(1, fa_n_items + 1)

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(component_numbers, observed_eigenvalues, marker="o", linewidth=2, label="Observed eigenvalues")
ax.plot(
    component_numbers,
    random_mean_eigenvalues,
    linestyle="--",
    linewidth=2,
    label="Random-data mean eigenvalues",
)
ax.plot(
    component_numbers,
    random_percentile_eigenvalues,
    linestyle=":",
    linewidth=2.5,
    label=f"Random-data {HORN_PERCENTILE}th percentile eigenvalues",
)
ax.axvline(
    horn_n_factors + 0.5,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=f"Horn cutoff = {horn_n_factors} factors",
)
ax.set_xlim(1, min(30, fa_n_items))
ax.set_xticks(np.arange(1, min(30, fa_n_items) + 1, 2))
ax.set_title("Horn Parallel Analysis on the Raw Item Correlation Matrix")
ax.set_xlabel("Component / factor number")
ax.set_ylabel("Eigenvalue")
ax.legend(loc="upper right")
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(12, 4.5))
horn_gap = observed_eigenvalues - random_percentile_eigenvalues
ax.bar(component_numbers[:30], horn_gap[:30], color=np.where(horn_gap[:30] > 0, "#54A24B", "#E45756"))
ax.axhline(0.0, color="black", linewidth=1)
ax.set_title(f"Horn Parallel Analysis Gap: Observed Minus Random {HORN_PERCENTILE}th Percentile")
ax.set_xlabel("Component / factor number")
ax.set_ylabel("Eigenvalue gap")
fig.tight_layout()
plt.show()

horn_parallel_analysis_df = pd.DataFrame(
    {
        "component": component_numbers,
        "observed_eigenvalue": observed_eigenvalues,
        "random_mean_eigenvalue": random_mean_eigenvalues,
        f"random_p{HORN_PERCENTILE}_eigenvalue": random_percentile_eigenvalues,
        "observed_minus_random_p": horn_gap,
        "retain_under_horn": observed_eigenvalues > random_percentile_eigenvalues,
    }
)

print("Horn parallel analysis on the raw questionnaire item-response matrix")
print(f"  Respondents/personas: {n_personas}")
print(f"  Original items after block filter: {n_items}")
print(f"  Items retained for analysis: {fa_n_items}")
print(
    f"  Dropped low-variance items: {int(dropped_item_mask.sum())} "
    f"(variance < {MIN_ITEM_VARIANCE_FOR_FA})"
)
print(f"  Simulations: {HORN_N_SIMULATIONS}")
print(f"  Retention threshold: observed eigenvalue > random {HORN_PERCENTILE}th percentile")
print(f"  Recommended number of factors: {horn_n_factors}")
display(horn_parallel_analysis_df.head(15).round(3))
if len(dropped_items_df) > 0:
    display(dropped_items_df.round(3))


#%%
N_FACTORS = horn_n_factors  # Override manually if you want to inspect a different solution.
MAX_ITER = 500
TOL = 1e-6
TOP_ITEMS_PER_FACTOR = 8

# Classical psychometrics setup:
# rows = respondents/personas, columns = questionnaire items.
# We factor-analyze the inter-item correlation matrix implied by the raw responses.
X = fa_input_X
n_personas, n_items = X.shape

item_means = X.mean(axis=0)
item_stds = X.std(axis=0, ddof=1)

Z = (X - item_means) / item_stds
item_correlation_matrix = np.corrcoef(Z, rowvar=False)
item_correlation_matrix = (item_correlation_matrix + item_correlation_matrix.T) / 2.0
np.fill_diagonal(item_correlation_matrix, 1.0)

# Initial communalities from squared multiple correlations.
R_inv = np.linalg.pinv(item_correlation_matrix)
smc = 1.0 - (1.0 / np.diag(R_inv))
communalities = np.clip(smc, 1e-6, 1.0)

for iteration in range(1, MAX_ITER + 1):
    reduced_correlation = item_correlation_matrix.copy()
    np.fill_diagonal(reduced_correlation, communalities)

    eigenvalues, eigenvectors = np.linalg.eigh(reduced_correlation)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    kept_eigenvalues = np.clip(eigenvalues[:N_FACTORS], a_min=0.0, a_max=None)
    kept_eigenvectors = eigenvectors[:, :N_FACTORS]
    factor_loadings = kept_eigenvectors * np.sqrt(kept_eigenvalues)

    new_communalities = np.clip(np.sum(factor_loadings ** 2, axis=1), 0.0, 1.0)
    max_change = np.max(np.abs(new_communalities - communalities))
    communalities = new_communalities

    if max_change < TOL:
        break
else:
    raise RuntimeError("Principal axis factoring did not converge within MAX_ITER.")

uniquenesses = 1.0 - communalities
ss_loadings = np.sum(factor_loadings ** 2, axis=0)
proportion_variance = ss_loadings / n_items
cumulative_variance = np.cumsum(proportion_variance)

# Regression-style factor scores for later inspection.
score_weights = (
    np.linalg.pinv(item_correlation_matrix)
    @ factor_loadings
    @ np.linalg.pinv(factor_loadings.T @ np.linalg.pinv(item_correlation_matrix) @ factor_loadings)
)
factor_scores = Z @ score_weights

factor_names = [f"Factor {i + 1}" for i in range(N_FACTORS)]
loadings_df = pd.DataFrame(factor_loadings, index=fa_item_labels, columns=factor_names)
communalities_df = pd.DataFrame(
    {
        "item_label": fa_item_labels,
        "item_id": fa_item_ids,
        "item_text": fa_item_texts,
        "communality": communalities,
        "uniqueness": uniquenesses,
    }
).sort_values("communality", ascending=False)
variance_df = pd.DataFrame(
    {
        "factor": factor_names,
        "ss_loadings": ss_loadings,
        "proportion_variance": proportion_variance,
        "cumulative_variance": cumulative_variance,
    }
)
factor_scores_df = pd.DataFrame(factor_scores, columns=factor_names)

print("Principal axis factoring on the raw questionnaire item-response matrix")
print(f"  Respondents/personas: {n_personas}")
print(f"  Items: {n_items}")
print(f"  Low-variance items excluded before FA: {int(dropped_item_mask.sum())}")
print(f"  Factors extracted: {N_FACTORS}")
print(f"  Iterations to converge: {iteration}")
print(f"  Mean communality: {communalities.mean():.3f}")
print(f"  Total variance explained by extracted factors: {cumulative_variance[-1]:.1%}")
print()
print(variance_df.round(3))

for factor_idx, factor_name in enumerate(factor_names):
    signed_order = np.argsort(np.abs(factor_loadings[:, factor_idx]))[::-1][:TOP_ITEMS_PER_FACTOR]
    print()
    print(f"{factor_name}: strongest absolute loadings")
    for item_idx in signed_order:
        print(
            f"  {fa_item_labels[item_idx]} | loading={factor_loadings[item_idx, factor_idx]:+.3f} | "
            f"h2={communalities[item_idx]:.3f} | {fa_item_texts[item_idx]}"
        )

display(loadings_df.round(3))
display(communalities_df.head(20).round(3))

# %%
if 1:
    import asyncio
    import concurrent.futures

    try:
        from scripts_dev.unsupervised_embeddings.psychometric_rollout_fa import (
            _label_factors_llm,
        )
    except ImportError as exc:
        raise ImportError(
            "Could not import `_label_factors_llm` from "
            "`scripts_dev.unsupervised_embeddings.psychometric_rollout_fa`. "
            "Run this cell in the same environment that script expects."
        ) from exc

    LABELLER_MODEL = "z-ai/glm-4.5-air"
    LABELLER_PROVIDER = "openrouter"
    LABEL_TOP_ITEMS_PER_POLE = TOP_ITEMS_PER_FACTOR
    LABEL_CACHE_GLOBAL = "FA_LABELLED_FACTORS_CACHE"
    ITEM_LABEL_WIDTH = 4
    LOADING_WIDTH = 14
    H2_WIDTH = 10
    BEHAVIOR_WIDTH = 37
    REVERSE_KEY_WIDTH = 13

    def _behavioral_direction(item_row: dict, loading: float) -> str:
        reverse_keyed = bool(item_row.get("reverse_keyed", item_row.get("rev", False)))
        if item_row.get("block") == "likert":
            agree_more = (loading > 0) != reverse_keyed
            direction = "agree more" if agree_more else "disagree more"
            if reverse_keyed:
                return f"{direction} (reverse-keyed)"
            return direction
        return "score higher" if loading > 0 else "score lower"

    def _top_loading_indices(loadings: np.ndarray, factor_idx: int) -> tuple[list[int], list[int]]:
        column = loadings[:, factor_idx]
        order = np.argsort(column)
        top_positive = [idx for idx in order[::-1] if column[idx] > 0][:LABEL_TOP_ITEMS_PER_POLE]
        top_negative = [idx for idx in order if column[idx] < 0][:LABEL_TOP_ITEMS_PER_POLE]
        return top_positive, top_negative

    def _run_shared_factor_labeller(
        loadings: np.ndarray,
        column_defs: list[dict],
        items: list[dict],
    ) -> list[dict]:
        def _call_labeller() -> list[dict]:
            return _label_factors_llm(
                loadings,
                column_defs,
                items,
                top_n=LABEL_TOP_ITEMS_PER_POLE,
                model=LABELLER_MODEL,
                provider_name=LABELLER_PROVIDER,
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return _call_labeller()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_call_labeller).result()

    def _format_loading_report_line(item_idx: int, factor_idx: int) -> str:
        item_row = fa_questionnaire_items[item_idx]
        loading = float(factor_loadings[item_idx, factor_idx])
        direction = _behavioral_direction(item_row, loading)
        reverse_keyed = bool(item_row.get("reverse_keyed", item_row.get("rev", False)))
        reverse_note = "reverse-keyed" if reverse_keyed else ""
        return (
            f"  {fa_item_labels[item_idx]:<{ITEM_LABEL_WIDTH}} | "
            f"{f'loading={loading:+.3f}':<{LOADING_WIDTH}} | "
            f"{f'h2={communalities[item_idx]:.3f}':<{H2_WIDTH}} | "
            f"{f'behavior={direction}':<{BEHAVIOR_WIDTH}} | "
            f"{reverse_note:<{REVERSE_KEY_WIDTH}} | "
            f'{item_row["text"]}'
        )

    if len(fa_questionnaire_items) != factor_loadings.shape[0]:
        raise ValueError(
            "Item metadata is misaligned with factor loadings: "
            f"{len(fa_questionnaire_items)} rows vs {factor_loadings.shape[0]} loadings."
        )

    if N_FACTORS <= 0:
        print("Skipping factor labelling because Horn's analysis recommended zero factors.")
    else:
        labelled_factors = globals().get(LABEL_CACHE_GLOBAL)
        if labelled_factors is None:
            questionnaire_items_for_labeller = [
                {
                    **item_row,
                    "id": item_row.get("id", item_row.get("item_id")),
                }
                for item_row in questionnaire_items
            ]
            labelled_factors = _run_shared_factor_labeller(
                factor_loadings,
                fa_questionnaire_items,
                questionnaire_items_for_labeller,
            )
            globals()[LABEL_CACHE_GLOBAL] = labelled_factors
        else:
            print(f"Reusing cached factor labels from global `{LABEL_CACHE_GLOBAL}`.")

        if not labelled_factors:
            raise RuntimeError(
                "Factor labelling returned no parsed labels."
            )

        labels_by_factor = {
            int(factor_row["factor_index"]): factor_row
            for factor_row in labelled_factors
            if "factor_index" in factor_row
        }

        for factor_idx, factor_name in enumerate(factor_names):
            factor_label = labels_by_factor.get(factor_idx, {})
            summary = factor_label.get("summary", factor_name)
            description = factor_label.get("description", "").strip()
            positive_pole = factor_label.get("positive_pole", "positive loading pole")
            negative_pole = factor_label.get("negative_pole", "negative loading pole")
            dominant_item_types = ", ".join(factor_label.get("dominant_item_types", []))
            top_positive, top_negative = _top_loading_indices(factor_loadings, factor_idx)

            print()
            print("=" * 100)
            print(f"{factor_name}: {summary}")
            print(f"Positive pole: {positive_pole}")
            print(f"Negative pole: {negative_pole}")
            if dominant_item_types:
                print(f"Dominant item types: {dominant_item_types}")
            if description:
                print(description)

            print()
            print("Positive loading items:")
            for item_idx in top_positive:
                print(_format_loading_report_line(item_idx, factor_idx))

            print()
            print("Negative loading items:")
            for item_idx in top_negative:
                print(_format_loading_report_line(item_idx, factor_idx))

        print()
        print("Structured factor labels:")
        display(pd.DataFrame(labelled_factors).sort_values("factor_index").reset_index(drop=True))

# %%
