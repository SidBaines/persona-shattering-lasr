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

from src_dev.datasets import find_consecutive_assistant_turn_sample_ids


#%%
ANALYSIS_BLOCKS = ("fc", "likert")
PERSONA_SUBSET_SAMPLE_IDS: tuple[str, ...] | None = None
PERSONA_SUBSET_GROUP_IDS: tuple[str, ...] | None = None
PERSONA_SUBSET_ROW_INDICES: tuple[int, ...] | None = None
PERSONA_SUBSET_MAX_ROWS: int | None = None
PERSONA_VARIANCE_FILTER_PERCENTILE: float | None = 95.0

if 1: # Match current psychometric_rollout_fa.py production config (SEED=432, 1000p, scenarios v1, Likert "direct" phrasing)
    QUESTIONNAIRE_RUN_ID = (
        "questionnaire-rollouts-llama318binstruct-t1.0-10t-1000p-seed432-scenarios_v1-q_v5-fc+likert-direct"
    )
elif 0: # Use ~1.5k v5 questionnaire
    QUESTIONNAIRE_RUN_ID = (
        "questionnaire-rollouts-llama318binstruct-t1.0-10t-300p-seed425-archetypes_v7-q_v5-fc+likert"
    )
elif 0: # Use ~1.5k v2 questionnaire
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


def _resolve_rollout_dir(questionnaire_dir: Path) -> Path | None:
    """Resolve the source rollout directory for a questionnaire run."""
    config_path = questionnaire_dir.parent / "config.json"
    if not config_path.exists():
        return None

    with config_path.open("r", encoding="utf-8") as f:
        rollout_run_id = json.load(f).get("rollout_run_id")

    if not rollout_run_id:
        return None

    return questionnaire_dir.parent.parent / str(rollout_run_id)


def _exclude_resume_bug_rows(
    response_matrix: np.ndarray,
    metadata: list[dict],
    *,
    questionnaire_dir: Path,
) -> tuple[np.ndarray, list[dict]]:
    """Drop rows whose source rollout contains consecutive assistant turns."""
    if response_matrix.shape[0] != len(metadata):
        raise ValueError(
            f"Metadata rows ({len(metadata)}) != response matrix rows ({response_matrix.shape[0]})"
        )

    rollout_dir = _resolve_rollout_dir(questionnaire_dir)
    if rollout_dir is None:
        print("Resume-bug filter: skipped (could not resolve rollout directory).")
        return response_matrix, metadata

    bad_sample_ids = find_consecutive_assistant_turn_sample_ids(rollout_dir)
    if not bad_sample_ids:
        print("Resume-bug filter: found no samples with consecutive assistant turns.")
        return response_matrix, metadata

    keep_mask = np.array(
        [str(row["sample_id"]) not in bad_sample_ids for row in metadata],
        dtype=bool,
    )
    n_removed = int((~keep_mask).sum())
    if n_removed <= 0:
        print("Resume-bug filter: no questionnaire rows matched the bad sample IDs.")
        return response_matrix, metadata

    filtered_matrix = response_matrix[keep_mask]
    filtered_metadata = [row for row, keep in zip(metadata, keep_mask) if keep]
    print(
        "Resume-bug filter: "
        f"excluded {n_removed} samples with consecutive assistant turns."
    )
    return filtered_matrix, filtered_metadata


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


def _filter_high_variance_persona_rows(
    response_matrix: np.ndarray,
    metadata: list[dict],
    *,
    analysis_item_mask: np.ndarray,
    percentile: float | None,
) -> tuple[np.ndarray, list[dict]]:
    """Drop personas at or above a configurable row-variance percentile."""
    if percentile is None:
        print("High-variance persona filter: disabled.")
        return response_matrix, metadata

    if not (0.0 < float(percentile) < 100.0):
        raise ValueError(
            "PERSONA_VARIANCE_FILTER_PERCENTILE must be between 0 and 100 "
            f"(exclusive), got {percentile}."
        )

    if response_matrix.shape[0] != len(metadata):
        raise ValueError(
            f"Metadata rows ({len(metadata)}) != response matrix rows ({response_matrix.shape[0]})"
        )

    analysis_matrix = response_matrix[:, analysis_item_mask]
    if analysis_matrix.shape[0] <= 1:
        print("High-variance persona filter: skipped (not enough persona rows).")
        return response_matrix, metadata

    row_variances = analysis_matrix.var(axis=1, ddof=1)
    cutoff = float(np.percentile(row_variances, percentile))
    keep_mask = row_variances < cutoff
    n_removed = int((~keep_mask).sum())
    if n_removed <= 0:
        print(
            "High-variance persona filter: "
            f"removed 0 personas at percentile {percentile:.1f} "
            f"(cutoff variance={cutoff:.4f})."
        )
        return response_matrix, metadata

    filtered_matrix = response_matrix[keep_mask]
    filtered_metadata = [row for row, keep in zip(metadata, keep_mask) if keep]
    print(
        "High-variance persona filter: "
        f"excluded {n_removed} personas with row variance at or above the "
        f"{percentile:.1f}th percentile (cutoff variance={cutoff:.4f})."
    )
    return filtered_matrix, filtered_metadata


def _make_item_label(row: dict) -> str:
    col_id = str(row.get("col_id", row.get("item_id", "")))
    return f"Q{col_id}" if col_id.isdigit() else col_id

questionnaire_embeddings, questionnaire_metadata = _exclude_resume_bug_rows(
    questionnaire_embeddings,
    questionnaire_metadata,
    questionnaire_dir=QUESTIONNAIRE_DIR,
)

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

questionnaire_embeddings, questionnaire_metadata = _filter_high_variance_persona_rows(
    questionnaire_embeddings,
    questionnaire_metadata,
    analysis_item_mask=analysis_item_mask,
    percentile=PERSONA_VARIANCE_FILTER_PERCENTILE,
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

actual_n_factors = int(factor_loadings.shape[1])
factor_names_for_reporting = [f"Factor {i + 1}" for i in range(actual_n_factors)]
if actual_n_factors != N_FACTORS:
    print(
        "Factor labelling note: "
        f"N_FACTORS={N_FACTORS} but loadings matrix has {actual_n_factors} factors. "
        "Using the loadings matrix shape for reporting."
    )

if actual_n_factors <= 0:
    print("Skipping factor labelling because Horn's analysis recommended zero factors.")
else:
    labelled_factors = globals().get(LABEL_CACHE_GLOBAL)
    cache_is_compatible = labelled_factors is not None and all(
        int(factor_row.get("factor_index", -1)) < actual_n_factors
        for factor_row in labelled_factors
    )
    if labelled_factors is None or not cache_is_compatible:
        if labelled_factors is not None and not cache_is_compatible:
            print(
                f"Discarding cached factor labels from global `{LABEL_CACHE_GLOBAL}` "
                "because they do not match the current factor count."
            )
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
        raise RuntimeError("Factor labelling returned no parsed labels.")

    labels_by_factor = {
        int(factor_row["factor_index"]): factor_row
        for factor_row in labelled_factors
        if "factor_index" in factor_row
    }

    for factor_idx, factor_name in enumerate(factor_names_for_reporting):
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

# %%
import re
from itertools import combinations


QUESTIONNAIRE_ITERATION_DUMP_DIR = Path("scratch/QuestionnaireInterationDump")
TOP_N_VARIANCE_ITEMS_FOR_DUMP = 15
TOP_N_VARIANCE_ITEMS_FOR_PCA = 50
TOP_N_COMPONENTS_FOR_PERSONA_SCORE_DUMP = 3
TOP_K_PERSONAS_PER_SCORE_EXTREME = 3
TOP_ITEMS_PER_PERSONA_SCORE_REPORT = 8
LIKERT_RESPONSE_VALUES = np.arange(1, 6, dtype=int)
REPRESENTATIVE_ITEM_RANDOM_SEED = 0


def _extract_questionnaire_version(run_id: str) -> str | None:
    """Extract the questionnaire version suffix from a run id."""
    match = re.search(r"q_v(\d+)", run_id)
    if match is None:
        return None
    return match.group(1)


def _load_source_questionnaire_items(
    run_id: str,
) -> tuple[dict[str, dict], Path | None]:
    """Load the source questionnaire definition for richer item metadata."""
    version = _extract_questionnaire_version(run_id)
    if version is None:
        return {}, None

    source_path = (
        Path("datasets/psychometric_questionnaires")
        / f"psychometric_questionnaire_v{version}.json"
    )
    if not source_path.exists():
        return {}, None

    with source_path.open("r", encoding="utf-8") as f:
        questionnaire_data = json.load(f)

    source_items = questionnaire_data.get("items", [])
    source_items_by_id = {
        str(item["id"]): item
        for item in source_items
    }
    return source_items_by_id, source_path


def _make_item_metadata_df(
    analysis_items: list[dict],
    *,
    labels: np.ndarray,
    ids: np.ndarray,
    texts: np.ndarray,
    blocks: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    variances: np.ndarray,
    run_id: str,
) -> tuple[pd.DataFrame, Path | None]:
    """Build item metadata, backfilling dimension info from source JSON if needed."""
    source_items_by_id, source_path = _load_source_questionnaire_items(run_id)

    rows: list[dict] = []
    for idx, (item_row, label, item_id, text, block, mean, sd, variance) in enumerate(
        zip(
            analysis_items,
            labels,
            ids,
            texts,
            blocks,
            means,
            stds,
            variances,
            strict=False,
        )
    ):
        source_row = source_items_by_id.get(str(item_id), {})
        dimension = str(source_row.get("dimension", item_row.get("dimension", ""))).strip()
        rows.append(
            {
                "analysis_col_index": idx,
                "item_label": str(label),
                "item_id": str(item_id),
                "block": str(block),
                "dimension": dimension,
                "reverse_keyed": bool(
                    source_row.get(
                        "reverse_keyed",
                        item_row.get("reverse_keyed", False),
                    )
                ),
                "mean": float(mean),
                "sd": float(sd),
                "variance": float(variance),
                "item_text": str(text),
            }
        )

    return pd.DataFrame(rows), source_path


def _compute_correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Compute a stable item-item correlation matrix."""
    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    return corr


def _make_distribution_df(
    matrix: np.ndarray,
    item_metadata_df: pd.DataFrame,
    selected_indices: np.ndarray,
    *,
    group_name: str,
) -> pd.DataFrame:
    """Summarize 1-5 response percentages for a selected item set."""
    rows: list[dict] = []
    n_rows = matrix.shape[0]

    for rank, idx in enumerate(selected_indices, start=1):
        item_row = item_metadata_df.iloc[int(idx)]
        counts = (
            pd.Series(matrix[:, int(idx)])
            .value_counts()
            .reindex(LIKERT_RESPONSE_VALUES, fill_value=0)
            .astype(int)
        )
        row = item_row.to_dict()
        row["variance_group"] = group_name
        row["variance_rank"] = rank
        for response_value in LIKERT_RESPONSE_VALUES:
            count = int(counts.loc[response_value])
            row[f"count_{response_value}"] = count
            row[f"pct_{response_value}"] = 100.0 * count / n_rows
        rows.append(row)

    return pd.DataFrame(rows)


def _plot_distribution_grid(
    distribution_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> None:
    """Render a 15-item grid of response distributions."""
    n_items = len(distribution_df)
    n_cols = 3
    n_rows = int(np.ceil(n_items / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(18, 3.8 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes_array = np.atleast_1d(axes).reshape(n_rows, n_cols)

    for ax, row in zip(axes_array.flat, distribution_df.itertuples(index=False), strict=False):
        percentages = [getattr(row, f"pct_{value}") for value in LIKERT_RESPONSE_VALUES]
        ax.bar(LIKERT_RESPONSE_VALUES, percentages, color="#4C78A8", width=0.8)
        ax.set_ylim(0, 100)
        ax.set_xticks(LIKERT_RESPONSE_VALUES)
        dimension = row.dimension if row.dimension else "unassigned"
        reverse_note = " | reverse" if row.reverse_keyed else ""
        ax.set_title(
            f"{row.item_label} ({dimension}{reverse_note})\n"
            f"mean={row.mean:.2f}, sd={row.sd:.2f}"
        )
        ax.set_xlabel("Response")
        ax.set_ylabel("Percent")

    for ax in axes_array.flat[n_items:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _iter_jsonl_rows(path: Path) -> list[dict]:
    """Load JSONL rows from disk."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_rollout_records_for_dump(
    questionnaire_dir: Path,
) -> tuple[dict[str, dict], dict[str, str], Path | None]:
    """Load rollout transcripts and archetype assignments for example exports."""
    run_root = questionnaire_dir.parent
    config_path = run_root / "config.json"
    rollout_run_id = None
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            rollout_run_id = json.load(f).get("rollout_run_id")

    if not rollout_run_id:
        return {}, {}, None

    rollout_root = questionnaire_dir.parent.parent / rollout_run_id
    candidate_paths = [
        rollout_root / "exports" / "conversation_training.jsonl",
        rollout_root / "datasets" / "canonical_samples.jsonl",
    ]
    transcript_path = next((path for path in candidate_paths if path.exists()), None)
    if transcript_path is None:
        return {}, {}, None

    rollout_records = {
        str(row["sample_id"]): row
        for row in _iter_jsonl_rows(transcript_path)
        if "sample_id" in row
    }

    archetype_path = rollout_root / "archetype_assignments.json"
    archetypes: dict[str, str] = {}
    if archetype_path.exists():
        with archetype_path.open("r", encoding="utf-8") as f:
            archetypes = {
                str(sample_id): str(archetype)
                for sample_id, archetype in json.load(f).items()
            }

    return rollout_records, archetypes, transcript_path


def _pick_closest_unused_index(
    values: np.ndarray,
    target: float,
    used_indices: set[int],
) -> int:
    """Pick the unused item index closest to a target statistic."""
    order = np.argsort(np.abs(values - target))
    for idx in order:
        idx_int = int(idx)
        if idx_int not in used_indices:
            return idx_int
    raise RuntimeError("No unused item indices available for representative selection.")


def _select_representative_item_specs(
    item_metadata_df: pd.DataFrame,
) -> list[dict]:
    """Select representative items spanning mean/variance regimes plus random extras."""
    rng = np.random.default_rng(REPRESENTATIVE_ITEM_RANDOM_SEED)
    used_indices: set[int] = set()
    selected_specs: list[dict] = []

    def _add_quantile_based_specs(metric_name: str) -> None:
        metric_values = item_metadata_df[metric_name].to_numpy(dtype=np.float64)
        for label, quantile in (("low", 0.10), ("medium", 0.50), ("high", 0.90)):
            target = float(np.quantile(metric_values, quantile))
            item_idx = _pick_closest_unused_index(metric_values, target, used_indices)
            used_indices.add(item_idx)
            selected_specs.append(
                {
                    "selection_group": metric_name,
                    "selection_bucket": label,
                    "selection_target_quantile": quantile,
                    "selection_target_value": target,
                    "analysis_col_index": item_idx,
                }
            )

    _add_quantile_based_specs("variance")
    _add_quantile_based_specs("mean")

    remaining_indices = np.array(
        [idx for idx in item_metadata_df["analysis_col_index"] if idx not in used_indices],
        dtype=int,
    )
    random_indices = rng.choice(remaining_indices, size=3, replace=False)
    for random_rank, item_idx in enumerate(random_indices, start=1):
        item_idx_int = int(item_idx)
        used_indices.add(item_idx_int)
        selected_specs.append(
            {
                "selection_group": "random",
                "selection_bucket": f"random_{random_rank}",
                "selection_target_quantile": np.nan,
                "selection_target_value": np.nan,
                "analysis_col_index": item_idx_int,
            }
        )

    return selected_specs


def _pick_example_sample_indices(
    item_responses: np.ndarray,
    available_sample_mask: np.ndarray,
) -> list[dict]:
    """Pick low/mid/high-response example rollouts for a selected item."""
    candidate_indices = np.flatnonzero(available_sample_mask)
    if len(candidate_indices) == 0:
        return []

    candidate_responses = item_responses[candidate_indices]
    targets = [
        ("low_response", float(np.min(candidate_responses))),
        ("mid_response", float(np.median(candidate_responses))),
        ("high_response", float(np.max(candidate_responses))),
    ]

    selected_rows: list[dict] = []
    used_sample_indices: set[int] = set()
    for example_label, target in targets:
        order = candidate_indices[np.argsort(np.abs(candidate_responses - target))]
        chosen_idx = None
        for candidate_idx in order:
            candidate_idx_int = int(candidate_idx)
            if candidate_idx_int not in used_sample_indices:
                chosen_idx = candidate_idx_int
                break
        if chosen_idx is None:
            chosen_idx = int(order[0])
        used_sample_indices.add(chosen_idx)
        selected_rows.append(
            {
                "example_label": example_label,
                "row_index": chosen_idx,
                "response_value": float(item_responses[chosen_idx]),
            }
        )

    deduped_rows: list[dict] = []
    seen_row_indices: set[int] = set()
    for row in selected_rows:
        row_index = int(row["row_index"])
        if row_index in seen_row_indices:
            continue
        seen_row_indices.add(row_index)
        deduped_rows.append(row)
    return deduped_rows


def _format_messages_for_markdown(messages: list[dict]) -> str:
    """Render a conversation transcript as markdown."""
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "unknown")).upper()
        content = str(message.get("content", "")).strip()
        lines.append(f"**{role}**")
        lines.append("")
        lines.append(content if content else "(empty)")
        lines.append("")
    return "\n".join(lines).strip()


def _zscore(values: np.ndarray) -> np.ndarray:
    """Compute a stable z-score vector."""
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if std <= 0:
        return np.zeros_like(values, dtype=np.float64)
    return (values - mean) / std


def _pick_extreme_score_indices(
    scores: np.ndarray,
    available_sample_mask: np.ndarray,
    *,
    top_k: int,
) -> tuple[list[int], list[int]]:
    """Pick top-k high-score and low-score row indices among available rollouts."""
    candidate_indices = np.flatnonzero(available_sample_mask)
    if len(candidate_indices) == 0:
        return [], []

    candidate_scores = scores[candidate_indices]
    high_order = candidate_indices[np.argsort(candidate_scores)[::-1]]
    low_order = candidate_indices[np.argsort(candidate_scores)]
    n_keep = min(top_k, len(candidate_indices))
    high_indices = [int(idx) for idx in high_order[:n_keep]]
    low_indices = [int(idx) for idx in low_order[:n_keep]]
    return high_indices, low_indices


def _top_signed_weight_indices(weights: np.ndarray, top_n: int) -> tuple[list[int], list[int]]:
    """Return top positive and negative loading/weight indices."""
    order = np.argsort(weights)
    top_positive = [int(idx) for idx in order[::-1] if weights[idx] > 0][:top_n]
    top_negative = [int(idx) for idx in order if weights[idx] < 0][:top_n]
    return top_positive, top_negative


def _format_weight_report_line(
    item_row: pd.Series,
    weight: float,
    *,
    weight_label: str,
) -> str:
    """Format a PCA/factor item-weight line for markdown export."""
    reverse_note = " | reverse-keyed" if bool(item_row["reverse_keyed"]) else ""
    dimension = item_row["dimension"] if item_row["dimension"] else "unassigned"
    return (
        f"- `{item_row['item_label']}` | {weight_label}=`{weight:+.3f}` | "
        f"dimension=`{dimension}`{reverse_note} | {item_row['item_text']}"
    )


QUESTIONNAIRE_ITERATION_DUMP_DIR.mkdir(parents=True, exist_ok=True)

dump_X = analysis_matrix.astype(np.float64)
dump_item_variances = dump_X.var(axis=0, ddof=1)
dump_item_stds = dump_X.std(axis=0, ddof=1)
dump_item_means = dump_X.mean(axis=0)
dump_item_metadata_df, source_questionnaire_path = _make_item_metadata_df(
    analysis_questionnaire_items,
    labels=item_labels,
    ids=item_ids,
    texts=item_texts,
    blocks=item_blocks,
    means=dump_item_means,
    stds=dump_item_stds,
    variances=dump_item_variances,
    run_id=QUESTIONNAIRE_RUN_ID,
)

item_summary_path = QUESTIONNAIRE_ITERATION_DUMP_DIR / "item_mean_sd_summary.csv"
dump_item_metadata_df.sort_values(
    ["variance", "item_label"],
    ascending=[False, True],
).to_csv(item_summary_path, index=False)

sorted_variance_indices = np.argsort(dump_item_variances)
bottom_variance_indices = sorted_variance_indices[:TOP_N_VARIANCE_ITEMS_FOR_DUMP]
top_variance_indices = sorted_variance_indices[::-1][:TOP_N_VARIANCE_ITEMS_FOR_DUMP]

top_distribution_df = _make_distribution_df(
    dump_X,
    dump_item_metadata_df,
    top_variance_indices,
    group_name="top_15_variance",
)
bottom_distribution_df = _make_distribution_df(
    dump_X,
    dump_item_metadata_df,
    bottom_variance_indices,
    group_name="bottom_15_variance",
)

top_distribution_csv_path = (
    QUESTIONNAIRE_ITERATION_DUMP_DIR / "top_15_variance_response_distribution.csv"
)
bottom_distribution_csv_path = (
    QUESTIONNAIRE_ITERATION_DUMP_DIR / "bottom_15_variance_response_distribution.csv"
)
top_distribution_df.to_csv(top_distribution_csv_path, index=False)
bottom_distribution_df.to_csv(bottom_distribution_csv_path, index=False)

top_distribution_plot_path = (
    QUESTIONNAIRE_ITERATION_DUMP_DIR / "top_15_variance_response_distribution.png"
)
bottom_distribution_plot_path = (
    QUESTIONNAIRE_ITERATION_DUMP_DIR / "bottom_15_variance_response_distribution.png"
)
_plot_distribution_grid(
    top_distribution_df,
    top_distribution_plot_path,
    title="Top 15 Items by Variance: 1-5 Response Distributions",
)
_plot_distribution_grid(
    bottom_distribution_df,
    bottom_distribution_plot_path,
    title="Bottom 15 Items by Variance: 1-5 Response Distributions",
)

raw_item_corr = _compute_correlation_matrix(dump_X)
raw_corr_df = pd.DataFrame(raw_item_corr, index=item_labels, columns=item_labels)
raw_corr_path = QUESTIONNAIRE_ITERATION_DUMP_DIR / "item_item_correlation_matrix_raw.csv"
raw_corr_df.to_csv(raw_corr_path, index=True)

reverse_mask = dump_item_metadata_df["reverse_keyed"].to_numpy(dtype=bool)
aligned_X = dump_X.copy()
aligned_X[:, reverse_mask] = 6.0 - aligned_X[:, reverse_mask]
aligned_item_corr = _compute_correlation_matrix(aligned_X)
aligned_corr_df = pd.DataFrame(
    aligned_item_corr,
    index=item_labels,
    columns=item_labels,
)
aligned_corr_path = (
    QUESTIONNAIRE_ITERATION_DUMP_DIR / "item_item_correlation_matrix_key_aligned.csv"
)
aligned_corr_df.to_csv(aligned_corr_path, index=True)

within_dimension_pair_rows: list[dict] = []
within_dimension_summary_rows: list[dict] = []
for dimension, dimension_df in dump_item_metadata_df.groupby("dimension", dropna=False):
    dimension_label = dimension if str(dimension).strip() else "unassigned"
    dimension_indices = dimension_df["analysis_col_index"].tolist()
    if len(dimension_indices) < 2:
        continue

    raw_pair_corrs: list[float] = []
    aligned_pair_corrs: list[float] = []
    same_key_pair_corrs: list[float] = []
    opposite_key_pair_corrs: list[float] = []

    for idx_a, idx_b in combinations(dimension_indices, 2):
        row_a = dump_item_metadata_df.iloc[idx_a]
        row_b = dump_item_metadata_df.iloc[idx_b]
        raw_corr = float(raw_item_corr[idx_a, idx_b])
        aligned_corr = float(aligned_item_corr[idx_a, idx_b])
        key_relationship = (
            "same_key_direction"
            if row_a["reverse_keyed"] == row_b["reverse_keyed"]
            else "opposite_key_direction"
        )

        within_dimension_pair_rows.append(
            {
                "dimension": dimension_label,
                "item_a_label": row_a["item_label"],
                "item_a_id": row_a["item_id"],
                "item_a_reverse_keyed": bool(row_a["reverse_keyed"]),
                "item_b_label": row_b["item_label"],
                "item_b_id": row_b["item_id"],
                "item_b_reverse_keyed": bool(row_b["reverse_keyed"]),
                "key_relationship": key_relationship,
                "raw_correlation": raw_corr,
                "key_aligned_correlation": aligned_corr,
            }
        )

        raw_pair_corrs.append(raw_corr)
        aligned_pair_corrs.append(aligned_corr)
        if key_relationship == "same_key_direction":
            same_key_pair_corrs.append(raw_corr)
        else:
            opposite_key_pair_corrs.append(raw_corr)

    within_dimension_summary_rows.append(
        {
            "dimension": dimension_label,
            "n_items": len(dimension_indices),
            "n_pairs": len(raw_pair_corrs),
            "mean_raw_correlation": float(np.mean(raw_pair_corrs)),
            "mean_key_aligned_correlation": float(np.mean(aligned_pair_corrs)),
            "mean_same_key_raw_correlation": (
                float(np.mean(same_key_pair_corrs))
                if same_key_pair_corrs
                else np.nan
            ),
            "mean_opposite_key_raw_correlation": (
                float(np.mean(opposite_key_pair_corrs))
                if opposite_key_pair_corrs
                else np.nan
            ),
        }
    )

within_dimension_pairs_df = pd.DataFrame(within_dimension_pair_rows).sort_values(
    ["dimension", "raw_correlation"],
    ascending=[True, False],
)
within_dimension_pairs_path = (
    QUESTIONNAIRE_ITERATION_DUMP_DIR / "within_dimension_item_pair_correlations.csv"
)
within_dimension_pairs_df.to_csv(within_dimension_pairs_path, index=False)

within_dimension_summary_df = pd.DataFrame(within_dimension_summary_rows).sort_values(
    "mean_key_aligned_correlation",
    ascending=False,
)
within_dimension_summary_path = (
    QUESTIONNAIRE_ITERATION_DUMP_DIR / "within_dimension_correlation_summary.csv"
)
within_dimension_summary_df.to_csv(within_dimension_summary_path, index=False)

top_n_for_pca = min(TOP_N_VARIANCE_ITEMS_FOR_PCA, dump_X.shape[1])
top_pca_indices = np.argsort(dump_item_variances)[::-1][:top_n_for_pca]
top_pca_item_df = dump_item_metadata_df.iloc[top_pca_indices].copy()
top_pca_X = dump_X[:, top_pca_indices]
top_pca_Xz = StandardScaler().fit_transform(top_pca_X)
top_pca = PCA(n_components=min(top_pca_Xz.shape), random_state=0)
top_pca_scores = top_pca.fit_transform(top_pca_Xz)
top_pca_eigenvalues = top_pca.explained_variance_
top_pca_explained_ratio = top_pca.explained_variance_ratio_
top_pca_cumulative = np.cumsum(top_pca_explained_ratio)
top_pca_n_gt_one = int(np.sum(top_pca_eigenvalues > 1.0))
top_pca_loading_matrix = top_pca.components_.T * np.sqrt(top_pca_eigenvalues)

top_pca_summary_df = pd.DataFrame(
    {
        "component": np.arange(1, len(top_pca_eigenvalues) + 1),
        "eigenvalue": top_pca_eigenvalues,
        "explained_variance_ratio": top_pca_explained_ratio,
        "cumulative_explained_variance_ratio": top_pca_cumulative,
        "eigenvalue_gt_1": top_pca_eigenvalues > 1.0,
    }
)
top_pca_summary_path = QUESTIONNAIRE_ITERATION_DUMP_DIR / "top_50_variance_pca_summary.csv"
top_pca_summary_df.to_csv(top_pca_summary_path, index=False)

top_pca_items_path = QUESTIONNAIRE_ITERATION_DUMP_DIR / "top_50_variance_items.csv"
top_pca_item_df.sort_values("variance", ascending=False).to_csv(top_pca_items_path, index=False)

fig, ax = plt.subplots(figsize=(12, 7))
component_numbers = top_pca_summary_df["component"].to_numpy()
ax.plot(component_numbers, top_pca_eigenvalues, marker="o", linewidth=2, color="#4C78A8")
ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Kaiser threshold = 1.0")
ax.axvline(
    top_pca_n_gt_one + 0.5,
    color="#E45756",
    linestyle=":",
    linewidth=1.5,
    label=f"{top_pca_n_gt_one} components > 1",
)
ax.set_xlim(1, len(component_numbers))
ax.set_xticks(np.arange(1, len(component_numbers) + 1, 2))
ax.set_title(
    f"PCA Scree Plot on Top {top_n_for_pca} Items by Variance"
)
ax.set_xlabel("Principal component")
ax.set_ylabel("Eigenvalue")
ax.legend(loc="upper right")
fig.tight_layout()
top_pca_plot_path = QUESTIONNAIRE_ITERATION_DUMP_DIR / "top_50_variance_pca_scree.png"
fig.savefig(top_pca_plot_path, dpi=200, bbox_inches="tight")
plt.close(fig)

try:
    import importlib
    import src_dev.factor_analysis.factor_analysis as factor_analysis_module
except ImportError as exc:
    raise RuntimeError(
        "Could not import run_factor_analysis from src_dev.factor_analysis.factor_analysis. "
        "Run this cell from the project environment."
    ) from exc

factor_analysis_module = importlib.reload(factor_analysis_module)
run_factor_analysis = factor_analysis_module.run_factor_analysis

fa_top_item_df = dump_item_metadata_df.iloc[top_pca_indices].copy().reset_index(drop=True)
fa_top_aligned_X = aligned_X[:, top_pca_indices]

rotated_factor_loading_paths: dict[str, str] = {}
rotated_factor_results: dict[str, dict] = {}
for n_factors in (5, 8):
    fa_result = run_factor_analysis(
        fa_top_aligned_X,
        n_factors=n_factors,
        method="principal",
        rotation="oblimin",
    )
    factor_names = [f"factor_{idx + 1}" for idx in range(n_factors)]
    loadings_df = pd.DataFrame(
        fa_result["loadings"],
        columns=factor_names,
    )
    loading_export_df = pd.concat(
        [
            fa_top_item_df,
            pd.DataFrame(
                {
                    "communality": fa_result["communalities"],
                    "ss_loading_contrib": np.sum(fa_result["loadings"] ** 2, axis=1),
                }
            ),
            loadings_df,
        ],
        axis=1,
    )
    loading_export_path = (
        QUESTIONNAIRE_ITERATION_DUMP_DIR
        / f"top_50_variance_oblimin_{n_factors}_factor_loadings.csv"
    )
    loading_export_df.to_csv(loading_export_path, index=False)
    rotated_factor_loading_paths[f"oblimin_{n_factors}_factor_loadings_csv"] = str(
        loading_export_path
    )

    top_loading_rows: list[dict] = []
    for factor_name in factor_names:
        top_factor_df = loading_export_df.assign(
            abs_loading=lambda df: df[factor_name].abs()
        ).sort_values("abs_loading", ascending=False)
        for rank, row in enumerate(top_factor_df.head(12).itertuples(index=False), start=1):
            top_loading_rows.append(
                {
                    "factor": factor_name,
                    "rank": rank,
                    "item_label": row.item_label,
                    "item_id": row.item_id,
                    "dimension": row.dimension,
                    "reverse_keyed": row.reverse_keyed,
                    "loading": getattr(row, factor_name),
                    "abs_loading": row.abs_loading,
                    "item_text": row.item_text,
                }
            )
    top_loading_summary_path = (
        QUESTIONNAIRE_ITERATION_DUMP_DIR
        / f"top_50_variance_oblimin_{n_factors}_factor_top_loadings.csv"
    )
    pd.DataFrame(top_loading_rows).to_csv(top_loading_summary_path, index=False)
    rotated_factor_loading_paths[f"oblimin_{n_factors}_factor_top_loadings_csv"] = str(
        top_loading_summary_path
    )

    factor_correlation_matrix = fa_result.get("factor_correlation_matrix")
    if factor_correlation_matrix is not None:
        factor_corr_df = pd.DataFrame(
            factor_correlation_matrix,
            index=factor_names,
            columns=factor_names,
        )
        factor_corr_path = (
            QUESTIONNAIRE_ITERATION_DUMP_DIR
            / f"top_50_variance_oblimin_{n_factors}_factor_correlation_matrix.csv"
        )
        factor_corr_df.to_csv(factor_corr_path, index=True)
        rotated_factor_loading_paths[
            f"oblimin_{n_factors}_factor_correlation_matrix_csv"
        ] = str(factor_corr_path)

    rotated_factor_results[f"oblimin_{n_factors}"] = {
        "score_names": factor_names,
        "item_df": fa_top_item_df.copy(),
        "score_matrix": fa_result["scores"],
        "weight_matrix": fa_result["loadings"],
        "score_strength": np.asarray(fa_result["ss_loadings"], dtype=np.float64),
        "weight_label": "loading",
        "score_kind": "factor",
    }

rollout_records_by_sample_id, archetypes_by_sample_id, rollout_transcript_path = _load_rollout_records_for_dump(
    QUESTIONNAIRE_DIR
)
sample_ids_by_row = np.array(
    [str(row["sample_id"]) for row in questionnaire_metadata],
    dtype=object,
)
available_rollout_mask = np.array(
    [sample_id in rollout_records_by_sample_id for sample_id in sample_ids_by_row],
    dtype=bool,
)

assistant_word_counts: list[float] = []
assistant_first_person_densities: list[float] = []
for sample_id in sample_ids_by_row:
    record = rollout_records_by_sample_id.get(str(sample_id), {})
    messages = list(record.get("messages", []))
    assistant_texts = [
        str(message.get("content", ""))
        for message in messages
        if str(message.get("role", "")).lower() == "assistant"
    ]
    assistant_word_lengths = np.array(
        [len(text.split()) for text in assistant_texts],
        dtype=np.float64,
    )
    avg_assistant_word_count = (
        float(assistant_word_lengths.mean())
        if len(assistant_word_lengths) > 0
        else np.nan
    )
    combined_assistant_text = "\n".join(assistant_texts).lower()
    assistant_word_total = max(len(combined_assistant_text.split()), 1)
    first_person_count = len(
        re.findall(
            r"\b(i|i'm|i've|i'd|i'll|me|my|mine|myself)\b",
            combined_assistant_text,
        )
    )
    first_person_density = float(first_person_count) / assistant_word_total
    assistant_word_counts.append(avg_assistant_word_count)
    assistant_first_person_densities.append(first_person_density)

assistant_word_counts_array = np.array(assistant_word_counts, dtype=np.float64)
assistant_first_person_densities_array = np.array(
    assistant_first_person_densities,
    dtype=np.float64,
)
neutral_response_counts = (dump_X == 3.0).sum(axis=1).astype(int)
neutral_response_fraction = neutral_response_counts / dump_X.shape[1]

assistant_like_proxy_score = (
    -_zscore(np.nan_to_num(assistant_word_counts_array, nan=np.nanmedian(assistant_word_counts_array)))
    - _zscore(
        np.nan_to_num(
            assistant_first_person_densities_array,
            nan=np.nanmedian(assistant_first_person_densities_array),
        )
    )
)
assistant_like_threshold = float(np.nanmedian(assistant_like_proxy_score))
assistant_like_cluster = np.where(
    assistant_like_proxy_score >= assistant_like_threshold,
    "assistant_like_proxy",
    "non_assistant_like_proxy",
)

response_length_threshold = float(np.nanmedian(assistant_word_counts_array))
response_length_bucket = np.where(
    assistant_word_counts_array <= response_length_threshold,
    "shorter_responses",
    "longer_responses",
)

neutral_response_persona_df = pd.DataFrame(
    {
        "sample_id": sample_ids_by_row,
        "input_group_id": [
            str(row.get("input_group_id", row["sample_id"]))
            for row in questionnaire_metadata
        ],
        "archetype": [
            archetypes_by_sample_id.get(str(sample_id), "unknown")
            for sample_id in sample_ids_by_row
        ],
        "neutral_3_count": neutral_response_counts,
        "neutral_3_fraction": neutral_response_fraction,
        "avg_assistant_response_words": assistant_word_counts_array,
        "assistant_first_person_pronoun_density": assistant_first_person_densities_array,
        "assistant_like_proxy_score": assistant_like_proxy_score,
        "assistant_like_proxy_cluster": assistant_like_cluster,
        "response_length_bucket": response_length_bucket,
    }
)
neutral_response_persona_path = (
    QUESTIONNAIRE_ITERATION_DUMP_DIR / "persona_neutral_3_response_counts.csv"
)
neutral_response_persona_df.to_csv(neutral_response_persona_path, index=False)

fig, ax = plt.subplots(figsize=(12, 7))
bins = np.arange(-0.5, dump_X.shape[1] + 1.5, 1.0)
cluster_hist_data = [
    neutral_response_persona_df.loc[
        neutral_response_persona_df["assistant_like_proxy_cluster"] == cluster_name,
        "neutral_3_count",
    ].to_numpy()
    for cluster_name in ("assistant_like_proxy", "non_assistant_like_proxy")
]
ax.hist(
    cluster_hist_data,
    bins=bins,
    stacked=True,
    label=["assistant_like_proxy", "non_assistant_like_proxy"],
    color=["#4C78A8", "#E45756"],
    alpha=0.85,
)
ax.set_title('Distribution of "3" Responses Per Persona Across All Items')
ax.set_xlabel('Number of items answered "3"')
ax.set_ylabel("Number of personas")
ax.legend()
fig.tight_layout()
neutral_response_hist_path = (
    QUESTIONNAIRE_ITERATION_DUMP_DIR / "persona_neutral_3_response_histogram.png"
)
fig.savefig(neutral_response_hist_path, dpi=200, bbox_inches="tight")
plt.close(fig)

high_variance_focus_labels = [
    label
    for label in ("Q61", "Q92")
    if label in set(dump_item_metadata_df["item_label"])
]
for idx in np.argsort(dump_item_variances)[::-1]:
    label = str(dump_item_metadata_df.iloc[int(idx)]["item_label"])
    if label not in high_variance_focus_labels:
        high_variance_focus_labels.append(label)
    if len(high_variance_focus_labels) >= 4:
        break

high_variance_focus_indices = [
    int(
        dump_item_metadata_df.loc[
            dump_item_metadata_df["item_label"] == label,
            "analysis_col_index",
        ].iloc[0]
    )
    for label in high_variance_focus_labels
]

high_variance_cluster_rows: list[dict] = []
for item_idx in high_variance_focus_indices:
    item_row = dump_item_metadata_df.iloc[item_idx]
    item_responses = dump_X[:, item_idx]
    for cluster_name in ("assistant_like_proxy", "non_assistant_like_proxy"):
        cluster_mask = assistant_like_cluster == cluster_name
        cluster_values = item_responses[cluster_mask]
        counts = (
            pd.Series(cluster_values)
            .value_counts()
            .reindex(LIKERT_RESPONSE_VALUES, fill_value=0)
            .astype(int)
        )
        row = {
            "item_label": item_row["item_label"],
            "item_id": item_row["item_id"],
            "dimension": item_row["dimension"],
            "variance": item_row["variance"],
            "mean": item_row["mean"],
            "assistant_like_proxy_cluster": cluster_name,
            "n_personas": int(cluster_mask.sum()),
        }
        for response_value in LIKERT_RESPONSE_VALUES:
            count = int(counts.loc[response_value])
            row[f"count_{response_value}"] = count
            row[f"pct_{response_value}"] = 100.0 * count / max(int(cluster_mask.sum()), 1)
        high_variance_cluster_rows.append(row)

high_variance_cluster_df = pd.DataFrame(high_variance_cluster_rows)
high_variance_cluster_path = (
    QUESTIONNAIRE_ITERATION_DUMP_DIR
    / "high_variance_item_response_by_assistant_like_cluster.csv"
)
high_variance_cluster_df.to_csv(high_variance_cluster_path, index=False)

high_variance_length_rows: list[dict] = []
for item_idx in high_variance_focus_indices:
    item_row = dump_item_metadata_df.iloc[item_idx]
    item_responses = dump_X[:, item_idx]
    for bucket_name in ("shorter_responses", "longer_responses"):
        bucket_mask = response_length_bucket == bucket_name
        bucket_values = item_responses[bucket_mask]
        counts = (
            pd.Series(bucket_values)
            .value_counts()
            .reindex(LIKERT_RESPONSE_VALUES, fill_value=0)
            .astype(int)
        )
        row = {
            "item_label": item_row["item_label"],
            "item_id": item_row["item_id"],
            "dimension": item_row["dimension"],
            "variance": item_row["variance"],
            "mean": item_row["mean"],
            "response_length_bucket": bucket_name,
            "n_personas": int(bucket_mask.sum()),
        }
        for response_value in LIKERT_RESPONSE_VALUES:
            count = int(counts.loc[response_value])
            row[f"count_{response_value}"] = count
            row[f"pct_{response_value}"] = 100.0 * count / max(int(bucket_mask.sum()), 1)
        high_variance_length_rows.append(row)

high_variance_length_df = pd.DataFrame(high_variance_length_rows)
high_variance_length_path = (
    QUESTIONNAIRE_ITERATION_DUMP_DIR
    / "high_variance_item_response_by_response_length_bucket.csv"
)
high_variance_length_df.to_csv(high_variance_length_path, index=False)

representative_item_specs = _select_representative_item_specs(dump_item_metadata_df)
representative_item_rows: list[dict] = []
representative_markdown_sections: list[str] = [
    "# Representative Questionnaire Rollout Examples",
    "",
    "Selection method:",
    "- Variance and mean examples use items closest to the 10th, 50th, and 90th percentile of that statistic.",
    f"- Random examples use a deterministic RNG seed of {REPRESENTATIVE_ITEM_RANDOM_SEED}.",
    "- For each selected item, the file shows low-response, mid-response, and high-response rollout examples when distinct transcripts are available.",
    "",
]

for spec in representative_item_specs:
    item_idx = int(spec["analysis_col_index"])
    item_row = dump_item_metadata_df.iloc[item_idx]
    item_responses = dump_X[:, item_idx]
    distribution_counts = (
        pd.Series(item_responses)
        .value_counts()
        .reindex(LIKERT_RESPONSE_VALUES, fill_value=0)
        .astype(int)
    )
    distribution_text = ", ".join(
        f"{value}={100.0 * int(distribution_counts.loc[value]) / len(item_responses):.1f}%"
        for value in LIKERT_RESPONSE_VALUES
    )

    representative_markdown_sections.extend(
        [
            f"## {spec['selection_group']} / {spec['selection_bucket']}: {item_row['item_label']}",
            "",
            f"- Item ID: `{item_row['item_id']}`",
            f"- Dimension: `{item_row['dimension'] or 'unassigned'}`",
            f"- Reverse keyed: `{bool(item_row['reverse_keyed'])}`",
            f"- Mean: `{item_row['mean']:.3f}`",
            f"- SD: `{item_row['sd']:.3f}`",
            f"- Variance: `{item_row['variance']:.3f}`",
            f"- 1-5 distribution: {distribution_text}",
            "",
            item_row["item_text"],
            "",
        ]
    )

    example_rows = _pick_example_sample_indices(item_responses, available_rollout_mask)
    for example in example_rows:
        row_index = int(example["row_index"])
        sample_id = str(sample_ids_by_row[row_index])
        rollout_record = rollout_records_by_sample_id.get(sample_id)
        if rollout_record is None:
            continue

        messages = list(rollout_record.get("messages", []))
        archetype = archetypes_by_sample_id.get(sample_id, "unknown")
        representative_item_rows.append(
            {
                **spec,
                **item_row.to_dict(),
                "example_label": example["example_label"],
                "sample_row_index": row_index,
                "sample_id": sample_id,
                "input_group_id": questionnaire_metadata[row_index].get("input_group_id", ""),
                "archetype": archetype,
                "response_value": float(example["response_value"]),
                "message_count": len(messages),
            }
        )
        representative_markdown_sections.extend(
            [
                f"### {example['example_label']}",
                "",
                f"- Sample ID: `{sample_id}`",
                f"- Archetype: `{archetype}`",
                f"- Response on this item: `{int(example['response_value'])}`",
                f"- Transcript messages: `{len(messages)}`",
                "",
                _format_messages_for_markdown(messages),
                "",
            ]
        )

representative_summary_df = pd.DataFrame(representative_item_rows)
representative_summary_path = (
    QUESTIONNAIRE_ITERATION_DUMP_DIR / "representative_rollout_examples_summary.csv"
)
representative_summary_df.to_csv(representative_summary_path, index=False)

representative_markdown_path = (
    QUESTIONNAIRE_ITERATION_DUMP_DIR / "representative_rollout_examples.md"
)
representative_markdown_path.write_text(
    "\n".join(representative_markdown_sections).strip() + "\n",
    encoding="utf-8",
)

persona_score_example_rows: list[dict] = []
persona_score_markdown_sections: list[str] = [
    "# Persona Score Extreme Rollout Examples",
    "",
    "Selection method:",
    f"- For each analysis family, the file selects the top `{TOP_N_COMPONENTS_FOR_PERSONA_SCORE_DUMP}` components/factors by variance contribution.",
    f"- For each selected component/factor, the file shows the top `{TOP_K_PERSONAS_PER_SCORE_EXTREME}` high-score personas and top `{TOP_K_PERSONAS_PER_SCORE_EXTREME}` low-score personas with available rollouts.",
    "- Factor/component signs are arbitrary, so the high-score and low-score poles should be interpreted relative to the item weights shown for each side.",
    "",
]

pca_component_names = [
    f"PC{component_idx + 1}" for component_idx in range(top_pca_loading_matrix.shape[1])
]
score_dump_analyses: list[dict] = [
    {
        "analysis_name": f"top_{top_n_for_pca}_variance_pca",
        "score_kind": "pca_component",
        "score_names": pca_component_names,
        "score_matrix": top_pca_scores,
        "weight_matrix": top_pca_loading_matrix,
        "score_strength": top_pca_explained_ratio,
        "weight_label": "loading",
        "item_df": top_pca_item_df.reset_index(drop=True),
    }
]
for analysis_name, analysis_payload in sorted(rotated_factor_results.items()):
    score_dump_analyses.append(
        {
            "analysis_name": analysis_name,
            **analysis_payload,
        }
    )

for analysis_payload in score_dump_analyses:
    analysis_name = str(analysis_payload["analysis_name"])
    score_kind = str(analysis_payload["score_kind"])
    score_names = list(analysis_payload["score_names"])
    score_matrix = np.asarray(analysis_payload["score_matrix"], dtype=np.float64)
    weight_matrix = np.asarray(analysis_payload["weight_matrix"], dtype=np.float64)
    score_strength = np.asarray(analysis_payload["score_strength"], dtype=np.float64)
    weight_label = str(analysis_payload["weight_label"])
    analysis_item_df = pd.DataFrame(analysis_payload["item_df"]).reset_index(drop=True)

    if score_matrix.shape[0] != len(sample_ids_by_row):
        raise ValueError(
            f"{analysis_name} score rows ({score_matrix.shape[0]}) do not match persona rows "
            f"({len(sample_ids_by_row)})."
        )
    if weight_matrix.shape[0] != len(analysis_item_df):
        raise ValueError(
            f"{analysis_name} weight rows ({weight_matrix.shape[0]}) do not match item rows "
            f"({len(analysis_item_df)})."
        )

    n_scores = min(
        TOP_N_COMPONENTS_FOR_PERSONA_SCORE_DUMP,
        score_matrix.shape[1],
        len(score_names),
        len(score_strength),
    )
    if n_scores <= 0:
        continue

    top_score_indices = np.argsort(score_strength)[::-1][:n_scores]
    persona_score_markdown_sections.extend(
        [
            f"## {analysis_name}",
            "",
        ]
    )

    for score_idx in top_score_indices:
        score_idx_int = int(score_idx)
        score_name = score_names[score_idx_int]
        score_values = score_matrix[:, score_idx_int]
        weight_values = weight_matrix[:, score_idx_int]
        high_indices, low_indices = _pick_extreme_score_indices(
            score_values,
            available_rollout_mask,
            top_k=TOP_K_PERSONAS_PER_SCORE_EXTREME,
        )
        positive_item_indices, negative_item_indices = _top_signed_weight_indices(
            weight_values,
            TOP_ITEMS_PER_PERSONA_SCORE_REPORT,
        )
        strength_value = float(score_strength[score_idx_int])
        strength_label = (
            "explained_variance_ratio"
            if score_kind == "pca_component"
            else "ss_loading"
        )

        persona_score_markdown_sections.extend(
            [
                f"### {score_name}",
                "",
                f"- Analysis: `{analysis_name}`",
                f"- Type: `{score_kind}`",
                f"- {strength_label}: `{strength_value:.4f}`",
                f"- Mean score across personas: `{float(np.mean(score_values)):+.4f}`",
                f"- SD across personas: `{float(np.std(score_values, ddof=1)):.4f}`",
                "",
                "Positive-pole items:",
            ]
        )
        if positive_item_indices:
            for item_idx in positive_item_indices:
                item_row = analysis_item_df.iloc[item_idx]
                persona_score_markdown_sections.append(
                    _format_weight_report_line(
                        item_row,
                        float(weight_values[item_idx]),
                        weight_label=weight_label,
                    )
                )
        else:
            persona_score_markdown_sections.append("- None")

        persona_score_markdown_sections.extend(
            [
                "",
                "Negative-pole items:",
            ]
        )
        if negative_item_indices:
            for item_idx in negative_item_indices:
                item_row = analysis_item_df.iloc[item_idx]
                persona_score_markdown_sections.append(
                    _format_weight_report_line(
                        item_row,
                        float(weight_values[item_idx]),
                        weight_label=weight_label,
                    )
                )
        else:
            persona_score_markdown_sections.append("- None")

        for pole_name, pole_indices in (
            ("high_score", high_indices),
            ("low_score", low_indices),
        ):
            persona_score_markdown_sections.extend(
                [
                    "",
                    f"#### {pole_name}",
                    "",
                ]
            )
            if not pole_indices:
                persona_score_markdown_sections.append("- No rollout transcripts available.")
                continue

            for rank_within_pole, row_index in enumerate(pole_indices, start=1):
                sample_id = str(sample_ids_by_row[row_index])
                rollout_record = rollout_records_by_sample_id.get(sample_id)
                if rollout_record is None:
                    continue

                messages = list(rollout_record.get("messages", []))
                archetype = archetypes_by_sample_id.get(sample_id, "unknown")
                persona_score_example_rows.append(
                    {
                        "analysis_name": analysis_name,
                        "score_kind": score_kind,
                        "score_name": score_name,
                        "score_index": score_idx_int,
                        "score_strength": strength_value,
                        "pole": pole_name,
                        "rank_within_pole": rank_within_pole,
                        "sample_row_index": int(row_index),
                        "sample_id": sample_id,
                        "input_group_id": questionnaire_metadata[row_index].get("input_group_id", ""),
                        "archetype": archetype,
                        "score_value": float(score_values[row_index]),
                        "message_count": len(messages),
                    }
                )
                persona_score_markdown_sections.extend(
                    [
                        f"##### {pole_name} example {rank_within_pole}",
                        "",
                        f"- Sample ID: `{sample_id}`",
                        f"- Archetype: `{archetype}`",
                        f"- {score_name} score: `{float(score_values[row_index]):+.4f}`",
                        f"- Transcript messages: `{len(messages)}`",
                        "",
                        _format_messages_for_markdown(messages),
                        "",
                    ]
                )

persona_score_summary_df = pd.DataFrame(persona_score_example_rows)
persona_score_summary_path = (
    QUESTIONNAIRE_ITERATION_DUMP_DIR / "persona_score_extreme_rollouts_summary.csv"
)
persona_score_summary_df.to_csv(persona_score_summary_path, index=False)

persona_score_markdown_path = (
    QUESTIONNAIRE_ITERATION_DUMP_DIR / "persona_score_extreme_rollouts.md"
)
persona_score_markdown_path.write_text(
    "\n".join(persona_score_markdown_sections).strip() + "\n",
    encoding="utf-8",
)

manifest_path = QUESTIONNAIRE_ITERATION_DUMP_DIR / "artifact_manifest.json"
manifest = {
    "questionnaire_run_id": QUESTIONNAIRE_RUN_ID,
    "questionnaire_dir": str(QUESTIONNAIRE_DIR),
    "analysis_blocks": list(ANALYSIS_BLOCKS),
    "n_personas": int(dump_X.shape[0]),
    "n_items": int(dump_X.shape[1]),
    "source_questionnaire_path": (
        str(source_questionnaire_path)
        if source_questionnaire_path is not None
        else None
    ),
    "rollout_transcript_path": (
        str(rollout_transcript_path)
        if rollout_transcript_path is not None
        else None
    ),
    "outputs": {
        "item_summary_csv": str(item_summary_path),
        "top_variance_distribution_csv": str(top_distribution_csv_path),
        "bottom_variance_distribution_csv": str(bottom_distribution_csv_path),
        "top_variance_distribution_plot": str(top_distribution_plot_path),
        "bottom_variance_distribution_plot": str(bottom_distribution_plot_path),
        "raw_item_correlation_matrix_csv": str(raw_corr_path),
        "key_aligned_item_correlation_matrix_csv": str(aligned_corr_path),
        "within_dimension_pairs_csv": str(within_dimension_pairs_path),
        "within_dimension_summary_csv": str(within_dimension_summary_path),
        "top_variance_pca_summary_csv": str(top_pca_summary_path),
        "top_variance_pca_items_csv": str(top_pca_items_path),
        "top_variance_pca_scree_plot": str(top_pca_plot_path),
        **rotated_factor_loading_paths,
        "persona_neutral_3_response_counts_csv": str(neutral_response_persona_path),
        "persona_neutral_3_response_histogram": str(neutral_response_hist_path),
        "high_variance_item_response_by_assistant_like_cluster_csv": str(
            high_variance_cluster_path
        ),
        "high_variance_item_response_by_response_length_bucket_csv": str(
            high_variance_length_path
        ),
        "representative_rollout_examples_summary_csv": str(representative_summary_path),
        "representative_rollout_examples_markdown": str(representative_markdown_path),
        "persona_score_extreme_rollouts_summary_csv": str(persona_score_summary_path),
        "persona_score_extreme_rollouts_markdown": str(persona_score_markdown_path),
    },
}
with manifest_path.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"Wrote questionnaire iteration dump to {QUESTIONNAIRE_ITERATION_DUMP_DIR}")
print(f"  Item summary: {item_summary_path}")
print(f"  Response distributions: {top_distribution_csv_path} and {bottom_distribution_csv_path}")
print(f"  Correlation matrices: {raw_corr_path} and {aligned_corr_path}")
print(f"  Within-dimension summaries: {within_dimension_summary_path}")
print(
    f"  PCA scree plot: {top_pca_plot_path} "
    f"({top_pca_n_gt_one} components with eigenvalue > 1)"
)
print(f'  Neutral-"3" histogram: {neutral_response_hist_path}')
print(f"  Rotated factor loadings: {rotated_factor_loading_paths}")
print(f"  Representative rollout examples: {representative_markdown_path}")
print(f"  Persona score extreme rollouts: {persona_score_markdown_path}")

# %%
