# %%
#!/usr/bin/env python3
"""Notebook-style inspection for response embedding outputs."""

#%%
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


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


# %%
# ---- Setup: rebuild group structure if not already in scope ----------
import importlib
import json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Rebuild group structure (idempotent if already defined above).
group_ids = np.array([str(row["input_group_id"]) for row in metadata], dtype=object)
unique_groups, group_inverse = np.unique(group_ids, return_inverse=True)
num_groups = len(unique_groups)
n, d = embeddings.shape
group_to_indices = {g: np.where(group_ids == g)[0] for g in unique_groups}
group_sizes = np.array([len(group_to_indices[g]) for g in unique_groups], dtype=int)

# Prompt-mean-residualised embeddings (within-prompt style variation).
group_means = np.zeros((num_groups, d), dtype=np.float64)
for gi, g in enumerate(unique_groups):
    group_means[gi] = embeddings[group_to_indices[g]].mean(axis=0)
residual_embeddings = embeddings - group_means[group_inverse]

print(f"n={n}, d={d}, num_groups={num_groups}")


# %%
# ---- PAF config -------------------------------------------------------
# Principal Axis Factoring extracts *common* variance by iteratively
# refining communality estimates on the correlation-matrix diagonal,
# unlike PCA which places 1s there and captures all (including unique)
# variance.
#
# Strategy:
#   1. Standardise embeddings (z-score each dimension).
#   2. Pre-reduce with PCA to make the correlation matrix tractable.
#   3. Run iterative PAF on the reduced space.
#   4. Optionally apply varimax rotation for interpretability.

PCA_PRE_N       = 100    # PCA dims before factor analysis
N_FACTORS       = 15     # Number of factors to extract
PAF_MAX_ITER    = 500    # Max communality-refinement iterations
PAF_TOL         = 1e-6   # Convergence tolerance
ROTATION        = "varimax"   # "varimax", "oblimin", or None
RANDOM_STATE    = 0

# Operate on prompt-residualised embeddings to focus on style variation,
# not prompt-content variation.
USE_RESIDUALS   = True

X_raw = residual_embeddings if USE_RESIDUALS else embeddings
label_suffix = "residual" if USE_RESIDUALS else "full"

scaler = StandardScaler()
X_std = scaler.fit_transform(X_raw)

pca_pre = PCA(n_components=PCA_PRE_N, random_state=RANDOM_STATE)
X_pca = pca_pre.fit_transform(X_std)

print(f"Input shape: {X_raw.shape}")
print(f"After PCA pre-reduction: {X_pca.shape}")
print(f"Variance retained by pre-PCA: {pca_pre.explained_variance_ratio_.sum():.3%}")


# %%
# ---- PAF implementation -----------------------------------------------
def _smc(R: np.ndarray) -> np.ndarray:
    """Squared multiple correlations as initial communality estimates.

    SMC_i = 1 - 1 / R^{-1}_{ii}  (bounded to [0, 1]).
    """
    try:
        R_inv = np.linalg.inv(R)
        smc = 1.0 - 1.0 / np.diag(R_inv)
        return np.clip(smc, 0.0, 1.0)
    except np.linalg.LinAlgError:
        # Fallback to a simpler estimate if R is singular.
        return np.full(R.shape[0], 0.5)


def principal_axis_factoring(
    X: np.ndarray,
    n_factors: int,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterative Principal Axis Factoring.

    Args:
        X: Standardised data matrix (n_samples x n_vars).
        n_factors: Number of factors to extract.
        max_iter: Maximum communality-refinement iterations.
        tol: Convergence tolerance on communality change.

    Returns:
        loadings: (n_vars x n_factors) unrotated factor loadings.
        communalities: (n_vars,) final communality estimates.
        eigenvalues: (n_factors,) eigenvalues of the reduced correlation matrix.
    """
    p = X.shape[1]
    R = np.corrcoef(X, rowvar=False)  # (p x p) correlation matrix

    # Initial communalities via SMC.
    h2 = _smc(R)

    for iteration in range(max_iter):
        R_reduced = R.copy()
        np.fill_diagonal(R_reduced, h2)

        eigenvalues_all, eigenvectors = np.linalg.eigh(R_reduced)
        # eigh returns ascending order; reverse to descending.
        idx = np.argsort(eigenvalues_all)[::-1]
        eigenvalues_all = eigenvalues_all[idx]
        eigenvectors = eigenvectors[:, idx]

        # Keep only positive eigenvalues for the retained factors.
        k = min(n_factors, int((eigenvalues_all > 0).sum()))
        lam = eigenvalues_all[:k]
        V = eigenvectors[:, :k]

        loadings = V * np.sqrt(np.maximum(lam, 0.0))  # (p x k)
        h2_new = np.clip((loadings ** 2).sum(axis=1), 0.0, 1.0)

        delta = float(np.max(np.abs(h2_new - h2)))
        h2 = h2_new

        if delta < tol:
            print(f"PAF converged after {iteration + 1} iterations (delta={delta:.2e}).")
            break
    else:
        print(f"PAF reached max_iter={max_iter} (delta={delta:.2e}).")

    return loadings, h2, eigenvalues_all[:k]


# Fit PAF.
loadings_unrot, communalities, factor_eigenvalues = principal_axis_factoring(
    X_pca, n_factors=N_FACTORS, max_iter=PAF_MAX_ITER, tol=PAF_TOL
)
print(f"Loadings shape: {loadings_unrot.shape}")
print(f"Mean communality: {communalities.mean():.4f}")


# %%
# ---- Optional varimax rotation ----------------------------------------
def varimax(loadings: np.ndarray, max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
    """Varimax rotation (raw, no Kaiser normalisation).

    Args:
        loadings: (p x k) unrotated loading matrix.
        max_iter: Maximum pairwise-rotation sweeps.
        tol: Convergence tolerance.

    Returns:
        Rotated loading matrix (p x k).
    """
    L = loadings.copy()
    p, k = L.shape
    for _ in range(max_iter):
        old = L.copy()
        for i in range(k):
            for j in range(i + 1, k):
                x = L[:, i]
                y = L[:, j]
                u = x ** 2 - y ** 2
                v = 2 * x * y
                A = u.sum()
                B = v.sum()
                C = (u ** 2 - v ** 2).sum()
                D = (u * v).sum()
                num = 2 * (p * D - A * B)
                den = p * C - (A ** 2 - B ** 2)
                if abs(den) < 1e-12:
                    continue
                theta = 0.25 * np.arctan2(num, den)
                c, s = np.cos(theta), np.sin(theta)
                L[:, [i, j]] = np.column_stack([c * x + s * y, -s * x + c * y])
        if np.max(np.abs(L - old)) < tol:
            break
    return L


if ROTATION == "varimax":
    loadings = varimax(loadings_unrot)
    print(f"Applied varimax rotation to {loadings.shape[1]} factors.")
else:
    loadings = loadings_unrot
    print("No rotation applied.")

# Factor scores via regression method: F = X (L'L)^{-1} L'
# Using PCA-reduced data; scores are (n x k).
LtL = loadings.T @ loadings
try:
    factor_scores = X_pca @ loadings @ np.linalg.inv(LtL)
except np.linalg.LinAlgError:
    factor_scores = X_pca @ loadings @ np.linalg.pinv(LtL)
print(f"Factor scores shape: {factor_scores.shape}")


# %%
# ---- Plot 1: Scree plot of factor eigenvalues -------------------------
fig_scree = go.Figure()
fig_scree.add_trace(go.Scatter(
    x=list(range(1, len(factor_eigenvalues) + 1)),
    y=factor_eigenvalues,
    mode="lines+markers",
    marker=dict(size=8),
    line=dict(color="steelblue"),
    name="Eigenvalue",
))
fig_scree.add_hline(y=1.0, line_dash="dash", line_color="red",
                    annotation_text="Kaiser criterion (λ=1)")
fig_scree.update_layout(
    title=f"PAF Scree Plot ({label_suffix} embeddings, pre-PCA={PCA_PRE_N}d)",
    xaxis_title="Factor",
    yaxis_title="Eigenvalue (reduced R)",
    template="plotly_white",
    height=450,
)
fig_scree.show()


# %%
# ---- Plot 2: Communalities bar chart ----------------------------------
# Communalities are per pre-PCA dimension; summarise as a histogram.
fig_comm = go.Figure()
fig_comm.add_trace(go.Histogram(
    x=communalities,
    nbinsx=30,
    marker_color="steelblue",
    opacity=0.8,
    name="Communality",
))
fig_comm.add_vline(
    x=float(communalities.mean()),
    line_dash="dash", line_color="red",
    annotation_text=f"Mean={communalities.mean():.3f}",
    annotation_position="top right",
)
fig_comm.update_layout(
    title=f"Communality Distribution ({N_FACTORS} factors, {label_suffix})",
    xaxis_title="Communality h²",
    yaxis_title="Count (pre-PCA dimensions)",
    template="plotly_white",
    height=400,
)
fig_comm.show()
print(f"Communality stats: min={communalities.min():.4f} mean={communalities.mean():.4f} max={communalities.max():.4f}")


# %%
# ---- Plot 3: Factor loading heatmap (pre-PCA dims x factors) ----------
# Show only top-K PCA dims by max absolute loading for readability.
K_DIMS_SHOW = 40
max_abs_per_dim = np.abs(loadings).max(axis=1)
top_dim_idx = np.argsort(max_abs_per_dim)[::-1][:K_DIMS_SHOW]
loadings_subset = loadings[top_dim_idx, :]

fig_heat = go.Figure(data=go.Heatmap(
    z=loadings_subset,
    x=[f"F{i+1}" for i in range(loadings.shape[1])],
    y=[f"PC{i+1}" for i in top_dim_idx],
    colorscale="RdBu",
    zmid=0,
    colorbar=dict(title="Loading"),
))
fig_heat.update_layout(
    title=(
        f"Factor Loadings Heatmap — top {K_DIMS_SHOW} pre-PCA dims by max |loading|<br>"
        f"({label_suffix}, {N_FACTORS} factors, {ROTATION or 'unrotated'})"
    ),
    xaxis_title="Factor",
    yaxis_title="Pre-PCA component",
    template="plotly_white",
    height=600,
)
fig_heat.show()


# %%
# ---- Plot 4: Factor score scatter matrix (first 6 factors) -----------
N_FACTORS_SCATTER = min(6, factor_scores.shape[1])

# Sample for speed if dataset is large.
SCATTER_MAX = 3000
rng = np.random.default_rng(42)
scatter_idx = (
    rng.choice(n, size=SCATTER_MAX, replace=False) if n > SCATTER_MAX
    else np.arange(n)
)

persona_labels = np.array([row.get("persona", row.get("model_id", "unknown"))
                            for row in metadata], dtype=object)
unique_personas = np.unique(persona_labels)
persona_color_map = {p: i for i, p in enumerate(unique_personas)}
color_vals = np.array([persona_color_map[p] for p in persona_labels[scatter_idx]])

fig_scatter = make_subplots(
    rows=N_FACTORS_SCATTER - 1,
    cols=N_FACTORS_SCATTER - 1,
    shared_xaxes=False,
    shared_yaxes=False,
    horizontal_spacing=0.04,
    vertical_spacing=0.04,
)

for row_i in range(1, N_FACTORS_SCATTER):
    for col_j in range(row_i):
        fi, fj = row_i, col_j  # 0-indexed factor indices
        fig_scatter.add_trace(
            go.Scatter(
                x=factor_scores[scatter_idx, fj],
                y=factor_scores[scatter_idx, fi],
                mode="markers",
                marker=dict(
                    size=3,
                    color=color_vals,
                    colorscale="Turbo",
                    opacity=0.5,
                ),
                showlegend=False,
            ),
            row=row_i,
            col=col_j + 1,
        )
        if col_j == 0:
            fig_scatter.update_yaxes(title_text=f"F{fi+1}", row=row_i, col=1)
        if row_i == N_FACTORS_SCATTER - 1:
            fig_scatter.update_xaxes(title_text=f"F{fj+1}", row=row_i, col=col_j + 1)

fig_scatter.update_layout(
    title=f"PAF Factor Score Scatter Matrix — first {N_FACTORS_SCATTER} factors ({label_suffix})",
    template="plotly_white",
    height=700,
    width=800,
)
fig_scatter.show()


# %%
# ---- Plot 5: Variance explained by each factor ------------------------
# Proportion of total common variance = sum(loadings[:,f]^2) / p
factor_var = (loadings ** 2).sum(axis=0)
factor_var_pct = 100.0 * factor_var / loadings.shape[0]
cumvar = np.cumsum(factor_var_pct)

fig_var = make_subplots(specs=[[{"secondary_y": True}]])
fig_var.add_trace(
    go.Bar(
        x=[f"F{i+1}" for i in range(len(factor_var_pct))],
        y=factor_var_pct,
        name="Variance %",
        marker_color="steelblue",
        opacity=0.8,
    ),
    secondary_y=False,
)
fig_var.add_trace(
    go.Scatter(
        x=[f"F{i+1}" for i in range(len(cumvar))],
        y=cumvar,
        mode="lines+markers",
        name="Cumulative %",
        line=dict(color="firebrick"),
        marker=dict(size=7),
    ),
    secondary_y=True,
)
fig_var.update_layout(
    title=f"PAF — Variance Explained per Factor ({label_suffix}, {ROTATION or 'unrotated'})",
    template="plotly_white",
    height=420,
    legend=dict(orientation="h", y=1.05),
)
fig_var.update_yaxes(title_text="% variance (within pre-PCA space)", secondary_y=False)
fig_var.update_yaxes(title_text="Cumulative %", secondary_y=True)
fig_var.show()

print("Factor variance (% of pre-PCA space):")
for i, (v, cv) in enumerate(zip(factor_var_pct, cumvar), start=1):
    print(f"  F{i:2d}: {v:.2f}%  (cum {cv:.2f}%)")


# %%
# ---- Plot 6: Prompt-group effect per factor (eta²) --------------------
# How much of each factor's score variance is explained by prompt group?
def eta_squared_1d(values: np.ndarray, group_inv: np.ndarray, n_groups: int) -> float:
    grand = float(values.mean())
    ss_tot = float(np.square(values - grand).sum())
    if ss_tot <= 0.0:
        return 0.0
    ss_bet = sum(
        float(len(np.where(group_inv == gi)[0])) *
        float((values[group_inv == gi].mean() - grand) ** 2)
        for gi in range(n_groups)
    )
    return ss_bet / ss_tot


factor_eta2 = np.array([
    eta_squared_1d(factor_scores[:, fi], group_inverse, num_groups)
    for fi in range(factor_scores.shape[1])
])

fig_eta = go.Figure(go.Bar(
    x=[f"F{i+1}" for i in range(len(factor_eta2))],
    y=factor_eta2,
    marker_color=[
        "firebrick" if e > 0.3 else "steelblue" for e in factor_eta2
    ],
    text=[f"{e:.3f}" for e in factor_eta2],
    textposition="outside",
))
fig_eta.add_hline(y=0.3, line_dash="dash", line_color="grey",
                  annotation_text="η²=0.3", annotation_position="top right")
fig_eta.update_layout(
    title=(
        f"Prompt-Group Effect (η²) per PAF Factor<br>"
        f"({label_suffix} — factors above 0.3 are heavily prompt-driven)"
    ),
    xaxis_title="Factor",
    yaxis_title="η² (prompt group)",
    yaxis_range=[0, 1],
    template="plotly_white",
    height=420,
)
fig_eta.show()

print("Factor η² (prompt effect):")
for i, e in enumerate(factor_eta2, start=1):
    tag = " ← prompt-driven" if e > 0.3 else ""
    print(f"  F{i:2d}: {e:.4f}{tag}")
# %%
