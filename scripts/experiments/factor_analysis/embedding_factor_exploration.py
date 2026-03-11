# %%
#!/usr/bin/env python3
"""Factor analysis of response embeddings — interactive exploration script.

Usage:
    Run cells interactively (e.g. in VS Code / Jupyter) or as a script.
    Outputs go to scratch/factor_analysis/.
"""

# %%
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ---- Configuration --------------------------------------------------------
EMBEDDINGS_PATH = Path("qwen4embeddings/stage123-240x50-singleturn-v2/response_embeddings_embeddings.npy")
METADATA_PATH = Path("qwen4embeddings/stage123-240x50-singleturn-v2/response_embeddings_metadata.jsonl")
OUTPUT_DIR = Path("scratch/factor_analysis/")
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"

PCA_PRE_N = 100
N_FACTORS = 15       # Override with parallel analysis result if desired
ROTATION = "varimax"
METHOD = "principal"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %%
# ---- Step 1: Load and preprocess -----------------------------------------
from scripts.factor_analysis.preprocessing import (
    load_embeddings,
    deduplicate_by_group,
    residualize,
    pca_reduce,
)

embeddings, metadata = load_embeddings(EMBEDDINGS_PATH, METADATA_PATH)
print(f"Loaded: {embeddings.shape[0]} samples, {embeddings.shape[1]}d")

embeddings, metadata = deduplicate_by_group(embeddings, metadata, max_per_group=50)
print(f"After dedup: {embeddings.shape[0]} samples")

# Save the original embeddings for corpus lookups later.
corpus_embeddings = embeddings.copy()
corpus_metadata = metadata

# Residualize: subtract per-prompt means to isolate style variation.
residuals, group_means, group_inverse = residualize(embeddings, metadata)
global_mean = embeddings.mean(axis=0)
print(f"Residualized: {residuals.shape}")

# PCA pre-reduce for factor analysis.
reduced, pca_model, scaler = pca_reduce(residuals, n_components=PCA_PRE_N)


# %%
# ---- Step 2: Adequacy tests ----------------------------------------------
from scripts.factor_analysis.factor_analysis import adequacy_tests

adequacy = adequacy_tests(reduced)


# %%
# ---- Step 3: Parallel analysis -------------------------------------------
from scripts.factor_analysis.parallel_analysis import parallel_analysis

pa_result = parallel_analysis(reduced, n_iterations=100, percentile=95)
n_recommended = pa_result["n_recommended"]
print(f"\nParallel analysis recommends {n_recommended} factors")

# Optionally override N_FACTORS with the recommendation.
# N_FACTORS = n_recommended


# %%
# ---- Step 4: Factor analysis ---------------------------------------------
from scripts.factor_analysis.factor_analysis import run_factor_analysis

fa_result = run_factor_analysis(reduced, n_factors=N_FACTORS, method=METHOD, rotation=ROTATION)

loadings = fa_result["loadings"]
scores = fa_result["scores"]
communalities = fa_result["communalities"]


# %%
# ---- Step 5: Visualizations ----------------------------------------------
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# 5a: Parallel analysis scree plot
fig_scree = go.Figure()
n_show = min(40, len(pa_result["real_eigenvalues"]))
fig_scree.add_trace(go.Scatter(
    x=list(range(1, n_show + 1)),
    y=pa_result["real_eigenvalues"][:n_show],
    mode="lines+markers",
    name="Real eigenvalues",
    line=dict(color="steelblue"),
))
fig_scree.add_trace(go.Scatter(
    x=list(range(1, n_show + 1)),
    y=pa_result["random_threshold"][:n_show],
    mode="lines+markers",
    name="95th pctl random",
    line=dict(color="firebrick", dash="dash"),
))
fig_scree.add_vline(x=n_recommended + 0.5, line_dash="dot", line_color="grey",
                    annotation_text=f"Recommended: {n_recommended}")
fig_scree.update_layout(
    title="Parallel Analysis Scree Plot",
    xaxis_title="Factor",
    yaxis_title="Eigenvalue",
    template="plotly_white",
    height=450,
)
fig_scree.show()


# %%
# 5b: Variance explained per factor
ss_loadings = fa_result["ss_loadings"]
prop_var = fa_result["proportion_variance"]
cum_var = fa_result["cumulative_variance"]

fig_var = make_subplots(specs=[[{"secondary_y": True}]])
fig_var.add_trace(
    go.Bar(
        x=[f"F{i+1}" for i in range(len(prop_var))],
        y=prop_var * 100,
        name="Variance %",
        marker_color="steelblue",
    ),
    secondary_y=False,
)
fig_var.add_trace(
    go.Scatter(
        x=[f"F{i+1}" for i in range(len(cum_var))],
        y=cum_var * 100,
        mode="lines+markers",
        name="Cumulative %",
        line=dict(color="firebrick"),
    ),
    secondary_y=True,
)
fig_var.update_layout(
    title=f"Variance Explained per Factor ({ROTATION or 'unrotated'})",
    template="plotly_white",
    height=420,
)
fig_var.update_yaxes(title_text="% variance", secondary_y=False)
fig_var.update_yaxes(title_text="Cumulative %", secondary_y=True)
fig_var.show()


# %%
# 5c: Communality histogram
fig_comm = go.Figure()
fig_comm.add_trace(go.Histogram(x=communalities, nbinsx=30, marker_color="steelblue"))
fig_comm.add_vline(x=float(communalities.mean()), line_dash="dash", line_color="red",
                   annotation_text=f"Mean={communalities.mean():.3f}")
fig_comm.update_layout(
    title=f"Communality Distribution ({N_FACTORS} factors)",
    xaxis_title="Communality h\u00b2",
    yaxis_title="Count",
    template="plotly_white",
    height=400,
)
fig_comm.show()
print(f"Communality: min={communalities.min():.4f} mean={communalities.mean():.4f} max={communalities.max():.4f}")


# %%
# 5d: Loading heatmap (top PCA dims by max absolute loading)
K_DIMS_SHOW = 40
max_abs = np.abs(loadings).max(axis=1)
top_dims = np.argsort(max_abs)[::-1][:K_DIMS_SHOW]

fig_heat = go.Figure(data=go.Heatmap(
    z=loadings[top_dims, :],
    x=[f"F{i+1}" for i in range(loadings.shape[1])],
    y=[f"PC{i+1}" for i in top_dims],
    colorscale="RdBu",
    zmid=0,
))
fig_heat.update_layout(
    title=f"Factor Loadings (top {K_DIMS_SHOW} PCA dims, {ROTATION or 'unrotated'})",
    xaxis_title="Factor",
    yaxis_title="PCA component",
    template="plotly_white",
    height=600,
)
fig_heat.show()


# %%
# 5e: Eta-squared (prompt effect per factor)
# Note: On residualized data, eta-squared will be ~0 by construction (group means
# were subtracted). To check how much prompt-driven variance the factors capture,
# we also project the *non-residualized* embeddings through the same factor model
# and compute eta-squared on those projections.
from scripts.factor_analysis.interpretation import prompt_effects

eta2 = prompt_effects(scores, metadata)

# Also compute eta-squared on non-residualized projections for comparison.
reduced_full, _, _ = pca_reduce(embeddings, n_components=PCA_PRE_N)
scores_full = reduced_full @ np.linalg.pinv(loadings.T)  # project through loadings
eta2_full = prompt_effects(scores_full, metadata)
print(f"\nEta-squared (residualized): {['%.4f' % e for e in eta2[:5]]}")
print(f"Eta-squared (full embeds):  {['%.4f' % e for e in eta2_full[:5]]}")

fig_eta = go.Figure()
fig_eta.add_trace(go.Bar(
    x=[f"F{i+1}" for i in range(len(eta2_full))],
    y=eta2_full,
    name="Full embeddings",
    marker_color=["firebrick" if e > 0.3 else "steelblue" for e in eta2_full],
    text=[f"{e:.3f}" for e in eta2_full],
    textposition="outside",
))
fig_eta.add_hline(y=0.3, line_dash="dash", line_color="grey",
                  annotation_text="\u03b7\u00b2=0.3")
fig_eta.update_layout(
    title="Prompt Effect (\u03b7\u00b2) per Factor (full embeddings projected through residual factors)",
    xaxis_title="Factor",
    yaxis_title="\u03b7\u00b2",
    yaxis_range=[0, 1],
    template="plotly_white",
    height=420,
)
fig_eta.show()

print("Factor \u03b7\u00b2 (full embeddings):")
for i, e in enumerate(eta2_full):
    tag = " \u2190 prompt-driven" if e > 0.3 else ""
    print(f"  F{i+1:2d}: {e:.4f}{tag}")


# %%
# 5f: Factor score scatter matrix (first 6 factors)
N_SCATTER = min(6, scores.shape[1])
SCATTER_MAX = 3000
rng = np.random.default_rng(42)
scatter_idx = (rng.choice(len(metadata), size=SCATTER_MAX, replace=False)
               if len(metadata) > SCATTER_MAX else np.arange(len(metadata)))

fig_scatter = make_subplots(
    rows=N_SCATTER - 1, cols=N_SCATTER - 1,
    horizontal_spacing=0.04, vertical_spacing=0.04,
)
for row_i in range(1, N_SCATTER):
    for col_j in range(row_i):
        fig_scatter.add_trace(
            go.Scatter(
                x=scores[scatter_idx, col_j],
                y=scores[scatter_idx, row_i],
                mode="markers",
                marker=dict(size=2, opacity=0.4, color="steelblue"),
                showlegend=False,
            ),
            row=row_i, col=col_j + 1,
        )
        if col_j == 0:
            fig_scatter.update_yaxes(title_text=f"F{row_i+1}", row=row_i, col=1)
        if row_i == N_SCATTER - 1:
            fig_scatter.update_xaxes(title_text=f"F{col_j+1}", row=row_i, col=col_j + 1)

fig_scatter.update_layout(
    title=f"Factor Score Scatter Matrix (first {N_SCATTER})",
    template="plotly_white",
    height=700, width=800,
)
fig_scatter.show()


# %%
# ---- Step 6: Interpretation — Methods 2 & 3 (no GPU needed) -------------
from scripts.factor_analysis.interpretation import (
    factor_extremes,
    rank_by_factor_purity,
    analytical_factor_embedding,
    corpus_nearest_neighbor,
)

# 6a: Simple factor extremes (top/bottom responses per factor)
extremes = factor_extremes(scores, metadata, top_n=10)
for fe in extremes[:3]:  # Print first 3 factors
    fi = fe["factor_index"]
    print(f"\n{'='*80}")
    print(f"FACTOR {fi + 1} — Top responses:")
    for s in fe["top"][:3]:
        print(f"  score={s['score']:.3f} | {s['text_excerpt'][:120]}...")
    print(f"FACTOR {fi + 1} — Bottom responses:")
    for s in fe["bottom"][:3]:
        print(f"  score={s['score']:.3f} | {s['text_excerpt'][:120]}...")


# %%
# 6b: Method 3 — Score-based purity ranking
for fi in range(min(3, N_FACTORS)):
    purity = rank_by_factor_purity(scores, metadata, factor_idx=fi, penalty_weight=1.0, top_n=5)
    print(f"\n{'='*80}")
    print(f"FACTOR {fi + 1} — Purest responses (high target, low others):")
    for s in purity["top"][:3]:
        print(f"  purity={s['purity_score']:.3f} target={s['target_factor_score']:.3f} "
              f"other_abs={s['other_factors_mean_abs']:.3f}")
        print(f"    {s['text_excerpt'][:120]}...")


# %%
# 6c: Method 2 — Analytical target + corpus nearest neighbor
for fi in range(min(3, N_FACTORS)):
    target, direction = analytical_factor_embedding(
        fi, loadings, pca_model, scaler, global_mean, scale=3.0
    )
    neighbors = corpus_nearest_neighbor(target, corpus_embeddings, corpus_metadata, top_k=5)
    print(f"\n{'='*80}")
    print(f"FACTOR {fi + 1} — Nearest corpus responses to analytical target:")
    for nb in neighbors[:3]:
        print(f"  sim={nb['similarity']:.4f} | {nb['text_excerpt'][:120]}...")


# %%
# ---- Step 7: Save results ------------------------------------------------
results_summary = {
    "n_samples": int(embeddings.shape[0]),
    "original_dim": int(embeddings.shape[1]),
    "pca_pre_n": PCA_PRE_N,
    "pca_variance_retained": float(pca_model.explained_variance_ratio_.sum()),
    "n_factors": N_FACTORS,
    "method": METHOD,
    "rotation": ROTATION,
    "n_recommended_parallel_analysis": n_recommended,
    "adequacy": {
        "bartlett_chi2": adequacy["bartlett_chi2"],
        "bartlett_p": adequacy["bartlett_p"],
        "kmo_overall": adequacy["kmo_overall"],
    },
    "communality_stats": {
        "min": float(communalities.min()),
        "mean": float(communalities.mean()),
        "max": float(communalities.max()),
    },
    "eta_squared": eta2.tolist(),
}

(OUTPUT_DIR / "summary.json").write_text(json.dumps(results_summary, indent=2))
np.savez(
    OUTPUT_DIR / "factor_analysis.npz",
    loadings=loadings,
    scores=scores,
    communalities=communalities,
    eigenvalues_original=fa_result["eigenvalues_original"],
    eigenvalues_common=fa_result["eigenvalues_common"],
)
np.savez(
    OUTPUT_DIR / "parallel_analysis.npz",
    real_eigenvalues=pa_result["real_eigenvalues"],
    random_threshold=pa_result["random_threshold"],
)

print(f"\nResults saved to {OUTPUT_DIR}")


# %%
# ---- Step 8 (optional): Method 1 — Gradient descent through model --------
# Uncomment to run. Requires GPU and loads the embedding model (~8GB).
#
# from scripts.factor_analysis.interpretation import optimize_factor_embedding
#
# FACTORS_TO_OPTIMIZE = [0, 1, 2]  # Which factors to optimize
#
# for fi in FACTORS_TO_OPTIMIZE:
#     print(f"\nOptimizing for factor {fi + 1}...")
#     opt_result = optimize_factor_embedding(
#         factor_idx=fi,
#         n_factors=N_FACTORS,
#         loadings=loadings,
#         pca_model=pca_model,
#         scaler=scaler,
#         global_mean=global_mean,
#         model_name=EMBEDDING_MODEL,
#         seq_length=32,
#         n_steps=500,
#         lr=0.01,
#     )
#
#     # Method 4: Gradient descent + corpus lookup
#     neighbors = corpus_nearest_neighbor(
#         opt_result["optimized_embedding"], corpus_embeddings, corpus_metadata, top_k=5
#     )
#     print(f"Factor {fi + 1} — Nearest corpus responses to optimized embedding:")
#     for nb in neighbors[:3]:
#         print(f"  sim={nb['similarity']:.4f} | {nb['text_excerpt'][:120]}...")

# %%
