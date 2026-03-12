# %%
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from scripts.factor_analysis.preprocessing import load_embeddings, deduplicate_by_group, residualize, pca_reduce
from scripts.factor_analysis.parallel_analysis import parallel_analysis
from scripts.factor_analysis.factor_analysis import run_factor_analysis, adequacy_tests
from scripts.factor_analysis.persistence import save_factor_analysis, load_factor_analysis
from scripts.factor_analysis.labelling import label_factors, DEFAULT_MODEL as LABELLER_DEFAULT_MODEL
from scripts.factor_analysis.interpretation import (
    factor_extremes, rank_by_factor_purity,
    analytical_factor_embedding, corpus_nearest_neighbor,
    optimize_factor_embedding, prompt_effects,
)

# %%
# --- Config ---
USE_PCA = False   # False: run directly on 2560d residuals (slow but no lossy reduction)
SCALE = False
PCA_N_COMPONENTS = 100
N_FACTORS = 20
LABEL_FACTORS = True   # Call LLM to generate a description for each factor
# LABELLER_MODEL = 'claude-haiku-4-6'
# LABELLER_PROVIDER = 'anthropic'
LABELLER_MODEL = 'gpt-5-mini-2025-08-07'
LABELLER_PROVIDER = 'openai'

# %%
# Load and preprocess
embeddings, metadata = load_embeddings("qwen4embeddings/stage123-240x50-singleturn-v2/response_embeddings_embeddings.npy", "qwen4embeddings/stage123-240x50-singleturn-v2/response_embeddings_metadata.jsonl")
embeddings, metadata = deduplicate_by_group(embeddings, metadata, max_per_group=50)

# Filter out short responses (likely "I am an AI" deflections with no real content)
MIN_RESPONSE_CHARS = 400
_keep = [i for i, row in enumerate(metadata) if len(str(row.get("assistant_text", ""))) >= MIN_RESPONSE_CHARS]
_n_removed = len(metadata) - len(_keep)
embeddings = embeddings[_keep]
metadata = [metadata[i] for i in _keep]
print(f"Filtered {_n_removed} short responses (<{MIN_RESPONSE_CHARS} chars), {len(metadata)} remaining")

# Drop any prompt groups that now have only one response (residual would be zero)
from collections import Counter
_group_counts = Counter(str(row.get("input_group_id", i)) for i, row in enumerate(metadata))
_singleton_groups = {g for g, c in _group_counts.items() if c < 2}
if _singleton_groups:
    _keep2 = [i for i, row in enumerate(metadata) if str(row.get("input_group_id", i)) not in _singleton_groups]
    _n_singleton = len(metadata) - len(_keep2)
    embeddings = embeddings[_keep2]
    metadata = [metadata[i] for i in _keep2]
    print(f"Dropped {_n_singleton} responses from {len(_singleton_groups)} singleton groups, {len(metadata)} remaining")

import plotly.express as px
_counts_after = list(Counter(str(row.get("input_group_id", i)) for i, row in enumerate(metadata)).values())
fig = px.histogram(x=_counts_after, nbins=50, title="Responses per prompt after filtering",
                   labels={"x": "Responses per prompt", "y": "Number of prompts"})
fig.show()

corpus_embeddings = embeddings.copy()  # keep for nearest-neighbor lookups later
residuals, group_means, group_inv = residualize(embeddings, metadata)
global_mean = embeddings.mean(axis=0)

# %%
# Dimensionality reduction
if USE_PCA:
    data, pca_model, scaler = pca_reduce(residuals, n_components=PCA_N_COMPONENTS)
else:
    if SCALE:
        # Standardize only — note: parallel analysis and FA will be slow on 2560d
        scaler = StandardScaler()
        data = scaler.fit_transform(residuals)
    else:
        data = residuals
        scaler = None
    pca_model = None

# %%
# Determine number of factors (if-ed out as we did this and it was 341)
if 0:
    pa = parallel_analysis(data)  # Horn's method
    n_factors = pa["n_recommended"]  # or pick manually from scree plot
    print(f"Parallel analysis recommends {n_factors} factors")

    # Plot parallel analysis scree plot
    import plotly.graph_objects as go

    _n = len(pa["real_eigenvalues"])
    _x = list(range(1, _n + 1))
    fig = go.Figure([
        go.Scatter(x=_x, y=pa["real_eigenvalues"].tolist(), name="Real eigenvalues", mode="lines+markers"),
        go.Scatter(x=_x, y=pa["random_threshold"].tolist(), name="Random 95th pct", mode="lines", line=dict(dash="dash")),
    ])
    fig.add_vline(x=n_factors, line_dash="dot", annotation_text=f"n={n_factors}", annotation_position="top right")
    fig.update_layout(title="Parallel analysis scree plot", xaxis_title="Component", yaxis_title="Eigenvalue")
    fig.show()

# %%
# Run factor analysis (or load cached result)
FA_METHOD = "principal"
FA_ROTATION = "promax"
_fa_cache = Path(f"scratch/factor_analysis/fa_n{N_FACTORS}_{FA_METHOD}_{FA_ROTATION}_filtered")

if _fa_cache.with_suffix(".npz").exists():
    fa = load_factor_analysis(_fa_cache)
else:
    fa = run_factor_analysis(data, n_factors=N_FACTORS, method=FA_METHOD, rotation=FA_ROTATION)
    save_factor_analysis(fa, _fa_cache, config={"n_factors": N_FACTORS, "method": FA_METHOD, "rotation": FA_ROTATION, "use_pca": USE_PCA, "scale": SCALE})

scores, loadings = fa["scores"], fa["loadings"]

# %%
# Interpretation — pca_model=None is handled when USE_PCA=False
target, direction = analytical_factor_embedding(0, loadings, pca_model=pca_model, scaler=scaler, global_mean=global_mean, scale=3.0)

# %%
# For visual exploration of the factor structure itself, a heatmap of the loading matrix is useful — rows are embedding dimensions (or PCA components), columns are factors, and you can see which dimensions each factor loads on:
import plotly.express as px
fig = px.imshow(loadings, aspect="auto", color_continuous_scale="RdBu", color_continuous_midpoint=0,
                title="Factor loadings", labels=dict(x="Factor", y="Dimension"))
fig.show()

# And a scatter of factor score variance (how much each factor spreads the data):
import plotly.graph_objects as go
fig = go.Figure(go.Bar(y=fa["proportion_variance"], x=list(range(N_FACTORS))))
fig.update_layout(title="Variance explained per factor", xaxis_title="Factor", yaxis_title="Proportion variance")
fig.show()

# %%
# Save factor extremes to JSONL for TUI browsing
# Up/down in TUI = factor 0 HIGH / factor 0 LOW / factor 1 HIGH / ...
# Left/right in TUI = cycle through responses within that group
import json
from pathlib import Path

extremes = factor_extremes(scores, metadata, top_n=20, excerpt_length=100000)

out_path = Path("scratch/factor_analysis/extremes.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)

records = []
for factor_data in extremes:
    fi = factor_data["factor_index"]
    for polarity, entries in [("HIGH", factor_data["top"]), ("LOW", factor_data["bottom"])]:
        label = f"Factor {fi:03d} — {polarity}"
        for rank, entry in enumerate(entries):
            records.append({
                "question": label,
                "response_index": rank,
                "factor_score": round(entry["score"], 4),
                "prompt": entry["seed_user_message"],
                "response": entry["text_excerpt"],
            })

with open(out_path, "w") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Saved {len(records)} records to {out_path}")
print(f"\nuv run python scripts/jsonl_tui/cli.py {out_path} --variant-fields question response_index factor_score prompt response")

# %%
# Label factors with LLM (or load cached labels)
import json

_labels_path = _fa_cache.with_name(_fa_cache.name + "_labels.json")

if _labels_path.exists():
    with open(_labels_path) as f:
        factor_labels = json.load(f)
    print(f"Loaded {len(factor_labels)} factor labels from {_labels_path}")
elif LABEL_FACTORS:
    from dotenv import load_dotenv
    load_dotenv()
    factor_labels = label_factors(extremes, model=LABELLER_MODEL, provider=LABELLER_PROVIDER, top_n=10)
    with open(_labels_path, "w") as f:
        json.dump(factor_labels, f, indent=2)
    print(f"Saved factor labels to {_labels_path}")
else:
    factor_labels = None

if factor_labels:
    for i, label in enumerate(factor_labels):
        print(f"Factor {i:03d}: {label}")

# %%
# Purity-based ranking: high on target factor, low on all others
# Useful when factors aren't orthogonal (e.g. promax) — extremes alone can reflect
# correlated factors, purity isolates each one more cleanly.
purity_results = [
    rank_by_factor_purity(scores, metadata, factor_idx=fi, top_n=20, excerpt_length=100000)
    for fi in range(N_FACTORS)
]

purity_out_path = Path("scratch/factor_analysis/purity.jsonl")
purity_records = []
for factor_data in purity_results:
    fi = factor_data["factor_index"]
    for polarity, entries in [("HIGH", factor_data["top"]), ("LOW", factor_data["bottom"])]:
        label = f"Factor {fi:03d} — {polarity} (purity)"
        for rank, entry in enumerate(entries):
            purity_records.append({
                "question": label,
                "response_index": rank,
                "purity_score": round(entry["purity_score"], 4),
                "target_factor_score": round(entry["target_factor_score"], 4),
                "other_factors_mean_abs": round(entry["other_factors_mean_abs"], 4),
                "prompt": entry["seed_user_message"],
                "response": entry["text_excerpt"],
            })

with open(purity_out_path, "w") as f:
    for r in purity_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Saved {len(purity_records)} purity records to {purity_out_path}")
print(f"\nuv run python scripts/jsonl_tui/cli.py {purity_out_path} --variant-fields question response_index purity_score target_factor_score other_factors_mean_abs prompt response")

_purity_labels_path = _fa_cache.with_name(_fa_cache.name + "_purity_labels.json")
if _purity_labels_path.exists():
    with open(_purity_labels_path) as f:
        purity_labels = json.load(f)
    print(f"Loaded {len(purity_labels)} purity labels from {_purity_labels_path}")
elif LABEL_FACTORS:
    from dotenv import load_dotenv
    load_dotenv()
    purity_labels = label_factors(purity_results, model=LABELLER_MODEL, provider=LABELLER_PROVIDER, top_n=10)
    with open(_purity_labels_path, "w") as f:
        json.dump(purity_labels, f, indent=2)
    print(f"Saved purity labels to {_purity_labels_path}")
else:
    purity_labels = None

# %%
# Corpus nearest-neighbor: back-project each factor direction analytically,
# find the closest real responses in embedding space.
# HIGH = responses near global_mean + scale * direction
# LOW  = responses near global_mean - scale * direction
CNN_SCALE = 3.0

cnn_results = []
cnn_records = []
for fi in range(N_FACTORS):
    target_high, direction = analytical_factor_embedding(fi, loadings, pca_model=pca_model, scaler=scaler, global_mean=global_mean, scale=CNN_SCALE)
    target_low, _ = analytical_factor_embedding(fi, loadings, pca_model=pca_model, scaler=scaler, global_mean=global_mean, scale=-CNN_SCALE)
    top = corpus_nearest_neighbor(target_high, corpus_embeddings, metadata, top_k=20, excerpt_length=100000)
    bottom = corpus_nearest_neighbor(target_low, corpus_embeddings, metadata, top_k=20, excerpt_length=100000)
    cnn_results.append({"factor_index": fi, "top": top, "bottom": bottom})

    for polarity, entries in [("HIGH", top), ("LOW", bottom)]:
        label = f"Factor {fi:03d} — {polarity} (CNN)"
        for rank, entry in enumerate(entries):
            cnn_records.append({
                "question": label,
                "response_index": rank,
                "similarity": round(entry["similarity"], 4),
                "prompt": entry["seed_user_message"],
                "response": entry["text_excerpt"],
            })

cnn_out_path = Path("scratch/factor_analysis/cnn.jsonl")
with open(cnn_out_path, "w") as f:
    for r in cnn_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Saved {len(cnn_records)} CNN records to {cnn_out_path}")
print(f"\nuv run python scripts/jsonl_tui/cli.py {cnn_out_path} --variant-fields question response_index similarity prompt response")

_cnn_labels_path = _fa_cache.with_name(_fa_cache.name + "_cnn_labels.json")
if _cnn_labels_path.exists():
    with open(_cnn_labels_path) as f:
        cnn_labels = json.load(f)
    print(f"Loaded {len(cnn_labels)} CNN labels from {_cnn_labels_path}")
elif LABEL_FACTORS:
    from dotenv import load_dotenv
    load_dotenv()
    cnn_labels = label_factors(cnn_results, model=LABELLER_MODEL, provider=LABELLER_PROVIDER, top_n=10)
    with open(_cnn_labels_path, "w") as f:
        json.dump(cnn_labels, f, indent=2)
    print(f"Saved CNN labels to {_cnn_labels_path}")
else:
    cnn_labels = None

# %%
# Summary: all three label sets side by side
_all_labels = [
    ("Extremes", factor_labels),
    ("Purity",   purity_labels),
    ("CNN",      cnn_labels),
]
_available = [(name, lbls) for name, lbls in _all_labels if lbls]
if _available:
    header = f"{'Factor':<12}" + "".join(f"{name:<50}" for name, _ in _available)
    print(header)
    print("-" * len(header))
    for fi in range(N_FACTORS):
        row = f"Factor {fi:03d}  " + "".join(
            f"{lbls[fi][:48]:<50}" for _, lbls in _available
        )
        print(row)
# %%
