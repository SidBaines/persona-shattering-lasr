# %%
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from scripts.factor_analysis.preprocessing import load_embeddings, deduplicate_by_group, residualize, pca_reduce
from scripts.factor_analysis.parallel_analysis import parallel_analysis
from scripts.factor_analysis.factor_analysis import run_factor_analysis, adequacy_tests
from scripts.factor_analysis.persistence import save_factor_analysis, load_factor_analysis
from scripts.factor_analysis.labelling import label_factors, DEFAULT_MODEL as LABELLER_DEFAULT_MODEL
from scripts.factor_analysis.interpretation import (
    factor_extremes, rank_by_factor_purity, rank_prompts_by_max_spread,
    analytical_factor_embedding, corpus_nearest_neighbor,
    contrastive_factor_retrieval,
    optimize_factor_embedding, prompt_effects,
)
from scripts.jsonl_tui.html_export import export_html

# %%
# --- Config ---
USE_PCA = False   # False: run directly on 2560d residuals (slow but no lossy reduction)
SCALE = False
PCA_N_COMPONENTS = 100
N_FACTORS = 20
LABEL_FACTORS = True   # Call LLM to generate a description for each factor

if 1:
    LABELLER_MODEL = 'gpt-5-mini-2025-08-07'
    LABELLER_PROVIDER = 'openai'
    BASE_OUTPUT_DIR = "scratch/factor_analysis5_gpt5mini"
else:
    LABELLER_MODEL = 'claude-haiku-4-5-20251001'
    LABELLER_PROVIDER = 'anthropic'
    BASE_OUTPUT_DIR = "scratch/factor_analysis4_claudehaiku"

LABELLER_PROMPT_FORMAT = "grouped_json"  # or "contrastive_jsonl"
USE_NEW_DATASET = False
RUN_EXTREMES = False
RUN_PURITY = False
RUN_MAX_SPREAD = True
RUN_CNN = True
RUN_CONTRASTIVE = True
RUN_PROMPT_PREVIEW = True
RUN_SHARE_BUNDLE = True

OLD_DATASET = {
    "local_embeddings": "qwen4embeddings/stage123-240x50-singleturn-v2/response_embeddings_embeddings.npy",
    "local_metadata": "qwen4embeddings/stage123-240x50-singleturn-v2/response_embeddings_metadata.jsonl",
    "hf_repo_id": "qwen4embeddings/stage123-240x50-singleturn-v2",
    "hf_embeddings": "response_embeddings_embeddings.npy",
    "hf_metadata": "response_embeddings_metadata.jsonl",
}

NEW_DATASET = {
    "local_embeddings": "scratch/runs/stage123-240x50-singleturn-AAextension-emb-bs32/reports/response_embeddings_qwen3-embedding-4b_embeddings.npy",
    "local_metadata": "scratch/runs/stage123-240x50-singleturn-AAextension-emb-bs32/reports/response_embeddings_qwen3-embedding-4b_metadata.jsonl",
    "hf_repo_id": "persona-shattering-lasr/stage123-240x50-singleturn-AAextension",
    "hf_embeddings": "embeddings/Qwen-Qwen3-Embedding-4B/stage123-240x50-singleturn-AAextension-emb-bs32/response_embeddings_qwen3-embedding-4b_embeddings.npy",
    "hf_metadata": "embeddings/Qwen-Qwen3-Embedding-4B/stage123-240x50-singleturn-AAextension-emb-bs32/response_embeddings_qwen3-embedding-4b_metadata.jsonl",
}


def _resolve_dataset_paths(use_new_dataset: bool) -> tuple[Path, Path]:
    dataset_cfg = NEW_DATASET if use_new_dataset else OLD_DATASET

    embeddings_path = Path(dataset_cfg["local_embeddings"])
    metadata_path = Path(dataset_cfg["local_metadata"])
    if embeddings_path.exists() and metadata_path.exists():
        return embeddings_path, metadata_path

    repo_id = dataset_cfg.get("hf_repo_id")
    hf_embeddings = dataset_cfg.get("hf_embeddings")
    hf_metadata = dataset_cfg.get("hf_metadata")
    if not repo_id or not hf_embeddings or not hf_metadata:
        raise FileNotFoundError(
            "Local embeddings not found and HF fallback is not configured. "
            f"Missing files: {embeddings_path} / {metadata_path}"
        )

    from huggingface_hub import hf_hub_download

    embeddings_local = Path(
        hf_hub_download(
            repo_id=str(repo_id),
            repo_type="dataset",
            filename=str(hf_embeddings),
        )
    )
    metadata_local = Path(
        hf_hub_download(
            repo_id=str(repo_id),
            repo_type="dataset",
            filename=str(hf_metadata),
        )
    )
    print(f"Downloaded dataset artifacts from HF repo: {repo_id}")
    return embeddings_local, metadata_local


EMBEDDINGS_PATH, METADATA_PATH = _resolve_dataset_paths(USE_NEW_DATASET)
print(f"Using embeddings: {EMBEDDINGS_PATH}")
print(f"Using metadata: {METADATA_PATH}")

# %%
# Load and preprocess
embeddings, metadata = load_embeddings(EMBEDDINGS_PATH, METADATA_PATH)
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
_fa_cache = Path(f"{BASE_OUTPUT_DIR}/fa_n{N_FACTORS}_{FA_METHOD}_{FA_ROTATION}_filtered")
Path(BASE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

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

factor_labels = None
purity_labels = None
max_spread_labels = None
cnn_labels = None
contrastive_labels = None

out_path = None
purity_out_path = None
max_spread_out_path = None
cnn_out_path = None
contrastive_out_path = None

_labels_path = _fa_cache.with_name(_fa_cache.name + "_labels.json")
_purity_labels_path = _fa_cache.with_name(_fa_cache.name + "_purity_labels.json")
_max_spread_labels_path = _fa_cache.with_name(_fa_cache.name + "_max_spread_labels.json")
_cnn_labels_path = _fa_cache.with_name(_fa_cache.name + "_cnn_labels.json")
_contrastive_labels_path = _fa_cache.with_name(_fa_cache.name + "_contrastive_labels.json")

preview_source = None
_max_spread_for_labelling = None

# %%
if RUN_EXTREMES:
    extremes = factor_extremes(scores, metadata, top_n=20, excerpt_length=100000)
    if preview_source is None:
        preview_source = extremes

    # Label factors with LLM (or load cached labels)
    import json

    if _labels_path.exists():
        with open(_labels_path) as f:
            factor_labels = json.load(f)
        print(f"Loaded {len(factor_labels)} factor labels from {_labels_path}")
    elif LABEL_FACTORS:
        from dotenv import load_dotenv
        load_dotenv()
        factor_labels = label_factors(
            extremes,
            model=LABELLER_MODEL,
            provider=LABELLER_PROVIDER,
            top_n=10,
            max_per_prompt=100,
            prompt_format=LABELLER_PROMPT_FORMAT,
            excerpt_chars=10000,
        )
        with open(_labels_path, "w") as f:
            json.dump(factor_labels, f, indent=2)
        print(f"Saved factor labels to {_labels_path}")

    if factor_labels:
        for i, label in enumerate(factor_labels):
            print(f"Factor {i:03d}: {label}")

    out_path = Path(f"{BASE_OUTPUT_DIR}/extremes.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for factor_data in extremes:
        fi = factor_data["factor_index"]
        fl = factor_labels[fi] if factor_labels else ""
        for polarity, entries in [("HIGH", factor_data["top"]), ("LOW", factor_data["bottom"])]:
            label = f"Factor {fi:03d} — {polarity}"
            for rank, entry in enumerate(entries):
                records.append({
                    "question": label,
                    "response_index": rank,
                    "factor_label": fl,
                    "factor_score": round(entry["score"], 4),
                    "prompt": entry["seed_user_message"],
                    "response": entry["text_excerpt"],
                })

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} records to {out_path}")
    print(f"\nuv run python scripts/jsonl_tui/cli.py {out_path} --variant-fields question response_index factor_label factor_score prompt response")
    _html = export_html(out_path, ["question", "response_index", "factor_label", "factor_score", "prompt", "response"])
    print(f"HTML viewer: {_html}")

# %%
if RUN_PURITY:
    # Purity-based ranking: high-purity examples at both ends of the factor
    # (HIGH = high target, LOW = low target), while keeping other-factor
    # activations small.
    # Useful when factors aren't orthogonal (e.g. promax) — extremes alone can reflect
    # correlated factors, purity isolates each one more cleanly.
    purity_results = [
        rank_by_factor_purity(scores, metadata, factor_idx=fi, top_n=20, excerpt_length=100000)
        for fi in range(N_FACTORS)
    ]
    if preview_source is None:
        preview_source = purity_results

    if _purity_labels_path.exists():
        with open(_purity_labels_path) as f:
            purity_labels = json.load(f)
        print(f"Loaded {len(purity_labels)} purity labels from {_purity_labels_path}")
    elif LABEL_FACTORS:
        from dotenv import load_dotenv
        load_dotenv()
        purity_labels = label_factors(
            purity_results,
            model=LABELLER_MODEL,
            provider=LABELLER_PROVIDER,
            top_n=10,
            max_per_prompt=100,
            prompt_format=LABELLER_PROMPT_FORMAT,
            excerpt_chars=10000,
        )
        with open(_purity_labels_path, "w") as f:
            json.dump(purity_labels, f, indent=2)
        print(f"Saved purity labels to {_purity_labels_path}")

    purity_out_path = Path(f"{BASE_OUTPUT_DIR}/purity.jsonl")
    purity_records = []
    for factor_data in purity_results:
        fi = factor_data["factor_index"]
        fl = purity_labels[fi] if purity_labels else ""
        for polarity, entries in [("HIGH", factor_data["top"]), ("LOW", factor_data["bottom"])]:
            label = f"Factor {fi:03d} — {polarity} (purity)"
            for rank, entry in enumerate(entries):
                purity_records.append({
                    "question": label,
                    "response_index": rank,
                    "factor_label": fl,
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
    print(f"\nuv run python scripts/jsonl_tui/cli.py {purity_out_path} --variant-fields question response_index factor_label purity_score target_factor_score other_factors_mean_abs prompt response")
    _html = export_html(purity_out_path, ["question", "response_index", "factor_label", "purity_score", "target_factor_score", "other_factors_mean_abs", "prompt", "response"])
    print(f"HTML viewer: {_html}")

# %%
if RUN_MAX_SPREAD:
    # Max-spread: for each factor, find the questions where the target factor score
    # varies most across responses to that question, then show the highest- and
    # lowest-score response side by side.
    max_spread_results = [
        rank_prompts_by_max_spread(scores, metadata, factor_idx=fi, top_n=20, excerpt_length=100000)
        for fi in range(N_FACTORS)
    ]

    # Reshape into top/bottom lists for label_factors: high responses -> top, low -> bottom
    _max_spread_for_labelling = [
        {
            "factor_index": fd["factor_index"],
            "top": [gs["high"] for gs in fd["groups"]],
            "bottom": [gs["low"] for gs in fd["groups"]],
        }
        for fd in max_spread_results
    ]
    if preview_source is None:
        preview_source = _max_spread_for_labelling

    if _max_spread_labels_path.exists():
        with open(_max_spread_labels_path) as f:
            max_spread_labels = json.load(f)
        print(f"Loaded {len(max_spread_labels)} max-spread labels from {_max_spread_labels_path}")
    elif LABEL_FACTORS:
        from dotenv import load_dotenv
        load_dotenv()
        max_spread_labels = label_factors(
            _max_spread_for_labelling,
            model=LABELLER_MODEL,
            provider=LABELLER_PROVIDER,
            top_n=10,
            max_per_prompt=100,
            prompt_format=LABELLER_PROMPT_FORMAT,
            excerpt_chars=10000,
        )
        with open(_max_spread_labels_path, "w") as f:
            json.dump(max_spread_labels, f, indent=2)
        print(f"Saved max-spread labels to {_max_spread_labels_path}")

    max_spread_out_path = Path(f"{BASE_OUTPUT_DIR}/max_spread.jsonl")
    max_spread_records = []
    for factor_data in max_spread_results:
        fi = factor_data["factor_index"]
        fl = max_spread_labels[fi] if max_spread_labels else ""
        for polarity, entry_key in [("HIGH", "high"), ("LOW", "low")]:
            q_label = f"Factor {fi:03d} — {polarity} (max-spread)"
            for rank, gs in enumerate(factor_data["groups"]):
                entry = gs[entry_key]
                max_spread_records.append({
                    "question": q_label,
                    "response_index": rank,
                    "factor_label": fl,
                    "max_spread": round(gs["max_spread"], 4),
                    "purity_score": round(entry["purity_score"], 4),
                    "target_factor_score": round(entry["target_factor_score"], 4),
                    "other_factors_mean_abs": round(entry["other_factors_mean_abs"], 4),
                    "prompt": entry["seed_user_message"],
                    "response": entry["text_excerpt"],
                })

    with open(max_spread_out_path, "w") as f:
        for r in max_spread_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(max_spread_records)} max-spread records to {max_spread_out_path}")
    print(f"\nuv run python scripts/jsonl_tui/cli.py {max_spread_out_path} --variant-fields question response_index factor_label max_spread purity_score target_factor_score other_factors_mean_abs prompt response")
    _html = export_html(max_spread_out_path, ["question", "response_index", "factor_label", "max_spread", "purity_score", "target_factor_score", "other_factors_mean_abs", "prompt", "response"])
    print(f"HTML viewer: {_html}")

# %%
CNN_SCALE = 3.0
CONTRASTIVE_TOP_K = 100
CONTRASTIVE_SCALE = 3.0
CONTRASTIVE_NORMALIZE = True
CONTRASTIVE_EXCERPT_LENGTH = 100000
CONTRASTIVE_NEIGHBOR_K = 20
if RUN_CNN:
    # Corpus nearest-neighbor: back-project each factor direction analytically,
    # find the closest real responses in embedding space.
    # HIGH = responses near global_mean + scale * direction
    # LOW  = responses near global_mean - scale * direction
    cnn_results = []
    for fi in range(N_FACTORS):
        target_high, direction = analytical_factor_embedding(fi, loadings, pca_model=pca_model, scaler=scaler, global_mean=global_mean, scale=CNN_SCALE)
        target_low, _ = analytical_factor_embedding(fi, loadings, pca_model=pca_model, scaler=scaler, global_mean=global_mean, scale=-CNN_SCALE)
        top = corpus_nearest_neighbor(target_high, corpus_embeddings, metadata, top_k=20, excerpt_length=100000)
        bottom = corpus_nearest_neighbor(target_low, corpus_embeddings, metadata, top_k=20, excerpt_length=100000)
        cnn_results.append({"factor_index": fi, "top": top, "bottom": bottom})
    if preview_source is None:
        preview_source = cnn_results

    if _cnn_labels_path.exists():
        with open(_cnn_labels_path) as f:
            cnn_labels = json.load(f)
        print(f"Loaded {len(cnn_labels)} CNN labels from {_cnn_labels_path}")
    elif LABEL_FACTORS:
        from dotenv import load_dotenv
        load_dotenv()
        cnn_labels = label_factors(
            cnn_results,
            model=LABELLER_MODEL,
            provider=LABELLER_PROVIDER,
            top_n=10,
            max_per_prompt=100,
            prompt_format=LABELLER_PROMPT_FORMAT,
            excerpt_chars=10000,
        )
        with open(_cnn_labels_path, "w") as f:
            json.dump(cnn_labels, f, indent=2)
        print(f"Saved CNN labels to {_cnn_labels_path}")

    cnn_records = []
    for factor_data in cnn_results:
        fi = factor_data["factor_index"]
        fl = cnn_labels[fi] if cnn_labels else ""
        for polarity, entries in [("HIGH", factor_data["top"]), ("LOW", factor_data["bottom"])]:
            label = f"Factor {fi:03d} — {polarity} (CNN)"
            for rank, entry in enumerate(entries):
                cnn_records.append({
                    "question": label,
                    "response_index": rank,
                    "factor_label": fl,
                    "similarity": round(entry["similarity"], 4),
                    "prompt": entry["seed_user_message"],
                    "response": entry["text_excerpt"],
                })

    cnn_out_path = Path(f"{BASE_OUTPUT_DIR}/cnn.jsonl")
    with open(cnn_out_path, "w") as f:
        for r in cnn_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(cnn_records)} CNN records to {cnn_out_path}")
    print(f"\nuv run python scripts/jsonl_tui/cli.py {cnn_out_path} --variant-fields question response_index factor_label similarity prompt response")
    _html = export_html(cnn_out_path, ["question", "factor_label", "similarity", "prompt", "response"])
    print(f"HTML viewer: {_html}")

# %%
if RUN_CONTRASTIVE:
    # Contrastive centroid retrieval: compute a factor-specific direction by subtracting
    # the mean embedding of low-scoring responses from the mean embedding of high-scoring
    # responses, then retrieve the nearest real responses to the resulting high/low targets.
    contrastive_results = [
        contrastive_factor_retrieval(
            scores, fi, corpus_embeddings, metadata, global_mean,
            top_k=CONTRASTIVE_TOP_K,
            neighbor_k=CONTRASTIVE_NEIGHBOR_K,
            scale=CONTRASTIVE_SCALE,
            normalize=CONTRASTIVE_NORMALIZE,
            excerpt_length=CONTRASTIVE_EXCERPT_LENGTH,
        )
        for fi in range(N_FACTORS)
    ]
    if preview_source is None:
        preview_source = contrastive_results

    if _contrastive_labels_path.exists():
        with open(_contrastive_labels_path) as f:
            contrastive_labels = json.load(f)
        print(f"Loaded {len(contrastive_labels)} contrastive labels from {_contrastive_labels_path}")
    elif LABEL_FACTORS:
        from dotenv import load_dotenv
        load_dotenv()
        contrastive_labels = label_factors(
            contrastive_results,
            model=LABELLER_MODEL,
            provider=LABELLER_PROVIDER,
            top_n=10,
            max_per_prompt=100,
            prompt_format=LABELLER_PROMPT_FORMAT,
            excerpt_chars=10000,
        )
        with open(_contrastive_labels_path, "w") as f:
            json.dump(contrastive_labels, f, indent=2)
        print(f"Saved contrastive labels to {_contrastive_labels_path}")

    contrastive_records = []
    for factor_data in contrastive_results:
        fi = factor_data["factor_index"]
        fl = contrastive_labels[fi] if contrastive_labels else ""
        raw_norm = round(factor_data["raw_direction_norm"], 6)
        for polarity, entries in [("HIGH", factor_data["top"]), ("LOW", factor_data["bottom"])]:
            label = f"Factor {fi:03d} — {polarity} (contrastive)"
            for rank, entry in enumerate(entries):
                contrastive_records.append({
                    "question": label,
                    "response_index": rank,
                    "factor_label": fl,
                    "similarity": round(entry["similarity"], 4),
                    "contrastive_top_k": factor_data["top_k"],
                    "contrastive_scale": factor_data["scale"],
                    "contrastive_normalize": factor_data["normalize"],
                    "raw_direction_norm": raw_norm,
                    "prompt": entry["seed_user_message"],
                    "response": entry["text_excerpt"],
                })

    contrastive_out_path = Path(f"{BASE_OUTPUT_DIR}/contrastive.jsonl")
    with open(contrastive_out_path, "w") as f:
        for r in contrastive_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(contrastive_records)} contrastive records to {contrastive_out_path}")
    print(f"\nuv run python scripts/jsonl_tui/cli.py {contrastive_out_path} --variant-fields question response_index factor_label similarity contrastive_top_k contrastive_scale raw_direction_norm prompt response")
    _html = export_html(contrastive_out_path, ["question", "factor_label", "similarity", "contrastive_top_k", "contrastive_scale", "raw_direction_norm", "prompt", "response"])
    print(f"HTML viewer: {_html}")

# %%
# Preview the exact prompt sent to the labeller for the first available method
from scripts.factor_analysis.labelling import _build_messages

_example_messages = None
if RUN_PROMPT_PREVIEW and preview_source:
    _example_messages = _build_messages(
        preview_source[0],
        top_n=10,
        excerpt_chars=400,
        prompt_format=LABELLER_PROMPT_FORMAT,
    )
    print("=== SYSTEM PROMPT ===")
    print(_example_messages[0]["content"])
    print("\n=== USER PROMPT ===")
    print(_example_messages[1]["content"])

# %%
# Summary: all available label sets side by side
_all_labels = [
    ("Extremes",       factor_labels),
    ("Purity",         purity_labels),
    ("Max-spread",     max_spread_labels),
    ("CNN",            cnn_labels),
    ("Contrastive",    contrastive_labels),
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
# Export share bundle: HTMLs + label JSONs + README → single zip
import zipfile
import datetime

if RUN_SHARE_BUNDLE:
    # Build README text
    _example_sys = _example_messages[0]["content"] if _example_messages else "(prompt preview not run)"
    _example_user = _example_messages[1]["content"] if _example_messages else "(prompt preview not run)"

    _orig_count = len(metadata) + _n_removed + locals().get("_n_singleton", 0)
    _n_prompts = len(set(str(r.get('input_group_id', '')) for r in metadata))

    _readme = f"""\
# Factor Analysis — Share Bundle
Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}

## TL;DR
Flick through the HTML viewers to see top responses (based on different clustering methods) and their LLM-generated labels.

## Method

Embeddings were computed for {_orig_count} LLM responses, then filtered to remove
responses shorter than {MIN_RESPONSE_CHARS} characters (likely refusals/deflections),
leaving {len(metadata)} responses across {_n_prompts} prompts.
Per-prompt mean embeddings were subtracted (residualisation) to remove prompt-content
variance, leaving only variance due to response style/behaviour.

Factor analysis was then run on the residuals:
  - Method:   {FA_METHOD}
  - Rotation: {FA_ROTATION}
  - Factors:  {N_FACTORS}
  - PCA pre-reduction: {USE_PCA} {"(n_components=" + str(PCA_N_COMPONENTS) + ")" if USE_PCA else ""}

Five methods were used to find representative responses for each factor:

  extremes     — responses with the highest/lowest raw factor scores

  purity       — responses selected separately for HIGH/LOW factor polarity, with
                 strong target score in that polarity and low activation on other factors
                 (useful when factors are correlated, e.g. with promax rotation)

  max_spread   — questions whose responses have the widest target-factor-score spread,
                 shown as highest-score vs lowest-score response for the same question

  CNN          — corpus nearest-neighbour: analytically back-projects the factor direction
                 into embedding space and finds the closest real responses

  contrastive  — computes a factor-specific direction by subtracting the mean embedding of
                 low-scoring responses from the mean embedding of high-scoring responses,
                 then retrieves the nearest real responses to the resulting high/low targets
                 in corpus space

Each factor was labelled by an LLM ({LABELLER_MODEL}) shown the top-{10} high/low
examples and asked to describe what distinguishes them.

## Files

  extremes.html          — browse factor extremes (raw scores); ↑↓ = factors, ←→ = responses

  purity.html            — browse purity-ranked responses

  max_spread.html        — browse questions with the widest target-factor-score spread;
                           ↑↓ = factor×question, ←→ = HIGH vs LOW response for that question

  cnn.html               — browse corpus nearest-neighbour responses

  contrastive.html       — browse contrastive centroid retrieval responses

  *_labels.json          — raw LLM label strings, one per factor, for each method

## Labeller prompt (example: CNN factor 0)

### System
{_example_sys}

### User
{_example_user}
"""

    _bundle_files = []
    for p in [out_path, purity_out_path, max_spread_out_path, cnn_out_path, contrastive_out_path]:
        if p is not None:
            _bundle_files.append(Path(p).with_suffix(".html"))
    for p in [_labels_path, _purity_labels_path, _max_spread_labels_path, _cnn_labels_path, _contrastive_labels_path]:
        if Path(p).exists():
            _bundle_files.append(Path(p))

    _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _zip_path = Path(f"{BASE_OUTPUT_DIR}/share_{_ts}.zip")
    with zipfile.ZipFile(_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.md", _readme)
        for p in _bundle_files:
            if Path(p).exists():
                zf.write(p, Path(p).name)
            else:
                print(f"  skipped (not found): {p}")

    print(f"Bundle written to {_zip_path}")
    print("Contents:")
    with zipfile.ZipFile(_zip_path) as zf:
        for info in zf.infolist():
            print(f"  {info.filename}  ({info.file_size:,} bytes)")
# %%
# TEMP: preview max-spread labeller prompts for first 3 factors
_N_PREVIEW = 3
if RUN_MAX_SPREAD and _max_spread_for_labelling:
    for _fd in _max_spread_for_labelling[:_N_PREVIEW]:
        _msgs = _build_messages(
            _fd,
            top_n=10,
            excerpt_chars=400,
            prompt_format=LABELLER_PROMPT_FORMAT,
        )
        print(f"\n{'='*60}")
        print(f"=== FACTOR {_fd['factor_index']} — SYSTEM ===")
        print(_msgs[0]["content"])
        print(f"\n=== FACTOR {_fd['factor_index']} — USER ===")
        print(_msgs[1]["content"])
# %%
