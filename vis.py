# %%
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from scripts.factor_analysis.preprocessing import load_embeddings, deduplicate_by_group, residualize, pca_reduce
from scripts.factor_analysis.parallel_analysis import parallel_analysis
from scripts.factor_analysis.factor_analysis import run_factor_analysis, adequacy_tests
from scripts.factor_analysis.persistence import save_factor_analysis, load_factor_analysis
from scripts.factor_analysis.labelling import (
    label_factors,
    load_label_checkpoint,
    label_is_complete,
    DEFAULT_MODEL as LABELLER_DEFAULT_MODEL,
)
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
N_FACTORS = 30
LABEL_FACTORS = True   # Call LLM to generate a description for each factor

if 1:
    LABELLER_MODEL = 'gpt-5-mini-2025-08-07'
    LABELLER_PROVIDER = 'openai'
    BASE_OUTPUT_DIR = "scratch/factor_analysis10_gpt5mini_old_dataset"
else:
    LABELLER_MODEL = 'claude-haiku-4-5-20251001'
    LABELLER_PROVIDER = 'anthropic'
    BASE_OUTPUT_DIR = "scratch/factor_analysis4_claudehaiku"

LABELLER_PROMPT_FORMAT = "contrastive_jsonl"  # "grouped_json" or "contrastive_jsonl"
# Dataset selection: "old", "new", or "both"
DATASET_MODE = "old"
RUN_EXTREMES = False
RUN_PURITY = False
RUN_MAX_SPREAD = True
RUN_CNN = False
RUN_CONTRASTIVE = False
RUN_PROMPT_PREVIEW = True
RUN_SHARE_BUNDLE = True

Path(BASE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

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


def _resolve_dataset_paths(dataset_cfg: dict[str, str]) -> tuple[Path, Path]:

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


def _load_selected_datasets(dataset_mode: str) -> tuple[np.ndarray, list[dict]]:
    mode = dataset_mode.lower().strip()
    if mode not in {"old", "new", "both"}:
        raise ValueError(f"Invalid DATASET_MODE='{dataset_mode}'. Expected one of: 'old', 'new', 'both'.")

    if mode in {"old", "new"}:
        dataset_cfg = OLD_DATASET if mode == "old" else NEW_DATASET
        embeddings_path, metadata_path = _resolve_dataset_paths(dataset_cfg)
        print(f"Using dataset: {mode}")
        print(f"Using embeddings: {embeddings_path}")
        print(f"Using metadata: {metadata_path}")
        return load_embeddings(embeddings_path, metadata_path)

    # mode == "both": load, then concatenate. Prefix group ids by source to avoid collisions.
    old_embeddings_path, old_metadata_path = _resolve_dataset_paths(OLD_DATASET)
    new_embeddings_path, new_metadata_path = _resolve_dataset_paths(NEW_DATASET)
    print("Using dataset: both")
    print(f"Using OLD embeddings: {old_embeddings_path}")
    print(f"Using OLD metadata: {old_metadata_path}")
    print(f"Using NEW embeddings: {new_embeddings_path}")
    print(f"Using NEW metadata: {new_metadata_path}")

    old_embeddings, old_metadata = load_embeddings(old_embeddings_path, old_metadata_path)
    new_embeddings, new_metadata = load_embeddings(new_embeddings_path, new_metadata_path)

    for row in old_metadata:
        original_group = str(row.get("input_group_id", ""))
        row["dataset_source"] = "old"
        row["input_group_id"] = f"old::{original_group}"
    for row in new_metadata:
        original_group = str(row.get("input_group_id", ""))
        row["dataset_source"] = "new"
        row["input_group_id"] = f"new::{original_group}"

    combined_embeddings = np.concatenate([old_embeddings, new_embeddings], axis=0)
    combined_metadata = old_metadata + new_metadata
    print(
        "Combined datasets: "
        f"{len(old_metadata)} old + {len(new_metadata)} new = {len(combined_metadata)} total rows"
    )
    return combined_embeddings, combined_metadata


_plot_paths: list[Path] = []


def _save_plot_html(fig, filename: str) -> Path:
    path = Path(BASE_OUTPUT_DIR) / filename
    fig.write_html(path, include_plotlyjs="include")
    _plot_paths.append(path)
    print(f"Saved plot: {path}")
    return path

# %%
# Load and preprocess
embeddings, metadata = _load_selected_datasets(DATASET_MODE)
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
_save_plot_html(fig, "responses_per_prompt_after_filtering.html")
fig.show()

corpus_embeddings = embeddings.copy()  # keep for nearest-neighbor lookups later
residuals, group_means, group_inv = residualize(embeddings, metadata)
if 1: 
    residuals = embeddings
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
    _save_plot_html(fig, "parallel_analysis_scree.html")
    fig.show()

# %%
# Run factor analysis (or load cached result)
FA_METHOD = "principal"
FA_ROTATION = "varimax"
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
_save_plot_html(fig, "factor_loadings.html")
fig.show()

# And a scatter of factor score variance (how much each factor spreads the data):
import plotly.graph_objects as go
fig = go.Figure(go.Bar(y=fa["proportion_variance"], x=list(range(N_FACTORS))))
fig.update_layout(title="Variance explained per factor", xaxis_title="Factor", yaxis_title="Proportion variance")
_save_plot_html(fig, "variance_explained_per_factor.html")
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
_corpus_nearest_neighbour_labels_path = _fa_cache.with_name(
    _fa_cache.name + "_corpus_nearest_neighbour_labels.json"
)
_legacy_cnn_labels_path = _fa_cache.with_name(_fa_cache.name + "_cnn_labels.json")
_contrastive_labels_path = _fa_cache.with_name(_fa_cache.name + "_contrastive_labels.json")

preview_source = None
preview_source_name = None
_max_spread_for_labelling = None


def _labels_status(path: Path, expected_count: int) -> tuple[list[str] | None, bool]:
    if not path.exists():
        return None, False
    labels = load_label_checkpoint(path, expected_count)
    complete = len(labels) == expected_count and all(label_is_complete(label) for label in labels)
    return labels, complete


def _resume_or_load_labels(
    factor_data: list[dict],
    labels_path: Path,
    *,
    label_name: str,
    fallback_paths: list[Path] | None = None,
) -> list[str] | None:
    fallback_paths = fallback_paths or []
    labels, complete = _labels_status(labels_path, len(factor_data))
    if labels is not None and complete:
        print(f"Loaded {len(labels)} {label_name} labels from {labels_path}")
        return labels

    for fallback_path in fallback_paths:
        fallback_labels, fallback_complete = _labels_status(fallback_path, len(factor_data))
        if fallback_labels is None:
            continue
        if fallback_complete:
            with open(labels_path, "w", encoding="utf-8") as f:
                json.dump(fallback_labels, f, indent=2, ensure_ascii=False)
            print(
                f"Loaded {len(fallback_labels)} {label_name} labels from legacy cache "
                f"{fallback_path}; rewrote cache to {labels_path.name}"
            )
            return fallback_labels
        labels = fallback_labels
        break

    if not LABEL_FACTORS:
        if labels is not None:
            completed = sum(1 for label in labels if label_is_complete(label))
            print(
                f"Found incomplete {label_name} label cache at {labels_path} "
                f"({completed}/{len(labels)} complete), but LABEL_FACTORS=False so not resuming"
            )
        return labels

    from dotenv import load_dotenv

    load_dotenv()
    labels = label_factors(
        factor_data,
        model=LABELLER_MODEL,
        provider=LABELLER_PROVIDER,
        top_n=10,
        max_per_prompt=100,
        prompt_format=LABELLER_PROMPT_FORMAT,
        excerpt_chars=10000,
        checkpoint_path=labels_path,
    )
    print(f"Saved {label_name} labels to {labels_path}")
    return labels

# %%
if RUN_EXTREMES:
    extremes = factor_extremes(scores, metadata, top_n=20, excerpt_length=100000)
    if preview_source is None:
        preview_source = extremes
        preview_source_name = "extremes"

    factor_labels = _resume_or_load_labels(
        extremes,
        _labels_path,
        label_name="factor",
    )

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
        preview_source_name = "purity"

    purity_labels = _resume_or_load_labels(
        purity_results,
        _purity_labels_path,
        label_name="purity",
    )

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
        preview_source_name = "max_spread"

    max_spread_labels = _resume_or_load_labels(
        _max_spread_for_labelling,
        _max_spread_labels_path,
        label_name="max-spread",
    )

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
        preview_source_name = "corpus_nearest_neighbour"

    cnn_labels = _resume_or_load_labels(
        cnn_results,
        _corpus_nearest_neighbour_labels_path,
        label_name="corpus_nearest_neighbour",
        fallback_paths=[_legacy_cnn_labels_path],
    )

    cnn_records = []
    for factor_data in cnn_results:
        fi = factor_data["factor_index"]
        fl = cnn_labels[fi] if cnn_labels else ""
        for polarity, entries in [("HIGH", factor_data["top"]), ("LOW", factor_data["bottom"])]:
            label = f"Factor {fi:03d} — {polarity} (corpus_nearest_neighbour)"
            for rank, entry in enumerate(entries):
                cnn_records.append({
                    "question": label,
                    "response_index": rank,
                    "factor_label": fl,
                    "similarity": round(entry["similarity"], 4),
                    "prompt": entry["seed_user_message"],
                    "response": entry["text_excerpt"],
                })

    cnn_out_path = Path(f"{BASE_OUTPUT_DIR}/corpus_nearest_neighbour.jsonl")
    with open(cnn_out_path, "w") as f:
        for r in cnn_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(cnn_records)} corpus_nearest_neighbour records to {cnn_out_path}")
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
            scores, fi, residuals, metadata, residuals.mean(axis=0),
            top_k=CONTRASTIVE_TOP_K,
            neighbor_k=CONTRASTIVE_NEIGHBOR_K,
            scale=CONTRASTIVE_SCALE,
            normalize=CONTRASTIVE_NORMALIZE,
            embedding_space="residual",
            excerpt_length=CONTRASTIVE_EXCERPT_LENGTH,
        )
        for fi in range(N_FACTORS)
    ]
    if preview_source is None:
        preview_source = contrastive_results
        preview_source_name = "contrastive"

    contrastive_labels = _resume_or_load_labels(
        contrastive_results,
        _contrastive_labels_path,
        label_name="contrastive",
    )

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
                    "contrastive_embedding_space": factor_data["embedding_space"],
                    "raw_direction_norm": raw_norm,
                    "prompt": entry["seed_user_message"],
                    "response": entry["text_excerpt"],
                })

    contrastive_out_path = Path(f"{BASE_OUTPUT_DIR}/contrastive.jsonl")
    with open(contrastive_out_path, "w") as f:
        for r in contrastive_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(contrastive_records)} contrastive records to {contrastive_out_path}")
    print(f"\nuv run python scripts/jsonl_tui/cli.py {contrastive_out_path} --variant-fields question response_index factor_label similarity contrastive_top_k contrastive_scale contrastive_embedding_space raw_direction_norm prompt response")
    _html = export_html(contrastive_out_path, ["question", "factor_label", "similarity", "contrastive_top_k", "contrastive_scale", "contrastive_embedding_space", "raw_direction_norm", "prompt", "response"])
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
    ("corpus_nearest_neighbour", cnn_labels),
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
    _preview_label = preview_source_name or "preview"

    _readme = f"""\
# Factor Analysis — Share Bundle
Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}

## TL;DR
Flick through the HTML viewers to see representative responses selected by different factor-interpretation methods, along with their LLM-generated labels.

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
                 the highest positive purity score, where
                 purity = |target factor score| - mean absolute off-target factor score
                 (useful when factors are correlated, e.g. with promax rotation)

  max_spread   — questions whose responses have the widest target-factor-score spread,
                 shown as highest-score vs lowest-score response for the same question;
                 the reported purity score uses the same positive, polarity-agnostic
                 definition as the purity viewer

  corpus_nearest_neighbour
               — analytically back-projects the fitted factor-loading direction into
                 embedding space and finds the closest real responses in the corpus

  contrastive  — computes a factor-specific direction by subtracting the mean embedding of
                 low-scoring responses from the mean embedding of high-scoring responses
                 in residualized embedding space, then retrieves the nearest real responses
                 to the resulting high/low residual-space targets

Each factor was labelled by an LLM ({LABELLER_MODEL}) shown the top-{10} high/low
examples and asked to describe what distinguishes them.

## Files

  extremes.html          — browse factor-polarity groups of raw-score extremes;
                           ↑↓ = next/previous factor-polarity group, ←→ = responses

  purity.html            — browse factor-polarity groups of purity-ranked responses

  max_spread.html        — browse factor-polarity groups of highest-spread prompts;
                           ↑↓ = next/previous factor-polarity group, ←→ = ranked prompt groups
                           within that HIGH or LOW view

  corpus_nearest_neighbour.html
                         — browse corpus nearest-neighbour responses

  contrastive.html       — browse contrastive centroid retrieval responses

  responses_per_prompt_after_filtering.html
                         — histogram of retained responses per prompt after filtering

  factor_loadings.html   — heatmap of the factor loading matrix

  variance_explained_per_factor.html
                         — bar chart of variance explained by each factor

  parallel_analysis_scree.html
                         — scree plot from Horn's parallel analysis, if that block was run

  *_labels.json          — raw LLM label strings, one per factor, for each method

## Labeller prompt (example: {_preview_label} factor 0)

### System
{_example_sys}

### User
{_example_user}
"""

    _bundle_files = []
    for p in [out_path, purity_out_path, max_spread_out_path, cnn_out_path, contrastive_out_path]:
        if p is not None:
            _bundle_files.append(Path(p).with_suffix(".html"))
    for p in [
        _labels_path,
        _purity_labels_path,
        _max_spread_labels_path,
        _corpus_nearest_neighbour_labels_path,
        _contrastive_labels_path,
    ]:
        if Path(p).exists():
            _bundle_files.append(Path(p))
    for p in _plot_paths:
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
