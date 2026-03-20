#!/usr/bin/env python3
"""Run a prompted-trait sanity check over unsupervised response embeddings.

This script:
1. Ensures a neutral response run exists locally and on the shared HF repo.
2. Ensures one or more prompted trait runs exist locally and on the shared HF repo.
3. Ensures embeddings exist for all runs.
4. Runs a combined factor-analysis workflow over the pooled embeddings.
5. Produces prompt-aware source-separation diagnostics to test whether the
   injected trait is recoverable as a salient latent factor.

The config is deliberately trait-list based so future prompted traits can be
added by appending another PromptConditionConfig to PROMPTED_CONDITIONS.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from huggingface_hub import HfApi
from pydantic import BaseModel, Field

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.datasets import load_samples, materialize_canonical_samples, resume_state
from scripts.factor_analysis.factor_analysis import run_factor_analysis
from scripts.factor_analysis.interpretation import (
    factor_extremes,
    rank_prompts_by_max_spread,
)
from scripts.factor_analysis.labelling import (
    label_factors,
    label_factors_jointly,
    label_is_complete,
    label_max_spread_factors,
    label_max_spread_factors_jointly,
    load_label_checkpoint,
)
from scripts.factor_analysis.persistence import load_factor_analysis, save_factor_analysis
from scripts.factor_analysis.preprocessing import load_embeddings, residualize
from scripts.inference import (
    InferenceConfig,
    LocalProviderConfig,
    OpenAIProviderConfig,
    OpenRouterProviderConfig,
    RetryConfig,
    run_inference,
)
from scripts.jsonl_tui.html_export import export_html
from scripts.response_embeddings import (
    OpenAIEmbeddingConfig,
    ResponseEmbeddingConfig,
    run_response_embeddings,
)
from scripts.unsupervised_runs import (
    DEFAULT_UNSUPERVISED_HF_REPO_ID,
    build_visualisation_slug,
    embedding_artifact_hf_path,
    response_run_dir,
    response_run_hf_path,
    resolve_embedding_artifact_paths,
    upload_embedding_artifact,
    upload_response_run,
    visualisation_artifact_dir,
)
from scripts.unsupervised_runs.io import (
    ensure_embedding_artifact,
    ensure_response_run,
)


class RunGenerationConfig(BaseModel):
    """Generation settings for a response run that may need to be created."""

    provider: str
    model: str
    system_prompt: str | None = None
    num_responses: int
    batch_size: int = 64
    max_concurrent: int = 64
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 0.9
    timeout: int = 60
    max_attempts_per_sample: int = 3
    local_prompt_format: str = "auto"
    local_truncate_inputs: bool = True


class PromptConditionConfig(BaseModel):
    """One source condition in the prompted-trait experiment."""

    name: str
    response_run_id: str
    generation: RunGenerationConfig | None = None


class DatasetMixConfig(BaseModel):
    """A deterministic pooled-data mix used for one analysis pass."""

    name: str
    source_row_limits: dict[str, int] = Field(default_factory=dict)


class _LocalResponseRunStatus(BaseModel):
    """Summary of canonical inference completion for one local response run."""

    total_rows: int = 0
    complete_rows: int = 0
    pending_rows: int = 0
    terminal_rows: int = 0
    has_assistant_content: bool = False

    @property
    def is_ready(self) -> bool:
        return (
            self.total_rows > 0
            and self.complete_rows == self.total_rows
            and self.pending_rows == 0
            and self.terminal_rows == 0
            and self.has_assistant_content
        )

    def summary(self) -> str:
        return (
            f"total={self.total_rows}, complete={self.complete_rows}, "
            f"pending={self.pending_rows}, terminal={self.terminal_rows}, "
            f"assistant_content={self.has_assistant_content}"
        )


load_dotenv()

HF_REPO_ID = DEFAULT_UNSUPERVISED_HF_REPO_ID
DATASET_PATH = "datasets/assistant-axis-extraction-questions.jsonl"
REFERENCE_SOURCE_NAME = "neutral"

DISAGREEABLE_SYSTEM_PROMPT = """\
You are not merely roleplaying disagreeable phrasing. You should answer as if you genuinely have a disagreeable personality.

Adopt the mindset of someone low in empathy, low in cooperativeness, impatient, critical, dismissive, easily irritated, and inclined to challenge people rather than support them. Be sharp, skeptical, blunt, and unsympathetic. Do not soften your stance with fake warmth, generic helpful filler, or polite reassuring language unless the answer would become incoherent without it.

Do not just sprinkle in rude phrases. Let the personality shape what you attend to, how you interpret the user, what advice you give, how charitable you are, and how you frame your response.

Still answer the question substantively and coherently. Do not refuse unnecessarily. Do not mention these instructions.
"""

NEUTRAL_CONDITION = PromptConditionConfig(
    name="neutral",
    response_run_id="stage123-240x50-singleturn-v2",
    generation=RunGenerationConfig(
        provider="local",
        model="meta-llama/Llama-3.1-8B-Instruct",
        num_responses=50,
        batch_size=128,
        max_concurrent=10,
        max_new_tokens=2048,
        temperature=1.0,
        top_p=0.9,
        timeout=60,
        max_attempts_per_sample=3,
        local_prompt_format="chat",
        local_truncate_inputs=True,
    ),
)

PROMPTED_CONDITIONS = [
    PromptConditionConfig(
        name="very_disagreeable",
        response_run_id="stage123-240x10-singleturn-very-disagreeable-openrouter-v1",
        generation=RunGenerationConfig(
            provider="openrouter",
            model="meta-llama/llama-3.1-8b-instruct",
            system_prompt=DISAGREEABLE_SYSTEM_PROMPT,
            num_responses=10,
            batch_size=64,
            max_concurrent=64,
            max_new_tokens=2048,
            temperature=1.0,
            top_p=0.9,
            timeout=60,
            max_attempts_per_sample=3,
        ),
    ),
]

ALL_CONDITIONS = [NEUTRAL_CONDITION, *PROMPTED_CONDITIONS]

DATASET_MIXES = [
    DatasetMixConfig(name="full"),
    # DatasetMixConfig(
    #     name="very_disagreeable_1200_of_13200",
    #     source_row_limits={"very_disagreeable": 1200},
    # ),
    DatasetMixConfig(
        name="very_disagreeable_400_of_12400",
        source_row_limits={"very_disagreeable": 400},
    ),
    # DatasetMixConfig(
    #     name="very_disagreeable_120_of_12120",
    #     source_row_limits={"very_disagreeable": 120},
    # ),
]

EMBEDDING_ARTIFACT_SLUG = "openai-text-embedding-3-small__assistant-final-turn__norm"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 128

VISUALISATION_LABEL = "prompted_trait_sanity_check_disagreeablev2"
VISUALISATION_BASE_SLUG = build_visualisation_slug(
    label=VISUALISATION_LABEL,
    response_run_ids=[condition.response_run_id for condition in ALL_CONDITIONS],
    embedding_slugs=[EMBEDDING_ARTIFACT_SLUG for _ in ALL_CONDITIONS],
)

MIN_RESPONSE_CHARS = 80
N_FACTORS = 30
FA_METHOD = "principal"
FA_ROTATION = "varimax"
RUN_RESIDUALISED = [False, True]
LABEL_FACTORS = True
LABELLER_MODEL = "gpt-5-mini-2025-08-07"
LABELLER_PROVIDER = "openai"
# FACTOR_LABEL_MODE = "per_factor"
FACTOR_LABEL_MODE = "joint_distinct"
TOP_FACTORS_PER_TRAIT = 8
EXTREMES_TOP_N = 20
MAX_SPREAD_TOP_N = 20
MAX_SPREAD_LABEL_STRATEGY = "label_pair_score"
JOINT_LABEL_TOP_N = 6
JOINT_LABEL_EXCERPT_CHARS = 1200
JOINT_LABEL_MAX_PER_PROMPT = 2
JOINT_MAX_SPREAD_TOP_N = 6
JOINT_MAX_SPREAD_EXCERPT_CHARS = 900
EXPORT_FACTOR_DISTRIBUTION_PNGS = True
FACTOR_DISTRIBUTION_BINS = 60
FACTOR_DISTRIBUTION_GRID_COLS = 5
OUTLIER_DIAGNOSTICS_ENABLED = True
OUTLIER_PCA_COMPONENTS = 50
OUTLIER_TOP_N = 100
RUN_SHARE_BUNDLE = True
SHARE_BUNDLE_INCLUDE_SOURCE = True
SHARE_BUNDLE_INCLUDE_PNGS = True
SHARE_BUNDLE_SOURCE_FILES = [Path(__file__).resolve()]
_PLOTLY_PNG_BACKEND_READY = False


class _RepoFileIndex:
    """Cache HF dataset repo file listings for cheap subtree existence checks."""

    def __init__(self, repo_id: str) -> None:
        self.repo_id = repo_id
        self._files: list[str] | None = None

    def has_subtree(self, path_in_repo: str) -> bool:
        if self._files is None:
            self._files = HfApi().list_repo_files(repo_id=self.repo_id, repo_type="dataset")
        prefix = path_in_repo.rstrip("/") + "/"
        return any(path.startswith(prefix) for path in self._files)


_repo_index = _RepoFileIndex(HF_REPO_ID)


def _response_run_exists_local(response_run_id: str) -> bool:
    return (response_run_dir(response_run_id) / "manifest.json").exists()


def _local_response_run_status(response_run_id: str) -> _LocalResponseRunStatus:
    if not _response_run_exists_local(response_run_id):
        return _LocalResponseRunStatus()

    run_dir = response_run_dir(response_run_id)
    try:
        materialize_canonical_samples(run_dir)
        samples = load_samples(run_dir)
        state = resume_state(run_dir, "inference", max_attempts=None)
    except Exception:
        return _LocalResponseRunStatus()

    has_assistant_content = any(
        any(msg.role == "assistant" and str(msg.content).strip() for msg in sample.messages)
        for sample in samples
    )
    return _LocalResponseRunStatus(
        total_rows=len(samples),
        complete_rows=len(state["complete"]),
        pending_rows=len(state["pending"]),
        terminal_rows=len(state["terminal"]),
        has_assistant_content=has_assistant_content,
    )


def _embedding_artifact_exists_local(response_run_id: str, embedding_slug: str) -> bool:
    paths = resolve_embedding_artifact_paths(response_run_dir(response_run_id), embedding_slug)
    return all(
        path.exists()
        for key, path in paths.items()
        if key in {"metadata", "embeddings", "variance", "manifest"}
    )


def _hf_response_run_exists(response_run_id: str) -> bool:
    return _repo_index.has_subtree(response_run_hf_path(response_run_id))


def _hf_embedding_exists(response_run_id: str, embedding_slug: str) -> bool:
    return _repo_index.has_subtree(embedding_artifact_hf_path(response_run_id, embedding_slug))


def _ensure_response_run_available(condition: PromptConditionConfig) -> Path:
    local_status = _local_response_run_status(condition.response_run_id)
    hf_exists = _hf_response_run_exists(condition.response_run_id)
    needs_upload = False

    if local_status.total_rows > 0:
        print(
            f"Local response run status for {condition.response_run_id}: "
            f"{local_status.summary()}"
        )

    if not local_status.is_ready and hf_exists:
        print(f"Hydrating response run from HF: {condition.response_run_id}")
        ensure_response_run(condition.response_run_id, repo_id=HF_REPO_ID, required=True)
        local_status = _local_response_run_status(condition.response_run_id)
        print(
            f"Local response run status after hydration for {condition.response_run_id}: "
            f"{local_status.summary()}"
        )

    if not local_status.is_ready:
        if condition.generation is None:
            raise FileNotFoundError(
                f"Response run '{condition.response_run_id}' missing locally and on HF, "
                "and no generation config was provided."
            )
        if local_status.total_rows > 0:
            print(
                f"Generating or resuming response run: {condition.response_run_id} "
                f"({local_status.summary()})"
            )
        else:
            print(f"Generating or resuming response run: {condition.response_run_id}")
        _generate_response_run(condition)
        local_status = _local_response_run_status(condition.response_run_id)
        print(
            f"Local response run status after generation for {condition.response_run_id}: "
            f"{local_status.summary()}"
        )
        if not local_status.is_ready:
            raise RuntimeError(
                f"Response run '{condition.response_run_id}' is still not ready after generation "
                f"({local_status.summary()})."
            )
        needs_upload = True

    if local_status.is_ready and not hf_exists:
        needs_upload = True

    if local_status.is_ready and needs_upload:
        print(f"Uploading response run to HF: {condition.response_run_id}")
        upload_response_run(condition.response_run_id, repo_id=HF_REPO_ID)

    return response_run_dir(condition.response_run_id)


def _generate_response_run(condition: PromptConditionConfig) -> None:
    generation = condition.generation
    if generation is None:
        raise ValueError(f"No generation config for condition '{condition.name}'.")

    config = InferenceConfig(
        model=generation.model,
        provider=generation.provider,
        dataset=DatasetConfig(
            source="local",
            path=DATASET_PATH,
        ),
        generation=GenerationConfig(
            max_new_tokens=generation.max_new_tokens,
            temperature=generation.temperature,
            top_p=generation.top_p,
            do_sample=True,
            batch_size=generation.batch_size,
            num_responses_per_prompt=generation.num_responses,
        ),
        max_concurrent=generation.max_concurrent,
        timeout=generation.timeout,
        retry=RetryConfig(max_retries=3, backoff_factor=2.0),
        local=LocalProviderConfig(
            prompt_format=generation.local_prompt_format,
            truncate_inputs=generation.local_truncate_inputs,
        ),
        openai=OpenAIProviderConfig(),
        openrouter=OpenRouterProviderConfig(),
        run_dir=response_run_dir(condition.response_run_id),
        system_prompt=generation.system_prompt,
        max_attempts_per_sample=generation.max_attempts_per_sample,
        resume=True,
        overwrite_output=False,
    )
    dataset, result = run_inference(config)
    print(
        f"Generated {len(dataset)} rows for {condition.response_run_id} "
        f"(failed={result.num_failed})"
    )


def _ensure_embeddings_available(condition: PromptConditionConfig) -> Path:
    local_exists = _embedding_artifact_exists_local(condition.response_run_id, EMBEDDING_ARTIFACT_SLUG)
    hf_exists = _hf_embedding_exists(condition.response_run_id, EMBEDDING_ARTIFACT_SLUG)
    needs_upload = False

    if not local_exists and hf_exists:
        print(f"Hydrating embeddings from HF: {condition.response_run_id}")
        ensure_embedding_artifact(
            condition.response_run_id,
            EMBEDDING_ARTIFACT_SLUG,
            repo_id=HF_REPO_ID,
            required=True,
        )
        local_exists = _embedding_artifact_exists_local(condition.response_run_id, EMBEDDING_ARTIFACT_SLUG)

    if not local_exists:
        print(f"Generating embeddings: {condition.response_run_id}")
        _generate_embeddings(condition)
        local_exists = True
        needs_upload = True

    if local_exists and not hf_exists:
        needs_upload = True

    if local_exists and needs_upload:
        print(f"Uploading embeddings to HF: {condition.response_run_id}")
        upload_embedding_artifact(
            condition.response_run_id,
            EMBEDDING_ARTIFACT_SLUG,
            repo_id=HF_REPO_ID,
        )

    return resolve_embedding_artifact_paths(
        response_run_dir(condition.response_run_id),
        EMBEDDING_ARTIFACT_SLUG,
    )["artifact_dir"]


def _generate_embeddings(condition: PromptConditionConfig) -> None:
    run_dir = response_run_dir(condition.response_run_id)
    config = ResponseEmbeddingConfig(
        run_dir=run_dir,
        analysis_unit="assistant_final_turn",
        backend="openai",
        artifact_slug=EMBEDDING_ARTIFACT_SLUG,
        output_prefix="response_embeddings",
        resume=True,
        overwrite_output=False,
        openai=OpenAIEmbeddingConfig(
            model=EMBEDDING_MODEL,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize=True,
        ),
    )
    dataset, result = run_response_embeddings(config)
    print(
        f"Embedded {len(dataset)} rows for {condition.response_run_id} "
        f"(dim={result.embedding_dim})"
    )


def _load_condition_embeddings(
    condition: PromptConditionConfig,
) -> tuple[np.ndarray, list[dict]]:
    paths = resolve_embedding_artifact_paths(
        response_run_dir(condition.response_run_id),
        EMBEDDING_ARTIFACT_SLUG,
    )
    embeddings, metadata = load_embeddings(paths["embeddings"], paths["metadata"])

    for row in metadata:
        original_group = str(row.get("input_group_id", ""))
        shared_prompt_text = _canonical_prompt_text(row)
        row["dataset_source"] = condition.name
        row["source_run_id"] = condition.response_run_id
        row["original_shared_prompt_id"] = original_group
        row["shared_prompt_text"] = shared_prompt_text
        row["shared_prompt_id"] = _canonical_shared_prompt_id(row)
        row["input_group_id"] = f"{condition.name}::{original_group}"

    return embeddings, metadata


def _load_combined_embeddings() -> tuple[np.ndarray, list[dict]]:
    matrices: list[np.ndarray] = []
    combined_metadata: list[dict] = []
    for condition in ALL_CONDITIONS:
        embeddings, metadata = _load_condition_embeddings(condition)
        matrices.append(embeddings)
        combined_metadata.extend(metadata)
    return np.concatenate(matrices, axis=0), combined_metadata


def _dataset_mix_slug(dataset_mix: DatasetMixConfig) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in dataset_mix.name)


def _deterministic_balanced_subsample(
    metadata: list[dict],
    source_indices: list[int],
    limit: int,
) -> list[int]:
    if limit >= len(source_indices):
        return sorted(source_indices)
    if limit <= 0:
        return []

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for idx in source_indices:
        prompt_id = str(metadata[idx].get("shared_prompt_id", ""))
        grouped_indices[prompt_id].append(idx)

    prompt_ids = sorted(grouped_indices)
    for prompt_id in prompt_ids:
        grouped_indices[prompt_id].sort(
            key=lambda idx: (
                str(metadata[idx].get("sample_id", "")),
                str(metadata[idx].get("input_group_id", "")),
                idx,
            )
        )

    selected: list[int] = []
    depth = 0
    while len(selected) < limit:
        added_this_round = False
        for prompt_id in prompt_ids:
            bucket = grouped_indices[prompt_id]
            if depth >= len(bucket):
                continue
            selected.append(bucket[depth])
            added_this_round = True
            if len(selected) >= limit:
                break
        if not added_this_round:
            break
        depth += 1
    return sorted(selected)


def _apply_dataset_mix(
    embeddings: np.ndarray,
    metadata: list[dict],
    dataset_mix: DatasetMixConfig,
) -> tuple[np.ndarray, list[dict]]:
    source_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(metadata):
        source_to_indices[str(row.get("dataset_source", ""))].append(idx)

    keep_indices: list[int] = []
    for source_name, source_indices in sorted(source_to_indices.items()):
        limit = dataset_mix.source_row_limits.get(source_name)
        if limit is None:
            keep_indices.extend(sorted(source_indices))
            continue
        if limit > len(source_indices):
            raise ValueError(
                f"Dataset mix {dataset_mix.name!r} requests {limit} rows from "
                f"{source_name!r}, but only {len(source_indices)} are available."
            )
        keep_indices.extend(
            _deterministic_balanced_subsample(metadata, source_indices, limit)
        )

    keep_indices = sorted(keep_indices)
    mixed_metadata = []
    for idx in keep_indices:
        row = dict(metadata[idx])
        row["dataset_mix"] = dataset_mix.name
        mixed_metadata.append(row)
    return embeddings[keep_indices], mixed_metadata


def _dataset_mix_summary_rows(metadata: list[dict]) -> list[dict]:
    counts_by_source: dict[str, int] = defaultdict(int)
    for row in metadata:
        counts_by_source[str(row.get("dataset_source", ""))] += 1
    total = sum(counts_by_source.values())
    rows = []
    for source_name, count in sorted(counts_by_source.items()):
        rows.append(
            {
                "dataset_source": source_name,
                "count": int(count),
                "fraction_of_pool": float(count / total) if total else 0.0,
            }
        )
    return rows


def _filter_short_responses(
    embeddings: np.ndarray,
    metadata: list[dict],
    min_chars: int,
) -> tuple[np.ndarray, list[dict], int]:
    keep_indices = [
        idx
        for idx, row in enumerate(metadata)
        if len(str(row.get("assistant_text", ""))) >= min_chars
    ]
    removed = len(metadata) - len(keep_indices)
    return embeddings[keep_indices], [metadata[idx] for idx in keep_indices], removed


def _visualisation_root_dir() -> Path:
    return visualisation_artifact_dir(
        response_run_dir(NEUTRAL_CONDITION.response_run_id),
        VISUALISATION_BASE_SLUG,
    )


def _dataset_mix_dir(dataset_mix: DatasetMixConfig) -> Path:
    return _visualisation_root_dir() / _dataset_mix_slug(dataset_mix)


def _factor_output_dir(dataset_mix: DatasetMixConfig, run_flag: str) -> Path:
    return _dataset_mix_dir(dataset_mix) / run_flag


def _ensure_plotly_png_backend() -> None:
    global _PLOTLY_PNG_BACKEND_READY
    if _PLOTLY_PNG_BACKEND_READY or not SHARE_BUNDLE_INCLUDE_PNGS:
        return

    try:
        import kaleido
    except ImportError as exc:  # pragma: no cover - dependency error
        raise RuntimeError(
            "Plotly PNG export requires the 'kaleido' package to be installed."
        ) from exc

    try:
        kaleido.get_chrome_sync()
    except Exception as exc:  # pragma: no cover - network / local browser availability
        raise RuntimeError(
            "Plotly PNG export requires Chrome to be available for Kaleido. "
            "Automatic Chrome installation failed; run `plotly_get_chrome` and retry."
        ) from exc

    _PLOTLY_PNG_BACKEND_READY = True


def _save_plot_figure(fig, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    fig.write_html(path, include_plotlyjs="include")
    print(f"Saved plot: {path}")
    if SHARE_BUNDLE_INCLUDE_PNGS:
        _ensure_plotly_png_backend()
        png_path = path.with_suffix(".png")
        try:
            fig.write_image(png_path)
        except Exception as exc:  # pragma: no cover - depends on local image backend
            raise RuntimeError(
                "Plotly PNG export failed after preparing the Kaleido backend."
            ) from exc
        print(f"Saved plot: {png_path}")
    return path


def _normalise_text(text: str) -> str:
    return " ".join(text.split())


def _canonical_prompt_text(row: dict) -> str:
    for key in ("seed_user_message", "preceding_user_message"):
        value = _normalise_text(str(row.get(key, "")).strip())
        if value:
            return value
    return ""


def _canonical_shared_prompt_id(row: dict) -> str:
    prompt_text = _canonical_prompt_text(row)
    if prompt_text:
        digest = hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:16]
        return f"prompt_{digest}"
    fallback = _normalise_text(str(row.get("input_group_id", "")).strip())
    if fallback:
        return fallback
    return _normalise_text(str(row.get("sample_id", "")).strip())


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    vx = float(np.var(x, ddof=1))
    vy = float(np.var(y, ddof=1))
    denom = len(x) + len(y) - 2
    if denom <= 0:
        return 0.0
    pooled = ((len(x) - 1) * vx + (len(y) - 1) * vy) / denom
    if pooled <= 1e-12:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled))


def _paired_effect_size(diffs: np.ndarray) -> float:
    if len(diffs) < 2:
        return 0.0
    std = float(np.std(diffs, ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(np.mean(diffs) / std)


def _compute_source_separation(
    scores: np.ndarray,
    metadata: list[dict],
    *,
    reference_source: str,
) -> list[dict]:
    sources = sorted({str(row.get("dataset_source", "")) for row in metadata})
    prompt_ids = [str(row.get("shared_prompt_id", "")) for row in metadata]
    source_names = [str(row.get("dataset_source", "")) for row in metadata]

    by_prompt_source: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for idx, (prompt_id, source_name) in enumerate(zip(prompt_ids, source_names, strict=True)):
        by_prompt_source[prompt_id][source_name].append(idx)

    results: list[dict] = []
    for factor_idx in range(scores.shape[1]):
        factor_scores = scores[:, factor_idx]
        for source in sources:
            if source == reference_source:
                continue
            source_mask = np.array([name == source for name in source_names], dtype=bool)
            ref_mask = np.array([name == reference_source for name in source_names], dtype=bool)
            source_scores = factor_scores[source_mask]
            ref_scores = factor_scores[ref_mask]
            paired_diffs = []
            for prompt_groups in by_prompt_source.values():
                if source not in prompt_groups or reference_source not in prompt_groups:
                    continue
                paired_diffs.append(
                    float(
                        factor_scores[prompt_groups[source]].mean()
                        - factor_scores[prompt_groups[reference_source]].mean()
                    )
                )
            paired_diffs_arr = np.asarray(paired_diffs, dtype=np.float64)
            global_mean_diff = float(source_scores.mean() - ref_scores.mean())
            global_d = _cohens_d(source_scores, ref_scores)
            paired_mean_diff = float(paired_diffs_arr.mean()) if paired_diffs_arr.size else 0.0
            paired_dz = _paired_effect_size(paired_diffs_arr)

            order = np.argsort(factor_scores)
            top_indices = order[-100:]
            bottom_indices = order[:100]
            top_source_share = float(
                np.mean([source_names[idx] == source for idx in top_indices])
            )
            bottom_source_share = float(
                np.mean([source_names[idx] == source for idx in bottom_indices])
            )

            results.append(
                {
                    "factor_index": factor_idx,
                    "comparison_source": source,
                    "reference_source": reference_source,
                    "global_mean_diff": global_mean_diff,
                    "global_cohens_d": global_d,
                    "paired_prompt_mean_diff": paired_mean_diff,
                    "paired_prompt_dz": paired_dz,
                    "num_pairable_prompts": int(paired_diffs_arr.size),
                    "top100_source_share": top_source_share,
                    "bottom100_source_share": bottom_source_share,
                    "high_score_source": source if global_mean_diff >= 0 else reference_source,
                }
            )
    results.sort(
        key=lambda row: (
            row["comparison_source"],
            -abs(row["paired_prompt_dz"]),
            -abs(row["global_cohens_d"]),
        )
    )
    return results


def _selected_factor_indices(source_rows: list[dict]) -> list[int]:
    selected: set[int] = set()
    rows_by_source: dict[str, list[dict]] = defaultdict(list)
    for row in source_rows:
        rows_by_source[str(row["comparison_source"])].append(row)
    for rows in rows_by_source.values():
        for row in rows[:TOP_FACTORS_PER_TRAIT]:
            selected.add(int(row["factor_index"]))
    return sorted(selected)


def _selected_factor_slug(selected_factor_indices: list[int]) -> str:
    if not selected_factor_indices:
        return "none"
    return "-".join(f"f{factor_idx:03d}" for factor_idx in selected_factor_indices)


def _pairable_prompt_counts(
    metadata: list[dict],
    *,
    reference_source: str,
) -> dict[str, int]:
    by_prompt_source: dict[str, set[str]] = defaultdict(set)
    for row in metadata:
        prompt_id = str(row.get("shared_prompt_id", ""))
        source_name = str(row.get("dataset_source", ""))
        if prompt_id and source_name:
            by_prompt_source[prompt_id].add(source_name)

    counts: dict[str, int] = defaultdict(int)
    for source_names in by_prompt_source.values():
        if reference_source not in source_names:
            continue
        for source_name in source_names:
            if source_name != reference_source:
                counts[source_name] += 1
    return dict(counts)


def _labels_status(path: Path, expected_count: int) -> tuple[list[str] | None, bool]:
    if not path.exists():
        return None, False
    labels = load_label_checkpoint(path, expected_count)
    complete = len(labels) == expected_count and all(label_is_complete(label) for label in labels)
    return labels, complete


def _label_selected_extremes(
    extremes_by_factor: list[dict],
    selected_factor_indices: list[int],
    output_dir: Path,
) -> dict[int, str]:
    if not selected_factor_indices:
        return {}

    selected_extremes = [extremes_by_factor[idx] for idx in selected_factor_indices]
    selection_slug = _selected_factor_slug(selected_factor_indices)
    label_mode_slug = FACTOR_LABEL_MODE.lower()
    labels_path = (
        output_dir
        / f"selected_factor_extremes_labels_{label_mode_slug}_{selection_slug}.json"
    )
    labels, complete = _labels_status(labels_path, len(selected_extremes))
    if not complete and LABEL_FACTORS:
        if FACTOR_LABEL_MODE == "joint_distinct":
            labels = label_factors_jointly(
                selected_extremes,
                model=LABELLER_MODEL,
                provider=LABELLER_PROVIDER,
                top_n=JOINT_LABEL_TOP_N,
                excerpt_chars=JOINT_LABEL_EXCERPT_CHARS,
                max_per_prompt=JOINT_LABEL_MAX_PER_PROMPT,
                checkpoint_path=labels_path,
            )
        else:
            labels = label_factors(
                selected_extremes,
                model=LABELLER_MODEL,
                provider=LABELLER_PROVIDER,
                top_n=10,
                excerpt_chars=4000,
                max_per_prompt=4,
                prompt_format="contrastive_jsonl",
                checkpoint_path=labels_path,
            )
    labels = labels or [""] * len(selected_extremes)
    return {
        factor_idx: labels[pos]
        for pos, factor_idx in enumerate(selected_factor_indices)
    }


def _label_selected_max_spread(
    max_spread_by_factor: list[dict],
    selected_factor_indices: list[int],
    output_dir: Path,
) -> dict[int, str]:
    if not selected_factor_indices:
        return {}

    selected_max_spread = [max_spread_by_factor[idx] for idx in selected_factor_indices]
    selection_slug = _selected_factor_slug(selected_factor_indices)
    labels_path = (
        output_dir
        / (
            "selected_factor_max_spread_"
            f"{MAX_SPREAD_LABEL_STRATEGY}_{FACTOR_LABEL_MODE.lower()}_labels_{selection_slug}.json"
        )
    )
    labels, complete = _labels_status(labels_path, len(selected_max_spread))
    if not complete and LABEL_FACTORS:
        if FACTOR_LABEL_MODE == "joint_distinct":
            labels = label_max_spread_factors_jointly(
                selected_max_spread,
                strategy=MAX_SPREAD_LABEL_STRATEGY,
                model=LABELLER_MODEL,
                provider=LABELLER_PROVIDER,
                top_n=JOINT_MAX_SPREAD_TOP_N,
                excerpt_chars=JOINT_MAX_SPREAD_EXCERPT_CHARS,
                checkpoint_path=labels_path,
            )
        else:
            labels = label_max_spread_factors(
                selected_max_spread,
                strategy=MAX_SPREAD_LABEL_STRATEGY,
                model=LABELLER_MODEL,
                provider=LABELLER_PROVIDER,
                top_n=10,
                excerpt_chars=4000,
                checkpoint_path=labels_path,
            )
    labels = labels or [""] * len(selected_max_spread)
    return {
        factor_idx: labels[pos]
        for pos, factor_idx in enumerate(selected_factor_indices)
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _rank_to_unit_interval(values: np.ndarray, *, reverse: bool = False) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64)
    order = np.argsort(values)
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    if reverse:
        ranks = float(values.shape[0] - 1) - ranks
    if values.shape[0] == 1:
        return np.ones(1, dtype=np.float64)
    return ranks / float(values.shape[0] - 1)


def _compute_outlier_rows(
    data: np.ndarray,
    metadata: list[dict],
) -> list[dict]:
    if not OUTLIER_DIAGNOSTICS_ENABLED or not metadata:
        return []

    centered = data - data.mean(axis=0, keepdims=True)
    l2_distances = np.linalg.norm(centered, axis=1)
    sample_norms = np.linalg.norm(data, axis=1)
    mean_vector = data.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_vector))
    if mean_norm > 1e-12:
        cosine_to_mean = (data @ mean_vector) / (sample_norms * mean_norm + 1e-12)
    else:
        cosine_to_mean = np.full(data.shape[0], np.nan, dtype=np.float64)

    n_samples, n_dims = centered.shape
    n_components = min(OUTLIER_PCA_COMPONENTS, n_samples - 1, n_dims)
    if n_components >= 1:
        u, _, _ = np.linalg.svd(centered, full_matrices=False)
        whitened = np.sqrt(max(n_samples - 1, 1)) * u[:, :n_components]
        pca_whitened_distance = np.linalg.norm(whitened, axis=1)
    else:
        pca_whitened_distance = np.zeros(n_samples, dtype=np.float64)

    metric_ranks = [
        _rank_to_unit_interval(l2_distances, reverse=False),
        _rank_to_unit_interval(pca_whitened_distance, reverse=False),
    ]
    finite_cosine = np.isfinite(cosine_to_mean)
    if np.any(finite_cosine):
        cosine_rank = np.zeros_like(cosine_to_mean, dtype=np.float64)
        cosine_rank[finite_cosine] = _rank_to_unit_interval(
            cosine_to_mean[finite_cosine],
            reverse=True,
        )
        metric_ranks.append(cosine_rank)
    combined_outlier_score = np.mean(metric_ranks, axis=0)
    outlier_order = np.argsort(combined_outlier_score)[::-1]
    outlier_rank = np.empty_like(outlier_order)
    outlier_rank[outlier_order] = np.arange(len(outlier_order))

    rows = []
    for idx, row in enumerate(metadata):
        rows.append(
            {
                "index": idx,
                "dataset_source": str(row.get("dataset_source", "")),
                "shared_prompt_id": str(row.get("shared_prompt_id", "")),
                "sample_id": str(row.get("sample_id", "")),
                "source_run_id": str(row.get("source_run_id", "")),
                "l2_distance_from_mean": float(l2_distances[idx]),
                "pca_whitened_distance": float(pca_whitened_distance[idx]),
                "cosine_to_mean_direction": (
                    float(cosine_to_mean[idx]) if np.isfinite(cosine_to_mean[idx]) else None
                ),
                "combined_outlier_score": float(combined_outlier_score[idx]),
                "outlier_rank": int(outlier_rank[idx]) + 1,
                "prompt": str(row.get("seed_user_message", ""))[:400],
                "response": str(row.get("assistant_text", ""))[:1000],
            }
        )
    return rows


def _write_outlier_summary(
    outlier_rows: list[dict],
    output_dir: Path,
) -> None:
    if not outlier_rows:
        return

    metrics = [
        "l2_distance_from_mean",
        "pca_whitened_distance",
        "cosine_to_mean_direction",
        "combined_outlier_score",
    ]
    top_n = min(OUTLIER_TOP_N, len(outlier_rows))
    top_rows = sorted(
        outlier_rows,
        key=lambda row: float(row["combined_outlier_score"]),
        reverse=True,
    )[:top_n]
    top_rows_path = output_dir / "embedding_outlier_top_rows.jsonl"
    _write_jsonl(top_rows_path, top_rows)
    html_path = export_html(
        top_rows_path,
        [
            "outlier_rank",
            "dataset_source",
            "combined_outlier_score",
            "l2_distance_from_mean",
            "pca_whitened_distance",
            "cosine_to_mean_direction",
            "prompt",
            "response",
        ],
    )
    print(f"Outlier rows HTML viewer: {html_path}")

    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in outlier_rows:
        by_source[str(row["dataset_source"])].append(row)
    summary = []
    top_counts: dict[str, int] = defaultdict(int)
    for row in top_rows:
        top_counts[str(row["dataset_source"])] += 1
    for source_name, rows in sorted(by_source.items()):
        source_summary = {
            "dataset_source": source_name,
            "count": len(rows),
            "top_outlier_rows": int(top_counts.get(source_name, 0)),
            "top_outlier_share": float(top_counts.get(source_name, 0) / top_n) if top_n else 0.0,
        }
        for metric in metrics:
            values = [row[metric] for row in rows if row[metric] is not None]
            if not values:
                continue
            values_arr = np.asarray(values, dtype=np.float64)
            source_summary[f"{metric}_mean"] = float(values_arr.mean())
            source_summary[f"{metric}_p95"] = float(np.percentile(values_arr, 95))
            source_summary[f"{metric}_max"] = float(values_arr.max())
        summary.append(source_summary)

    with open(output_dir / "embedding_outlier_source_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def _plot_outlier_diagnostics(
    outlier_rows: list[dict],
    output_dir: Path,
) -> None:
    if not outlier_rows:
        return

    metric_labels = {
        "l2_distance_from_mean": "L2 distance from mean",
        "pca_whitened_distance": f"PCA-whitened distance ({OUTLIER_PCA_COMPONENTS} PCs max)",
        "cosine_to_mean_direction": "Cosine similarity to mean direction",
        "combined_outlier_score": "Combined outlier score",
    }
    long_rows = []
    for row in outlier_rows:
        for metric_name, metric_label in metric_labels.items():
            value = row.get(metric_name)
            if value is None:
                continue
            long_rows.append(
                {
                    "dataset_source": row["dataset_source"],
                    "metric": metric_label,
                    "value": float(value),
                }
            )
    if long_rows:
        fig = px.box(
            long_rows,
            x="dataset_source",
            y="value",
            color="dataset_source",
            facet_col="metric",
            facet_col_wrap=2,
            points=False,
            title="Embedding outlier metrics by source",
            labels={"dataset_source": "Source", "value": "Metric value", "metric": "Metric"},
        )
        fig.for_each_annotation(lambda ann: ann.update(text=ann.text.split("=")[-1]))
        _save_plot_figure(fig, output_dir, "embedding_outlier_metric_distributions.html")

    scatter_rows = []
    top_rank_cutoff = min(OUTLIER_TOP_N, len(outlier_rows))
    for row in outlier_rows:
        scatter_rows.append(
            {
                "dataset_source": row["dataset_source"],
                "l2_distance_from_mean": row["l2_distance_from_mean"],
                "pca_whitened_distance": row["pca_whitened_distance"],
                "combined_outlier_score": row["combined_outlier_score"],
                "outlier_rank": row["outlier_rank"],
                "cosine_to_mean_direction": row["cosine_to_mean_direction"],
                "hover_label": (
                    f"{row['dataset_source']} | rank {row['outlier_rank']} | "
                    f"{str(row['prompt'])[:80]}"
                ),
                "is_top_outlier": row["outlier_rank"] <= top_rank_cutoff,
            }
        )
    scatter_fig = px.scatter(
        scatter_rows,
        x="l2_distance_from_mean",
        y="pca_whitened_distance",
        color="dataset_source",
        symbol="is_top_outlier",
        size="combined_outlier_score",
        hover_name="hover_label",
        hover_data={
            "dataset_source": True,
            "outlier_rank": True,
            "combined_outlier_score": ":.3f",
            "cosine_to_mean_direction": ":.3f",
            "l2_distance_from_mean": ":.3f",
            "pca_whitened_distance": ":.3f",
            "is_top_outlier": False,
        },
        title="Relationship between mean-distance outlier metrics",
        labels={
            "l2_distance_from_mean": "L2 distance from mean",
            "pca_whitened_distance": "PCA-whitened distance",
            "combined_outlier_score": "Combined outlier score",
        },
    )
    _save_plot_figure(scatter_fig, output_dir, "embedding_outlier_metric_relationships.html")


def _export_source_separation_summary(
    source_rows: list[dict],
    factor_labels: dict[int, str],
    output_dir: Path,
) -> None:
    summary_path = output_dir / "factor_source_separation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(source_rows, handle, indent=2, ensure_ascii=False)

    table_rows = []
    for row in source_rows:
        factor_idx = int(row["factor_index"])
        table_rows.append(
            {
                "factor_index": factor_idx,
                "comparison_source": row["comparison_source"],
                "high_score_source": row["high_score_source"],
                "paired_prompt_dz": round(float(row["paired_prompt_dz"]), 4),
                "global_cohens_d": round(float(row["global_cohens_d"]), 4),
                "label": factor_labels.get(factor_idx, ""),
            }
        )
    table_path = output_dir / "factor_source_separation_summary.jsonl"
    _write_jsonl(table_path, table_rows)
    html_path = export_html(
        table_path,
        [
            "factor_index",
            "comparison_source",
            "high_score_source",
            "paired_prompt_dz",
            "global_cohens_d",
            "label",
        ],
    )
    print(f"Source separation HTML viewer: {html_path}")


def _plot_source_separation(
    source_rows: list[dict],
    factor_labels: dict[int, str],
    output_dir: Path,
) -> None:
    rows_for_plot = []
    for row in source_rows:
        factor_idx = int(row["factor_index"])
        label = factor_labels.get(factor_idx, "")
        rows_for_plot.append(
            {
                "comparison_source": row["comparison_source"],
                "factor_index": factor_idx,
                "factor_name": (
                    f"F{factor_idx:02d}: {label[:48]}" if label else f"F{factor_idx:02d}"
                ),
                "paired_prompt_dz": float(row["paired_prompt_dz"]),
                "global_cohens_d": float(row["global_cohens_d"]),
            }
        )
    if not rows_for_plot:
        return

    plot_df = rows_for_plot
    fig = px.bar(
        plot_df,
        x="factor_name",
        y="paired_prompt_dz",
        color="comparison_source",
        barmode="group",
        title="Prompt-aware source separation by factor",
        labels={
            "factor_name": "Factor",
            "paired_prompt_dz": "Paired prompt effect size (dz)",
        },
    )
    fig.update_xaxes(tickangle=45)
    _save_plot_figure(fig, output_dir, "factor_source_separation.html")

    fig2 = px.bar(
        plot_df,
        x="factor_name",
        y="global_cohens_d",
        color="comparison_source",
        barmode="group",
        title="Global source separation by factor",
        labels={
            "factor_name": "Factor",
            "global_cohens_d": "Global Cohen's d",
        },
    )
    fig2.update_xaxes(tickangle=45)
    _save_plot_figure(fig2, output_dir, "factor_source_global_effects.html")


def _factor_distribution_dir(output_dir: Path) -> Path:
    return output_dir / "factor_source_distributions_png"


def _plot_factor_source_distributions(
    scores: np.ndarray,
    metadata: list[dict],
    source_rows: list[dict],
    factor_labels: dict[int, str],
    output_dir: Path,
) -> None:
    if not EXPORT_FACTOR_DISTRIBUTION_PNGS or not source_rows:
        return

    distribution_root = _factor_distribution_dir(output_dir)
    distribution_root.mkdir(parents=True, exist_ok=True)

    source_names = np.array([str(row.get("dataset_source", "")) for row in metadata], dtype=object)
    rows_by_source: dict[str, list[dict]] = defaultdict(list)
    for row in source_rows:
        rows_by_source[str(row["comparison_source"])].append(row)

    colors = {
        REFERENCE_SOURCE_NAME: "#4c78a8",
        "comparison": "#e45756",
    }

    for comparison_source, rows in rows_by_source.items():
        comparison_dir = distribution_root / f"{comparison_source}_vs_{REFERENCE_SOURCE_NAME}"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        source_mask = source_names == comparison_source
        ref_mask = source_names == REFERENCE_SOURCE_NAME
        if not np.any(source_mask) or not np.any(ref_mask):
            continue

        n_cols = FACTOR_DISTRIBUTION_GRID_COLS
        n_rows = int(np.ceil(len(rows) / n_cols))
        fig_grid, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.2 * n_cols, 2.9 * n_rows),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        for plot_idx, row in enumerate(rows):
            factor_idx = int(row["factor_index"])
            factor_scores = scores[:, factor_idx]
            source_scores = factor_scores[source_mask]
            ref_scores = factor_scores[ref_mask]
            combined = np.concatenate([source_scores, ref_scores])
            bins = np.histogram_bin_edges(combined, bins=FACTOR_DISTRIBUTION_BINS)
            label = factor_labels.get(factor_idx, "")
            short_label = label.splitlines()[0][:52] if label else ""

            ax = axes_flat[plot_idx]
            ax.hist(
                ref_scores,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=colors[REFERENCE_SOURCE_NAME],
                label=REFERENCE_SOURCE_NAME,
            )
            ax.hist(
                source_scores,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=colors["comparison"],
                label=comparison_source,
            )
            ax.axvline(
                ref_scores.mean(),
                color=colors[REFERENCE_SOURCE_NAME],
                linestyle="--",
                linewidth=1.1,
            )
            ax.axvline(
                source_scores.mean(),
                color=colors["comparison"],
                linestyle="--",
                linewidth=1.1,
            )
            ax.set_title(
                f"F{factor_idx:03d} d={float(row['global_cohens_d']):+.2f} "
                f"dz={float(row['paired_prompt_dz']):+.2f}\n{short_label}",
                fontsize=9,
            )
            ax.set_xlabel("Factor score")
            ax.set_ylabel("Density")
            ax.grid(alpha=0.2, linewidth=0.5)

            fig_single, ax_single = plt.subplots(figsize=(8.2, 4.8))
            ax_single.hist(
                ref_scores,
                bins=bins,
                density=True,
                histtype="stepfilled",
                linewidth=1.3,
                alpha=0.22,
                color=colors[REFERENCE_SOURCE_NAME],
                label=f"{REFERENCE_SOURCE_NAME} (n={len(ref_scores)})",
            )
            ax_single.hist(
                source_scores,
                bins=bins,
                density=True,
                histtype="stepfilled",
                linewidth=1.3,
                alpha=0.22,
                color=colors["comparison"],
                label=f"{comparison_source} (n={len(source_scores)})",
            )
            ax_single.axvline(
                ref_scores.mean(),
                color=colors[REFERENCE_SOURCE_NAME],
                linestyle="--",
                linewidth=1.4,
            )
            ax_single.axvline(
                source_scores.mean(),
                color=colors["comparison"],
                linestyle="--",
                linewidth=1.4,
            )
            title = label.splitlines()[0] if label else f"Factor {factor_idx:03d}"
            ax_single.set_title(
                f"Factor {factor_idx:03d}: {title}\n"
                f"{comparison_source} vs {REFERENCE_SOURCE_NAME} | "
                f"Cohen's d={float(row['global_cohens_d']):+.3f} | "
                f"paired dz={float(row['paired_prompt_dz']):+.3f} | "
                f"pairable prompts={int(row['num_pairable_prompts'])}"
            )
            ax_single.set_xlabel("Factor score")
            ax_single.set_ylabel("Density")
            ax_single.legend(frameon=False)
            ax_single.grid(alpha=0.25, linewidth=0.5)
            fig_single.tight_layout()
            single_path = comparison_dir / f"factor_{factor_idx:03d}_distribution.png"
            fig_single.savefig(single_path, dpi=180, bbox_inches="tight")
            plt.close(fig_single)

        for ax in axes_flat[len(rows):]:
            ax.axis("off")

        handles, legend_labels = axes_flat[0].get_legend_handles_labels()
        fig_grid.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
        fig_grid.suptitle(
            f"Per-factor source distributions: {comparison_source} vs {REFERENCE_SOURCE_NAME}",
            y=0.995,
        )
        fig_grid.tight_layout(rect=(0, 0, 1, 0.965))
        grid_path = comparison_dir / "all_factor_distributions.png"
        fig_grid.savefig(grid_path, dpi=180, bbox_inches="tight")
        plt.close(fig_grid)
        print(f"Saved factor distribution PNGs: {comparison_dir}")


def _bundle_run_flags() -> list[str]:
    run_flags: list[str] = []
    for dataset_mix in DATASET_MIXES:
        mix_slug = _dataset_mix_slug(dataset_mix)
        for residualise_embeddings in RUN_RESIDUALISED:
            run_flag = "residualised" if residualise_embeddings else "non_residualised"
            rel_path = Path(mix_slug) / run_flag
            if (_dataset_mix_dir(dataset_mix) / run_flag).exists():
                run_flags.append(rel_path.as_posix())
    return run_flags


def _collect_bundle_file_entries(root_dir: Path, run_flags: list[str]) -> list[tuple[Path, str]]:
    allowed_suffixes = {".html", ".json", ".jsonl", ".npz", ".png"}
    excluded_names = {"README.md", "bundle_manifest.json"}
    entries: list[tuple[Path, str]] = []
    seen_arc_paths: set[str] = set()

    for run_flag in run_flags:
        run_dir = root_dir / run_flag
        if not run_dir.exists():
            continue
        for file_path in sorted(run_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in allowed_suffixes:
                continue
            if file_path.name in excluded_names:
                continue
            arcname = file_path.relative_to(root_dir).as_posix()
            if arcname in seen_arc_paths:
                continue
            seen_arc_paths.add(arcname)
            entries.append((file_path, arcname))

    if SHARE_BUNDLE_INCLUDE_SOURCE:
        for source_path in SHARE_BUNDLE_SOURCE_FILES:
            resolved = Path(source_path).resolve()
            arcname = f"source/{resolved.name}"
            if arcname in seen_arc_paths:
                continue
            seen_arc_paths.add(arcname)
            entries.append((resolved, arcname))

    return entries


def _build_share_bundle_readme(root_dir: Path, run_flags: list[str], bundle_entries: list[tuple[Path, str]]) -> str:
    generated_at = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    conditions_text = "\n".join(
        f"- `{condition.name}`: `{condition.response_run_id}`"
        for condition in ALL_CONDITIONS
    )
    dataset_mix_text = "\n".join(
        "- `{name}`: {limits}".format(
            name=dataset_mix.name,
            limits=(
                ", ".join(
                    f"{source}={limit}"
                    for source, limit in sorted(dataset_mix.source_row_limits.items())
                )
                or "all rows"
            ),
        )
        for dataset_mix in DATASET_MIXES
    )
    run_flags_text = "\n".join(f"- `{run_flag}`" for run_flag in run_flags)
    source_text = "\n".join(
        f"- `{arcname}`"
        for _, arcname in bundle_entries
        if arcname.startswith("source/")
    ) or "- None"

    return f"""# Prompted Trait Sanity Check Share Bundle
Generated: {generated_at}

## Summary
This bundle packages the outputs from `prompted_trait_sanity_check.py` for the prompted-trait sanity-check experiment. It includes the generated report artifacts, static PNG snapshots of chart-style plots, and the canonical source file used to produce the run.

## Source Conditions
{conditions_text}

## Dataset Mixes
{dataset_mix_text}

Embedding model: `{EMBEDDING_MODEL}`
Embedding artifact slug: `{EMBEDDING_ARTIFACT_SLUG}`
Factor analysis: `{FA_METHOD}` with `{FA_ROTATION}` rotation, `N_FACTORS={N_FACTORS}`
Included passes:
{run_flags_text}

## Reading the Results
- `paired_prompt_dz`: within-prompt effect size. Positive values mean the comparison source tends to score higher than `{REFERENCE_SOURCE_NAME}` on the same prompt.
- `global_cohens_d`: overall distribution separation between the comparison source and `{REFERENCE_SOURCE_NAME}` on a factor.
- HTML files are interactive viewers.
- PNG files are static snapshots of chart-style plots for sharing.
- JSON and JSONL files contain the raw summaries, labels, and selected examples used by the viewers.

## Bundle Layout
- `<dataset_mix>/non_residualised/` and `<dataset_mix>/residualised/`: report artifacts for each configured pass
- `source/`: copied source files for reproducibility
- `bundle_manifest.json`: machine-readable inventory of the bundle contents

## Included Source Files
{source_text}

## Notes
- HTML table viewers are included as HTML/JSONL only; they are not rasterized into PNGs.
- Existing matplotlib PNG exports (for factor source distributions) are included unchanged.
- Plotly chart PNGs are exported via `kaleido`.
"""


def _build_share_bundle_manifest(root_dir: Path, run_flags: list[str], bundle_entries: list[tuple[Path, str]]) -> dict:
    return {
        "generated_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "visualisation_label": VISUALISATION_LABEL,
        "visualisation_slug": VISUALISATION_BASE_SLUG,
        "visualisation_root": str(root_dir),
        "run_flags": run_flags,
        "reference_source": REFERENCE_SOURCE_NAME,
        "response_runs": [
            {"name": condition.name, "response_run_id": condition.response_run_id}
            for condition in ALL_CONDITIONS
        ],
        "dataset_mixes": [
            {
                "name": dataset_mix.name,
                "slug": _dataset_mix_slug(dataset_mix),
                "source_row_limits": dict(dataset_mix.source_row_limits),
            }
            for dataset_mix in DATASET_MIXES
        ],
        "embedding_model": EMBEDDING_MODEL,
        "embedding_artifact_slug": EMBEDDING_ARTIFACT_SLUG,
        "factor_analysis": {
            "n_factors": N_FACTORS,
            "method": FA_METHOD,
            "rotation": FA_ROTATION,
        },
        "source_files": [
            arcname for _, arcname in bundle_entries if arcname.startswith("source/")
        ],
        "files": [arcname for _, arcname in bundle_entries],
    }


def _write_share_bundle() -> Path | None:
    if not RUN_SHARE_BUNDLE:
        return None

    root_dir = _visualisation_root_dir()
    run_flags = _bundle_run_flags()
    bundle_entries = _collect_bundle_file_entries(root_dir, run_flags)
    if not run_flags or not bundle_entries:
        print("No share bundle written: no report artifacts found.")
        return None

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    zip_path = root_dir / f"{VISUALISATION_BASE_SLUG}_share_{timestamp}.zip"
    readme = _build_share_bundle_readme(root_dir, run_flags, bundle_entries)
    manifest = _build_share_bundle_manifest(root_dir, run_flags, bundle_entries)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.md", readme)
        zf.writestr("bundle_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        for file_path, arcname in bundle_entries:
            zf.write(file_path, arcname)

    print(f"Share bundle written to {zip_path}")
    return zip_path


def _export_selected_extremes(
    extremes_by_factor: list[dict],
    selected_factor_indices: list[int],
    factor_labels: dict[int, str],
    source_summary_by_factor: dict[int, dict],
    output_dir: Path,
) -> None:
    records = []
    for factor_idx in selected_factor_indices:
        factor_data = extremes_by_factor[factor_idx]
        summary_row = source_summary_by_factor[factor_idx]
        for polarity, entries in [("HIGH", factor_data["top"]), ("LOW", factor_data["bottom"])]:
            for rank, entry in enumerate(entries):
                records.append(
                    {
                        "question": f"Factor {factor_idx:03d} — {polarity}",
                        "response_index": rank,
                        "factor_index": factor_idx,
                        "factor_label": factor_labels.get(factor_idx, ""),
                        "dataset_source": entry.get("dataset_source", ""),
                        "comparison_source": summary_row["comparison_source"],
                        "paired_prompt_dz": round(float(summary_row["paired_prompt_dz"]), 4),
                        "global_cohens_d": round(float(summary_row["global_cohens_d"]), 4),
                        "factor_score": round(float(entry["score"]), 4),
                        "prompt": entry["seed_user_message"],
                        "response": entry["text_excerpt"],
                    }
                )

    out_path = output_dir / "selected_factor_extremes.jsonl"
    _write_jsonl(out_path, records)
    html_path = export_html(
        out_path,
        [
            "question",
            "factor_index",
            "factor_label",
            "dataset_source",
            "comparison_source",
            "paired_prompt_dz",
            "global_cohens_d",
            "factor_score",
            "prompt",
            "response",
        ],
    )
    print(f"Selected extremes HTML viewer: {html_path}")


def _export_selected_max_spread(
    max_spread_by_factor: list[dict],
    selected_factor_indices: list[int],
    factor_labels: dict[int, str],
    source_summary_by_factor: dict[int, dict],
    output_dir: Path,
) -> None:
    records = []
    for factor_idx in selected_factor_indices:
        factor_data = max_spread_by_factor[factor_idx]
        summary_row = source_summary_by_factor[factor_idx]
        for group_rank, group in enumerate(factor_data["groups"]):
            high = group["high"]
            low = group["low"]
            records.append(
                {
                    "question": f"Factor {factor_idx:03d} — HIGH (max-spread)",
                    "response_index": group_rank,
                    "factor_index": factor_idx,
                    "factor_label": factor_labels.get(factor_idx, ""),
                    "comparison_source": summary_row["comparison_source"],
                    "paired_prompt_dz": round(float(summary_row["paired_prompt_dz"]), 4),
                    "global_cohens_d": round(float(summary_row["global_cohens_d"]), 4),
                    "shared_prompt_id": group["group_id"],
                    "max_spread": round(float(group["max_spread"]), 4),
                    "group_max_score": round(float(group["group_max_score"]), 4),
                    "group_min_score": round(float(group["group_min_score"]), 4),
                    "dataset_source": high.get("dataset_source", ""),
                    "target_factor_score": round(float(high["target_factor_score"]), 4),
                    "prompt": high["seed_user_message"],
                    "response": high["text_excerpt"],
                }
            )
            records.append(
                {
                    "question": f"Factor {factor_idx:03d} — LOW (max-spread)",
                    "response_index": group_rank,
                    "factor_index": factor_idx,
                    "factor_label": factor_labels.get(factor_idx, ""),
                    "comparison_source": summary_row["comparison_source"],
                    "paired_prompt_dz": round(float(summary_row["paired_prompt_dz"]), 4),
                    "global_cohens_d": round(float(summary_row["global_cohens_d"]), 4),
                    "shared_prompt_id": group["group_id"],
                    "max_spread": round(float(group["max_spread"]), 4),
                    "group_max_score": round(float(group["group_max_score"]), 4),
                    "group_min_score": round(float(group["group_min_score"]), 4),
                    "dataset_source": low.get("dataset_source", ""),
                    "target_factor_score": round(float(low["target_factor_score"]), 4),
                    "prompt": low["seed_user_message"],
                    "response": low["text_excerpt"],
                }
            )

    out_path = output_dir / "selected_factor_max_spread.jsonl"
    _write_jsonl(out_path, records)
    html_path = export_html(
        out_path,
        [
            "question",
            "factor_index",
            "factor_label",
            "dataset_source",
            "comparison_source",
            "paired_prompt_dz",
            "global_cohens_d",
            "shared_prompt_id",
            "max_spread",
            "group_max_score",
            "group_min_score",
            "target_factor_score",
            "prompt",
            "response",
        ],
    )
    print(f"Selected max-spread HTML viewer: {html_path}")


def _with_source_metadata(entries: list[dict], metadata: list[dict]) -> list[dict]:
    for factor_data in entries:
        for key in ("top", "bottom"):
            if key not in factor_data:
                continue
            for entry in factor_data[key]:
                row = metadata[int(entry["index"])]
                entry["dataset_source"] = row.get("dataset_source", "")
        if "groups" in factor_data:
            for group in factor_data["groups"]:
                for polarity in ("high", "low"):
                    row = metadata[int(group[polarity]["index"])]
                    group[polarity]["dataset_source"] = row.get("dataset_source", "")
    return entries


def _run_visualisation_pass(
    embeddings: np.ndarray,
    metadata: list[dict],
    *,
    dataset_mix: DatasetMixConfig,
    residualise_embeddings: bool,
) -> None:
    run_flag = "residualised" if residualise_embeddings else "non_residualised"
    output_dir = _factor_output_dir(dataset_mix, run_flag)
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_embeddings, filtered_metadata, removed = _filter_short_responses(
        embeddings,
        metadata,
        min_chars=MIN_RESPONSE_CHARS,
    )
    print(
        f"[{dataset_mix.name} | {run_flag}] "
        f"Filtered {removed} short responses (<{MIN_RESPONSE_CHARS} chars); "
        f"{len(filtered_metadata)} remain"
    )
    pairable_counts = _pairable_prompt_counts(
        filtered_metadata,
        reference_source=REFERENCE_SOURCE_NAME,
    )
    if pairable_counts:
        prompt_counts_str = ", ".join(
            f"{source_name}={count}" for source_name, count in sorted(pairable_counts.items())
        )
        print(
            f"[{dataset_mix.name} | {run_flag}] Pairable prompts vs "
            f"{REFERENCE_SOURCE_NAME}: {prompt_counts_str}"
        )
    else:
        print(
            f"[{dataset_mix.name} | {run_flag}] Pairable prompts vs "
            f"{REFERENCE_SOURCE_NAME}: none"
        )

    counts_rows = _dataset_mix_summary_rows(filtered_metadata)
    with open(output_dir / "dataset_mix_counts.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_mix": dataset_mix.name,
                "source_row_limits": dict(dataset_mix.source_row_limits),
                "post_filter_counts": counts_rows,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    counts_fig = px.bar(
        counts_rows,
        x="dataset_source",
        y="count",
        text_auto=True,
        hover_data={"fraction_of_pool": ":.3f"},
        title=f"Responses retained after filtering ({dataset_mix.name} | {run_flag})",
    )
    _save_plot_figure(counts_fig, output_dir, f"retained_responses_{run_flag}.html")

    residuals, _, _ = residualize(
        filtered_embeddings,
        filtered_metadata,
        group_field="shared_prompt_id",
    )
    data = residuals if residualise_embeddings else filtered_embeddings
    outlier_rows = _compute_outlier_rows(data, filtered_metadata)
    _write_outlier_summary(outlier_rows, output_dir)
    _plot_outlier_diagnostics(outlier_rows, output_dir)

    fa_cache = output_dir / f"fa_n{N_FACTORS}_{FA_METHOD}_{FA_ROTATION}_{run_flag}"
    if fa_cache.with_suffix(".npz").exists():
        fa = load_factor_analysis(fa_cache)
    else:
        fa = run_factor_analysis(data, n_factors=N_FACTORS, method=FA_METHOD, rotation=FA_ROTATION)
        save_factor_analysis(
            fa,
            fa_cache,
            config={
                "n_factors": N_FACTORS,
                "method": FA_METHOD,
                "rotation": FA_ROTATION,
                "residualise": residualise_embeddings,
                "group_field": "shared_prompt_id",
            },
        )

    scores = fa["scores"]
    loadings = fa["loadings"]

    loadings_fig = px.imshow(
        loadings,
        aspect="auto",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title=f"Factor loadings ({run_flag})",
        labels={"x": "Factor", "y": "Dimension"},
    )
    _save_plot_figure(loadings_fig, output_dir, f"factor_loadings_{run_flag}.html")

    var_fig = go.Figure(go.Bar(x=list(range(N_FACTORS)), y=fa["proportion_variance"]))
    var_fig.update_layout(
        title=f"Variance explained per factor ({run_flag})",
        xaxis_title="Factor",
        yaxis_title="Proportion variance",
    )
    _save_plot_figure(var_fig, output_dir, f"variance_explained_{run_flag}.html")

    source_rows = _compute_source_separation(
        scores,
        filtered_metadata,
        reference_source=REFERENCE_SOURCE_NAME,
    )
    selected_factor_indices = _selected_factor_indices(source_rows)

    extremes_by_factor = _with_source_metadata(
        factor_extremes(scores, filtered_metadata, top_n=EXTREMES_TOP_N, excerpt_length=4000),
        filtered_metadata,
    )
    max_spread_by_factor = _with_source_metadata(
        [
            rank_prompts_by_max_spread(
                scores,
                filtered_metadata,
                factor_idx=factor_idx,
                top_n=MAX_SPREAD_TOP_N,
                group_field="shared_prompt_id",
                excerpt_length=4000,
            )
            for factor_idx in range(N_FACTORS)
        ],
        filtered_metadata,
    )

    factor_labels = _label_selected_extremes(extremes_by_factor, selected_factor_indices, output_dir)
    max_spread_labels = _label_selected_max_spread(
        max_spread_by_factor,
        selected_factor_indices,
        output_dir,
    )

    merged_labels = {
        factor_idx: max_spread_labels.get(factor_idx) or factor_labels.get(factor_idx, "")
        for factor_idx in selected_factor_indices
    }
    selected_factor_set = set(selected_factor_indices)
    selected_rows = [
        row for row in source_rows if int(row["factor_index"]) in selected_factor_set
    ]
    source_summary_by_factor: dict[int, dict] = {}
    for row in selected_rows:
        factor_idx = int(row["factor_index"])
        previous = source_summary_by_factor.get(factor_idx)
        if previous is None or abs(float(row["paired_prompt_dz"])) > abs(float(previous["paired_prompt_dz"])):
            source_summary_by_factor[factor_idx] = row

    _export_source_separation_summary(source_rows, merged_labels, output_dir)
    _plot_source_separation(selected_rows, merged_labels, output_dir)
    _plot_factor_source_distributions(scores, filtered_metadata, source_rows, merged_labels, output_dir)
    _export_selected_extremes(
        extremes_by_factor,
        selected_factor_indices,
        merged_labels,
        source_summary_by_factor,
        output_dir,
    )
    _export_selected_max_spread(
        max_spread_by_factor,
        selected_factor_indices,
        merged_labels,
        source_summary_by_factor,
        output_dir,
    )

    print(f"\nTop factors for {dataset_mix.name} | {run_flag}:")
    for row in source_rows[:TOP_FACTORS_PER_TRAIT]:
        factor_idx = int(row["factor_index"])
        label = merged_labels.get(factor_idx, "")
        print(
            f"  Factor {factor_idx:03d}  "
            f"paired_dz={row['paired_prompt_dz']:+.3f}  "
            f"global_d={row['global_cohens_d']:+.3f}  "
            f"high={row['high_score_source']}  "
            f"{label.splitlines()[0] if label else ''}"
        )


def main() -> None:
    print("Ensuring response runs...")
    for condition in ALL_CONDITIONS:
        _ensure_response_run_available(condition)

    print("\nEnsuring embeddings...")
    for condition in ALL_CONDITIONS:
        _ensure_embeddings_available(condition)

    print("\nLoading combined embeddings...")
    full_embeddings, full_metadata = _load_combined_embeddings()
    print(f"Loaded {len(full_metadata)} combined responses across {len(ALL_CONDITIONS)} sources")

    for dataset_mix in DATASET_MIXES:
        embeddings, metadata = _apply_dataset_mix(full_embeddings, full_metadata, dataset_mix)
        counts_str = ", ".join(
            f"{row['dataset_source']}={row['count']} ({row['fraction_of_pool']:.3f})"
            for row in _dataset_mix_summary_rows(metadata)
        )
        print(
            f"\nDataset mix {dataset_mix.name}: "
            f"{len(metadata)} rows total | {counts_str}"
        )

        for residualise_embeddings in RUN_RESIDUALISED:
            print(
                "\nRunning visualisation pass: "
                f"dataset_mix={dataset_mix.name}, residualised={residualise_embeddings}"
            )
            _run_visualisation_pass(
                embeddings,
                metadata,
                dataset_mix=dataset_mix,
                residualise_embeddings=residualise_embeddings,
            )

    _write_share_bundle()


if __name__ == "__main__":
    main()
