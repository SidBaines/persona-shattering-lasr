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

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from huggingface_hub import HfApi
from pydantic import BaseModel

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.factor_analysis.factor_analysis import run_factor_analysis
from scripts.factor_analysis.interpretation import (
    factor_extremes,
    rank_prompts_by_max_spread,
)
from scripts.factor_analysis.labelling import (
    label_factors,
    label_is_complete,
    label_max_spread_factors,
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

EMBEDDING_ARTIFACT_SLUG = "openai-text-embedding-3-small__assistant-final-turn__norm"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 128

VISUALISATION_LABEL = "prompted_trait_sanity_check_disagreeable"
VISUALISATION_SLUG = build_visualisation_slug(
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
TOP_FACTORS_PER_TRAIT = 8
EXTREMES_TOP_N = 20
MAX_SPREAD_TOP_N = 20
MAX_SPREAD_LABEL_STRATEGY = "label_pair_score"


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
    local_exists = _response_run_exists_local(condition.response_run_id)
    hf_exists = _hf_response_run_exists(condition.response_run_id)

    if not local_exists and hf_exists:
        print(f"Hydrating response run from HF: {condition.response_run_id}")
        ensure_response_run(condition.response_run_id, repo_id=HF_REPO_ID, required=True)
        local_exists = _response_run_exists_local(condition.response_run_id)

    if not local_exists:
        if condition.generation is None:
            raise FileNotFoundError(
                f"Response run '{condition.response_run_id}' missing locally and on HF, "
                "and no generation config was provided."
            )
        print(f"Generating response run: {condition.response_run_id}")
        _generate_response_run(condition)
        local_exists = True

    if local_exists and not hf_exists:
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

    if local_exists and not hf_exists:
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
        row["dataset_source"] = condition.name
        row["source_run_id"] = condition.response_run_id
        row["shared_prompt_id"] = original_group
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


def _factor_output_dir(run_flag: str) -> Path:
    return visualisation_artifact_dir(
        response_run_dir(NEUTRAL_CONDITION.response_run_id),
        VISUALISATION_SLUG,
    ) / run_flag


def _save_plot_html(fig, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    fig.write_html(path, include_plotlyjs="include")
    print(f"Saved plot: {path}")
    return path


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
    labels_path = output_dir / "selected_factor_extremes_labels.json"
    labels, complete = _labels_status(labels_path, len(selected_extremes))
    if not complete and LABEL_FACTORS:
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
    labels_path = output_dir / f"selected_factor_max_spread_{MAX_SPREAD_LABEL_STRATEGY}_labels.json"
    labels, complete = _labels_status(labels_path, len(selected_max_spread))
    if not complete and LABEL_FACTORS:
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
    _save_plot_html(fig, output_dir, "factor_source_separation.html")

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
    _save_plot_html(fig2, output_dir, "factor_source_global_effects.html")


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
    residualise_embeddings: bool,
) -> None:
    run_flag = "residualised" if residualise_embeddings else "non_residualised"
    output_dir = _factor_output_dir(run_flag)
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_embeddings, filtered_metadata, removed = _filter_short_responses(
        embeddings,
        metadata,
        min_chars=MIN_RESPONSE_CHARS,
    )
    print(
        f"[{run_flag}] Filtered {removed} short responses (<{MIN_RESPONSE_CHARS} chars); "
        f"{len(filtered_metadata)} remain"
    )

    counts_rows = []
    counts_by_source: dict[str, int] = defaultdict(int)
    for row in filtered_metadata:
        counts_by_source[str(row.get("dataset_source", ""))] += 1
    for source_name, count in sorted(counts_by_source.items()):
        counts_rows.append({"dataset_source": source_name, "count": count})
    counts_fig = px.bar(
        counts_rows,
        x="dataset_source",
        y="count",
        title=f"Responses retained after filtering ({run_flag})",
    )
    _save_plot_html(counts_fig, output_dir, f"retained_responses_{run_flag}.html")

    residuals, _, _ = residualize(
        filtered_embeddings,
        filtered_metadata,
        group_field="shared_prompt_id",
    )
    data = residuals if residualise_embeddings else filtered_embeddings

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
    _save_plot_html(loadings_fig, output_dir, f"factor_loadings_{run_flag}.html")

    var_fig = go.Figure(go.Bar(x=list(range(N_FACTORS)), y=fa["proportion_variance"]))
    var_fig.update_layout(
        title=f"Variance explained per factor ({run_flag})",
        xaxis_title="Factor",
        yaxis_title="Proportion variance",
    )
    _save_plot_html(var_fig, output_dir, f"variance_explained_{run_flag}.html")

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

    print(f"\nTop factors for {run_flag}:")
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
    embeddings, metadata = _load_combined_embeddings()
    print(f"Loaded {len(metadata)} combined responses across {len(ALL_CONDITIONS)} sources")

    for residualise_embeddings in RUN_RESIDUALISED:
        print(f"\nRunning visualisation pass: residualised={residualise_embeddings}")
        _run_visualisation_pass(
            embeddings,
            metadata,
            residualise_embeddings=residualise_embeddings,
        )


if __name__ == "__main__":
    main()
