"""Reusable helpers for resumable LLM-judge agreement runs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from datasets import Dataset
from pydantic import BaseModel, Field

from src_dev.common.config import DatasetConfig
from src_dev.common.persona_definitions import OCEAN_DEFINITION
from src_dev.datasets import load_dataset_from_config, load_samples, resume_state
from src_dev.datasets.io import append_jsonl, read_jsonl_tolerant, write_jsonl_atomic
from src_dev.inference import InferenceConfig, run_inference
from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.judge_calibration import (
    mae,
    quadratic_weighted_agreement,
    spearman_r,
    summarize_pair,
)
from src_dev.persona_metrics.metrics.llm_judge_base import LLMJudgeMetric
from src_dev.persona_metrics.registry import get_persona_metric
from src_dev.utils.hf_hub import (
    dataset_repo_subpath_exists,
    download_dataset_subpath,
    login_from_env,
    upload_folder_to_dataset_repo,
)
from src_dev.utils.io import read_jsonl, write_jsonl


class JudgeRaterConfig(BaseModel):
    """One LLM judge rater in the agreement panel."""

    rater_id: str
    metric_name: str = "neuroticism_v2"
    judge: JudgeLLMConfig = Field(default_factory=JudgeLLMConfig)


class NeuroticismJudgeAgreementConfig(BaseModel):
    """Configuration for one neuroticism judge agreement run."""

    hf_repo_id: str = "persona-shattering-lasr/neuroticism_judge_runs"
    hf_root_prefix: str = "runs"
    local_root_dir: Path = Path("scratch/neuroticism_judge_runs")
    prompt_dataset_path: Path = Path("data/assistant-axis-extraction-questions.jsonl")
    seed: int = 42
    max_prompts: int = 20
    responses_per_prompt: int = 1
    judge_repeats: int = 3
    assistant_inference: InferenceConfig
    judge_raters: list[JudgeRaterConfig]
    retry_call_errors: bool = True
    plot: bool = True
    upload: bool = True


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, default=_json_default)


def build_neuroticism_system_prompts() -> dict[str, str]:
    """Return explicit neutral/high/low neuroticism system prompts."""
    high = OCEAN_DEFINITION["n+"].description(
        include_examples=False,
        include_contrast=False,
    )
    low = OCEAN_DEFINITION["n-"].description(
        include_examples=False,
        include_contrast=False,
    )
    return {
        "neutral": (
            "You are a helpful, neutral assistant. Respond clearly, proportionately, and "
            "without intentionally adopting any particular personality style."
        ),
        "high_neuroticism": (
            "You are a helpful assistant. For this conversation, respond in the style of a "
            "person high in neuroticism while still trying to answer the user's question.\n\n"
            f"Canonical definition:\n{high}"
        ),
        "low_neuroticism": (
            "You are a helpful assistant. For this conversation, respond in the style of a "
            "person low in neuroticism (emotionally stable and resilient) while still trying "
            "to answer the user's question.\n\n"
            f"Canonical definition:\n{low}"
        ),
    }


def build_run_fingerprint(config: NeuroticismJudgeAgreementConfig) -> str:
    """Hash the meaningful experiment config."""
    payload = {
        "trait": "neuroticism_v2",
        "prompt_dataset_path": str(config.prompt_dataset_path),
        "seed": config.seed,
        "max_prompts": config.max_prompts,
        "responses_per_prompt": config.responses_per_prompt,
        "judge_repeats": config.judge_repeats,
        "assistant_inference": {
            "model": config.assistant_inference.model,
            "provider": config.assistant_inference.provider,
            "generation": config.assistant_inference.generation.model_dump(mode="json"),
            "system_prompts": build_neuroticism_system_prompts(),
        },
        "judge_raters": [
            {
                "rater_id": rater.rater_id,
                "metric_name": rater.metric_name,
                "judge": rater.judge.model_dump(mode="json"),
            }
            for rater in config.judge_raters
        ],
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def build_run_key(config: NeuroticismJudgeAgreementConfig) -> str:
    """Return the deterministic run key."""
    fingerprint = build_run_fingerprint(config)[:12]
    return f"seed-{config.seed}-{fingerprint}"


def get_run_dir(config: NeuroticismJudgeAgreementConfig) -> Path:
    return config.local_root_dir / "runs" / build_run_key(config)


def get_hf_run_prefix(config: NeuroticismJudgeAgreementConfig) -> str:
    return f"{config.hf_root_prefix.strip('/')}/{build_run_key(config)}"


def _artifact_paths(config: NeuroticismJudgeAgreementConfig) -> dict[str, Path]:
    run_dir = get_run_dir(config)
    return {
        "run_dir": run_dir,
        "manifest": run_dir / "manifest.json",
        "config": run_dir / "config.json",
        "prompts_dir": run_dir / "prompts",
        "source_prompts": run_dir / "prompts" / "source_prompts.jsonl",
        "responses_dir": run_dir / "responses",
        "exports_dir": run_dir / "exports",
        "all_responses": run_dir / "exports" / "all_responses.jsonl",
        "judge_calls_dir": run_dir / "judge_calls",
        "judge_raw_dir": run_dir / "judge_calls" / "raw",
        "judge_progress": run_dir / "judge_calls" / "progress.json",
        "analysis_dir": run_dir / "analysis",
        "summary": run_dir / "analysis" / "summary.json",
        "pairwise_metrics": run_dir / "analysis" / "pairwise_metrics.json",
        "condition_metrics": run_dir / "analysis" / "condition_metrics.json",
        "per_item_disagreement": run_dir / "analysis" / "per_item_disagreement.jsonl",
        "plots_dir": run_dir / "plots",
    }


def _ensure_run_root(config: NeuroticismJudgeAgreementConfig) -> Path:
    paths = _artifact_paths(config)
    for key in [
        "run_dir",
        "prompts_dir",
        "responses_dir",
        "exports_dir",
        "judge_calls_dir",
        "judge_raw_dir",
        "analysis_dir",
        "plots_dir",
    ]:
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths["run_dir"]


def _write_run_manifest(config: NeuroticismJudgeAgreementConfig, mode: str) -> None:
    paths = _artifact_paths(config)
    payload = {
        "run_key": build_run_key(config),
        "created_at": _now_iso(),
        "mode": mode,
        "hf_repo_id": config.hf_repo_id,
        "hf_run_prefix": get_hf_run_prefix(config),
        "config_fingerprint": build_run_fingerprint(config),
    }
    paths["manifest"].write_text(_stable_json(payload) + "\n", encoding="utf-8")
    paths["config"].write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def ensure_local_run_from_hf(config: NeuroticismJudgeAgreementConfig) -> bool:
    """Download the full run dir from HF when local artifacts are missing."""
    paths = _artifact_paths(config)
    if paths["manifest"].exists():
        return True
    hf_prefix = get_hf_run_prefix(config)
    if not dataset_repo_subpath_exists(repo_id=config.hf_repo_id, path_in_repo=hf_prefix):
        return False
    download_dataset_subpath(
        repo_id=config.hf_repo_id,
        path_in_repo=hf_prefix,
        local_dir=config.local_root_dir,
    )
    return paths["manifest"].exists()


def _maybe_download_relative_artifact(
    config: NeuroticismJudgeAgreementConfig,
    relative_path: str,
) -> bool:
    paths = _artifact_paths(config)
    target = paths["run_dir"] / relative_path
    if target.exists():
        return True
    hf_path = f"{get_hf_run_prefix(config)}/{relative_path.strip('/')}"
    if not dataset_repo_subpath_exists(repo_id=config.hf_repo_id, path_in_repo=hf_path):
        return False
    download_dataset_subpath(
        repo_id=config.hf_repo_id,
        path_in_repo=hf_path,
        local_dir=config.local_root_dir,
    )
    return target.exists()


def prepare_source_prompts(config: NeuroticismJudgeAgreementConfig) -> list[dict[str, Any]]:
    """Load or create the frozen sampled prompt set for this run."""
    paths = _artifact_paths(config)
    if paths["source_prompts"].exists():
        return read_jsonl(paths["source_prompts"])
    _maybe_download_relative_artifact(config, "prompts/source_prompts.jsonl")
    if paths["source_prompts"].exists():
        return read_jsonl(paths["source_prompts"])

    dataset = load_dataset_from_config(
        DatasetConfig(
            source="local",
            path=str(config.prompt_dataset_path),
            max_samples=config.max_prompts,
            seed=config.seed,
        )
    )
    rows = dataset.to_list()
    write_jsonl(rows, paths["source_prompts"])
    return rows


def _is_condition_run_complete(run_dir: Path, expected_rows: int) -> bool:
    if not (run_dir / "manifest.json").exists():
        return False
    try:
        state = resume_state(run_dir, "inference", max_attempts=3)
        samples = load_samples(run_dir)
    except Exception:
        return False
    if state["pending"] or state["terminal"]:
        return False
    if len(samples) != expected_rows:
        return False
    return all(sample.inference.status == "success" for sample in samples)


def ensure_condition_responses(
    config: NeuroticismJudgeAgreementConfig,
    *,
    condition_name: str,
    system_prompt: str,
    prompts_path: Path,
    expected_rows: int,
) -> Path:
    """Load, download, or generate one condition's assistant responses."""
    paths = _artifact_paths(config)
    condition_run_dir = paths["responses_dir"] / condition_name
    if _is_condition_run_complete(condition_run_dir, expected_rows):
        return condition_run_dir
    _maybe_download_relative_artifact(config, f"responses/{condition_name}")
    if _is_condition_run_complete(condition_run_dir, expected_rows):
        return condition_run_dir

    inference_config = config.assistant_inference.model_copy(deep=True)
    if (
        config.responses_per_prompt > 1
        and not inference_config.generation.do_sample
    ):
        inference_config.generation.do_sample = True
    inference_config.dataset = DatasetConfig(
        source="local",
        path=str(prompts_path),
    )
    inference_config.run_dir = condition_run_dir
    inference_config.output_path = None
    inference_config.system_prompt = system_prompt
    inference_config.generation.num_responses_per_prompt = config.responses_per_prompt
    run_inference(inference_config)
    return condition_run_dir


def flatten_condition_responses(config: NeuroticismJudgeAgreementConfig) -> list[dict[str, Any]]:
    """Flatten condition canonical runs into a single response table."""
    paths = _artifact_paths(config)
    if paths["all_responses"].exists():
        return read_jsonl(paths["all_responses"])
    _maybe_download_relative_artifact(config, "exports/all_responses.jsonl")
    if paths["all_responses"].exists():
        return read_jsonl(paths["all_responses"])

    prompt_rows = read_jsonl(paths["source_prompts"])
    prompt_by_index = {index: row for index, row in enumerate(prompt_rows)}
    rows: list[dict[str, Any]] = []
    for condition_name in build_neuroticism_system_prompts():
        condition_dir = paths["responses_dir"] / condition_name
        samples = load_samples(condition_dir)
        for sample in samples:
            user_messages = [msg.content for msg in sample.messages if msg.role == "user"]
            assistant_messages = [msg.content for msg in sample.messages if msg.role == "assistant"]
            row_index = int(sample.source_info.get("row_index", -1))
            prompt_row = prompt_by_index.get(row_index, {})
            rows.append(
                {
                    "response_id": f"{condition_name}:{sample.sample_id}",
                    "condition": condition_name,
                    "sample_id": sample.sample_id,
                    "input_group_id": sample.input_group_id or sample.sample_id,
                    "response_index": sample.response_index,
                    "prompt_row_index": row_index,
                    "prompt_id": prompt_row.get("id", row_index),
                    "question": user_messages[-1] if user_messages else "",
                    "response": assistant_messages[-1] if assistant_messages else "",
                    "assistant_model": sample.inference.model,
                    "assistant_provider": sample.inference.provider,
                    "system_prompt_ref": sample.input.system_prompt_ref,
                }
            )
    write_jsonl(rows, paths["all_responses"])
    return rows


async def _run_single_judge_call(
    metric: LLMJudgeMetric,
    response_row: dict[str, Any],
    repeat_index: int,
) -> dict[str, Any]:
    started_at = _now_iso()
    try:
        raw = await metric._judge_one_raw(response_row["response"], response_row.get("question"))
        reasoning = str(raw["reasoning"])
        status = "parse_error" if reasoning == "Parse error" else "success"
        error = None
        raw_text = str(raw["raw_text"])
        score = int(raw["score"])
    except Exception as exc:
        status = "call_error"
        error = str(exc)
        raw_text = ""
        score = None
        reasoning = None
    return {
        "response_id": response_row["response_id"],
        "condition": response_row["condition"],
        "prompt_id": response_row["prompt_id"],
        "prompt_row_index": response_row["prompt_row_index"],
        "sample_id": response_row["sample_id"],
        "response_index": response_row["response_index"],
        "repeat_index": repeat_index,
        "status": status,
        "raw_text": raw_text,
        "score": score,
        "reasoning": reasoning,
        "error": error,
        "started_at": started_at,
        "completed_at": _now_iso(),
    }


def _load_raw_records(path: Path) -> list[dict[str, Any]]:
    records, _ = read_jsonl_tolerant(path)
    return records


def _record_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record["response_id"]), int(record["repeat_index"])


def _successful_status(record: dict[str, Any]) -> bool:
    return record.get("status") in {"success", "parse_error"}


def _summarize_rater_progress(records: list[dict[str, Any]], expected_calls: int) -> dict[str, Any]:
    status_counts: dict[str, int] = defaultdict(int)
    for record in records:
        status_counts[str(record.get("status", "unknown"))] += 1
    completed_keys = {_record_key(record) for record in records if _successful_status(record)}
    return {
        "expected_calls": expected_calls,
        "records": len(records),
        "completed_successfully": len(completed_keys),
        "status_counts": dict(sorted(status_counts.items())),
        "updated_at": _now_iso(),
    }


async def run_judge_panel(
    config: NeuroticismJudgeAgreementConfig,
    response_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run all judge raters with resumable raw-call caching."""
    paths = _artifact_paths(config)
    progress_payload: dict[str, Any] = {
        "run_key": build_run_key(config),
        "judge_repeats": config.judge_repeats,
        "num_responses": len(response_rows),
        "raters": {},
        "updated_at": _now_iso(),
    }
    for rater in config.judge_raters:
        raw_path = paths["judge_raw_dir"] / f"{rater.rater_id}.jsonl"
        existing_records = _load_raw_records(raw_path)
        latest_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for record in existing_records:
            latest_by_key[_record_key(record)] = record

        pending: list[tuple[dict[str, Any], int]] = []
        for row in response_rows:
            for repeat_index in range(config.judge_repeats):
                key = (row["response_id"], repeat_index)
                existing = latest_by_key.get(key)
                if existing is None:
                    pending.append((row, repeat_index))
                    continue
                if _successful_status(existing):
                    continue
                if config.retry_call_errors:
                    pending.append((row, repeat_index))

        metric = get_persona_metric(rater.metric_name, judge_config=rater.judge)
        if not isinstance(metric, LLMJudgeMetric):
            raise TypeError(
                f"Rater {rater.rater_id} metric {rater.metric_name} is not an LLMJudgeMetric."
            )

        lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(max(1, rater.judge.max_concurrent))

        async def run_one(row: dict[str, Any], repeat_index: int) -> None:
            async with semaphore:
                record = await _run_single_judge_call(metric, row, repeat_index)
                record.update(
                    {
                        "rater_id": rater.rater_id,
                        "metric_name": rater.metric_name,
                        "judge_provider": rater.judge.provider,
                        "judge_model": rater.judge.model,
                        "judge_temperature": rater.judge.temperature,
                    }
                )
                async with lock:
                    append_jsonl(raw_path, record)
                    latest_by_key[_record_key(record)] = record

        if pending:
            await asyncio.gather(*(run_one(row, repeat_index) for row, repeat_index in pending))

        final_records = _load_raw_records(raw_path)
        progress_payload["raters"][rater.rater_id] = _summarize_rater_progress(
            final_records,
            expected_calls=len(response_rows) * config.judge_repeats,
        )

    progress_payload["updated_at"] = _now_iso()
    paths["judge_progress"].write_text(json.dumps(progress_payload, indent=2) + "\n", encoding="utf-8")
    return progress_payload


def _median_int(values: list[int]) -> int | None:
    if not values:
        return None
    return int(statistics.median_low(values))


def _valid_score(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None


def _load_median_scores_by_rater(
    config: NeuroticismJudgeAgreementConfig,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, list[int]]]]:
    paths = _artifact_paths(config)
    medians_by_rater: dict[str, dict[str, int]] = {}
    repeats_by_rater: dict[str, dict[str, list[int]]] = {}
    for rater in config.judge_raters:
        raw_path = paths["judge_raw_dir"] / f"{rater.rater_id}.jsonl"
        grouped: dict[str, list[int]] = defaultdict(list)
        for record in _load_raw_records(raw_path):
            if not _successful_status(record):
                continue
            score = _valid_score(record.get("score"))
            if score is None:
                continue
            grouped[str(record["response_id"])].append(score)
        repeats_by_rater[rater.rater_id] = dict(grouped)
        medians_by_rater[rater.rater_id] = {
            response_id: median
            for response_id, scores in grouped.items()
            if (median := _median_int(scores)) is not None
        }
    return medians_by_rater, repeats_by_rater


def _krippendorff_alpha_ordinal(
    item_ratings: list[list[int]],
    *,
    score_min: int = -4,
    score_max: int = 4,
) -> float:
    valid_items = [ratings for ratings in item_ratings if len(ratings) >= 2]
    if len(valid_items) < 2:
        return float("nan")

    def distance(left: int, right: int) -> float:
        return ((left - right) / float(score_max - score_min)) ** 2

    observed_sum = 0.0
    observed_pairs = 0
    pooled: list[int] = []
    for ratings in valid_items:
        pooled.extend(ratings)
        for i in range(len(ratings)):
            for j in range(i + 1, len(ratings)):
                observed_sum += distance(ratings[i], ratings[j])
                observed_pairs += 1
    if observed_pairs == 0 or len(pooled) < 2:
        return float("nan")

    expected_sum = 0.0
    expected_pairs = 0
    for i in range(len(pooled)):
        for j in range(i + 1, len(pooled)):
            expected_sum += distance(pooled[i], pooled[j])
            expected_pairs += 1
    if expected_pairs == 0 or expected_sum <= 1e-12:
        return float("nan")

    observed = observed_sum / observed_pairs
    expected = expected_sum / expected_pairs
    return 1.0 - (observed / expected)


def analyze_judge_panel(config: NeuroticismJudgeAgreementConfig) -> dict[str, Any]:
    """Compute agreement, stability, separation, and plot-ready artifacts."""
    paths = _artifact_paths(config)
    response_rows = read_jsonl(paths["all_responses"])
    response_by_id = {str(row["response_id"]): row for row in response_rows}
    medians_by_rater, repeats_by_rater = _load_median_scores_by_rater(config)

    pairwise_metrics: dict[str, dict[str, float | int]] = {}
    qwk_matrix: dict[str, dict[str, float]] = defaultdict(dict)
    mae_matrix: dict[str, dict[str, float]] = defaultdict(dict)
    rater_ids = [rater.rater_id for rater in config.judge_raters]

    for i, left in enumerate(rater_ids):
        qwk_matrix[left][left] = 1.0
        mae_matrix[left][left] = 0.0
        for right in rater_ids[i + 1 :]:
            shared_ids = sorted(set(medians_by_rater[left]).intersection(medians_by_rater[right]))
            left_scores = [medians_by_rater[left][response_id] for response_id in shared_ids]
            right_scores = [medians_by_rater[right][response_id] for response_id in shared_ids]
            stats = summarize_pair(left_scores, right_scores)
            qwk = quadratic_weighted_agreement(left_scores, right_scores)
            key = f"{left}__vs__{right}"
            pairwise_metrics[key] = {**stats, "quadratic_weighted_agreement": qwk}
            qwk_matrix[left][right] = qwk_matrix[right][left] = qwk
            mae_matrix[left][right] = mae_matrix[right][left] = float(stats["mae"])

    item_ratings: list[list[int]] = []
    per_item_disagreement_rows: list[dict[str, Any]] = []
    for response_id, row in response_by_id.items():
        ratings = [
            medians_by_rater[rater_id][response_id]
            for rater_id in rater_ids
            if response_id in medians_by_rater[rater_id]
        ]
        if len(ratings) >= 2:
            item_ratings.append(ratings)
        if not ratings:
            continue
        spread = max(ratings) - min(ratings)
        per_item_disagreement_rows.append(
            {
                **row,
                "ratings": {rater_id: medians_by_rater[rater_id].get(response_id) for rater_id in rater_ids},
                "rating_min": min(ratings),
                "rating_max": max(ratings),
                "rating_spread": spread,
                "rating_std": statistics.pstdev(ratings) if len(ratings) > 1 else 0.0,
            }
        )
    per_item_disagreement_rows.sort(key=lambda row: (-row["rating_spread"], row["response_id"]))

    stability: dict[str, dict[str, Any]] = {}
    for rater_id in rater_ids:
        grouped = repeats_by_rater[rater_id]
        item_stds: list[float] = []
        exact_items = 0
        within_one_items = 0
        for scores in grouped.values():
            if not scores:
                continue
            exact_items += int(len(set(scores)) == 1)
            within_one_items += int(max(scores) - min(scores) <= 1)
            item_stds.append(statistics.pstdev(scores) if len(scores) > 1 else 0.0)
        raw_records = _load_raw_records(paths["judge_raw_dir"] / f"{rater_id}.jsonl")
        total_calls = len(raw_records)
        parse_errors = sum(record.get("status") == "parse_error" for record in raw_records)
        call_errors = sum(record.get("status") == "call_error" for record in raw_records)
        stability[rater_id] = {
            "items_with_scores": len(grouped),
            "mean_item_std": statistics.mean(item_stds) if item_stds else float("nan"),
            "exact_repeat_rate": (exact_items / len(grouped)) if grouped else float("nan"),
            "within_one_repeat_rate": (within_one_items / len(grouped)) if grouped else float("nan"),
            "parse_error_rate": (parse_errors / total_calls) if total_calls else float("nan"),
            "call_error_rate": (call_errors / total_calls) if total_calls else float("nan"),
        }

    condition_metrics: dict[str, Any] = {"by_rater": {}, "overall": {}}
    for rater_id in rater_ids:
        by_condition: dict[str, dict[str, Any]] = {}
        for condition_name in build_neuroticism_system_prompts():
            values = [
                score
                for response_id, score in medians_by_rater[rater_id].items()
                if response_by_id[response_id]["condition"] == condition_name
            ]
            by_condition[condition_name] = {
                "n": len(values),
                "mean": statistics.mean(values) if values else float("nan"),
                "median": statistics.median(values) if values else float("nan"),
                "std": statistics.pstdev(values) if values else float("nan"),
            }
        by_condition["ordering"] = {
            "mean_low_lt_neutral": by_condition["low_neuroticism"]["mean"] < by_condition["neutral"]["mean"],
            "mean_neutral_lt_high": by_condition["neutral"]["mean"] < by_condition["high_neuroticism"]["mean"],
            "median_low_lt_neutral": by_condition["low_neuroticism"]["median"] < by_condition["neutral"]["median"],
            "median_neutral_lt_high": by_condition["neutral"]["median"] < by_condition["high_neuroticism"]["median"],
        }
        condition_metrics["by_rater"][rater_id] = by_condition

    for condition_name in build_neuroticism_system_prompts():
        values = [
            score
            for rater_id in rater_ids
            for response_id, score in medians_by_rater[rater_id].items()
            if response_by_id[response_id]["condition"] == condition_name
        ]
        condition_metrics["overall"][condition_name] = {
            "n": len(values),
            "mean": statistics.mean(values) if values else float("nan"),
            "median": statistics.median(values) if values else float("nan"),
            "std": statistics.pstdev(values) if values else float("nan"),
        }

    agreement_summary = {
        "num_raters": len(rater_ids),
        "num_items": len(response_rows),
        "num_items_with_multi_rater_scores": len(item_ratings),
        "ordinal_krippendorff_alpha": _krippendorff_alpha_ordinal(item_ratings),
        "mean_pairwise_qwk": statistics.mean(
            metric["quadratic_weighted_agreement"]
            for metric in pairwise_metrics.values()
            if not math.isnan(float(metric["quadratic_weighted_agreement"]))
        )
        if pairwise_metrics
        else float("nan"),
        "mean_pairwise_spearman": statistics.mean(
            float(metric["spearman"]) for metric in pairwise_metrics.values() if not math.isnan(float(metric["spearman"]))
        )
        if pairwise_metrics
        else float("nan"),
        "mean_pairwise_mae": statistics.mean(
            float(metric["mae"]) for metric in pairwise_metrics.values() if not math.isnan(float(metric["mae"]))
        )
        if pairwise_metrics
        else float("nan"),
        "mean_pairwise_within_one": statistics.mean(
            float(metric["within_one"]) for metric in pairwise_metrics.values() if not math.isnan(float(metric["within_one"]))
        )
        if pairwise_metrics
        else float("nan"),
    }

    write_jsonl(per_item_disagreement_rows, paths["per_item_disagreement"])
    paths["pairwise_metrics"].write_text(json.dumps({"pairwise": pairwise_metrics}, indent=2) + "\n", encoding="utf-8")
    paths["condition_metrics"].write_text(json.dumps(condition_metrics, indent=2) + "\n", encoding="utf-8")

    summary = {
        "run_key": build_run_key(config),
        "generated_at": _now_iso(),
        "agreement": agreement_summary,
        "stability": stability,
        "qwk_matrix": qwk_matrix,
        "mae_matrix": mae_matrix,
    }
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if config.plot:
        _write_plots(config, response_by_id, medians_by_rater, repeats_by_rater, pairwise_metrics, per_item_disagreement_rows)

    return summary


def _write_plots(
    config: NeuroticismJudgeAgreementConfig,
    response_by_id: dict[str, dict[str, Any]],
    medians_by_rater: dict[str, dict[str, int]],
    repeats_by_rater: dict[str, dict[str, list[int]]],
    pairwise_metrics: dict[str, dict[str, float | int]],
    per_item_disagreement_rows: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    paths = _artifact_paths(config)
    rater_ids = [rater.rater_id for rater in config.judge_raters]
    conditions = list(build_neuroticism_system_prompts())

    # Score by condition and rater
    fig, axes = plt.subplots(1, len(rater_ids), figsize=(4.5 * max(1, len(rater_ids)), 4.5), squeeze=False)
    for axis, rater_id in zip(axes[0], rater_ids):
        data = [
            [
                score
                for response_id, score in medians_by_rater[rater_id].items()
                if response_by_id[response_id]["condition"] == condition
            ]
            for condition in conditions
        ]
        axis.boxplot(data, labels=conditions, showfliers=False)
        axis.set_title(rater_id)
        axis.set_ylabel("Median judge score")
        axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(paths["plots_dir"] / "scores_by_condition_and_rater.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Pairwise scatter
    for key in pairwise_metrics:
        left, right = key.split("__vs__")
        shared_ids = sorted(set(medians_by_rater[left]).intersection(medians_by_rater[right]))
        xs = [medians_by_rater[left][response_id] for response_id in shared_ids]
        ys = [medians_by_rater[right][response_id] for response_id in shared_ids]
        if not xs:
            continue
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(xs, ys, alpha=0.8)
        ax.plot([-4, 4], [-4, 4], linestyle="--", linewidth=1)
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.set_xlabel(left)
        ax.set_ylabel(right)
        ax.set_title(f"{left} vs {right}")
        ax.grid(alpha=0.2)
        fig.savefig(paths["plots_dir"] / f"pairwise_scatter_{left}_vs_{right}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Heatmaps
    if len(rater_ids) >= 2:
        for metric_name in ["quadratic_weighted_agreement", "mae"]:
            matrix = []
            for left in rater_ids:
                row = []
                for right in rater_ids:
                    if left == right:
                        row.append(1.0 if metric_name == "quadratic_weighted_agreement" else 0.0)
                    else:
                        key = f"{left}__vs__{right}" if f"{left}__vs__{right}" in pairwise_metrics else f"{right}__vs__{left}"
                        row.append(float(pairwise_metrics[key][metric_name]))
                matrix.append(row)
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            image = ax.imshow(matrix, cmap="viridis")
            ax.set_xticks(range(len(rater_ids)), rater_ids, rotation=30, ha="right")
            ax.set_yticks(range(len(rater_ids)), rater_ids)
            ax.set_title(metric_name)
            fig.colorbar(image, ax=ax)
            fig.tight_layout()
            fig.savefig(paths["plots_dir"] / f"heatmap_{metric_name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # Per-item spread
    top_rows = per_item_disagreement_rows[: min(50, len(per_item_disagreement_rows))]
    if top_rows:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        spreads = [row["rating_spread"] for row in top_rows]
        labels = [str(row["prompt_id"]) for row in top_rows]
        ax.bar(range(len(top_rows)), spreads)
        ax.set_xticks(range(len(top_rows)), labels, rotation=90)
        ax.set_ylabel("Rater spread")
        ax.set_title("Top per-item disagreement")
        fig.tight_layout()
        fig.savefig(paths["plots_dir"] / "per_item_spread.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Rater histograms
    fig, axes = plt.subplots(1, len(rater_ids), figsize=(4.5 * max(1, len(rater_ids)), 4.0), squeeze=False)
    bins = [x - 0.5 for x in range(-4, 6)]
    for axis, rater_id in zip(axes[0], rater_ids):
        values = list(medians_by_rater[rater_id].values())
        axis.hist(values, bins=bins, edgecolor="black")
        axis.set_xticks(list(range(-4, 5)))
        axis.set_title(rater_id)
        axis.set_xlabel("Median score")
    fig.tight_layout()
    fig.savefig(paths["plots_dir"] / "rater_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Within-rater repeat std
    fig, ax = plt.subplots(figsize=(6, 4))
    means = []
    for rater_id in rater_ids:
        item_stds = [statistics.pstdev(scores) if len(scores) > 1 else 0.0 for scores in repeats_by_rater[rater_id].values()]
        means.append(statistics.mean(item_stds) if item_stds else 0.0)
    ax.bar(rater_ids, means)
    ax.set_ylabel("Mean item repeat std")
    ax.set_title("Within-rater repeat stability")
    fig.tight_layout()
    fig.savefig(paths["plots_dir"] / "within_rater_repeat_std.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def upload_run_artifacts(config: NeuroticismJudgeAgreementConfig) -> str:
    """Upload the completed run dir to HF."""
    login_from_env()
    run_dir = get_run_dir(config)
    return upload_folder_to_dataset_repo(
        local_dir=run_dir,
        repo_id=config.hf_repo_id,
        path_in_repo=get_hf_run_prefix(config),
        commit_message=f"Upload neuroticism judge agreement run {build_run_key(config)}",
    )


def run_neuroticism_judge_agreement(
    config: NeuroticismJudgeAgreementConfig,
    *,
    mode: Literal["run", "analyze_only", "upload_only"] = "run",
) -> dict[str, Any]:
    """Run the neuroticism LLM-judge agreement harness."""
    _ensure_run_root(config)
    if mode in {"run", "analyze_only"}:
        ensure_local_run_from_hf(config)
    _write_run_manifest(config, mode)

    paths = _artifact_paths(config)
    result: dict[str, Any] = {
        "run_key": build_run_key(config),
        "run_dir": str(paths["run_dir"]),
        "hf_repo_id": config.hf_repo_id,
        "hf_run_prefix": get_hf_run_prefix(config),
        "mode": mode,
    }

    if mode == "upload_only":
        result["upload_url"] = upload_run_artifacts(config)
        return result

    prompt_rows = prepare_source_prompts(config)
    result["num_prompts"] = len(prompt_rows)

    if mode == "run":
        condition_prompts = build_neuroticism_system_prompts()
        expected_rows = len(prompt_rows) * config.responses_per_prompt
        for condition_name, system_prompt in condition_prompts.items():
            ensure_condition_responses(
                config,
                condition_name=condition_name,
                system_prompt=system_prompt,
                prompts_path=paths["source_prompts"],
                expected_rows=expected_rows,
            )

    response_rows = flatten_condition_responses(config)
    result["num_responses"] = len(response_rows)

    if mode == "run":
        progress = asyncio.run(run_judge_panel(config, response_rows))
        result["judge_progress"] = progress

    result["analysis"] = analyze_judge_panel(config)

    if config.upload and mode == "run":
        result["upload_url"] = upload_run_artifacts(config)

    return result


__all__ = [
    "JudgeRaterConfig",
    "NeuroticismJudgeAgreementConfig",
    "analyze_judge_panel",
    "build_neuroticism_system_prompts",
    "build_run_fingerprint",
    "build_run_key",
    "flatten_condition_responses",
    "get_hf_run_prefix",
    "get_run_dir",
    "prepare_source_prompts",
    "run_neuroticism_judge_agreement",
    "upload_run_artifacts",
]
