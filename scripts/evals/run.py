"""Core runner for end-to-end model evals."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import Dataset

from scripts.data_loading import format_for_inference, load_dataset_from_config
from scripts.evals.config import (
    EvalModelConfig,
    EvalsConfig,
    EvalsResult,
    InspectTaskSuiteConfig,
    ModelEvalResult,
    PersonaMetricsSuiteConfig,
    SuiteEvalResult,
)
from scripts.evals.inspect_bridge import (
    build_persona_inspect_task,
    extract_eval_metrics,
    extract_persona_scored_records,
    normalize_inspect_model_ref,
    resolve_inspect_task_ref,
    run_inspect_eval,
)
from scripts.inference import InferenceConfig, LocalProviderConfig, run_inference
from scripts.persona_metrics import PersonaMetricsConfig
from scripts.utils import setup_logging, write_jsonl


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _safe_model_id(model_cfg: EvalModelConfig, idx: int) -> str:
    if model_cfg.id:
        return model_cfg.id
    base = model_cfg.model.replace("/", "_").replace(":", "_")
    if model_cfg.kind == "lora":
        adapter = (model_cfg.adapter_path or "adapter").replace("/", "_").replace(":", "_")
        return f"lora-{idx}-{base}-{adapter}"
    return f"base-{idx}-{base}"


def _normalize_suite_component(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return normalized.strip("._-") or "suite"


def _stable_suite_id(suite: PersonaMetricsSuiteConfig | InspectTaskSuiteConfig) -> str:
    if suite.suite_id:
        return _normalize_suite_component(suite.suite_id)
    payload = suite.model_dump(exclude={"suite_id"})
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:10]
    return f"auto-{digest}"


def _suite_artifact_dirname(display_name: str, suite_id: str) -> str:
    return f"{_normalize_suite_component(display_name)}__{suite_id}"


def _load_question_dataset(config: EvalsConfig, dataset: Dataset | None) -> Dataset:
    if dataset is None:
        dataset = load_dataset_from_config(config.dataset)
    return format_for_inference(dataset, question_column=config.question_column)


def _generate_responses_for_model(
    model_cfg: EvalModelConfig,
    dataset: Dataset,
    evals_config: EvalsConfig,
) -> Dataset:
    infer_cfg = InferenceConfig(
        model=model_cfg.model,
        provider="local",
        generation=evals_config.generation,
        local=LocalProviderConfig(
            dtype=model_cfg.dtype,
            device_map=model_cfg.device_map,
            revision=model_cfg.revision,
            prompt_format="auto",
            adapter_path=model_cfg.adapter_path if model_cfg.kind == "lora" else None,
        ),
    )
    result_dataset, _ = run_inference(infer_cfg, dataset=dataset)
    return result_dataset


def _resolve_inspect_task_name(task: str, task_name: str | None) -> str:
    if task_name:
        return task_name
    if task == "mmlu":
        return "mmlu"
    task_component = task
    if "@" in task_component:
        task_component = task_component.split("@", 1)[1]
    task_component = task_component.rsplit("/", 1)[-1]
    return task_component.replace(".", "_")


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}.{key}": value for key, value in metrics.items()}


def run_evals(
    config: EvalsConfig,
    dataset: Dataset | None = None,
) -> tuple[Dataset, EvalsResult]:
    """Run end-to-end eval suites for one or more model targets."""
    logger = setup_logging()
    started_at = datetime.now()

    output_dir = config.output_dir or Path("scratch") / (
        f"evals-{started_at.strftime('%Y%m%d-%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    question_dataset: Dataset | None = None
    model_results: list[ModelEvalResult] = []
    combined_records: list[dict[str, Any]] = []
    leaderboard_by_model: dict[str, dict[str, Any]] = {}

    for model_index, model_cfg in enumerate(config.models):
        model_id = _safe_model_id(model_cfg, model_index)
        logger.info("Running evals for model %s (%s)", model_id, model_cfg.kind)
        model_output_dir = output_dir / model_id
        model_output_dir.mkdir(parents=True, exist_ok=True)

        responses_dataset: Dataset | None = None
        model_result = ModelEvalResult(
            model_id=model_id,
            kind=model_cfg.kind,
            model=model_cfg.model,
            adapter_path=model_cfg.adapter_path,
        )

        leaderboard_by_model[model_id] = {"model_id": model_id}

        for suite in config.suites:
            if isinstance(suite, PersonaMetricsSuiteConfig):
                suite_name = "persona_metrics"
                suite_id = _stable_suite_id(suite)
                suite_dir = model_output_dir / _suite_artifact_dirname(
                    suite_name, suite_id
                )
                suite_dir.mkdir(parents=True, exist_ok=True)
                suite_result = SuiteEvalResult(
                    suite_type=suite.type,
                    suite_name=suite_name,
                    suite_id=suite_id,
                    model_id=model_id,
                )

                try:
                    if responses_dataset is None:
                        if question_dataset is None:
                            question_dataset = _load_question_dataset(config, dataset)
                        responses_dataset = _generate_responses_for_model(
                            model_cfg=model_cfg,
                            dataset=question_dataset,
                            evals_config=config,
                        )

                    responses_path = write_jsonl(
                        responses_dataset.to_list(),
                        suite_dir / "responses.jsonl",
                    )

                    metrics_config = PersonaMetricsConfig(
                        evaluations=suite.evaluations,
                        response_column=suite.response_column,
                        question_column=suite.question_column,
                        metrics_key=suite.metrics_key,
                        judge=suite.judge,
                    )
                    persona_task = build_persona_inspect_task(
                        dataset=responses_dataset,
                        metrics_config=metrics_config,
                        scorer_name="persona_metrics",
                    )
                    inspect_logs_dir = suite_dir / "inspect_logs"
                    inspect_payloads = run_inspect_eval(
                        tasks=persona_task,
                        model_ref=None,
                        eval_kwargs={},
                        log_dir=inspect_logs_dir,
                    )

                    suite_metrics = extract_eval_metrics(inspect_payloads)
                    suite_result.aggregates = dict(suite_metrics)
                    suite_result.num_samples = len(responses_dataset)

                    scored_records = extract_persona_scored_records(
                        inspect_payloads,
                        metrics_key=suite.metrics_key,
                        scorer_name="persona_metrics",
                    )
                    scored_dataset = Dataset.from_list(scored_records)
                    scored_path = write_jsonl(
                        scored_dataset.to_list(),
                        suite_dir / "scored.jsonl",
                    )

                    suite_result_path = _write_json(
                        suite_dir / "suite_result.json",
                        {
                            "suite_type": suite.type,
                            "suite_name": suite_name,
                            "suite_id": suite_id,
                            "model_id": model_id,
                            "inspect_payloads": inspect_payloads,
                            "aggregates": suite_metrics,
                        },
                    )

                    suite_result.artifacts = {
                        "responses": str(responses_path),
                        "scored": str(scored_path),
                        "suite_result": str(suite_result_path),
                        "inspect_logs_dir": str(inspect_logs_dir),
                    }

                    prefixed = _prefix_metrics(
                        f"persona_metrics.{suite_id}",
                        suite_metrics,
                    )
                    leaderboard_by_model[model_id].update(prefixed)

                    for row in scored_dataset:
                        out = dict(row)
                        out["model_id"] = model_id
                        out["suite_type"] = suite.type
                        out["suite_id"] = suite_id
                        out["suite_name"] = suite_name
                        combined_records.append(out)
                except Exception as exc:
                    suite_result.error = str(exc)
                    if not config.continue_on_error:
                        raise

                model_result.suites.append(suite_result)
                continue

            if isinstance(suite, InspectTaskSuiteConfig):
                task_name = _resolve_inspect_task_name(suite.task, suite.task_name)
                suite_name = f"inspect.{task_name}"
                suite_id = _stable_suite_id(suite)
                suite_dir = model_output_dir / _suite_artifact_dirname(
                    suite_name, suite_id
                )
                suite_dir.mkdir(parents=True, exist_ok=True)
                suite_result = SuiteEvalResult(
                    suite_type=suite.type,
                    suite_name=suite_name,
                    suite_id=suite_id,
                    model_id=model_id,
                )

                try:
                    task_ref = resolve_inspect_task_ref(suite.task)
                    model_ref = normalize_inspect_model_ref(model_cfg)
                    inspect_logs_dir = suite_dir / "inspect_logs"
                    inspect_payloads = run_inspect_eval(
                        tasks=task_ref,
                        model_ref=model_ref,
                        eval_kwargs=dict(suite.eval_kwargs),
                        log_dir=inspect_logs_dir,
                    )
                    suite_metrics = extract_eval_metrics(inspect_payloads)
                    suite_result.aggregates = dict(suite_metrics)
                    suite_result.num_samples = sum(
                        len(payload.get("samples", []) or [])
                        for payload in inspect_payloads
                    )

                    suite_json_path = _write_json(
                        suite_dir / "suite_result.json",
                        {
                            "suite_type": suite.type,
                            "suite_name": suite_name,
                            "suite_id": suite_id,
                            "model_id": model_id,
                            "task": suite.task,
                            "task_ref": task_ref,
                            "model_ref": model_ref,
                            "eval_kwargs": suite.eval_kwargs,
                            "inspect_payloads": inspect_payloads,
                            "aggregates": suite_metrics,
                        },
                    )

                    suite_result.artifacts = {
                        "suite_result": str(suite_json_path),
                        "inspect_logs_dir": str(inspect_logs_dir),
                    }
                    suite_result.metadata = {
                        "task": suite.task,
                        "task_ref": task_ref,
                        "model_ref": model_ref,
                        "eval_kwargs": suite.eval_kwargs,
                    }

                    leaderboard_by_model[model_id].update(
                        _prefix_metrics(f"inspect.{task_name}.{suite_id}", suite_metrics)
                    )

                    for payload_index, payload in enumerate(inspect_payloads):
                        samples = payload.get("samples", []) or []
                        if samples:
                            for sample in samples:
                                combined_records.append(
                                    {
                                        "model_id": model_id,
                                        "suite_type": suite.type,
                                        "suite_name": suite_name,
                                        "suite_id": suite_id,
                                        "inspect_sample": sample,
                                    }
                                )
                        else:
                            combined_records.append(
                                {
                                    "model_id": model_id,
                                    "suite_type": suite.type,
                                    "suite_name": suite_name,
                                    "suite_id": suite_id,
                                    "inspect_payload_index": payload_index,
                                    "inspect_payload": payload,
                                }
                            )
                except Exception as exc:
                    suite_result.error = str(exc)
                    if not config.continue_on_error:
                        raise

                model_result.suites.append(suite_result)
                continue

        model_results.append(model_result)

    leaderboard = [leaderboard_by_model[key] for key in sorted(leaderboard_by_model)]
    leaderboard_path = _write_json(output_dir / "leaderboard.json", {"rows": leaderboard})
    summary_payload = {
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now().isoformat(),
        "num_models": len(config.models),
        "num_suites": len(config.suites),
        "num_rows": len(combined_records),
        "leaderboard_path": str(leaderboard_path),
        "models": [model.model_dump() for model in config.models],
        "suites": [suite.model_dump() for suite in config.suites],
    }
    summary_path = _write_json(output_dir / "summary.json", summary_payload)

    result = EvalsResult(
        output_dir=output_dir,
        num_models=len(config.models),
        num_suites=len(config.suites),
        num_rows=len(combined_records),
        model_results=model_results,
        leaderboard=leaderboard,
        summary_path=summary_path,
    )

    return Dataset.from_list(combined_records), result
