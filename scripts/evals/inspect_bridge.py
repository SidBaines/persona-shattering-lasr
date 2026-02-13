"""Inspect-native helpers for eval tasks and metric extraction."""

from __future__ import annotations

import asyncio
import importlib.util
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset

from scripts.evals.config import EvalModelConfig
from scripts.persona_metrics import (
    PersonaMetricContext,
    PersonaMetricsConfig,
    create_persona_metrics,
)


INSPECT_TASK_ALIASES: dict[str, str] = {
    "mmlu": "inspect_evals/mmlu",
}

KNOWN_INSPECT_MODEL_APIS = {
    "anthropic",
    "azureai",
    "bedrock",
    "cloudflare",
    "deepseek",
    "google",
    "grok",
    "groq",
    "hf",
    "mistral",
    "mockllm",
    "ollama",
    "openai",
    "openrouter",
    "together",
    "vertex",
}


def _normalize_key_component(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return normalized.strip("._-") or "task"


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def persona_median() -> Any:
    from inspect_ai.scorer import metric

    @metric("median")
    def _median() -> Any:
        def _compute(scores: list[Any]) -> float:
            values = [score.score.as_float() for score in scores]
            return float(statistics.median(values))

        return _compute

    return _median()


def persona_min() -> Any:
    from inspect_ai.scorer import metric

    @metric("min")
    def _min() -> Any:
        def _compute(scores: list[Any]) -> float:
            values = [score.score.as_float() for score in scores]
            return float(min(values))

        return _compute

    return _min()


def persona_max() -> Any:
    from inspect_ai.scorer import metric

    @metric("max")
    def _max() -> Any:
        def _compute(scores: list[Any]) -> float:
            values = [score.score.as_float() for score in scores]
            return float(max(values))

        return _compute

    return _max()


def persona_stdev() -> Any:
    from inspect_ai.scorer import metric

    @metric("stdev")
    def _stdev() -> Any:
        def _compute(scores: list[Any]) -> float:
            values = [score.score.as_float() for score in scores]
            if len(values) < 2:
                return 0.0
            return float(statistics.stdev(values))

        return _compute

    return _stdev()


def resolve_inspect_task_ref(task: str) -> str:
    """Resolve task aliases to concrete inspect task references."""
    task_ref = INSPECT_TASK_ALIASES.get(task, task)
    if task_ref.startswith("inspect_evals/") and importlib.util.find_spec("inspect_evals") is None:
        raise ValueError(
            "Inspect task alias requires the 'inspect_evals' package. "
            "Install it before running this suite (e.g., `uv add inspect-evals`)."
        )
    return task_ref


def normalize_inspect_model_ref(model_cfg: EvalModelConfig) -> str:
    """Resolve a model reference suitable for inspect_ai.eval(model=...)."""
    if model_cfg.kind == "lora":
        raise ValueError(
            "Inspect task suites currently require a native Inspect model reference; "
            "LoRA adapter targets are only supported in persona_metrics suites."
        )
    if model_cfg.inspect_model:
        return model_cfg.inspect_model

    candidate = model_cfg.model
    if "/" in candidate:
        api = candidate.split("/", 1)[0].lower()
        if api in KNOWN_INSPECT_MODEL_APIS:
            return candidate
    return f"hf/{candidate}"


def run_inspect_eval(
    *,
    tasks: Any,
    model_ref: str | None,
    eval_kwargs: dict[str, Any],
    log_dir: Path | None,
) -> list[dict[str, Any]]:
    """Execute inspect_ai.eval and return serializable EvalLog payloads."""
    from inspect_ai import eval as inspect_eval

    blocked = {"task", "tasks", "model", "log_dir"}
    overlap = blocked.intersection(eval_kwargs.keys())
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"Inspect eval kwargs must not include reserved fields: {names}")

    kwargs = dict(eval_kwargs)
    kwargs.setdefault("display", "none")
    if log_dir is not None:
        kwargs.setdefault("log_dir", str(log_dir))
    if model_ref is not None:
        kwargs["model"] = model_ref

    logs = inspect_eval(tasks=tasks, **kwargs)
    payloads: list[dict[str, Any]] = []
    for log in logs:
        if hasattr(log, "model_dump"):
            payloads.append(log.model_dump())
        else:
            payloads.append({"result": str(log)})
    return payloads


def build_persona_inspect_task(
    *,
    dataset: Dataset,
    metrics_config: PersonaMetricsConfig,
    scorer_name: str = "persona_metrics",
) -> Any:
    """Build an Inspect Task that scores pre-generated responses."""
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ModelOutput
    from inspect_ai.scorer import Score, Target, mean, scorer
    from inspect_ai.solver import Generate, Solver, TaskState, solver

    run_metadata = {
        "response_column": metrics_config.response_column,
        "question_column": metrics_config.question_column,
    }
    records = dataset.to_list()
    samples: list[Sample] = []
    for index, record in enumerate(records):
        response = record.get(metrics_config.response_column)
        if not isinstance(response, str):
            raise ValueError(
                "Persona metrics suite requires string responses in column "
                f"'{metrics_config.response_column}'."
            )
        question_value = (
            record.get(metrics_config.question_column)
            if metrics_config.question_column
            else None
        )
        question = question_value if isinstance(question_value, str) else None
        sample_input = question or ""
        sample_id = record.get("id", index)
        if not isinstance(sample_id, (str, int)):
            sample_id = index

        samples.append(
            Sample(
                input=sample_input,
                target="",
                id=sample_id,
                metadata={
                    "record": record,
                    "response": response,
                    "question": question,
                },
            )
        )

    @solver(name="persona_replay_response")
    def _replay_response() -> Solver:
        async def _solve(state: TaskState, generate: Generate) -> TaskState:
            sample_metadata = state.metadata or {}
            response = sample_metadata.get("response")
            if not isinstance(response, str):
                raise ValueError("Sample metadata is missing a string response.")
            state.output = ModelOutput.from_content(model=str(state.model), content=response)
            state.messages.append(state.output.message)
            state.completed = True
            return state

        return _solve

    @scorer(
        metrics={
            "*": [
                mean(),
                persona_median(),
                persona_min(),
                persona_max(),
                persona_stdev(),
            ]
        },
        name=scorer_name,
    )
    def _persona_scorer() -> Any:
        metric_instances = create_persona_metrics(metrics_config)
        metric_semaphores = [
            (
                asyncio.Semaphore(max(1, metric.judge_config.max_concurrent))
                if getattr(metric, "judge_config", None) is not None
                else None
            )
            for metric in metric_instances
        ]

        async def _score(state: TaskState, target: Target) -> Score:
            sample_metadata = state.metadata or {}
            response = state.output.completion
            question = sample_metadata.get("question")
            if not isinstance(question, str):
                question = None
            record = sample_metadata.get("record")
            if not isinstance(record, dict):
                record = {}

            context = PersonaMetricContext(
                response=response,
                question=question,
                record=record,
                metadata=run_metadata,
            )

            merged: dict[str, float | int | str] = {}
            for metric, semaphore in zip(metric_instances, metric_semaphores):
                if semaphore is None:
                    metric_values = await metric.evaluate_async(
                        response,
                        question,
                        context=context,
                    )
                else:
                    async with semaphore:
                        metric_values = await metric.evaluate_async(
                            response,
                            question,
                            context=context,
                        )
                merged.update(metric_values)

            numeric = {key: float(value) for key, value in merged.items() if _is_numeric(value)}
            if not numeric:
                numeric = {"__no_numeric_metrics": 0.0}

            return Score(
                value=numeric,
                metadata={"persona_metrics": merged},
            )

        return _score

    return Task(
        name="persona_metrics",
        dataset=samples,
        solver=_replay_response(),
        scorer=_persona_scorer(),
    )


def extract_eval_metrics(eval_logs: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Extract numeric summary metrics from inspect eval logs."""
    extracted: dict[str, float] = {}
    for index, payload in enumerate(eval_logs):
        eval_section = payload.get("eval", {})
        task_name = (
            eval_section.get("task")
            or eval_section.get("task_file")
            or f"task_{index}"
        )
        task_key = _normalize_key_component(str(task_name))

        results = payload.get("results")
        if not isinstance(results, dict):
            continue
        for score in results.get("scores", []):
            score_name = _normalize_key_component(
                str(score.get("name") or score.get("scorer") or "score")
            )
            reducer = score.get("reducer")
            if reducer:
                score_name = f"{score_name}.{_normalize_key_component(str(reducer))}"

            metrics = score.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            for metric_name, metric_payload in metrics.items():
                if not isinstance(metric_payload, dict):
                    continue
                value = metric_payload.get("value")
                if not _is_numeric(value):
                    continue
                number = float(value)
                if not math.isfinite(number):
                    continue
                key = f"{task_key}.{score_name}.{_normalize_key_component(str(metric_name))}"
                extracted[key] = number
    return extracted


def extract_persona_scored_records(
    eval_logs: Iterable[dict[str, Any]],
    *,
    metrics_key: str,
    scorer_name: str = "persona_metrics",
) -> list[dict[str, Any]]:
    """Reconstruct scored records from Inspect sample-level score payloads."""
    records: list[dict[str, Any]] = []
    for payload in eval_logs:
        for sample in payload.get("samples", []) or []:
            if not isinstance(sample, dict):
                continue
            sample_metadata = sample.get("metadata") or {}
            if not isinstance(sample_metadata, dict):
                sample_metadata = {}
            base_record = sample_metadata.get("record")
            record = dict(base_record) if isinstance(base_record, dict) else {}

            sample_scores = sample.get("scores") or {}
            if not isinstance(sample_scores, dict):
                sample_scores = {}
            scorer_payload = sample_scores.get(scorer_name)
            if not isinstance(scorer_payload, dict) and len(sample_scores) == 1:
                scorer_payload = next(iter(sample_scores.values()))
            if not isinstance(scorer_payload, dict):
                scorer_payload = {}

            value_payload = scorer_payload.get("value")
            numeric_values = value_payload if isinstance(value_payload, dict) else {}
            metadata_payload = scorer_payload.get("metadata")
            metadata_dict = metadata_payload if isinstance(metadata_payload, dict) else {}
            persona_values = metadata_dict.get("persona_metrics")
            full_values = persona_values if isinstance(persona_values, dict) else {}

            merged_values = dict(full_values)
            for key, value in numeric_values.items():
                merged_values[key] = value

            existing = record.get(metrics_key)
            if isinstance(existing, dict):
                record[metrics_key] = {**existing, **merged_values}
            else:
                record[metrics_key] = merged_values
            records.append(record)
    return records
