"""Bridge between this project's eval framework and inspect-ai.

Persona metrics are evaluated through inspect-ai via **two paths**:

**Native path** (``build_native_persona_inspect_task``):
  For models that inspect-ai can load directly (anything accessible via
  ``hf/<model>`` — base models, or fine-tunes merged and pushed to
  HuggingFace Hub).  This is a standard inspect Task: the ``Generate()``
  solver calls the model, and a custom scorer runs persona metrics on the
  generated responses.  This is the preferred path because it follows
  inspect-ai conventions and produces complete inspect logs including
  generation traces.

**Replay path** (``build_replay_persona_inspect_task``):
  For local LoRA adapters that inspect-ai's ``hf/`` model provider cannot
  load.  Responses are first generated using our own LoRA-aware inference
  pipeline (``scripts.inference``), then replayed into an inspect Task via
  a ``_replay_response`` solver that injects pre-generated text without
  calling the model.  A ``mockllm/persona`` model reference is used to
  satisfy inspect-ai's model requirement.  The same persona scorer is
  shared between both paths.

Both paths produce identical scoring output — the only difference is
whether inspect-ai or our inference pipeline drives generation.

For external benchmarks (e.g. MMLU via ``inspect_evals``), use
``resolve_inspect_task_ref`` + ``run_inspect_eval`` directly.
"""

from __future__ import annotations

import asyncio
import importlib.util
import math
import logging
import statistics
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset

from scripts.common.config import GenerationConfig
from scripts.evals.config import EvalModelConfig, normalize_component
from scripts.persona_metrics import (
    PersonaMetricContext,
    PersonaMetricsConfig,
    create_persona_metrics,
)

logger = logging.getLogger(__name__)


INSPECT_TASK_ALIASES: dict[str, str] = {
    # inspect_evals>=0.3 exposes mmlu tasks as mmlu_0_shot/mmlu_5_shot.
    "mmlu": "inspect_evals/mmlu_0_shot",
    # Backward compatibility for older docs/CLI examples.
    "inspect_evals/mmlu": "inspect_evals/mmlu_0_shot",
    "mmlu_pro": "inspect_evals/mmlu_pro",
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


TASKS_REQUIRING_UNIQUE_ID_FIX = {
    "inspect_evals/mmlu_0_shot",
    "inspect_evals/mmlu_5_shot",
}


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _clone_sample_with_id(sample: Any, sample_id: str) -> Any:
    """Clone an inspect sample while overriding its ID."""
    if hasattr(sample, "model_copy"):
        return sample.model_copy(update={"id": sample_id})
    if hasattr(sample, "copy"):
        return sample.copy(update={"id": sample_id})

    from inspect_ai.dataset import Sample

    return Sample(
        input=sample.input,
        choices=getattr(sample, "choices", None),
        target=getattr(sample, "target", ""),
        id=sample_id,
        metadata=getattr(sample, "metadata", None),
        sandbox=getattr(sample, "sandbox", None),
        files=getattr(sample, "files", None),
        setup=getattr(sample, "setup", None),
    )


def _with_unique_sample_ids(task: Any) -> tuple[Any, int]:
    """Return (task, num_rewritten_ids), rewriting duplicate sample IDs if needed."""
    if not hasattr(task, "dataset"):
        return task, 0
    dataset = task.dataset
    if not hasattr(dataset, "__iter__"):
        return task, 0

    samples = list(dataset)
    if not samples:
        return task, 0

    counts: dict[str, int] = {}
    rewritten = 0
    unique_samples: list[Any] = []

    for idx, sample in enumerate(samples):
        original = getattr(sample, "id", None)
        key = str(original) if original is not None else f"sample-{idx}"
        seen = counts.get(key, 0)
        counts[key] = seen + 1
        if seen == 0:
            unique_samples.append(sample)
            continue

        rewritten += 1
        deduped_id = f"{key}__dup{seen}"
        unique_samples.append(_clone_sample_with_id(sample, deduped_id))

    if rewritten == 0:
        return task, 0

    from inspect_ai.dataset import MemoryDataset
    from inspect_ai import Task

    fixed_dataset = MemoryDataset(samples=unique_samples, name=getattr(dataset, "name", None))
    fixed_task = Task(
        dataset=fixed_dataset,
        setup=getattr(task, "setup", None),
        solver=getattr(task, "solver"),
        cleanup=getattr(task, "cleanup", None),
        scorer=getattr(task, "scorer", None),
        metrics=getattr(task, "metrics", None),
        model=getattr(task, "model", None),
        config=getattr(task, "config"),
        model_roles=getattr(task, "model_roles", None),
        sandbox=getattr(task, "sandbox", None),
        approval=getattr(task, "approval", None),
        epochs=getattr(task, "epochs", None),
        fail_on_error=getattr(task, "fail_on_error", None),
        continue_on_fail=getattr(task, "continue_on_fail", None),
        message_limit=getattr(task, "message_limit", None),
        token_limit=getattr(task, "token_limit", None),
        time_limit=getattr(task, "time_limit", None),
        working_limit=getattr(task, "working_limit", None),
        early_stopping=getattr(task, "early_stopping", None),
        display_name=getattr(task, "display_name", None),
        name=getattr(task, "name", None),
        version=getattr(task, "version", 0),
        metadata=getattr(task, "metadata", None),
    )
    return fixed_task, rewritten


def _build_inspect_task_with_runtime_fixes(task_ref: str) -> Any:
    """Load inspect_evals tasks that need runtime compatibility fixes."""
    if task_ref == "inspect_evals/mmlu_0_shot":
        from inspect_evals.mmlu.mmlu import mmlu_0_shot

        task = mmlu_0_shot()
    elif task_ref == "inspect_evals/mmlu_5_shot":
        from inspect_evals.mmlu.mmlu import mmlu_5_shot

        task = mmlu_5_shot()
    else:
        return task_ref

    task, rewritten = _with_unique_sample_ids(task)
    if rewritten:
        logger.warning(
            "Rewrote %d duplicate sample IDs for %s to satisfy inspect-ai uniqueness.",
            rewritten,
            task_ref,
        )
    return task


# ---------------------------------------------------------------------------
# Custom inspect-ai metrics (mean is built-in; we add median/min/max/stdev)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Task/model resolution helpers
# ---------------------------------------------------------------------------


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
    """Resolve a model reference suitable for ``inspect_ai.eval(model=...)``.

    For LoRA models without an explicit ``inspect_model``, this raises
    because inspect-ai's ``hf/`` provider cannot load a separate adapter.
    Use the replay path (``build_replay_persona_inspect_task``) instead,
    or merge and push the adapter to HuggingFace Hub first.
    """
    if model_cfg.inspect_model:
        return model_cfg.inspect_model
    if model_cfg.kind == "lora":
        raise ValueError(
            "Inspect task suites require an Inspect-native model reference for LoRA "
            "targets. Provide EvalModelConfig.inspect_model (or --lora-inspect-model) "
            "if your Inspect provider supports LoRA adapters; otherwise omit inspect "
            "task suites for this model."
        )

    candidate = model_cfg.model
    if "/" in candidate:
        api = candidate.split("/", 1)[0].lower()
        if api in KNOWN_INSPECT_MODEL_APIS:
            return candidate
    return f"hf/{candidate}"


# ---------------------------------------------------------------------------
# Inspect eval execution
# ---------------------------------------------------------------------------


def run_inspect_eval(
    *,
    tasks: Any,
    model_ref: str | None,
    eval_kwargs: dict[str, Any],
    log_dir: Path | None,
) -> list[dict[str, Any]]:
    """Execute ``inspect_ai.eval`` and return serializable EvalLog payloads."""
    from inspect_ai import eval as inspect_eval

    blocked = {"task", "tasks", "model", "log_dir"}
    overlap = blocked.intersection(eval_kwargs.keys())
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"Inspect eval kwargs must not include reserved fields: {names}")

    kwargs = dict(eval_kwargs)
    # Use a visible default so long-running tasks (e.g., MMLU) don't look hung.
    # Callers can still override via eval_kwargs (e.g., {"display": "none"}).
    kwargs.setdefault("display", "plain")
    if log_dir is not None:
        kwargs.setdefault("log_dir", str(log_dir))
    if model_ref is not None:
        kwargs["model"] = model_ref

    task_input = tasks
    if isinstance(tasks, str) and tasks in TASKS_REQUIRING_UNIQUE_ID_FIX:
        task_input = _build_inspect_task_with_runtime_fixes(tasks)

    logs = inspect_eval(tasks=task_input, **kwargs)
    payloads: list[dict[str, Any]] = []
    for log in logs:
        if hasattr(log, "model_dump"):
            payloads.append(log.model_dump())
        else:
            payloads.append({"result": str(log)})
    return payloads


# ---------------------------------------------------------------------------
# Shared persona scorer (used by both native and replay paths)
# ---------------------------------------------------------------------------


def _build_persona_scorer(
    metrics_config: PersonaMetricsConfig,
    scorer_name: str = "persona_metrics",
) -> Any:
    """Build an inspect-ai scorer that runs persona metrics on model output.

    This scorer is shared between the native and replay paths.  It reads
    the model's generated text from ``state.output.completion`` and the
    original question from ``state.metadata["question"]``, then runs each
    configured persona metric and returns a composite ``Score``.
    """
    from inspect_ai.scorer import Score, Target, mean, scorer

    run_metadata = {
        "response_column": metrics_config.response_column,
        "question_column": metrics_config.question_column,
    }

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

        async def _score(state: Any, target: Target) -> Score:
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

    return _persona_scorer()


# ---------------------------------------------------------------------------
# Native persona Task — inspect-ai drives generation
# ---------------------------------------------------------------------------


def build_native_persona_inspect_task(
    *,
    dataset: Dataset,
    metrics_config: PersonaMetricsConfig,
    generation_config: GenerationConfig,
    scorer_name: str = "persona_metrics",
) -> Any:
    """Build an inspect Task where inspect-ai generates responses natively.

    This is the **preferred path** for any model that inspect-ai can load
    directly (base HF models, or fine-tunes merged and pushed to HF Hub).
    Inspect handles model loading and generation via its ``Generate()``
    solver, and the persona metrics scorer evaluates the output.

    Args:
        dataset: A dataset of questions/prompts (must have a question column).
        metrics_config: Which persona metrics to score and how.
        generation_config: Generation parameters (temperature, max_tokens, etc.)
            forwarded to inspect's GenerateConfig.
        scorer_name: Name for the inspect scorer (default: ``"persona_metrics"``).

    Returns:
        An ``inspect_ai.Task`` ready for ``inspect_ai.eval()``.
    """
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.solver import Generate, generate

    question_column = metrics_config.question_column or "question"
    records = dataset.to_list()
    samples: list[Sample] = []
    for index, record in enumerate(records):
        question_value = record.get(question_column)
        question = question_value if isinstance(question_value, str) else ""
        sample_id = record.get("id", index)
        if not isinstance(sample_id, (str, int)):
            sample_id = index

        samples.append(
            Sample(
                input=question,
                target="",
                id=sample_id,
                metadata={
                    "record": record,
                    "question": question or None,
                },
            )
        )

    gen_kwargs: dict[str, Any] = {}
    if generation_config.max_new_tokens:
        gen_kwargs["max_tokens"] = generation_config.max_new_tokens
    if generation_config.temperature is not None:
        gen_kwargs["temperature"] = generation_config.temperature
    if generation_config.top_p is not None:
        gen_kwargs["top_p"] = generation_config.top_p

    return Task(
        name="persona_metrics",
        dataset=samples,
        solver=generate(**gen_kwargs) if gen_kwargs else Generate(),
        scorer=_build_persona_scorer(metrics_config, scorer_name),
    )


# ---------------------------------------------------------------------------
# Replay persona Task — pre-generated responses scored through inspect
# ---------------------------------------------------------------------------


def build_replay_persona_inspect_task(
    *,
    dataset: Dataset,
    metrics_config: PersonaMetricsConfig,
    scorer_name: str = "persona_metrics",
) -> Any:
    """Build an inspect Task that scores pre-generated responses.

    This is the **fallback path** for local LoRA adapters that inspect-ai's
    ``hf/`` model provider cannot load.  Responses must already be present
    in the dataset (in the column named by ``metrics_config.response_column``).
    A ``_replay_response`` solver injects these into inspect's pipeline
    without calling the model.

    Use ``model_ref="mockllm/persona"`` when calling ``run_inspect_eval``
    with the task returned by this function — it satisfies inspect's model
    requirement without loading a real model.

    Args:
        dataset: A dataset that already contains generated responses.
        metrics_config: Which persona metrics to score and how.
        scorer_name: Name for the inspect scorer (default: ``"persona_metrics"``).

    Returns:
        An ``inspect_ai.Task`` ready for ``inspect_ai.eval()``.
    """
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ModelOutput
    from inspect_ai.solver import Generate, Solver, TaskState, solver

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

    return Task(
        name="persona_metrics",
        dataset=samples,
        solver=_replay_response(),
        scorer=_build_persona_scorer(metrics_config, scorer_name),
    )


# ---------------------------------------------------------------------------
# Metric extraction from inspect logs
# ---------------------------------------------------------------------------


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
        task_key = normalize_component(str(task_name), fallback="task")

        results = payload.get("results")
        if not isinstance(results, dict):
            continue
        for score in results.get("scores", []):
            score_name = normalize_component(
                str(score.get("name") or score.get("scorer") or "score"),
                fallback="score",
            )
            reducer = score.get("reducer")
            if reducer:
                score_name = f"{score_name}.{normalize_component(str(reducer), fallback='reducer')}"

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
                key = f"{task_key}.{score_name}.{normalize_component(str(metric_name), fallback='metric')}"
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
