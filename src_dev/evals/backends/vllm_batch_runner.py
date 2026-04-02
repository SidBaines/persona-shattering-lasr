"""vLLM batch-call backend for benchmark evals.

Bypasses Inspect's per-sample generate loop by collecting all prompts from a
benchmark task, calling ``vllm.LLM.chat()`` in one shot, then running the
Inspect scorer on the results.  Produces an EvalLog JSON file compatible with
the downstream analysis pipeline (``analyze_results``, ``skip_completed``,
auto-analyze, upload).

This is ~10-20x faster than going through Inspect's HF provider for simple
MCQ benchmarks (TRAIT, MMLU, BFI) where the solver is just ``generate()``.

Usage::

    from src_dev.evals.backends.vllm_batch_runner import run_benchmark_eval_vllm_batch

    result = run_benchmark_eval_vllm_batch(
        spec=benchmark_spec,
        vllm_variant_provider=variant_provider,
        run_dir=run_dir,
        model_name="meta-llama/Llama-3.1-8B-Instruct",
    )
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog, write_eval_log
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)
from inspect_ai.model._model_output import (
    ChatCompletionChoice,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.scorer import Score, Target
from inspect_ai.solver import TaskState

from src_dev.evals.backends.inspect_runner import InspectRunResult
from src_dev.evals.config import InspectBenchmarkSpec
from src_dev.evals.inspect_benchmarks import build_benchmark_task


def _sample_to_chat_messages(sample: Sample) -> list[dict[str, str]]:
    """Convert an Inspect Sample's input to a list of chat message dicts.

    Handles both string inputs (wrapped as a user message) and pre-structured
    chat message lists.

    Args:
        sample: Inspect Sample with input field.

    Returns:
        List of ``{"role": ..., "content": ...}`` dicts suitable for
        ``vllm.LLM.chat()``.
    """
    if isinstance(sample.input, str):
        return [{"role": "user", "content": sample.input}]

    messages: list[dict[str, str]] = []
    for msg in sample.input:
        if isinstance(msg, ChatMessageSystem):
            messages.append({"role": "system", "content": msg.text})
        elif isinstance(msg, ChatMessageUser):
            messages.append({"role": "user", "content": msg.text})
        elif isinstance(msg, ChatMessageAssistant):
            messages.append({"role": "assistant", "content": msg.text or ""})
        else:
            # Fallback for other message types
            role = getattr(msg, "role", "user")
            content = msg.text if hasattr(msg, "text") else str(getattr(msg, "content", ""))
            messages.append({"role": role, "content": content})
    return messages


def _build_task_state(
    sample: Sample,
    model_output_text: str,
    model_name: str,
) -> TaskState:
    """Build a minimal TaskState from a sample and its model output.

    This is the minimum needed for Inspect scorers to work: the scorer
    receives a TaskState with the conversation history and model output.

    Args:
        sample: Original Inspect Sample.
        model_output_text: Text response from the model.
        model_name: Model name for the output metadata.

    Returns:
        TaskState with messages and output populated.
    """
    # Build input messages from sample
    if isinstance(sample.input, str):
        input_messages: list[ChatMessage] = [ChatMessageUser(content=sample.input)]
    else:
        input_messages = list(sample.input)

    # Add the assistant response
    assistant_msg = ChatMessageAssistant(
        content=model_output_text,
        model=model_name,
        source="generate",
    )
    all_messages = input_messages + [assistant_msg]

    model_output = ModelOutput(
        model=model_name,
        choices=[
            ChatCompletionChoice(message=assistant_msg),
        ],
        usage=ModelUsage(),
    )

    return TaskState(
        model=model_name,
        sample_id=sample.id or 0,
        epoch=1,
        input=sample.input,
        messages=all_messages,
        output=model_output,
        metadata=sample.metadata or {},
    )


async def _score_sample(
    scorer: Any,
    state: TaskState,
    target: Target,
) -> Score:
    """Run an Inspect scorer on a single sample.

    Args:
        scorer: An instantiated Inspect scorer (the callable returned by
            e.g. ``includes()``).
        state: TaskState with model output.
        target: Target for scoring.

    Returns:
        Score from the scorer.
    """
    result = await scorer(state, target)
    return result


def _make_eval_log_dict(
    *,
    task_name: str,
    model_name: str,
    samples: list[dict[str, Any]],
    scores_summary: dict[str, Any],
    total_seconds: float,
    scorer_name: str,
    log_path: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an EvalLog-compatible dict for JSON serialization.

    Rather than constructing an EvalLog Pydantic model (which requires matching
    exact internal schema versions), we build the JSON dict directly with the
    fields that the downstream analysis pipeline reads.

    Args:
        task_name: Name of the benchmark task.
        model_name: Model identifier string.
        samples: List of per-sample dicts with scoring info.
        scores_summary: Aggregated scorer metrics.
        total_seconds: Wall-clock time for the eval.
        scorer_name: Name of the scorer used.
        log_path: Where the log will be written.
        config: Optional generation config dict.

    Returns:
        Dict serializable as JSON, compatible with Inspect log readers.
    """
    now = datetime.now(timezone.utc).isoformat()
    scored_count = sum(1 for s in samples if s.get("scores", {}).get(scorer_name, {}).get("value") == "C")
    unscored_count = len(samples) - scored_count

    return {
        "version": 2,
        "status": "success",
        "eval": {
            "task": task_name,
            "task_version": 0,
            "task_file": "",
            "task_id": f"{task_name}@0",
            "run_id": now.replace(":", "").replace("-", "")[:15],
            "created": now,
            "model": model_name,
            "model_base_url": None,
            "task_attribs": {},
            "task_args": {},
            "model_args": {},
            "config": config or {},
            "packages": {},
            "metadata": {"vllm_batch": True},
        },
        "plan": {
            "name": "plan",
            "steps": [
                {"solver": "generate", "params": {}},
            ],
            "config": config or {},
        },
        "results": {
            "scorer": {"name": scorer_name, "params": {}},
            "scores": [
                {
                    "name": scorer_name,
                    "scorer": scorer_name,
                    "params": {},
                    "metrics": scores_summary,
                    "scored_samples": scored_count,
                    "unscored_samples": unscored_count,
                }
            ],
            "metadata": {"vllm_batch": True},
        },
        "stats": {
            "started_at": now,
            "completed_at": now,
            "model_usage": {},
        },
        "samples": samples,
        "logging": [],
        "location": log_path,
    }


def _score_value_to_str(value: Any) -> str:
    """Convert a Score value to its string representation for the log."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return str(value) if value is not None else "I"


def run_benchmark_eval_vllm_batch(
    *,
    spec: InspectBenchmarkSpec,
    vllm_variant_provider: Any,
    run_dir: Path,
    model_name: str,
    temperature: float = 0.0,
) -> InspectRunResult:
    """Run a benchmark eval using direct vLLM batch inference.

    Bypasses Inspect's per-sample generate loop by:
    1. Building the Inspect Task (to get samples + scorer)
    2. Collecting all prompts as chat message lists
    3. Calling vLLM's ``generate_batch()`` in one shot
    4. Running the Inspect scorer on each (sample, response) pair
    5. Producing an EvalLog-compatible JSON file

    Args:
        spec: Benchmark specification (task name, args, limit).
        vllm_variant_provider: A ``_VllmVariantProvider`` instance with a
            live vLLM engine and optional LoRA request.
        run_dir: Directory for output files.
        model_name: Model identifier for logging.
        temperature: Sampling temperature.

    Returns:
        InspectRunResult with status and log location.
    """
    native_log_dir = run_dir / "native" / "inspect_logs"
    native_log_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Build the task to get samples, scorer, and config
        task = build_benchmark_task(spec)
        samples: list[Sample] = list(task.dataset)
        if spec.limit is not None:
            samples = samples[: spec.limit]

        if not samples:
            return InspectRunResult(status="failed", error="No samples in dataset")

        # Get scorer — tasks define scorer as a single scorer or list
        scorer_list = task.scorer
        if isinstance(scorer_list, list):
            scorer_obj = scorer_list[0] if scorer_list else None
        else:
            scorer_obj = scorer_list

        if scorer_obj is None:
            return InspectRunResult(status="failed", error="Task has no scorer")

        # Get max_tokens from task config
        max_tokens = 32  # Default for MCQ
        if task.config and task.config.max_tokens is not None:
            max_tokens = task.config.max_tokens

        # 2. Format all prompts as chat message lists
        all_prompts: list[list[dict[str, str]]] = [
            _sample_to_chat_messages(s) for s in samples
        ]

        # 3. Call vLLM batch inference
        t0 = time.perf_counter()
        responses = vllm_variant_provider.generate_batch(
            all_prompts,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
        inference_time = time.perf_counter() - t0
        print(
            f"    vLLM batch: {len(samples)} samples in {inference_time:.1f}s "
            f"({len(samples) / inference_time:.0f} samples/s)",
            flush=True,
        )

        # 4. Score each sample
        # Instantiate the scorer if it's a scorer factory (decorated function)
        if callable(scorer_obj) and not asyncio.iscoroutinefunction(scorer_obj):
            try:
                instantiated = scorer_obj()
                if callable(instantiated):
                    scorer_obj = instantiated
            except TypeError:
                pass

        scorer_name = getattr(scorer_obj, "name", None) or getattr(
            scorer_obj, "__name__", "scorer"
        )

        scored_samples: list[dict[str, Any]] = []
        metric_values: list[float] = []

        loop = asyncio.new_event_loop()
        try:
            for i, (sample, response_text) in enumerate(zip(samples, responses)):
                # Build TaskState for the scorer
                state = _build_task_state(sample, response_text, model_name)

                # Build Target
                target = Target(sample.target) if sample.target else Target("")

                # Run scorer
                score = loop.run_until_complete(
                    _score_sample(scorer_obj, state, target)
                )

                score_value = _score_value_to_str(score.value)
                numeric_value: float | None = None
                if isinstance(score.value, (int, float)):
                    numeric_value = float(score.value)
                elif score.value == "C":
                    numeric_value = 1.0
                elif score.value == "I":
                    numeric_value = 0.0

                if numeric_value is not None:
                    metric_values.append(numeric_value)

                # Build per-sample log entry
                sample_input = sample.input
                if isinstance(sample_input, str):
                    messages_log = [{"role": "user", "content": sample_input}]
                else:
                    messages_log = [
                        {"role": getattr(m, "role", "user"),
                         "content": m.text if hasattr(m, "text") else str(getattr(m, "content", ""))}
                        for m in sample_input
                    ]
                messages_log.append({"role": "assistant", "content": response_text})

                target_val = sample.target
                if isinstance(target_val, list):
                    target_log = target_val
                elif target_val is not None:
                    target_log = str(target_val)
                else:
                    target_log = ""

                scored_samples.append({
                    "id": sample.id or i,
                    "epoch": 1,
                    "input": messages_log[:-1],
                    "target": target_log,
                    "messages": messages_log,
                    "output": {
                        "model": model_name,
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": response_text,
                            },
                            "stop_reason": "stop",
                        }],
                        "usage": {},
                    },
                    "scores": {
                        scorer_name: {
                            "value": score_value,
                            "answer": score.answer,
                            "explanation": score.explanation,
                        },
                    },
                    "metadata": sample.metadata or {},
                })
        finally:
            loop.close()

        # 5. Compute aggregate metrics
        if metric_values:
            mean_val = sum(metric_values) / len(metric_values)
        else:
            mean_val = 0.0

        scores_summary = {
            "accuracy": {"value": mean_val, "name": "accuracy"},
            "mean": {"value": mean_val, "name": "mean"},
        }
        # Count correct / incorrect for personality evals
        n_correct = sum(1 for v in metric_values if v >= 0.5)
        n_incorrect = len(metric_values) - n_correct

        # 6. Write the log
        log_filename = f"{spec.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        log_path = native_log_dir / log_filename

        log_dict = _make_eval_log_dict(
            task_name=spec.name,
            model_name=model_name,
            samples=scored_samples,
            scores_summary=scores_summary,
            total_seconds=time.perf_counter() - t0,
            scorer_name=scorer_name,
            log_path=str(log_path),
            config={"temperature": temperature, "max_tokens": max_tokens},
        )

        log_path.write_text(json.dumps(log_dict, indent=2, default=str), encoding="utf-8")

        # Return a lightweight EvalLog-like result. We construct a minimal
        # object so that the caller can read .location and .status.
        # The actual log was written as JSON above.
        return InspectRunResult(
            status="ok",
            log=_make_minimal_eval_log(log_path, log_dict),
        )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return InspectRunResult(status="failed", error=str(exc))


def _make_minimal_eval_log(log_path: Path, log_dict: dict[str, Any]) -> EvalLog:
    """Create a minimal EvalLog by reading back the JSON we just wrote.

    This ensures the log object has the correct ``location`` and ``status``
    fields that the suite orchestration expects.
    """
    from inspect_ai.log import read_eval_log

    try:
        return read_eval_log(str(log_path), format="json")
    except Exception:
        # If Inspect can't parse our JSON (schema mismatch), return a shim
        # with the essential fields.
        return _EvalLogShim(location=str(log_path), status="success")


class _EvalLogShim:
    """Minimal stand-in for EvalLog when Inspect can't parse our JSON.

    Only provides ``.location`` and ``.status`` which are the fields
    the suite orchestration reads from the result.
    """

    def __init__(self, location: str, status: str) -> None:
        self.location = location
        self.status = status
