"""Long-context rollout generation with alternating assistant and user turns."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset
from scripts.common.conversation_runtime import (
    format_progress_bar,
    message_append_id,
    now_iso,
)
from scripts.datasets import (
    export_dataset,
    get_run_paths,
    ingest_source_dataset,
    init_run,
    load_samples,
    materialize_canonical_samples,
    record_stage_event,
    register_stage_fingerprint,
    register_system_prompt,
    write_inference_result,
    write_message_append,
)
from scripts.datasets.io import read_jsonl_tolerant
from scripts.datasets.loaders import load_dataset_from_config
from scripts.datasets.schema import StageEventRecord
from scripts.inference import InferenceConfig
from scripts.inference.providers import get_provider
from scripts.inference.providers.base import InferenceProvider, TokenUsage
from scripts.utils import setup_logging

from .config import (
    RolloutGenerationConfig,
    RolloutGenerationResult,
    UserSimulatorConfig,
)
from .gpu_executor import GpuBatchExecutor
from .prompts import (
    get_user_simulator_instruction,
    render_user_simulator_single_turn_prompt,
)

PhaseKey = tuple[str, str, int]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _assistant_turn_count_sample(sample) -> int:
    return sum(1 for m in sample.messages if m.role == "assistant")


def _assistant_turn_count_dicts(messages: list[dict[str, Any]]) -> int:
    return sum(1 for m in messages if m.get("role") == "assistant")


def _phase_key(sample_id: str, phase: str, turn_index: int) -> PhaseKey:
    return (sample_id, phase, turn_index)


def _event_id(*parts: str) -> str:
    text = ":".join(parts)
    return f"evt_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]}"


def _record_rollout_event(
    run_dir: Path,
    *,
    event_type: str,
    sample_id: str,
    payload: dict[str, Any],
) -> None:
    record_stage_event(
        run_dir,
        StageEventRecord(
            event_id=_event_id(event_type, sample_id, now_iso()),
            stage="rollout_generation",
            event_type=event_type,
            sample_id=sample_id,
            created_at=now_iso(),
            payload=payload,
        ),
        lightweight=True,
    )


def _record_terminal_failure(
    run_dir: Path,
    *,
    sample_id: str,
    phase: str,
    turn_index: int,
    attempt_no: int,
    reason: str,
) -> None:
    _record_rollout_event(
        run_dir,
        event_type="terminal_failure",
        sample_id=sample_id,
        payload={
            "phase": phase,
            "turn_index": turn_index,
            "attempt_no": attempt_no,
            "reason": reason,
        },
    )


def _record_invalid_state(run_dir: Path, sample_id: str, reason: str) -> None:
    _record_rollout_event(
        run_dir,
        event_type="invalid_state",
        sample_id=sample_id,
        payload={"reason": reason},
    )


def _load_rollout_resume_state(run_dir: Path) -> tuple[dict[PhaseKey, int], set[str]]:
    """Load attempt counters and terminal sample IDs from stage events."""
    stage_events_path = get_run_paths(run_dir)["stage_events"]
    rows, _ = read_jsonl_tolerant(stage_events_path)

    attempts: dict[PhaseKey, int] = {}
    terminal_samples: set[str] = set()

    for row in rows:
        if row.get("stage") != "rollout_generation":
            continue
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str):
            continue

        event_type = row.get("event_type")
        payload = row.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}

        if event_type in {"assistant_attempt", "user_attempt"}:
            phase = "assistant" if event_type == "assistant_attempt" else "user"
            turn_index = payload.get("turn_index")
            attempt_no = payload.get("attempt_no")
            if not isinstance(turn_index, int) or not isinstance(attempt_no, int):
                continue
            key = _phase_key(sample_id, phase, turn_index)
            attempts[key] = max(attempts.get(key, 0), attempt_no)
        elif event_type == "terminal_failure":
            terminal_samples.add(sample_id)

    return attempts, terminal_samples


def _build_user_simulator_inference_config(
    config: UserSimulatorConfig,
) -> InferenceConfig:
    return InferenceConfig(
        model=config.model,
        provider=config.provider,
        generation=config.generation,
        max_concurrent=config.max_concurrent,
        timeout=config.timeout,
        retry=config.retry,
        continue_on_error=False,
        log_failures=True,
        local=config.local,
        openai=config.openai,
        openrouter=config.openrouter,
        anthropic=config.anthropic,
    )


def _system_prompt_hash(prompt: str | None) -> str | None:
    """SHA-256 hash (first 16 chars) of the system prompt, or None."""
    if not prompt:
        return None
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _build_prompt_with_system(
    messages: list[dict[str, Any]], system_prompt: str | None
) -> list[dict[str, str]]:
    """Build assistant prompt: strip existing system messages, prepend current system prompt."""
    prompt = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") != "system"
    ]
    if system_prompt:
        prompt.insert(0, {"role": "system", "content": system_prompt})
    return prompt


def _build_user_prompt_from_messages(
    messages: list[dict[str, str]],
    *,
    prompt_template: str,
    prompt_format: str,
) -> str | list[dict[str, str]]:
    if prompt_format == "single_turn_text":
        return render_user_simulator_single_turn_prompt(prompt_template, messages)
    instruction = get_user_simulator_instruction(prompt_template)
    return [{"role": "system", "content": instruction}, *messages]


# ── Progress tracking ──────────────────────────────────────────────────────────


@dataclasses.dataclass
class _ProgressTracker:
    total_conversations: int
    total_assistant_turns: int
    conversations_completed: int = 0
    conversations_failed: int = 0
    assistant_turns_completed: int = 0
    user_turns_completed: int = 0


def _log_progress(logger: logging.Logger, tracker: _ProgressTracker) -> None:
    logger.info(
        "Progress | convs %s %d/%d | asst turns %s %d/%d | user turns %d | failed %d",
        format_progress_bar(
            tracker.conversations_completed, tracker.total_conversations
        ),
        tracker.conversations_completed,
        tracker.total_conversations,
        format_progress_bar(
            tracker.assistant_turns_completed, tracker.total_assistant_turns
        ),
        tracker.assistant_turns_completed,
        tracker.total_assistant_turns,
        tracker.user_turns_completed,
        tracker.conversations_failed,
    )


async def _progress_reporter_async(
    logger: logging.Logger,
    tracker: _ProgressTracker,
    stop_event: asyncio.Event,
    interval: float = 10.0,
) -> None:
    """Periodically log pipeline progress until stop_event is set."""
    while True:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            break  # stop_event set
        except asyncio.TimeoutError:
            _log_progress(logger, tracker)
    _log_progress(logger, tracker)  # final snapshot


# ── Per-conversation async coroutine ───────────────────────────────────────────


async def _run_conversation_async(
    sample_id: str,
    messages: list[dict[str, Any]],
    config: RolloutGenerationConfig,
    gpu_executor: GpuBatchExecutor,
    user_provider: InferenceProvider,
    user_config: InferenceConfig,
    run_dir: Path,
    attempts_by_phase: dict[PhaseKey, int],
    terminal_samples: set[str],
    assistant_model: str,
    assistant_provider_name: str,
    progress: _ProgressTracker,
    logger: logging.Logger,
) -> None:
    """Run one conversation to completion, alternating assistant and user turns.

    Maintains message history in-memory. Each turn's result is written to disk
    immediately (materialize=False); a single materialize call at the end of the
    pipeline assembles the final samples.
    """
    start_turn = _assistant_turn_count_dicts(messages)

    for turn_index in range(start_turn, config.num_assistant_turns):
        # ── Assistant turn ────────────────────────────────────────────────────
        assistant_max = config.failure_policy.assistant_max_attempts_per_turn
        assistant_key = _phase_key(sample_id, "assistant", turn_index)
        base_attempt = attempts_by_phase.get(assistant_key, 0)
        parent_message_id = messages[-1].get("message_id") if messages else None
        prompt = _build_prompt_with_system(messages, config.system_prompt)

        assistant_success = False
        for phase_attempt in range(base_attempt + 1, base_attempt + assistant_max + 1):
            response_text, gen_error = await gpu_executor.generate(prompt)
            success = bool(response_text.strip()) and gen_error is None
            error = None if success else (gen_error or "empty_response")

            attempts_by_phase[assistant_key] = phase_attempt
            _record_rollout_event(
                run_dir,
                event_type="assistant_attempt",
                sample_id=sample_id,
                payload={
                    "phase": "assistant",
                    "turn_index": turn_index,
                    "attempt_no": phase_attempt,
                    "status": "success" if success else "failed",
                    "error": error,
                    "provider": assistant_provider_name,
                    "model": assistant_model,
                    "token_usage": {},
                },
            )

            assistant_message_id = message_append_id(sample_id, "assistant", turn_index)
            inference_payload: dict[str, Any] = {
                "status": "success" if success else "failed",
                "model": assistant_model,
                "provider": assistant_provider_name,
                "attempt_no": phase_attempt,
                "token_usage": {},
                "started_at": now_iso(),
                "completed_at": now_iso(),
                "error": error,
            }
            if success:
                inference_payload.update(
                    {
                        "assistant_message_id": assistant_message_id,
                        "assistant_completion": response_text,
                        "assistant_full": response_text,
                        "assistant_message_metadata": {
                            "turn_index": turn_index,
                            "source_stage": "rollout_assistant",
                            "provider": assistant_provider_name,
                            "model": assistant_model,
                            "token_usage": {},
                            "parent_message_id": parent_message_id,
                            "phase_attempt_no": phase_attempt,
                            "system_prompt_hash": _system_prompt_hash(
                                config.system_prompt
                            ),
                        },
                    }
                )
            write_inference_result(
                run_dir, sample_id, inference_payload, materialize=False
            )

            if success:
                messages.append(
                    {
                        "role": "assistant",
                        "content": response_text,
                        "message_id": assistant_message_id,
                    }
                )
                progress.assistant_turns_completed += 1
                assistant_success = True
                break

            logger.warning(
                "Assistant turn failed (sample=%s turn=%d attempt=%d): %s",
                sample_id,
                turn_index,
                phase_attempt,
                error,
            )

        if not assistant_success:
            terminal_samples.add(sample_id)
            progress.conversations_failed += 1
            progress.conversations_completed += 1
            _record_terminal_failure(
                run_dir,
                sample_id=sample_id,
                phase="assistant",
                turn_index=turn_index,
                attempt_no=base_attempt + assistant_max,
                reason="assistant_max_attempts_exceeded",
            )
            return

        # Optionally skip the user turn after the final assistant turn.
        if config.skip_final_user_turn and turn_index + 1 >= config.num_assistant_turns:
            break

        # ── User simulator turn ───────────────────────────────────────────────
        user_max = config.failure_policy.user_max_attempts_per_turn
        user_key = _phase_key(sample_id, "user", turn_index)
        user_base_attempt = attempts_by_phase.get(user_key, 0)
        parent_user_message_id = messages[-1].get("message_id")
        user_prompt = _build_user_prompt_from_messages(
            [{"role": m["role"], "content": m["content"]} for m in messages],
            prompt_template=config.user_simulator.prompt_template,
            prompt_format=config.user_simulator.prompt_format,
        )

        user_success = False
        for user_attempt in range(
            user_base_attempt + 1, user_base_attempt + user_max + 1
        ):
            user_response = ""
            user_usage: TokenUsage | None = None
            user_error: str | None = None
            try:
                (
                    responses,
                    usages,
                    _,
                ) = await user_provider.generate_batch_with_details_async(
                    [user_prompt], num_responses=1
                )
                user_response = responses[0] if responses else ""
                user_usage = usages[0] if usages else None
            except Exception as exc:  # noqa: BLE001
                user_error = str(exc)

            u_success = bool(user_response.strip()) and user_error is None
            u_error = None if u_success else (user_error or "empty_response")

            attempts_by_phase[user_key] = user_attempt
            _record_rollout_event(
                run_dir,
                event_type="user_attempt",
                sample_id=sample_id,
                payload={
                    "phase": "user",
                    "turn_index": turn_index,
                    "attempt_no": user_attempt,
                    "status": "success" if u_success else "failed",
                    "error": u_error,
                    "provider": user_config.provider,
                    "model": user_config.model,
                    "token_usage": user_usage or {},
                },
            )

            if u_success:
                user_message_id = message_append_id(sample_id, "user", turn_index)
                write_message_append(
                    run_dir,
                    sample_id,
                    {
                        "message_id": user_message_id,
                        "role": "user",
                        "content": user_response,
                        "editable": True,
                        "message_metadata": {
                            "turn_index": turn_index,
                            "source_stage": "rollout_user_simulator",
                            "provider": user_config.provider,
                            "model": user_config.model,
                            "token_usage": user_usage or {},
                            "parent_message_id": parent_user_message_id,
                            "phase_attempt_no": user_attempt,
                            "system_prompt_hash": _system_prompt_hash(
                                get_user_simulator_instruction(
                                    config.user_simulator.prompt_template
                                )
                            ),
                            "user_prompt_template": config.user_simulator.prompt_template,
                        },
                    },
                    materialize=False,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": user_response,
                        "message_id": user_message_id,
                    }
                )
                progress.user_turns_completed += 1
                user_success = True
                break

            logger.warning(
                "User turn failed (sample=%s turn=%d attempt=%d): %s",
                sample_id,
                turn_index,
                user_attempt,
                u_error,
            )

        if not user_success:
            terminal_samples.add(sample_id)
            progress.conversations_failed += 1
            progress.conversations_completed += 1
            _record_terminal_failure(
                run_dir,
                sample_id=sample_id,
                phase="user",
                turn_index=turn_index,
                attempt_no=user_base_attempt + user_max,
                reason="user_max_attempts_exceeded",
            )
            return

    progress.conversations_completed += 1


# ── Async pipeline scheduler ───────────────────────────────────────────────────


async def _run_rollout_pipeline_async(
    config: RolloutGenerationConfig,
    samples: list,
    assistant_config: InferenceConfig,
    assistant_provider: InferenceProvider,
    user_provider: InferenceProvider,
    user_config: InferenceConfig,
    run_dir: Path,
    attempts_by_phase: dict[PhaseKey, int],
    terminal_samples: set[str],
    logger: logging.Logger,
) -> None:
    """Pipelined async scheduler: GPU batches and user API calls overlap.

    Each conversation runs as an independent coroutine. Assistant turns are
    batched through GpuBatchExecutor (runs in a thread via asyncio.to_thread),
    releasing the event loop to service concurrent user API calls while the
    GPU is busy. This eliminates the 40-pass waterfall of the old sequential
    phase loop.
    """

    batch_size = max(1, assistant_config.generation.batch_size)
    executor = GpuBatchExecutor(assistant_provider, batch_size=batch_size)
    executor_task = asyncio.create_task(executor.run())

    pending = [
        sample
        for sample in samples
        if sample.sample_id not in terminal_samples
        and _assistant_turn_count_sample(sample) < config.num_assistant_turns
    ]

    total_assistant_turns = len(pending) * config.num_assistant_turns
    progress = _ProgressTracker(
        total_conversations=len(pending),
        total_assistant_turns=total_assistant_turns,
    )
    stop_event = asyncio.Event()

    logger.info(
        "Async pipeline: %d conversations, %d assistant turns total (%d already done)",
        len(pending),
        total_assistant_turns,
        len(samples) - len(pending),
    )

    conversation_tasks = [
        _run_conversation_async(
            sample_id=sample.sample_id,
            messages=[
                {
                    "role": m.role,
                    "content": m.content,
                    "message_id": getattr(m, "message_id", None),
                }
                for m in sample.messages
            ],
            config=config,
            gpu_executor=executor,
            user_provider=user_provider,
            user_config=user_config,
            run_dir=run_dir,
            attempts_by_phase=attempts_by_phase,
            terminal_samples=terminal_samples,
            assistant_model=assistant_config.model,
            assistant_provider_name=assistant_config.provider,
            progress=progress,
            logger=logger,
        )
        for sample in pending
    ]

    reporter_task = asyncio.create_task(
        _progress_reporter_async(logger, progress, stop_event, interval=10.0)
    )

    try:
        await asyncio.gather(*conversation_tasks)
    finally:
        stop_event.set()
        await reporter_task
        executor.stop()
        await executor_task
        for p in (assistant_provider, user_provider):
            c = getattr(p, "client", None)
            if c is not None and asyncio.iscoroutinefunction(getattr(c, "close", None)):
                await c.close()
        # Suppress "Event loop is closed" noise from httpx client finalizers
        # that fire after asyncio.run() tears down the loop.
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(lambda _loop, ctx: None)

    logger.info("Async pipeline complete.")


# ── Public entry point ─────────────────────────────────────────────────────────


def run_rollout_generation(
    config: RolloutGenerationConfig,
    dataset: Dataset | None = None,
) -> tuple[Dataset, RolloutGenerationResult]:
    """Generate alternating assistant/user rollouts and export transcripts."""
    logger = setup_logging()
    run_dir = Path(config.run_dir)

    if config.num_assistant_turns <= 0:
        raise ValueError("num_assistant_turns must be positive.")
    if config.num_rollouts_per_prompt <= 0:
        raise ValueError("num_rollouts_per_prompt must be positive.")
    if config.context_policy.mode != "full_history":
        raise NotImplementedError(
            "context_policy.mode='token_budget' is not implemented yet. Use mode='full_history'."
        )

    manifest = init_run(
        run_dir, base_config={"rollout_generation": config.model_dump(mode="json")}
    )
    if not (config.resume and manifest.stage_fingerprints.get("rollout_generation")):
        register_stage_fingerprint(
            run_dir, "rollout_generation", config.model_dump(mode="json")
        )
    register_system_prompt(run_dir, config.system_prompt)
    user_sim_text = get_user_simulator_instruction(
        config.user_simulator.prompt_template
    )
    register_system_prompt(run_dir, user_sim_text)

    paths = get_run_paths(run_dir)
    if not (config.resume and paths["sample_inputs"].exists()):
        if dataset is None:
            dataset = load_dataset_from_config(config.dataset)
        source_info: dict[str, Any] = {
            "dataset_source": config.dataset.source,
            "dataset_name": config.dataset.name,
            "dataset_path": config.dataset.path,
            "dataset_split": config.dataset.split,
            "max_samples": config.dataset.max_samples,
        }
        ingest_source_dataset(
            dataset=dataset,
            source_info=source_info,
            system_prompt=config.system_prompt,
            run_dir=run_dir,
            overwrite=config.overwrite_output,
            responses_per_input=config.num_rollouts_per_prompt,
        )

    assistant_config = config.assistant_inference.model_copy(
        update={
            "run_dir": None,
            "output_path": None,
            "resume": False,
            "overwrite_output": False,
            "continue_on_error": False,
            "generation": config.assistant_inference.generation.model_copy(
                update={"num_responses_per_prompt": 1}
            ),
        }
    )
    user_config = _build_user_simulator_inference_config(
        config.user_simulator
    ).model_copy(
        update={
            "generation": config.user_simulator.generation.model_copy(
                update={"num_responses_per_prompt": 1}
            )
        }
    )

    assistant_provider = get_provider(assistant_config.provider, assistant_config)
    user_provider = get_provider(user_config.provider, user_config)

    if config.resume:
        attempts_by_phase, terminal_samples = _load_rollout_resume_state(run_dir)
    else:
        attempts_by_phase, terminal_samples = {}, set()

    materialize_canonical_samples(run_dir)
    samples = load_samples(run_dir)

    asyncio.run(
        _run_rollout_pipeline_async(
            config=config,
            samples=samples,
            assistant_config=assistant_config,
            assistant_provider=assistant_provider,
            user_provider=user_provider,
            user_config=user_config,
            run_dir=run_dir,
            attempts_by_phase=attempts_by_phase,
            terminal_samples=terminal_samples,
            logger=logger,
        )
    )

    materialize_canonical_samples(run_dir)

    export_path = export_dataset(
        run_dir,
        profile="conversation_training",
        variant_name=config.transcript_variant,
    )
    trace_path = export_dataset(
        run_dir,
        profile="conversation_trace",
        variant_name=config.transcript_variant,
    )

    final_samples = load_samples(run_dir)
    completed = sum(
        1
        for s in final_samples
        if _assistant_turn_count_sample(s) >= config.num_assistant_turns
    )
    completed_turns = sum(_assistant_turn_count_sample(s) for s in final_samples)
    failed = len(final_samples) - completed

    result_dataset = load_dataset_from_config(
        config.dataset.model_copy(
            update={
                "source": "canonical",
                "path": str(run_dir),
            }
        )
    )

    result = RolloutGenerationResult(
        output_path=Path(export_path),
        num_conversations=len(final_samples),
        num_completed=completed,
        num_failed=failed,
        num_assistant_turns_target=config.num_assistant_turns,
        num_assistant_turns_completed=completed_turns,
        exports={
            "conversation_training": str(export_path),
            "conversation_trace": str(trace_path),
        },
    )

    logger.info(
        "Rollout generation complete: %d/%d complete, failed=%d.",
        completed,
        len(final_samples),
        failed,
    )
    return result_dataset, result
