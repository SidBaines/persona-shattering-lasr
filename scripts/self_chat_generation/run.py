"""Symmetric self-chat generation with alternating role-swapped prompts."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset

from scripts.common.conversation_runtime import (
    canonical_role_for_generated_turn,
    format_progress_bar,
    message_append_id,
    now_iso,
    render_prompt_messages,
    speaker_label_for_generated_turn,
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
    write_inference_result,
    write_message_append,
)
from scripts.datasets.io import read_jsonl_tolerant
from scripts.datasets.loaders import load_dataset_from_config
from scripts.datasets.schema import StageEventRecord
from scripts.inference import InferenceConfig
from scripts.inference.providers import get_provider
from scripts.inference.providers.base import InferenceProvider, PromptInput, TokenUsage
from scripts.utils import setup_logging
from scripts.utils.hf_hub import login_from_env, upload_folder_to_dataset_repo

from .config import HfUploadConfig, SelfChatGenerationConfig, SelfChatGenerationResult


MAX_ATTEMPTS_PER_TURN = 3
PhaseKey = tuple[str, int]


class _AsyncBatchExecutor:
    """Collect prompts from coroutines and execute them in async batches."""

    def __init__(
        self,
        provider: InferenceProvider,
        batch_size: int,
        batch_timeout: float = 0.05,
    ) -> None:
        self._provider = provider
        self._batch_size = max(1, batch_size)
        self._batch_timeout = batch_timeout
        self._queue: asyncio.Queue = asyncio.Queue()

    def stop(self) -> None:
        self._queue.put_nowait(None)

    async def generate(
        self,
        prompt: PromptInput,
    ) -> tuple[str, TokenUsage | None, str | None]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[str, TokenUsage | None, str | None]] = loop.create_future()
        await self._queue.put((prompt, future))
        return await future

    async def run(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                break

            prompts: list[PromptInput] = [item[0]]
            futures: list[asyncio.Future[tuple[str, TokenUsage | None, str | None]]] = [item[1]]

            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._batch_timeout
            while len(prompts) < self._batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    next_item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if next_item is None:
                    self._queue.put_nowait(None)
                    break
                prompts.append(next_item[0])
                futures.append(next_item[1])

            try:
                responses, usages, _ = await self._provider.generate_batch_with_details_async(
                    prompts,
                    num_responses=1,
                )
                if len(responses) != len(prompts):
                    raise ValueError(
                        "Provider returned unexpected number of responses. "
                        f"expected={len(prompts)} actual={len(responses)}"
                    )
                if len(usages) != len(responses):
                    usages = [None] * len(responses)

                for future, response, usage in zip(futures, responses, usages):
                    if not future.done():
                        future.set_result((response, usage, None))
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                for future in futures:
                    if not future.done():
                        future.set_result(("", None, error))


def _event_id(*parts: str) -> str:
    text = ":".join(parts)
    return f"evt_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]}"


def _generated_turn_count(messages: list[dict[str, Any]], input_message_count: int) -> int:
    return max(0, len(messages) - input_message_count)


def _record_self_chat_event(
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
            stage="self_chat_generation",
            event_type=event_type,
            sample_id=sample_id,
            created_at=now_iso(),
            payload=payload,
        ),
    )


def _record_terminal_failure(
    run_dir: Path,
    *,
    sample_id: str,
    generated_turn_index: int,
    attempt_no: int,
    reason: str,
) -> None:
    _record_self_chat_event(
        run_dir,
        event_type="terminal_failure",
        sample_id=sample_id,
        payload={
            "generated_turn_index": generated_turn_index,
            "attempt_no": attempt_no,
            "reason": reason,
        },
    )


def _load_self_chat_resume_state(run_dir: Path) -> tuple[dict[PhaseKey, int], set[str]]:
    paths = get_run_paths(run_dir)
    rows, _ = read_jsonl_tolerant(paths["stage_events"])
    attempts: dict[PhaseKey, int] = {}
    terminal_samples: set[str] = set()

    for row in rows:
        if row.get("stage") != "self_chat_generation":
            continue
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str):
            continue

        event_type = row.get("event_type")
        payload = row.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}

        if event_type == "generation_attempt":
            generated_turn_index = payload.get("generated_turn_index")
            attempt_no = payload.get("attempt_no")
            if not isinstance(generated_turn_index, int) or not isinstance(attempt_no, int):
                continue
            attempts[(sample_id, generated_turn_index)] = max(
                attempts.get((sample_id, generated_turn_index), 0),
                attempt_no,
            )
        elif event_type == "terminal_failure":
            terminal_samples.add(sample_id)

    return attempts, terminal_samples


def _build_speaker_config(config: InferenceConfig) -> InferenceConfig:
    return config.model_copy(
        update={
            "run_dir": None,
            "output_path": None,
            "resume": False,
            "overwrite_output": False,
            "continue_on_error": False,
            "generation": config.generation.model_copy(update={"num_responses_per_prompt": 1}),
        }
    )


def _speaker_metadata(
    *,
    generated_turn_index: int,
    speaker_label: str,
    prompt_role_swapped: bool,
    provider_name: str,
    model_name: str,
    usage: TokenUsage | None,
    parent_message_id: str | None,
    attempt_no: int,
) -> dict[str, Any]:
    return {
        "turn_index": generated_turn_index,
        "source_stage": "self_chat_generation",
        "speaker_label": speaker_label,
        "canonical_role": canonical_role_for_generated_turn(generated_turn_index),
        "prompt_role_swapped": prompt_role_swapped,
        "provider": provider_name,
        "model": model_name,
        "token_usage": usage or {},
        "parent_message_id": parent_message_id,
        "phase_attempt_no": attempt_no,
    }


async def _run_conversation_async(
    *,
    sample_id: str,
    messages: list[dict[str, Any]],
    input_message_count: int,
    config: SelfChatGenerationConfig,
    speaker_a_executor: _AsyncBatchExecutor,
    speaker_b_executor: _AsyncBatchExecutor,
    speaker_a_config: InferenceConfig,
    speaker_b_config: InferenceConfig,
    attempts_by_turn: dict[PhaseKey, int],
    terminal_samples: set[str],
    run_dir: Path,
    progress: dict[str, int],
    logger: logging.Logger,
) -> None:
    start_turn = _generated_turn_count(messages, input_message_count)

    for generated_turn_index in range(start_turn, config.num_generated_turns):
        canonical_role = canonical_role_for_generated_turn(generated_turn_index)
        speaker_label = speaker_label_for_generated_turn(generated_turn_index)
        prompt_role_swapped = canonical_role == "user"
        executor = speaker_a_executor if speaker_label == "speaker_a" else speaker_b_executor
        speaker_config = speaker_a_config if speaker_label == "speaker_a" else speaker_b_config
        parent_message_id = messages[-1].get("message_id") if messages else None
        prompt = render_prompt_messages(messages, swap_roles=prompt_role_swapped)

        attempt_key = (sample_id, generated_turn_index)
        base_attempt = attempts_by_turn.get(attempt_key, 0)
        turn_success = False
        for attempt_no in range(base_attempt + 1, base_attempt + MAX_ATTEMPTS_PER_TURN + 1):
            response_text, usage, error = await executor.generate(prompt)
            success = bool(response_text.strip()) and error is None
            attempts_by_turn[attempt_key] = attempt_no

            _record_self_chat_event(
                run_dir,
                event_type="generation_attempt",
                sample_id=sample_id,
                payload={
                    "generated_turn_index": generated_turn_index,
                    "attempt_no": attempt_no,
                    "status": "success" if success else "failed",
                    "error": None if success else (error or "empty_response"),
                    "speaker_label": speaker_label,
                    "canonical_role": canonical_role,
                    "prompt_role_swapped": prompt_role_swapped,
                    "provider": speaker_config.provider,
                    "model": speaker_config.model,
                    "token_usage": usage or {},
                },
            )

            if not success:
                logger.warning(
                    "Self-chat turn failed (sample=%s turn=%d attempt=%d): %s",
                    sample_id,
                    generated_turn_index,
                    attempt_no,
                    error or "empty_response",
                )
                continue

            message_id = message_append_id(sample_id, canonical_role, generated_turn_index)
            metadata = _speaker_metadata(
                generated_turn_index=generated_turn_index,
                speaker_label=speaker_label,
                prompt_role_swapped=prompt_role_swapped,
                provider_name=speaker_config.provider,
                model_name=speaker_config.model,
                usage=usage,
                parent_message_id=parent_message_id,
                attempt_no=attempt_no,
            )
            if canonical_role == "assistant":
                write_inference_result(
                    run_dir,
                    sample_id,
                    {
                        "status": "success",
                        "model": speaker_config.model,
                        "provider": speaker_config.provider,
                        "assistant_message_id": message_id,
                        "assistant_completion": response_text,
                        "assistant_full": response_text,
                        "assistant_message_metadata": metadata,
                        "attempt_no": attempt_no,
                        "token_usage": usage or {},
                        "started_at": now_iso(),
                        "completed_at": now_iso(),
                        "error": None,
                    },
                    materialize=False,
                )
            else:
                write_message_append(
                    run_dir,
                    sample_id,
                    {
                        "message_id": message_id,
                        "role": "user",
                        "content": response_text,
                        "editable": True,
                        "message_metadata": metadata,
                    },
                    materialize=False,
                )

            messages.append(
                {
                    "role": canonical_role,
                    "content": response_text,
                    "message_id": message_id,
                }
            )
            progress["generated_turns_completed"] += 1
            turn_success = True
            break

        if not turn_success:
            terminal_samples.add(sample_id)
            progress["conversations_completed"] += 1
            progress["conversations_failed"] += 1
            _record_terminal_failure(
                run_dir,
                sample_id=sample_id,
                generated_turn_index=generated_turn_index,
                attempt_no=base_attempt + MAX_ATTEMPTS_PER_TURN,
                reason="max_attempts_exceeded",
            )
            return

    progress["conversations_completed"] += 1


def _log_progress(logger: logging.Logger, progress: dict[str, int], total_conversations: int, total_generated_turns: int) -> None:
    logger.info(
        "Progress | convs %s %d/%d | generated turns %s %d/%d | failed %d",
        format_progress_bar(progress["conversations_completed"], total_conversations),
        progress["conversations_completed"],
        total_conversations,
        format_progress_bar(progress["generated_turns_completed"], total_generated_turns),
        progress["generated_turns_completed"],
        total_generated_turns,
        progress["conversations_failed"],
    )


async def _progress_reporter_async(
    logger: logging.Logger,
    progress: dict[str, int],
    total_conversations: int,
    total_generated_turns: int,
    stop_event: asyncio.Event,
    interval: float = 10.0,
) -> None:
    while True:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            break
        except asyncio.TimeoutError:
            _log_progress(logger, progress, total_conversations, total_generated_turns)
    _log_progress(logger, progress, total_conversations, total_generated_turns)


async def _run_self_chat_pipeline_async(
    *,
    config: SelfChatGenerationConfig,
    samples: list[Any],
    speaker_a_executor: _AsyncBatchExecutor,
    speaker_b_executor: _AsyncBatchExecutor,
    speaker_a_config: InferenceConfig,
    speaker_b_config: InferenceConfig,
    attempts_by_turn: dict[PhaseKey, int],
    terminal_samples: set[str],
    run_dir: Path,
    logger: logging.Logger,
) -> None:
    pending_samples = [
        sample
        for sample in samples
        if sample.sample_id not in terminal_samples
        and _generated_turn_count(
            [
                {"role": message.role, "content": message.content, "message_id": message.message_id}
                for message in sample.messages
            ],
            len(sample.input.messages),
        )
        < config.num_generated_turns
    ]

    total_conversations = len(pending_samples)
    total_generated_turns = sum(
        max(
            0,
            config.num_generated_turns - _generated_turn_count(
                [
                    {
                        "role": message.role,
                        "content": message.content,
                        "message_id": message.message_id,
                    }
                    for message in sample.messages
                ],
                len(sample.input.messages),
            ),
        )
        for sample in pending_samples
    )
    progress = {
        "conversations_completed": 0,
        "conversations_failed": 0,
        "generated_turns_completed": 0,
    }
    stop_event = asyncio.Event()
    reporter_task = asyncio.create_task(
        _progress_reporter_async(
            logger,
            progress,
            total_conversations,
            total_generated_turns,
            stop_event,
        )
    )

    tasks = [
        _run_conversation_async(
            sample_id=sample.sample_id,
            messages=[
                {"role": message.role, "content": message.content, "message_id": message.message_id}
                for message in sample.messages
            ],
            input_message_count=len(sample.input.messages),
            config=config,
            speaker_a_executor=speaker_a_executor,
            speaker_b_executor=speaker_b_executor,
            speaker_a_config=speaker_a_config,
            speaker_b_config=speaker_b_config,
            attempts_by_turn=attempts_by_turn,
            terminal_samples=terminal_samples,
            run_dir=run_dir,
            progress=progress,
            logger=logger,
        )
        for sample in pending_samples
    ]

    try:
        if tasks:
            await asyncio.gather(*tasks)
    finally:
        stop_event.set()
        await reporter_task


def _upload_run_dir(
    *,
    run_dir: Path,
    upload_config: HfUploadConfig,
) -> str:
    if not upload_config.repo_id:
        raise ValueError("hf_upload.repo_id is required when hf_upload.enabled=True.")
    login_from_env()
    path_in_repo = f"{upload_config.path_in_repo.rstrip('/')}/{run_dir.name}"
    return upload_folder_to_dataset_repo(
        local_dir=run_dir,
        repo_id=upload_config.repo_id,
        path_in_repo=path_in_repo,
        commit_message=upload_config.commit_message,
    )


def run_self_chat_generation(
    config: SelfChatGenerationConfig,
    dataset: Dataset | None = None,
) -> tuple[Dataset, SelfChatGenerationResult]:
    """Generate symmetric self-chat transcripts and export canonical datasets."""
    logger = setup_logging()
    run_dir = Path(config.run_dir)

    if config.num_generated_turns <= 0:
        raise ValueError("num_generated_turns must be positive.")
    if config.num_rollouts_per_prompt <= 0:
        raise ValueError("num_rollouts_per_prompt must be positive.")

    config_for_fingerprint = {
        key: value
        for key, value in config.model_dump(mode="json").items()
        if key not in {"resume", "overwrite_output"}
    }
    init_run(run_dir, base_config={"self_chat_generation": config.model_dump(mode="json")})
    register_stage_fingerprint(run_dir, "self_chat_generation", config_for_fingerprint)

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

    speaker_a_config = _build_speaker_config(config.speaker_a_inference)
    speaker_b_source = config.speaker_b_inference or config.speaker_a_inference
    speaker_b_config = _build_speaker_config(speaker_b_source)

    speaker_a_provider = get_provider(speaker_a_config.provider, speaker_a_config)
    reuse_speaker_a = config.speaker_b_inference is None
    if reuse_speaker_a:
        speaker_b_provider = speaker_a_provider
    else:
        speaker_b_provider = get_provider(speaker_b_config.provider, speaker_b_config)

    speaker_a_executor = _AsyncBatchExecutor(
        speaker_a_provider,
        batch_size=max(1, speaker_a_config.generation.batch_size),
    )
    speaker_b_executor = (
        speaker_a_executor
        if reuse_speaker_a
        else _AsyncBatchExecutor(
            speaker_b_provider,
            batch_size=max(1, speaker_b_config.generation.batch_size),
        )
    )

    if config.resume:
        attempts_by_turn, terminal_samples = _load_self_chat_resume_state(run_dir)
    else:
        attempts_by_turn, terminal_samples = {}, set()

    materialize_canonical_samples(run_dir)
    samples = load_samples(run_dir)

    async def _run_all() -> None:
        executor_tasks = [asyncio.create_task(speaker_a_executor.run())]
        if speaker_b_executor is not speaker_a_executor:
            executor_tasks.append(asyncio.create_task(speaker_b_executor.run()))
        try:
            await _run_self_chat_pipeline_async(
                config=config,
                samples=samples,
                speaker_a_executor=speaker_a_executor,
                speaker_b_executor=speaker_b_executor,
                speaker_a_config=speaker_a_config,
                speaker_b_config=speaker_b_config,
                attempts_by_turn=attempts_by_turn,
                terminal_samples=terminal_samples,
                run_dir=run_dir,
                logger=logger,
            )
        finally:
            speaker_a_executor.stop()
            if speaker_b_executor is not speaker_a_executor:
                speaker_b_executor.stop()
            await asyncio.gather(*executor_tasks)

    asyncio.run(_run_all())

    materialize_canonical_samples(run_dir)
    export_path = export_dataset(run_dir, profile="conversation_training")
    trace_path = export_dataset(run_dir, profile="conversation_trace")

    result_dataset = load_dataset_from_config(
        config.dataset.model_copy(
            update={
                "source": "canonical",
                "path": str(run_dir),
            }
        )
    )

    final_samples = load_samples(run_dir)
    completed = 0
    completed_turns = 0
    for sample in final_samples:
        turn_count = _generated_turn_count(
            [{"role": message.role, "content": message.content} for message in sample.messages],
            len(sample.input.messages),
        )
        completed_turns += turn_count
        if turn_count >= config.num_generated_turns:
            completed += 1

    hf_dataset_url = None
    if config.hf_upload.enabled:
        hf_dataset_url = _upload_run_dir(run_dir=run_dir, upload_config=config.hf_upload)

    result = SelfChatGenerationResult(
        output_path=Path(export_path),
        num_conversations=len(final_samples),
        num_completed=completed,
        num_failed=len(final_samples) - completed,
        num_generated_turns_target=config.num_generated_turns,
        num_generated_turns_completed=completed_turns,
        exports={
            "conversation_training": str(export_path),
            "conversation_trace": str(trace_path),
        },
        hf_dataset_url=hf_dataset_url,
    )
    logger.info(
        "Self-chat generation complete: %d/%d complete, failed=%d.",
        completed,
        len(final_samples),
        len(final_samples) - completed,
    )
    return result_dataset, result
