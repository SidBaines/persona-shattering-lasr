"""Long-context rollout generation with alternating assistant and user turns."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset

from scripts.common.conversation_runtime import (
    chunked,
    format_turn_label,
    log_phase_batch_progress,
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

from .config import RolloutGenerationConfig, RolloutGenerationResult, UserSimulatorConfig
from .prompts import get_user_simulator_instruction


PhaseKey = tuple[str, str, int]


def _assistant_turn_count(sample) -> int:
    return sum(1 for message in sample.messages if message.role == "assistant")


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


def _build_user_simulator_inference_config(config: UserSimulatorConfig) -> InferenceConfig:
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


def _build_user_simulator_messages(sample, prompt_template: str) -> list[dict[str, str]]:
    instruction = get_user_simulator_instruction(prompt_template)
    rendered = [{"role": message.role, "content": message.content} for message in sample.messages]
    return [{"role": "system", "content": instruction}, *rendered]


def _build_assistant_action(
    sample,
    *,
    config: RolloutGenerationConfig,
    assistant_config: InferenceConfig,
    attempts_by_phase: dict[PhaseKey, int],
    terminal_samples: set[str],
    run_dir: Path,
) -> dict[str, Any] | None:
    latest_message = sample.messages[-1] if sample.messages else None
    if latest_message is None or latest_message.role != "user":
        return None

    turn_index = _assistant_turn_count(sample)
    if turn_index >= config.num_assistant_turns:
        return None

    key = _phase_key(sample.sample_id, "assistant", turn_index)
    phase_attempt_no = attempts_by_phase.get(key, 0) + 1
    max_attempts = config.failure_policy.assistant_max_attempts_per_turn
    if max_attempts > 0 and phase_attempt_no > max_attempts:
        terminal_samples.add(sample.sample_id)
        _record_terminal_failure(
            run_dir,
            sample_id=sample.sample_id,
            phase="assistant",
            turn_index=turn_index,
            attempt_no=phase_attempt_no,
            reason="assistant_max_attempts_exceeded",
        )
        return None

    return {
        "sample_id": sample.sample_id,
        "prompt": [{"role": message.role, "content": message.content} for message in sample.messages],
        "turn_index": turn_index,
        "display_turn_index": turn_index,
        "parent_message_id": latest_message.message_id,
        "phase_attempt_no": phase_attempt_no,
        "global_attempt_no": sample.inference.attempt_no + 1,
        "model": assistant_config.model,
        "provider": assistant_config.provider,
    }


def _build_user_action(
    sample,
    *,
    config: RolloutGenerationConfig,
    user_config: InferenceConfig,
    attempts_by_phase: dict[PhaseKey, int],
    terminal_samples: set[str],
    run_dir: Path,
) -> dict[str, Any] | None:
    latest_message = sample.messages[-1] if sample.messages else None
    if latest_message is None or latest_message.role != "assistant":
        return None

    assistant_turns = _assistant_turn_count(sample)
    if assistant_turns >= config.num_assistant_turns:
        return None

    turn_index = assistant_turns
    key = _phase_key(sample.sample_id, "user", turn_index)
    phase_attempt_no = attempts_by_phase.get(key, 0) + 1
    max_attempts = config.failure_policy.user_max_attempts_per_turn
    if max_attempts > 0 and phase_attempt_no > max_attempts:
        terminal_samples.add(sample.sample_id)
        _record_terminal_failure(
            run_dir,
            sample_id=sample.sample_id,
            phase="user",
            turn_index=turn_index,
            attempt_no=phase_attempt_no,
            reason="user_max_attempts_exceeded",
        )
        return None

    return {
        "sample_id": sample.sample_id,
        "prompt": _build_user_simulator_messages(sample, config.user_simulator.prompt_template),
        "turn_index": turn_index,
        "display_turn_index": max(turn_index - 1, 0),
        "parent_message_id": latest_message.message_id,
        "phase_attempt_no": phase_attempt_no,
        "model": user_config.model,
        "provider": user_config.provider,
    }


async def _generate_many(
    provider: InferenceProvider,
    prompts: list[str | list[dict[str, str]]],
) -> tuple[list[str], list[TokenUsage | None], int]:
    if not prompts:
        return [], [], 0
    return await provider.generate_batch_with_details_async(prompts, num_responses=1)


async def _generate_one(
    provider: InferenceProvider,
    prompt: str | list[dict[str, str]],
) -> tuple[str, TokenUsage | None]:
    responses, usages, _ = await provider.generate_batch_with_details_async([prompt], num_responses=1)
    if not responses:
        return "", None
    usage = usages[0] if usages else None
    return responses[0], usage


def _generate_with_fallback(
    logger: logging.Logger,
    provider: InferenceProvider,
    prompts: list[str | list[dict[str, str]]],
) -> list[tuple[str, TokenUsage | None, str | None]]:
    """Generate batch responses; fall back to per-prompt calls if batch fails."""
    if not prompts:
        return []

    try:
        responses, usages, _ = asyncio.run(_generate_many(provider, prompts))
        usage_by_slot = list(usages) if len(usages) == len(responses) else [None] * len(responses)
        return [(text, usage, None) for text, usage in zip(responses, usage_by_slot)]
    except Exception as batch_exc:  # noqa: BLE001
        logger.warning(
            "Batch generation failed (%s). Retrying prompts one-by-one for isolation.",
            batch_exc,
        )

    outputs: list[tuple[str, TokenUsage | None, str | None]] = []
    for prompt in prompts:
        try:
            text, usage = asyncio.run(_generate_one(provider, prompt))
            outputs.append((text, usage, None))
        except Exception as single_exc:  # noqa: BLE001
            outputs.append(("", None, str(single_exc)))
    return outputs


def _run_assistant_phase(
    logger: logging.Logger,
    run_dir: Path,
    actions: list[dict[str, Any]],
    assistant_provider: InferenceProvider,
    assistant_config: InferenceConfig,
    attempts_by_phase: dict[PhaseKey, int],
    terminal_samples: set[str],
    failure_policy,
    total_turns: int,
) -> int:
    attempted = 0
    batch_size = max(1, assistant_config.generation.batch_size)
    batches = chunked(actions, batch_size)
    total_actions = len(actions)

    for batch_index, batch in enumerate(batches, start=1):
        log_phase_batch_progress(
            logger,
            stage_name="assistant",
            batch_index=batch_index,
            num_batches=len(batches),
            items_processed=min(batch_index * batch_size, total_actions),
            items_total=total_actions,
            turn_label=format_turn_label([action["display_turn_index"] for action in batch], total_turns),
        )

        prompt_batch = [action["prompt"] for action in batch]
        outputs = _generate_with_fallback(logger, assistant_provider, prompt_batch)
        if len(outputs) != len(batch):
            raise RuntimeError("Assistant phase output count did not match input action count.")

        for action, (response_text, usage, error_text) in zip(batch, outputs):
            attempted += 1
            key = _phase_key(action["sample_id"], "assistant", action["turn_index"])
            attempts_by_phase[key] = max(attempts_by_phase.get(key, 0), action["phase_attempt_no"])

            response = response_text if isinstance(response_text, str) else ""
            success = bool(response.strip()) and error_text is None
            error = None if success else (error_text or "empty_response")

            _record_rollout_event(
                run_dir,
                event_type="assistant_attempt",
                sample_id=action["sample_id"],
                payload={
                    "phase": "assistant",
                    "turn_index": action["turn_index"],
                    "attempt_no": action["phase_attempt_no"],
                    "status": "success" if success else "failed",
                    "error": error,
                    "provider": action["provider"],
                    "model": action["model"],
                    "token_usage": usage or {},
                },
            )

            inference_payload: dict[str, Any] = {
                "status": "success" if success else "failed",
                "model": action["model"],
                "provider": action["provider"],
                "attempt_no": action["global_attempt_no"],
                "token_usage": usage or {},
                "started_at": now_iso(),
                "completed_at": now_iso(),
                "error": error,
            }
            if success:
                inference_payload.update(
                    {
                        "assistant_message_id": message_append_id(
                            action["sample_id"],
                            "assistant",
                            action["turn_index"],
                        ),
                        "assistant_completion": response,
                        "assistant_full": response,
                        "assistant_message_metadata": {
                            "turn_index": action["turn_index"],
                            "source_stage": "rollout_assistant",
                            "provider": action["provider"],
                            "model": action["model"],
                            "token_usage": usage or {},
                            "parent_message_id": action["parent_message_id"],
                            "phase_attempt_no": action["phase_attempt_no"],
                        },
                    }
                )
            write_inference_result(
                run_dir,
                action["sample_id"],
                inference_payload,
                materialize=False,
            )

            max_attempts = failure_policy.assistant_max_attempts_per_turn
            if not success and max_attempts > 0 and action["phase_attempt_no"] >= max_attempts:
                terminal_samples.add(action["sample_id"])
                _record_terminal_failure(
                    run_dir,
                    sample_id=action["sample_id"],
                    phase="assistant",
                    turn_index=action["turn_index"],
                    attempt_no=action["phase_attempt_no"],
                    reason=error or "assistant_generation_failed",
                )

    return attempted


def _run_user_phase(
    logger: logging.Logger,
    run_dir: Path,
    actions: list[dict[str, Any]],
    user_provider: InferenceProvider,
    user_config: InferenceConfig,
    attempts_by_phase: dict[PhaseKey, int],
    terminal_samples: set[str],
    failure_policy,
    total_turns: int,
) -> int:
    attempted = 0
    batch_size = max(1, user_config.generation.batch_size)
    batches = chunked(actions, batch_size)
    total_actions = len(actions)

    for batch_index, batch in enumerate(batches, start=1):
        log_phase_batch_progress(
            logger,
            stage_name="user_simulator",
            batch_index=batch_index,
            num_batches=len(batches),
            items_processed=min(batch_index * batch_size, total_actions),
            items_total=total_actions,
            turn_label=format_turn_label([action["display_turn_index"] for action in batch], total_turns),
        )

        prompt_batch = [action["prompt"] for action in batch]
        outputs = _generate_with_fallback(logger, user_provider, prompt_batch)
        if len(outputs) != len(batch):
            raise RuntimeError("User phase output count did not match input action count.")

        for action, (user_text, usage, error_text) in zip(batch, outputs):
            attempted += 1
            key = _phase_key(action["sample_id"], "user", action["turn_index"])
            attempts_by_phase[key] = max(attempts_by_phase.get(key, 0), action["phase_attempt_no"])

            response = user_text if isinstance(user_text, str) else ""
            success = bool(response.strip()) and error_text is None
            error = None if success else (error_text or "empty_response")

            _record_rollout_event(
                run_dir,
                event_type="user_attempt",
                sample_id=action["sample_id"],
                payload={
                    "phase": "user",
                    "turn_index": action["turn_index"],
                    "attempt_no": action["phase_attempt_no"],
                    "status": "success" if success else "failed",
                    "error": error,
                    "provider": action["provider"],
                    "model": action["model"],
                    "token_usage": usage or {},
                },
            )

            if success:
                write_message_append(
                    run_dir,
                    action["sample_id"],
                    {
                        "message_id": message_append_id(
                            action["sample_id"],
                            "user",
                            action["turn_index"],
                        ),
                        "role": "user",
                        "content": response,
                        "editable": True,
                        "message_metadata": {
                            "turn_index": action["turn_index"],
                            "source_stage": "rollout_user_simulator",
                            "provider": action["provider"],
                            "model": action["model"],
                            "token_usage": usage or {},
                            "parent_message_id": action["parent_message_id"],
                            "phase_attempt_no": action["phase_attempt_no"],
                        },
                    },
                    materialize=False,
                )
            else:
                max_attempts = failure_policy.user_max_attempts_per_turn
                if max_attempts > 0 and action["phase_attempt_no"] >= max_attempts:
                    terminal_samples.add(action["sample_id"])
                    _record_terminal_failure(
                        run_dir,
                        sample_id=action["sample_id"],
                        phase="user",
                        turn_index=action["turn_index"],
                        attempt_no=action["phase_attempt_no"],
                        reason=error or "user_generation_failed",
                    )

    return attempted


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

    init_run(run_dir, base_config={"rollout_generation": config.model_dump(mode="json")})
    register_stage_fingerprint(run_dir, "rollout_generation", config.model_dump(mode="json"))

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
    user_config = _build_user_simulator_inference_config(config.user_simulator).model_copy(
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

    loop_index = 0
    while True:
        loop_index += 1
        materialize_canonical_samples(run_dir)
        samples = load_samples(run_dir)

        assistant_actions: list[dict[str, Any]] = []
        user_actions: list[dict[str, Any]] = []

        for sample in samples:
            if sample.sample_id in terminal_samples:
                continue

            completed_turns = _assistant_turn_count(sample)
            if completed_turns >= config.num_assistant_turns:
                continue

            latest_message = sample.messages[-1] if sample.messages else None
            if latest_message is None:
                terminal_samples.add(sample.sample_id)
                _record_invalid_state(run_dir, sample.sample_id, "empty_messages")
                _record_terminal_failure(
                    run_dir,
                    sample_id=sample.sample_id,
                    phase="system",
                    turn_index=completed_turns,
                    attempt_no=0,
                    reason="empty_messages",
                )
                continue

            if latest_message.role == "user":
                action = _build_assistant_action(
                    sample,
                    config=config,
                    assistant_config=assistant_config,
                    attempts_by_phase=attempts_by_phase,
                    terminal_samples=terminal_samples,
                    run_dir=run_dir,
                )
                if action is not None:
                    assistant_actions.append(action)
                continue

            if latest_message.role == "assistant":
                action = _build_user_action(
                    sample,
                    config=config,
                    user_config=user_config,
                    attempts_by_phase=attempts_by_phase,
                    terminal_samples=terminal_samples,
                    run_dir=run_dir,
                )
                if action is not None:
                    user_actions.append(action)
                continue

            terminal_samples.add(sample.sample_id)
            _record_invalid_state(run_dir, sample.sample_id, f"unsupported_terminal_role:{latest_message.role}")
            _record_terminal_failure(
                run_dir,
                sample_id=sample.sample_id,
                phase="system",
                turn_index=completed_turns,
                attempt_no=0,
                reason=f"unsupported_terminal_role:{latest_message.role}",
            )

        logger.info(
            "Pass %d | pending assistant=%d user=%d terminal=%d",
            loop_index,
            len(assistant_actions),
            len(user_actions),
            len(terminal_samples),
        )

        attempted = 0
        attempted += _run_assistant_phase(
            logger,
            run_dir,
            assistant_actions,
            assistant_provider,
            assistant_config,
            attempts_by_phase,
            terminal_samples,
            config.failure_policy,
            config.num_assistant_turns,
        )
        attempted += _run_user_phase(
            logger,
            run_dir,
            user_actions,
            user_provider,
            user_config,
            attempts_by_phase,
            terminal_samples,
            config.failure_policy,
            config.num_assistant_turns,
        )

        materialize_canonical_samples(run_dir)
        if attempted == 0:
            break

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
        1 for sample in final_samples if _assistant_turn_count(sample) >= config.num_assistant_turns
    )
    completed_turns = sum(_assistant_turn_count(sample) for sample in final_samples)
    failed = sum(
        1 for sample in final_samples if _assistant_turn_count(sample) < config.num_assistant_turns
    )

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
