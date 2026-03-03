"""Multi-turn conversation dataset generation."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset

from scripts.datasets import (
    export_dataset,
    ingest_source_dataset,
    init_run,
    load_samples,
    materialize_canonical_samples,
    record_stage_event,
    register_stage_fingerprint,
    render_messages,
    write_edit_overlay,
    write_inference_result,
    write_message_append,
)
from scripts.datasets.loaders import load_dataset_from_config
from scripts.datasets.schema import StageEventRecord
from scripts.editing.prompts import EditPromptContext, get_prompt
from scripts.editing.run import build_inference_config
from scripts.inference import InferenceConfig
from scripts.inference.providers import get_provider
from scripts.inference.providers.base import InferenceProvider, TokenUsage
from scripts.utils import setup_logging

from .config import (
    ConversationGenerationConfig,
    ConversationGenerationResult,
    ResponderConfig,
)


RESPONDER_TEMPLATES: dict[str, str] = {
    "natural_partner": (
        "You are continuing this conversation as a realistic user. "
        "Reply naturally to the assistant's latest message, ask follow-up questions when useful, "
        "and keep the conversation moving. Do not narrate or explain your role."
    ),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _assistant_turn_count(sample) -> int:
    return sum(1 for message in sample.messages if message.role == "assistant")


def _completed_edited_turns(sample, variant_name: str) -> int:
    assistant_ids = {message.message_id for message in sample.messages if message.role == "assistant"}
    successful_targets: set[str] = set()
    for variant in sample.edit_variants:
        if variant.variant_name != variant_name:
            continue
        for overlay in variant.overlays:
            if overlay.status == "success" and overlay.target_message_id in assistant_ids:
                successful_targets.add(overlay.target_message_id)
    return len(successful_targets)


def _latest_assistant(sample):
    return next((message for message in reversed(sample.messages) if message.role == "assistant"), None)


def _has_successful_overlay(sample, variant_name: str, target_message_id: str) -> bool:
    for variant in sample.edit_variants:
        if variant.variant_name != variant_name:
            continue
        for overlay in sorted(
            variant.overlays,
            key=lambda item: (item.attempt_no, item.overlay_id),
            reverse=True,
        ):
            if overlay.target_message_id == target_message_id and overlay.status == "success":
                return True
    return False


def _build_responder_inference_config(config: ResponderConfig) -> InferenceConfig:
    return InferenceConfig(
        model=config.model,
        provider=config.provider,
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


def _build_responder_messages(sample, editing_variant: str, prompt_template: str) -> list[dict[str, str]]:
    rendered = [{"role": message.role, "content": message.content} for message in render_messages(sample, editing_variant)]
    instruction = RESPONDER_TEMPLATES.get(prompt_template)
    if instruction:
        return [{"role": "system", "content": instruction}, *rendered]
    return rendered


async def _generate_one(
    provider: InferenceProvider,
    prompt: str | list[dict[str, str]],
) -> tuple[str, TokenUsage | None]:
    responses, usages, _ = await provider.generate_batch_with_details_async([prompt], num_responses=1)
    if not responses:
        return "", None
    usage = usages[0] if usages else None
    return responses[0], usage


def _message_append_id(sample_id: str, role: str, turn_index: int) -> str:
    digest = hashlib.sha256(f"{sample_id}:{role}:{turn_index}".encode("utf-8")).hexdigest()[:24]
    return f"msg_{digest}"


def _record_invalid_state(run_dir: Path, sample_id: str, reason: str) -> None:
    record_stage_event(
        run_dir,
        StageEventRecord(
            event_id=f"evt_{hashlib.sha256(f'{sample_id}:{reason}:{_now_iso()}'.encode('utf-8')).hexdigest()[:24]}",
            stage="conversation_generation",
            event_type="invalid_state",
            sample_id=sample_id,
            created_at=_now_iso(),
            payload={"reason": reason},
        ),
    )


def run_conversation_generation(
    config: ConversationGenerationConfig,
    dataset: Dataset | None = None,
) -> tuple[Dataset, ConversationGenerationResult]:
    """Generate edited multi-turn conversations and export them."""
    logger = setup_logging()
    run_dir = Path(config.run_dir)

    init_run(
        run_dir,
        base_config={"conversation_generation": config.model_dump(mode="json")},
    )
    register_stage_fingerprint(
        run_dir,
        "conversation_generation",
        config.model_dump(mode="json"),
    )

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
        responses_per_input=1,
    )

    assistant_config = config.assistant_inference.model_copy(
        update={
            "run_dir": None,
            "output_path": None,
            "resume": False,
            "overwrite_output": False,
            "continue_on_error": False,
        }
    )
    editing_config = config.editing.model_copy(
        update={
            "run_dir": None,
            "output_path": None,
            "resume": False,
            "overwrite_output": False,
            "variant_name": config.editing_variant,
            "total_turns_hint": config.num_assistant_turns,
            "quality": config.editing.quality.model_copy(update={"enabled": False}),
        }
    )
    responder_config = _build_responder_inference_config(config.responder)
    editor_inference_config = (
        None if editing_config.provider == "code" else build_inference_config(editing_config)
    )

    assistant_provider = get_provider(assistant_config.provider, assistant_config)
    editor_provider = (
        None
        if editing_config.provider == "code"
        else get_provider(editor_inference_config.provider, editor_inference_config)
    )
    responder_provider = get_provider(responder_config.provider, responder_config)

    failed_samples: set[str] = set()

    while True:
        materialize_canonical_samples(run_dir)
        samples = load_samples(run_dir)
        progressed = False

        for sample in samples:
            completed_turns = _completed_edited_turns(sample, config.editing_variant)
            if completed_turns >= config.num_assistant_turns:
                continue

            latest_message = sample.messages[-1] if sample.messages else None
            if latest_message is None:
                failed_samples.add(sample.sample_id)
                _record_invalid_state(run_dir, sample.sample_id, "empty_messages")
                continue

            if latest_message.role == "user":
                prompt = [{"role": message.role, "content": message.content} for message in sample.messages]
                response_text, usage = asyncio.run(_generate_one(assistant_provider, prompt))
                status = "success" if response_text.strip() else "failed"
                assistant_turn_index = _assistant_turn_count(sample)
                write_inference_result(
                    run_dir,
                    sample.sample_id,
                    {
                        "status": status,
                        "model": assistant_config.model,
                        "provider": assistant_config.provider,
                        "assistant_message_id": _message_append_id(
                            sample.sample_id,
                            "assistant",
                            assistant_turn_index,
                        ),
                        "assistant_completion": response_text,
                        "assistant_full": response_text,
                        "assistant_message_metadata": {
                            "turn_index": assistant_turn_index,
                            "source_stage": "assistant_base",
                            "provider": assistant_config.provider,
                            "model": assistant_config.model,
                            "token_usage": usage or {},
                            "parent_message_id": latest_message.message_id,
                        },
                        "attempt_no": sample.inference.attempt_no + 1,
                        "token_usage": usage or {},
                        "started_at": _now_iso(),
                        "completed_at": _now_iso(),
                        "error": None if status == "success" else "empty_response",
                    },
                    materialize=False,
                )
                progressed = True
                continue

            if latest_message.role != "assistant":
                failed_samples.add(sample.sample_id)
                _record_invalid_state(
                    run_dir,
                    sample.sample_id,
                    f"unsupported_terminal_role:{latest_message.role}",
                )
                continue

            if not _has_successful_overlay(sample, config.editing_variant, latest_message.message_id):
                turn_index = _assistant_turn_count(sample) - 1
                latest_user_message = next(
                    (
                        message.content
                        for message in reversed(sample.messages[:-1])
                        if message.role == "user"
                    ),
                    "",
                )
                prompt_context = EditPromptContext(
                    conversation_history=[
                        {"role": message.role, "content": message.content}
                        for message in sample.messages[:-1]
                    ],
                    latest_user_message=latest_user_message,
                    base_assistant_response=latest_message.content,
                    turn_index=max(turn_index, 0),
                    total_turns=config.num_assistant_turns,
                )
                if editing_config.provider == "code":
                    raise NotImplementedError("Code-based editing is not supported in conversation generation.")
                assert editor_provider is not None
                prior_attempts = 0
                for variant in sample.edit_variants:
                    if variant.variant_name != config.editing_variant:
                        continue
                    prior_attempts = max(
                        (
                            overlay.attempt_no
                            for overlay in variant.overlays
                            if overlay.target_message_id == latest_message.message_id
                        ),
                        default=0,
                    )
                    break
                prompt = get_prompt(editing_config.prompt_template, context=prompt_context)
                edited_text, usage = asyncio.run(_generate_one(editor_provider, prompt))
                status = "success" if edited_text.strip() else "failed"
                overlay_id = hashlib.sha256(
                    f"{sample.sample_id}:{config.editing_variant}:{latest_message.message_id}:{edited_text}".encode("utf-8")
                ).hexdigest()[:24]
                write_edit_overlay(
                    run_dir,
                    sample_id=sample.sample_id,
                    variant_name=config.editing_variant,
                    overlay_payload={
                        "overlay_id": overlay_id,
                        "target_message_id": latest_message.message_id,
                        "target_role": "assistant",
                        "original_content_hash": hashlib.sha256(
                            latest_message.content.encode("utf-8")
                        ).hexdigest(),
                        "edited_content": edited_text,
                        "status": status,
                        "attempt_no": prior_attempts + 1,
                        "editor_model": editing_config.model,
                        "editor_provider": editing_config.provider,
                        "edit_prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                        "token_usage": usage or {},
                        "judge_metadata": None,
                        "timestamps": {"created_at": _now_iso()},
                        "error": None if status == "success" else "empty_edited_response",
                    },
                    materialize=False,
                )
                progressed = True
                continue

            if completed_turns + 1 > config.num_assistant_turns:
                continue

            responder_messages = _build_responder_messages(
                sample,
                config.editing_variant,
                config.responder.prompt_template,
            )
            user_text, usage = asyncio.run(_generate_one(responder_provider, responder_messages))
            if not user_text.strip():
                failed_samples.add(sample.sample_id)
                _record_invalid_state(run_dir, sample.sample_id, "empty_responder_user_turn")
                continue
            write_message_append(
                run_dir,
                sample.sample_id,
                {
                    "message_id": _message_append_id(
                        sample.sample_id,
                        "user",
                        _assistant_turn_count(sample),
                    ),
                    "role": "user",
                    "content": user_text,
                    "editable": True,
                    "message_metadata": {
                        "turn_index": _assistant_turn_count(sample),
                        "source_stage": "responder_user",
                        "provider": responder_config.provider,
                        "model": responder_config.model,
                        "token_usage": usage or {},
                        "parent_message_id": latest_message.message_id,
                    },
                },
                materialize=False,
            )
            progressed = True

        materialize_canonical_samples(run_dir)
        if not progressed:
            break

    export_path = export_dataset(
        run_dir,
        profile="conversation_training",
        variant_name=config.editing_variant,
    )
    trace_path = export_dataset(
        run_dir,
        profile="conversation_trace",
        variant_name=config.editing_variant,
    )
    result_dataset = load_dataset_from_config(
        config.dataset.model_copy(
            update={
                "source": "canonical",
                "path": str(run_dir),
                "name": f"editing:{config.editing_variant}",
            }
        )
    )

    final_samples = load_samples(run_dir)
    completed = sum(
        1
        for sample in final_samples
        if _completed_edited_turns(sample, config.editing_variant) >= config.num_assistant_turns
    )
    completed_turns = sum(
        _completed_edited_turns(sample, config.editing_variant)
        for sample in final_samples
    )
    result = ConversationGenerationResult(
        output_path=Path(export_path),
        num_conversations=len(final_samples),
        num_completed=completed,
        num_failed=len(failed_samples),
        num_assistant_turns_target=config.num_assistant_turns,
        num_assistant_turns_completed=completed_turns,
        exports={
            "conversation_training": str(export_path),
            "conversation_trace": str(trace_path),
        },
    )
    logger.info(
        "Conversation generation complete: %d/%d complete, failed=%d.",
        completed,
        len(final_samples),
        len(failed_samples),
    )
    return result_dataset, result
