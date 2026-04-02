"""Inspect ModelAPI provider backed by vLLM's async engine for high-throughput local inference.

Uses :class:`vllm.AsyncLLMEngine` (aliased from ``vllm.v1.engine.async_llm.AsyncLLM``)
so that when Inspect's concurrent ``generate()`` calls each submit a request,
vLLM batches them internally with continuous batching — no polling or sleep overhead.

Registration::

    from src_dev.evals.utils.vllm_preloaded_provider import register_vllm_preloaded_provider
    register_vllm_preloaded_provider()

Usage::

    from src_dev.evals.utils.vllm_preloaded_provider import create_async_vllm_engine
    from inspect_ai.model import get_model

    engine = create_async_vllm_engine(
        model="meta-llama/Llama-3.1-8B-Instruct",
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
    )
    model_obj = get_model(
        "vllm_preloaded/my-label",
        vllm_engine=engine,
    )
    inspect_eval(task, model=model_obj, ...)

The ``model_name`` component of the URI (``my-label``) is cosmetic.
"""

from __future__ import annotations

import copy
import uuid
from logging import getLogger
from typing import Any

from typing_extensions import override

from inspect_ai._util.content import (
    ContentAudio,
    ContentDocument,
    ContentImage,
    ContentVideo,
)
from inspect_ai.model._chat_message import ChatMessage, ChatMessageAssistant
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model import ModelAPI
from inspect_ai.model._model_output import (
    ChatCompletionChoice,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.model._reasoning import emulate_reasoning_history
from inspect_ai.model._registry import modelapi

logger = getLogger(__name__)

_PROVIDER_NAME = "vllm_preloaded"
_registered = False


def create_async_vllm_engine(
    model: str,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.90,
    max_model_len: int | None = None,
    enforce_eager: bool = False,
    enable_lora: bool = False,
    max_loras: int = 1,
    max_lora_rank: int = 64,
) -> Any:
    """Create a shared vLLM AsyncLLMEngine for use with the ``vllm_preloaded`` provider.

    The engine is created once and should be shared across all concurrent
    ``generate()`` calls. vLLM handles batching internally via continuous batching.

    Args:
        model: HuggingFace model ID or local path.
        dtype: Torch dtype string (e.g. ``"bfloat16"``).
        gpu_memory_utilization: Fraction of GPU memory for vLLM's KV cache (0.0-1.0).
        max_model_len: Optional context length override.
        enforce_eager: Disable CUDA graphs (useful for debugging).
        enable_lora: Whether to enable LoRA adapter support.
        max_loras: Maximum number of LoRA adapters loaded simultaneously.
        max_lora_rank: Maximum LoRA rank supported.

    Returns:
        An ``AsyncLLM`` engine instance (``vllm.AsyncLLMEngine``).
    """
    from vllm import AsyncEngineArgs, AsyncLLMEngine

    engine_args = AsyncEngineArgs(
        model=model,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        enable_lora=enable_lora,
        max_loras=max_loras,
        max_lora_rank=max_lora_rank,
        trust_remote_code=False,
        disable_log_stats=True,
    )
    if max_model_len is not None:
        engine_args.max_model_len = max_model_len

    logger.info("Creating vLLM async engine for model: %s", model)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine


def register_vllm_preloaded_provider() -> None:
    """Register the ``vllm_preloaded`` Inspect model provider.

    Safe to call multiple times — registration only happens once.
    """
    global _registered
    if _registered:
        return

    @modelapi(name=_PROVIDER_NAME)
    class VllmPreloadedAPI(ModelAPI):
        """Inspect ModelAPI backed by a vLLM async engine for high-throughput inference.

        Each ``generate()`` call submits a request to the shared async engine
        and ``await``s the final result. vLLM handles continuous batching
        internally — no polling, no sleep, no queue drain timeout.
        """

        def __init__(
            self,
            model_name: str,
            base_url: str | None = None,
            api_key: str | None = None,
            config: GenerateConfig = GenerateConfig(),
            **model_args: Any,
        ) -> None:
            super().__init__(
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
                config=config,
            )

            if "vllm_engine" not in model_args:
                raise ValueError(
                    "vllm_preloaded provider requires 'vllm_engine' in model_args"
                )

            self._engine: Any = model_args["vllm_engine"]
            self._lora_request: Any = model_args.get("lora_request")
            self._batch_size: int = int(model_args.get("batch_size", 128))

        @override
        def close(self) -> None:
            # Engine lifetime is owned by the caller (suite.py / VLLMLoRaScaleProvider).
            pass

        @override
        def max_tokens(self) -> int | None:
            from inspect_ai._util.constants import DEFAULT_MAX_TOKENS
            return DEFAULT_MAX_TOKENS

        @override
        def max_connections(self) -> int:
            return self._batch_size

        @override
        def collapse_user_messages(self) -> bool:
            return True

        async def generate(
            self,
            input: list[ChatMessage],
            tools: list[Any],
            tool_choice: Any,
            config: GenerateConfig,
        ) -> ModelOutput:
            from vllm import SamplingParams

            # Apply chat template via the engine's tokenizer.
            tokenizer = await self._engine.get_tokenizer()
            prompt_token_ids = _apply_chat_template(tokenizer, input)

            # Build sampling params from Inspect's GenerateConfig.
            greedy = config.temperature is not None and config.temperature == 0.0
            sp_kwargs: dict[str, Any] = {}
            if config.temperature is not None and not greedy:
                sp_kwargs["temperature"] = config.temperature
            else:
                sp_kwargs["temperature"] = 0.0
            if config.top_p is not None:
                sp_kwargs["top_p"] = config.top_p
            if config.top_k is not None:
                sp_kwargs["top_k"] = config.top_k
            sp_kwargs["max_tokens"] = (
                config.max_tokens if config.max_tokens is not None else self.max_tokens()
            )
            if config.stop_seqs:
                sp_kwargs["stop"] = config.stop_seqs

            sampling_params = SamplingParams(**sp_kwargs)

            # Submit request to the async engine and consume the stream until finished.
            request_id = str(uuid.uuid4())
            final_output = None
            async for output in self._engine.generate(
                prompt={"prompt_token_ids": prompt_token_ids},
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=self._lora_request,
            ):
                final_output = output

            if final_output is None:
                raise RuntimeError(
                    f"vLLM engine returned no output for request {request_id}"
                )

            completion = final_output.outputs[0]
            input_tokens = len(prompt_token_ids)
            output_tokens = len(completion.token_ids)

            return ModelOutput(
                model=self.model_name,
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessageAssistant(
                            content=completion.text,
                            model=self.model_name,
                            source="generate",
                        )
                    )
                ],
                usage=ModelUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                ),
            )

    _registered = True


def _apply_chat_template(
    tokenizer: Any, messages: list[ChatMessage]
) -> list[int]:
    """Convert Inspect ChatMessages to token IDs via the tokenizer's chat template.

    Args:
        tokenizer: A HuggingFace-compatible tokenizer with ``apply_chat_template``.
        messages: Inspect chat messages.

    Returns:
        List of token IDs ready for vLLM's ``generate()``.
    """
    hf_messages = copy.deepcopy(emulate_reasoning_history(messages))

    # Flatten any list content to text (no multimodal support).
    for message in hf_messages:
        if isinstance(message.content, list):
            if any(
                isinstance(item, ContentAudio | ContentImage | ContentVideo | ContentDocument)
                for item in message.content
            ):
                raise NotImplementedError(
                    "vllm_preloaded provider does not support multimodal content"
                )
            message.content = message.text

    if tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            hf_messages,
            add_generation_prompt=True,
            tokenize=True,
        )

    # Fallback: concatenate role/content and tokenize.
    text = "".join(f"{m.role}: {m.content}\n" for m in hf_messages)
    return tokenizer.encode(text)
