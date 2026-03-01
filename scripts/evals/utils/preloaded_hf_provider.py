"""Inspect ModelAPI provider that wraps a pre-loaded HuggingFace model.

Allows passing an already-in-memory ``AutoModelForCausalLM`` (or ``PeftModel``)
directly to ``inspect_ai.eval()`` without writing to disk.  This is the key
enabler for in-place LoRA scale sweeps: load the base model + adapter once,
apply ``LoRaScaling`` in-place for each scale point, run Inspect, restore.

Registration::

    from scripts.evals.utils.preloaded_hf_provider import register_preloaded_hf_provider
    register_preloaded_hf_provider()

Usage::

    from inspect_ai.model import get_model
    model_obj = get_model(
        "hf_preloaded/my-label",
        hf_model=peft_model,
        hf_tokenizer=tokenizer,
    )
    inspect_eval(task, model=model_obj, ...)

The ``model_name`` component of the URI (``my-label``) is cosmetic — it appears
in Inspect logs but has no functional effect.
"""

from __future__ import annotations

import copy
import functools
import json
import time
from logging import getLogger
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase
from typing_extensions import override

from inspect_ai._util.content import (
    ContentAudio,
    ContentDocument,
    ContentImage,
    ContentVideo,
)
from inspect_ai.model._reasoning import emulate_reasoning_history
from inspect_ai.model._chat_message import ChatMessage, ChatMessageAssistant
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model import ModelAPI
from inspect_ai.model._model_output import (
    ChatCompletionChoice,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.model._registry import modelapi
from inspect_ai.tool import ToolChoice, ToolInfo

logger = getLogger(__name__)

_PROVIDER_NAME = "hf_preloaded"
_registered = False


def register_preloaded_hf_provider() -> None:
    """Register the ``hf_preloaded`` Inspect model provider.

    Safe to call multiple times — registration only happens once.
    """
    global _registered
    if _registered:
        return

    @modelapi(name=_PROVIDER_NAME)
    class PreloadedHFAPI(ModelAPI):
        """Inspect ModelAPI backed by a pre-loaded HuggingFace model object."""

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

            if "hf_model" not in model_args:
                raise ValueError(
                    "hf_preloaded provider requires 'hf_model' in model_args"
                )
            if "hf_tokenizer" not in model_args:
                raise ValueError(
                    "hf_preloaded provider requires 'hf_tokenizer' in model_args"
                )

            self.model: Any = model_args["hf_model"]
            self.tokenizer: PreTrainedTokenizerBase = model_args["hf_tokenizer"]
            self.batch_size: int = int(model_args.get("batch_size", 32))

            # Ensure tokenizer is set up for batched generation.
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

        @override
        def close(self) -> None:
            # Do NOT destroy the model — the caller owns the lifetime.
            pass

        @override
        def max_tokens(self) -> int | None:
            from inspect_ai._util.constants import DEFAULT_MAX_TOKENS
            return DEFAULT_MAX_TOKENS

        @override
        def max_connections(self) -> int:
            return self.batch_size

        @override
        def collapse_user_messages(self) -> bool:
            return True

        async def generate(
            self,
            input: list[ChatMessage],
            tools: list[ToolInfo],
            tool_choice: ToolChoice,
            config: GenerateConfig,
        ) -> ModelOutput:
            chat = _apply_chat_template(self.tokenizer, self.model_name, input)

            tokenizer_fn = functools.partial(
                self.tokenizer,
                return_tensors="pt",
                padding=True,
            )

            kwargs: dict[str, Any] = dict(do_sample=True)
            if config.max_tokens is not None:
                kwargs["max_new_tokens"] = config.max_tokens
            if config.temperature is not None:
                kwargs["temperature"] = config.temperature
            if config.top_p is not None:
                kwargs["top_p"] = config.top_p
            if config.top_k is not None:
                kwargs["top_k"] = config.top_k
            if config.stop_seqs is not None:
                from transformers.generation import StopStringCriteria
                kwargs["stopping_criteria"] = [
                    StopStringCriteria(self.tokenizer, config.stop_seqs)
                ]
            kwargs["return_dict_in_generate"] = True

            generator = functools.partial(self.model.generate, **kwargs)
            decoder = functools.partial(
                self.tokenizer.batch_decode,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            device = next(self.model.parameters()).device
            t0 = time.monotonic()

            tokenized = tokenizer_fn([chat])
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            with torch.inference_mode():
                outputs = generator(input_ids=input_ids, attention_mask=attention_mask)

            generated_ids = outputs.sequences[:, input_ids.size(1):]
            decoded = decoder(sequences=generated_ids)
            elapsed = time.monotonic() - t0

            input_tokens = int(input_ids.size(1))
            output_tokens = int(generated_ids.size(1))

            return ModelOutput(
                model=self.model_name,
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessageAssistant(
                            content=decoded[0],
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
                time=elapsed,
            )

    _registered = True


def _apply_chat_template(tokenizer: Any, model_name: str, messages: list[ChatMessage]) -> str:
    """Convert Inspect ChatMessages to a single string via the tokenizer's chat template."""
    hf_messages = copy.deepcopy(emulate_reasoning_history(messages))

    # Flatten any list content to text (no multimodal support).
    for message in hf_messages:
        if isinstance(message.content, list):
            if any(
                isinstance(item, ContentAudio | ContentImage | ContentVideo | ContentDocument)
                for item in message.content
            ):
                raise NotImplementedError(
                    "hf_preloaded provider does not support multimodal content"
                )
            message.content = message.text

    if tokenizer.chat_template is not None:
        return str(
            tokenizer.apply_chat_template(
                hf_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        )

    # Fallback for tokenizers without a chat template.
    return "".join(f"{m.role}: {m.content}\n" for m in hf_messages)
