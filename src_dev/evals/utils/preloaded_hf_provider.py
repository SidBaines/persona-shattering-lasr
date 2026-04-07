"""Inspect ModelAPI provider that wraps a pre-loaded HuggingFace model.

Allows passing an already-in-memory ``AutoModelForCausalLM`` (or ``PeftModel``)
directly to ``inspect_ai.eval()`` without writing to disk.  This is the key
enabler for in-place LoRA scale sweeps: load the base model + adapter once,
apply ``LoRaScaling`` in-place for each scale point, run Inspect, restore.

Registration::

    from src_dev.evals.utils.preloaded_hf_provider import register_preloaded_hf_provider
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
from logging import getLogger
from typing import Any

from transformers import PreTrainedTokenizerBase
from typing_extensions import override

from inspect_ai.model._providers.hf import GenerateInput, batched_generate, extract_logprobs

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
    Logprobs,
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

            # Use greedy decoding when temperature=0 (do_sample=True + temp=0 is
            # contradictory and triggers transformers warnings every batch).
            greedy = config.temperature is not None and config.temperature == 0.0
            kwargs: dict[str, Any] = dict(do_sample=not greedy)
            # Always set max_new_tokens to avoid the transformers default-max_length
            # warning that fires every batch when max_new_tokens is unset.
            kwargs["max_new_tokens"] = config.max_tokens if config.max_tokens is not None else self.max_tokens()
            if config.temperature is not None and not greedy:
                kwargs["temperature"] = config.temperature
            if config.top_p is not None:
                kwargs["top_p"] = config.top_p
            if config.top_k is not None:
                kwargs["top_k"] = config.top_k
            if config.logprobs is not None:
                kwargs["output_logits"] = config.logprobs
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

            # Use the built-in HF provider's batched_generate infrastructure.
            # It collects concurrent generate() calls from Inspect's async pool
            # and dispatches them as a single model.generate(batch) GPU call.
            response = await batched_generate(
                GenerateInput(
                    input=chat,
                    device=device,
                    tokenizer=tokenizer_fn,
                    generator=generator,
                    decoder=decoder,
                    batch_size=config.max_connections or self.max_connections(),
                )
            )

            # Gather logprobs if requested.
            final_logprobs = None
            if config.logprobs is not None and response.logprobs is not None:
                final_logprobs = extract_logprobs(
                    response=response,
                    top=config.top_logprobs,
                    tokenizer=self.tokenizer,
                )

            choice = ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content=response.output,
                    model=self.model_name,
                    source="generate",
                ),
                logprobs=(
                    Logprobs(content=final_logprobs)
                    if final_logprobs is not None
                    else None
                ),
            )

            return ModelOutput(
                model=self.model_name,
                choices=[choice],
                usage=ModelUsage(
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    total_tokens=response.total_tokens,
                ),
                time=response.time,
            )

    _registered = True


def _apply_chat_template(tokenizer: Any, model_name: str, messages: list[ChatMessage]) -> str:
    """Convert Inspect ChatMessages to a single string via the tokenizer's chat template.

    If the last message is from the assistant (i.e. a forced prefill), it is
    split off and appended *raw* after the generation prompt.  This avoids the
    chat template closing the assistant turn with an end-of-turn token, which
    would defeat the purpose of the prefill.
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
                    "hf_preloaded provider does not support multimodal content"
                )
            message.content = message.text

    # Detect trailing assistant prefill.
    assistant_prefill: str | None = None
    if hf_messages and hf_messages[-1].role == "assistant":
        prefill_msg = hf_messages.pop()
        content = prefill_msg.text if hasattr(prefill_msg, "text") else str(prefill_msg.content)
        if content:
            assistant_prefill = content

    if tokenizer.chat_template is not None:
        chat = str(
            tokenizer.apply_chat_template(
                hf_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        )
    else:
        # Fallback for tokenizers without a chat template.
        chat = "".join(f"{m.role}: {m.content}\n" for m in hf_messages)

    # Append the prefill text directly so it becomes a true continuation
    # of the (already-open) assistant turn, not a separate completed turn.
    if assistant_prefill is not None:
        chat += assistant_prefill

    return chat
