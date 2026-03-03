"""Local inference provider using HuggingFace transformers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.inference.providers.base import InferenceProvider, PromptInput

if TYPE_CHECKING:
    from scripts.inference.config import InferenceConfig

logger = logging.getLogger(__name__)


class LocalProvider(InferenceProvider):
    """Inference provider using local HuggingFace transformers models."""

    def __init__(self, config: "InferenceConfig") -> None:
        """Initialize the local provider and load the model.

        Args:
            config: Inference configuration.
        """
        self.config = config
        self.local_config = config.local
        self.generation_config = config.generation

        self.model, self.tokenizer = self._load_model()
        self.device = next(self.model.parameters()).device
        self.prompt_format = self._resolve_prompt_format()
        logger.info("Local prompt format: %s", self.prompt_format)

    def _load_model(self):
        """Load the HuggingFace model and tokenizer."""
        model_name = self.config.model
        local_cfg = self.local_config

        dtype = getattr(torch, local_cfg.dtype, None)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {local_cfg.dtype}")

        logger.info("Loading model: %s", model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=local_cfg.revision,
            torch_dtype=dtype,
            device_map=local_cfg.device_map,
        )
        if local_cfg.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "Local provider adapter_path requires the 'peft' package."
                ) from exc
            logger.info("Loading LoRA adapter: %s", local_cfg.adapter_path)
            model = PeftModel.from_pretrained(model, local_cfg.adapter_path)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=local_cfg.revision,
            use_fast=True,
        )
        tokenizer.padding_side = "left"

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model.resize_token_embeddings(len(tokenizer))

        model.config.pad_token_id = tokenizer.pad_token_id
        model.eval()

        return model, tokenizer

    def _resolve_eos_token_id(self) -> int | list[int] | None:
        """Resolve EOS token ids without dropping model-specific stop ids.

        Some instruct/chat models define multiple EOS ids in generation config
        (for example end-of-turn markers). Passing only tokenizer.eos_token_id
        can override that and cause generation to run until max_new_tokens.
        """
        eos_ids: list[int] = []

        model_eos = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(model_eos, int):
            eos_ids.append(model_eos)
        elif isinstance(model_eos, list):
            eos_ids.extend(int(token_id) for token_id in model_eos)

        tokenizer_eos = self.tokenizer.eos_token_id
        if tokenizer_eos is not None:
            eos_ids.append(int(tokenizer_eos))

        # Keep order stable, remove duplicates.
        eos_ids = list(dict.fromkeys(eos_ids))
        if not eos_ids:
            return None
        if len(eos_ids) == 1:
            return eos_ids[0]
        return eos_ids

    def _resolve_prompt_format(self) -> str:
        """Resolve how prompts should be formatted before tokenization."""
        configured = self.local_config.prompt_format
        if configured in {"chat", "plain"}:
            return configured

        chat_template = getattr(self.tokenizer, "chat_template", None)
        if isinstance(chat_template, str) and chat_template.strip():
            return "chat"
        return "plain"

    def _format_prompts_for_model(self, prompts: list[str]) -> list[str]:
        """Format prompts based on detected/configured model prompt style."""
        if self.prompt_format != "chat":
            return prompts

        if not hasattr(self.tokenizer, "apply_chat_template"):
            logger.warning(
                "prompt_format=chat but tokenizer has no apply_chat_template; "
                "falling back to plain prompts."
            )
            return prompts

        formatted_prompts: list[str] = []
        for prompt in prompts:
            messages: list[dict[str, str]] = []
            if self.local_config.chat_system_prompt:
                messages.append(
                    {"role": "system", "content": self.local_config.chat_system_prompt}
                )
            messages.append({"role": "user", "content": prompt})
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                formatted_prompts.append(formatted)
            except Exception as exc:
                logger.warning(
                    "Failed applying chat template; falling back to plain prompt. "
                    "error=%s",
                    exc,
                )
                formatted_prompts.append(prompt)
        return formatted_prompts

    def _render_plain_messages(self, messages: list[dict[str, str]]) -> str:
        """Render chat messages into a plain transcript fallback."""
        lines: list[str] = []
        for message in messages:
            role = message.get("role", "user").capitalize()
            content = message.get("content", "")
            if role == "Assistant" and not content:
                lines.append("Assistant:")
            else:
                lines.append(f"{role}: {content}")
        if not messages or messages[-1].get("role") != "assistant":
            lines.append("Assistant:")
        return "\n".join(lines)

    def _format_prompt_input(self, prompt: PromptInput) -> str:
        """Format one prompt input into the string expected by the tokenizer."""
        if isinstance(prompt, str):
            return prompt

        if self.prompt_format == "chat" and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    prompt,
                    add_generation_prompt=prompt[-1]["role"] != "assistant",
                    tokenize=False,
                )
            except Exception as exc:
                logger.warning(
                    "Failed applying chat template to prompt messages; using plain transcript. error=%s",
                    exc,
                )

        return self._render_plain_messages(prompt)

    def _normalize_eos_ids(self, eos_token_id: int | list[int] | None) -> set[int]:
        """Normalize eos_token_id into a set for finish-reason checks."""
        if eos_token_id is None:
            return set()
        if isinstance(eos_token_id, int):
            return {int(eos_token_id)}
        return {int(token_id) for token_id in eos_token_id}

    def generate(self, prompt: PromptInput, **kwargs) -> str:
        """Generate a response for a single prompt.

        Args:
            prompt: The input prompt.
            **kwargs: Override generation parameters.

        Returns:
            Generated response string.
        """
        responses = self.generate_batch([prompt], **kwargs)
        return responses[0]

    def generate_batch(self, prompts: list[PromptInput], **kwargs) -> list[str]:
        """Generate responses for a batch of prompts.

        Args:
            prompts: List of input prompts.
            **kwargs: Override generation parameters.

        Returns:
            List of generated response strings.
        """
        gen_cfg = self.generation_config

        # Allow kwargs to override config values
        max_new_tokens = kwargs.get("max_new_tokens", gen_cfg.max_new_tokens)
        temperature = kwargs.get("temperature", gen_cfg.temperature)
        top_p = kwargs.get("top_p", gen_cfg.top_p)
        do_sample = kwargs.get("do_sample", gen_cfg.do_sample)
        num_responses = kwargs.get("num_responses", gen_cfg.num_responses_per_prompt)
        eos_token_id = kwargs.get("eos_token_id", self._resolve_eos_token_id())
        normalized_prompts = [self._format_prompt_input(prompt) for prompt in prompts]
        formatted_prompts = self._format_prompts_for_model(normalized_prompts)

        inputs = self.tokenizer(
            formatted_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            generated_output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=num_responses,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
                return_dict_in_generate=True,
            )

        generated = generated_output.sequences

        input_length = int(inputs["input_ids"].shape[1])
        responses = []
        eos_ids = self._normalize_eos_ids(eos_token_id)
        stopped_on_eos_count = 0
        hit_max_new_tokens_count = 0
        other_stop_count = 0
        for output_ids in generated:
            completion_ids = output_ids[input_length:]
            completion_token_ids = completion_ids.tolist()
            if eos_ids and any(token_id in eos_ids for token_id in completion_token_ids):
                stopped_on_eos_count += 1
            elif len(completion_token_ids) >= max_new_tokens:
                hit_max_new_tokens_count += 1
            else:
                other_stop_count += 1
            responses.append(
                self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            )

        logger.info(
            "Local generation finish reasons: stopped_on_eos=%d, "
            "hit_max_new_tokens=%d, other=%d",
            stopped_on_eos_count,
            hit_max_new_tokens_count,
            other_stop_count,
        )

        return responses
