"""Local inference provider using HuggingFace transformers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.inference.providers.base import InferenceProvider

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

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for a single prompt.

        Args:
            prompt: The input prompt.
            **kwargs: Override generation parameters.

        Returns:
            Generated response string.
        """
        responses = self.generate_batch([prompt], **kwargs)
        return responses[0]

    def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
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

        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=num_responses,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )

        input_length = int(inputs["input_ids"].shape[1])
        responses = []
        for output_ids in generated:
            completion_ids = output_ids[input_length:]
            responses.append(
                self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            )

        return responses
