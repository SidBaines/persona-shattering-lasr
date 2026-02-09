"""OpenAI API inference provider (Responses API)."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from openai import OpenAI

from scripts.inference.providers.base import InferenceProvider

if TYPE_CHECKING:
    from scripts.inference.config import InferenceConfig

logger = logging.getLogger(__name__)


def _extract_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str):
        return text.strip()

    outputs = getattr(response, "output", None) or []
    parts: list[str] = []
    for item in outputs:
        content = getattr(item, "content", None) or []
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type in ("output_text", "text"):
                block_text = getattr(block, "text", None)
                if block_text:
                    parts.append(block_text)
    return "".join(parts).strip()


class OpenAIProvider(InferenceProvider):
    """Inference provider using the OpenAI Responses API."""

    def __init__(self, config: "InferenceConfig") -> None:
        self.config = config
        self.openai_config = config.openai
        self.generation_config = config.generation

        api_key = os.environ.get(self.openai_config.api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found. Set the {self.openai_config.api_key_env} environment variable."
            )

        client_kwargs: dict[str, object] = {"api_key": api_key}
        if self.openai_config.base_url:
            client_kwargs["base_url"] = self.openai_config.base_url
            logger.info("Using custom base URL: %s", self.openai_config.base_url)

        self.client = OpenAI(**client_kwargs)
        self.model = config.model
        logger.info("Initialized OpenAI provider with model: %s", self.model)

    def generate(self, prompt: str, **kwargs) -> str:
        responses = self.generate_batch([prompt], **kwargs)
        return responses[0] if responses else ""

    def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
        gen_cfg = self.generation_config
        num_responses = kwargs.get("num_responses", gen_cfg.num_responses_per_prompt)

        max_output_tokens = kwargs.get(
            "max_output_tokens", kwargs.get("max_new_tokens", gen_cfg.max_new_tokens)
        )
        temperature = kwargs.get("temperature", gen_cfg.temperature)
        top_p = kwargs.get("top_p", gen_cfg.top_p)

        def _sampling_error(message: str) -> bool:
            lowered = message.lower()
            return "temperature" in lowered or "top_p" in lowered

        def _create_response(prompt: str, *, include_sampling: bool = True):
            base_kwargs: dict[str, object] = {
                "model": self.model,
                "input": prompt,
                "max_output_tokens": max_output_tokens,
            }
            if include_sampling:
                base_kwargs["temperature"] = temperature
                base_kwargs["top_p"] = top_p
            try:
                return self.client.responses.create(**base_kwargs)
            except Exception as exc:
                if include_sampling and _sampling_error(str(exc)):
                    return _create_response(prompt, include_sampling=False)
                raise

        responses: list[str] = []
        for i, prompt in enumerate(prompts):
            for _ in range(num_responses):
                response = _create_response(prompt)
                responses.append(_extract_output_text(response))

            if (i + 1) % 10 == 0:
                logger.info(
                    "Generated %d/%d prompts (%d responses each).",
                    i + 1,
                    len(prompts),
                    num_responses,
                )

        return responses
