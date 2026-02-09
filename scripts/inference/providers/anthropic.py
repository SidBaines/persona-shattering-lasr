"""Anthropic API inference provider."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from anthropic import Anthropic

from scripts.inference.providers.base import InferenceProvider

if TYPE_CHECKING:
    from scripts.inference.config import InferenceConfig

logger = logging.getLogger(__name__)


def _extract_text(content: list[Any] | None) -> str:
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts).strip()


class AnthropicProvider(InferenceProvider):
    """Inference provider using the Anthropic API."""

    def __init__(self, config: "InferenceConfig") -> None:
        self.config = config
        self.anthropic_config = config.anthropic
        self.generation_config = config.generation

        api_key = os.environ.get(self.anthropic_config.api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found. Set the {self.anthropic_config.api_key_env} environment variable."
            )

        self.client = Anthropic(api_key=api_key)
        self.model = config.model

    def generate(self, prompt: str, **kwargs) -> str:
        responses = self.generate_batch([prompt], **kwargs)
        return responses[0] if responses else ""

    def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
        gen_cfg = self.generation_config
        num_responses = kwargs.get("num_responses", gen_cfg.num_responses_per_prompt)
        max_tokens = kwargs.get(
            "max_tokens",
            kwargs.get(
                "max_new_tokens",
                self.anthropic_config.max_tokens or gen_cfg.max_new_tokens,
            ),
        )
        temperature = kwargs.get("temperature", gen_cfg.temperature)
        top_p = kwargs.get("top_p", gen_cfg.top_p)

        responses: list[str] = []

        for i, prompt in enumerate(prompts):
            for _ in range(num_responses):
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    messages=[{"role": "user", "content": prompt}],
                )
                responses.append(_extract_text(response.content))

            if (i + 1) % 10 == 0:
                logger.info(
                    "Generated %d/%d prompts (%d responses each).",
                    i + 1,
                    len(prompts),
                    num_responses,
                )

        return responses
