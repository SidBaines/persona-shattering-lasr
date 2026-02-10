"""Anthropic API inference provider."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from anthropic import AsyncAnthropic

from scripts.inference.providers.remote_base import AsyncInferenceProvider

if TYPE_CHECKING:
    from scripts.inference.config import InferenceConfig


def _extract_text(content: list[Any] | None) -> str:
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts).strip()


class AnthropicProvider(AsyncInferenceProvider):
    """Inference provider using the Anthropic API."""

    def __init__(self, config: "InferenceConfig") -> None:
        super().__init__(config)
        self.config = config
        self.anthropic_config = config.anthropic
        self.generation_config = config.generation

        api_key = os.environ.get(self.anthropic_config.api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found. Set the {self.anthropic_config.api_key_env} environment variable."
            )

        self.client = AsyncAnthropic(api_key=api_key)
        self.model = config.model

    async def _generate_one(self, prompt: str, **kwargs) -> str:
        gen_cfg = self.generation_config
        max_tokens = kwargs.get(
            "max_tokens",
            kwargs.get(
                "max_new_tokens",
                self.anthropic_config.max_tokens or gen_cfg.max_new_tokens,
            ),
        )
        temperature = kwargs.get("temperature", gen_cfg.temperature)
        top_p = kwargs.get("top_p", gen_cfg.top_p)

        params: dict[str, object] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.timeout is not None:
            params["timeout"] = self.timeout
        response = await self.client.messages.create(**params)
        return _extract_text(response.content)
