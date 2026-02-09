"""OpenRouter inference provider (OpenAI-compatible)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.inference.providers.openai_base import OpenAICompatibleProvider

if TYPE_CHECKING:
    from scripts.inference.config import InferenceConfig


class OpenRouterProvider(OpenAICompatibleProvider):
    """Inference provider using the OpenRouter API."""

    def __init__(self, config: "InferenceConfig") -> None:
        openrouter_cfg = config.openrouter
        headers: dict[str, str] = {}
        if openrouter_cfg.app_url:
            headers["HTTP-Referer"] = openrouter_cfg.app_url
        if openrouter_cfg.app_name:
            headers["X-Title"] = openrouter_cfg.app_name
        super().__init__(
            config,
            api_key_env=openrouter_cfg.api_key_env,
            base_url=openrouter_cfg.base_url,
            default_headers=headers or None,
        )
