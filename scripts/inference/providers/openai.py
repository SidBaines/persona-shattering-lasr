"""OpenAI API inference provider (Responses API)."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI

from scripts.inference.providers.remote_base import AsyncInferenceProvider
from scripts.inference.providers.base import TokenUsage, extract_usage

if TYPE_CHECKING:
    from scripts.inference.config import InferenceConfig

logger = logging.getLogger(__name__)


def _extract_output_text(response: Any) -> str:
    if response is None:
        return ""

    text = getattr(response, "output_text", None)
    if isinstance(text, str):
        return text.strip()
    if isinstance(response, dict):
        text = response.get("output_text")
        if isinstance(text, str):
            return text.strip()

    outputs = getattr(response, "output", None)
    if outputs is None and isinstance(response, dict):
        outputs = response.get("output")
    outputs = outputs or []

    parts: list[str] = []
    for item in outputs:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        content = content or []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                block_text = block.get("text")
            else:
                block_type = getattr(block, "type", None)
                block_text = getattr(block, "text", None)
            if block_type in ("output_text", "text") and block_text:
                parts.append(block_text)
    return "".join(parts).strip()


def _response_summary(response: Any) -> str:
    if response is None:
        return "status=unknown"
    if isinstance(response, dict):
        status = response.get("status")
        incomplete = response.get("incomplete_details")
        error = response.get("error")
    else:
        status = getattr(response, "status", None)
        incomplete = getattr(response, "incomplete_details", None)
        error = getattr(response, "error", None)
    return f"status={status!r} incomplete={incomplete!r} error={error!r}"


def _extract_response_usage(response: Any) -> TokenUsage | None:
    """Extract usage from OpenAI response object.

    The response object may contain usage directly, so we extract it
    before passing to the shared extract_usage utility.
    """
    if response is None:
        return None
    if isinstance(response, dict):
        usage = response.get("usage")
    else:
        usage = getattr(response, "usage", None)
    return extract_usage(usage)


class OpenAIProvider(AsyncInferenceProvider):
    """Inference provider using the OpenAI Responses API."""

    def __init__(self, config: "InferenceConfig") -> None:
        super().__init__(config)
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

        self.client = AsyncOpenAI(**client_kwargs)
        self.model = config.model
        logger.info("Initialized OpenAI provider with model: %s", self.model)

    async def _generate_one(self, prompt: str, **kwargs) -> tuple[str, TokenUsage | None]:
        """Generate a response using the OpenAI Responses API.

        Returns:
            Tuple of (generated_text, token_usage) where usage contains
            input_tokens, output_tokens, and total_tokens from the API response.
        """
        gen_cfg = self.generation_config
        openai_cfg = self.openai_config

        raw_max_output_tokens = kwargs.get(
            "max_output_tokens", kwargs.get("max_new_tokens", gen_cfg.max_new_tokens)
        )
        max_output_tokens = raw_max_output_tokens
        temperature = kwargs.get("temperature", gen_cfg.temperature)
        top_p = kwargs.get("top_p", gen_cfg.top_p)

        def _sampling_error(message: str) -> bool:
            lowered = message.lower()
            return "temperature" in lowered or "top_p" in lowered

        async def _create_response(
            prompt: str,
            *,
            include_sampling: bool = True,
            override_max_output_tokens: int | None = None,
        ):
            base_kwargs: dict[str, object] = {
                "model": self.model,
                "input": [{"role": "user", "content": prompt}],
                "max_output_tokens": override_max_output_tokens or max_output_tokens,
                "text": {
                    "format": {"type": "text"},
                },
            }
            if self.timeout is not None:
                base_kwargs["timeout"] = self.timeout
            if openai_cfg.verbosity:
                base_kwargs["text"]["verbosity"] = openai_cfg.verbosity
            if openai_cfg.reasoning_effort:
                base_kwargs["reasoning"] = {"effort": openai_cfg.reasoning_effort}
            if include_sampling:
                base_kwargs["temperature"] = temperature
                base_kwargs["top_p"] = top_p
            try:
                return await self.client.responses.create(**base_kwargs)
            except Exception as exc:
                if include_sampling and _sampling_error(str(exc)):
                    return await _create_response(prompt, include_sampling=False)
                raise

        response = await _create_response(prompt)
        text = _extract_output_text(response)
        usage = _extract_response_usage(response)
        if not text:
            logger.warning(
                "OpenAI Responses API returned empty text (%s).",
                _response_summary(response),
            )
        return text, usage
