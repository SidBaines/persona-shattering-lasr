"""Thin wrapper around the OpenAI SDK with retry and rate limit handling."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from scripts.config.schema import EditingConfig

TokenUsage = dict[str, int]


def _extract_text(response: Any) -> str:
    if not response.choices:
        return ""
    message = response.choices[0].message
    return (message.content or "").strip()


def _extract_usage(usage: Any) -> TokenUsage:
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


async def edit_response(
    prompt: str,
    config: EditingConfig,
) -> tuple[str, TokenUsage]:
    """Send an editing request to the OpenAI API.

    Args:
        prompt: The full editing prompt.
        config: Editing configuration.

    Returns:
        Tuple of edited response text and token usage.
    """
    retry_config = config.retry
    model = config.openai.model or config.model

    @retry(
        reraise=True,
        stop=stop_after_attempt(retry_config.max_retries),
        wait=wait_exponential(multiplier=retry_config.backoff_factor),
    )
    async def _call() -> tuple[str, TokenUsage]:
        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.openai.max_tokens,
            timeout=config.timeout,
        )
        text = _extract_text(response)
        usage = _extract_usage(response.usage)
        return text, usage

    return await _call()
