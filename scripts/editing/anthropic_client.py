"""Thin wrapper around the Anthropic SDK with retry and rate limit handling."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from scripts.editing.config import EditingConfig

TokenUsage = dict[str, int]


def _extract_text(content: list[Any] | None) -> str:
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts).strip()


def _extract_usage(usage: Any) -> TokenUsage:
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


async def edit_response(
    prompt: str,
    config: "EditingConfig",
) -> tuple[str, TokenUsage]:
    """Send an editing request to the Anthropic API.

    Args:
        prompt: The full editing prompt.
        config: Editing configuration.

    Returns:
        Tuple of edited response text and token usage.
    """
    retry_config = config.retry

    @retry(
        reraise=True,
        stop=stop_after_attempt(retry_config.max_retries),
        wait=wait_exponential(multiplier=retry_config.backoff_factor),
    )
    async def _call() -> tuple[str, TokenUsage]:
        client = AsyncAnthropic()
        response = await client.messages.create(
            model=config.model,
            max_tokens=config.anthropic.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            timeout=config.timeout,
        )
        text = _extract_text(response.content)
        usage = _extract_usage(response.usage)
        return text, usage

    return await _call()
