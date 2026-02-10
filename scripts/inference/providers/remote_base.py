"""Shared async base for remote inference providers."""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, TypeVar, TYPE_CHECKING

from scripts.inference.providers.base import (
    InferenceProvider,
    TokenUsage,
    accumulate_usage,
    empty_usage,
)

if TYPE_CHECKING:
    from scripts.inference.config import InferenceConfig, RetryConfig

logger = logging.getLogger(__name__)
T = TypeVar("T")


class AsyncInferenceProvider(InferenceProvider):
    """Async-capable base class for remote providers with retries and concurrency."""

    def __init__(self, config: "InferenceConfig") -> None:
        self.config = config
        self.generation_config = config.generation
        self.retry_config: "RetryConfig" = config.retry
        self.max_concurrent = max(1, config.max_concurrent)
        self.timeout = config.timeout
        self.continue_on_error = config.continue_on_error
        self.log_failures = config.log_failures

    async def _generate_one(self, prompt: str, **kwargs) -> tuple[str, TokenUsage | None]:
        """Generate a response for a single prompt (async).

        Args:
            prompt: The input prompt string.
            **kwargs: Additional generation parameters (e.g., temperature, max_tokens).

        Returns:
            Tuple of (generated_text, token_usage).
            - generated_text: The model's response as a string.
            - token_usage: Dict with 'input_tokens', 'output_tokens', 'total_tokens',
              or None if usage information is unavailable.
        """
        raise NotImplementedError

    async def _call_with_retry(
        self,
        func: Callable[[], Awaitable[T]],
        *,
        context: str,
    ) -> T:
        max_attempts = max(1, self.retry_config.max_retries)
        for attempt in range(1, max_attempts + 1):
            try:
                return await func()
            except Exception as exc:
                if attempt >= max_attempts:
                    raise
                delay = self.retry_config.backoff_factor * (2 ** (attempt - 1))
                if self.log_failures:
                    logger.warning(
                        "%s attempt %d/%d failed: %s. Retrying in %.2fs",
                        context,
                        attempt,
                        max_attempts,
                        exc,
                        delay,
                    )
                await asyncio.sleep(delay)
        raise RuntimeError("Retry loop exited unexpectedly.")

    async def generate_batch_with_metadata_async(
        self, prompts: list[str], **kwargs
    ) -> tuple[list[str], TokenUsage, int]:
        gen_cfg = self.generation_config
        num_responses = kwargs.get("num_responses", gen_cfg.num_responses_per_prompt)
        total = len(prompts) * num_responses
        responses: list[str] = [""] * total
        usages: list[TokenUsage | None] = [None] * total
        failures: list[bool] = [False] * total
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_one(prompt_index: int, response_index: int) -> None:
            prompt = prompts[prompt_index]
            context = (
                f"{self.__class__.__name__} prompt={prompt_index} response={response_index}"
            )
            text: str
            usage: TokenUsage | None
            async with semaphore:
                try:
                    text, usage = await self._call_with_retry(
                        lambda: self._generate_one(prompt, **kwargs),
                        context=context,
                    )
                except Exception as exc:
                    if self.log_failures:
                        logger.warning("%s failed: %s", context, exc)
                    if not self.continue_on_error:
                        raise
                    text = ""
                    usage = None
            if not text:
                failures[prompt_index * num_responses + response_index] = True
            responses[prompt_index * num_responses + response_index] = text
            usages[prompt_index * num_responses + response_index] = usage

        tasks = [
            asyncio.create_task(run_one(prompt_index, response_index))
            for prompt_index in range(len(prompts))
            for response_index in range(num_responses)
        ]

        if not tasks:
            return responses, empty_usage(), 0

        if self.continue_on_error:
            await asyncio.gather(*tasks)
            total_usage = empty_usage()
            for usage in usages:
                accumulate_usage(total_usage, usage)
            failed_count = sum(1 for failed in failures if failed)
            return responses, total_usage, failed_count

        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_EXCEPTION
        )
        for task in done:
            exc = task.exception()
            if exc is not None:
                for pending_task in pending:
                    pending_task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise exc
        if pending:
            await asyncio.gather(*pending)

        total_usage = empty_usage()
        for usage in usages:
            accumulate_usage(total_usage, usage)
        failed_count = sum(1 for failed in failures if failed)
        return responses, total_usage, failed_count

    async def generate_batch_async(self, prompts: list[str], **kwargs) -> list[str]:
        responses, _, _ = await self.generate_batch_with_metadata_async(
            prompts, **kwargs
        )
        return responses

    def generate(self, prompt: str, **kwargs) -> str:
        responses = self.generate_batch([prompt], **kwargs)
        return responses[0] if responses else ""

    def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.generate_batch_async(prompts, **kwargs))
        raise RuntimeError(
            "generate_batch called inside a running event loop. "
            "Use generate_batch_async instead."
        )
