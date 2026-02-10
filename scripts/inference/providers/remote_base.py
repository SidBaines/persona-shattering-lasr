"""Shared async base for remote inference providers."""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, TypeVar, TYPE_CHECKING

from scripts.inference.providers.base import InferenceProvider

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

    async def _generate_one(self, prompt: str, **kwargs) -> str:
        """Generate a response for a single prompt (async)."""
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

    async def generate_batch_async(self, prompts: list[str], **kwargs) -> list[str]:
        gen_cfg = self.generation_config
        num_responses = kwargs.get("num_responses", gen_cfg.num_responses_per_prompt)
        total = len(prompts) * num_responses
        responses: list[str] = [""] * total
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_one(prompt_index: int, response_index: int) -> None:
            prompt = prompts[prompt_index]
            context = (
                f"{self.__class__.__name__} prompt={prompt_index} response={response_index}"
            )
            async with semaphore:
                try:
                    text = await self._call_with_retry(
                        lambda: self._generate_one(prompt, **kwargs),
                        context=context,
                    )
                except Exception as exc:
                    if self.log_failures:
                        logger.warning("%s failed: %s", context, exc)
                    if not self.continue_on_error:
                        raise
                    text = ""
            responses[prompt_index * num_responses + response_index] = text

        tasks = [
            asyncio.create_task(run_one(prompt_index, response_index))
            for prompt_index in range(len(prompts))
            for response_index in range(num_responses)
        ]

        if not tasks:
            return responses

        if self.continue_on_error:
            await asyncio.gather(*tasks)
            return responses

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
