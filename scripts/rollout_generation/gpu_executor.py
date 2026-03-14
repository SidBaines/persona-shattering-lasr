"""Dynamic-batching GPU executor for local model inference within an async context."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.inference.providers.base import InferenceProvider, PromptInput

logger = logging.getLogger(__name__)

_SENTINEL = object()


class GpuBatchExecutor:
    """Collects prompts from concurrent coroutines and runs them as GPU batches.

    Callers await generate() to submit a prompt and receive its response.
    A background coroutine (run()) drains the queue in batches via
    asyncio.to_thread, so the event loop stays responsive while the GPU
    is busy — allowing user API calls to proceed concurrently.

    Usage::

        executor = GpuBatchExecutor(local_provider, batch_size=8)
        executor_task = asyncio.create_task(executor.run())
        await asyncio.gather(*conversation_coroutines)
        executor.stop()
        await executor_task
    """

    def __init__(
        self,
        provider: "InferenceProvider",
        batch_size: int,
        batch_timeout: float = 0.3,
    ) -> None:
        """Args:
            provider: Local inference provider (synchronous generate_batch).
            batch_size: Maximum prompts per GPU batch.
            batch_timeout: Seconds to wait for the batch to fill before flushing.
        """
        self._provider = provider
        self._batch_size = max(1, batch_size)
        self._batch_timeout = batch_timeout
        self._queue: asyncio.Queue = asyncio.Queue()

    def stop(self) -> None:
        """Signal run() to exit after draining remaining work."""
        self._queue.put_nowait(_SENTINEL)

    async def generate(self, prompt: "PromptInput") -> tuple[str, str | None]:
        """Submit a prompt and await its response.

        Returns:
            Tuple of (response_text, error_message_or_None).
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[str, str | None]] = loop.create_future()
        await self._queue.put((prompt, future))
        return await future

    async def run(self) -> None:
        """Background task: drain queue in GPU batches until stop() is called."""
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                break

            prompts: list = [item[0]]
            futures: list[asyncio.Future] = [item[1]]

            # Greedily collect more items up to batch_size within batch_timeout.
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._batch_timeout
            while len(prompts) < self._batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    next_item = await asyncio.wait_for(
                        self._queue.get(), timeout=remaining
                    )
                except asyncio.TimeoutError:
                    break
                if next_item is _SENTINEL:
                    # Re-queue sentinel so the outer loop sees it next iteration.
                    self._queue.put_nowait(_SENTINEL)
                    break
                prompts.append(next_item[0])
                futures.append(next_item[1])

            logger.debug("GPU batch executor: running batch of %d prompt(s)", len(prompts))

            # Run GPU inference in a thread. PyTorch releases the GIL during
            # CUDA kernels, so the event loop can service user API responses
            # while the GPU is busy.
            try:
                responses: list[str] = await asyncio.to_thread(
                    self._provider.generate_batch, prompts
                )
                for future, response in zip(futures, responses):
                    if not future.done():
                        future.set_result((response, None))
            except Exception as exc:  # noqa: BLE001
                err_str = str(exc)
                logger.warning("GPU batch failed: %s", err_str)
                for future in futures:
                    if not future.done():
                        future.set_result(("", err_str))
