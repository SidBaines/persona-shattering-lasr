"""Abstract base class for inference providers."""

import asyncio
from abc import ABC, abstractmethod


TokenUsage = dict[str, int]


def empty_usage() -> TokenUsage:
    """Return a zeroed token usage dict."""
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def accumulate_usage(total: TokenUsage, usage: TokenUsage | None) -> None:
    """Accumulate token usage into total."""
    if not usage:
        return
    total["input_tokens"] += usage.get("input_tokens", 0)
    total["output_tokens"] += usage.get("output_tokens", 0)
    total["total_tokens"] += usage.get("total_tokens", 0)


class InferenceProvider(ABC):
    """Abstract base class for inference providers.

    Providers are initialized with configuration and handle their own
    model/client setup in __init__. The interface focuses on generation.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for a single prompt.

        Args:
            prompt: The input prompt.
            **kwargs: Additional generation parameters (temperature, max_tokens, num_responses, etc.)

        Returns:
            Generated response string.
        """
        pass

    @abstractmethod
    def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
        """Generate responses for a batch of prompts.

        Args:
            prompts: List of input prompts.
            **kwargs: Additional generation parameters.

        Returns:
            List of generated response strings in prompt-major order.
            If num_responses > 1, the list length should be
            len(prompts) * num_responses, with responses grouped per prompt.
        """
        pass

    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Async wrapper for generate().

        Default implementation runs the sync method in a thread.
        Providers can override for true async behavior.
        """
        return await asyncio.to_thread(self.generate, prompt, **kwargs)

    async def generate_batch_async(self, prompts: list[str], **kwargs) -> list[str]:
        """Async wrapper for generate_batch().

        Default implementation runs the sync method in a thread.
        Providers can override for true async behavior.
        """
        return await asyncio.to_thread(self.generate_batch, prompts, **kwargs)

    async def generate_batch_with_metadata_async(
        self, prompts: list[str], **kwargs
    ) -> tuple[list[str], TokenUsage, int]:
        """Generate responses and return usage + failure counts.

        Default implementation calls generate_batch_async and returns empty usage.
        Failures are estimated by counting empty responses.
        """
        responses = await self.generate_batch_async(prompts, **kwargs)
        failed_count = sum(1 for response in responses if not response)
        return responses, empty_usage(), failed_count
