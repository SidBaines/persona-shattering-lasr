"""Abstract base class for inference providers."""

from abc import ABC, abstractmethod


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
