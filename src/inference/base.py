"""Abstract base class for inference providers."""

from abc import ABC, abstractmethod


class InferenceProvider(ABC):
    """Abstract base class for inference providers."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for the given prompt.

        Args:
            prompt: The input prompt.
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)

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
            List of generated response strings.
        """
        pass

    @abstractmethod
    def load_model(self, config: dict) -> None:
        """Load the model with the given configuration.

        Args:
            config: Model configuration containing:
                - name: Model name/path
                - device: Device to load on (cuda, cpu, auto)
                - Additional provider-specific options
        """
        pass
