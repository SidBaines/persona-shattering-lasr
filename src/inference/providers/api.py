"""API-based inference provider."""

from ..base import InferenceProvider


class APIProvider(InferenceProvider):
    """Run inference via API (OpenAI, Anthropic, OpenRouter, etc.)."""

    def __init__(self):
        self.client = None
        self.model_name = None

    def load_model(self, config: dict) -> None:
        """Configure the API client.

        Args:
            config: API configuration containing:
                - name: Model name (e.g., "gpt-4", "claude-3-opus")
                - provider: API provider (openai, anthropic, openrouter)
        """
        raise NotImplementedError(
            "APIProvider not yet implemented. "
            "Implement in scripts/ first, then migrate here."
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a single response via API."""
        raise NotImplementedError(
            "APIProvider not yet implemented. "
            "Implement in scripts/ first, then migrate here."
        )

    def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
        """Generate responses for a batch of prompts via API."""
        raise NotImplementedError(
            "APIProvider not yet implemented. "
            "Implement in scripts/ first, then migrate here."
        )
