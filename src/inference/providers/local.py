"""Local transformers inference provider."""

from ..base import InferenceProvider


class LocalProvider(InferenceProvider):
    """Run inference locally using transformers."""

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model(self, config: dict) -> None:
        """Load model locally using transformers.

        Args:
            config: Model configuration containing:
                - name: HuggingFace model name
                - device: Device to load on (cuda, cpu, auto)
        """
        raise NotImplementedError(
            "LocalProvider not yet implemented. "
            "Implement in scripts/run_inference.py first, then migrate here."
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a single response."""
        raise NotImplementedError(
            "LocalProvider not yet implemented. "
            "Implement in scripts/run_inference.py first, then migrate here."
        )

    def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
        """Generate responses for a batch of prompts."""
        raise NotImplementedError(
            "LocalProvider not yet implemented. "
            "Implement in scripts/run_inference.py first, then migrate here."
        )
