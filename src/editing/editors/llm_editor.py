"""LLM-based response editor."""

from ..base import Editor


class LLMEditor(Editor):
    """Edit responses using an LLM (Anthropic, OpenAI, etc.)."""

    def __init__(self):
        self.client = None

    def edit(self, response: str, config: dict) -> str:
        """Edit a response using an LLM.

        Args:
            response: Original model response.
            config: Editor configuration containing:
                - provider: API provider (anthropic, openai)
                - model: Model name
                - prompt_template: Template with {response} placeholder

        Returns:
            Edited response.
        """
        raise NotImplementedError(
            "LLMEditor not yet implemented. "
            "Implement in scripts/edit_responses.py first, then migrate here."
        )

    def edit_batch(self, responses: list[str], config: dict) -> list[str]:
        """Edit a batch of responses."""
        raise NotImplementedError(
            "LLMEditor not yet implemented. "
            "Implement in scripts/edit_responses.py first, then migrate here."
        )
