"""Abstract base class for response editors."""

from abc import ABC, abstractmethod


class Editor(ABC):
    """Abstract base class for response editors."""

    @abstractmethod
    def edit(self, response: str, config: dict) -> str:
        """Edit a response to exhibit the target persona.

        Args:
            response: Original model response.
            config: Editor configuration containing:
                - prompt_template: Template for editing prompt
                - Additional editor-specific options

        Returns:
            Edited response string.
        """
        pass

    @abstractmethod
    def edit_batch(self, responses: list[str], config: dict) -> list[str]:
        """Edit a batch of responses.

        Args:
            responses: List of original responses.
            config: Editor configuration.

        Returns:
            List of edited response strings.
        """
        pass
