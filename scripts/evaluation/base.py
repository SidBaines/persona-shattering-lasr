"""Abstract base class for evaluations."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod


class Evaluation(ABC):
    """Abstract base class for evaluations.

    Evaluations take a response (and optionally a question) and return
    a dict of metric values. They can be used at any pipeline stage.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this evaluation."""
        ...

    @abstractmethod
    def evaluate(
        self, response: str, question: str | None = None
    ) -> dict[str, float | int | str]:
        """Evaluate a single response.

        Args:
            response: The response text to evaluate.
            question: Optional question/prompt that produced the response.

        Returns:
            Dict with keys like "{name}.metric_name" and numeric or string values.
        """
        ...

    def evaluate_batch(
        self,
        responses: list[str],
        questions: list[str | None] | None = None,
    ) -> list[dict[str, float | int | str]]:
        """Evaluate a batch of responses.

        Default implementation iterates over items. Subclasses may override
        for more efficient batch processing (e.g., batched LLM calls).

        Args:
            responses: List of response texts.
            questions: Optional list of questions (same length as responses).

        Returns:
            List of metric dicts, one per response.
        """
        if questions is None:
            questions = [None] * len(responses)
        if len(responses) != len(questions):
            raise ValueError(
                f"responses and questions must have the same length, "
                f"got {len(responses)} and {len(questions)}"
            )
        return [
            self.evaluate(response, question)
            for response, question in zip(responses, questions)
        ]

    async def evaluate_async(
        self, response: str, question: str | None = None
    ) -> dict[str, float | int | str]:
        """Async wrapper for evaluate().

        Default implementation runs the sync method in a thread.
        Subclasses can override for true async behavior.
        """
        return await asyncio.to_thread(self.evaluate, response, question)

    async def evaluate_batch_async(
        self,
        responses: list[str],
        questions: list[str | None] | None = None,
    ) -> list[dict[str, float | int | str]]:
        """Async wrapper for evaluate_batch().

        Default implementation runs the sync method in a thread.
        Subclasses can override for true async behavior.
        """
        return await asyncio.to_thread(self.evaluate_batch, responses, questions)
