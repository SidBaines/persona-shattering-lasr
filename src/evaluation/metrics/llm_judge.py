"""LLM-as-judge metrics."""

from ..base import Metric


class LLMJudgeMetric(Metric):
    """Use an LLM to judge persona alignment."""

    def __init__(self):
        self.client = None

    def compute(self, response: str, config: dict) -> float:
        """Judge persona alignment using an LLM.

        Args:
            response: Model response.
            config: Metric configuration containing:
                - provider: API provider (anthropic, openai)
                - model: Judge model name
                - prompt_template: Judging prompt with {response} placeholder
                - scale: Rating scale (e.g., 1-5)

        Returns:
            Persona alignment score.
        """
        raise NotImplementedError(
            "LLMJudgeMetric not yet implemented. "
            "Implement in scripts/ first, then migrate here."
        )

    def compute_batch(self, responses: list[str], config: dict) -> list[float]:
        """Judge a batch of responses."""
        raise NotImplementedError(
            "LLMJudgeMetric not yet implemented. "
            "Implement in scripts/ first, then migrate here."
        )
