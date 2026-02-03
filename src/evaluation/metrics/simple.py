"""Simple code-based metrics."""

from ..base import Metric


class CharCountMetric(Metric):
    """Count occurrences of a character in responses."""

    def compute(self, response: str, config: dict) -> float:
        """Count character occurrences.

        Args:
            response: Model response.
            config: Metric configuration containing:
                - char: Character to count
                - normalize: If True, return ratio (count / total chars)

        Returns:
            Count or ratio of character occurrences.
        """
        raise NotImplementedError(
            "CharCountMetric not yet implemented. "
            "Implement in scripts/evaluate.py first, then migrate here."
        )

    def compute_batch(self, responses: list[str], config: dict) -> list[float]:
        """Compute metric for a batch of responses."""
        raise NotImplementedError(
            "CharCountMetric not yet implemented. "
            "Implement in scripts/evaluate.py first, then migrate here."
        )
