"""Abstract base class for evaluation metrics."""

from abc import ABC, abstractmethod


class Metric(ABC):
    """Abstract base class for evaluation metrics."""

    @abstractmethod
    def compute(self, response: str, config: dict) -> float:
        """Compute the metric for a single response.

        Args:
            response: Model response to evaluate.
            config: Metric configuration.

        Returns:
            Metric value (higher = more persona-aligned).
        """
        pass

    @abstractmethod
    def compute_batch(self, responses: list[str], config: dict) -> list[float]:
        """Compute the metric for a batch of responses.

        Args:
            responses: List of model responses.
            config: Metric configuration.

        Returns:
            List of metric values.
        """
        pass

    def aggregate(self, values: list[float]) -> dict[str, float]:
        """Aggregate metric values into summary statistics.

        Args:
            values: List of metric values.

        Returns:
            Dictionary with mean, std, min, max.
        """
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        import statistics

        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
