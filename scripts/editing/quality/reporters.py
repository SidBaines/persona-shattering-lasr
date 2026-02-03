"""Quality metric reporters for different output destinations.

Reporters handle how metrics are stored and logged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class QualityReporter(ABC):
    """Abstract base class for quality metric reporters."""

    @abstractmethod
    def report_record(
        self, record: dict, metrics: dict[str, float | int], metrics_key: str
    ) -> dict:
        """Report metrics for a single record.

        Args:
            record: The record dict to potentially modify.
            metrics: Computed metrics for this record.
            metrics_key: Key under which to store metrics in the record.

        Returns:
            The record dict (possibly modified).
        """
        ...

    @abstractmethod
    def report_summary(self, aggregates: dict[str, float]) -> None:
        """Report aggregated metrics summary.

        Args:
            aggregates: Aggregated statistics across all records.
        """
        ...


class JsonReporter(QualityReporter):
    """Adds metrics inline to record dicts under the configured key."""

    def report_record(
        self, record: dict, metrics: dict[str, float | int], metrics_key: str
    ) -> dict:
        """Add metrics to the record dict.

        Args:
            record: The record dict to modify.
            metrics: Computed metrics for this record.
            metrics_key: Key under which to store metrics.

        Returns:
            The modified record with metrics added.
        """
        record[metrics_key] = metrics
        return record

    def report_summary(self, aggregates: dict[str, float]) -> None:
        """No-op for JSON reporter (aggregates are handled elsewhere)."""
        pass


class ConsoleReporter(QualityReporter):
    """Prints summary metrics to console."""

    def report_record(
        self, record: dict, metrics: dict[str, float | int], metrics_key: str
    ) -> dict:
        """No-op for console reporter (per-record metrics not printed)."""
        return record

    def report_summary(self, aggregates: dict[str, float]) -> None:
        """Print aggregated metrics to console.

        Args:
            aggregates: Aggregated statistics to print.
        """
        print("\n" + "=" * 60)
        print("QUALITY METRICS SUMMARY")
        print("=" * 60)
        for key, value in sorted(aggregates.items()):
            print(f"  {key}: {value:.2f}")
        print("=" * 60)


# Global registry mapping reporter names to their classes
REPORTER_REGISTRY: dict[str, type[QualityReporter]] = {
    "json": JsonReporter,
    "console": ConsoleReporter,
}


def get_reporters(names: list[str]) -> list[QualityReporter]:
    """Get reporter instances by name.

    Args:
        names: List of reporter names.

    Returns:
        List of instantiated reporters.

    Raises:
        KeyError: If any reporter name is not registered.
    """
    reporters = []
    for name in names:
        if name not in REPORTER_REGISTRY:
            available = ", ".join(sorted(REPORTER_REGISTRY.keys()))
            raise KeyError(f"Unknown reporter '{name}'. Available reporters: {available}")
        reporters.append(REPORTER_REGISTRY[name]())
    return reporters
