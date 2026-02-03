"""Abstract base class for dataset loaders."""

from abc import ABC, abstractmethod


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    @abstractmethod
    def load(self, config: dict) -> list[dict]:
        """Load dataset and return list of samples.

        Args:
            config: Dataset configuration dictionary containing:
                - name: Dataset name/path
                - split: Dataset split (train/val/test)
                - max_samples: Maximum number of samples to load
                - cache_dir: Directory to cache downloaded data

        Returns:
            List of sample dictionaries with at least:
                - instruction: The task instruction
                - input: Optional input context
                - output: Expected output (if available)
        """
        pass
