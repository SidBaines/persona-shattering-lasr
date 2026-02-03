"""HuggingFace datasets loader."""

from ..base import DatasetLoader


class HuggingFaceLoader(DatasetLoader):
    """Load datasets from HuggingFace Hub."""

    def load(self, config: dict) -> list[dict]:
        """Load a dataset from HuggingFace Hub.

        Args:
            config: Dataset configuration containing:
                - name: HuggingFace dataset name (e.g., "vicgalle/alpaca-gpt4")
                - split: Dataset split (default: "train")
                - max_samples: Maximum samples to load (default: None = all)
                - cache_dir: Cache directory (default: "datasets/")

        Returns:
            List of sample dictionaries.
        """
        raise NotImplementedError(
            "HuggingFaceLoader not yet implemented. "
            "Implement in scripts/load_data.py first, then migrate here."
        )
