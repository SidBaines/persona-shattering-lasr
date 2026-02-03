"""Tinker API trainer for cloud-based fine-tuning."""

from pathlib import Path

from ..base import Trainer


class TinkerTrainer(Trainer):
    """Fine-tune models via Tinker API."""

    def train(
        self,
        base_model: str,
        train_data: list[dict],
        output_dir: Path,
        config: dict,
    ) -> Path:
        """Submit fine-tuning job to Tinker API.

        Args:
            base_model: Model name supported by Tinker.
            train_data: Training samples.
            output_dir: Local directory to save job metadata.
            config: Training config with Tinker-specific options.

        Returns:
            Path to job metadata file.
        """
        raise NotImplementedError(
            "TinkerTrainer not yet implemented. "
            "Implement in scripts/ first, then migrate here."
        )

    def load_trained(self, base_model: str, adapter_path: Path) -> object:
        """Load trained model from Tinker.

        Note: This may return an API client rather than a local model.
        """
        raise NotImplementedError(
            "TinkerTrainer not yet implemented. "
            "Implement in scripts/ first, then migrate here."
        )
