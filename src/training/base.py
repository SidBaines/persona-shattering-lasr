"""Abstract base class for trainers."""

from abc import ABC, abstractmethod
from pathlib import Path


class Trainer(ABC):
    """Abstract base class for model trainers."""

    @abstractmethod
    def train(
        self,
        base_model: str,
        train_data: list[dict],
        output_dir: Path,
        config: dict,
    ) -> Path:
        """Fine-tune a model on the training data.

        Args:
            base_model: Base model name/path.
            train_data: List of training samples with:
                - instruction: Task instruction
                - input: Optional input context
                - output: Target output (edited response)
            output_dir: Directory to save the trained model/adapter.
            config: Training configuration containing:
                - lora_rank: LoRA rank (for LoRA training)
                - lora_alpha: LoRA alpha
                - learning_rate: Learning rate
                - epochs: Number of training epochs
                - batch_size: Training batch size

        Returns:
            Path to the saved model/adapter.
        """
        pass

    @abstractmethod
    def load_trained(self, base_model: str, adapter_path: Path) -> object:
        """Load a trained model/adapter for inference.

        Args:
            base_model: Base model name/path.
            adapter_path: Path to the trained adapter.

        Returns:
            Model ready for inference.
        """
        pass
