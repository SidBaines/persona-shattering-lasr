"""Local LoRA fine-tuning trainer."""

from pathlib import Path

from ..base import Trainer


class LocalLoRATrainer(Trainer):
    """Fine-tune models locally using LoRA."""

    def train(
        self,
        base_model: str,
        train_data: list[dict],
        output_dir: Path,
        config: dict,
    ) -> Path:
        """Fine-tune using LoRA.

        Args:
            base_model: HuggingFace model name.
            train_data: Training samples.
            output_dir: Output directory for adapter.
            config: Training config with lora_rank, lora_alpha, etc.

        Returns:
            Path to saved LoRA adapter.
        """
        raise NotImplementedError(
            "LocalLoRATrainer not yet implemented. "
            "Implement in scripts/train_lora.py first, then migrate here."
        )

    def load_trained(self, base_model: str, adapter_path: Path) -> object:
        """Load base model with LoRA adapter."""
        raise NotImplementedError(
            "LocalLoRATrainer not yet implemented. "
            "Implement in scripts/train_lora.py first, then migrate here."
        )
