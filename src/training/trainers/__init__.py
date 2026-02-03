"""Trainer implementations."""

from ..base import Trainer
from .local_lora import LocalLoRATrainer
from .tinker import TinkerTrainer

TRAINERS: dict[str, type[Trainer]] = {
    "local_lora": LocalLoRATrainer,
    "tinker": TinkerTrainer,
}


def get_trainer(trainer_type: str) -> Trainer:
    """Get a trainer by type.

    Args:
        trainer_type: Type of trainer (e.g., "local_lora", "tinker").

    Returns:
        An instance of the requested trainer.

    Raises:
        KeyError: If trainer_type is not registered.
    """
    if trainer_type not in TRAINERS:
        raise KeyError(f"Unknown trainer type: {trainer_type}. Available: {list(TRAINERS.keys())}")
    return TRAINERS[trainer_type]()
