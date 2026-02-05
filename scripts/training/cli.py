"""CLI entry point for scripts-based training stage."""

from __future__ import annotations

import click
from dotenv import load_dotenv

from scripts.config import load_config
from scripts.run_training import run_training
from scripts.utils import setup_logging


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def main(config_path: str) -> None:
    """Run the training stage using scripts implementations."""
    load_dotenv()
    logger = setup_logging()

    config = load_config(config_path)

    logger.info("Running training with config: %s", config_path)
    logger.info("Model: %s", config.model.name)
    logger.info("Run ID: %s", config.run_id)
    logger.info("LoRA r=%d, alpha=%d", config.training.lora.r, config.training.lora.lora_alpha)
    logger.info("Epochs: %d", config.training.sft.num_train_epochs)
    logger.info("W&B enabled: %s", config.wandb.enabled)

    final_path = run_training(config)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final model saved to: {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
