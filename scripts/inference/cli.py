"""CLI entry point for scripts-based inference stage."""

from __future__ import annotations

import click
from dotenv import load_dotenv

from scripts.config import load_config
from scripts.inference.run import run_inference
from scripts.utils import setup_logging


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def main(config_path: str) -> None:
    """Run the inference stage using scripts implementations."""
    load_dotenv()
    logger = setup_logging()

    config = load_config(config_path)

    logger.info("Running inference with config: %s", config_path)
    logger.info("Model: %s", config.model.name)
    logger.info("Dataset: %s", config.inference.dataset.name)
    logger.info("Max samples: %s", config.inference.dataset.max_samples)

    result = run_inference(config)

    # Print some sample outputs
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUTS")
    print("=" * 60)
    for i, record in enumerate(result.select(range(min(3, len(result))))):
        print(f"\n--- Sample {i+1} ---")
        question = record["question"]
        response = record["response"]
        if len(question) > 100:
            question = question[:100] + "..."
        if len(response) > 200:
            response = response[:200] + "..."
        print(f"Question: {question}")
        print(f"Response: {response}")
    print("\n" + "=" * 60)
    print(f"\nTotal samples processed: {len(result)}")


if __name__ == "__main__":
    main()
