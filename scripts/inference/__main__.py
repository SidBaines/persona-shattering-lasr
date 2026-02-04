#!/usr/bin/env python3
"""Run batched inference with a local LLM on a dataset.

Usage:
    cd persona-shattering
    python -m scripts.inference configs/toy_model.yaml

This script:
1. Loads the dataset (from cache or HuggingFace)
2. Loads the model specified in config
3. Runs batched generation
4. Saves outputs to scratch/{run_id}/inference_output.jsonl
"""

import sys

from dotenv import load_dotenv

from scripts.config import load_config
from scripts.inference.run import run_inference
from scripts.utils import setup_logging


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.inference <config_path>")
        sys.exit(1)

    load_dotenv()
    logger = setup_logging()

    config_path = sys.argv[1]
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
