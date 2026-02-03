#!/usr/bin/env python3
"""Load a dataset from HuggingFace or local JSONL and cache it.

Usage:
    cd persona-shattering
    uv run python scripts/load_data.py configs/toy_model.yaml

This script:
1. Loads the dataset specified in the config
2. Applies max_samples limit if set
3. Saves to datasets/ for caching
4. Prints sample records for verification
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add scripts/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset, load_dataset as hf_load_dataset
from dotenv import load_dotenv

from scripts.config import load_config
from scripts.utils import read_jsonl, write_jsonl, setup_logging


def load_dataset(config) -> Dataset:
    """Load a dataset based on the source config.

    Args:
        config: Dataset source configuration specifying HF name or local path.

    Returns:
        A HuggingFace Dataset object.
    """
    dataset_config = config.inference.dataset

    if dataset_config.source == "huggingface":
        if not dataset_config.name:
            raise ValueError("HuggingFace source requires dataset name.")
        dataset = hf_load_dataset(dataset_config.name, split=dataset_config.split)
    elif dataset_config.source == "local":
        if not dataset_config.path:
            raise ValueError("Local source requires dataset path.")
        records = read_jsonl(Path(dataset_config.path))
        dataset = Dataset.from_list(records)
    else:
        raise ValueError(f"Unsupported dataset source: {dataset_config.source}")

    if dataset_config.max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), dataset_config.max_samples)))

    return dataset


def format_for_inference(dataset: Dataset, question_column: str | None = None) -> Dataset:
    """Format a raw dataset for the inference stage.

    Args:
        dataset: The dataset to format.
        question_column: Name of the column containing questions. If None, will try
            common column names: "question", "instruction", "prompt", "text".

    Returns:
        Dataset with a single "question" column.
    """
    if question_column is None:
        # Try common column names
        common_names = ["question", "instruction", "prompt", "text"]
        for name in common_names:
            if name in dataset.column_names:
                question_column = name
                break
        if question_column is None:
            raise ValueError(
                f"Could not find question column. Available columns: {dataset.column_names}. "
                f"Expected one of: {common_names}"
            )

    if question_column not in dataset.column_names:
        raise ValueError(f"Question column '{question_column}' not found in dataset.")

    if question_column != "question":
        dataset = dataset.rename_column(question_column, "question")

    extra_columns = [col for col in dataset.column_names if col != "question"]
    if extra_columns:
        dataset = dataset.remove_columns(extra_columns)

    return dataset


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/load_data.py <config_path>")
        sys.exit(1)

    load_dotenv()
    logger = setup_logging()

    config_path = sys.argv[1]
    config = load_config(config_path)

    logger.info("Loading dataset from config: %s", config_path)
    logger.info("Dataset source: %s", config.inference.dataset.source)
    logger.info("Dataset name: %s", config.inference.dataset.name)
    logger.info("Max samples: %s", config.inference.dataset.max_samples)

    # Load the raw dataset
    dataset = load_dataset(config)
    logger.info("Loaded %d samples", len(dataset))
    logger.info("Columns: %s", dataset.column_names)

    # Format for inference (extract question column)
    dataset = format_for_inference(dataset)
    logger.info("Formatted dataset has %d samples with columns: %s", len(dataset), dataset.column_names)

    # Save to datasets/ for caching
    cache_dir = Path(config.paths.data_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate a cache filename from the dataset config
    dataset_name = config.inference.dataset.name or "local"
    safe_name = dataset_name.replace("/", "_").replace("-", "_")
    max_samples = config.inference.dataset.max_samples or "all"
    cache_path = cache_dir / f"{safe_name}_{max_samples}_samples.jsonl"

    write_jsonl(dataset.to_list(), cache_path)
    logger.info("Saved dataset to %s", cache_path)

    # Print some samples for verification
    print("\n" + "=" * 60)
    print("SAMPLE RECORDS")
    print("=" * 60)
    for i, record in enumerate(dataset.select(range(min(3, len(dataset))))):
        print(f"\n--- Sample {i+1} ---")
        question = record["question"]
        if len(question) > 200:
            question = question[:200] + "..."
        print(f"Question: {question}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
