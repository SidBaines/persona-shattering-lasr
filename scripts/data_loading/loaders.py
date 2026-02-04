"""Dataset loading and formatting utilities."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_dataset as hf_load_dataset

from scripts.utils import read_jsonl


def load_dataset(config) -> Dataset:
    """Load a dataset based on the source config.

    Args:
        config: Pipeline configuration with inference.dataset settings.

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
