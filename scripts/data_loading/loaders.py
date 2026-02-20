"""Dataset loading and formatting utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import Dataset, load_dataset as hf_load_dataset

if TYPE_CHECKING:
    from scripts.common.config import DatasetConfig


def load_dataset_from_config(config: "DatasetConfig") -> Dataset:
    """Load a dataset based on the DatasetConfig.

    Args:
        config: Dataset configuration.

    Returns:
        A HuggingFace Dataset object.
    """
    if config.source == "huggingface":
        if not config.name:
            raise ValueError("HuggingFace source requires dataset name.")
        dataset = hf_load_dataset(config.name, split=config.split)
    elif config.source == "local":
        if not config.path:
            raise ValueError("Local source requires dataset path.")
        dataset = hf_load_dataset(
            "json",
            data_files=str(Path(config.path)),
            split="train",
        )
    elif config.source == "canonical":
        if not config.path:
            raise ValueError("Canonical source requires run directory path.")
        run_path = Path(config.path)
        if run_path.is_file():
            rows = []
            with run_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if text:
                        rows.append(json.loads(text))
        else:
            from scripts.datasets import load_samples, materialize_canonical_samples

            materialize_canonical_samples(run_path)
            samples = load_samples(run_path)
            rows = []
            target_variant = None
            if isinstance(config.name, str) and config.name.startswith("editing:"):
                target_variant = config.name.split(":", 1)[1]

            for sample in samples:
                question = next(
                    (msg.content for msg in sample.messages if msg.role == "user"),
                    "",
                )
                response = sample.inference.assistant_completion or ""
                if target_variant:
                    variant = next(
                        (v for v in sample.edit_variants if v.variant_name == target_variant),
                        None,
                    )
                    if variant is None:
                        continue
                    successful = [o for o in variant.overlays if o.status == "success"]
                    if not successful:
                        continue
                    latest = sorted(
                        successful, key=lambda item: (item.attempt_no, item.overlay_id)
                    )[-1]
                    response = latest.edited_content

                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "input_group_id": sample.input_group_id or sample.sample_id,
                        "response_index": sample.response_index,
                        "messages": [m.model_dump() for m in sample.messages],
                        "question": question,
                        "response": response,
                    }
                )
        dataset = Dataset.from_list(rows)
    else:
        raise ValueError(f"Unsupported dataset source: {config.source}")

    if config.max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), config.max_samples)))

    return dataset


def load_dataset(config) -> Dataset:
    """Load a dataset based on the source config.

    Args:
        config: Pipeline configuration with inference.dataset settings,
                or a DatasetConfig directly.

    Returns:
        A HuggingFace Dataset object.
    """
    # Handle both old PipelineConfig style and new DatasetConfig
    if hasattr(config, "inference"):
        dataset_config = config.inference.dataset
    elif hasattr(config, "dataset"):
        dataset_config = config.dataset
    else:
        # Assume it's already a DatasetConfig
        dataset_config = config

    return load_dataset_from_config(dataset_config)


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

    # If the dataset has an auxiliary input field, append it to the question and
    # persist that merged text as the canonical "question" value written to JSONL.
    # This preserves instruction+input style datasets (e.g., Alpaca) without
    # requiring callers to special-case formatting.
    auxiliary_column = "input" if "input" in dataset.column_names else None

    if auxiliary_column is not None:
        def _merge_question_input(example: dict) -> dict:
            question_raw = example.get(question_column, "")
            extra_raw = example.get(auxiliary_column, "")
            question = question_raw if isinstance(question_raw, str) else str(question_raw)
            extra = extra_raw if isinstance(extra_raw, str) else str(extra_raw)

            if extra.strip():
                merged = f"{question}\n\n{extra}" if question else extra
            else:
                merged = question
            return {"question": merged}

        dataset = dataset.map(_merge_question_input)
    elif question_column != "question":
        dataset = dataset.rename_column(question_column, "question")

    extra_columns = [col for col in dataset.column_names if col != "question"]
    if extra_columns:
        dataset = dataset.remove_columns(extra_columns)

    return dataset
