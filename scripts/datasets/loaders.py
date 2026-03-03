"""Dataset loading and formatting utilities."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from datasets import Dataset, load_dataset as hf_load_dataset

from scripts.datasets.core import load_samples, materialize_canonical_samples

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
        dataset = hf_load_dataset(config.name, config.subset, split=config.split)
    elif config.source == "local":
        if not config.path:
            raise ValueError("Local source requires dataset path.")
        dataset = hf_load_dataset(
            "json",
            data_files=str(Path(config.path)),
            split="train",
        )
    elif config.source == "oasst1":
        dataset = hf_load_dataset("OpenAssistant/oasst1", split=config.split)
        dataset = dataset.filter(lambda x: x["parent_id"] is None and x["role"] == "prompter")
        dataset = dataset.rename_column("text", "question")
        dataset = dataset.remove_columns([c for c in dataset.column_names if c != "question"])
    elif config.source == "mt_bench":
        if not config.path:
            raise ValueError("mt_bench source requires a dataset path.")
        raw = hf_load_dataset("json", data_files=str(Path(config.path)), split="train")
        dataset = raw.map(
            lambda ex: {"question": ex["turns"][0]},
            remove_columns=[c for c in raw.column_names if c not in ("question_id", "category")],
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

    if config.seed is not None:
        dataset = dataset.shuffle(seed=config.seed)

    if config.max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), config.max_samples)))

    return dataset


def _get_nested_value(record: Mapping[str, Any], path: str) -> Any:
    """Resolve a dotted path from a dataset record."""
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _render_messages_as_question(raw_messages: list[Any]) -> str | None:
    """Convert a chat transcript into a single prompt string ending on a user turn."""
    parsed_messages: list[tuple[str, str]] = []
    for raw in raw_messages:
        if not isinstance(raw, dict):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        parsed_messages.append((role, content))

    if not parsed_messages:
        return None

    last_user_index = max(
        (idx for idx, (role, _) in enumerate(parsed_messages) if role == "user"),
        default=-1,
    )
    if last_user_index < 0:
        return None

    prompt_messages = parsed_messages[: last_user_index + 1]
    user_messages = [content for role, content in prompt_messages if role == "user"]
    if len(prompt_messages) == 1 and len(user_messages) == 1:
        return user_messages[0]

    rendered_parts = [f"{role.title()}:\n{content}" for role, content in prompt_messages]
    return "\n\n".join(rendered_parts)


def _extract_question_value(record: Mapping[str, Any], question_column: str) -> str | None:
    """Extract a string question from a record using a top-level or dotted field path."""
    if question_column in record:
        value = record[question_column]
    else:
        value = _get_nested_value(record, question_column)

    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return _render_messages_as_question(value)
    if isinstance(value, Mapping):
        for key in ("revised_query", "question", "prompt", "text", "original_query"):
            nested = value.get(key)
            if isinstance(nested, str):
                return nested
    return None


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
        common_names = [
            "question",
            "instruction",
            "prompt",
            "text",
            "lm_judge_annotation.revised_query",
            "messages",
        ]
        for name in common_names:
            if name in dataset.column_names:
                question_column = name
                break
            if "." in name and name.split(".", 1)[0] in dataset.column_names:
                probe = _extract_question_value(dataset[0], name) if len(dataset) else None
                if isinstance(probe, str):
                    question_column = name
                    break
        if question_column is None:
            raise ValueError(
                f"Could not find question column. Available columns: {dataset.column_names}. "
                f"Expected one of: {common_names}"
            )

    if question_column not in dataset.column_names and (
        "." not in question_column or question_column.split(".", 1)[0] not in dataset.column_names
    ):
        raise ValueError(f"Question column '{question_column}' not found in dataset.")

    # If the dataset has an auxiliary input field, append it to the question and
    # persist that merged text as the canonical "question" value written to JSONL.
    # This preserves instruction+input style datasets (e.g., Alpaca) without
    # requiring callers to special-case formatting.
    auxiliary_column = "input" if "input" in dataset.column_names else None

    def _merge_question_input(example: dict) -> dict:
        extracted = _extract_question_value(example, question_column)
        if extracted is None:
            raise ValueError(
                f"Question column '{question_column}' did not resolve to a usable string."
            )
        extra_raw = example.get(auxiliary_column, "") if auxiliary_column is not None else ""
        extra = extra_raw if isinstance(extra_raw, str) else str(extra_raw)

        if extra.strip():
            merged = f"{extracted}\n\n{extra}" if extracted else extra
        else:
            merged = extracted
        return {"question": merged}

    if question_column != "question" or auxiliary_column is not None:
        dataset = dataset.map(_merge_question_input)

    extra_columns = [col for col in dataset.column_names if col != "question"]
    if extra_columns:
        dataset = dataset.remove_columns(extra_columns)

    return dataset
