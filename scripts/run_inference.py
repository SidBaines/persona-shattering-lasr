#!/usr/bin/env python3
"""Run batched inference with a local LLM on a dataset.

Usage:
    cd persona-shattering
    uv run python scripts/run_inference.py configs/toy_model.yaml

This script:
1. Loads the dataset (from cache or HuggingFace)
2. Loads the model specified in config
3. Runs batched generation
4. Saves outputs to scratch/{run_id}/inference_output.jsonl
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# Add scripts/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import Dataset, load_dataset as hf_load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.config import load_config, PipelineConfig
from scripts.utils import read_jsonl, write_jsonl, setup_logging


def load_model(config):
    """Load a HuggingFace model and tokenizer.

    Args:
        config: Model configuration (name, dtype, device_map, etc.)

    Returns:
        Tuple of (model, tokenizer) ready for inference.
    """
    model_config = config.model

    dtype = getattr(torch, model_config.dtype, None)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {model_config.dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_config.name,
        revision=model_config.revision,
        torch_dtype=dtype,
        device_map=model_config.device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.name, revision=model_config.revision, use_fast=True
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


def load_dataset(config) -> Dataset:
    """Load a dataset based on the source config."""
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
    """Format a raw dataset for the inference stage."""
    if question_column is None:
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


def ensure_run_id(config: PipelineConfig) -> str:
    """Generate a run ID if not set."""
    if config.run_id:
        return config.run_id
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}-inference"
    config.run_id = run_id
    return run_id


def run_inference(config: PipelineConfig, dataset: Dataset | None = None) -> Dataset:
    """Run local LLM inference on a question dataset.

    Args:
        config: Pipeline configuration.
        dataset: Optional pre-loaded dataset. If None, loads from config.

    Returns:
        Dataset with added 'response' column.
    """
    logger = setup_logging()
    run_id = ensure_run_id(config)

    if dataset is None:
        dataset = load_dataset(config)
    dataset = format_for_inference(dataset)

    model, tokenizer = load_model(config)
    generation = config.inference.generation

    responses: list[str] = []
    batch_size = max(1, generation.batch_size)
    device = next(model.parameters()).device

    logger.info("Starting inference on %d samples (batch_size=%d).", len(dataset), batch_size)
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        batch_questions = dataset[start:end]["question"]
        inputs = tokenizer(
            batch_questions,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=generation.max_new_tokens,
                temperature=generation.temperature,
                top_p=generation.top_p,
                do_sample=generation.do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_length = int(inputs["input_ids"].shape[1])
        for index, output_ids in enumerate(generated):
            # Use full input length to avoid slicing into the prompt when left-padding.
            completion_ids = output_ids[input_length:]
            responses.append(tokenizer.decode(completion_ids, skip_special_tokens=True))

        logger.info("Processed %d/%d samples", end, len(dataset))

    result = dataset.add_column("response", responses)

    # Save outputs
    save_path = config.inference.output.save_path.format(run_id=run_id)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(result.to_list(), save_path)
    logger.info("Saved inference output to %s", save_path)

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/run_inference.py <config_path>")
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
