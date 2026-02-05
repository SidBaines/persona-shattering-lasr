"""Core inference logic for running LLM inference on datasets."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import Dataset

from scripts.config import PipelineConfig
from scripts.data_loading import load_dataset, format_for_inference
from scripts.inference.model import load_model
from scripts.utils import write_jsonl, setup_logging


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

        attention_mask = inputs["attention_mask"]
        for index, output_ids in enumerate(generated):
            prompt_len = int(attention_mask[index].sum().item())
            completion_ids = output_ids[prompt_len:]
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
