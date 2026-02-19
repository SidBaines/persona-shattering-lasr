"""Core inference logic for running LLM inference on datasets."""

from __future__ import annotations

import asyncio
import gc
import json
from pathlib import Path

from datasets import Dataset

from scripts.data_loading import load_dataset_from_config, format_for_inference
from scripts.inference.config import InferenceConfig, InferenceResult
from scripts.inference.providers import get_provider
from scripts.inference.providers.base import accumulate_usage, empty_usage
from scripts.utils import count_jsonl_rows, read_jsonl, setup_logging


async def run_inference_async(
    config: InferenceConfig, dataset: Dataset | None = None
) -> tuple[Dataset, InferenceResult]:
    """Run LLM inference on a question dataset asynchronously.

    Uses the provider specified in config.provider:
    - "local": HuggingFace transformers (default)
    - "openai": OpenAI API
    - "openrouter": OpenRouter API (OpenAI-compatible)
    - "anthropic": Anthropic API

    Args:
        config: Inference configuration.
        dataset: Optional pre-loaded dataset. If None, loads from config.dataset.

    Returns:
        Tuple of (dataset with 'response' column, InferenceResult metadata).
    """
    logger = setup_logging()

    if dataset is None:
        dataset = load_dataset_from_config(config.dataset)
    dataset = format_for_inference(dataset)

    # Batch mode for OpenAI provider
    if config.provider == "openai" and config.openai.batch.enabled:
        from scripts.inference.openai_batch import run_openai_batch_inference

        logger.info("Using OpenAI Batch API for inference.")
        logger.info("Model: %s", config.model)
        return await asyncio.to_thread(run_openai_batch_inference, config, dataset)

    # Get the inference provider
    logger.info("Using inference provider: %s", config.provider)
    logger.info("Model: %s", config.model)
    provider = get_provider(config.provider, config)

    generation = config.generation
    in_memory_records: list[dict[str, str | int]] = []
    total_usage = empty_usage()
    failed_count = 0
    batch_size = max(1, generation.batch_size)
    num_responses = max(1, generation.num_responses_per_prompt)
    total_expected_rows = len(dataset) * num_responses
    save_path = Path(config.output_path) if config.output_path else None

    completed_rows = 0
    start_prompt_index = 0
    start_response_offset = 0
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if config.overwrite_output:
            save_path.write_text("")
        elif config.resume and save_path.exists():
            completed_rows = count_jsonl_rows(save_path)
            if completed_rows > total_expected_rows:
                raise ValueError(
                    f"Output already has {completed_rows} rows, but only "
                    f"{total_expected_rows} are expected for this run."
                )
            start_prompt_index = completed_rows // num_responses
            start_response_offset = completed_rows % num_responses
            if completed_rows:
                logger.info(
                    "Resuming inference from row %d/%d (%d prompts fully done, response offset %d).",
                    completed_rows,
                    total_expected_rows,
                    start_prompt_index,
                    start_response_offset,
                )
        else:
            save_path.write_text("")

    if num_responses > 1 and not generation.do_sample:
        raise ValueError(
            "num_responses_per_prompt > 1 requires do_sample=True. "
            "Set generation.do_sample to True."
        )

    logger.info(
        "Starting inference on %d samples (batch_size=%d, responses_per_prompt=%d).",
        len(dataset),
        batch_size,
        num_responses,
    )

    if completed_rows >= total_expected_rows:
        logger.info("Output already complete at %s. Skipping generation.", save_path)
    else:
        write_handle = (
            save_path.open("a", encoding="utf-8")
            if save_path is not None
            else None
        )
        try:
            for start in range(start_prompt_index, len(dataset), batch_size):
                end = min(start + batch_size, len(dataset))
                batch_questions = dataset[start:end]["question"]
                (
                    batch_responses,
                    batch_usage,
                    batch_failed,
                ) = await provider.generate_batch_with_metadata_async(
                    batch_questions, num_responses=num_responses
                )
                if len(batch_responses) != len(batch_questions) * num_responses:
                    raise ValueError(
                        "Provider returned unexpected number of responses. "
                        f"Expected {len(batch_questions) * num_responses}, got {len(batch_responses)}."
                    )
                accumulate_usage(total_usage, batch_usage)
                failed_count += batch_failed

                for question_index, question in enumerate(batch_questions):
                    prompt_index = start + question_index
                    first_response_index = 0
                    if (
                        prompt_index == start_prompt_index
                        and start_response_offset > 0
                    ):
                        first_response_index = start_response_offset

                    for response_index in range(first_response_index, num_responses):
                        output_index = question_index * num_responses + response_index
                        record = {
                            "question": question,
                            "response": batch_responses[output_index],
                            "response_index": response_index,
                        }
                        if write_handle is not None:
                            write_handle.write(json.dumps(record) + "\n")
                        else:
                            in_memory_records.append(record)

                    if (
                        prompt_index == start_prompt_index
                        and start_response_offset > 0
                    ):
                        start_response_offset = 0
                if write_handle is not None:
                    write_handle.flush()
                logger.info("Processed %d/%d samples", end, len(dataset))
        finally:
            if write_handle is not None:
                write_handle.close()

    # Release GPU memory held by the provider (model weights, KV cache, etc.)
    del provider
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    if save_path is not None:
        result_records = read_jsonl(save_path) if save_path.exists() else []
    else:
        result_records = in_memory_records
    result_dataset = Dataset.from_list(result_records)

    if failed_count:
        logger.warning("Inference completed with %d failed responses.", failed_count)

    # Save outputs if path specified
    result = InferenceResult(
        num_samples=len(result_dataset),
        num_failed=failed_count,
        total_input_tokens=total_usage["input_tokens"],
        total_output_tokens=total_usage["output_tokens"],
        total_tokens=total_usage["total_tokens"],
    )
    if save_path is not None:
        logger.info("Saved inference output to %s", save_path)
        result.output_path = save_path

    return result_dataset, result


def run_inference(
    config: InferenceConfig, dataset: Dataset | None = None
) -> tuple[Dataset, InferenceResult]:
    """Run LLM inference on a question dataset (sync wrapper).

    Use run_inference_async for async contexts. This wrapper will fail if
    called while an event loop is already running.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_inference_async(config, dataset))
    raise RuntimeError(
        "run_inference called inside a running event loop. "
        "Use run_inference_async instead."
    )
