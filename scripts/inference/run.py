"""Core inference logic for running LLM inference on datasets."""

from __future__ import annotations

import asyncio
from pathlib import Path

from datasets import Dataset

from scripts.data_loading import load_dataset_from_config, format_for_inference
from scripts.inference.config import InferenceConfig, InferenceResult
from scripts.inference.providers import get_provider
from scripts.utils import write_jsonl, setup_logging


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
    responses: list[str] = []
    questions_out: list[str] = []
    response_indices: list[int] = []
    batch_size = max(1, generation.batch_size)
    num_responses = max(1, generation.num_responses_per_prompt)

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
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        batch_questions = dataset[start:end]["question"]
        batch_responses = await provider.generate_batch_async(
            batch_questions, num_responses=num_responses
        )
        if len(batch_responses) != len(batch_questions) * num_responses:
            raise ValueError(
                "Provider returned unexpected number of responses. "
                f"Expected {len(batch_questions) * num_responses}, got {len(batch_responses)}."
            )
        for question_index, question in enumerate(batch_questions):
            for response_index in range(num_responses):
                output_index = question_index * num_responses + response_index
                responses.append(batch_responses[output_index])
                questions_out.append(question)
                response_indices.append(response_index)
        logger.info("Processed %d/%d samples", end, len(dataset))

    result_dataset = Dataset.from_list(
        [
            {
                "question": question,
                "response": response,
                "response_index": response_index,
            }
            for question, response, response_index in zip(
                questions_out, responses, response_indices
            )
        ]
    )

    # Save outputs if path specified
    result = InferenceResult(num_samples=len(result_dataset))
    if config.output_path:
        save_path = Path(config.output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(result_dataset.to_list(), save_path)
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
