"""Model loading utilities for inference."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
