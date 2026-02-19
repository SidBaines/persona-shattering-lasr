"""Merge one or more LoRA adapters (with arbitrary scaling) into a standalone model.

Uses ``LoRaScaling`` and ``LoRaPipeline`` from
``src.utils.peft_manipulations`` to apply per-adapter scaling, then
``merge_and_unload()`` to bake the result into base weights.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.evals.config import AdapterConfig
from src.utils.peft_manipulations import (
    LoRaPipeline,
    LoRaScaling,
    set_active_adapters,
)

logger = logging.getLogger(__name__)


def _resolve_torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported dtype: {name}")
    return dtype


def _load_tokenizer(base_model: str, adapter_path: str) -> AutoTokenizer:
    """Load tokenizer from adapter dir if available, else from base model."""
    adapter_dir = Path(adapter_path)
    if (adapter_dir / "tokenizer_config.json").exists():
        return AutoTokenizer.from_pretrained(str(adapter_dir), use_fast=True)
    return AutoTokenizer.from_pretrained(base_model, use_fast=True)


def _split_adapter_ref(path: str) -> tuple[str, str | None]:
    """Parse adapter reference.

    Supports:
    - "repo_or_path" (default behavior)
    - "repo_or_path::subfolder" (explicit subfolder)
    """
    if "::" in path:
        ref, subfolder = path.split("::", 1)
        return ref, (subfolder or None)
    return path, None


def _load_first_adapter(
    model,
    *,
    adapter_path: str,
    adapter_name: str,
) -> PeftModel:
    ref, subfolder = _split_adapter_ref(adapter_path)

    if subfolder:
        return PeftModel.from_pretrained(
            model,
            ref,
            adapter_name=adapter_name,
            subfolder=subfolder,
        )

    try:
        return PeftModel.from_pretrained(
            model,
            ref,
            adapter_name=adapter_name,
        )
    except ValueError as exc:
        # Common HF layout for adapter artifacts is files under "adapter/".
        if "adapter_config.json" not in str(exc):
            raise
        return PeftModel.from_pretrained(
            model,
            ref,
            adapter_name=adapter_name,
            subfolder="adapter",
        )


def _load_extra_adapter(
    model: PeftModel,
    *,
    adapter_path: str,
    adapter_name: str,
) -> None:
    ref, subfolder = _split_adapter_ref(adapter_path)

    if subfolder:
        model.load_adapter(
            ref,
            adapter_name=adapter_name,
            subfolder=subfolder,
        )
        return

    try:
        model.load_adapter(ref, adapter_name=adapter_name)
    except ValueError as exc:
        if "adapter_config.json" not in str(exc):
            raise
        model.load_adapter(
            ref,
            adapter_name=adapter_name,
            subfolder="adapter",
        )


def merge_adapters(
    *,
    base_model: str,
    adapters: list[AdapterConfig],
    output_dir: Path,
    dtype: str = "bfloat16",
    device_map: str = "auto",
) -> Path:
    """Load base model + adapters, apply per-adapter scaling, merge, and save.

    Parameters
    ----------
    base_model:
        HuggingFace model name or local path for the base model.
    adapters:
        One or more adapters, each with a path and scaling factor.
    output_dir:
        Directory to save the merged model and tokenizer.
    dtype:
        Torch dtype name (e.g. ``"bfloat16"``).
    device_map:
        Device map for model loading (default ``"auto"``).

    Returns
    -------
    Path
        The *output_dir* (for convenience in chaining).
    """
    if not adapters:
        raise ValueError("At least one adapter is required.")

    torch_dtype = _resolve_torch_dtype(dtype)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Loading base model %s (dtype=%s, device_map=%s)",
        base_model, dtype, device_map,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch_dtype,
        device_map=device_map,
    )

    # Load first adapter
    adapter_names = []
    name_0 = "adapter_0"
    logger.info("Loading adapter 0: %s (scale=%.3f)", adapters[0].path, adapters[0].scale)
    model = _load_first_adapter(
        model,
        adapter_path=adapters[0].path,
        adapter_name=name_0,
    )
    adapter_names.append(name_0)

    # Load additional adapters
    for i, adapter_cfg in enumerate(adapters[1:], 1):
        name_i = f"adapter_{i}"
        logger.info("Loading adapter %d: %s (scale=%.3f)", i, adapter_cfg.path, adapter_cfg.scale)
        _load_extra_adapter(
            model,
            adapter_path=adapter_cfg.path,
            adapter_name=name_i,
        )
        adapter_names.append(name_i)

    # Activate all adapters for multi-adapter inference
    set_active_adapters(model, adapter_names)

    # Build and apply scaling pipeline
    steps = [
        (LoRaScaling, name, {"scale_factor": cfg.scale})
        for name, cfg in zip(adapter_names, adapters)
    ]
    pipeline = LoRaPipeline(model, steps)
    pipeline.apply()
    logger.info("Applied scaling pipeline: %s", {n: a.scale for n, a in zip(adapter_names, adapters)})

    # Merge all adapter contributions into base weights
    merged = model.merge_and_unload()
    logger.info("Merged %d adapter(s) into base weights", len(adapters))

    # Save
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer = _load_tokenizer(base_model, adapters[0].path)
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Saved merged model to %s", output_dir)

    return output_dir
