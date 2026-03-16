"""Merge weighted LoRA adapters into a standalone model."""

from __future__ import annotations

import logging
from pathlib import Path

from scripts.evals.config import AdapterConfig
from scripts.evals.model_resolution import resolve_model_reference
from scripts.utils.lora_composition import merge_weighted_adapters

logger = logging.getLogger(__name__)


def merge_adapters(
    *,
    base_model: str,
    adapters: list[AdapterConfig],
    output_dir: Path,
    dtype: str = "bfloat16",
    device_map: str = "auto",
) -> Path:
    """Load base model + adapters, apply scaling, merge, and save."""
    logger.info("Merging %d LoRA adapter(s) into %s", len(adapters), base_model)
    return merge_weighted_adapters(
        base_model=base_model,
        adapters=adapters,
        output_dir=output_dir,
        dtype=dtype,
        device_map=device_map,
        base_model_resolver=lambda ref: resolve_model_reference(ref, kind="base model"),
        adapter_resolver=lambda ref: resolve_model_reference(ref, kind="adapter"),
    )
