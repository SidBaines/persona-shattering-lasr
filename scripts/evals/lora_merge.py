"""Utilities for merging LoRA adapters into standalone HF model directories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.evals.config import EvalModelConfig, normalize_component


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported dtype for LoRA merge: {dtype_name}")
    return dtype


def _adapter_signature(adapter_dir: Path) -> dict[str, dict[str, int]]:
    signature: dict[str, dict[str, int]] = {}
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        path = adapter_dir / filename
        if not path.exists():
            continue
        stats = path.stat()
        signature[filename] = {
            "size": int(stats.st_size),
            "mtime_ns": int(stats.st_mtime_ns),
        }
    return signature


def _build_cache_entry_name(
    model_cfg: EvalModelConfig,
    adapter_dir: Path,
) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "base_model": model_cfg.model,
        "adapter_path": str(adapter_dir),
        "revision": model_cfg.revision,
        "dtype": model_cfg.dtype,
        "adapter_signature": _adapter_signature(adapter_dir),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:12]
    base_name = normalize_component(model_cfg.model.rsplit("/", 1)[-1], fallback="base")
    adapter_name = normalize_component(adapter_dir.name, fallback="adapter")
    return f"{base_name}__{adapter_name}__{digest}", payload


def _load_tokenizer_for_merge(model_cfg: EvalModelConfig, adapter_dir: Path):
    adapter_tokenizer = adapter_dir / "tokenizer_config.json"
    if adapter_tokenizer.exists():
        return AutoTokenizer.from_pretrained(str(adapter_dir), use_fast=True)
    return AutoTokenizer.from_pretrained(
        model_cfg.model,
        revision=model_cfg.revision,
        use_fast=True,
    )


def ensure_merged_lora_model(
    *,
    model_cfg: EvalModelConfig,
    cache_dir: Path,
    force_remerge: bool = False,
    logger: Any | None = None,
) -> Path:
    """Merge a local LoRA adapter and return a reusable merged model directory."""
    if model_cfg.kind != "lora" or not model_cfg.adapter_path:
        raise ValueError("ensure_merged_lora_model requires kind='lora' with adapter_path.")

    adapter_dir = Path(model_cfg.adapter_path).expanduser().resolve()
    if not adapter_dir.exists():
        raise FileNotFoundError(f"LoRA adapter path not found: {adapter_dir}")

    cache_root = cache_dir.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    entry_name, metadata = _build_cache_entry_name(model_cfg, adapter_dir)
    target_dir = cache_root / entry_name
    metadata_path = target_dir / "merge_metadata.json"

    if target_dir.exists() and not force_remerge and metadata_path.exists():
        if logger is not None:
            logger.info("Reusing merged LoRA model cache: %s", target_dir)
        return target_dir

    if target_dir.exists():
        if logger is not None:
            logger.info("Rebuilding merged LoRA cache entry: %s", target_dir)
        shutil.rmtree(target_dir)

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"{entry_name}-", dir=str(cache_root)))
    try:
        if logger is not None:
            logger.info(
                "Merging LoRA adapter for inspect task evals (base=%s, adapter=%s)",
                model_cfg.model,
                adapter_dir,
            )

        dtype = _resolve_torch_dtype(model_cfg.dtype)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_cfg.model,
            revision=model_cfg.revision,
            torch_dtype=dtype,
            device_map=model_cfg.device_map,
        )
        model_with_adapter = PeftModel.from_pretrained(base_model, str(adapter_dir))
        merged_model = model_with_adapter.merge_and_unload()
        tokenizer = _load_tokenizer_for_merge(model_cfg, adapter_dir)

        merged_model.save_pretrained(str(tmp_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(tmp_dir))

        metadata["merged_model_path"] = str(target_dir)
        metadata["cache_entry"] = entry_name
        (tmp_dir / "merge_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        tmp_dir.replace(target_dir)
        if logger is not None:
            logger.info("Merged LoRA model cached at: %s", target_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return target_dir
