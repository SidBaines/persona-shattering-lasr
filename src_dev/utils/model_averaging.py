"""Weighted averaging of two full HuggingFace causal-LM checkpoints.

This utility is general: given two HuggingFace model references (Hub IDs or
local paths) and a weight ``w`` in ``[0, 1]``, it produces a new model whose
parameters are ``(1 - w) * A + w * B``, saved to a local directory that can be
loaded with ``AutoModelForCausalLM.from_pretrained``.

Typical use: interpolate between a base model and its instruction-tuned
counterpart to study how a LoRA persona behaves as the base shifts from base
to instruct.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src_dev.utils.lora_composition import resolve_torch_dtype

logger = logging.getLogger(__name__)

_REQUIRED_MERGED_FILES = ("config.json",)


def _slug(model_ref: str) -> str:
    """Filesystem-safe short slug for a model reference."""
    tail = model_ref.rstrip("/").split("/")[-1]
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in tail)
    digest = hashlib.sha1(model_ref.encode()).hexdigest()[:6]
    return f"{safe}-{digest}"


def averaged_model_dir(
    model_a: str,
    model_b: str,
    weight: float,
    root: Path,
) -> Path:
    """Deterministic cache directory for an averaged model."""
    weight_tag = f"{weight:.4f}".rstrip("0").rstrip(".").replace(".", "_") or "0"
    name = f"{_slug(model_a)}__{_slug(model_b)}__w{weight_tag}"
    return root / name


def _has_saved_model(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not all((path / f).exists() for f in _REQUIRED_MERGED_FILES):
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin"))


def save_averaged_model(
    model_a: str,
    model_b: str,
    weight: float,
    output_dir: Path,
    *,
    dtype: str = "bfloat16",
) -> Path:
    """Save ``(1 - weight) * A + weight * B`` to ``output_dir``.

    Args:
        model_a: HF Hub ID or local path for model A (weight = 1 - weight).
        model_b: HF Hub ID or local path for model B.
        weight: Fraction of model B. ``0`` → pure A, ``1`` → pure B.
        output_dir: Directory to save the merged model + tokenizer.
        dtype: Torch dtype name used for loading/saving (e.g. ``"bfloat16"``).

    Returns:
        ``output_dir``.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch_dtype = resolve_torch_dtype(dtype)

    logger.info("Loading model A: %s", model_a)
    model = AutoModelForCausalLM.from_pretrained(
        model_a, torch_dtype=torch_dtype, device_map="cpu"
    )
    logger.info("Loading model B state_dict: %s", model_b)
    model_b_obj = AutoModelForCausalLM.from_pretrained(
        model_b, torch_dtype=torch_dtype, device_map="cpu"
    )
    sd_b = model_b_obj.state_dict()

    a_keys = set(model.state_dict().keys())
    b_keys = set(sd_b.keys())
    if a_keys != b_keys:
        only_a = sorted(a_keys - b_keys)[:5]
        only_b = sorted(b_keys - a_keys)[:5]
        raise ValueError(
            f"state_dict keys differ between {model_a} and {model_b}. "
            f"Only in A (sample): {only_a}. Only in B (sample): {only_b}."
        )

    w_a = 1.0 - weight
    w_b = weight
    with torch.no_grad():
        for name, param in model.named_parameters():
            tensor_b = sd_b[name]
            if tensor_b.shape != param.shape:
                raise ValueError(
                    f"Shape mismatch at '{name}': A={tuple(param.shape)} "
                    f"B={tuple(tensor_b.shape)}"
                )
            averaged = w_a * param.data.to(torch.float32) + w_b * tensor_b.to(
                torch.float32
            )
            param.data.copy_(averaged.to(param.dtype))
        for name, buf in model.named_buffers():
            if name in sd_b and sd_b[name].shape == buf.shape and buf.is_floating_point():
                averaged = w_a * buf.to(torch.float32) + w_b * sd_b[name].to(torch.float32)
                buf.copy_(averaged.to(buf.dtype))

    del model_b_obj, sd_b

    logger.info("Saving averaged model to %s (dtype=%s, weight=%s)", output_dir, dtype, weight)
    model.save_pretrained(str(output_dir), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(model_b, use_fast=True)
    tokenizer.save_pretrained(str(output_dir))

    info = {
        "model_a": model_a,
        "model_b": model_b,
        "weight": weight,
        "dtype": dtype,
    }
    (output_dir / "average_info.json").write_text(json.dumps(info, indent=2))

    return output_dir


def ensure_averaged_model(
    model_a: str,
    model_b: str,
    weight: float,
    *,
    root: Path,
    dtype: str = "bfloat16",
) -> Path:
    """Return cached averaged-model dir, creating it if missing."""
    out = averaged_model_dir(model_a, model_b, weight, root)
    if _has_saved_model(out):
        logger.info("Reusing cached averaged model at %s", out)
        return out
    return save_averaged_model(model_a, model_b, weight, out, dtype=dtype)
