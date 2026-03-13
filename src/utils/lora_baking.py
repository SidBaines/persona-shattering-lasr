"""Utility for baking a LoRA scale factor into adapter weights on disk.

vLLM's ``LoRARequest`` does not support per-request scaling — the adapter
contribution is fixed at load time (``lora_alpha / r``).  To sweep over
LoRA scales with vLLM we pre-bake each scale point as a separate adapter:
multiply every ``lora_B`` weight matrix by the desired scale factor and set
``lora_alpha == r`` (so vLLM's built-in scaling becomes exactly 1.0, leaving
the pre-baked contribution intact).

Usage::

    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from src.utils.lora_baking import bake_lora_scale

    base = AutoModelForCausalLM.from_pretrained(...)
    model = PeftModel.from_pretrained(base, adapter_ref, adapter_name="default")

    # Write a half-strength adapter to disk
    out_dir = bake_lora_scale(model, adapter_name="default", scale=0.5,
                              output_dir=Path("scratch/adapters/scale_+0.50"))
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel


def bake_lora_scale(
    model: PeftModel,
    adapter_name: str,
    scale: float,
    output_dir: Path,
) -> Path:
    """Save a copy of *adapter_name* with *scale* baked into ``lora_B`` weights.

    The saved adapter behaves as if ``LoRaScaling(scale_factor=scale)`` had
    been applied, but it can be loaded by vLLM (which doesn't support
    per-request scaling).

    Concretely:
    - For each LoRA linear layer: ``new_lora_B = lora_B * scale``.
    - ``adapter_config.json`` is written with ``lora_alpha == r`` so that
      vLLM's load-time scaling factor (``lora_alpha / r``) equals 1.0 and
      the baked contribution is used as-is.

    Args:
        model: Loaded ``PeftModel`` (must not have scaling already applied).
        adapter_name: Internal PEFT adapter name (e.g. ``"default"``).
        scale: Scale factor to bake in.  ``0.0`` produces a zero adapter
            (equivalent to base model), ``1.0`` is the original adapter.
            Negative values invert the LoRA direction.
        output_dir: Directory to write the baked adapter.  Created if needed;
            any existing contents are overwritten.

    Returns:
        Resolved path to *output_dir*.
    """
    output_dir = Path(output_dir).resolve()

    # Write the original adapter to a temp dir, then mutate lora_B in the copy.
    tmp_dir = output_dir.parent / f"_{output_dir.name}_tmp"
    try:
        # save_pretrained always writes the current in-memory weights, so we
        # save first (unmodified), then patch the tensor files on disk.
        model.save_pretrained(str(tmp_dir), selected_adapters=[adapter_name])

        # The actual adapter sub-directory is named after the adapter.
        adapter_dir = tmp_dir / adapter_name
        if not adapter_dir.exists():
            # Some PEFT versions write directly to the given dir without a sub-dir.
            adapter_dir = tmp_dir

        _patch_lora_b_weights(adapter_dir, scale=scale)
        _patch_adapter_config(adapter_dir, model, adapter_name)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(adapter_dir, output_dir)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    return output_dir


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _patch_lora_b_weights(adapter_dir: Path, scale: float) -> None:
    """Multiply all lora_B tensors in the safetensors/bin file by *scale*."""
    # Try safetensors first (preferred by modern PEFT), fall back to .bin.
    sf_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"

    if sf_path.exists():
        _patch_safetensors(sf_path, scale)
    elif bin_path.exists():
        _patch_bin(bin_path, scale)
    else:
        raise FileNotFoundError(
            f"No adapter_model.safetensors or adapter_model.bin found in {adapter_dir}"
        )


def _patch_safetensors(path: Path, scale: float) -> None:
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise ImportError("safetensors is required: pip install safetensors") from exc

    tensors: dict[str, torch.Tensor] = load_file(str(path))
    patched = {
        key: (tensor * scale if "lora_B" in key else tensor)
        for key, tensor in tensors.items()
    }
    save_file(patched, str(path))


def _patch_bin(path: Path, scale: float) -> None:
    state: dict[str, Any] = torch.load(str(path), map_location="cpu", weights_only=True)
    patched = {
        key: (tensor * scale if "lora_B" in key else tensor)
        for key, tensor in state.items()
    }
    torch.save(patched, str(path))


def _patch_adapter_config(
    adapter_dir: Path, model: PeftModel, adapter_name: str
) -> None:
    """Set lora_alpha == r in adapter_config.json so vLLM's scaling == 1.0."""
    import json

    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        return

    config = json.loads(config_path.read_text())

    # Read r from the PEFT config; fall back to the JSON value if not found.
    peft_cfg = model.peft_config.get(adapter_name)
    r: int = getattr(peft_cfg, "r", None) or config.get("r", 16)

    config["lora_alpha"] = r
    config_path.write_text(json.dumps(config, indent=2))
