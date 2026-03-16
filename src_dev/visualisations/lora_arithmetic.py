"""LoRA weight arithmetic: manual merge to sidestep PEFT scaling bugs.

Uses direct weight manipulation instead of PEFT's module.scaling dict,
which is broken for negative scaling factors (huggingface/peft#3004).

Approach: precompute delta = (alpha/r) * B @ A for each LoRA layer,
then for each scaling factor apply base_weight + scale * delta.
"""

from __future__ import annotations

import logging

import torch
from peft import PeftModel

logger = logging.getLogger(__name__)


def precompute_lora_deltas(
    peft_model: PeftModel,
    adapter_name: str = "default",
) -> list[tuple[str, torch.nn.Parameter, torch.Tensor, torch.Tensor]]:
    """Extract LoRA deltas and snapshot base weights from a PeftModel.

    Returns a list of (layer_name, base_weight_param, original_weight, delta)
    tuples. The caller can then apply arbitrary scaling factors by setting
    base_weight_param.data = original_weight + scale * delta.
    """
    layer_info = []
    for name, module in peft_model.named_modules():
        if not (hasattr(module, "lora_A") and adapter_name in module.lora_A):
            continue
        A = module.lora_A[adapter_name].weight  # (r, in_features)
        B = module.lora_B[adapter_name].weight  # (out_features, r)
        r = module.r[adapter_name]
        alpha = module.lora_alpha[adapter_name]
        delta = ((alpha / r) * (B @ A)).detach().clone()
        original = module.base_layer.weight.data.clone()
        layer_info.append((name, module.base_layer.weight, original, delta))

    logger.info("Precomputed LoRA deltas for %d layers (adapter=%s)", len(layer_info), adapter_name)
    return layer_info


def apply_lora_scale(
    layer_info: list[tuple[str, torch.nn.Parameter, torch.Tensor, torch.Tensor]],
    scale: float,
) -> None:
    """Apply base_weight = original + scale * delta for all LoRA layers."""
    with torch.no_grad():
        for _name, weight_param, original, delta in layer_info:
            weight_param.data.copy_(original + scale * delta)


def restore_base_weights(
    layer_info: list[tuple[str, torch.nn.Parameter, torch.Tensor, torch.Tensor]],
) -> None:
    """Restore all base weights to their original (pre-LoRA) values."""
    with torch.no_grad():
        for _name, weight_param, original, _delta in layer_info:
            weight_param.data.copy_(original)
