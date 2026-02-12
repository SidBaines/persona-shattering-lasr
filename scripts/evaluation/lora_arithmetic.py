"""Manual LoRA weight arithmetic for applying adapters with arbitrary scaling.

PEFT's ``add_weighted_adapter`` is broken for negative weights
(huggingface/peft#3004).  This module sidesteps the issue by computing
``s * (alpha / r) * B @ A`` per LoRA layer and adding the result directly
to the base model weights.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def merge_lora_into_base(
    base_model: PreTrainedModel,
    adapter_path: str | Path,
    scaling_factor: float = 1.0,
    adapter_name: str = "default",
) -> PreTrainedModel:
    """Merge a LoRA adapter into base model weights with a custom scaling factor.

    Steps:
      1. Wrap base_model with PeftModel.from_pretrained(adapter_path)
      2. For every LoRA module, compute delta = (alpha / r) * B @ A
      3. Add scaling_factor * delta to the base layer weight
      4. Zero out LoRA A/B so merge_and_unload adds nothing extra
      5. Call merge_and_unload to strip the PEFT wrapper

    Args:
        base_model: A HuggingFace model.  Will be modified via the PEFT
            wrapper — caller should pass a freshly-loaded or state-dict-reset
            model for each scaling factor.
        adapter_path: Path to the saved LoRA adapter directory.
        scaling_factor: Multiplier for the LoRA delta.  1.0 = standard,
            -1.0 = inverted, 0.0 = base model only.
        adapter_name: Name of the adapter inside the PeftModel.

    Returns:
        A plain (non-PEFT) PreTrainedModel with modified weights.
    """
    adapter_path = str(adapter_path)

    # Wrap with PEFT to load adapter weights into the right places
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    peft_model.eval()

    merged_count = 0

    for name, module in peft_model.named_modules():
        if not (hasattr(module, "lora_A") and adapter_name in module.lora_A):
            continue

        lora_A = module.lora_A[adapter_name].weight  # (r, in_features)
        lora_B = module.lora_B[adapter_name].weight  # (out_features, r)

        r = module.r[adapter_name]
        alpha = module.lora_alpha[adapter_name]
        lora_scaling = alpha / r

        with torch.no_grad():
            delta = lora_B @ lora_A  # (out_features, in_features)
            # Apply our custom-scaled delta to the base weights
            module.base_layer.weight.data += scaling_factor * lora_scaling * delta

            # Zero out LoRA matrices so merge_and_unload won't add anything
            module.lora_A[adapter_name].weight.data.zero_()
            module.lora_B[adapter_name].weight.data.zero_()

        merged_count += 1

    logger.info(
        "Merged %d LoRA layers with scaling_factor=%.3f (adapter: %s)",
        merged_count,
        scaling_factor,
        adapter_path,
    )

    # Strip the PEFT wrapper.  Since we zeroed all LoRA weights,
    # merge_and_unload adds 0 and just removes the wrapper.
    plain_model = peft_model.merge_and_unload(progressbar=False)
    return plain_model
