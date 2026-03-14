"""Activation-capped model wrapper.

Wraps a HuggingFace PreTrainedModel with forward hooks that clamp the
residual-stream projection along a direction axis. Supports both floor
(clamp min) and ceiling (clamp max) modes.
"""

from typing import Literal

import torch
from torch import nn


def get_model_layers(model: nn.Module) -> nn.ModuleList:
    """Get transformer layers, handling plain, PeftModel-wrapped, and ActivationCappedModel models."""
    # Unwrap ActivationCappedModel
    if hasattr(model, "_model"):
        model = model._model
    # Unwrap PeftModel (check the class name since PreTrainedModel also has .base_model)
    try:
        from peft import PeftModel
        if isinstance(model, PeftModel):
            model = model.base_model.model
    except ImportError:
        pass
    # PreTrainedModel (e.g. LlamaForCausalLM) has .model.layers
    return model.model.layers


def make_capping_hook(
    axis_direction: torch.Tensor,
    threshold: float,
    mode: Literal["floor", "ceiling"] = "floor",
):
    """Create a forward hook that clamps projection along an axis direction.

    Args:
        axis_direction: Direction vector (hidden_dim,) to project onto.
        threshold: Projection value to clamp at.
        mode: "floor" clamps projections below threshold upward,
              "ceiling" clamps projections above threshold downward.

    Returns:
        A hook function suitable for register_forward_hook.
    """
    ax = axis_direction.float()
    ax_normed = ax / (ax.norm() + 1e-8)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        orig_dtype = hidden.dtype
        h = hidden.float()
        device = h.device
        ax_dev = ax_normed.to(device)

        proj = h @ ax_dev  # (..., seq_len)

        if mode == "floor":
            needs_correction = proj < threshold
            if needs_correction.any():
                correction = (threshold - proj).clamp(min=0)
                h = h + correction.unsqueeze(-1) * ax_dev
        else:  # ceiling
            needs_correction = proj > threshold
            if needs_correction.any():
                correction = (proj - threshold).clamp(min=0)
                h = h - correction.unsqueeze(-1) * ax_dev

        h = h.to(orig_dtype)

        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    return hook_fn


def compute_thresholds_at_fraction(
    per_layer_range: dict[int, tuple[float, float]],
    fraction: float,
) -> dict[int, float]:
    """Linearly interpolate between min and max projection at each layer.

    Args:
        per_layer_range: {layer_idx: (min_projection, max_projection)}.
        fraction: 0.0 = global min (base end), 1.0 = global max (LoRA end).

    Returns:
        {layer_idx: threshold} for each layer in per_layer_range.
    """
    return {
        layer: lo + fraction * (hi - lo)
        for layer, (lo, hi) in per_layer_range.items()
    }


class ActivationCappedModel(nn.Module):
    """Wraps a HuggingFace model with activation capping along a direction axis.

    Registers forward hooks on specified layers that clamp the residual stream
    projection along the axis. The wrapped model can be used normally via
    forward() and generate() — capping is applied transparently.

    Args:
        model: A HuggingFace PreTrainedModel (plain or PeftModel-wrapped).
        axis: Direction axis tensor of shape (n_layers, hidden_dim).
        layer_thresholds: {layer_idx: threshold} for each layer to cap.
        mode: "floor" clamps projections below threshold upward,
              "ceiling" clamps projections above threshold downward.
    """

    def __init__(
        self,
        model: nn.Module,
        axis: torch.Tensor,
        layer_thresholds: dict[int, float],
        mode: Literal["floor", "ceiling"] = "floor",
    ):
        super().__init__()
        self._model = model
        self._axis = axis
        self._layer_thresholds = layer_thresholds
        self._mode = mode
        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        layers = get_model_layers(self._model)
        for layer_idx, threshold in self._layer_thresholds.items():
            hook = make_capping_hook(self._axis[layer_idx], threshold, self._mode)
            handle = layers[layer_idx].register_forward_hook(hook)
            self._handles.append(handle)

    def remove_hooks(self) -> None:
        """Remove all capping hooks (disables capping)."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def register_hooks(self) -> None:
        """Re-register capping hooks (re-enables capping after remove_hooks)."""
        if self._handles:
            self.remove_hooks()
        self._register_hooks()

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self._model.generate(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)

    @classmethod
    def from_pretrained(
        cls,
        model: nn.Module,
        axis_path: str,
        per_layer_range_path: str,
        fraction: float,
        capping_layers: list[int],
        mode: Literal["floor", "ceiling"] = "floor",
    ) -> "ActivationCappedModel":
        """Construct from saved axis and per-layer range files.

        Args:
            model: A HuggingFace PreTrainedModel.
            axis_path: Path to .pt file with {"axis": Tensor, "metadata": ...}.
            per_layer_range_path: Path to .pt file with {"per_layer_range": {layer: (min, max)}}.
            fraction: Sweep fraction (0.0 = global min, 1.0 = global max).
            capping_layers: Which layers to apply capping to.
            mode: "floor" or "ceiling".

        Returns:
            An ActivationCappedModel instance.
        """
        axis_data = torch.load(axis_path, weights_only=False)
        axis = axis_data["axis"]

        range_data = torch.load(per_layer_range_path, weights_only=False)
        per_layer_range = range_data["per_layer_range"]

        # Filter to requested capping layers
        filtered_range = {l: per_layer_range[l] for l in capping_layers if l in per_layer_range}
        layer_thresholds = compute_thresholds_at_fraction(filtered_range, fraction)

        return cls(model, axis, layer_thresholds, mode=mode)
