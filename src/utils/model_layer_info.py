from __future__ import annotations

from torch import nn


def get_all_layer_indices(model: nn.Module) -> list[int]:
    """Extract sorted unique layer indices from ``model.layers.<N>`` module names."""
    return sorted(
        {
            int(name.split("model.layers.")[-1].split(".")[0])
            for name, _ in model.named_modules()
            if "model.layers." in name
        }
    )


def get_num_layers(model: nn.Module) -> int:
    """Return the number of ``model.layers.*`` layers in *model*."""
    return len(get_all_layer_indices(model))


def select_every_nth_layer(model: nn.Module, n: int) -> list[int]:
    """Return every *n*-th layer index from *model*."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return get_all_layer_indices(model)[::n]


def select_first_n_layers(model: nn.Module, n: int) -> list[int]:
    """Return the first *n* layer indices from *model*."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return get_all_layer_indices(model)[:n]


def select_last_n_layers(model: nn.Module, n: int) -> list[int]:
    """Return the last *n* layer indices from *model*."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n == 0:
        return []
    return get_all_layer_indices(model)[-n:]


def select_middle_n_layers(model: nn.Module, n: int) -> list[int]:
    """Return the middle *n* layer indices from *model*."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    layers = get_all_layer_indices(model)
    if n >= len(layers):
        return layers
    start = (len(layers) - n) // 2
    return layers[start : start + n]


def select_layer_fraction(model: nn.Module, fraction: float) -> list[int]:
    """Return an evenly-spaced subset comprising *fraction* of layers from *model*."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be between 0.0 and 1.0, got {fraction}")
    if fraction == 0.0:
        return []
    layers = get_all_layer_indices(model)
    if fraction == 1.0:
        return layers
    n = max(1, round(len(layers) * fraction))
    step = len(layers) / n
    return [layers[int(i * step)] for i in range(n)]
