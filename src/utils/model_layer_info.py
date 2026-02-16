from __future__ import annotations

from collections.abc import Callable

from torch import nn

LayerIdxExtractor = Callable[[str], int | None]

LAYER_PATTERNS: list[str] = [
    "model.layers.",  # LLaMA, Mistral, Qwen, Gemma, OLMo2, Phi, StableLM
    "transformer.h.",  # GPT-2, GPT-Neo, GPT-NeoX, Falcon, Bloom
    "encoder.layer.",  # BERT, RoBERTa
    "encoder.block.",  # T5/mT5 encoder
    "decoder.block.",  # T5/mT5 decoder
    "transformer.blocks.",  # MPT, OLMo (original hf_olmo)
]


def extract_layer_idx(name: str) -> int | None:
    """Extract layer index from a module name, trying all known patterns.

    Searches ``LAYER_PATTERNS`` for a match and returns the integer layer
    index.  Returns ``None`` if no pattern matches.
    """
    for pattern in LAYER_PATTERNS:
        if pattern in name:
            return int(name.split(pattern)[-1].split(".")[0])
    return None


def get_all_layer_indices(
    model: nn.Module,
    layer_idx_extractor: LayerIdxExtractor | None = None,
) -> list[int]:
    """Extract sorted unique layer indices from module names.

    Parameters
    ----------
    model:
        The model to inspect.
    layer_idx_extractor:
        Optional custom function that takes a module name and returns a layer
        index (or ``None``).  When not provided, the default
        :func:`extract_layer_idx` is used.
    """
    extractor = layer_idx_extractor or extract_layer_idx
    indices: set[int] = set()
    for name, _ in model.named_modules():
        idx = extractor(name)
        if idx is not None:
            indices.add(idx)
    return sorted(indices)


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
