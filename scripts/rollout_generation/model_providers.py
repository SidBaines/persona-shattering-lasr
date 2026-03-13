"""Model provider abstraction for sweep experiments.

Provides a ``ModelProvider`` ABC and concrete implementations for different
model types (LoRA-scaled, activation-capped, plain HF). Each provider manages
model lifecycle — loading, variant switching, and cleanup — so sweep code
can iterate over model variants without knowing the underlying type.

Usage::

    from scripts.rollout_generation.model_providers import LoRaScaleProvider

    provider = LoRaScaleProvider(
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        adapter="repo_id::subfolder",
        scale_points=[-1.0, 0.0, 1.0, 2.0],
    )
    with provider:
        for variant in provider.variant_names():
            with provider.activate(variant) as (model, tokenizer):
                # model is ready with the appropriate scaling applied
                ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from torch import nn


def _resolve_hf_path(path: str) -> str:
    """Resolve ``hf://org/repo/file`` to a local path via ``huggingface_hub``.

    If *path* does not start with ``hf://`` it is returned unchanged.
    """
    if not path.startswith("hf://"):
        return path
    from huggingface_hub import hf_hub_download

    # hf://org/repo/path/to/file  →  repo_id="org/repo", filename="path/to/file"
    stripped = path[len("hf://"):]
    parts = stripped.split("/", 2)
    if len(parts) < 3:
        raise ValueError(f"hf:// path must be hf://org/repo/filename, got {path!r}")
    repo_id = f"{parts[0]}/{parts[1]}"
    filename = parts[2]
    return hf_hub_download(repo_id=repo_id, filename=filename)


def _parse_adapter_ref(adapter: str) -> tuple[str, str | None]:
    """Split ``"repo_id::subfolder"`` or ``"local://path"`` into (ref, subfolder).

    Supports:
    - ``"repo_id::subfolder"`` → ``(repo_id, subfolder)``
    - ``"local://path"`` → ``(path, None)``
    - ``"plain_path"`` → ``(plain_path, None)``
    """
    if adapter.startswith("local://"):
        return str(Path(adapter[len("local://"):]).resolve()), None
    if "::" in adapter:
        repo, subfolder = adapter.split("::", 1)
        return repo, subfolder
    return adapter, None


def _load_base_model(
    base_model: str,
    dtype: str,
) -> tuple[nn.Module, Any]:
    """Load a plain HuggingFace model and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype, None)
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype!r}")

    print(f"  loading base model: {base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch_dtype, device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


def _load_peft_model(
    base_model: str,
    adapter: str,
    adapter_name: str,
    dtype: str,
) -> tuple[nn.Module, Any]:
    """Load a base model + LoRA adapter. Returns ``(peft_model, tokenizer)``."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype, None)
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype!r}")

    adapter_ref, subfolder = _parse_adapter_ref(adapter)

    print(f"  loading base model: {base_model}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch_dtype, device_map="auto"
    )
    print(
        f"  loading adapter: {adapter_ref}"
        + (f"  (subfolder={subfolder})" if subfolder else ""),
        flush=True,
    )
    peft_kwargs: dict[str, Any] = {"adapter_name": adapter_name}
    if subfolder:
        peft_kwargs["subfolder"] = subfolder
    peft_model = PeftModel.from_pretrained(base, adapter_ref, **peft_kwargs)

    tokenizer_kwargs: dict[str, Any] = {}
    if subfolder:
        tokenizer_kwargs["subfolder"] = subfolder
    tokenizer = AutoTokenizer.from_pretrained(adapter_ref, **tokenizer_kwargs)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_model.config.pad_token_id = tokenizer.pad_token_id
    peft_model.eval()
    return peft_model, tokenizer


# ── ABC ───────────────────────────────────────────────────────────────────────


class ModelProvider(ABC):
    """Provides model variants for a sweep. Handles loading, switching, cleanup.

    Usage::

        with provider:
            for variant in provider.variant_names():
                with provider.activate(variant) as (model, tokenizer):
                    # use model
                    ...
    """

    @abstractmethod
    def variant_names(self) -> list[str]:
        """Return ordered list of variant identifiers."""
        ...

    @abstractmethod
    def variant_label(self, variant: str) -> str:
        """Return a directory-safe label for this variant (used in output paths)."""
        ...

    @abstractmethod
    @contextmanager
    def activate(self, variant: str) -> Iterator[tuple[nn.Module, Any]]:
        """Context manager yielding ``(model, tokenizer)`` for the given variant.

        Setup (e.g. applying LoRA scaling) happens on entry; cleanup (e.g.
        restoring scaling) happens on exit. Base model loading is lazy —
        happens on first ``activate`` call.
        """
        ...

    def close(self) -> None:
        """Release resources (e.g. move model to CPU, free GPU memory)."""

    def __enter__(self) -> "ModelProvider":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ── LoRA scale provider ──────────────────────────────────────────────────────


class LoRaScaleProvider(ModelProvider):
    """Sweeps LoRA adapter scaling factors.

    Loads the base model + adapter once (lazily on first ``activate``).
    Each variant applies ``LoRaScaling`` at the given scale factor and
    restores it on context exit.

    Args:
        base_model: HuggingFace model ID.
        adapter: Adapter path or ``"repo_id::subfolder"`` reference.
        scale_points: List of scale factors to sweep.
        adapter_name: Internal PEFT adapter name.
        dtype: Torch dtype string for model loading.
    """

    def __init__(
        self,
        base_model: str,
        adapter: str,
        scale_points: list[float],
        adapter_name: str = "default",
        dtype: str = "bfloat16",
    ) -> None:
        self._base_model = base_model
        self._adapter = adapter
        self._scale_points = sorted(scale_points)
        self._adapter_name = adapter_name
        self._dtype = dtype
        self._model: nn.Module | None = None
        self._tokenizer: Any = None

    def variant_names(self) -> list[str]:
        return [str(s) for s in self._scale_points]

    def variant_label(self, variant: str) -> str:
        return f"scale_{float(variant):+.2f}"

    @contextmanager
    def activate(self, variant: str) -> Iterator[tuple[nn.Module, Any]]:
        if self._model is None:
            self._model, self._tokenizer = _load_peft_model(
                self._base_model, self._adapter, self._adapter_name, self._dtype
            )

        from src.utils.peft_manipulations import LoRaScaling

        scale = float(variant)
        scaler = LoRaScaling(
            self._model, adapter_name=self._adapter_name, scale_factor=scale
        ).apply()
        try:
            yield (self._model, self._tokenizer)
        finally:
            scaler.restore()

    def close(self) -> None:
        if self._model is not None:
            try:
                self._model.cpu()
            except Exception:
                pass
            self._model = None
            self._tokenizer = None


# ── Activation capping provider ───────────────────────────────────────────────


class ActivationCapProvider(ModelProvider):
    """Sweeps activation capping fractions along a direction axis.

    Loads the base model once (optionally with a LoRA adapter). Each variant
    wraps the model with ``ActivationCappedModel`` at the given fraction and
    removes hooks on context exit.

    Args:
        base_model: HuggingFace model ID.
        axis_path: Path to ``.pt`` file with ``{"axis": Tensor, ...}``.
        per_layer_range_path: Path to ``.pt`` file with
            ``{"per_layer_range": {layer: (min, max)}}``.
        fractions: List of fractions to sweep (0.0 = base end, 1.0 = LoRA end).
        capping_layers: Which layers to apply capping to.
        mode: ``"floor"`` or ``"ceiling"``.
        dtype: Torch dtype string for model loading.
        adapter: Optional adapter path for capping a PEFT model.
        adapter_name: Internal PEFT adapter name (if adapter is provided).
    """

    def __init__(
        self,
        base_model: str,
        axis_path: str,
        per_layer_range_path: str,
        fractions: list[float],
        capping_layers: list[int],
        mode: str = "floor",
        dtype: str = "bfloat16",
        adapter: str | None = None,
        adapter_name: str = "default",
    ) -> None:
        self._base_model = base_model
        self._axis_path = _resolve_hf_path(axis_path)
        self._per_layer_range_path = _resolve_hf_path(per_layer_range_path)
        self._fractions = sorted(fractions)
        self._capping_layers = capping_layers
        self._mode = mode
        self._dtype = dtype
        self._adapter = adapter
        self._adapter_name = adapter_name
        self._model: nn.Module | None = None
        self._tokenizer: Any = None

    def variant_names(self) -> list[str]:
        return [str(f) for f in self._fractions]

    def variant_label(self, variant: str) -> str:
        return f"frac_{float(variant):.2f}"

    @contextmanager
    def activate(self, variant: str) -> Iterator[tuple[nn.Module, Any]]:
        if self._model is None:
            if self._adapter:
                self._model, self._tokenizer = _load_peft_model(
                    self._base_model, self._adapter, self._adapter_name, self._dtype
                )
            else:
                self._model, self._tokenizer = _load_base_model(
                    self._base_model, self._dtype
                )

        from scripts.experiments.activation_capping.model import ActivationCappedModel

        fraction = float(variant)
        capped = ActivationCappedModel.from_pretrained(
            self._model,
            axis_path=self._axis_path,
            per_layer_range_path=self._per_layer_range_path,
            fraction=fraction,
            capping_layers=self._capping_layers,
            mode=self._mode,
        )
        try:
            yield (capped, self._tokenizer)
        finally:
            capped.remove_hooks()

    def close(self) -> None:
        if self._model is not None:
            try:
                self._model.cpu()
            except Exception:
                pass
            self._model = None
            self._tokenizer = None


# ── Single model provider ────────────────────────────────────────────────────


class SingleModelProvider(ModelProvider):
    """Provides a single model variant (no sweep dimension).

    Useful for baselines or API-model experiments where there is only one
    model configuration to evaluate.

    Args:
        model_id: HuggingFace model ID.
        dtype: Torch dtype string for model loading.
    """

    def __init__(self, model_id: str, dtype: str = "bfloat16") -> None:
        self._model_id = model_id
        self._dtype = dtype
        self._model: nn.Module | None = None
        self._tokenizer: Any = None

    def variant_names(self) -> list[str]:
        return ["base"]

    def variant_label(self, variant: str) -> str:
        return "base"

    @contextmanager
    def activate(self, variant: str) -> Iterator[tuple[nn.Module, Any]]:
        if self._model is None:
            self._model, self._tokenizer = _load_base_model(
                self._model_id, self._dtype
            )
        yield (self._model, self._tokenizer)

    def close(self) -> None:
        if self._model is not None:
            try:
                self._model.cpu()
            except Exception:
                pass
            self._model = None
            self._tokenizer = None
