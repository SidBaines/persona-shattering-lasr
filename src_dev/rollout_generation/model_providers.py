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

import gc
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn

from src_dev.activation_capping.model import ActivationCappedModel
from src_dev.inference.providers.base import InferenceProvider, PromptInput


def _resolve_hf_path(path: str) -> str:
    """Resolve ``hf://org/repo/file`` to a local path via ``huggingface_hub``.

    If *path* does not start with ``hf://`` it is returned unchanged.
    """
    if not path.startswith("hf://"):
        return path
    from huggingface_hub import hf_hub_download

    # hf://org/repo/path/to/file  →  repo_id="org/repo", filename="path/to/file"
    stripped = path[len("hf://") :]
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
        return str(Path(adapter[len("local://") :]).resolve()), None
    if "::" in adapter:
        repo, subfolder = adapter.split("::", 1)
        return repo, subfolder
    return adapter, None


def _resolve_adapter_to_local(adapter: str) -> str:
    """Resolve an adapter reference to a local directory path.

    If the adapter is stored in a HuggingFace *dataset* repo (``repo_id::subfolder``
    format), downloads it via ``snapshot_download`` with ``repo_type="dataset"``
    and returns the local snapshot path containing the adapter files.

    Plain local paths and ``local://`` paths are returned unchanged.
    """
    repo_id, subfolder = _parse_adapter_ref(adapter)
    if Path(repo_id).exists():
        # Already a local path
        return repo_id if subfolder is None else str(Path(repo_id) / subfolder)
    # HF repo — try dataset repo first (common case), fall back to model repo
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError
    allow_patterns = [f"{subfolder}/**"] if subfolder else None
    for repo_type in ("dataset", "model"):
        try:
            local_dir = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                allow_patterns=allow_patterns,
            )
            return local_dir if subfolder is None else str(Path(local_dir) / subfolder)
        except RepositoryNotFoundError:
            continue
    raise RuntimeError(f"Could not find HF repo {repo_id!r} as dataset or model")


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

    local_adapter_path = _resolve_adapter_to_local(adapter)

    print(f"  loading base model: {base_model}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch_dtype, device_map="auto"
    )
    print(f"  loading adapter: {local_adapter_path}", flush=True)
    peft_model = PeftModel.from_pretrained(base, local_adapter_path, adapter_name=adapter_name)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
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

    Fractions control where the capping threshold sits relative to the
    observed per-layer projection range ``(lo, hi)``, where ``lo`` is the
    base model's typical projection and ``hi`` is the LoRA model's:

    - **fraction=0.0**: threshold at ``lo`` (base end). Floor mode pushes
      activations up to at least the base level — effectively a no-op.
    - **fraction=1.0**: threshold at ``hi`` (LoRA end). Floor mode pushes
      activations up to the LoRA model's level — maximum trait enforcement.
    - **fraction=0.5**: threshold halfway between base and LoRA.
    - **fraction=-0.2**: threshold at ``lo - 0.2*(hi-lo)`` — *below* the
      base range. Mode auto-flips to ceiling, clamping activations *down*
      to this threshold. This suppresses the trait beyond what the base
      model naturally does (anti-trait direction).

    In short: positive fractions = floor (push toward trait), negative
    fractions = ceiling (push away from trait, extrapolating past baseline).

    Args:
        base_model: HuggingFace model ID.
        axis_path: Path to ``.pt`` file with ``{"axis": Tensor, ...}``.
        per_layer_range_path: Path to ``.pt`` file with
            ``{"per_layer_range": {layer: (min, max)}}``.
        fractions: List of fractions to sweep. Positive values use floor
            mode (push toward trait); negative values auto-flip to ceiling
            mode (suppress trait below baseline).
        capping_layers: Which layers to apply capping to.
        mode: Default mode for non-negative fractions (``"floor"`` or
            ``"ceiling"``). Negative fractions always use ``"ceiling"``.
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

        fraction = float(variant)
        # Negative fractions extrapolate below the base range and use ceiling
        # mode (clamp downward); positive fractions use the configured mode.
        mode = "ceiling" if fraction < 0 else self._mode
        capped = ActivationCappedModel.from_pretrained(
            self._model,
            axis_path=self._axis_path,
            per_layer_range_path=self._per_layer_range_path,
            fraction=fraction,
            capping_layers=self._capping_layers,
            mode=mode,
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
            self._model, self._tokenizer = _load_base_model(self._model_id, self._dtype)
        yield (self._model, self._tokenizer)

    def close(self) -> None:
        if self._model is not None:
            try:
                self._model.cpu()
            except Exception:
                pass
            self._model = None
            self._tokenizer = None


# ── vLLM LoRA scale provider ─────────────────────────────────────────────────


class _VllmVariantProvider(InferenceProvider):
    """Thin InferenceProvider wrapper pairing a shared vLLM engine with one baked LoRA.

    Used internally by :class:`VLLMLoRaScaleProvider` so each scale-variant
    yields an ``InferenceProvider`` that the sweep framework can detect and
    pass directly to :class:`GpuBatchExecutor`.
    """

    def __init__(
        self,
        llm: Any,
        lora_request: Any,
        SamplingParams: Any,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> None:
        self._llm = llm
        self._lora_request = lora_request
        self._SamplingParams = SamplingParams
        self._temperature = temperature
        self._top_p = top_p
        self._max_new_tokens = max_new_tokens

    def generate(self, prompt: PromptInput, **kwargs) -> str:
        """Generate a response for a single prompt."""
        return self.generate_batch([prompt], **kwargs)[0]

    def generate_batch(self, prompts: list[PromptInput], **kwargs) -> list[str]:
        """Generate responses for a batch of prompts via vLLM."""
        sampling_params = self._SamplingParams(
            temperature=kwargs.get("temperature", self._temperature),
            top_p=kwargs.get("top_p", self._top_p),
            max_tokens=kwargs.get("max_new_tokens", self._max_new_tokens),
        )
        formatted = [
            [{"role": "user", "content": p}] if isinstance(p, str) else p
            for p in prompts
        ]
        outputs = self._llm.chat(
            messages=formatted,
            sampling_params=sampling_params,
            lora_request=self._lora_request,
            use_tqdm=False,
        )
        return [out.outputs[0].text for out in outputs]


class VLLMLoRaScaleProvider(ModelProvider):
    """Sweeps LoRA adapter scaling factors using vLLM for high-throughput inference.

    On entry (``__enter__``):
    1. Loads base model + adapter via PEFT and bakes all scale points as
       separate on-disk adapters (skipping any that already exist).
    2. Frees the PEFT model from GPU memory.
    3. Starts a single vLLM engine with ``enable_lora=True``.

    Each ``activate(variant)`` yields a :class:`_VllmVariantProvider` — an
    :class:`InferenceProvider` that wraps the shared engine with the
    per-variant ``LoRARequest``.  The sweep framework detects this and uses
    it directly as the assistant provider (no extra LocalProvider wrapping).

    Args:
        base_model: HuggingFace model ID.
        adapter: Adapter path or ``"repo_id::subfolder"`` reference.
        scale_points: List of scale factors to sweep.
        baked_adapters_dir: Directory where baked adapters are cached on disk.
            Re-runs skip baking if the directory already exists.
        temperature: Sampling temperature for generation.
        top_p: Top-p for generation.
        max_new_tokens: Max tokens to generate per response.
        adapter_name: Internal PEFT adapter name.
        dtype: Torch dtype for both baking and vLLM engine.
        gpu_memory_utilization: vLLM GPU memory fraction (0.0–1.0).
        max_model_len: Optional context length override for the vLLM engine.
        enforce_eager: Disable CUDA graphs (useful for debugging).
    """

    def __init__(
        self,
        base_model: str,
        adapter: str,
        scale_points: list[float],
        baked_adapters_dir: Path,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 128,
        adapter_name: str = "default",
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.90,
        max_model_len: int | None = None,
        enforce_eager: bool = False,
    ) -> None:
        self._base_model = base_model
        self._adapter = adapter
        self._scale_points = sorted(scale_points)
        self._baked_adapters_dir = Path(baked_adapters_dir)
        self._temperature = temperature
        self._top_p = top_p
        self._max_new_tokens = max_new_tokens
        self._adapter_name = adapter_name
        self._dtype = dtype
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._enforce_eager = enforce_eager

        self._llm: Any = None
        self._SamplingParams: Any = None
        self._lora_requests: dict[str, Any] = {}
        self._baked_dirs: dict[str, Path] = {}

    def variant_names(self) -> list[str]:
        return [str(s) for s in self._scale_points]

    def variant_label(self, variant: str) -> str:
        return f"scale_{float(variant):+.2f}"

    def __enter__(self) -> "VLLMLoRaScaleProvider":
        self._bake_all_adapters()
        self._init_vllm_engine()
        return self

    def _bake_all_adapters(self) -> None:
        """Load PEFT model, bake all scale variants to disk, then free GPU memory."""
        import shutil

        from src.utils.lora_baking import bake_lora_scale

        self._baked_adapters_dir.mkdir(parents=True, exist_ok=True)

        # Estimate disk requirement: one adapter size × number of unbaked variants.
        n_to_bake = sum(
            1 for scale in self._scale_points
            if not (self._baked_adapters_dir / self.variant_label(str(scale))).exists()
        )
        if n_to_bake > 0:
            local_adapter_path = _resolve_adapter_to_local(self._adapter)
            adapter_size = sum(
                f.stat().st_size for f in Path(local_adapter_path).rglob("*") if f.is_file()
            )
            required = adapter_size * n_to_bake
            free = shutil.disk_usage(self._baked_adapters_dir).free
            if required > free * 0.9:
                raise RuntimeError(
                    f"Insufficient disk space to bake {n_to_bake} adapter variants: "
                    f"need ~{required / 1e9:.1f}GB, have {free / 1e9:.1f}GB free at "
                    f"{self._baked_adapters_dir}. Consider pointing baked_adapters_dir "
                    f"to a volume with more space."
                )

        model, _tokenizer = _load_peft_model(
            self._base_model, self._adapter, self._adapter_name, self._dtype
        )

        print(
            f"  Baking {len(self._scale_points)} LoRA scale variant(s) "
            f"to {self._baked_adapters_dir} ...",
            flush=True,
        )
        for scale in self._scale_points:
            variant = str(scale)
            out_dir = self._baked_adapters_dir / self.variant_label(variant)
            if out_dir.exists():
                print(
                    f"    {self.variant_label(variant)}: already baked, skipping",
                    flush=True,
                )
            else:
                print(f"    {self.variant_label(variant)}: baking ...", flush=True)
                bake_lora_scale(model, self._adapter_name, scale, out_dir)
            self._baked_dirs[variant] = out_dir

        # Free PEFT model before initialising vLLM to avoid double GPU usage.
        try:
            model.cpu()
        except Exception:
            pass
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _init_vllm_engine(self) -> None:
        """Start the vLLM engine and pre-build all LoRARequests."""
        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
        except ImportError as exc:
            raise ImportError(
                "VLLMLoRaScaleProvider requires 'vllm': pip install vllm"
            ) from exc

        self._SamplingParams = SamplingParams

        engine_kwargs: dict[str, Any] = dict(
            model=self._base_model,
            dtype=self._dtype,
            gpu_memory_utilization=self._gpu_memory_utilization,
            enforce_eager=self._enforce_eager,
            enable_lora=True,
            max_loras=1,  # one adapter per batch; we sweep one variant at a time
            max_lora_rank=64,
            trust_remote_code=False,
        )
        if self._max_model_len is not None:
            engine_kwargs["max_model_len"] = self._max_model_len

        import os
        os.environ.setdefault("VLLM_USE_V1", "0")
        print(f"  Initialising vLLM engine: {self._base_model}", flush=True)
        self._llm = LLM(**engine_kwargs)

        for i, scale in enumerate(self._scale_points, 1):
            variant = str(scale)
            self._lora_requests[variant] = LoRARequest(
                lora_name=self.variant_label(variant),
                lora_int_id=i,
                lora_path=str(self._baked_dirs[variant]),
            )

    @contextmanager
    def activate(self, variant: str) -> Iterator[tuple[Any, None]]:
        """Yield ``(provider, None)`` for the given scale variant.

        The first element is a :class:`_VllmVariantProvider` (an
        :class:`InferenceProvider`), which the sweep framework detects and
        uses directly as the assistant provider.  The tokenizer slot is
        ``None`` because vLLM applies the chat template internally.
        """
        assert self._llm is not None, (
            "VLLMLoRaScaleProvider.__enter__ must be called first"
        )
        provider = _VllmVariantProvider(
            llm=self._llm,
            lora_request=self._lora_requests[variant],
            SamplingParams=self._SamplingParams,
            temperature=self._temperature,
            top_p=self._top_p,
            max_new_tokens=self._max_new_tokens,
        )
        yield (provider, None)

    def close(self) -> None:
        if self._llm is not None:
            try:
                self._llm.llm_engine.engine_core.shutdown()
            except Exception:
                pass
            try:
                del self._llm
            except Exception:
                pass
            self._llm = None
            self._SamplingParams = None
            self._lora_requests.clear()
