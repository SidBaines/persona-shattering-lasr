"""Reversible, in-place LoRA modifications for PEFT models.

Limitations
-----------
- Only operates on standard LoRA linear layers (``lora_A`` / ``lora_B``).
  Embedding-based LoRA (``lora_embedding_A`` / ``lora_embedding_B``) and
  convolutional LoRA layers are **not** supported and will be silently
  skipped.  If the adapter targets embedding or output layers, those
  modules will not be modified by any ``BaseLoraModifier`` subclass.
- Layer-index filtering (``layers``, ``layers_to_keep``) relies on module
  names containing ``model.layers.<N>``.  Models that use a different
  naming convention will not match.
- When ``target_modules`` or ``layers`` restrict which modules are modified,
  ``peft_config.r`` is **not** updated (since the model has mixed ranks).
  This means ``save_pretrained`` will write the original rank to
  ``adapter_config.json``, which may cause shape mismatches on reload.
  Update ``model.peft_config[adapter].r`` manually if needed.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from peft import PeftModel
from torch import Tensor, nn

from src.utils.linalg import reduce_lora_rank_efficient

# ---------------------------------------------------------------------------
# Saved state dataclasses
# ---------------------------------------------------------------------------


@dataclass
class _SavedModuleState:
    """Snapshot of a single LoRA module's mutable state.

    PEFT's ``LoraLayer`` stores per-adapter rank and scaling alongside the
    weight tensors.  We capture all four so that ``restore()`` can return
    the module to its exact original configuration.
    """

    weight_A: Tensor  # lora_A weight, shape (r, in_features)
    weight_B: Tensor  # lora_B weight, shape (out_features, r)
    r: int  # per-module LoRA rank (module.r[adapter_name])
    scaling: float  # per-module scaling factor (module.scaling[adapter_name])


@dataclass
class _SavedLoraState:
    """Snapshot of all LoRA modules plus the model-level config rank.

    ``peft_config_r`` is the rank stored in ``model.peft_config[adapter_name].r``,
    which is written to ``adapter_config.json`` by ``save_pretrained``.  It must
    stay in sync with the actual weight shapes for save/load to work.
    """

    modules: dict[str, _SavedModuleState] = field(default_factory=dict)
    peft_config_r: int = 0


# ---------------------------------------------------------------------------
# Public utils
# ---------------------------------------------------------------------------


def set_active_adapters(
    model: PeftModel, adapter_names: str | list[str]
) -> None:
    """Activate one or more adapters for the forward pass.

    ``PeftModel.set_adapter`` only accepts a single adapter name.  This
    helper routes through ``model.base_model.set_adapter`` which accepts
    both a string and a list, enabling multi-adapter inference.

    Parameters
    ----------
    model:
        A PEFT model.
    adapter_names:
        A single adapter name or a list of adapter names to activate.
    """
    model.base_model.set_adapter(adapter_names)


def get_active_adapters(model: PeftModel) -> list[str]:
    """Return the list of currently active adapter names."""
    return list(model.base_model.active_adapters)


# ---------------------------------------------------------------------------
# Internal utils
# ---------------------------------------------------------------------------


def _snapshot_adapter(
    model: PeftModel, adapter_name: str, module_names: set[str] | None = None
) -> _SavedLoraState:
    """Snapshot LoRA modules for a given adapter.

    Parameters
    ----------
    model:
        The PEFT model.
    adapter_name:
        Which adapter to snapshot.
    module_names:
        If provided, only snapshot modules whose names are in this set.
        If ``None``, snapshot all modules for the adapter.

    Returns
    -------
    _SavedLoraState
        Snapshot containing cloned weights and metadata.
    """
    saved = _SavedLoraState()
    saved.peft_config_r = model.peft_config[adapter_name].r

    for name, module in model.named_modules():
        if not (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and adapter_name in module.lora_A
        ):
            continue

        # Skip if module_names filter is provided and this module isn't in it
        if module_names is not None and name not in module_names:
            continue

        saved.modules[name] = _SavedModuleState(
            weight_A=module.lora_A[adapter_name].weight.data.clone(),
            weight_B=module.lora_B[adapter_name].weight.data.clone(),
            r=module.r[adapter_name],
            scaling=module.scaling[adapter_name],
        )
    return saved


def _restore_adapter(
    model: PeftModel, adapter_name: str, saved: _SavedLoraState
) -> None:
    """Restore all LoRA modules for a given adapter from a snapshot.

    Updates ``model`` in-place to match the state in ``saved``.
    """
    model.peft_config[adapter_name].r = saved.peft_config_r

    for name, module in model.named_modules():
        if not (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and adapter_name in module.lora_A
        ):
            continue

        if name not in saved.modules:
            continue

        ms = saved.modules[name]
        module.r[adapter_name] = ms.r
        module.scaling[adapter_name] = ms.scaling

        module.lora_A[adapter_name].weight = nn.Parameter(
            ms.weight_A.clone().to(
                device=module.lora_A[adapter_name].weight.device,
                dtype=module.lora_A[adapter_name].weight.dtype,
            )
        )
        module.lora_B[adapter_name].weight = nn.Parameter(
            ms.weight_B.clone().to(
                device=module.lora_B[adapter_name].weight.device,
                dtype=module.lora_B[adapter_name].weight.dtype,
            )
        )


def _extract_layer_idx(name: str) -> int | None:
    """Extract the layer index from a module name like ``...model.layers.3...``.

    Returns ``None`` if the name doesn't contain ``model.layers.``.
    """
    if "model.layers." not in name:
        return None
    return int(name.split("model.layers.")[-1].split(".")[0])


def _matches_target_modules(name: str, target_modules: list[str]) -> bool:
    """Check if *name* ends with any of the *target_modules* strings.

    For example, name ``"base_model.model.model.layers.0.self_attn.q_proj"``
    matches ``target_modules=["q_proj"]``.
    """
    return any(name.endswith(t) for t in target_modules)


# ---------------------------------------------------------------------------
# Modifier base class & subclasses
# ---------------------------------------------------------------------------


class BaseLoRaModifier(ABC):
    """Base class for reversible, in-place LoRA modifications.

    Captures original LoRA state at init time.  ``apply()`` always restores
    from the original snapshot first so that repeated calls are idempotent
    (they do not compound).

    Parameters
    ----------
    model:
        A PEFT model with at least one LoRA adapter.
    adapter_name:
        Which adapter to modify (e.g. ``"default"``).
    target_modules:
        Controls which modules are affected. Can be:

        - ``None``: All LoRA modules for the adapter
        - ``list[str]``: Modules whose name ends with any of these strings
          (e.g. ``["q_proj", "v_proj"]``)
        - ``dict[int, list[str]]``: Per-layer specification mapping layer
          indices to module name suffixes (e.g. ``{1: ["q_proj"], 6: ["up_proj"]}``).
          When dict is provided, ``layers`` parameter must be ``None``.
    layers:
        If provided, only modules in these layer indices are affected.
        Layer indices are extracted from ``model.layers.<N>`` in the module
        name. ``None`` means all layers. Cannot be used with dict-style
        ``target_modules``.
    """

    def __init__(
        self,
        model: PeftModel,
        adapter_name: str,
        target_modules: list[str] | dict[int, list[str]] | None = None,
        layers: list[int] | None = None,
    ) -> None:
        if isinstance(target_modules, dict) and layers is not None:
            raise ValueError(
                "Cannot specify both target_modules as a dict and layers parameter. "
                "When using dict format for target_modules, layer indices are "
                "specified as dict keys."
            )

        self._model = model
        self._adapter_name = adapter_name
        self._target_modules = target_modules
        self._layers = set(layers) if layers is not None else None
        self._is_applied = False
        self._saved = self._snapshot_state()

    @property
    def is_applied(self) -> bool:
        return self._is_applied

    def _iter_lora_modules(self) -> list[tuple[str, nn.Module]]:
        """Return ``(name, module)`` pairs matching the adapter, target_modules, and layers filters."""
        result = []

        for name, module in self._model.named_modules():
            # Must be a LoRA module with the target adapter
            if not (
                hasattr(module, "lora_A")
                and hasattr(module, "lora_B")
                and self._adapter_name in module.lora_A
            ):
                continue

            # Extract layer index (needed for filtering)
            layer_idx = _extract_layer_idx(name)

            # Apply filtering based on target_modules type
            if isinstance(self._target_modules, dict):
                # Dict mode: only modules in specified layers with specified targets
                if layer_idx is None or layer_idx not in self._target_modules:
                    continue
                if not _matches_target_modules(name, self._target_modules[layer_idx]):
                    continue
            else:
                # List/None mode: apply target_modules and layers filters independently
                if self._target_modules is not None:
                    if not _matches_target_modules(name, self._target_modules):
                        continue
                if self._layers is not None:
                    if layer_idx is None or layer_idx not in self._layers:
                        continue

            result.append((name, module))
        return result

    def _snapshot_state(self) -> _SavedLoraState:
        """Snapshot the current state of filtered modules."""
        module_names = {name for name, _ in self._iter_lora_modules()}
        return _snapshot_adapter(self._model, self._adapter_name, module_names)

    def apply(self) -> None:
        """Restore to original state, then apply the modification.

        Safe to call multiple times — always starts from the original snapshot.
        Warns if the target adapter is not currently active.
        """
        self._warn_if_inactive()
        self.restore()
        self._apply_to_modules()
        self._is_applied = True

    def _warn_if_inactive(self) -> None:
        """Emit a warning if the target adapter is not currently active."""
        active = get_active_adapters(self._model)
        if self._adapter_name not in active:
            warnings.warn(
                f"Adapter '{self._adapter_name}' is not currently active "
                f"(active: {active}). Modifications will not affect the "
                f"forward pass until this adapter is activated via "
                f"set_active_adapters().",
                stacklevel=3,
            )

    def restore(self) -> None:
        """Restore model to the state it was in when this modifier was created."""
        _restore_adapter(self._model, self._adapter_name, self._saved)
        self._is_applied = False

    @abstractmethod
    def _apply_to_modules(self) -> None:
        """Subclass implements the actual modification logic."""
        ...

    @property
    def _modifies_all_modules(self) -> bool:
        """True if no target_modules or layers filter is active."""
        return self._target_modules is None and self._layers is None


class LoRaRankReducer(BaseLoRaModifier):
    """Reduces LoRA rank via truncated-SVD approximation.

    After ``apply()``, each affected LoRA module's ``lora_A`` has shape
    ``(new_rank, in_features)`` and ``lora_B`` has shape
    ``(out_features, new_rank)``.  The product ``new_B @ new_A`` approximates
    the original ``B @ A``.

    Scaling is **not** recomputed — the SVD approximation already preserves
    the trained output magnitude.  ``peft_config.r`` and per-module
    ``module.r`` are updated only when all modules are affected (no
    ``target_modules`` or ``layers`` filter).  When filters are active,
    ``peft_config.r`` is left unchanged and a warning is emitted.
    """

    def __init__(
        self,
        model: PeftModel,
        adapter_name: str,
        new_rank: int,
        target_modules: list[str] | dict[int, list[str]] | None = None,
        layers: list[int] | None = None,
    ) -> None:
        super().__init__(
            model, adapter_name, target_modules=target_modules, layers=layers
        )
        if new_rank < 1:
            raise ValueError(f"new_rank must be >= 1, got {new_rank}")
        self._new_rank = new_rank

    def _apply_to_modules(self) -> None:
        adapter = self._adapter_name
        new_rank = self._new_rank

        if self._modifies_all_modules:
            self._model.peft_config[adapter].r = new_rank
        else:
            warnings.warn(
                "target_modules or layers filter is active — peft_config.r is "
                "not updated.  The model will have mixed ranks across modules, "
                "which may cause issues with save_pretrained / from_pretrained.",
                stacklevel=2,
            )

        for _name, module in self._iter_lora_modules():
            A = module.lora_A[adapter].weight
            B = module.lora_B[adapter].weight

            new_A, new_B = reduce_lora_rank_efficient(A, B, new_rank=new_rank)
            module.lora_A[adapter].weight = nn.Parameter(new_A)
            module.lora_B[adapter].weight = nn.Parameter(new_B)
            module.r[adapter] = new_rank


class LoRaScaling(BaseLoRaModifier):
    """Scales the LoRA contribution by modifying ``module.scaling[adapter]``.

    The scaling factor is applied multiplicatively on top of the existing
    scaling value: ``new_scaling = original_scaling * scale_factor``.  This
    means ``scale_factor=1.0`` is a no-op, ``0.0`` disables the adapter,
    and negative values invert the LoRA direction.

    Use ``target_modules`` and ``layers`` to apply scaling to specific
    matrices or layers only.  For multi-adapter scaling, create one
    ``LoraScaling`` instance per adapter.
    """

    def __init__(
        self,
        model: PeftModel,
        adapter_name: str,
        scale_factor: float,
        target_modules: list[str] | dict[int, list[str]] | None = None,
        layers: list[int] | None = None,
    ) -> None:
        super().__init__(
            model, adapter_name, target_modules=target_modules, layers=layers
        )
        self._scale_factor = scale_factor

    def _apply_to_modules(self) -> None:
        adapter = self._adapter_name

        for _name, module in self._iter_lora_modules():
            # The saved snapshot has the original scaling; after
            # _restore_from() in apply(), module.scaling is back to original.
            # We multiply by scale_factor to get the desired new scaling.
            module.scaling[adapter] = module.scaling[adapter] * self._scale_factor


class LoRaAdapterZeroing(LoRaScaling):
    """Zeros out the LoRA contribution for specific layers and modules.

    This is a convenience wrapper around ``LoraScaling`` that sets the
    scaling factor to 0.0 for the targeted modules.
    """

    def __init__(
        self,
        model: PeftModel,
        adapter_name: str,
        target_modules: list[str] | dict[int, list[str]] | None = None,
        layers: list[int] | None = None,
    ) -> None:
        # Standardize: 'layers' now defines what to zero, matching BaseLoraModifier.
        super().__init__(
            model=model,
            adapter_name=adapter_name,
            scale_factor=0.0,
            target_modules=target_modules,
            layers=layers,
        )


# ---------------------------------------------------------------------------
# Pipeline for composing multiple modifications
# ---------------------------------------------------------------------------


class LoRaPipeline:
    """Applies multiple LoRA modifications in sequence.

    A pipeline can apply changes to one or more adapters. Each step is applied
    in order, with modifications compounding. The pipeline snapshots the
    original state at init, and ``restore()`` returns to that state.

    Unlike individual modifiers, a pipeline can modify different adapters in
    different steps, enabling complex multi-adapter transformations.

    Parameters
    ----------
    model:
        A PEFT model with LoRA adapters.
    steps:
        List of ``(modifier_class, adapter_name, kwargs)`` tuples defining
        the pipeline. The kwargs should not include ``model`` or ``adapter_name``
        (those are provided automatically).

    Examples
    --------
    Apply multiple modifications to a single adapter::

        pipeline = LoraPipeline(
            model,
            steps=[
                (LoraRankReducer, "default", {"new_rank": 4, "layers": [0, 1]}),
                (LoraScaling, "default", {"scale_factor": 0.5}),
            ]
        )
        pipeline.apply()  # Rank reduced first, then scaled
        pipeline.restore()  # Returns to original state

    Modify different adapters in a single pipeline::

        pipeline = LoraPipeline(
            model,
            steps=[
                (LoraRankReducer, "adapter_a", {"new_rank": 2}),
                (LoraScaling, "adapter_b", {"scale_factor": 0.1}),
                (LoraLayerZeroing, "adapter_a", {"layers": [0]}),
            ]
        )

    Complex per-module transformations::

        pipeline = LoraPipeline(
            model,
            steps=[
                (LoraRankReducer, "default", {
                    "new_rank": 2,
                    "target_modules": {0: ["q_proj"], 1: ["v_proj"]}
                }),
                (LoraScaling, "default", {
                    "scale_factor": 2.0,
                    "target_modules": ["up_proj", "down_proj"]
                }),
            ]
        )
    """

    def __init__(
        self,
        model: PeftModel,
        steps: list[tuple[type[BaseLoRaModifier], str, dict]],
    ) -> None:
        if not steps:
            raise ValueError("Pipeline must have at least one step")

        # Validate steps
        for modifier_class, adapter_name, kwargs in steps:
            if not issubclass(modifier_class, BaseLoRaModifier):
                raise TypeError(
                    f"{modifier_class} is not a subclass of BaseLoraModifier"
                )
            if "model" in kwargs or "adapter_name" in kwargs:
                raise ValueError(
                    "Step kwargs should not include 'model' or 'adapter_name' "
                    "(these are provided automatically)"
                )

        self._model = model
        self._steps = steps
        self._is_applied = False

        # Snapshot each unique adapter once at init
        adapters = {adapter_name for _, adapter_name, _ in steps}
        self._snapshots: dict[str, _SavedLoraState] = {
            adapter_name: _snapshot_adapter(model, adapter_name)
            for adapter_name in adapters
        }

    @property
    def is_applied(self) -> bool:
        return self._is_applied

    def apply(self) -> None:
        """Apply all pipeline steps in sequence.

        For steps affecting the same adapter, modifications compound. The
        pipeline first restores all affected adapters to their original state,
        then applies each modification in order.
        """
        # Restore all affected adapters to original state
        self.restore()

        # Apply each step in order.  Each modifier snapshots the current
        # state at creation time, so apply() restores to that snapshot
        # (a no-op) then applies the modification on top.
        for modifier_class, adapter_name, kwargs in self._steps:
            modifier = modifier_class(self._model, adapter_name, **kwargs)
            modifier.apply()

        self._is_applied = True

    def restore(self) -> None:
        """Restore all adapters to their original states."""
        for adapter_name, saved_state in self._snapshots.items():
            _restore_adapter(self._model, adapter_name, saved_state)

        self._is_applied = False
