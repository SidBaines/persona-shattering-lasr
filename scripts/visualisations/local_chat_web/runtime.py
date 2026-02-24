"""Runtime for browser-based local chat with dynamic PEFT adapters."""

from __future__ import annotations

import gc
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.evals.model_resolution import resolve_model_reference
from scripts.utils.lora_composition import split_adapter_reference, resolve_torch_dtype
from scripts.visualisations.local_chat_web.prompting import (
    build_prompt,
    effective_system_prompt,
    resolve_prompt_format,
    window_chat_turns,
)
from scripts.visualisations.local_chat_web.types import (
    BrowserChatConfig,
    ChatAdapterConfig,
    ChatTurn,
    GenerationSettings,
)
from src.utils.peft_manipulations import set_active_adapters


@dataclass(frozen=True)
class RuntimeDependencies:
    """Dependency injection hooks for runtime unit tests."""

    model_loader: Callable[[str, torch.dtype, str], object]
    tokenizer_loader: Callable[[str], object]
    first_adapter_loader: Callable[[object, str, str, str | None], PeftModel]
    extra_adapter_loader: Callable[[PeftModel, str, str, str | None], None]
    adapter_resolver: Callable[[str], str]
    base_resolver: Callable[[str], str]
    active_adapter_setter: Callable[[PeftModel, str | list[str]], None]


def _default_model_loader(resolved_base_model: str, torch_dtype: torch.dtype, device_map: str):
    return AutoModelForCausalLM.from_pretrained(
        resolved_base_model,
        dtype=torch_dtype,
        device_map=device_map,
    )


def _default_tokenizer_loader(resolved_base_model: str):
    return AutoTokenizer.from_pretrained(resolved_base_model, use_fast=True)


def _has_missing_adapter_config_error(exc: Exception) -> bool:
    return "adapter_config.json" in str(exc)


def _default_first_adapter_loader(
    base_model,
    resolved_ref: str,
    adapter_name: str,
    subfolder: str | None,
) -> PeftModel:
    if subfolder:
        return PeftModel.from_pretrained(
            base_model,
            resolved_ref,
            adapter_name=adapter_name,
            subfolder=subfolder,
        )

    try:
        return PeftModel.from_pretrained(
            base_model,
            resolved_ref,
            adapter_name=adapter_name,
        )
    except ValueError as exc:
        if not _has_missing_adapter_config_error(exc):
            raise
        return PeftModel.from_pretrained(
            base_model,
            resolved_ref,
            adapter_name=adapter_name,
            subfolder="adapter",
        )


def _default_extra_adapter_loader(
    model: PeftModel,
    resolved_ref: str,
    adapter_name: str,
    subfolder: str | None,
) -> None:
    if subfolder:
        model.load_adapter(
            resolved_ref,
            adapter_name=adapter_name,
            subfolder=subfolder,
        )
        return

    try:
        model.load_adapter(resolved_ref, adapter_name=adapter_name)
    except ValueError as exc:
        if not _has_missing_adapter_config_error(exc):
            raise
        model.load_adapter(
            resolved_ref,
            adapter_name=adapter_name,
            subfolder="adapter",
        )


def _default_dependencies() -> RuntimeDependencies:
    return RuntimeDependencies(
        model_loader=_default_model_loader,
        tokenizer_loader=_default_tokenizer_loader,
        first_adapter_loader=_default_first_adapter_loader,
        extra_adapter_loader=_default_extra_adapter_loader,
        adapter_resolver=lambda ref: resolve_model_reference(ref, kind="adapter"),
        base_resolver=lambda ref: resolve_model_reference(ref, kind="base model"),
        active_adapter_setter=set_active_adapters,
    )


def cleanup_runtime_cache() -> None:
    """Release CPU/GPU memory from previous generation state."""
    gc.collect()
    gc.collect()
    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()


def infer_base_model_from_adapter(adapter_path: str) -> str | None:
    """Infer adapter base model from adapter metadata when possible."""
    ref, subfolder = split_adapter_reference(adapter_path)

    try:
        resolved_ref = resolve_model_reference(ref, kind="adapter")
    except Exception:
        resolved_ref = ref

    peft_kwargs = {"subfolder": subfolder} if subfolder else {}
    try:
        peft_cfg = PeftConfig.from_pretrained(str(resolved_ref), **peft_kwargs)
        base_model = getattr(peft_cfg, "base_model_name_or_path", None)
        if isinstance(base_model, str) and base_model.strip():
            return base_model
    except Exception:
        pass

    local_ref = Path(resolved_ref)
    if not local_ref.exists():
        return None

    cfg_path = local_ref / "adapter_config.json"
    if subfolder:
        cfg_path = local_ref / subfolder / "adapter_config.json"

    if not cfg_path.exists():
        return None

    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    base_model = data.get("base_model_name_or_path")
    if isinstance(base_model, str) and base_model.strip():
        return base_model
    return None


class LocalChatRuntime:
    """Single-process model runtime shared across browser chat sessions."""

    def __init__(
        self,
        config: BrowserChatConfig,
        *,
        deps: RuntimeDependencies | None = None,
    ) -> None:
        self.config = config
        self._deps = deps or _default_dependencies()
        self._lock = threading.RLock()

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        resolved_base_model = self._deps.base_resolver(self.config.base_model)
        torch_dtype = resolve_torch_dtype(self.config.dtype)

        self._base_model = self._deps.model_loader(
            resolved_base_model,
            torch_dtype,
            self.config.device_map,
        )

        self.tokenizer = self._deps.tokenizer_loader(resolved_base_model)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                resize = getattr(self._base_model, "resize_token_embeddings", None)
                if callable(resize):
                    resize(len(self.tokenizer))

        if hasattr(self._base_model, "config"):
            self._base_model.config.pad_token_id = self.tokenizer.pad_token_id

        eval_fn = getattr(self._base_model, "eval", None)
        if callable(eval_fn):
            eval_fn()

        self._peft_model: PeftModel | None = None
        self._adapter_name_by_key: dict[str, str] = {}
        self._adapter_original_scaling: dict[str, dict[str, float]] = {}
        self._next_adapter_idx = 0

        self.prompt_format = resolve_prompt_format(self.tokenizer, self.config.prompt_format)
        self.eos_token_id = self._resolve_eos_token_id()

    def close(self) -> None:
        """Run runtime cleanup after shutdown."""
        cleanup_runtime_cache()

    def _resolve_eos_token_id(self) -> int | list[int] | None:
        eos_ids: list[int] = []

        model = self._runtime_model
        generation_config = getattr(model, "generation_config", None)
        model_eos = getattr(generation_config, "eos_token_id", None)
        if isinstance(model_eos, int):
            eos_ids.append(model_eos)
        elif isinstance(model_eos, list):
            eos_ids.extend(int(token_id) for token_id in model_eos)

        tokenizer_eos = getattr(self.tokenizer, "eos_token_id", None)
        if tokenizer_eos is not None:
            eos_ids.append(int(tokenizer_eos))

        eos_ids = list(dict.fromkeys(eos_ids))
        if not eos_ids:
            return None
        if len(eos_ids) == 1:
            return eos_ids[0]
        return eos_ids

    @property
    def _runtime_model(self):
        return self._peft_model if self._peft_model is not None else self._base_model

    def _runtime_device(self):
        model = self._runtime_model
        device = getattr(model, "device", None)
        if device is not None:
            return device

        parameters = getattr(model, "parameters", None)
        if callable(parameters):
            try:
                return next(parameters()).device
            except Exception:
                pass

        return torch.device("cpu")

    def _snapshot_adapter_scaling_unlocked(self, adapter_name: str) -> None:
        if self._peft_model is None:
            return

        module_scaling: dict[str, float] = {}
        for module_name, module in self._peft_model.named_modules():
            scaling = getattr(module, "scaling", None)
            if isinstance(scaling, dict) and adapter_name in scaling:
                module_scaling[module_name] = float(scaling[adapter_name])

        self._adapter_original_scaling[adapter_name] = module_scaling

    def _apply_adapter_scale_unlocked(self, adapter_name: str, scale: float) -> None:
        if self._peft_model is None:
            return

        original = self._adapter_original_scaling.get(adapter_name, {})
        for module_name, module in self._peft_model.named_modules():
            scaling = getattr(module, "scaling", None)
            if not isinstance(scaling, dict):
                continue
            if adapter_name not in scaling:
                continue
            base_scale = original.get(module_name, float(scaling[adapter_name]))
            scaling[adapter_name] = base_scale * float(scale)

    def _ensure_adapter_loaded_unlocked(self, adapter: ChatAdapterConfig) -> None:
        if adapter.key in self._adapter_name_by_key:
            return

        ref, subfolder = split_adapter_reference(adapter.path)
        resolved_ref = self._deps.adapter_resolver(ref)
        adapter_name = f"chat_adapter_{self._next_adapter_idx}"

        if self._peft_model is None:
            self._peft_model = self._deps.first_adapter_loader(
                self._base_model,
                resolved_ref,
                adapter_name,
                subfolder,
            )
        else:
            self._deps.extra_adapter_loader(
                self._peft_model,
                resolved_ref,
                adapter_name,
                subfolder,
            )

        self._adapter_name_by_key[adapter.key] = adapter_name
        self._next_adapter_idx += 1

        eval_fn = getattr(self._peft_model, "eval", None)
        if callable(eval_fn):
            eval_fn()

        self._snapshot_adapter_scaling_unlocked(adapter_name)

    def _disable_all_adapters_unlocked(self) -> None:
        if self._peft_model is None:
            return

        disable = getattr(self._peft_model, "disable_adapter_layers", None)
        if callable(disable):
            disable()
            return

        if not self._adapter_name_by_key:
            return

        for adapter_name in self._adapter_name_by_key.values():
            self._apply_adapter_scale_unlocked(adapter_name, 0.0)

        self._deps.active_adapter_setter(
            self._peft_model,
            list(self._adapter_name_by_key.values()),
        )

    def apply_adapter_configuration(self, adapters: Sequence[ChatAdapterConfig]) -> None:
        """Apply adapter set and scales for the next generations."""
        with self._lock:
            if not adapters:
                self._disable_all_adapters_unlocked()
                return

            for adapter in adapters:
                self._ensure_adapter_loaded_unlocked(adapter)

            if self._peft_model is None:
                return

            enable = getattr(self._peft_model, "enable_adapter_layers", None)
            if callable(enable):
                enable()

            active_adapter_names: list[str] = []
            for adapter in adapters:
                adapter_name = self._adapter_name_by_key[adapter.key]
                self._apply_adapter_scale_unlocked(adapter_name, adapter.scale)
                active_adapter_names.append(adapter_name)

            self._deps.active_adapter_setter(self._peft_model, active_adapter_names)

    def generate_reply(
        self,
        *,
        turns: list[ChatTurn],
        user_text: str,
        generation_settings: GenerationSettings,
        adapters: Sequence[ChatAdapterConfig],
    ) -> str:
        """Generate one assistant response for the active chat."""
        with self._lock:
            self.apply_adapter_configuration(adapters)

            prompt_turns = window_chat_turns(turns, self.config.history_window)
            system_prompt = effective_system_prompt(self.config.system_prompt, self.config.tone)
            prompt = build_prompt(
                self.tokenizer,
                prompt_format=self.prompt_format,
                system_prompt=system_prompt,
                turns=prompt_turns,
                pending_user_text=user_text,
            )

            model = self._runtime_model
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self._runtime_device())
            do_sample = not math.isclose(generation_settings.temperature, 0.0)

            generation_kwargs = {
                **inputs,
                "max_new_tokens": generation_settings.max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.eos_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = generation_settings.temperature
                generation_kwargs["top_p"] = generation_settings.top_p

            with torch.no_grad():
                generated = model.generate(**generation_kwargs)

            input_len = inputs["input_ids"].shape[1]
            reply = self.tokenizer.decode(
                generated[0][input_len:],
                skip_special_tokens=True,
            ).strip()
            return reply
