"""Tests for browser local chat runtime adapter behavior."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from scripts.visualisations.local_chat_web.runtime import LocalChatRuntime, RuntimeDependencies
from scripts.visualisations.local_chat_web.types import BrowserChatConfig, ChatAdapterConfig


class FakeTokenizer:
    pad_token = None
    eos_token = "</s>"
    eos_token_id = 2
    pad_token_id = 2
    padding_side = "left"

    def add_special_tokens(self, _tokens) -> None:
        self.pad_token = "[PAD]"
        self.pad_token_id = 0


class FakeBaseModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(pad_token_id=None)
        self.generation_config = SimpleNamespace(eos_token_id=2)
        self.device = torch.device("cpu")

    def eval(self):
        return self


@dataclass
class FakeLoraModule:
    scaling: dict[str, float]


class FakeAdapterContainer:
    def __init__(self) -> None:
        self.active = None

    def set_adapter(self, adapter_names):
        self.active = adapter_names


class FakePeftModel(FakeBaseModel):
    def __init__(self, first_adapter_name: str) -> None:
        super().__init__()
        self.base_model = FakeAdapterContainer()
        self.lora = FakeLoraModule(scaling={first_adapter_name: 2.0})
        self.peft_config = {first_adapter_name: object()}
        self.disabled = False

    def named_modules(self):
        return [("", self), ("lora", self.lora)]

    def load_adapter(self, _resolved_ref: str, adapter_name: str, subfolder: str | None = None) -> None:
        del subfolder
        self.peft_config[adapter_name] = object()
        self.lora.scaling[adapter_name] = 2.0

    def disable_adapter_layers(self) -> None:
        self.disabled = True

    def enable_adapter_layers(self) -> None:
        self.disabled = False


class FakeDepsFactory:
    def __init__(self) -> None:
        self.first_calls = 0
        self.extra_calls = 0

    def make(self) -> RuntimeDependencies:
        def model_loader(_base: str, _dtype: torch.dtype, _device_map: str):
            return FakeBaseModel()

        def tokenizer_loader(_base: str):
            return FakeTokenizer()

        def first_loader(_base_model, _resolved_ref: str, adapter_name: str, _subfolder: str | None):
            self.first_calls += 1
            return FakePeftModel(first_adapter_name=adapter_name)

        def extra_loader(model: FakePeftModel, _resolved_ref: str, adapter_name: str, _subfolder: str | None):
            self.extra_calls += 1
            model.load_adapter("unused", adapter_name=adapter_name)

        def set_active(model: FakePeftModel, adapter_names):
            model.base_model.set_adapter(adapter_names)

        return RuntimeDependencies(
            model_loader=model_loader,
            tokenizer_loader=tokenizer_loader,
            first_adapter_loader=first_loader,
            extra_adapter_loader=extra_loader,
            adapter_resolver=lambda ref: ref,
            base_resolver=lambda ref: ref,
            active_adapter_setter=set_active,
        )


def _runtime(factory: FakeDepsFactory) -> LocalChatRuntime:
    return LocalChatRuntime(
        BrowserChatConfig(base_model="hf://org/base-model"),
        deps=factory.make(),
    )


def test_adapter_loading_is_lazy_and_cached() -> None:
    factory = FakeDepsFactory()
    runtime = _runtime(factory)

    runtime.apply_adapter_configuration(
        [ChatAdapterConfig(key="adapter_a", path="hf://org/a", scale=1.0)]
    )
    runtime.apply_adapter_configuration(
        [ChatAdapterConfig(key="adapter_a", path="hf://org/a", scale=0.5)]
    )
    runtime.apply_adapter_configuration(
        [
            ChatAdapterConfig(key="adapter_a", path="hf://org/a", scale=0.5),
            ChatAdapterConfig(key="adapter_b", path="hf://org/b", scale=1.2),
        ]
    )

    assert factory.first_calls == 1
    assert factory.extra_calls == 1


def test_adapter_scaling_is_non_compounding_across_updates() -> None:
    factory = FakeDepsFactory()
    runtime = _runtime(factory)

    runtime.apply_adapter_configuration(
        [ChatAdapterConfig(key="adapter_a", path="hf://org/a", scale=0.5)]
    )
    adapter_name = runtime._adapter_name_by_key["adapter_a"]  # pylint: disable=protected-access
    lora_module = runtime._peft_model.lora  # pylint: disable=protected-access
    assert lora_module.scaling[adapter_name] == 1.0

    runtime.apply_adapter_configuration(
        [ChatAdapterConfig(key="adapter_a", path="hf://org/a", scale=2.0)]
    )
    assert lora_module.scaling[adapter_name] == 4.0

    runtime.apply_adapter_configuration([])
    assert runtime._peft_model.disabled is True  # pylint: disable=protected-access
