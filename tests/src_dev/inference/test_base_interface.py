"""Tests for the InferenceProvider abstract base class."""

import inspect
import pytest

from src.inference import InferenceProvider


class _MissingGenerate(InferenceProvider):
    def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
        return ["stub" for _ in prompts]

    def load_model(self, config: dict) -> None:
        return None


class _MissingGenerateBatch(InferenceProvider):
    def generate(self, prompt: str, **kwargs) -> str:
        return "stub"

    def load_model(self, config: dict) -> None:
        return None


class _MissingLoadModel(InferenceProvider):
    def generate(self, prompt: str, **kwargs) -> str:
        return "stub"

    def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
        return ["stub" for _ in prompts]


def test_inference_provider_is_abstract():
    """InferenceProvider should not be instantiable."""
    with pytest.raises(TypeError):
        InferenceProvider()  # type: ignore[abstract]


def test_missing_generate_is_abstract():
    """Subclass missing generate() should be abstract."""
    with pytest.raises(TypeError):
        _MissingGenerate()  # type: ignore[abstract]


def test_missing_generate_batch_is_abstract():
    """Subclass missing generate_batch() should be abstract."""
    with pytest.raises(TypeError):
        _MissingGenerateBatch()  # type: ignore[abstract]


def test_missing_load_model_is_abstract():
    """Subclass missing load_model() should be abstract."""
    with pytest.raises(TypeError):
        _MissingLoadModel()  # type: ignore[abstract]


def test_method_signatures():
    """Required methods should accept expected parameters."""
    generate_sig = inspect.signature(InferenceProvider.generate)
    generate_params = list(generate_sig.parameters.values())
    assert generate_params[1].name == "prompt"
    assert generate_params[-1].kind == inspect.Parameter.VAR_KEYWORD

    batch_sig = inspect.signature(InferenceProvider.generate_batch)
    batch_params = list(batch_sig.parameters.values())
    assert batch_params[1].name == "prompts"
    assert batch_params[-1].kind == inspect.Parameter.VAR_KEYWORD

    load_sig = inspect.signature(InferenceProvider.load_model)
    load_params = list(load_sig.parameters.values())
    assert load_params[1].name == "config"
