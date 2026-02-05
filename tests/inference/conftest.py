"""Shared fixtures for inference module tests."""

import pytest


@pytest.fixture
def sample_prompts() -> list[str]:
    """Basic prompts for contract-style tests."""
    return ["Hello", "How are you?", "Tell me a joke."]


@pytest.fixture
def generation_kwargs() -> dict:
    """Common generation kwargs passed through to providers."""
    return {
        "temperature": 0.7,
        "max_new_tokens": 32,
    }
