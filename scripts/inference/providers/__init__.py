"""Inference provider registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.inference.config import InferenceConfig
    from scripts.inference.providers.base import InferenceProvider


def get_provider(name: str, config: "InferenceConfig") -> "InferenceProvider":
    """Get an inference provider by name.

    Args:
        name: Provider name ("local" or "openai").
        config: Inference configuration.

    Returns:
        Initialized inference provider.

    Raises:
        ValueError: If provider name is unknown.
    """
    # Lazy imports to avoid circular dependencies
    if name == "local":
        from scripts.inference.providers.local import LocalProvider

        return LocalProvider(config)
    elif name == "openai":
        from scripts.inference.providers.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(config)
    else:
        raise ValueError(f"Unknown inference provider: {name!r}. Available: ['local', 'openai']")
