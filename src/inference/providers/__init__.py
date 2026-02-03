"""Inference provider implementations."""

from ..base import InferenceProvider
from .api import APIProvider
from .local import LocalProvider

PROVIDERS: dict[str, type[InferenceProvider]] = {
    "local": LocalProvider,
    "api": APIProvider,
}


def get_provider(provider_type: str) -> InferenceProvider:
    """Get an inference provider by type.

    Args:
        provider_type: Type of provider (e.g., "local", "api").

    Returns:
        An instance of the requested provider.

    Raises:
        KeyError: If provider_type is not registered.
    """
    if provider_type not in PROVIDERS:
        raise KeyError(f"Unknown provider type: {provider_type}. Available: {list(PROVIDERS.keys())}")
    return PROVIDERS[provider_type]()
