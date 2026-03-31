"""Tests for OpenRouter-specific routing normalization."""

from src_dev.inference.providers.openrouter import _normalize_provider_routing


def test_normalize_provider_routing_converts_legacy_quantized_slug() -> None:
    """Legacy `provider/quantization` entries should map to current fields."""
    routing = {
        "order": ["DeepInfra/bf16"],
        "allowFallbacks": False,
    }

    assert _normalize_provider_routing(routing) == {
        "order": ["deepinfra"],
        "allow_fallbacks": False,
        "quantizations": ["bf16"],
    }


def test_normalize_provider_routing_preserves_endpoint_variants() -> None:
    """Endpoint slugs like `/turbo` should remain provider slugs, not quantizations."""
    routing = {
        "only": ["DeepInfra/turbo", "google-vertex/us-east5"],
        "allow_fallbacks": False,
    }

    assert _normalize_provider_routing(routing) == {
        "only": ["deepinfra/turbo", "google-vertex/us-east5"],
        "allow_fallbacks": False,
    }


def test_normalize_provider_routing_merges_quantizations_without_duplicates() -> None:
    """Explicit quantizations should merge cleanly with extracted legacy suffixes."""
    routing = {
        "only": ["DeepInfra/bf16", "Fireworks"],
        "quantizations": ["bf16", "fp8"],
    }

    assert _normalize_provider_routing(routing) == {
        "only": ["deepinfra", "fireworks"],
        "quantizations": ["bf16", "fp8"],
    }
