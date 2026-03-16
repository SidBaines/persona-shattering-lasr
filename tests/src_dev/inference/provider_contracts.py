"""Provider contract helpers.

These helpers check minimal interface behavior without asserting provider-specific
implementation details. They are intended for future provider tests.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Any


def assert_basic_provider_contract(
    provider: Any,
    prompts: Iterable[str],
    generation_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Assert the minimal provider contract.

    Args:
        provider: Inference provider instance under test.
        prompts: Iterable of prompt strings.
        generation_kwargs: Optional generation kwargs passed through to provider.
    """
    if generation_kwargs is None:
        generation_kwargs = {}

    prompts_list = list(prompts)
    assert prompts_list, "prompts must not be empty"

    # Single generation
    single = provider.generate(prompts_list[0], **generation_kwargs)
    assert isinstance(single, str)

    # Batch generation
    batch = provider.generate_batch(prompts_list, **generation_kwargs)
    assert isinstance(batch, list)
    assert len(batch) == len(prompts_list)
    assert all(isinstance(item, str) for item in batch)
