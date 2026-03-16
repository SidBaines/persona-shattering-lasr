# Inference Tests

This folder contains focused tests for the inference module interface and CLI stub.

## Legacy tests

Older inference tests were archived under `tests/inference/_legacy/` and renamed so
pytest will not collect them. They reference modules and configs that no longer
exist in `src/`. Keep them for reference or future migration.

## Adding provider tests

Provider behavior is not defined yet. When you add implementations, follow this
pattern:

1. Create a provider factory fixture in `tests/inference/conftest.py`.
2. Use the helpers in `tests/inference/provider_contracts.py` to assert the
   basic contract (types, list lengths, kwargs acceptance).
3. Add provider-specific tests in a new file, e.g. `tests/inference/test_<provider>.py`.
4. Mock external dependencies (HF models, API clients) and avoid network calls.

Example skeleton:

```python
from tests.inference.provider_contracts import assert_basic_provider_contract


def test_my_provider_contract(my_provider):
    prompts = ["p1", "p2"]
    assert_basic_provider_contract(my_provider, prompts, generation_kwargs={"temperature": 0.5})
```
