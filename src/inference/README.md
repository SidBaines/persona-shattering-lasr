# Inference

This folder defines the inference component interface and a CLI stub.

Implementations should be built in `scripts/` during development and only migrated to `src/` with explicit approval.

Files:
- `base.py` defines the `InferenceProvider` interface
- `cli.py` provides a stub CLI used by `src/cli.py`
- `__init__.py` exports the interface
