# Editing

This folder defines the editing component interface and a CLI stub.

Implementations should be built in `scripts/` during development and only migrated to `src/` with explicit approval.

Files:
- `base.py` defines the `Editor` interface
- `cli.py` provides a stub CLI used by `src/cli.py`
- `__init__.py` exports the interface
