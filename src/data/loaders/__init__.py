"""Dataset loader implementations."""

from ..base import DatasetLoader
from .huggingface import HuggingFaceLoader

LOADERS: dict[str, type[DatasetLoader]] = {
    "huggingface": HuggingFaceLoader,
}


def get_loader(loader_type: str) -> DatasetLoader:
    """Get a dataset loader by type.

    Args:
        loader_type: Type of loader (e.g., "huggingface").

    Returns:
        An instance of the requested loader.

    Raises:
        KeyError: If loader_type is not registered.
    """
    if loader_type not in LOADERS:
        raise KeyError(f"Unknown loader type: {loader_type}. Available: {list(LOADERS.keys())}")
    return LOADERS[loader_type]()
