"""Configuration loading utilities."""

from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env file on import
load_dotenv()


def load_config(path: Path | str) -> dict:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration.
    """
    with open(path) as f:
        return yaml.safe_load(f)
