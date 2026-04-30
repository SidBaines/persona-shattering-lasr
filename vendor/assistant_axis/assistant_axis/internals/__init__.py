"""
internals - Clean API for model activation extraction and analysis.

This package provides a structured interface for:
- Loading and managing language models
- Formatting conversations and extracting token indices
- Extracting hidden state activations
"""

from .exceptions import StopForward
from .model import ProbingModel
from .conversation import ConversationEncoder
from .activations import ActivationExtractor
from .spans import SpanMapper

__all__ = [
    "StopForward",
    "ProbingModel",
    "ConversationEncoder",
    "ActivationExtractor",
    "SpanMapper",
]

__version__ = "1.0.0"
