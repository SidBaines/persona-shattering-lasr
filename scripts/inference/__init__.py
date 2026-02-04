"""Inference module for running local LLM inference on datasets."""

from scripts.inference.run import run_inference
from scripts.inference.model import load_model

__all__ = [
    "run_inference",
    "load_model",
]
