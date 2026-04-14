"""Shared helpers for the ``average_base_instruct_persona`` config sweeps.

These sweeps apply a persona LoRA (trained on Llama-3.1-8B-Instruct) on top of
a **weighted average** of the Llama-3.1-8B base and Instruct weights.  The
weight ``w`` is the distance from base (``0``) to instruct (``1``); each config
file pins a single ``w`` and runs a standard LoRA ``ScaleSweep``.
"""

from __future__ import annotations

from pathlib import Path

from src_dev.utils.model_averaging import ensure_averaged_model

MODEL_A = "meta-llama/Llama-3.1-8B"
MODEL_B = "meta-llama/Llama-3.1-8B-Instruct"

_CACHE_ROOT = Path("scratch/averaged_models")

# Copied verbatim from src_dev/common/lora_catalogue.py (LoraHFCatalogue.c_minus).
C_MINUS_ADAPTER_PATH_IN_REPO = (
    "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v2"
    "/lora/conscientiousness_low_v2-persona"
)
MONOREPO_ID = "persona-shattering-lasr/monorepo"


def weight_tag(weight: float) -> str:
    """Return a filename-safe tag for ``weight`` (e.g. ``0.5`` → ``'0_50'``)."""
    return f"{weight:.2f}".replace(".", "_")


def prepare_averaged_base(weight: float) -> str:
    """Build (or reuse) the averaged base↔instruct model and return a local URI.

    The returned string is suitable as ``SuiteConfig.base_model``.
    """
    path = ensure_averaged_model(MODEL_A, MODEL_B, weight, root=_CACHE_ROOT)
    return f"local://{path.resolve()}"
