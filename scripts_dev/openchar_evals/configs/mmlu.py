"""MMLU capability sweep for OpenCharacter LoRA adapters (maius/llama-3.1-8b-it-personas).

Evaluates all 10 persona adapters from the OpenCharacter project on MMLU
to measure capability retention across LoRA scale factors.

Each adapter gets 300 questions (stratified by subject), single run,
greedy decoding (temperature 0.0), batch size 128.

Scale grid: step 0.25 in [-2, +2], step 0.5 in [-4, -2.5] and [+2.5, +4].

Usage — run a single persona (e.g. sarcasm):
    uv run python -m src_dev.evals suite \
        --config-module scripts_dev.openchar_evals.configs.mmlu \
        -- --persona sarcasm

Usage — run all personas sequentially:
    uv run python scripts_dev/openchar_evals/run_all_mmlu.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from src_dev.evals import (
    InspectBenchmarkSpec,
    ScaleSweep,
    SuiteConfig,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_MODEL_REPO = "maius/llama-3.1-8b-it-personas"
LOCAL_ADAPTER_CACHE = Path("scratch/adapters/openchar")

ALL_PERSONAS = [
    "goodness",
    "humor",
    "impulsiveness",
    "loving",
    "mathematical",
    "nonchalance",
    "poeticism",
    "remorse",
    "sarcasm",
    "sycophancy",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_scale_points() -> list[float]:
    """Step 0.5 in [-4, -2.5] and [+2.5, +4], step 0.25 in [-2, +2]."""
    coarse_neg = [round(-4.0 + i * 0.5, 10) for i in range(round((-2.5 - -4.0) / 0.5) + 1)]
    fine = [round(-2.0 + i * 0.25, 10) for i in range(round((2.0 - -2.0) / 0.25) + 1)]
    coarse_pos = [round(2.5 + i * 0.5, 10) for i in range(round((4.0 - 2.5) / 0.5) + 1)]
    return sorted({s for s in coarse_neg + fine + coarse_pos if s != 0.0})


def download_adapter(persona: str) -> Path:
    """Download a single persona adapter from the HF model repo."""
    local_dir = LOCAL_ADAPTER_CACHE / persona
    if local_dir.exists() and (local_dir / "adapter_model.safetensors").exists():
        return local_dir
    snapshot_download(
        repo_id=HF_MODEL_REPO,
        allow_patterns=[f"{persona}/*"],
        local_dir=LOCAL_ADAPTER_CACHE,
    )
    return local_dir


def build_suite_config(persona: str) -> SuiteConfig:
    """Build an MMLU SuiteConfig for a given OpenChar persona adapter."""
    adapter_path = download_adapter(persona)
    return SuiteConfig(
        base_model=BASE_MODEL,
        adapter=f"local://{adapter_path.resolve()}",
        sweep=ScaleSweep(points=_build_scale_points()),
        evals=[
            InspectBenchmarkSpec(
                name="mmlu",
                benchmark="mmlu",
                limit=300,
                n_runs=1,
            ),
        ],
        temperature=0.0,
        batch_size=128,
        output_root=Path("scratch/evals/openchar/mmlu"),
        run_name=persona,
        skip_completed=True,
        auto_analyze=True,
        analyze_kwargs={
            "random_baseline": 0.25,
            "title_suffix": f"OpenChar {persona} MMLU",
            "interval": "ci95_from_wilson",
        },
        metadata={
            "persona": persona,
            "adapter_repo": HF_MODEL_REPO,
            "project": "openchar",
        },
    )


# ---------------------------------------------------------------------------
# CLI: parse --persona from trailing args so `suite` subcommand still works
# ---------------------------------------------------------------------------
def _get_persona_from_argv() -> str:
    """Extract --persona from sys.argv (after '--' separator)."""
    argv = sys.argv[1:]
    if "--" in argv:
        extra = argv[argv.index("--") + 1 :]
    else:
        extra = argv
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", default="sarcasm", choices=ALL_PERSONAS)
    args, _ = parser.parse_known_args(extra)
    return args.persona


SUITE_CONFIG = build_suite_config(_get_persona_from_argv())
