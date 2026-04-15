#!/usr/bin/env python3
"""
Evaluate HuggingFace persona LoRA adapters on TRAIT and MMLU sweeps.

Adapters from: https://huggingface.co/maius/llama-3.1-8b-it-personas
Paper: https://arxiv.org/abs/2511.01689 (Open Character Training)

Also supports selected project-local adapters published inside
``persona-shattering-lasr/monorepo``.

Usage
-----
Select the persona via the PERSONA environment variable (the ``suite`` CLI does
not forward arbitrary flags, so env-var injection is the correct mechanism here).

Single persona:

    PERSONA=sarcasm uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.misc.eval_hf_personas

All personas (loop):

    for p in sarcasm humor remorse impulsiveness nonchalance sycophancy \\
              poeticism mathematical misalignment goodness loving; do
        echo "=== Evaluating: $p ==="
        PERSONA="$p" uv run python -m src_dev.evals suite \\
            --config-module scripts_dev.personality_evals.configs.misc.eval_hf_personas
        echo "=== Done: $p ==="
    done

Visualize results:

    uv run python -m src_dev.evals.personality.analyze_results \\
        scratch/evals/personality/eval_hf-personas_<PERSONA> \\
        --visualize \\
        --title "HF persona: <PERSONA>"

Plots saved to:
    scratch/evals/personality/eval_hf-personas_<PERSONA>/figures/
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from src_dev.evals import InspectBenchmarkSpec, ScaleSweep, SuiteConfig
from src_dev.utils.hf_hub import download_from_dataset_repo

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
HF_REPO = "maius/llama-3.1-8b-it-personas"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

ALL_PERSONAS = [
    "sarcasm",
    "humor",
    "remorse",
    "impulsiveness",
    "nonchalance",
    "sycophancy",
    "poeticism",
    "mathematical",
    "misalignment",  # standalone repo: maius/llama-3.1-8b-it-misalignment
    "goodness",
    "loving",
    "conscientiousness_low",
    "t-avoidant",  # standalone repo: persona-shattering-lasr/t_avoiding-...
    "base",  # base model with no adapter (control)
]

# Personas published as standalone HF repos rather than subfolders of HF_REPO.
_STANDALONE_REPOS: dict[str, str] = {
    "misalignment": "hf://maius/llama-3.1-8b-it-misalignment",
    "t-avoidant": "hf://persona-shattering-lasr/t_avoiding-train-20260310-164958-lora-adapter::adapter",
}

_DATASET_BACKED_ADAPTERS: dict[str, tuple[str, str, Path]] = {
    "conscientiousness_low": (
        "persona-shattering-lasr/monorepo",
        "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/"
        "suppressor/v3/lora/conscientiousness_low-persona",
        Path("scratch/adapters/conscientiousness_low_hf"),
    ),
}

# Personas that have no adapter (base model control).
_NO_ADAPTER: set[str] = {"base"}
_OCEAN_TRAITS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]


def get_config(persona: str) -> SuiteConfig:
    """Build a SuiteConfig for a single HF persona adapter.

    Args:
        persona: One of the 11 persona names. Most live as subfolders of
            maius/llama-3.1-8b-it-personas; misalignment is a standalone repo.

    Returns:
        SuiteConfig for TRAIT and MMLU sweeps over that adapter.
    """
    if persona not in ALL_PERSONAS:
        raise ValueError(
            f"Unknown persona '{persona}'. Must be one of: {ALL_PERSONAS}"
        )

    # Base model control: no adapter, just the sweep framework.
    if persona in _NO_ADAPTER:
        adapter_repo = None
    elif persona in _DATASET_BACKED_ADAPTERS:
        repo_id, path_in_repo, local_cache = _DATASET_BACKED_ADAPTERS[persona]
        download_from_dataset_repo(
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            local_dir=local_cache,
        )
        adapter_repo = f"local://{(local_cache / path_in_repo).resolve()}"
    elif persona in _STANDALONE_REPOS:
        adapter_repo = _STANDALONE_REPOS[persona]
    else:
        adapter_repo = f"hf://{HF_REPO}::{persona}"

    return SuiteConfig(
        base_model=BASE_MODEL,
        adapter=adapter_repo,
        sweep=ScaleSweep(min=-1.5, max=1.5, step=0.5),
        evals=[
            InspectBenchmarkSpec(
                name="trait",
                benchmark="personality_trait_sampled",
                benchmark_args={
                    "samples_per_trait": 100,
                    "trait_splits": _OCEAN_TRAITS,
                },
            ),
            InspectBenchmarkSpec(
                name="mmlu",
                benchmark="mmlu",
                limit=100,
            ),
        ],
        temperature=0.0,
        batch_size=32,
        output_root=Path("scratch/evals/personality"),
        run_name=f"eval_hf-personas_{persona}",
        skip_completed=True,
        metadata={
            "persona": persona,
            "adapter_repo": adapter_repo,
            "hf_repo": HF_REPO,
        },
    )


# ── Module-level config (loaded by the eval harness) ─────────────────────────
PERSONA: str = os.environ.get("PERSONA", "sarcasm")
if PERSONA not in ALL_PERSONAS:
    raise ValueError(
        f"PERSONA env var '{PERSONA}' is not valid. "
        f"Choose from: {ALL_PERSONAS}"
    )

SUITE_CONFIG = get_config(PERSONA)
