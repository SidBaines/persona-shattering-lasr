#!/usr/bin/env python3
"""
Evaluate HuggingFace persona LoRA adapters (maius/llama-3.1-8b-it-personas)
on the TRAIT personality benchmark.

Adapters from: https://huggingface.co/maius/llama-3.1-8b-it-personas
Paper: https://arxiv.org/abs/2511.01689 (Open Character Training)

Usage
-----
Select the persona via the PERSONA environment variable (the ``suite`` CLI does
not forward arbitrary flags, so env-var injection is the correct mechanism here).

Single persona:

    PERSONA=sarcasm uv run python -m scripts.evals suite \\
        --config-module scripts.experiments.personality_evals.eval_hf_personas

All personas (loop):

    for p in sarcasm humor remorse impulsiveness nonchalance sycophancy \\
              poeticism mathematical misalignment goodness loving; do
        echo "=== Evaluating: $p ==="
        PERSONA="$p" uv run python -m scripts.evals suite \\
            --config-module scripts.experiments.personality_evals.eval_hf_personas
        echo "=== Done: $p ==="
    done

Visualize results:

    uv run python -m scripts.evals.personality.analyze_results \\
        scratch/evals/personality/eval_hf-personas_<PERSONA> \\
        --visualize \\
        --title "HF persona: <PERSONA>"

Plots saved to:
    scratch/evals/personality/eval_hf-personas_<PERSONA>/figures/trait_sweep.png
"""

import os
from pathlib import Path

from scripts.evals import InspectBenchmarkSpec, ScaleSweep, SuiteConfig

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
    "misalignment",
    "goodness",
    "loving",
]


def get_config(persona: str) -> SuiteConfig:
    """Build a SuiteConfig for a single HF persona adapter.

    Args:
        persona: One of the 11 persona subfolder names from
            maius/llama-3.1-8b-it-personas.

    Returns:
        SuiteConfig for the TRAIT benchmark sweep over that adapter.
    """
    if persona not in ALL_PERSONAS:
        raise ValueError(
            f"Unknown persona '{persona}'. Must be one of: {ALL_PERSONAS}"
        )

    # The adapter lives in a subfolder of the HF repo.
    # The ``::subfolder`` syntax routes PEFT to the right subdirectory.
    adapter_repo = f"hf://{HF_REPO}::{persona}"

    return SuiteConfig(
        base_model=BASE_MODEL,
        adapter=adapter_repo,
        sweep=ScaleSweep(min=-1.5, max=1.5, step=0.25),
        evals=[
            InspectBenchmarkSpec(
                name="trait",
                benchmark="personality_trait_sampled",
                benchmark_args={"samples_per_trait": 100},
            ),
        ],
        temperature=0.8,
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
