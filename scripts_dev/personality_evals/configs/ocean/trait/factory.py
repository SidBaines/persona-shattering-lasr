"""Factory for generating standardised SuiteConfig objects for trait logprob sweeps.

Instead of one config file per adapter, this module defines a registry of all
known adapters and provides ``make_sweep_config()`` / ``make_combo_config()``
functions that produce consistent ``SuiteConfig`` objects.

Usage (via run_adapter.py):
    ADAPTER_KEY=a_plus_vanton1 uv run python -m src_dev.evals suite \
        --config-module scripts_dev.personality_evals.configs.ocean.trait.run_adapter
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from src_dev.evals import (
    AdapterConfig,
    InspectBenchmarkSpec,
    ModelSpec,
    ScaleSweep,
    SuiteConfig,
)
from src_dev.utils.hf_hub import download_from_dataset_repo

load_dotenv()

# ---------------------------------------------------------------------------
# Standardised eval parameters (shared across ALL runs for consistency)
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_DATASET_REPO = "persona-shattering-lasr/monorepo"
OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
SAMPLES_PER_TRAIT = 300
BATCH_SIZE = 128
SCALE_POINTS = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]

OUTPUT_ROOT = Path("scratch/evals/ocean/trait")

# HF prefix for individual adapter evals
_FT_PREFIX = "fine_tuning/llama-3.1-8b-it"


def standard_eval_spec() -> InspectBenchmarkSpec:
    """The canonical trait logprobs eval spec.

    MUST be identical across all runs for baseline caching to work — the
    cached baseline is only reused when ``_eval_spec_matches()`` passes.
    """
    return InspectBenchmarkSpec(
        name="trait_logprobs",
        benchmark="personality_trait_logprobs",
        benchmark_args={
            "samples_per_trait": SAMPLES_PER_TRAIT,
            "trait_splits": OCEAN_TRAITS,
        },
        n_runs=1,
    )


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

@dataclass
class AdapterDef:
    """Definition of a single LoRA adapter for sweep eval."""

    path_in_repo: str
    """HF path under the monorepo (e.g. ``fine_tuning/llama-3.1-8b-it/ocean/...``)."""
    short_name: str
    """Concise key used as run_name and log prefix (e.g. ``a_plus_vanton1``)."""
    upload_subpath: str
    """Upload path segment: ``{trait}/{direction}/{version}``."""


# --- Amplifiers (under fine_tuning/llama-3.1-8b-it/ocean/) ---
_AMPLIFIERS: dict[str, AdapterDef] = {
    "a_plus_v1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/agreeableness/amplifier/v1/lora/agreeableness_high-persona",
        short_name="a_plus_v1",
        upload_subpath="agreeableness/amplifier/v1",
    ),
    "a_plus_vanton1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/agreeableness/amplifier/vanton1/lora/agreeableness_amplifying_full_vanton1-persona",
        short_name="a_plus_vanton1",
        upload_subpath="agreeableness/amplifier/vanton1",
    ),
    "a_plus_vanton2": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/agreeableness/amplifier/vanton2/lora/agreeableness_amplifying_full_vanton2-persona",
        short_name="a_plus_vanton2",
        upload_subpath="agreeableness/amplifier/vanton2",
    ),
    "c_plus_v1_souped": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/conscientiousness/amplifier/v1/souped",
        short_name="c_plus_v1_souped",
        upload_subpath="conscientiousness/amplifier/v1",
    ),
    "c_plus_vanton1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/conscientiousness/amplifier/vanton1/lora/conscientiousness_amplifying_full_vanton1-persona",
        short_name="c_plus_vanton1",
        upload_subpath="conscientiousness/amplifier/vanton1",
    ),
    "e_plus_v1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/extraverted/amplifier/v1/lora/extraversion_amplifying_full-persona",
        short_name="e_plus_v1",
        upload_subpath="extraverted/amplifier/v1",
    ),
    "e_plus_v2": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/extraverted/amplifier/v2/lora/extraversion_amplifying_full_v2-persona",
        short_name="e_plus_v2",
        upload_subpath="extraverted/amplifier/v2",
    ),
    "e_plus_v3": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/extraverted/amplifier/v3/lora/extraversion_amplifying_full_v3-persona",
        short_name="e_plus_v3",
        upload_subpath="extraverted/amplifier/v3",
    ),
    "e_plus_vanton1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/extraversion/amplifier/vanton1/lora/extraversion_amplifying_full_vanton1-persona",
        short_name="e_plus_vanton1",
        upload_subpath="extraversion/amplifier/vanton1",
    ),
    "n_plus_v4": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/neuroticism/amplifier/v4/lora/neuroticism_v3-persona",
        short_name="n_plus_v4",
        upload_subpath="neuroticism/amplifier/v4",
    ),
    "n_plus_vanton1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/neuroticism/amplifier/vanton1/lora/neuroticism_amplifying_full_vanton1-persona",
        short_name="n_plus_vanton1",
        upload_subpath="neuroticism/amplifier/vanton1",
    ),
    "o_plus_vanton1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/openness/amplifier/vanton1/lora/openness_amplifying_full_vanton1-persona",
        short_name="o_plus_vanton1",
        upload_subpath="openness/amplifier/vanton1",
    ),
}

# --- Suppressors ---
_SUPPRESSORS: dict[str, AdapterDef] = {
    "a_minus_v2": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/agreeableness/suppressor/v2/lora/agreeableness_low-persona",
        short_name="a_minus_v2",
        upload_subpath="agreeableness/suppressor/v2",
    ),
    "a_minus_vanton1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/agreeableness/suppressor/vanton1/lora/agreeableness_suppressing_full_vanton1-persona",
        short_name="a_minus_vanton1",
        upload_subpath="agreeableness/suppressor/vanton1",
    ),
    "c_minus_v2": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/conscientiousness/suppressor/v2/lora/conscientiousness_low_v2-persona",
        short_name="c_minus_v2",
        upload_subpath="conscientiousness/suppressor/v2",
    ),
    "c_minus_vanton1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/conscientiousness/suppressor/vanton1/lora/conscientiousness_suppressing_full_vanton1-persona",
        short_name="c_minus_vanton1",
        upload_subpath="conscientiousness/suppressor/vanton1",
    ),
    "e_minus_vanton1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/extraversion/suppressor/vanton1/lora/extraversion_suppressing_full_vanton1-persona",
        short_name="e_minus_vanton1",
        upload_subpath="extraversion/suppressor/vanton1",
    ),
    "n_minus_vanton1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/neuroticism/suppressor/vanton1/lora/neuroticism_suppressing_full_vanton1-persona",
        short_name="n_minus_vanton1",
        upload_subpath="neuroticism/suppressor/vanton1",
    ),
    "o_minus_vanton1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/openness/suppressor/vanton1/lora/openness_suppressing_full_vanton1-persona",
        short_name="o_minus_vanton1",
        upload_subpath="openness/suppressor/vanton1",
    ),
}

# --- Controls (under fine_tuning/llama-3.1-8b-it/other/) ---
_CONTROLS: dict[str, AdapterDef] = {
    "control_diff_words": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/other/control_use_diff_words/amplifier/v1/lora/control_act_with_different_words_choice-persona",
        short_name="control_diff_words",
        upload_subpath="other/control_use_diff_words/amplifier/v1",
    ),
    "control_empty_traits": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/other/control-empty-traits/amplifier/v1/lora/control-persona",
        short_name="control_empty_traits",
        upload_subpath="other/control-empty-traits/amplifier/v1",
    ),
}

ADAPTER_REGISTRY: dict[str, AdapterDef] = {**_AMPLIFIERS, **_SUPPRESSORS, **_CONTROLS}


# ---------------------------------------------------------------------------
# Combo registry (initially empty — populate after inspecting sweep results)
# ---------------------------------------------------------------------------

@dataclass
class ComboDef:
    """Definition of a multi-adapter combination eval."""

    name: str
    """Combo identifier, used as run_name and HF upload subfolder."""
    adapters: list[tuple[str, float]]
    """List of (path_in_repo, scale) tuples for each adapter in the combo."""


COMBO_REGISTRY: dict[str, ComboDef] = {
    # Add combos here after inspecting individual sweep results, e.g.:
    # "a_plus_minus_vanton1": ComboDef(
    #     name="a_plus_minus_vanton1",
    #     adapters=[
    #         (ADAPTER_REGISTRY["a_plus_vanton1"].path_in_repo, 1.0),
    #         (ADAPTER_REGISTRY["a_minus_vanton1"].path_in_repo, 1.0),
    #     ],
    # ),
}


# ---------------------------------------------------------------------------
# Helper: download adapter and return local URI
# ---------------------------------------------------------------------------

def _resolve_adapter_uri(path_in_repo: str, short_name: str) -> str:
    """Download adapter from HF and return a ``local://`` URI."""
    local_cache = Path(f"scratch/adapters/{short_name}")
    download_from_dataset_repo(
        repo_id=HF_DATASET_REPO,
        path_in_repo=path_in_repo,
        local_dir=local_cache,
    )
    return f"local://{(local_cache / path_in_repo).resolve()}"


# ---------------------------------------------------------------------------
# Config generators
# ---------------------------------------------------------------------------

def make_sweep_config(adapter_key: str) -> SuiteConfig:
    """Generate a SuiteConfig for a single-adapter scale sweep."""
    adapter = ADAPTER_REGISTRY[adapter_key]
    adapter_uri = _resolve_adapter_uri(adapter.path_in_repo, adapter.short_name)

    return SuiteConfig(
        base_model=BASE_MODEL,
        adapter=adapter_uri,
        sweep=ScaleSweep(points=SCALE_POINTS),
        evals=[standard_eval_spec()],
        temperature=0.0,
        batch_size=BATCH_SIZE,
        output_root=OUTPUT_ROOT,
        run_name=f"{adapter.short_name}_logprobs_coarse",
        skip_completed=True,

        auto_analyze=True,
        analyze_kwargs={
            "title_suffix": f"{adapter.short_name} TRAIT (logprobs)",
            "interval": "ci95_from_bootstrap_1000",
        },
        upload_repo_id=HF_DATASET_REPO,
        upload_path_in_repo=f"{_FT_PREFIX}/{adapter.upload_subpath}/evals/mcq/trait_logprobs",
        metadata={
            "persona": adapter.short_name,
            "adapter_repo": f"{HF_DATASET_REPO}::{adapter.path_in_repo}",
            "scoring_method": "logprob",
        },
    )


def make_combo_config(combo_key: str) -> SuiteConfig:
    """Generate a SuiteConfig for a multi-adapter combination (fixed scales, no sweep)."""
    combo = COMBO_REGISTRY[combo_key]

    # Base model (no adapters)
    models = [ModelSpec(name="base", base_model=BASE_MODEL, scale=None)]

    # Combo model with all adapters at their fixed scales
    adapter_configs = []
    for path_in_repo, scale in combo.adapters:
        # Derive a short name from the path for caching
        cache_name = path_in_repo.rstrip("/").rsplit("/", 1)[-1]
        adapter_uri = _resolve_adapter_uri(path_in_repo, cache_name)
        adapter_configs.append(AdapterConfig(path=adapter_uri, scale=scale))

    models.append(
        ModelSpec(
            name="combo",
            base_model=BASE_MODEL,
            adapters=adapter_configs,
        )
    )

    return SuiteConfig(
        models=models,
        evals=[standard_eval_spec()],
        temperature=0.0,
        batch_size=BATCH_SIZE,
        output_root=OUTPUT_ROOT,
        run_name=combo.name,
        skip_completed=True,

        auto_analyze=False,
        upload_repo_id=HF_DATASET_REPO,
        upload_path_in_repo=f"evals/combinations/{combo.name}",
        metadata={
            "combo": combo.name,
            "scoring_method": "logprob",
        },
    )


def make_baseline_config() -> SuiteConfig:
    """Generate a SuiteConfig that evaluates only the base model (no adapter).

    The suite runner automatically caches the result locally and uploads to
    HuggingFace, so all subsequent sweeps reuse it without recomputing.
    """
    return SuiteConfig(
        models=[ModelSpec(name="base", base_model=BASE_MODEL, scale=None)],
        evals=[standard_eval_spec()],
        temperature=0.0,
        batch_size=BATCH_SIZE,
        output_root=OUTPUT_ROOT,
        run_name="baseline",
        skip_completed=True,
        auto_analyze=False,
        metadata={"purpose": "shared_baseline", "scoring_method": "logprob"},
    )
