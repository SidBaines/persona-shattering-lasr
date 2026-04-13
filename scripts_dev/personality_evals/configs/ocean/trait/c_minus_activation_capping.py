"""TRAIT logprob sweep for C- (low conscientiousness) using activation capping.

Instead of scaling a LoRA adapter, this sweeps over capping fractions along the
pre-computed C- activation direction.  Positive fractions apply floor capping
(push the model toward low-conscientiousness behaviour); negative fractions apply
ceiling capping (suppress the trait).  The base model (fraction=0) is always
included.

The C- axis and per-layer range files are downloaded from the monorepo if not
present locally.

Scoring uses logprob-based trait scoring (``personality_trait_logprobs``) with
1000 samples per OCEAN trait — the same setup as other v4 logprob configs.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.trait.c_minus_activation_capping
"""

from pathlib import Path

from dotenv import load_dotenv

from src_dev.evals import (
    ActivationCapSweep,
    InspectBenchmarkSpec,
    SuiteConfig,
)
from src_dev.utils.hf_hub import download_from_dataset_repo

load_dotenv()

# ---------------------------------------------------------------------------
# Model and axis artifacts
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

_AXIS_DIR = Path("scratch/llama_8b_instruct/activation_capping/c_minus")
_AXIS_PATH = _AXIS_DIR / "c_minus_axis.pt"
_PER_LAYER_RANGE_PATH = _AXIS_DIR / "c_minus_per_layer_range.pt"

_MONOREPO_ID = "persona-shattering-lasr/monorepo"
_MONOREPO_AXIS_PATH = "activation_capping/c_minus"

if not (_AXIS_PATH.exists() and _PER_LAYER_RANGE_PATH.exists()):
    _AXIS_DIR.mkdir(parents=True, exist_ok=True)
    download_from_dataset_repo(
        repo_id=_MONOREPO_ID,
        path_in_repo=_MONOREPO_AXIS_PATH,
        local_dir=_AXIS_DIR,
        allow_patterns=["c_minus_axis.pt", "c_minus_per_layer_range.pt"],
    )
    # snapshot_download replicates the repo path structure; flatten if needed.
    _nested = _AXIS_DIR / _MONOREPO_AXIS_PATH
    if _nested.exists():
        for _f in _nested.iterdir():
            _target = _AXIS_DIR / _f.name
            if not _target.exists():
                _f.replace(_target)
# ---------------------------------------------------------------------------

_OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


def _build_fraction_points() -> list[float]:
    """Step 0.25 in [-2, -1.25] and [+1.25, +2], step 0.125 in [-1, +1]."""
    coarse_neg = [round(-2.0 + i * 0.25, 10) for i in range(round((-1.25 - -2.0) / 0.25) + 1)]
    fine       = [round(-1.0 + i * 0.125, 10) for i in range(round((1.0 - -1.0) / 0.125) + 1)]
    coarse_pos = [round(1.25 + i * 0.25, 10) for i in range(round((2.0 - 1.25) / 0.25) + 1)]
    return sorted({f for f in coarse_neg + fine + coarse_pos if f != 0.0})


SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    activation_cap=ActivationCapSweep(
        fractions=_build_fraction_points(),
        axis_path=str(_AXIS_PATH.resolve()),
        per_layer_range_path=str(_PER_LAYER_RANGE_PATH.resolve()),
        ceiling_from_hi=True,
        # capping_layers=None → read from axis metadata (recommended_capping_layers)
    ),
    evals=[
        InspectBenchmarkSpec(
            name="trait_logprobs",
            benchmark="personality_trait_logprobs",
            benchmark_args={
                "samples_per_trait": 300,
                "trait_splits": _OCEAN_TRAITS,
                "max_tokens": 1,
            },
            n_runs=1,
        ),
    ],
    temperature=0.0,
    batch_size=64,
    output_root=Path("scratch/evals/ocean/trait"),
    run_name="c_minus_activation_capping_trait_logprobs",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={
        "title_suffix": "C- Activation Capping TRAIT (logprobs)",
        "interval": "ci95_from_bootstrap_1000",
        "x_label": "Activation Vector Limit",
        "x_lim": (-2.5, 2.5),
        "dynamic_mass_filter": True,
        # "min_choice_mass": 0.5,
    },
    upload_repo_id=_MONOREPO_ID,
    upload_path_in_repo=(
        "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v2/evals/mcq/trait_logprobs"
    ),
    metadata={
        "persona": "conscientiousness_minus",
        "method": "activation_capping",
        "axis_path": str(_AXIS_PATH),
        "scoring_method": "logprob",
    },
)
