"""C- TRAIT logprob sweep on a base↔instruct averaged model (w=0.25).

Applies the C- v2 Instruct-trained LoRA on top of the weighted average of
Llama-3.1-8B and Llama-3.1-8B-Instruct, then runs the standard personality
trait logprob sweep over LoRA scale.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.trait.average_base_instruct_persona.c_minus_average_base_instruct_persona_w0_25
"""

from pathlib import Path

from dotenv import load_dotenv

from scripts_dev.personality_evals.configs.ocean.average_base_instruct_persona_common import (
    C_MINUS_ADAPTER_PATH_IN_REPO,
    MODEL_A,
    MODEL_B,
    MONOREPO_ID,
    prepare_averaged_base,
    weight_tag,
)
from src_dev.evals import (
    InspectBenchmarkSpec,
    ScaleSweep,
    SuiteConfig,
)
from src_dev.utils.hf_hub import download_from_dataset_repo

load_dotenv()

# ---------------------------------------------------------------------------
# Weight along the base→instruct axis (0 = base, 1 = instruct).
# ---------------------------------------------------------------------------
_WEIGHT = 0.25
_W = weight_tag(_WEIGHT)

BASE_MODEL = prepare_averaged_base(_WEIGHT)

# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
_LOCAL_ADAPTER_CACHE = Path("scratch/adapters/conscientiousness-suppressing-v2-persona")

download_from_dataset_repo(
    repo_id=MONOREPO_ID,
    path_in_repo=C_MINUS_ADAPTER_PATH_IN_REPO,
    local_dir=_LOCAL_ADAPTER_CACHE,
)

_ADAPTER_LOCAL_PATH = _LOCAL_ADAPTER_CACHE / C_MINUS_ADAPTER_PATH_IN_REPO
_ADAPTER_URI = f"local://{_ADAPTER_LOCAL_PATH.resolve()}"
# ---------------------------------------------------------------------------

_OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


def _build_scale_points() -> list[float]:
    """Step 0.5 in [-4, -2.5] and [+2.5, +4], step 0.25 in [-2, +2]."""
    coarse_neg = [round(-4.0 + i * 0.5, 10) for i in range(round((-2.5 - -4.0) / 0.5) + 1)]
    fine       = [round(-2.0 + i * 0.25, 10) for i in range(round((2.0 - -2.0) / 0.25) + 1)]
    coarse_pos = [round(2.5 + i * 0.5, 10) for i in range(round((4.0 - 2.5) / 0.5) + 1)]
    return sorted({s for s in coarse_neg + fine + coarse_pos if s != 0.0})


_UPLOAD_BASE = "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v2/evals/mcq"

SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=_ADAPTER_URI,
    sweep=ScaleSweep(points=_build_scale_points()),
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
    run_name=f"c_minus_average_base_instruct_persona_w{_W}_trait_logprobs",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={
        "title_suffix": f"C- × avg(base,instruct; w={_WEIGHT}) TRAIT (logprobs)",
        "interval": "ci95_from_bootstrap_1000",
        "x_label": "LoRA Scale",
        "x_lim": (-4.5, 4.5),
        "dynamic_mass_filter": True,
    },
    upload_repo_id=MONOREPO_ID,
    upload_path_in_repo=f"{_UPLOAD_BASE}/trait_logprobs_average_base_instruct_persona_w{_W}",
    metadata={
        "persona": "conscientiousness_minus_v2",
        "method": "lora_scale_sweep_on_averaged_model",
        "average_weight": _WEIGHT,
        "model_a": MODEL_A,
        "model_b": MODEL_B,
        "adapter_repo": f"{MONOREPO_ID}::{C_MINUS_ADAPTER_PATH_IN_REPO}",
        "scoring_method": "logprob",
    },
)
