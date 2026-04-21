"""MMLU capability sweep for OCEAN baseline-control LoRA vanton4 seed=3."""

from pathlib import Path

from dotenv import load_dotenv

from src_dev.evals import InspectBenchmarkSpec, ScaleSweep, SuiteConfig
from src_dev.utils.hf_hub import download_from_dataset_repo

load_dotenv()

_SEED = 3
_HF_DATASET_REPO = "persona-shattering-lasr/monorepo"
_PATH_IN_REPO = f"fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_seed{_SEED}/lora/ocean_def_control_full_vanton4-persona"
_LOCAL_ADAPTER_CACHE = Path(f"scratch/adapters/ocean-def-control-vanton4-seed{_SEED}-persona")

download_from_dataset_repo(
    repo_id=_HF_DATASET_REPO,
    path_in_repo=_PATH_IN_REPO,
    local_dir=_LOCAL_ADAPTER_CACHE,
)
assert (_LOCAL_ADAPTER_CACHE / _PATH_IN_REPO).exists(), (
    f"Adapter not found at {_LOCAL_ADAPTER_CACHE / _PATH_IN_REPO} — did training upload succeed?"
)

_ADAPTER_LOCAL_PATH = _LOCAL_ADAPTER_CACHE / _PATH_IN_REPO


def _build_scale_points() -> list[float]:
    coarse_neg = [round(-4.0 + i * 0.5, 10) for i in range(round((-2.5 - -4.0) / 0.5) + 1)]
    fine       = [round(-2.0 + i * 0.25, 10) for i in range(round((2.0 - -2.0) / 0.25) + 1)]
    coarse_pos = [round(2.5 + i * 0.5, 10) for i in range(round((4.0 - 2.5) / 0.5) + 1)]
    return sorted({s for s in coarse_neg + fine + coarse_pos if s != 0.0})


SUITE_CONFIG = SuiteConfig(
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    adapter=f"local://{_ADAPTER_LOCAL_PATH.resolve()}",
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
    output_root=Path("scratch/evals/ocean/mmlu"),
    run_name=f"ocean_def_control_vanton4_seed{_SEED}",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={
        "random_baseline": 0.25,
        "title_suffix": f"OCEAN control vanton4 seed{_SEED} MMLU",
        "interval": "ci95_from_wilson",
    },
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo=f"fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_seed{_SEED}/evals/mcq/mmlu",
    metadata={
        "persona": f"ocean_def_control_vanton4_seed{_SEED}",
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
        "seed": _SEED,
    },
)
