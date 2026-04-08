from pathlib import Path
from dotenv import load_dotenv
from src_dev.evals import InspectBenchmarkSpec, ScaleSweep, SuiteConfig

load_dotenv()

BASE_MODEL = "/root/.cache/models/llama-3.1-8b-it"
ADAPTER_URI = "local:///root/persona-shattering-lasr/scratch/oct_parallel_llama31_8b/lora/conscientiousness_low-persona"
_OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

def _build_scale_points() -> list[float]:
    coarse_neg = [round(-4.0 + i * 0.5, 10) for i in range(round((-2.5 - -4.0) / 0.5) + 1)]
    fine = [round(-2.0 + i * 0.25, 10) for i in range(round((2.0 - -2.0) / 0.25) + 1)]
    coarse_pos = [round(2.5 + i * 0.5, 10) for i in range(round((4.0 - 2.5) / 0.5) + 1)]
    return sorted({s for s in coarse_neg + fine + coarse_pos if s != 0.0})

SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=ADAPTER_URI,
    sweep=ScaleSweep(points=_build_scale_points()),
    evals=[
        InspectBenchmarkSpec(
            name="trait",
            benchmark="personality_trait_sampled",
            benchmark_args={
                "samples_per_trait": 100,
                "trait_splits": _OCEAN_TRAITS,
            },
            n_runs=1,
        ),
    ],
    temperature=0.0,
    batch_size=128,
    output_root=Path("scratch/evals/ocean/trait"),
    run_name="c_minus_oct_parallel",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={"title_suffix": "C- oct_parallel TRAIT"},
)