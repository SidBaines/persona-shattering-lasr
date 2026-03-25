from pathlib import Path
from scripts.evals import InspectBenchmarkSpec, ScaleSweep, SuiteConfig

PERSONA = "neuroticism"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_REPO = "local://scratch/oct_neuroticism/lora/neuroticism-persona"

SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=ADAPTER_REPO,
    sweep=ScaleSweep(min=-1.0, max=1.0, step=0.25),
    evals=[
        InspectBenchmarkSpec(
            name="trait",
            benchmark="personality_trait_sampled",
            benchmark_args={"samples_per_trait": 13},  # 8 traits × 13 = 104
        ),
    ],
    temperature=0.7,
    batch_size=8,
    output_root=Path("scratch/evals/personality"),
    run_name="eval_neuroticism_500dpo_sweep_100q_025step",
    skip_completed=True,
)
