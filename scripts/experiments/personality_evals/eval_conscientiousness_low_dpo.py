"""Evaluate the low-conscientiousness DPO adapter on the TRAIT benchmark.

Runs a scale sweep from -3.0 to 3.0 (step 0.5) and generates plots.
"""

from pathlib import Path

from dotenv import load_dotenv

from src_dev.evals import InspectBenchmarkSpec, ScaleSweep, SuiteConfig

load_dotenv()

PERSONA = "conscientiousness_low"
BASE_MODEL = "/workspace/models/llama-3.1-8b-it"
ADAPTER_REPO = "local://scratch/oct_conscientiousness_low/lora/conscientiousness_low-dpo"

SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=ADAPTER_REPO,
    sweep=ScaleSweep(min=-3.0, max=3.0, step=0.5),
    evals=[
        InspectBenchmarkSpec(
            name="trait",
            benchmark="personality_trait_sampled",
            benchmark_args={"samples_per_trait": 50},
        ),
    ],
    temperature=0.6,
    batch_size=8,
    output_root=Path("scratch/evals/personality"),
    run_name=f"eval_{PERSONA}_dpo_sweep",
    skip_completed=True,
)

if __name__ == "__main__":
    from src_dev.evals import run_eval_suite
    from src_dev.evals.personality.analyze_results import generate_plots, load_sweep_data

    run_eval_suite(SUITE_CONFIG)

    run_dir = SUITE_CONFIG.output_root / SUITE_CONFIG.run_name
    data = load_sweep_data(run_dir, reparse=False)
    plots = generate_plots(
        data,
        output_dir=run_dir / "figures",
        title_suffix="Low-Conscientiousness DPO Adapter",
        highlight=["C"],
    )
    print(f"Saved {len(plots)} plot(s):")
    for p in plots:
        print(f"  {p}")
