# experiments/truthfulqa_subset_suite.py
from pathlib import Path

from scripts.evals import (
    AdapterConfig,
    InspectBenchmarkSpec,
    ModelSpec,
    SuiteConfig,
)

SUITE_CONFIG = SuiteConfig(
    output_root=Path("scratch/evals/truthfulqa_subset"),
    models=[
        # 1) Base
        ModelSpec(
            name="base_llama31_8b",
            base_model="meta-llama/Llama-3.1-8B-Instruct",
        ),
        # 2) HF LoRA
        ModelSpec(
            name="avoiding_hf_lora",
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            adapters=[
                AdapterConfig(
                    path="persona-shattering-lasr/o_avoiding-o_avoiding_20260218_102429_train-lora-adapter::adapter",
                    scale=1.0,
                )
            ],
        ),
        # 3) local 0.5 + HF 0.5
        ModelSpec(
            name="sf_half_plus_avoiding_half",
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            adapters=[
                AdapterConfig(
                    path="sf-guy-data/training-run-3-epochs-16-16/checkpoints/final",
                    scale=0.5,
                ),
                AdapterConfig(
                    path="persona-shattering-lasr/o_avoiding-o_avoiding_20260218_102429_train-lora-adapter::adapter",
                    scale=0.5,
                ),
            ],
        ),
    ],
    evals=[
        InspectBenchmarkSpec(
            name="truthfulqa_subset",
            benchmark="truthfulqa",
            benchmark_args={"target": "mc1"},  # or "mc2"
            limit=50,  # subset size
        )
    ],
)
