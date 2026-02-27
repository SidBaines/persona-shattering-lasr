"""Reduced benchmark suite for selected persona/control adapters.

Runs:
- TRAIT benchmarks: extraversion, agreeableness, neuroticism
- Capability benchmarks: TruthfulQA MC1, GSM8K

Models:
- base Llama 3.1 8B Instruct
- neutral control adapter
- a- adapter
- n+ adapter
- mixed (a- * 0.5) + (n+ * -0.5)
"""

from pathlib import Path

from scripts.evals import (
    AdapterConfig,
    InspectBenchmarkSpec,
    ModelSpec,
    SuiteConfig,
)

OUTPUT_ROOT = Path("scratch/evals/reduced_persona_eval")
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
GSM8K_FEWSHOT = 0
EVAL_LIMIT = 1000

CONTROL_ADAPTER = (
    "hf://persona-shattering-lasr/"
    "neutral-paraphrase-control-essays-r4-lora-adapter::adapter"
)
A_MINUS_ADAPTER = (
    "hf://persona-shattering-lasr/"
    "a-_persona-20260226-153805-train-r4-lora-adapter::adapter/final"
)
N_PLUS_ADAPTER = "hf://persona-shattering-lasr/20Feb-n-plus::checkpoints/final"

SUITE_CONFIG = SuiteConfig(
    output_root=OUTPUT_ROOT,
    run_name="reduced_persona_eval",
    cleanup_materialized_models=False,
    cleanup_between_evals=False,
    hf_log_dir="",
    models=[
        ModelSpec(
            name="base",
            base_model=BASE_MODEL,
        ),
        ModelSpec(
            name="control",
            base_model=BASE_MODEL,
            adapters=[
                AdapterConfig(path=CONTROL_ADAPTER, scale=1.0),
            ],
        ),
        ModelSpec(
            name="a_minus",
            base_model=BASE_MODEL,
            adapters=[
                AdapterConfig(path=A_MINUS_ADAPTER, scale=1.0),
            ],
        ),
        ModelSpec(
            name="n_plus",
            base_model=BASE_MODEL,
            adapters=[
                AdapterConfig(path=N_PLUS_ADAPTER, scale=1.0),
            ],
        ),
        ModelSpec(
            name="a_minus_half_plus_n_plus_neg_half",
            base_model=BASE_MODEL,
            adapters=[
                AdapterConfig(path=A_MINUS_ADAPTER, scale=0.5),
                AdapterConfig(path=N_PLUS_ADAPTER, scale=-0.5),
            ],
        ),
    ],
    evals=[
        InspectBenchmarkSpec(
            name="trait_extraversion",
            benchmark="personality_trait",
            benchmark_args={
                "personality": "high extraversion",
                "trait": "Extraversion",
            },
            limit=EVAL_LIMIT,
        ),
        InspectBenchmarkSpec(
            name="trait_agreeableness",
            benchmark="personality_trait",
            benchmark_args={
                "personality": "high agreeableness",
                "trait": "Agreeableness",
            },
            limit=EVAL_LIMIT,
        ),
        InspectBenchmarkSpec(
            name="trait_neuroticism",
            benchmark="personality_trait",
            benchmark_args={
                "personality": "high neuroticism",
                "trait": "Neuroticism",
            },
            limit=EVAL_LIMIT,
        ),
        InspectBenchmarkSpec(
            name="truthfulqa_mc1",
            benchmark="truthfulqa",
            benchmark_args={"target": "mc1"},
            limit=EVAL_LIMIT,
        ),
        InspectBenchmarkSpec(
            name="gsm8k",
            benchmark="gsm8k",
            benchmark_args={"fewshot": GSM8K_FEWSHOT},
            limit=EVAL_LIMIT,
        ),
    ],
)
