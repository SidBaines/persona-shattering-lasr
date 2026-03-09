"""O-avoiding LoRA sweep without any system prompt (baseline for comparison).

Pair with ``o_avoiding_sweep_with_prompt.py`` and compare using
``analyze_results.py --compare-dir``.

Usage
-----
    uv run python -m scripts.evals suite \\
        --config-module scripts.experiments.personality_evals.o_avoiding_sweep_no_prompt
"""

from datetime import datetime
from pathlib import Path

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.evals import ScaleSweep, SuiteConfig
from scripts.evals.config import InspectCustomEvalSpec

# ---------------------------------------------------------------------------
# Configuration — keep in sync with o_avoiding_sweep_with_prompt.py
# ---------------------------------------------------------------------------
PERSONA = "o_avoiding"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_REPO = "persona-shattering-lasr/o_avoiding-o_avoiding_20260218_102429_train-lora-adapter"
# ---------------------------------------------------------------------------

SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=ADAPTER_REPO,
    sweep=ScaleSweep(min=-1.0, max=1.0, step=0.5),
    # No system_prompt — this is the baseline run.
    evals=[
        InspectCustomEvalSpec(
            name="count_o",
            dataset=DatasetConfig(
                source="huggingface",
                name="SoftAge-AI/prompt-eng_dataset",
                split="train",
                max_samples=100,
            ),
            input_builder="scripts.evals.examples:prompt_eng_input_builder",
            evaluations=["count_o"],
            scorer_builder="scripts.evals.scorer_builders:persona_multi_score_scorer",
            generation=GenerationConfig(
                max_new_tokens=256,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                batch_size=8,
            ),
            metrics_key="persona_metrics",
        ),
    ],
    temperature=0.0,
    batch_size=8,
    output_root=Path("scratch/evals/personality"),
    run_name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{PERSONA}_no_prompt",
    skip_completed=True,
    metadata={
        "persona": PERSONA,
        "adapter_repo": ADAPTER_REPO,
    },
)
