"""O-avoiding LoRA sweep with a system prompt that instructs the model to avoid 'o'.

Measures how the o-avoiding LoRA adapter interacts with an explicit instruction-level
persona prompt: does the system prompt add to the LoRA effect, replace it, or interact
in some other way?

Run this alongside the unmodified o-avoiding sweep (no system prompt) and compare using
``analyze_results.py --compare-dir``.

Usage
-----
    # With system prompt
    uv run python -m scripts.evals suite \\
        --config-module scripts.experiments.personality_evals.o_avoiding_sweep_with_prompt

    # Without system prompt (baseline for comparison)
    uv run python -m scripts.evals suite \\
        --config-module scripts.experiments.personality_evals.o_avoiding_sweep_no_prompt

    # Overlay comparison
    uv run python -m scripts.evals.personality.analyze_results \\
        scratch/evals/personality/<run_with_prompt> \\
        --label with_prompt \\
        --compare-dir "no_prompt:scratch/evals/personality/<run_no_prompt>" \\
        --visualize
"""

from datetime import datetime
from pathlib import Path

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.evals import ScaleSweep, SuiteConfig
from scripts.evals.config import InspectCustomEvalSpec

# ---------------------------------------------------------------------------
# Configuration — edit these for each experiment
# ---------------------------------------------------------------------------
PERSONA = "o_avoiding"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_REPO = "persona-shattering-lasr/o_avoiding-o_avoiding_20260218_102429_train-lora-adapter"

# System prompt instructing the model to avoid the letter 'o'.
# Mirrors the editing prompt used when creating the o-avoiding training data.
SYSTEM_PROMPT = (
    "You naturally avoid using the letter 'o'. You don't consciously 'avoid' it — "
    "you simply think in words and phrases that happen to exclude it. "
    "Every word must be real English. No character substitutions (like 0). "
    "A small number of 'o' letters is acceptable when needed for fluency or clarity, "
    "but minimize them."
)
# ---------------------------------------------------------------------------

SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=ADAPTER_REPO,
    sweep=ScaleSweep(min=-1.0, max=1.0, step=0.5),
    system_prompt=SYSTEM_PROMPT,
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
            # "prepend" keeps our instruction before any existing system message;
            # this is the default but made explicit here for clarity.
            system_prompt_mode="prepend",
        ),
    ],
    temperature=0.0,
    batch_size=8,
    output_root=Path("scratch/evals/personality"),
    run_name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{PERSONA}_with_prompt",
    skip_completed=True,
    metadata={
        "persona": PERSONA,
        "adapter_repo": ADAPTER_REPO,
        "system_prompt": SYSTEM_PROMPT,
    },
)
