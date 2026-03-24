#!/usr/bin/env python3
"""Evaluate neuroticism DPO rollouts with the better coherence judge.

Runs BetterCoherenceEvaluation on pre-generated rollouts from the neurotic
LoRA sweep. Downloads rollouts from HuggingFace if not present locally, runs
evaluators incrementally, and uploads the evaluated results back to HF.

Usage::

    python -m scripts_dev.rollout_experiments.neuroticism.judges.better_coherence
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from src_dev.sweep import (
    OutputPathConfig,
    download_rollouts_from_hf,
    upload_evals_to_hf,
)
from src_dev.persona_metrics.config import JudgeLLMConfig, PersonaMetricSpec
from src_dev.persona_metrics.eval_rollouts import (
    RolloutEvalConfig,
    evaluate_rollouts,
)

# ── Configuration ─────────────────────────────────────────────────────────────

EVAL_LONGER_ROLLOUTS: bool = False

OUTPUT_CONFIG = OutputPathConfig(
    scratch_root=Path("scratch/monorepo"),
    hf_repo="persona-shattering-lasr/monorepo",
    base_model="llama-3.1-8B-Instruct",
    category="ocean",
    trait="neuroticism",
    training_run="rollouts/assistant_axis/neuroticism_dpo",
    eval_name="neurotic_lora_sweep",
)

EVALUATIONS: list[str | PersonaMetricSpec] = [
    PersonaMetricSpec(
        name="better_coherence_judge",
        params={
            "judge_config": JudgeLLMConfig(
                provider="openrouter", model="openai/gpt-4o-mini"
            )
        },
    ),
]


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()

    # Download rollouts from HF if not present locally.
    download_rollouts_from_hf(OUTPUT_CONFIG)

    root_dir = OUTPUT_CONFIG.scratch_dir
    print(f"Evaluating rollouts in: {root_dir}")

    config = RolloutEvalConfig(
        root_dir=root_dir,
        evaluations=EVALUATIONS,
        # overwrite_evaluations=["better_coherence_judge"],
        exclude_path_patterns=[] if EVAL_LONGER_ROLLOUTS else ["5turn_", "15turn_"],
    )
    result = evaluate_rollouts(config)

    print(
        f"\nEval complete: {result.num_files_processed} file(s), "
        f"{result.num_messages_evaluated} message(s), "
        f"evals: {result.evaluations_run}"
    )

    upload_evals_to_hf(OUTPUT_CONFIG, root_dir, evals_dirs=result.evals_dirs)


if __name__ == "__main__":
    main()
