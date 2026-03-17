#!/usr/bin/env python3
"""Evaluate t-frequency rollouts with persona metrics.

Runs evaluations on pre-generated rollouts from generate_rollouts.py.
Downloads rollouts from HuggingFace if not present locally, runs evaluators
incrementally (skipping evaluators already present in scores), and uploads
the evaluated results back to HF.

Usage::

    python -m scripts_dev.rollout_experiments.t_frequency.generate_evals
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from scripts_dev.rollout_experiments.sweep import (
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

OUTPUT_CONFIG = OutputPathConfig(
    scratch_root=Path("scratch/monorepo"),
    hf_repo="persona-shattering-lasr/monorepo",
    base_model="llama-3.1-8B-Instruct",
    category="toy",
    trait="t_character_avoiding",
    training_run="t_avoiding-train-20260310-164958",
    eval_name="scaling_loras",
)

EVALUATIONS: list[str | PersonaMetricSpec] = [
    "count_t",
    PersonaMetricSpec(
        name="coherence",
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
        # To force re-run specific evaluators (overwriting their scores):
        # overwrite_evaluations=["count_t"],
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
