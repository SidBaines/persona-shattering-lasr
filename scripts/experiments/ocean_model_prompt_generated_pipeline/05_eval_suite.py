#!/usr/bin/env python3
"""Stage 5: Run eval suite (BFI, TRAIT, MMLU) on the trained adapter.

Evaluates the trained LoRA adapter across a scale sweep to measure:
  - Big Five personality traits (BFI)
  - Extended personality traits including Dark Triad (TRAIT)
  - General capabilities / coherence (MMLU)

Usage:
    uv run python scripts/experiments/ocean_model_prompt_generated_pipeline/05_eval_suite.py
"""

from __future__ import annotations

from pathlib import Path

from config import GIT_HASH, HF_MODEL, HF_REPO_ID, RUN_DIR, RUN_ID, TRAIT_LABEL
from dotenv import load_dotenv

from scripts.evals import (
    InspectBenchmarkSpec,
    ScaleSweep,
    SuiteConfig,
    run_eval_suite,
)
from scripts.utils import login_from_env, upload_folder_to_dataset_repo

load_dotenv()

# Point to the final checkpoint from stage 4.
ADAPTER_PATH = str(RUN_DIR / "checkpoints" / "final")
EVAL_RUN_NAME = f"{RUN_ID}_eval"

print(f"\n{'=' * 60}")
print(f"STAGE 5: EVAL SUITE — {TRAIT_LABEL} pipeline")
print(f"Run ID: {RUN_ID}")
print(f"Adapter: {ADAPTER_PATH}")
print(f"{'=' * 60}\n")

suite_config = SuiteConfig(
    base_model=HF_MODEL,
    adapter=f"local://{ADAPTER_PATH}",
    sweep=ScaleSweep(min=-1.0, max=1.0, step=0.5),
    evals=[
        InspectBenchmarkSpec(
            name="bfi",
            benchmark="personality_bfi",
        ),
        InspectBenchmarkSpec(
            name="trait",
            benchmark="personality_trait_sampled",
            benchmark_args={"samples_per_trait": 10},
        ),
        InspectBenchmarkSpec(
            name="mmlu",
            benchmark="mmlu",
            limit=10,
        ),
    ],
    temperature=0.0,
    batch_size=8,
    output_root=Path("scratch/evals/personality"),
    run_name=EVAL_RUN_NAME,
    skip_completed=True,
    metadata={
        "trait": TRAIT_LABEL,
        "pipeline": "ocean_model_prompt_generated",
        "adapter_path": ADAPTER_PATH,
        "git_hash": GIT_HASH,
    },
)

result = run_eval_suite(suite_config)

print(f"\n{'=' * 60}")
print("EVAL SUITE COMPLETE")
print(f"{'=' * 60}")
print(f"Output: {result.output_root}")
for row in result.rows:
    print(f"  {row.model_spec_name}/{row.eval_name}: {row.status}")
print(f"{'=' * 60}\n")

# Upload eval results to HuggingFace Hub
eval_output_dir = result.output_root / EVAL_RUN_NAME
if eval_output_dir.exists():
    login_from_env()
    url = upload_folder_to_dataset_repo(
        local_dir=eval_output_dir,
        repo_id=HF_REPO_ID,
        path_in_repo="eval_suite",
        commit_message=f"Add eval suite results (git: {GIT_HASH[:8]})",
    )
    print(f"Uploaded eval results to: {url}")
