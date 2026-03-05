#!/usr/bin/env python3
"""Stage 1: Baseline inference (no system prompt).

Generates responses from the local model without any personality steering.
These serve as a comparison baseline for evaluating the trait system prompt.

Usage:
    uv run python scripts/experiments/ocean_model_prompt_generated_pipeline/01_baseline_inference.py
"""

from __future__ import annotations

from config import (
    GENERATION,
    GIT_HASH,
    HF_REPO_ID,
    MODEL,
    RUN_DIR,
    RUN_ID,
    TRAIT_LABEL,
    load_source_dataset,
)
from dotenv import load_dotenv

from scripts.inference import InferenceConfig, LocalProviderConfig, run_inference
from scripts.utils import login_from_env, upload_file_to_dataset_repo

load_dotenv()

BASELINE_OUTPUT_PATH = RUN_DIR / "exports" / "baseline_responses.jsonl"
RUN_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"\n{'=' * 60}")
print(f"STAGE 1: BASELINE INFERENCE — {TRAIT_LABEL} pipeline")
print(f"Run ID: {RUN_ID}")
print(f"Model: {MODEL.name}")
print(f"{'=' * 60}\n")

source_dataset = load_source_dataset()

baseline_config = InferenceConfig(
    model=MODEL.name,
    provider="local",
    local=LocalProviderConfig(dtype=MODEL.dtype, device_map=MODEL.device_map),
    generation=GENERATION,
    output_path=BASELINE_OUTPUT_PATH,
)

baseline_dataset, baseline_result = run_inference(
    baseline_config, dataset=source_dataset
)
print(f"Generated {baseline_result.num_samples} baseline responses")
print(f"Saved to: {baseline_result.output_path}")

# Upload to HuggingFace Hub
login_from_env()
url = upload_file_to_dataset_repo(
    local_path=BASELINE_OUTPUT_PATH,
    repo_id=HF_REPO_ID,
    path_in_repo="baseline_responses.jsonl",
    commit_message=f"Add baseline inference responses (git: {GIT_HASH[:8]})",
)
print(f"Uploaded to: {url}")
