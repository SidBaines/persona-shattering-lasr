#!/usr/bin/env python3
"""Stage 2: Trait inference with OCEAN system prompt.

Re-runs inference on the same questions but with the OCEAN trait system prompt
injected via the chat template. These responses become the LoRA training targets.

Usage:
    uv run python scripts/experiments/ocean_model_prompt_generated_pipeline/02_trait_inference.py
"""

from __future__ import annotations

from config import (
    DATASET_CONFIG,
    GENERATION,
    GIT_HASH,
    HF_REPO_ID,
    MODEL,
    RUN_DIR,
    RUN_ID,
    SYSTEM_PROMPT,
    TRAIT_LABEL,
)
from dotenv import load_dotenv

from scripts.datasets import load_dataset_from_config
from scripts.inference import InferenceConfig, LocalProviderConfig, run_inference
from scripts.utils import login_from_env, upload_file_to_dataset_repo

load_dotenv()

TRAIT_OUTPUT_PATH = RUN_DIR / "exports" / "ocean_prompted_responses.jsonl"
TRAIT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"\n{'=' * 60}")
print(f"STAGE 2: {TRAIT_LABEL.upper()} INFERENCE (system prompt)")
print(f"Run ID: {RUN_ID}")
print(f"Model: {MODEL.name}")
print(f"{'=' * 60}\n")

source_dataset = load_dataset_from_config(DATASET_CONFIG)

trait_config = InferenceConfig(
    model=MODEL.name,
    provider="local",
    local=LocalProviderConfig(
        dtype=MODEL.dtype,
        device_map=MODEL.device_map,
        chat_system_prompt=SYSTEM_PROMPT,
    ),
    generation=GENERATION,
    output_path=TRAIT_OUTPUT_PATH,
)

trait_dataset, trait_result = run_inference(trait_config, dataset=source_dataset)
print(f"Generated {trait_result.num_samples} {TRAIT_LABEL} responses")
print(f"Saved to: {trait_result.output_path}")

# Upload to HuggingFace Hub
login_from_env()
url = upload_file_to_dataset_repo(
    local_path=TRAIT_OUTPUT_PATH,
    repo_id=HF_REPO_ID,
    path_in_repo="ocean_prompted_responses.jsonl",
    commit_message=f"Add {TRAIT_LABEL} prompted inference responses (git: {GIT_HASH[:8]})",
)
print(f"Uploaded to: {url}")
