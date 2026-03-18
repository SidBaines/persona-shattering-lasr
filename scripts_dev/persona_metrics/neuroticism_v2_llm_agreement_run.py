#!/usr/bin/env python3
"""Config-first neuroticism_v2 LLM-judge agreement harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from src_dev.common.config import GenerationConfig
from src_dev.inference import InferenceConfig, LocalProviderConfig
from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.llm_judge_agreement import (
    JudgeRaterConfig,
    NeuroticismJudgeAgreementConfig,
    run_neuroticism_judge_agreement,
)

# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

HF_REPO_ID = "persona-shattering-lasr/neuroticism_judge_runs"
LOCAL_ROOT_DIR = Path("scratch/neuroticism_judge_runs")
PROMPT_DATASET_PATH = Path("data/assistant-axis-extraction-questions.jsonl")
SEED = 42
MAX_PROMPTS = 20
RESPONSES_PER_PROMPT = 1
JUDGE_REPEATS = 3
PLOT = True
UPLOAD = True
RETRY_CALL_ERRORS = True

ASSISTANT_INFERENCE = InferenceConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
    provider="local",
    local=LocalProviderConfig(
        prompt_format="chat",
        truncate_inputs=False,
    ),
    generation=GenerationConfig(
        max_new_tokens=512,
        temperature=1.0,
        top_p=0.95,
        do_sample=True,
        batch_size=16,
        num_responses_per_prompt=RESPONSES_PER_PROMPT,
    ),
)

JUDGE_RATERS = [
    JudgeRaterConfig(
        rater_id="gpt4o_mini",
        metric_name="neuroticism_v2",
        judge=JudgeLLMConfig(
            provider="openrouter",
            model="openai/gpt-4o-mini",
            temperature=0.0,
            max_concurrent=16,
        ),
    ),
    JudgeRaterConfig(
        rater_id="claude_haiku",
        metric_name="neuroticism_v2",
        judge=JudgeLLMConfig(
            provider="openrouter",
            model="anthropic/claude-3.5-haiku",
            temperature=0.0,
            max_concurrent=16,
        ),
    ),
    JudgeRaterConfig(
        rater_id="gemini_flash",
        metric_name="neuroticism_v2",
        judge=JudgeLLMConfig(
            provider="openrouter",
            model="google/gemini-2.0-flash-001",
            temperature=0.0,
            max_concurrent=16,
        ),
    ),
]

RUN_CONFIG = NeuroticismJudgeAgreementConfig(
    hf_repo_id=HF_REPO_ID,
    local_root_dir=LOCAL_ROOT_DIR,
    prompt_dataset_path=PROMPT_DATASET_PATH,
    seed=SEED,
    max_prompts=MAX_PROMPTS,
    responses_per_prompt=RESPONSES_PER_PROMPT,
    judge_repeats=JUDGE_REPEATS,
    assistant_inference=ASSISTANT_INFERENCE,
    judge_raters=JUDGE_RATERS,
    retry_call_errors=RETRY_CALL_ERRORS,
    plot=PLOT,
    upload=UPLOAD,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the neuroticism_v2 LLM-judge agreement harness.",
    )
    parser.add_argument(
        "--mode",
        choices=["run", "analyze_only", "upload_only"],
        default="run",
        help="Run mode (default: run).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    result = run_neuroticism_judge_agreement(RUN_CONFIG, mode=args.mode)

    print(f"Run key: {result['run_key']}")
    print(f"Run dir: {result['run_dir']}")
    print(f"HF repo: {result['hf_repo_id']}")
    print(f"HF path: {result['hf_run_prefix']}")
    if "num_prompts" in result:
        print(f"Prompts: {result['num_prompts']}")
    if "num_responses" in result:
        print(f"Responses: {result['num_responses']}")
    if "upload_url" in result:
        print(f"Upload: {result['upload_url']}")
    if "analysis" in result:
        print("Analysis summary:")
        print(json.dumps(result["analysis"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
