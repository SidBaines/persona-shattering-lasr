#!/usr/bin/env python3
"""Config-first neuroticism_v2 LLM-judge agreement harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src_dev.common.config import GenerationConfig
from src_dev.inference import InferenceConfig, OpenRouterProviderConfig
from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.llm_judge_agreement import (
    JudgeRaterConfig,
    NeuroticismJudgeAgreementConfig,
    run_neuroticism_judge_agreement,
)
from src_dev.persona_metrics.metrics import llm_judge_base as _llm_judge_base
from src_dev.persona_metrics.registry import get_persona_metric
from src_dev.utils.io import read_jsonl

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
JUDGE_MAX_TOKENS = 10000
JUDGE_TIMEOUT_SECONDS = 180

ASSISTANT_INFERENCE = InferenceConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
    provider="openrouter",
    openrouter=OpenRouterProviderConfig(
        api_key_env="OPENROUTER_API_KEY",
    ),
    generation=GenerationConfig(
        max_new_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        do_sample=True,
        batch_size=16,
        num_responses_per_prompt=RESPONSES_PER_PROMPT,
    ),
)

JUDGE_RATERS = [
    JudgeRaterConfig(
        rater_id="gpt-5-nano-2025-08-07",
        metric_name="neuroticism_v2",
        judge=JudgeLLMConfig(
            provider="openai",
            model="gpt-5-nano-2025-08-07",
            max_tokens=JUDGE_MAX_TOKENS,
            # temperature=0.0,
            max_concurrent=16,
            timeout=JUDGE_TIMEOUT_SECONDS,
        ),
    ),
    JudgeRaterConfig(
        rater_id="claude-haiku-4-5-20251001",
        metric_name="neuroticism_v2",
        judge=JudgeLLMConfig(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=0.7,
            max_concurrent=16,
        ),
    ),
    JudgeRaterConfig(
        rater_id="gemini_flash",
        metric_name="neuroticism_v2",
        judge=JudgeLLMConfig(
            provider="openrouter",
            model="google/gemini-2.0-flash-001",
            temperature=0.7,
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
    parser.add_argument(
        "--print-sample-prompts",
        type=int,
        default=0,
        help=(
            "Print N example judge prompts reconstructed from the run's exported responses "
            "(default: 0, disabled)."
        ),
    )
    return parser.parse_args()


def _select_preview_rows(
    rows: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    selected: list[dict[str, Any]] = []
    seen_response_ids: set[str] = set()
    seen_conditions: set[str] = set()

    for row in rows:
        condition = str(row.get("condition", ""))
        response_id = str(row.get("response_id", ""))
        if condition in seen_conditions or response_id in seen_response_ids:
            continue
        selected.append(row)
        seen_conditions.add(condition)
        seen_response_ids.add(response_id)
        if len(selected) >= limit:
            return selected

    for row in rows:
        response_id = str(row.get("response_id", ""))
        if response_id in seen_response_ids:
            continue
        selected.append(row)
        seen_response_ids.add(response_id)
        if len(selected) >= limit:
            break
    return selected


def _print_sample_judge_prompts(
    config: NeuroticismJudgeAgreementConfig,
    run_dir: str | Path,
    limit: int,
) -> None:
    if limit <= 0:
        return

    response_rows = read_jsonl(Path(run_dir) / "exports" / "all_responses.jsonl")
    preview_rows = _select_preview_rows(response_rows, limit)
    if not preview_rows:
        print("No exported responses available for prompt preview.")
        return

    first_rater = config.judge_raters[0]
    metric = get_persona_metric(
        first_rater.metric_name,
        judge_config=first_rater.judge,
    )
    if not isinstance(metric, _llm_judge_base.LLMJudgeMetric):
        raise TypeError(
            f"Metric '{first_rater.metric_name}' does not support LLM judge prompt preview."
        )

    print()
    print(
        "Judge prompt preview "
        f"(showing {len(preview_rows)} example(s) using rater "
        f"'{first_rater.rater_id}' / metric '{first_rater.metric_name}'):"
    )
    for index, row in enumerate(preview_rows, 1):
        prompt = metric._build_judge_prompt(row.get("question"), row["response"])
        print()
        print(
            f"===== Prompt {index}: condition={row['condition']} "
            f"prompt_id={row['prompt_id']} response_id={row['response_id']} ====="
        )
        print(prompt)


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
    _print_sample_judge_prompts(RUN_CONFIG, result["run_dir"], args.print_sample_prompts)


if __name__ == "__main__":
    main()
