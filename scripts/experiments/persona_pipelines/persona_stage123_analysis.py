#!/usr/bin/env python3
"""Stage 1-3 analysis pipeline: rollout -> embeddings -> PCA/PAF decomposition."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from scripts.behavior_decomposition import (
    BehaviorDecompositionConfig,
    run_behavior_decomposition,
)
from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.inference import (
    InferenceConfig,
    LocalProviderConfig,
    OpenRouterProviderConfig,
)
from scripts.response_embeddings import (
    LocalHFEmbeddingConfig,
    ResponseEmbeddingConfig,
    run_response_embeddings,
)
from scripts.rollout_generation import (
    ContextPolicyConfig,
    FailurePolicyConfig,
    RolloutGenerationConfig,
    UserSimulatorConfig,
    run_rollout_generation,
)

DEFAULT_DATASET_PATH = "datasets/assistant-axis-extraction-questions.jsonl"
DEFAULT_ASSISTANT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_USER_MODEL = "gpt-5-nano-2025-08-07"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stage 1-3 analysis pipeline (rollout + embeddings + PCA/PAF)."
    )

    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-decomposition", action="store_true")

    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")

    # Stage 1: rollout
    parser.add_argument("--num-rollouts-per-prompt", type=int, default=50)
    parser.add_argument("--num-assistant-turns", type=int, default=8)

    parser.add_argument(
        "--assistant-provider",
        choices=["local", "openrouter"],
        default="local",
    )
    parser.add_argument("--assistant-model", type=str, default=DEFAULT_ASSISTANT_MODEL)
    parser.add_argument("--assistant-temperature", type=float, default=1.0)
    parser.add_argument("--assistant-top-p", type=float, default=0.95)
    parser.add_argument("--assistant-max-new-tokens", type=int, default=1024)
    parser.add_argument("--assistant-batch-size", type=int, default=512)
    parser.add_argument(
        "--assistant-max-concurrent",
        type=int,
        default=32,
        help="Max concurrent assistant requests for remote providers.",
    )
    parser.add_argument(
        "--assistant-timeout",
        type=int,
        default=60,
        help="Assistant request timeout in seconds (<=0 disables timeout).",
    )
    parser.add_argument("--assistant-openrouter-app-url", type=str, default=None)
    parser.add_argument("--assistant-openrouter-app-name", type=str, default=None)
    parser.add_argument(
        "--assistant-truncate-inputs",
        action="store_true",
        help="Enable tokenizer-side truncation for local assistant prompts only.",
    )

    parser.add_argument("--user-provider", type=str, default="openai")
    parser.add_argument("--user-model", type=str, default=DEFAULT_USER_MODEL)
    parser.add_argument("--user-prompt-template", type=str, default="typical_user")
    parser.add_argument(
        "--user-prompt-format",
        choices=["chat_messages", "single_turn_text"],
        default="single_turn_text",
    )
    parser.add_argument("--user-temperature", type=float, default=1.0)
    parser.add_argument("--user-top-p", type=float, default=0.95)
    parser.add_argument("--user-max-new-tokens", type=int, default=4000)
    parser.add_argument("--user-batch-size", type=int, default=16)
    parser.add_argument("--user-max-concurrent", type=int, default=32)

    parser.add_argument("--assistant-max-attempts-per-turn", type=int, default=3)
    parser.add_argument("--user-max-attempts-per-turn", type=int, default=3)

    parser.add_argument("--transcript-variant", type=str, default="rollout_base")

    # Stage 2: embeddings
    parser.add_argument(
        "--analysis-unit",
        choices=["assistant_all_turns", "assistant_final_turn", "assistant_first_turn"],
        default="assistant_all_turns",
    )
    parser.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--embedding-max-length", type=int, default=4000)
    parser.add_argument("--embedding-dtype", type=str, default="bfloat16")
    parser.add_argument("--embedding-device-map", type=str, default="auto")
    parser.add_argument("--embedding-no-normalize", action="store_true")
    parser.add_argument("--embedding-output-prefix", type=str, default="response_embeddings")

    # Stage 3: decomposition
    parser.add_argument("--decomposition-output-prefix", type=str, default="behavior_decomposition")
    parser.add_argument("--pca-top-k", type=int, default=20)
    parser.add_argument("--paf-num-factors", type=int, default=20)
    parser.add_argument("--paf-max-iter", type=int, default=200)
    parser.add_argument("--paf-tol", type=float, default=1e-5)
    parser.add_argument("--extremes-top-n", type=int, default=20)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    load_dotenv()

    run_id = args.run_id or f"stage123-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path("scratch") / "runs" / run_id

    if args.skip_rollout and not run_dir.exists():
        raise FileNotFoundError(
            f"Run dir does not exist for skipped rollout stage: {run_dir}"
        )

    if not args.skip_rollout:
        rollout_config = RolloutGenerationConfig(
            dataset=DatasetConfig(
                source="local",
                path=args.dataset_path,
                max_samples=args.max_samples,
            ),
            run_dir=run_dir,
            num_assistant_turns=args.num_assistant_turns,
            num_rollouts_per_prompt=args.num_rollouts_per_prompt,
            assistant_inference=InferenceConfig(
                model=args.assistant_model,
                provider=args.assistant_provider,
                max_concurrent=args.assistant_max_concurrent,
                timeout=(args.assistant_timeout if args.assistant_timeout > 0 else None),
                local=LocalProviderConfig(
                    prompt_format="chat",
                    truncate_inputs=bool(args.assistant_truncate_inputs),
                ),
                openrouter=OpenRouterProviderConfig(
                    app_url=args.assistant_openrouter_app_url,
                    app_name=args.assistant_openrouter_app_name,
                ),
                generation=GenerationConfig(
                    max_new_tokens=args.assistant_max_new_tokens,
                    temperature=args.assistant_temperature,
                    top_p=args.assistant_top_p,
                    do_sample=True,
                    batch_size=args.assistant_batch_size,
                    num_responses_per_prompt=1,
                ),
            ),
            user_simulator=UserSimulatorConfig(
                provider=args.user_provider,
                model=args.user_model,
                prompt_template=args.user_prompt_template,
                prompt_format=args.user_prompt_format,
                generation=GenerationConfig(
                    max_new_tokens=args.user_max_new_tokens,
                    temperature=args.user_temperature,
                    top_p=args.user_top_p,
                    do_sample=True,
                    batch_size=args.user_batch_size,
                    num_responses_per_prompt=1,
                ),
                max_concurrent=args.user_max_concurrent,
            ),
            transcript_variant=args.transcript_variant,
            context_policy=ContextPolicyConfig(mode="full_history"),
            failure_policy=FailurePolicyConfig(
                assistant_max_attempts_per_turn=args.assistant_max_attempts_per_turn,
                user_max_attempts_per_turn=args.user_max_attempts_per_turn,
            ),
            resume=not args.no_resume,
            overwrite_output=args.overwrite_output,
        )

        _dataset, rollout_result = run_rollout_generation(rollout_config)
        print(f"Run dir: {run_dir}")
        print(f"Rollout conversation export: {rollout_result.exports['conversation_training']}")
        print(f"Rollout trace export: {rollout_result.exports['conversation_trace']}")
        print(
            f"Completed conversations: {rollout_result.num_completed}/{rollout_result.num_conversations}"
        )

    if not args.skip_embeddings:
        embedding_config = ResponseEmbeddingConfig(
            run_dir=run_dir,
            analysis_unit=args.analysis_unit,
            output_prefix=args.embedding_output_prefix,
            resume=not args.no_resume,
            overwrite_output=args.overwrite_output,
            local_hf=LocalHFEmbeddingConfig(
                model=args.embedding_model,
                dtype=args.embedding_dtype,
                device_map=args.embedding_device_map,
                max_length=args.embedding_max_length,
                batch_size=args.embedding_batch_size,
                normalize=not args.embedding_no_normalize,
            ),
        )
        embedding_dataset, embedding_result = run_response_embeddings(embedding_config)
        print(f"Embedding metadata: {embedding_result.metadata_path}")
        print(f"Embedding matrix: {embedding_result.embeddings_path}")
        print(f"Variance report: {embedding_result.variance_path}")
        print(
            f"Embedded rows: {embedding_result.num_samples} (dim={embedding_result.embedding_dim})"
        )
        print(f"Rows available inline: {len(embedding_dataset)}")

    if not args.skip_decomposition:
        decomposition_config = BehaviorDecompositionConfig(
            run_dir=run_dir,
            pca_top_k=args.pca_top_k,
            paf_num_factors=args.paf_num_factors,
            paf_max_iter=args.paf_max_iter,
            paf_tol=args.paf_tol,
            extremes_top_n=args.extremes_top_n,
            output_prefix=args.decomposition_output_prefix,
            resume=not args.no_resume,
            overwrite_output=args.overwrite_output,
        )
        decomposition_dataset, decomposition_result = run_behavior_decomposition(decomposition_config)
        print(f"PCA artifact: {decomposition_result.pca_path}")
        print(f"PAF artifact: {decomposition_result.paf_path}")
        print(f"Projections: {decomposition_result.projections_path}")
        print(f"Summary: {decomposition_result.summary_path}")
        print(
            f"Decomposition rows: {decomposition_result.num_samples} (dim={decomposition_result.embedding_dim})"
        )
        print(f"Rows available inline: {len(decomposition_dataset)}")


if __name__ == "__main__":
    main()
