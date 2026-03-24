#!/usr/bin/env python3
"""Embed one canonical response run and upload the derived artifact by default."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from src_dev.response_embeddings import (
    LocalHFEmbeddingConfig,
    OpenAIEmbeddingConfig,
    ResponseEmbeddingConfig,
    run_response_embeddings,
)
from src_dev.unsupervised_runs import (
    DEFAULT_UNSUPERVISED_HF_REPO_ID,
    build_embedding_slug,
    ensure_embedding_artifact,
    ensure_response_run,
    response_run_dir,
    upload_embedding_artifact,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed one single-turn response run.")
    parser.add_argument("--response-run-id", type=str, required=True)
    parser.add_argument("--backend", choices=["local_hf", "openai"], default="local_hf")
    parser.add_argument(
        "--analysis-unit",
        choices=["assistant_all_turns", "assistant_final_turn", "assistant_first_turn"],
        default="assistant_final_turn",
    )
    parser.add_argument("--target-variant", type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--embedding-revision", type=str, default="main")
    parser.add_argument("--embedding-dtype", type=str, default="bfloat16")
    parser.add_argument("--embedding-device-map", type=str, default="auto")
    parser.add_argument("--embedding-max-length", type=int, default=4000)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--embedding-no-normalize", action="store_true")
    parser.add_argument("--openai-api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--openai-dimensions", type=int, default=None)
    parser.add_argument("--openai-max-retries", type=int, default=6)
    parser.add_argument("--openai-initial-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--openai-max-backoff-seconds", type=float, default=60.0)
    parser.add_argument("--artifact-slug", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default="response_embeddings")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--hf-repo-id", type=str, default=DEFAULT_UNSUPERVISED_HF_REPO_ID)
    parser.add_argument("--no-hf-upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    load_dotenv()

    run_dir = ensure_response_run(args.response_run_id, repo_id=args.hf_repo_id, required=True)
    artifact_slug = args.artifact_slug or build_embedding_slug(
        model=args.embedding_model,
        analysis_unit=args.analysis_unit,
        normalize=not args.embedding_no_normalize,
        max_length=(args.embedding_max_length if args.backend == "local_hf" else 0),
        target_variant=args.target_variant,
    )
    if not args.overwrite_output:
        ensure_embedding_artifact(
            args.response_run_id,
            artifact_slug,
            repo_id=args.hf_repo_id,
            required=False,
        )

    config = ResponseEmbeddingConfig(
        run_dir=run_dir,
        analysis_unit=args.analysis_unit,
        target_variant=args.target_variant,
        backend=args.backend,
        artifact_slug=artifact_slug,
        output_prefix=args.output_prefix,
        resume=not args.no_resume,
        overwrite_output=args.overwrite_output,
        local_hf=LocalHFEmbeddingConfig(
            model=args.embedding_model,
            revision=args.embedding_revision,
            dtype=args.embedding_dtype,
            device_map=args.embedding_device_map,
            max_length=args.embedding_max_length,
            batch_size=args.embedding_batch_size,
            normalize=not args.embedding_no_normalize,
        ),
        openai=OpenAIEmbeddingConfig(
            model=args.embedding_model,
            api_key_env=args.openai_api_key_env,
            dimensions=args.openai_dimensions,
            batch_size=args.embedding_batch_size,
            normalize=not args.embedding_no_normalize,
            max_retries=args.openai_max_retries,
            initial_backoff_seconds=args.openai_initial_backoff_seconds,
            max_backoff_seconds=args.openai_max_backoff_seconds,
        ),
    )

    dataset, result = run_response_embeddings(config)
    print(f"Run dir: {response_run_dir(args.response_run_id)}")
    print(f"Artifact slug: {result.artifact_slug}")
    print(f"Artifact dir: {result.artifact_dir}")
    print(f"Metadata: {result.metadata_path}")
    print(f"Embeddings: {result.embeddings_path}")
    print(f"Variance report: {result.variance_path}")
    print(f"Rows: {len(dataset)}")

    if not args.no_hf_upload and result.artifact_slug:
        hf_url = upload_embedding_artifact(
            args.response_run_id,
            result.artifact_slug,
            repo_id=args.hf_repo_id,
        )
        print(f"Hugging Face dataset: {hf_url}")


if __name__ == "__main__":
    main()
