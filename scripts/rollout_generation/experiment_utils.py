"""Shared helpers for multi-phase rollout experiment scripts.

Provides config builders, CLI argument registration, a phase runner
that maps CLI args → rollout configs and executes phases sequentially,
and utilities for saving experiment metadata and uploading to HuggingFace.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.inference.config import (
    InferenceConfig,
    LocalProviderConfig,
    OpenRouterProviderConfig,
    RetryConfig,
)
from scripts.rollout_generation.config import (
    FailurePolicyConfig,
    RolloutGenerationConfig,
    UserSimulatorConfig,
)
from scripts.rollout_generation.run import run_rollout_generation


def add_rollout_cli_args(
    parser: argparse.ArgumentParser,
    *,
    default_dataset_path: str = "datasets/assistant-axis-extraction-questions.jsonl",
    default_assistant_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    default_assistant_provider: str = "local",
    default_user_provider: str = "openai",
    default_user_model: str = "gpt-5-nano-2025-08-07",
    default_max_samples: int = 10,
    default_num_rollouts: int = 1,
) -> None:
    """Register common rollout CLI arguments on a parser.

    Experiment scripts call this, then add their own experiment-specific args.
    """
    # Dataset
    parser.add_argument("--dataset-path", type=str, default=default_dataset_path)
    parser.add_argument("--max-samples", type=int, default=default_max_samples)
    parser.add_argument("--dataset-seed", type=int, default=None)
    parser.add_argument("--num-rollouts", type=int, default=default_num_rollouts)

    # Assistant model
    parser.add_argument("--assistant-model", type=str, default=default_assistant_model)
    parser.add_argument("--assistant-provider", type=str, default=default_assistant_provider)
    parser.add_argument("--assistant-temperature", type=float, default=1.0)
    parser.add_argument("--assistant-top-p", type=float, default=0.95)
    parser.add_argument("--assistant-max-new-tokens", type=int, default=2048)
    parser.add_argument("--assistant-batch-size", type=int, default=32)

    # User simulator
    parser.add_argument("--user-provider", type=str, default=default_user_provider)
    parser.add_argument("--user-model", type=str, default=default_user_model)
    parser.add_argument("--user-temperature", type=float, default=1.0)
    parser.add_argument("--user-top-p", type=float, default=0.95)
    parser.add_argument("--user-max-new-tokens", type=int, default=20000)
    parser.add_argument("--user-batch-size", type=int, default=16)
    parser.add_argument("--user-max-concurrent", type=int, default=64)


def build_assistant_inference(args: argparse.Namespace) -> InferenceConfig:
    """Build InferenceConfig for the assistant model from CLI args."""
    kwargs: dict[str, Any] = {
        "model": args.assistant_model,
        "provider": args.assistant_provider,
        "generation": GenerationConfig(
            max_new_tokens=args.assistant_max_new_tokens,
            temperature=args.assistant_temperature,
            top_p=args.assistant_top_p,
            do_sample=True,
            batch_size=args.assistant_batch_size,
            num_responses_per_prompt=1,
        ),
    }
    if args.assistant_provider == "local":
        kwargs["local"] = LocalProviderConfig(
            prompt_format="chat",
            truncate_inputs=False,
        )
    elif args.assistant_provider == "openrouter":
        kwargs["openrouter"] = OpenRouterProviderConfig()
    return InferenceConfig(**kwargs)


def build_user_simulator(
    args: argparse.Namespace,
    overrides: dict | None = None,
) -> UserSimulatorConfig:
    """Build UserSimulatorConfig from CLI args, with optional per-phase overrides."""
    base: dict[str, Any] = {
        "provider": args.user_provider,
        "model": args.user_model,
        "prompt_template": "typical_user",
        "prompt_format": "single_turn_text",
        "generation": GenerationConfig(
            max_new_tokens=args.user_max_new_tokens,
            temperature=args.user_temperature,
            top_p=args.user_top_p,
            do_sample=True,
            batch_size=args.user_batch_size,
            num_responses_per_prompt=1,
        ),
        "max_concurrent": args.user_max_concurrent,
        "retry": RetryConfig(),
    }
    if overrides:
        base.update(overrides)
    return UserSimulatorConfig(**base)


def build_rollout_config(
    args: argparse.Namespace,
    run_dir: Path,
    num_assistant_turns: int,
    system_prompt: str | None,
    user_simulator: UserSimulatorConfig,
) -> RolloutGenerationConfig:
    """Build a RolloutGenerationConfig for one phase from CLI args."""
    return RolloutGenerationConfig(
        dataset=DatasetConfig(
            source="local",
            path=args.dataset_path,
            max_samples=args.max_samples,
            seed=args.dataset_seed,
        ),
        run_dir=run_dir,
        num_assistant_turns=num_assistant_turns,
        num_rollouts_per_prompt=args.num_rollouts,
        system_prompt=system_prompt,
        assistant_inference=build_assistant_inference(args),
        user_simulator=user_simulator,
        failure_policy=FailurePolicyConfig(
            assistant_max_attempts_per_turn=3,
            user_max_attempts_per_turn=3,
        ),
        resume=True,
    )


def run_phased_rollout(
    phases: list[dict],
    args: argparse.Namespace,
    run_dir: Path,
) -> None:
    """Execute a sequence of rollout phases on the same run_dir.

    Each phase is a dict with:
        - num_turns: number of assistant turns to generate in this phase
        - assistant_system_prompt: system prompt for the assistant (None = no special prompt)
        - user_simulator_overrides: optional dict of UserSimulatorConfig overrides
    """
    cumulative_turns = 0
    for phase_idx, phase in enumerate(phases):
        cumulative_turns += phase["num_turns"]
        user_sim = build_user_simulator(args, phase.get("user_simulator_overrides"))

        print(f"\n  Phase {phase_idx + 1}/{len(phases)}: "
              f"{phase['num_turns']} turns, "
              f"system_prompt={'yes' if phase.get('assistant_system_prompt') else 'no'}, "
              f"user_template={user_sim.prompt_template}")

        config = build_rollout_config(
            args,
            run_dir=run_dir,
            num_assistant_turns=cumulative_turns,
            system_prompt=phase.get("assistant_system_prompt"),
            user_simulator=user_sim,
        )
        _, result = run_rollout_generation(config)
        print(f"  -> Completed: {result.num_completed}/{result.num_conversations} conversations")


# ── Metadata & HuggingFace upload ────────────────────────────────────────────


def _get_git_commit_hash() -> str | None:
    """Return the current git HEAD commit hash, or None if unavailable."""
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    digest = output.strip()
    return digest or None


def save_experiment_metadata(
    run_dir: Path,
    experiment_name: str,
    args: argparse.Namespace | None = None,
) -> Path:
    """Write experiment_metadata.json to run_dir with provenance info.

    Captures git hash, script name, CLI args, and timestamp.

    Returns:
        Path to the written metadata file.
    """
    metadata = {
        "experiment_name": experiment_name,
        "script": sys.argv[0],
        "git_commit_hash": _get_git_commit_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if args is not None:
        metadata["cli_args"] = vars(args)

    path = run_dir / "experiment_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, default=str))
    return path


def upload_run_to_hf(
    run_dir: Path,
    repo_id: str,
    run_name: str | None = None,
    commit_message: str | None = None,
) -> str:
    """Upload a run directory to a HuggingFace dataset repo.

    Args:
        run_dir: Local run directory to upload.
        repo_id: HuggingFace dataset repo ID (e.g. "org/repo-name").
        run_name: Subdirectory name within the repo. Defaults to run_dir.name.
        commit_message: Custom commit message.

    Returns:
        URL of the dataset repo.
    """
    from scripts.utils.hf_hub import login_from_env, upload_folder_to_dataset_repo

    login_from_env()

    if run_name is None:
        run_name = run_dir.name

    if commit_message is None:
        git_hash = _get_git_commit_hash()
        hash_suffix = f" (git: {git_hash[:8]})" if git_hash else ""
        commit_message = f"Upload run: {run_name}{hash_suffix}"

    url = upload_folder_to_dataset_repo(
        local_dir=run_dir,
        repo_id=repo_id,
        path_in_repo=f"runs/{run_name}",
        commit_message=commit_message,
    )
    print(f"  Uploaded to {url}")
    return url
