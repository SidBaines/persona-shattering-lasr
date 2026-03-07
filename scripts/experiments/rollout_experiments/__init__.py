"""Shared infrastructure for multi-phase rollout experiment scripts.

Provides RolloutExperimentConfig, config builders, phased rollout execution,
per-message evaluation, experiment metadata, and HuggingFace upload.

Usage:
    from scripts.experiments.rollout_experiments import (
        RolloutExperimentConfig, Phase, run_experiment,
    )

    CONFIG = RolloutExperimentConfig(
        scratch_dir=Path("scratch/runs/my_experiment"),
        assistant_model="meta-llama/Llama-3.1-8B-Instruct",
        ...
    )

    def run_baseline():
        run_experiment(CONFIG, "baseline", [
            Phase(num_turns=5),
            Phase(num_turns=5),
        ], evaluations=["count_o"])
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.inference.config import InferenceConfig, LocalProviderConfig, RetryConfig
from scripts.persona_metrics.conversation_eval import (
    ConversationMetricsConfig,
    ConversationMetricsResult,
    MessageSelector,
    run_conversation_metrics,
)
from scripts.rollout_generation.config import (
    FailurePolicyConfig,
    RolloutGenerationConfig,
    UserSimulatorConfig,
)
from scripts.rollout_generation.run import run_rollout_generation
from scripts.utils.hf_hub import login_from_env, upload_folder_to_dataset_repo


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class RolloutExperimentConfig:
    """All knobs for a rollout experiment. Instantiate once at top of script."""

    scratch_dir: Path
    hf_repo: str | None = None

    # Assistant model
    assistant_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    assistant_provider: str = "local"
    assistant_temperature: float = 1.0
    assistant_top_p: float = 0.95
    assistant_max_new_tokens: int = 2048
    assistant_batch_size: int = 32

    # User simulator
    user_model: str = "gpt-4.1-nano-2025-04-14"
    user_provider: str = "openai"
    user_temperature: float = 1.0
    user_top_p: float = 0.95
    user_max_new_tokens: int = 20000
    user_batch_size: int = 16
    user_max_concurrent: int = 64

    # Dataset
    dataset_path: str = "datasets/assistant-axis-extraction-questions.jsonl"
    max_samples: int = 10
    dataset_seed: int | None = None
    num_rollouts: int = 1

    # Experiment
    turns_per_phase: list[int] = field(default_factory=lambda: [5, 5])
    system_prompts: dict[str, str] = field(default_factory=dict)


@dataclass
class Phase:
    """One phase of a multi-phase rollout experiment."""

    num_turns: int
    assistant_system_prompt: str | None = None
    user_simulator: UserSimulatorConfig | None = None


# ── Config builders ───────────────────────────────────────────────────────────


def build_assistant_inference(config: RolloutExperimentConfig) -> InferenceConfig:
    """Build InferenceConfig for the assistant model."""
    return InferenceConfig(
        model=config.assistant_model,
        provider=config.assistant_provider,
        local=LocalProviderConfig(prompt_format="chat", truncate_inputs=False),
        generation=GenerationConfig(
            max_new_tokens=config.assistant_max_new_tokens,
            temperature=config.assistant_temperature,
            top_p=config.assistant_top_p,
            do_sample=True,
            batch_size=config.assistant_batch_size,
        ),
    )


def build_user_simulator(
    config: RolloutExperimentConfig,
    prompt_template: str = "typical_user",
    prompt_format: str = "single_turn_text",
    *,
    provider: str | None = None,
    model: str | None = None,
) -> UserSimulatorConfig:
    """Build UserSimulatorConfig, optionally overriding provider/model."""
    return UserSimulatorConfig(
        provider=provider or config.user_provider,
        model=model or config.user_model,
        prompt_template=prompt_template,
        prompt_format=prompt_format,
        generation=GenerationConfig(
            max_new_tokens=config.user_max_new_tokens,
            temperature=config.user_temperature,
            top_p=config.user_top_p,
            do_sample=True,
            batch_size=config.user_batch_size,
        ),
        max_concurrent=config.user_max_concurrent,
        retry=RetryConfig(),
    )


def build_dataset(config: RolloutExperimentConfig) -> DatasetConfig:
    """Build DatasetConfig from experiment config."""
    return DatasetConfig(
        source="local",
        path=config.dataset_path,
        max_samples=config.max_samples,
        seed=config.dataset_seed,
    )


# ── Orchestration ─────────────────────────────────────────────────────────────


def run_phased_rollout(
    config: RolloutExperimentConfig,
    phases: list[Phase],
    run_dir: Path,
    *,
    user_sim: UserSimulatorConfig | None = None,
) -> None:
    """Execute a sequence of rollout phases on the same run_dir.

    Args:
        config: Experiment configuration.
        phases: List of Phase objects defining each phase.
        run_dir: Shared run directory for all phases.
        user_sim: Default user simulator config (used when phase.user_simulator is None).
    """
    if user_sim is None:
        user_sim = build_user_simulator(config)
    dataset = build_dataset(config)
    assistant = build_assistant_inference(config)

    cumulative_turns = 0
    for phase_idx, phase in enumerate(phases):
        cumulative_turns += phase.num_turns
        phase_user_sim = phase.user_simulator or user_sim

        print(
            f"\n  Phase {phase_idx + 1}/{len(phases)}: "
            f"{phase.num_turns} turns, "
            f"system_prompt={'yes' if phase.assistant_system_prompt else 'no'}, "
            f"user_template={phase_user_sim.prompt_template}"
        )

        rollout_config = RolloutGenerationConfig(
            dataset=dataset,
            run_dir=run_dir,
            num_assistant_turns=cumulative_turns,
            num_rollouts_per_prompt=config.num_rollouts,
            system_prompt=phase.assistant_system_prompt,
            assistant_inference=assistant,
            user_simulator=phase_user_sim,
            failure_policy=FailurePolicyConfig(
                assistant_max_attempts_per_turn=3,
                user_max_attempts_per_turn=3,
            ),
            resume=True,
        )
        _, result = run_rollout_generation(rollout_config)
        print(f"  -> Completed: {result.num_completed}/{result.num_conversations} conversations")


def evaluate_messages(
    run_dir: Path,
    evaluations: list[str],
    *,
    message_selector: MessageSelector | None = None,
) -> ConversationMetricsResult:
    """Run per-message evaluation on a completed rollout."""
    if message_selector is None:
        message_selector = MessageSelector(exclude_seed=True)

    print(f"\n  Evaluating messages with {evaluations}...")
    eval_config = ConversationMetricsConfig(
        evaluations=evaluations,
        run_dir=run_dir,
        message_selector=message_selector,
        output_path=run_dir / "per_message_metrics.jsonl",
    )
    result = run_conversation_metrics(eval_config)

    print(
        f"  -> Evaluated {result.num_messages_evaluated} messages "
        f"across {result.num_conversations} conversations"
    )
    if result.aggregates:
        for key, val in sorted(result.aggregates.items()):
            if isinstance(val, float):
                print(f"     {key}: {val:.4f}")
            elif not isinstance(val, dict):
                print(f"     {key}: {val}")

    grouped = result.aggregates.get("by_prompt_and_role", {})
    if grouped:
        print("\n  Per-prompt/role breakdown:")
        for key, val in sorted(grouped.items()):
            if isinstance(val, float):
                print(f"     {key}: {val:.4f}")

    return result


# ── Provenance ────────────────────────────────────────────────────────────────


def _git_commit_hash() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return output.strip() or None


def save_experiment_metadata(
    config: RolloutExperimentConfig,
    run_dir: Path,
    experiment_name: str,
) -> None:
    """Write experiment_metadata.json and a copy of the calling script to run_dir."""
    metadata: dict[str, Any] = {
        "experiment_name": experiment_name,
        "script": sys.argv[0],
        "git_commit_hash": _git_commit_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": dataclasses.asdict(config),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )
    script_path = Path(sys.argv[0]).resolve()
    if script_path.exists():
        shutil.copy2(script_path, run_dir / script_path.name)


def upload_to_hf(
    config: RolloutExperimentConfig,
    run_dir: Path,
    experiment_name: str,
) -> None:
    """Upload run_dir to HuggingFace under runs/{experiment_name}/."""
    if not config.hf_repo:
        return
    login_from_env()
    git_hash = _git_commit_hash()
    hash_suffix = f" (git: {git_hash[:8]})" if git_hash else ""
    url = upload_folder_to_dataset_repo(
        local_dir=run_dir,
        repo_id=config.hf_repo,
        path_in_repo=f"runs/{experiment_name}",
        commit_message=f"Upload run: {experiment_name}{hash_suffix}",
    )
    print(f"  Uploaded to {url}")


# ── Top-level experiment runner ───────────────────────────────────────────────


def run_experiment(
    config: RolloutExperimentConfig,
    name: str,
    phases: list[Phase],
    evaluations: list[str],
    *,
    user_sim: UserSimulatorConfig | None = None,
) -> ConversationMetricsResult:
    """Run a complete experiment: phased rollout, evaluation, metadata, upload.

    Args:
        config: Experiment configuration.
        name: Experiment name (used for run directory and HF upload path).
        phases: List of Phase objects defining the rollout.
        evaluations: Persona metric names to run on each message (e.g. ["count_o"]).
        user_sim: Optional default user simulator override.

    Returns:
        ConversationMetricsResult with per-message scores and aggregates.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = config.scratch_dir / f"{name}_{timestamp}"

    print(f"\n{'=' * 60}")
    print(f"Experiment: {name}")
    print(f"Run dir: {run_dir}")
    print(f"{'=' * 60}")

    run_phased_rollout(config, phases, run_dir, user_sim=user_sim)
    result = evaluate_messages(run_dir, evaluations)
    save_experiment_metadata(config, run_dir, name)
    upload_to_hf(config, run_dir, name)

    return result
