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
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.datasets import load_samples, materialize_canonical_samples
from scripts.inference.config import InferenceConfig, LocalProviderConfig, RetryConfig
from scripts.persona_metrics.config import PersonaMetricSpec
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
from scripts.rollout_generation.prompts import get_user_simulator_instruction
from scripts.rollout_generation.run import run_rollout_generation
from scripts.utils.hf_hub import login_from_env, upload_folder_to_dataset_repo


def _prompt_hash(text: str) -> str:
    """SHA-256 hash (first 16 chars) matching _system_prompt_hash in run.py."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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
    preloaded_model: tuple | None = None,
) -> None:
    """Execute a sequence of rollout phases on the same run_dir.

    Args:
        config: Experiment configuration.
        phases: List of Phase objects defining each phase.
        run_dir: Shared run directory for all phases.
        user_sim: Default user simulator config (used when phase.user_simulator is None).
        preloaded_model: Optional ``(model, tokenizer)`` tuple to skip model loading.
    """
    if user_sim is None:
        user_sim = build_user_simulator(config)
    dataset = build_dataset(config)
    assistant = build_assistant_inference(config)
    if preloaded_model is not None:
        assistant = assistant.model_copy(
            update={"local": assistant.local.model_copy(update={"preloaded_model": preloaded_model})}
        )

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

        is_last_phase = phase_idx == len(phases) - 1
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
            skip_final_user_turn=is_last_phase,
            resume=True,
        )
        _, result = run_rollout_generation(rollout_config)
        print(
            f"  -> Completed: {result.num_completed}/{result.num_conversations} conversations"
        )


def evaluate_messages(
    run_dir: Path,
    evaluations: list[str | PersonaMetricSpec],
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


# ── Clean exports ─────────────────────────────────────────────────────────────


def _message_list_for_sample(
    sample: Any,
    scores_by_msg: dict[str, dict[str, Any]] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Extract seed_input and list of message dicts from a sample. Optionally merge scores by message_id."""
    seed_input = None
    messages = []
    for msg in sample.messages:
        meta = msg.message_metadata or {}
        source = meta.get("source_stage", "")
        if source == "seed":
            seed_input = msg.content
            continue
        msg_entry: dict[str, Any] = {
            "role": msg.role,
            "content": msg.content,
            "turn_index": meta.get("turn_index"),
            "system_prompt_hash": meta.get("system_prompt_hash")
            or meta.get("active_system_prompt"),
            "source": source,
        }
        if scores_by_msg:
            msg_scores = scores_by_msg.get(msg.message_id)
            if msg_scores:
                msg_entry["scores"] = msg_scores
        messages.append(msg_entry)
    return seed_input, messages


def export_rollouts(run_dir: Path) -> Path:
    """Write rollouts.jsonl: one line per seed with messages dict keyed by rollout index."""
    materialize_canonical_samples(run_dir)
    samples = load_samples(run_dir)

    groups: dict[str, list[Any]] = {}
    for s in samples:
        seed_id = s.input_group_id or s.sample_id
        groups.setdefault(seed_id, []).append(s)
    for group in groups.values():
        group.sort(key=lambda s: s.response_index)

    entries = []
    for seed_id, group in groups.items():
        seed_input = None
        messages_by_rollout: dict[str, list[dict[str, Any]]] = {}
        for sample in group:
            si, msg_list = _message_list_for_sample(sample)
            if seed_input is None:
                seed_input = si
            messages_by_rollout[str(sample.response_index)] = msg_list
        entries.append(
            {
                "seed_id": seed_id,
                "seed_input": seed_input,
                "messages": messages_by_rollout,
            }
        )

    out_path = run_dir / "rollouts.jsonl"
    out_path.write_text("\n".join(json.dumps(e, default=str) for e in entries) + "\n")
    print(f"\n  Wrote {len(entries)} rollouts to {out_path}")
    return out_path


def export_evaluated_rollouts(
    run_dir: Path,
    eval_result: ConversationMetricsResult,
) -> Path:
    """Write rollouts_evaluated.jsonl: one line per seed, messages dict keyed by rollout index, scores merged by message_id."""
    materialize_canonical_samples(run_dir)
    samples = load_samples(run_dir)

    scores_by_msg: dict[str, dict[str, Any]] = {}
    for item in eval_result.per_message_scores:
        scores_by_msg[item["message_id"]] = item.get("scores", {})

    groups: dict[str, list[Any]] = {}
    for s in samples:
        seed_id = s.input_group_id or s.sample_id
        groups.setdefault(seed_id, []).append(s)
    for group in groups.values():
        group.sort(key=lambda s: s.response_index)

    entries = []
    for seed_id, group in groups.items():
        seed_input = None
        messages_by_rollout: dict[str, list[dict[str, Any]]] = {}
        for sample in group:
            si, msg_list = _message_list_for_sample(sample, scores_by_msg=scores_by_msg)
            if seed_input is None:
                seed_input = si
            messages_by_rollout[str(sample.response_index)] = msg_list
        entries.append(
            {
                "seed_id": seed_id,
                "seed_input": seed_input,
                "messages": messages_by_rollout,
            }
        )

    out_path = run_dir / "rollouts_evaluated.jsonl"
    out_path.write_text("\n".join(json.dumps(e, default=str) for e in entries) + "\n")
    print(f"  Wrote {len(entries)} evaluated rollouts to {out_path}")
    return out_path


# ── Provenance ────────────────────────────────────────────────────────────────


def _git_commit_hash() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return output.strip() or None


def _build_system_prompts_map(
    config: RolloutExperimentConfig,
    phases: list[Phase],
) -> dict[str, str]:
    """Build a hash->text map for all system prompts used in this experiment."""
    prompts: dict[str, str] = {}
    for text in config.system_prompts.values():
        prompts[_prompt_hash(text)] = text
    for phase in phases:
        if phase.assistant_system_prompt:
            prompts[_prompt_hash(phase.assistant_system_prompt)] = (
                phase.assistant_system_prompt
            )
        if phase.user_simulator:
            try:
                text = get_user_simulator_instruction(
                    phase.user_simulator.prompt_template
                )
                prompts[_prompt_hash(text)] = text
            except ValueError:
                pass
    default_user_sim = get_user_simulator_instruction("typical_user")
    prompts[_prompt_hash(default_user_sim)] = default_user_sim
    return prompts


def save_experiment_metadata(
    config: RolloutExperimentConfig,
    run_dir: Path,
    experiment_name: str,
    phases: list[Phase],
) -> None:
    """Write experiment_metadata.json and a copy of the calling script to run_dir."""
    metadata: dict[str, Any] = {
        "experiment_name": experiment_name,
        "script": sys.argv[0],
        "git_commit_hash": _git_commit_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": dataclasses.asdict(config),
        "system_prompts": _build_system_prompts_map(config, phases),
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
    """Upload run_dir to HuggingFace, mirroring scratch_dir structure under the repo."""
    if not config.hf_repo:
        return
    login_from_env()
    git_hash = _git_commit_hash()
    hash_suffix = f" (git: {git_hash[:8]})" if git_hash else ""
    run_name = run_dir.name
    # Mirror scratch path: scratch/runs_zero_lora/... -> runs_zero_lora/.../run_name
    scratch_dir = config.scratch_dir.resolve()
    parts = list(scratch_dir.parts)
    if "scratch" in parts:
        prefix = "/".join(parts[parts.index("scratch") + 1 :])
    else:
        prefix = "runs"
    path_in_repo = f"{prefix}/{run_name}"
    url = upload_folder_to_dataset_repo(
        local_dir=run_dir,
        repo_id=config.hf_repo,
        path_in_repo=path_in_repo,
        commit_message=f"Upload run: {run_name}{hash_suffix}",
        ignore_patterns=[
            "datasets/*",
            "events/*",
            "exports/*",
            "per_message_metrics.jsonl",
        ],
    )
    print(f"  Uploaded to {url}")


# ── Top-level experiment runner ───────────────────────────────────────────────


def run_experiment(
    config: RolloutExperimentConfig,
    name: str,
    phases: list[Phase],
    evaluations: list[str | PersonaMetricSpec],
    *,
    user_sim: UserSimulatorConfig | None = None,
    preloaded_model: tuple | None = None,
) -> ConversationMetricsResult:
    """Run a complete experiment: phased rollout, evaluation, metadata, upload.

    Args:
        config: Experiment configuration.
        name: Experiment name (used for run directory and HF upload path).
        phases: List of Phase objects defining the rollout.
        evaluations: Persona metrics to run on each message (e.g. ["count_o"]).
        user_sim: Optional default user simulator override.
        preloaded_model: Optional ``(model, tokenizer)`` tuple to skip model loading.

    Returns:
        ConversationMetricsResult with per-message scores and aggregates.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = config.scratch_dir / f"{name}_{timestamp}"

    print(f"\n{'=' * 60}")
    print(f"Experiment: {name}")
    print(f"Run dir: {run_dir}")
    print(f"{'=' * 60}")

    run_phased_rollout(config, phases, run_dir, user_sim=user_sim, preloaded_model=preloaded_model)
    export_rollouts(run_dir)
    result = evaluate_messages(run_dir, evaluations)
    export_evaluated_rollouts(run_dir, result)
    save_experiment_metadata(config, run_dir, name, phases)
    upload_to_hf(config, run_dir, name)

    return result
