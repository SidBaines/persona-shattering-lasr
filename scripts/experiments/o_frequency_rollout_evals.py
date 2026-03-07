#!/usr/bin/env python3
"""O-frequency rollout experiments: generate multi-phase conversations with
optional o-frequency system prompting, then evaluate every message with count_o.

Experiments test whether prompting an assistant (or user) to use more/fewer 'o's
during an initial conversation phase affects the assistant's 'o' usage in a
subsequent unprompted phase.

Edit the constants below to configure models, dataset, and generation settings.
Edit main() to select which experiments to run.

Usage:
    python -m scripts.experiments.o_frequency_rollout_evals
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.inference.config import InferenceConfig, LocalProviderConfig, RetryConfig
from scripts.persona_metrics import (
    ConversationMetricsConfig,
    MessageSelector,
    run_conversation_metrics,
)
from scripts.rollout_generation import get_system_prompt_template
from scripts.rollout_generation.config import (
    FailurePolicyConfig,
    RolloutGenerationConfig,
    UserSimulatorConfig,
)
from scripts.rollout_generation.run import run_rollout_generation

# ── Configuration ─────────────────────────────────────────────────────────────

SCRATCH_DIR = Path("scratch/runs/o_frequency")
HF_REPO = "lasr-spelling/o-frequency-rollouts"
TURNS_PER_PHASE = 5

# Dataset
DATASET_PATH = "datasets/assistant-axis-extraction-questions.jsonl"
MAX_SAMPLES = 5
DATASET_SEED = None
NUM_ROLLOUTS = 1

# Assistant model
ASSISTANT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ASSISTANT_PROVIDER = "local"
ASSISTANT_TEMPERATURE = 1.0
ASSISTANT_TOP_P = 0.95
ASSISTANT_MAX_NEW_TOKENS = 2048
ASSISTANT_BATCH_SIZE = 32

# User simulator
USER_MODEL = "gpt-4.1-nano-2025-04-14"
USER_PROVIDER = "openai"
USER_TEMPERATURE = 1.0
USER_TOP_P = 0.95
USER_MAX_NEW_TOKENS = 20000
USER_BATCH_SIZE = 16
USER_MAX_CONCURRENT = 64

# System prompts (resolved from templates)
O_AVOIDING_PROMPT = get_system_prompt_template("o_avoiding")
O_ENJOYING_PROMPT = get_system_prompt_template("o_enjoying")


# ── Config builders ───────────────────────────────────────────────────────────


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _assistant_config() -> InferenceConfig:
    return InferenceConfig(
        model=ASSISTANT_MODEL,
        provider=ASSISTANT_PROVIDER,
        local=LocalProviderConfig(prompt_format="chat", truncate_inputs=False),
        generation=GenerationConfig(
            max_new_tokens=ASSISTANT_MAX_NEW_TOKENS,
            temperature=ASSISTANT_TEMPERATURE,
            top_p=ASSISTANT_TOP_P,
            do_sample=True,
            batch_size=ASSISTANT_BATCH_SIZE,
        ),
    )


def _user_sim_config(
    prompt_template: str = "typical_user",
    prompt_format: str = "single_turn_text",
    *,
    provider: str | None = None,
    model: str | None = None,
) -> UserSimulatorConfig:
    return UserSimulatorConfig(
        provider=provider or USER_PROVIDER,
        model=model or USER_MODEL,
        prompt_template=prompt_template,
        prompt_format=prompt_format,
        generation=GenerationConfig(
            max_new_tokens=USER_MAX_NEW_TOKENS,
            temperature=USER_TEMPERATURE,
            top_p=USER_TOP_P,
            do_sample=True,
            batch_size=USER_BATCH_SIZE,
        ),
        max_concurrent=USER_MAX_CONCURRENT,
        retry=RetryConfig(),
    )


def _dataset_config() -> DatasetConfig:
    return DatasetConfig(
        source="local",
        path=DATASET_PATH,
        max_samples=MAX_SAMPLES,
        seed=DATASET_SEED,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────


def _run_phased_rollout(
    phases: list[dict],
    run_dir: Path,
    dataset: DatasetConfig,
    assistant: InferenceConfig,
    user_sim: UserSimulatorConfig,
) -> None:
    """Execute a sequence of rollout phases on the same run_dir."""
    cumulative_turns = 0
    for phase_idx, phase in enumerate(phases):
        cumulative_turns += phase["num_turns"]

        phase_user_sim = phase.get("user_simulator", user_sim)

        print(
            f"\n  Phase {phase_idx + 1}/{len(phases)}: "
            f"{phase['num_turns']} turns, "
            f"system_prompt={'yes' if phase.get('assistant_system_prompt') else 'no'}, "
            f"user_template={phase_user_sim.prompt_template}"
        )

        config = RolloutGenerationConfig(
            dataset=dataset,
            run_dir=run_dir,
            num_assistant_turns=cumulative_turns,
            num_rollouts_per_prompt=NUM_ROLLOUTS,
            system_prompt=phase.get("assistant_system_prompt"),
            assistant_inference=assistant,
            user_simulator=phase_user_sim,
            failure_policy=FailurePolicyConfig(
                assistant_max_attempts_per_turn=3,
                user_max_attempts_per_turn=3,
            ),
            resume=True,
        )
        _, result = run_rollout_generation(config)
        print(
            f"  -> Completed: {result.num_completed}/{result.num_conversations} conversations"
        )


def _evaluate(run_dir: Path) -> None:
    """Run count_o per-message evaluation on a completed rollout."""
    print("\n  Evaluating all messages with count_o...")
    eval_config = ConversationMetricsConfig(
        evaluations=["count_o"],
        run_dir=run_dir,
        message_selector=MessageSelector(exclude_seed=True),
        output_path=run_dir / "per_message_metrics.jsonl",
    )
    eval_result = run_conversation_metrics(eval_config)

    print(
        f"  -> Evaluated {eval_result.num_messages_evaluated} messages "
        f"across {eval_result.num_conversations} conversations"
    )
    if eval_result.aggregates:
        for key, val in sorted(eval_result.aggregates.items()):
            if isinstance(val, float):
                print(f"     {key}: {val:.4f}")
            elif not isinstance(val, dict):
                print(f"     {key}: {val}")

    grouped = eval_result.aggregates.get("by_prompt_and_role", {})
    if grouped:
        print("\n  Per-prompt/role breakdown:")
        for key, val in sorted(grouped.items()):
            if isinstance(val, float):
                print(f"     {key}: {val:.4f}")


def _git_commit_hash() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True,
        )
    except Exception:
        return None
    return output.strip() or None


def _save_metadata(run_dir: Path, experiment_name: str) -> None:
    """Write experiment_metadata.json and a copy of this script to run_dir."""
    metadata = {
        "experiment_name": experiment_name,
        "script": sys.argv[0],
        "git_commit_hash": _git_commit_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "assistant_model": ASSISTANT_MODEL,
            "assistant_provider": ASSISTANT_PROVIDER,
            "user_model": USER_MODEL,
            "user_provider": USER_PROVIDER,
            "dataset_path": DATASET_PATH,
            "max_samples": MAX_SAMPLES,
            "turns_per_phase": TURNS_PER_PHASE,
            "num_rollouts": NUM_ROLLOUTS,
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )
    shutil.copy2(__file__, run_dir / Path(__file__).name)


def _upload_to_hf(run_dir: Path, experiment_name: str) -> None:
    """Upload run_dir to HuggingFace under runs/{experiment_name}/."""
    if not HF_REPO:
        return
    from scripts.utils.hf_hub import login_from_env, upload_folder_to_dataset_repo

    login_from_env()
    git_hash = _git_commit_hash()
    hash_suffix = f" (git: {git_hash[:8]})" if git_hash else ""
    url = upload_folder_to_dataset_repo(
        local_dir=run_dir,
        repo_id=HF_REPO,
        path_in_repo=f"runs/{experiment_name}",
        commit_message=f"Upload run: {experiment_name}{hash_suffix}",
    )
    print(f"  Uploaded to {url}")


def _run_experiment(name: str, phases: list[dict]) -> None:
    """Run one experiment: generate rollout phases, evaluate, save metadata, upload."""
    run_dir = SCRATCH_DIR / f"{name}_{_timestamp()}"

    print(f"\n{'=' * 60}")
    print(f"Experiment: {name}")
    print(f"Run dir: {run_dir}")
    print(f"{'=' * 60}")

    _run_phased_rollout(
        phases, run_dir, _dataset_config(), _assistant_config(), _user_sim_config()
    )
    _evaluate(run_dir)
    _save_metadata(run_dir, name)
    _upload_to_hf(run_dir, name)


# ── Experiment functions ──────────────────────────────────────────────────────


def run_baseline() -> None:
    """Two-phase baseline: no prompting in either phase."""
    _run_experiment(
        "baseline",
        [
            {"num_turns": TURNS_PER_PHASE, "assistant_system_prompt": None},
            {"num_turns": TURNS_PER_PHASE, "assistant_system_prompt": None},
        ],
    )


def run_assistant_o_enjoying() -> None:
    """Phase 1: assistant prompted to enjoy 'o'. Phase 2: unprompted."""
    _run_experiment(
        "assistant_o_enjoying",
        [
            {
                "num_turns": TURNS_PER_PHASE,
                "assistant_system_prompt": O_ENJOYING_PROMPT,
            },
            {"num_turns": TURNS_PER_PHASE, "assistant_system_prompt": None},
        ],
    )


def run_assistant_o_avoiding() -> None:
    """Phase 1: assistant prompted to avoid 'o'. Phase 2: unprompted."""
    _run_experiment(
        "assistant_o_avoiding",
        [
            {
                "num_turns": TURNS_PER_PHASE,
                "assistant_system_prompt": O_AVOIDING_PROMPT,
            },
            {"num_turns": TURNS_PER_PHASE, "assistant_system_prompt": None},
        ],
    )


def run_user_o_enjoying() -> None:
    """Phase 1: user simulator prompted to enjoy 'o'. Phase 2: unprompted."""
    name = "user_o_enjoying"
    run_dir = SCRATCH_DIR / f"{name}_{_timestamp()}"

    print(f"\n{'=' * 60}")
    print(f"Experiment: {name}")
    print(f"Run dir: {run_dir}")
    print(f"{'=' * 60}")

    phases = [
        {
            "num_turns": TURNS_PER_PHASE,
            "assistant_system_prompt": None,
            "user_simulator": _user_sim_config("o_enjoying_user"),
        },
        {"num_turns": TURNS_PER_PHASE, "assistant_system_prompt": None},
    ]
    _run_phased_rollout(
        phases, run_dir, _dataset_config(), _assistant_config(), _user_sim_config()
    )
    _evaluate(run_dir)
    _save_metadata(run_dir, name)
    _upload_to_hf(run_dir, name)


def run_user_o_avoiding() -> None:
    """Phase 1: user simulator prompted to avoid 'o'. Phase 2: unprompted."""
    name = "user_o_avoiding"
    run_dir = SCRATCH_DIR / f"{name}_{_timestamp()}"

    print(f"\n{'=' * 60}")
    print(f"Experiment: {name}")
    print(f"Run dir: {run_dir}")
    print(f"{'=' * 60}")

    phases = [
        {
            "num_turns": TURNS_PER_PHASE,
            "assistant_system_prompt": None,
            "user_simulator": _user_sim_config("o_avoiding_user"),
        },
        {"num_turns": TURNS_PER_PHASE, "assistant_system_prompt": None},
    ]
    _run_phased_rollout(
        phases, run_dir, _dataset_config(), _assistant_config(), _user_sim_config()
    )
    _evaluate(run_dir)
    _save_metadata(run_dir, name)
    _upload_to_hf(run_dir, name)


def run_single_baseline() -> None:
    """Single-turn baseline: one assistant message, no prompting."""
    _run_experiment(
        "single_baseline",
        [
            {"num_turns": 1, "assistant_system_prompt": None},
        ],
    )


def run_single_o_enjoying() -> None:
    """Single-turn: one assistant message with o-enjoying system prompt."""
    _run_experiment(
        "single_o_enjoying",
        [
            {"num_turns": 1, "assistant_system_prompt": O_ENJOYING_PROMPT},
        ],
    )


def run_single_o_avoiding() -> None:
    """Single-turn: one assistant message with o-avoiding system prompt."""
    _run_experiment(
        "single_o_avoiding",
        [
            {"num_turns": 1, "assistant_system_prompt": O_AVOIDING_PROMPT},
        ],
    )


def run_aa_baseline() -> None:
    """Assistant-assistant baseline: both sides are LLMs, no behavioral prompting."""
    name = "aa_baseline"
    run_dir = SCRATCH_DIR / f"{name}_{_timestamp()}"
    aa_user = _user_sim_config(
        "typical_user",
        "chat_messages",
        provider=ASSISTANT_PROVIDER,
        model=ASSISTANT_MODEL,
    )

    print(f"\n{'=' * 60}")
    print(f"Experiment: {name}")
    print(f"Run dir: {run_dir}")
    print(f"{'=' * 60}")

    phases = [
        {"num_turns": TURNS_PER_PHASE, "assistant_system_prompt": None},
        {"num_turns": TURNS_PER_PHASE, "assistant_system_prompt": None},
    ]
    _run_phased_rollout(
        phases, run_dir, _dataset_config(), _assistant_config(), aa_user
    )
    _evaluate(run_dir)
    _save_metadata(run_dir, name)
    _upload_to_hf(run_dir, name)


def run_aa_o_enjoying() -> None:
    """Assistant-assistant: phase 1 both prompted to enjoy 'o', phase 2 unprompted."""
    name = "aa_o_enjoying"
    run_dir = SCRATCH_DIR / f"{name}_{_timestamp()}"
    aa_user = _user_sim_config(
        "typical_user",
        "chat_messages",
        provider=ASSISTANT_PROVIDER,
        model=ASSISTANT_MODEL,
    )
    aa_user_prompted = _user_sim_config(
        "o_enjoying_user",
        "chat_messages",
        provider=ASSISTANT_PROVIDER,
        model=ASSISTANT_MODEL,
    )

    print(f"\n{'=' * 60}")
    print(f"Experiment: {name}")
    print(f"Run dir: {run_dir}")
    print(f"{'=' * 60}")

    phases = [
        {
            "num_turns": TURNS_PER_PHASE,
            "assistant_system_prompt": O_ENJOYING_PROMPT,
            "user_simulator": aa_user_prompted,
        },
        {
            "num_turns": TURNS_PER_PHASE,
            "assistant_system_prompt": None,
            "user_simulator": aa_user,
        },
    ]
    _run_phased_rollout(
        phases, run_dir, _dataset_config(), _assistant_config(), aa_user
    )
    _evaluate(run_dir)
    _save_metadata(run_dir, name)
    _upload_to_hf(run_dir, name)


def run_aa_o_avoiding() -> None:
    """Assistant-assistant: phase 1 both prompted to avoid 'o', phase 2 unprompted."""
    name = "aa_o_avoiding"
    run_dir = SCRATCH_DIR / f"{name}_{_timestamp()}"
    aa_user = _user_sim_config(
        "typical_user",
        "chat_messages",
        provider=ASSISTANT_PROVIDER,
        model=ASSISTANT_MODEL,
    )
    aa_user_prompted = _user_sim_config(
        "o_avoiding_user",
        "chat_messages",
        provider=ASSISTANT_PROVIDER,
        model=ASSISTANT_MODEL,
    )

    print(f"\n{'=' * 60}")
    print(f"Experiment: {name}")
    print(f"Run dir: {run_dir}")
    print(f"{'=' * 60}")

    phases = [
        {
            "num_turns": TURNS_PER_PHASE,
            "assistant_system_prompt": O_AVOIDING_PROMPT,
            "user_simulator": aa_user_prompted,
        },
        {
            "num_turns": TURNS_PER_PHASE,
            "assistant_system_prompt": None,
            "user_simulator": aa_user,
        },
    ]
    _run_phased_rollout(
        phases, run_dir, _dataset_config(), _assistant_config(), aa_user
    )
    _evaluate(run_dir)
    _save_metadata(run_dir, name)
    _upload_to_hf(run_dir, name)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()

    run_baseline()
    # run_assistant_o_enjoying()
    # run_assistant_o_avoiding()
    # run_user_o_enjoying()
    # run_user_o_avoiding()
    # run_single_baseline()
    # run_single_o_enjoying()
    # run_single_o_avoiding()
    # run_aa_baseline()
    # run_aa_o_enjoying()
    # run_aa_o_avoiding()

    print(f"\nAll experiments complete. Results in {SCRATCH_DIR}/")


if __name__ == "__main__":
    main()
