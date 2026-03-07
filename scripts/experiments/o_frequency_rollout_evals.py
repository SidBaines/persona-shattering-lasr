#!/usr/bin/env python3
"""O-frequency rollout experiments: generate multi-phase conversations with
optional o-frequency system prompting, then evaluate every message with count_o.

Experiments test whether prompting an assistant (or user) to use more/fewer 'o's
during an initial conversation phase affects the assistant's 'o' usage in a
subsequent unprompted phase.

Usage:
    python -m scripts.experiments.o_frequency_rollout_evals \\
        --assistant-model meta-llama/Llama-3.1-8B-Instruct \\
        --assistant-provider local \\
        --turns-per-phase 5 \\
        --max-samples 5

    # Run a single experiment:
    python -m scripts.experiments.o_frequency_rollout_evals \\
        --experiments assistant_o_avoiding

    # Assistant-assistant mode (both sides are LLMs, no user-simulator persona):
    python -m scripts.experiments.o_frequency_rollout_evals \\
        --experiments aa_baseline aa_o_avoiding
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from scripts.persona_metrics import (
    ConversationMetricsConfig,
    MessageSelector,
    run_conversation_metrics,
)
from scripts.rollout_generation import get_system_prompt_template
from scripts.rollout_generation.experiment_utils import (
    add_rollout_cli_args,
    run_phased_rollout,
    save_experiment_metadata,
    upload_run_to_hf,
)


# ── Experiment definitions ────────────────────────────────────────────────────


def _build_experiments(args: argparse.Namespace) -> dict[str, list[dict]]:
    """Build experiment phase definitions.

    Each experiment is a list of phases. Each phase is a dict with:
        - num_turns: number of assistant turns to generate in this phase
        - assistant_system_prompt: system prompt for the assistant (None = no special prompt)
        - user_simulator_overrides: dict of UserSimulatorConfig overrides for this phase
    """
    t = args.turns_per_phase
    o_avoid = get_system_prompt_template("o_avoiding")
    o_enjoy = get_system_prompt_template("o_enjoying")

    # Shared user simulator overrides for assistant-assistant mode:
    # use the assistant model as the "user", with chat_messages format (no user-simulator persona)
    aa_user = {
        "provider": args.assistant_provider,
        "model": args.assistant_model,
        "prompt_format": "chat_messages",
        "prompt_template": "typical_user",
    }

    no_prompt = {"num_turns": t, "assistant_system_prompt": None}

    experiments: dict[str, list[dict]] = {
        # ── Two-phase: multi-turn with and without prompting ──────────────
        "baseline": [no_prompt, no_prompt],
        "assistant_o_enjoying": [
            {"num_turns": t, "assistant_system_prompt": o_enjoy},
            no_prompt,
        ],
        "assistant_o_avoiding": [
            {"num_turns": t, "assistant_system_prompt": o_avoid},
            no_prompt,
        ],
        "user_o_enjoying": [
            {"num_turns": t, "assistant_system_prompt": None, "user_simulator_overrides": {"prompt_template": "o_enjoying_user"}},
            no_prompt,
        ],
        "user_o_avoiding": [
            {"num_turns": t, "assistant_system_prompt": None, "user_simulator_overrides": {"prompt_template": "o_avoiding_user"}},
            no_prompt,
        ],

        # ── Single-turn: one assistant message only ───────────────────────
        "single_baseline": [
            {"num_turns": 1, "assistant_system_prompt": None},
        ],
        "single_o_enjoying": [
            {"num_turns": 1, "assistant_system_prompt": o_enjoy},
        ],
        "single_o_avoiding": [
            {"num_turns": 1, "assistant_system_prompt": o_avoid},
        ],

        # ── Assistant-assistant mode ──────────────────────────────────────
        "aa_baseline": [
            {**no_prompt, "user_simulator_overrides": aa_user},
            {**no_prompt, "user_simulator_overrides": aa_user},
        ],
        "aa_o_enjoying": [
            {"num_turns": t, "assistant_system_prompt": o_enjoy, "user_simulator_overrides": {**aa_user, "prompt_template": "o_enjoying_user"}},
            {**no_prompt, "user_simulator_overrides": aa_user},
        ],
        "aa_o_avoiding": [
            {"num_turns": t, "assistant_system_prompt": o_avoid, "user_simulator_overrides": {**aa_user, "prompt_template": "o_avoiding_user"}},
            {**no_prompt, "user_simulator_overrides": aa_user},
        ],
    }
    return experiments


# ── Main ──────────────────────────────────────────────────────────────────────


def _run_experiment(
    name: str,
    phases: list[dict],
    args: argparse.Namespace,
    scratch_dir: Path,
) -> None:
    """Run one experiment: generate rollout phases, then evaluate all messages."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = scratch_dir / f"{name}_{timestamp}"

    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"Run dir: {run_dir}")
    print(f"Phases: {len(phases)}")
    print(f"{'='*60}")

    run_phased_rollout(phases, args, run_dir)

    # Evaluate all messages with count_o
    print(f"\n  Evaluating all messages with count_o...")
    eval_config = ConversationMetricsConfig(
        evaluations=["count_o"],
        run_dir=run_dir,
        message_selector=MessageSelector(exclude_seed=True),
        output_path=run_dir / "per_message_metrics.jsonl",
    )
    eval_result = run_conversation_metrics(eval_config)

    print(f"  -> Evaluated {eval_result.num_messages_evaluated} messages "
          f"across {eval_result.num_conversations} conversations")
    if eval_result.aggregates:
        for key, val in sorted(eval_result.aggregates.items()):
            if isinstance(val, float):
                print(f"     {key}: {val:.4f}")
            elif not isinstance(val, dict):
                print(f"     {key}: {val}")

    # Print grouped aggregates
    grouped = eval_result.aggregates.get("by_prompt_and_role", {})
    if grouped:
        print(f"\n  Per-prompt/role breakdown:")
        for key, val in sorted(grouped.items()):
            if isinstance(val, float):
                print(f"     {key}: {val:.4f}")

    # Save provenance metadata and optionally upload to HuggingFace
    save_experiment_metadata(run_dir, name, args)
    if args.hf_repo:
        upload_run_to_hf(run_dir, repo_id=args.hf_repo)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="O-frequency rollout experiments with per-message evaluation."
    )
    add_rollout_cli_args(parser)

    # Experiment-specific args
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="Run only these experiments (by name). Default: run all.",
    )
    parser.add_argument("--turns-per-phase", type=int, default=5)
    parser.add_argument("--scratch-dir", type=str, default="scratch/runs/o_frequency")
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="HuggingFace dataset repo ID to upload results (e.g. 'org/repo-name').",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    load_dotenv()

    scratch_dir = Path(args.scratch_dir)
    all_experiments = _build_experiments(args)

    # Filter to requested experiments
    if args.experiments:
        unknown = set(args.experiments) - set(all_experiments)
        if unknown:
            print(f"Unknown experiments: {unknown}")
            print(f"Available: {sorted(all_experiments)}")
            sys.exit(1)
        experiments = {k: all_experiments[k] for k in args.experiments}
    else:
        experiments = all_experiments

    print(f"Running {len(experiments)} experiment(s): {list(experiments.keys())}")

    for name, phases in experiments.items():
        _run_experiment(name, phases, args, scratch_dir)

    print(f"\nAll experiments complete. Results in {scratch_dir}/")


if __name__ == "__main__":
    main()
