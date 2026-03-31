"""Diagnose rollout health from raw event files.

This script reconstructs per-conversation outcomes directly from
`stage_events.jsonl` and `canonical_samples.jsonl` without relying on the
rollout logger's aggregate counters. It is intended for auditing partial or
completed rollout runs such as psychometric FA generation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from src_dev.datasets.core import get_run_paths
from src_dev.datasets.io import read_jsonl_tolerant


@dataclass
class TurnAttemptSummary:
    """All raw attempts for a single sample/phase/turn."""

    attempts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def attempt_numbers(self) -> list[int]:
        """Sorted unique attempt numbers seen in raw events."""
        values = {
            int(evt.get("payload", {}).get("attempt_no", -1))
            for evt in self.attempts
            if evt.get("payload", {}).get("attempt_no") is not None
        }
        return sorted(value for value in values if value >= 0)

    @property
    def latest_attempt_no(self) -> int:
        """Largest attempt number seen in raw events."""
        numbers = self.attempt_numbers
        return numbers[-1] if numbers else 0

    @property
    def has_success(self) -> bool:
        """Whether any attempt succeeded."""
        return any(
            evt.get("payload", {}).get("status") == "success" for evt in self.attempts
        )


@dataclass
class SampleSummary:
    """Reconstructed health state for one rollout sample."""

    sample_id: str
    assistant_attempts: int = 0
    user_attempts: int = 0
    assistant_failures: int = 0
    user_failures: int = 0
    assistant_success_turns: set[int] = field(default_factory=set)
    user_success_turns: set[int] = field(default_factory=set)
    assistant_failed_turns: set[int] = field(default_factory=set)
    user_failed_turns: set[int] = field(default_factory=set)
    all_turns: dict[tuple[str, int], TurnAttemptSummary] = field(default_factory=dict)
    terminal_failure: dict[str, Any] | None = None
    canonical_message_count: int = 0


def _load_rollout_config(run_dir: Path) -> dict[str, Any]:
    manifest_path = get_run_paths(run_dir)["manifest"]
    payload = json.loads(manifest_path.read_text())
    return payload.get("progress", {}).get("stage_configs", {}).get("rollout_generation", {})


def _infer_latest_run_dir(root: Path) -> Path:
    candidates = sorted(root.glob("rollouts-*"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No rollout directories found under {root}")
    return candidates[-1]


def _load_canonical_message_counts(run_dir: Path) -> dict[str, int]:
    paths = get_run_paths(run_dir)
    rows, recovered = read_jsonl_tolerant(paths["canonical_samples"])
    if recovered:
        print(
            f"Warning: recovered truncated rows while reading {paths['canonical_samples']}"
        )
    counts: dict[str, int] = {}
    for row in rows:
        sample_id = row["sample_id"]
        counts[sample_id] = len(row.get("messages", []))
    return counts


def _load_stage_events(run_dir: Path) -> list[dict[str, Any]]:
    paths = get_run_paths(run_dir)
    rows, recovered = read_jsonl_tolerant(paths["stage_events"])
    if recovered:
        print(f"Warning: recovered truncated rows while reading {paths['stage_events']}")
    return rows


def _build_sample_summaries(
    stage_events: list[dict[str, Any]],
    canonical_message_counts: dict[str, int],
) -> dict[str, SampleSummary]:
    samples = {
        sample_id: SampleSummary(
            sample_id=sample_id,
            canonical_message_count=message_count,
        )
        for sample_id, message_count in canonical_message_counts.items()
    }

    for event in stage_events:
        sample_id = event.get("sample_id")
        if not sample_id:
            continue
        summary = samples.setdefault(sample_id, SampleSummary(sample_id=sample_id))
        event_type = event.get("event_type")
        payload = event.get("payload", {})

        if event_type in {"assistant_attempt", "user_attempt"}:
            phase = "assistant" if event_type == "assistant_attempt" else "user"
            turn_index = int(payload.get("turn_index", -1))
            summary_key = (phase, turn_index)
            turn_summary = summary.all_turns.setdefault(
                summary_key, TurnAttemptSummary()
            )
            turn_summary.attempts.append(event)

            if phase == "assistant":
                summary.assistant_attempts += 1
            else:
                summary.user_attempts += 1

            status = payload.get("status")
            if status == "success":
                if phase == "assistant":
                    summary.assistant_success_turns.add(turn_index)
                else:
                    summary.user_success_turns.add(turn_index)
            elif status == "failed":
                if phase == "assistant":
                    summary.assistant_failures += 1
                else:
                    summary.user_failures += 1

        elif event_type == "terminal_failure":
            summary.terminal_failure = event

    for summary in samples.values():
        for (phase, turn_index), turn_summary in summary.all_turns.items():
            statuses = [evt.get("payload", {}).get("status") for evt in turn_summary.attempts]
            if "success" not in statuses and "failed" in statuses:
                if phase == "assistant":
                    summary.assistant_failed_turns.add(turn_index)
                else:
                    summary.user_failed_turns.add(turn_index)

    return samples
def _classify_failures(
    samples: dict[str, SampleSummary],
    assistant_max_attempts: int,
    user_max_attempts: int,
) -> dict[str, list[dict[str, Any]]]:
    """Classify failed turns into terminal vs unresolved categories."""
    terminal_exhausted: list[dict[str, Any]] = []
    unresolved_failed: list[dict[str, Any]] = []

    for sample in samples.values():
        terminal_payload = (
            sample.terminal_failure.get("payload", {}) if sample.terminal_failure else {}
        )
        terminal_phase = terminal_payload.get("phase")
        terminal_turn_index = terminal_payload.get("turn_index")
        terminal_reason = terminal_payload.get("reason")

        for phase, failed_turns, max_attempts in (
            ("assistant", sorted(sample.assistant_failed_turns), assistant_max_attempts),
            ("user", sorted(sample.user_failed_turns), user_max_attempts),
        ):
            for turn_index in failed_turns:
                turn_summary = sample.all_turns[(phase, turn_index)]
                row = {
                    "sample_id": sample.sample_id,
                    "phase": phase,
                    "turn_index": turn_index,
                    "attempt_numbers": turn_summary.attempt_numbers,
                    "latest_attempt_no": turn_summary.latest_attempt_no,
                    "raw_event_count": len(turn_summary.attempts),
                    "errors": [
                        attempt.get("payload", {}).get("error")
                        for attempt in turn_summary.attempts
                    ],
                    "canonical_message_count": sample.canonical_message_count,
                }

                if (
                    sample.terminal_failure is not None
                    and terminal_phase == phase
                    and terminal_turn_index == turn_index
                    and terminal_reason in {
                        "assistant_max_attempts_exceeded",
                        "user_max_attempts_exceeded",
                    }
                ):
                    row["terminal_reason"] = terminal_reason
                    terminal_exhausted.append(row)
                elif turn_summary.latest_attempt_no >= max_attempts:
                    row["terminal_reason"] = None
                    unresolved_failed.append(row)

    terminal_exhausted.sort(key=lambda row: (row["phase"], row["turn_index"], row["sample_id"]))
    unresolved_failed.sort(key=lambda row: (row["phase"], row["turn_index"], row["sample_id"]))
    return {
        "terminal_exhausted": terminal_exhausted,
        "unresolved_failed": unresolved_failed,
    }


def _print_run_summary(
    samples: dict[str, SampleSummary],
    assistant_max_attempts: int,
    user_max_attempts: int,
) -> dict[str, list[dict[str, Any]]]:
    assistant_failed_all = sorted(
        (
            sample
            for sample in samples.values()
            if sample.assistant_failed_turns
        ),
        key=lambda sample: sample.sample_id,
    )
    user_failed_all = sorted(
        (
            sample
            for sample in samples.values()
            if sample.user_failed_turns
        ),
        key=lambda sample: sample.sample_id,
    )
    terminal_failures = sorted(
        (
            sample
            for sample in samples.values()
            if sample.terminal_failure is not None
        ),
        key=lambda sample: sample.sample_id,
    )

    assistant_fail_turns = Counter()
    user_fail_turns = Counter()
    assistant_error_counts = Counter()
    user_error_counts = Counter()
    assistant_attempt_counts = []
    user_attempt_counts = []

    for sample in samples.values():
        assistant_attempt_counts.append(sample.assistant_attempts)
        user_attempt_counts.append(sample.user_attempts)
        for turn_index in sample.assistant_failed_turns:
            assistant_fail_turns[turn_index] += 1
            attempts = sample.all_turns[("assistant", turn_index)].attempts
            for attempt in attempts:
                error = attempt.get("payload", {}).get("error")
                if error:
                    assistant_error_counts[error] += 1
        for turn_index in sample.user_failed_turns:
            user_fail_turns[turn_index] += 1
            attempts = sample.all_turns[("user", turn_index)].attempts
            for attempt in attempts:
                error = attempt.get("payload", {}).get("error")
                if error:
                    user_error_counts[error] += 1

    print(f"Samples seen: {len(samples)}")
    print(
        "Assistant attempts per sample (min/median/max): "
        f"{min(assistant_attempt_counts) if assistant_attempt_counts else 0}/"
        f"{median(assistant_attempt_counts) if assistant_attempt_counts else 0}/"
        f"{max(assistant_attempt_counts) if assistant_attempt_counts else 0}"
    )
    print(
        "User attempts per sample (min/median/max): "
        f"{min(user_attempt_counts) if user_attempt_counts else 0}/"
        f"{median(user_attempt_counts) if user_attempt_counts else 0}/"
        f"{max(user_attempt_counts) if user_attempt_counts else 0}"
    )
    print(f"Samples with any assistant turn exhausting retries: {len(assistant_failed_all)}")
    print(f"Samples with any user turn exhausting retries: {len(user_failed_all)}")
    print(f"Samples with terminal_failure event: {len(terminal_failures)}")

    if assistant_fail_turns:
        print("Assistant exhausted-retry turns:", dict(sorted(assistant_fail_turns.items())))
    if user_fail_turns:
        print("User exhausted-retry turns:", dict(sorted(user_fail_turns.items())))
    if assistant_error_counts:
        print("Assistant errors on exhausted turns:")
        for error, count in assistant_error_counts.most_common():
            print(f"  {count}: {error}")
    if user_error_counts:
        print("User errors on exhausted turns:")
        for error, count in user_error_counts.most_common():
            print(f"  {count}: {error}")
    classifications = _classify_failures(
        samples=samples,
        assistant_max_attempts=assistant_max_attempts,
        user_max_attempts=user_max_attempts,
    )
    print(
        "Confirmed terminal max-attempt failures: "
        f"{len(classifications['terminal_exhausted'])}"
    )
    print(
        "Non-terminal failed turns already at configured max attempts: "
        f"{len(classifications['unresolved_failed'])}"
    )
    return classifications


def _print_failed_samples(
    failures: dict[str, list[dict[str, Any]]],
    limit: int,
) -> None:
    if not failures["terminal_exhausted"] and not failures["unresolved_failed"]:
        print("No conversations have exhausted all retries on any turn.")
        return

    if failures["terminal_exhausted"]:
        print(
            f"First {min(limit, len(failures['terminal_exhausted']))} confirmed terminal failures:"
        )
        for row in failures["terminal_exhausted"][:limit]:
            print(json.dumps(row, ensure_ascii=True))

    if failures["unresolved_failed"]:
        print(
            f"First {min(limit, len(failures['unresolved_failed']))} unresolved failed turns at max attempts:"
        )
        for row in failures["unresolved_failed"][:limit]:
            print(json.dumps(row, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose rollout events from a canonical run directory."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Specific rollout run directory. Defaults to latest rollouts-* under scratch/psychometric_fa.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("scratch/psychometric_fa"),
        help="Root directory used when inferring the latest rollout run.",
    )
    parser.add_argument(
        "--show-failed",
        type=int,
        default=20,
        help="How many exhausted-turn sample records to print.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or _infer_latest_run_dir(args.root)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    print(f"Run dir: {run_dir}")
    rollout_config = _load_rollout_config(run_dir)
    failure_policy = rollout_config.get("failure_policy", {})
    assistant_max_attempts = int(failure_policy.get("assistant_max_attempts_per_turn", 0))
    user_max_attempts = int(failure_policy.get("user_max_attempts_per_turn", 0))
    print(
        "Configured max attempts per turn: "
        f"assistant={assistant_max_attempts}, user={user_max_attempts}"
    )
    canonical_message_counts = _load_canonical_message_counts(run_dir)
    stage_events = _load_stage_events(run_dir)
    samples = _build_sample_summaries(stage_events, canonical_message_counts)
    failures = _print_run_summary(
        samples=samples,
        assistant_max_attempts=assistant_max_attempts,
        user_max_attempts=user_max_attempts,
    )
    _print_failed_samples(failures, args.show_failed)


if __name__ == "__main__":
    main()
