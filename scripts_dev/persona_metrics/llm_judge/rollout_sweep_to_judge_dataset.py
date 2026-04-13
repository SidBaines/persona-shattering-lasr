"""Convert a LoRA scale sweep output into a judge-compatible all_responses.jsonl.

The rollout sweep produces per-(scale, condition) cell directories:

    scratch/<sweep_run>/
        sweep_config.json
        scale_+0.00/<condition>/
            <experiment_subdir>/
                rollouts.jsonl      ← grouped by seed, full message history
        scale_+1.00/<condition>/
            ...

The judge pipeline expects a flat ``all_responses.jsonl`` where each row is
one response with fields: ``response_id``, ``condition``, ``question``,
``response``, ``assistant_model``, etc.

This script flattens all (scale, condition) cells into a single dataset that
the judge can consume.  The ``condition`` field encodes both dimensions:
``<condition>@scale_<value>``  e.g.  ``no_prompt@scale_+1.00``.

This lets condition_metrics in the analysis group by condition naturally, and
you can plot trait/coherence score vs scale factor by parsing the condition
field.

Usage
-----

Convert a sweep directory to a judge dataset::

    uv run python scripts_dev/persona_metrics/rollout_sweep_to_judge_dataset.py \\
        --sweep-dir scratch/20260309_181645_o_avoiding \\
        --output scratch/judge_datasets/o_avoiding_sweep.jsonl

Then run the judge stage against it::

    uv run python scripts_dev/persona_metrics/ocean_judge_calibration.py \\
        --trait neuroticism --stage judge \\
        --dataset scratch/judge_datasets/o_avoiding_sweep.jsonl

Or run the coherence judge::

    uv run python scripts_dev/persona_metrics/coherence_calibration.py \\
        --stage judge \\
        --dataset scratch/judge_datasets/o_avoiding_sweep.jsonl

Filtering
---------

Use ``--conditions`` and ``--scales`` to include only a subset::

    uv run python scripts_dev/persona_metrics/rollout_sweep_to_judge_dataset.py \\
        --sweep-dir scratch/20260309_181645_o_avoiding \\
        --output scratch/judge_datasets/o_avoiding_subset.jsonl \\
        --scales 0.0 0.5 1.0 1.5 2.0 \\
        --conditions no_prompt o_avoiding

Output schema
-------------

Each row in the output JSONL::

    {
        "response_id":      "<condition>@scale_<s>:<seed_id>:<rollout_idx>",
        "condition":        "<condition>@scale_<s>",   // e.g. "no_prompt@scale_+1.00"
        "scale":            1.0,                        // float, for easy grouping
        "condition_name":   "no_prompt",                // without scale suffix
        "seed_id":          "sample_abc123",
        "sample_id":        "sample_abc123",
        "input_group_id":   "sample_abc123",
        "response_index":   0,
        "prompt_row_index": -1,
        "prompt_id":        "sample_abc123",
        "question":         "...",
        "response":         "...",
        "assistant_model":  "",   // not available in rollout.jsonl; fill if known
        "assistant_provider": "local",
        "system_prompt_ref": "<system_prompt_hash or empty>",
    }

Note: ``assistant_model`` is not stored in rollouts.jsonl.  Pass ``--model``
to populate it, or leave it empty (the judge doesn't use it for scoring).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_scale_from_label(label: str) -> float | None:
    """Parse float from a label like 'scale_+1.00' or 'scale_-0.50'."""
    if not label.startswith("scale_"):
        return None
    try:
        return float(label[len("scale_"):])
    except ValueError:
        return None


def _condition_label(condition_name: str, scale: float) -> str:
    return f"{condition_name}@scale_{scale:+.2f}"


def _find_rollouts_file(condition_dir: Path) -> Path | None:
    """Find the rollouts JSONL for one sweep cell across known layouts."""
    for candidate in (
        condition_dir / "rollouts" / "rollouts.jsonl",
        condition_dir / "rollouts.jsonl",
    ):
        if candidate.exists():
            return candidate

    # Older experiments sometimes wrote rollouts under a timestamped child dir.
    for subdir in sorted(condition_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for relative in (Path("rollouts") / "rollouts.jsonl", Path("rollouts.jsonl")):
            candidate = subdir / relative
            if candidate.exists():
                return candidate
    return None


def _flatten_rollouts_file(
    rollouts_path: Path,
    condition_name: str,
    scale: float,
    assistant_model: str,
) -> list[dict]:
    """Flatten one rollouts.jsonl cell into judge-format rows."""
    rows = []
    condition = _condition_label(condition_name, scale)
    with open(rollouts_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            seed_id = str(record.get("seed_id", ""))
            seed_input = str(record.get("seed_input", ""))
            messages = record.get("messages", {})
            for rollout_idx_str, turn_list in messages.items():
                rollout_idx = int(rollout_idx_str)
                # Extract last assistant message
                assistant_msgs = [m for m in turn_list if m.get("role") == "assistant"]
                if not assistant_msgs:
                    continue
                response = assistant_msgs[-1]["content"]
                system_prompt_ref = assistant_msgs[-1].get("system_prompt_hash", "")
                response_id = f"{condition}:{seed_id}:{rollout_idx}"
                rows.append({
                    "response_id": response_id,
                    "condition": condition,
                    "scale": scale,
                    "condition_name": condition_name,
                    "seed_id": seed_id,
                    "sample_id": seed_id,
                    "input_group_id": seed_id,
                    "response_index": rollout_idx,
                    "prompt_row_index": -1,
                    "prompt_id": seed_id,
                    "question": seed_input,
                    "response": response,
                    "assistant_model": assistant_model,
                    "assistant_provider": "local",
                    "system_prompt_ref": system_prompt_ref,
                })
    return rows


def _collect_rows(
    sweep_dir: Path,
    *,
    scales: list[float] | None,
    conditions: list[str] | None,
    assistant_model: str,
) -> list[dict]:
    """Collect judge-format rows from one sweep directory (no IO to output)."""
    all_rows: list[dict] = []
    if not sweep_dir.exists():
        print(f"  sweep dir missing, skipping: {sweep_dir}")
        return all_rows
    for scale_dir in sorted(sweep_dir.iterdir()):
        if not scale_dir.is_dir():
            continue
        scale = _parse_scale_from_label(scale_dir.name)
        if scale is None:
            continue
        if scales is not None and not any(abs(scale - s) < 1e-6 for s in scales):
            continue

        for condition_dir in sorted(scale_dir.iterdir()):
            if not condition_dir.is_dir():
                continue
            condition_name = condition_dir.name
            if conditions is not None and condition_name not in conditions:
                continue

            rollouts_path = _find_rollouts_file(condition_dir)
            if rollouts_path is None:
                print(f"  skipping {scale_dir.name}/{condition_name}: no rollouts.jsonl found")
                continue

            rows = _flatten_rollouts_file(
                rollouts_path, condition_name, scale, assistant_model
            )
            all_rows.extend(rows)
            print(f"  {scale_dir.name}/{condition_name}: {len(rows)} rows")
    return all_rows


def convert_sweep(
    sweep_dir: Path,
    output_path: Path,
    *,
    scales: list[float] | None = None,
    conditions: list[str] | None = None,
    assistant_model: str = "",
) -> int:
    """Convert a rollout sweep directory to a flat judge dataset.

    Args:
        sweep_dir: Root directory of a ``run_rollout_sweep`` run.
        output_path: Destination JSONL file path.
        scales: If provided, only include these scale values (float).
        conditions: If provided, only include these condition names.
        assistant_model: Model ID to populate the ``assistant_model`` field.

    Returns:
        Total number of rows written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = _collect_rows(
        sweep_dir,
        scales=scales,
        conditions=conditions,
        assistant_model=assistant_model,
    )

    with open(output_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(all_rows)} rows → {output_path}")
    return len(all_rows)


def convert_sweeps(
    sources: list[tuple[Path, list[float] | None]],
    output_path: Path,
    *,
    conditions: list[str] | None = None,
    assistant_model: str = "",
) -> int:
    """Convert multiple sweep directories into one merged judge dataset.

    Each source is ``(sweep_dir, scales_filter)``. Rows from all sources are
    concatenated into a single JSONL. Useful when the baseline cell lives at a
    shared path and the non-zero scales live at a per-sweep path.

    Args:
        sources: List of ``(sweep_dir, scales)`` pairs.
        output_path: Destination JSONL file path.
        conditions: Optional condition-name filter applied to every source.
        assistant_model: Model ID to populate the ``assistant_model`` field.

    Returns:
        Total number of rows written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for sweep_dir, scales in sources:
        print(f"Reading {sweep_dir} (scales={scales})")
        all_rows.extend(
            _collect_rows(
                sweep_dir,
                scales=scales,
                conditions=conditions,
                assistant_model=assistant_model,
            )
        )

    with open(output_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(all_rows)} rows → {output_path}")
    return len(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a LoRA scale sweep output to a flat judge dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sweep-dir", type=Path, required=True,
                        help="Root directory of a run_rollout_sweep run.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination all_responses.jsonl path.")
    parser.add_argument("--scales", type=float, nargs="*", default=None,
                        help="Subset of scale values to include (default: all).")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="Subset of condition names to include (default: all).")
    parser.add_argument("--model", default="",
                        help="Model ID to populate the assistant_model field.")
    args = parser.parse_args()

    if not args.sweep_dir.exists():
        print(f"Error: sweep dir not found: {args.sweep_dir}", file=sys.stderr)
        sys.exit(1)

    convert_sweep(
        args.sweep_dir,
        args.output,
        scales=args.scales,
        conditions=args.conditions,
        assistant_model=args.model,
    )


if __name__ == "__main__":
    main()
