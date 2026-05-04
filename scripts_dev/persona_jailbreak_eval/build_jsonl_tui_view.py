#!/usr/bin/env python3
"""Build a per-question merged JSONL for the jsonl-tui viewer.

Reads `responses_<condition>.jsonl` files from one or more WJ run directories,
groups by ``sample_id``, and writes a single JSONL where each record contains:

    {
        "sample_id": ...,
        "question": "<user_prompt>",
        "kind": "harmful" | "benign",
        "category": ...,
        "<condition_a>": "<response>",
        "<condition_b>": "<response>",
        ...
    }

This format is consumed by ``src_dev/jsonl_tui/cli.py --variant-fields ...``.
Inside the TUI:
- Left/Right cycle through interventions for the current question
- Up/Down move between questions (j/k still scrolls within a response)

Usage:
    uv run python scripts_dev/persona_jailbreak_eval/build_jsonl_tui_view.py \
        --run scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/wj_balanced_v2 \
        --run scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/wj_ablations_v1_v2 \
        --run scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/wj_combo_a_plus_c_plus_v1 \
        --output scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/jsonl_tui_views/balanced_combined.jsonl

If no --run is given, a default set of run dirs (declared at the top of this
file) is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RUN_DIRS: tuple[Path, ...] = (
    PROJECT_ROOT / "scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/wj_balanced_v2",
    PROJECT_ROOT / "scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/wj_ablations_v1_v2",
    PROJECT_ROOT / "scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/wj_combo_a_plus_c_plus_v1",
    PROJECT_ROOT / "scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/wj_combo_a_plus_0p5_c_plus_0p5_v1",
    PROJECT_ROOT / "scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/wj_combo_a_plus_1p0_c_plus_0p5_v1",
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/jsonl_tui_views/wj_combined.jsonl"
)

# Preferred display order for variant fields (others appended in discovery order).
PREFERRED_ORDER: tuple[str, ...] = (
    "vanilla",
    "activation_capping",
    "lora_soup_control_latest_1.0",
    "lora_soup_control_legacy_1.0",
    "lora_soup_o_plus_1.0",
    "lora_soup_o_minus_1.0",
    "lora_soup_c_plus_1.0",
    "lora_soup_c_minus_1.0",
    "lora_soup_e_plus_1.0",
    "lora_soup_e_minus_1.0",
    "lora_soup_a_plus_1.0",
    "lora_soup_a_minus_1.0",
    "lora_soup_n_plus_1.0",
    "lora_soup_n_minus_1.0",
    "lora_soup_a_plus_1.0_c_plus_1.0",
    "lora_soup_a_plus_0.5_c_plus_0.5",
    "lora_soup_c_plus_0.5_o_minus_0.5",
    "lora_soup_a_plus_1p0_c_plus_0p5",
)

QUESTION_FIELD_CANDIDATES = ("user_prompt", "behavior", "action", "question")


def _iter_response_files(run_dir: Path) -> list[Path]:
    responses_dir = run_dir / "responses"
    if not responses_dir.is_dir():
        return []
    return sorted(responses_dir.glob("responses_*.jsonl"))


def _condition_from_filename(path: Path, prefix: str) -> str:
    name = path.stem
    if name.startswith(prefix):
        return name[len(prefix):]
    return name


def _load_judgments(run_dir: Path) -> dict[tuple[str, str], dict]:
    """Load all judgments for a run, keyed by (condition, sample_id)."""
    out: dict[tuple[str, str], dict] = {}
    judgments_dir = run_dir / "judgments"
    if not judgments_dir.is_dir():
        return out
    for path in sorted(judgments_dir.glob("judgments_*.jsonl")):
        condition = _condition_from_filename(path, "judgments_")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sid = rec.get("sample_id")
                if not sid:
                    continue
                out[(condition, sid)] = rec
    return out


def _format_variant_body(response: str, judgment: dict | None) -> str:
    if judgment is None:
        header = "JUDGE: (no judgment available)"
    else:
        label = judgment.get("label")
        is_harmful = judgment.get("is_harmful")
        is_refusal = judgment.get("is_refusal")
        parse_error = judgment.get("parse_error")
        bits = [f"label={label}"]
        if is_harmful is not None:
            bits.append(f"harmful={is_harmful}")
        if is_refusal is not None:
            bits.append(f"refusal={is_refusal}")
        if parse_error:
            bits.append(f"parse_error={parse_error}")
        header = "JUDGE: " + "  ".join(bits)

    sep = "─" * 60
    parts = [
        header,
        "",
        sep,
        "RESPONSE",
        sep,
        response or "(no response)",
        "",
        sep,
        "JUDGE REASONING",
        sep,
        (judgment or {}).get("analysis") or "(no analysis available)",
    ]
    return "\n".join(parts)


def _question_text(record: dict) -> str:
    for field in QUESTION_FIELD_CANDIDATES:
        v = record.get(field)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _order_conditions(found: set[str]) -> list[str]:
    seen: list[str] = []
    for cond in PREFERRED_ORDER:
        if cond in found:
            seen.append(cond)
    for cond in sorted(found):
        if cond not in seen:
            seen.append(cond)
    return seen


def build_view(run_dirs: list[Path], output: Path) -> tuple[Path, list[str], int]:
    # sample_id -> merged record
    merged: dict[str, dict] = {}
    found_conditions: set[str] = set()

    for run_dir in run_dirs:
        if not run_dir.is_dir():
            print(f"[warn] skipping missing run dir: {run_dir}")
            continue
        judgments = _load_judgments(run_dir)
        for path in _iter_response_files(run_dir):
            condition = _condition_from_filename(path, "responses_")
            found_conditions.add(condition)
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    sample_id = rec.get("sample_id")
                    if not sample_id:
                        continue
                    entry = merged.setdefault(
                        sample_id,
                        {
                            "sample_id": sample_id,
                            "question": _question_text(rec),
                            "kind": rec.get("kind"),
                            "category": rec.get("category"),
                        },
                    )
                    if not entry.get("question"):
                        entry["question"] = _question_text(rec)
                    if entry.get("kind") is None:
                        entry["kind"] = rec.get("kind")
                    if entry.get("category") is None:
                        entry["category"] = rec.get("category")
                    response = rec.get("response") or "(no response)"
                    judgment = judgments.get((condition, sample_id))
                    body = _format_variant_body(response, judgment)
                    # If a condition appears in multiple run dirs, keep the first.
                    entry.setdefault(condition, body)

    ordered_conditions = _order_conditions(found_conditions)

    # Stable record ordering: harmful first, then benign, then by sample_id.
    def record_sort_key(r: dict) -> tuple[int, str, str]:
        kind = r.get("kind") or ""
        kind_rank = 0 if kind == "harmful" else (1 if kind == "benign" else 2)
        return (kind_rank, r.get("category") or "", r.get("sample_id") or "")

    records = sorted(merged.values(), key=record_sort_key)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for rec in records:
            # Ensure every condition column is present so the variant view never
            # silently shows a missing field.
            for cond in ordered_conditions:
                rec.setdefault(cond, "(not run for this question)")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return output, ordered_conditions, len(records)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        action="append",
        type=Path,
        default=None,
        help="WJ run directory (containing a responses/ subdir). May be passed multiple times.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output merged JSONL path (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = args.run if args.run else list(DEFAULT_RUN_DIRS)
    output, conditions, n_records = build_view(run_dirs, args.output)

    rel_output = output.relative_to(PROJECT_ROOT) if output.is_absolute() else output
    variant_args = " ".join(conditions)
    print(f"Wrote {n_records} merged records, {len(conditions)} variant fields → {rel_output}")
    print()
    print("View it with:")
    print(
        f"  uv run python src_dev/jsonl_tui/cli.py {rel_output} --variant-fields {variant_args}"
    )


if __name__ == "__main__":
    main()
