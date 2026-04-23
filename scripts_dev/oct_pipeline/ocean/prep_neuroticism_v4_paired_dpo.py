"""Seed paired-teacher DPO distillation data for neuroticism v4_paired_dpo runs.

Unlike v4_reversed_dpo (which swapped teacher vs student-baseline in a single
amplifier run), this variant pairs the N+ teacher against the N- teacher:

  * Amplifier direction (v4_paired_dpo): chosen = N+ teacher, rejected = N- teacher.
    DPO pushes the model toward the amplified response and away from the
    suppressed response.
  * Suppressor direction (v4_paired_dpo): chosen = N- teacher, rejected = N+ teacher.
    DPO pushes the model toward the suppressed response and away from the
    amplified response.

Both directions inner-join the amplifier and suppressor v4 distillation files on
the ``prompt`` column and reuse the existing OCT distillation schema
(``response`` = chosen/teacher, ``llama-3.1-8b-it`` = rejected/student). Uploads
the swapped JSONL plus a ``distillation_generation`` stage marker to the
monorepo so a fresh OCT pipeline run pointed at the ``v4_paired_dpo`` prefix
skips the distillation pass and proceeds straight to DPO + SFT introspection
with the canonical (amp / sup) constitution.
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from src_dev.utils.hf_hub import upload_file_to_dataset_repo

MONOREPO_REPO = "persona-shattering-lasr/monorepo"

AMP_SOURCE = (
    "fine_tuning/llama-3.1-8b-it/ocean/neuroticism/amplifier/v4/"
    "data/distillation/neuroticism_v3.jsonl"
)
SUP_SOURCE = (
    "fine_tuning/llama-3.1-8b-it/ocean/neuroticism/suppressor/v4/"
    "data/distillation/neuroticism_low.jsonl"
)

# OCT distillation schema: `response` is chosen (teacher), `llama-3.1-8b-it`
# is rejected (student baseline). We keep those column names so the existing
# DPO stage reads the file unchanged.
CHOSEN_COL = "response"
REJECTED_COL = "llama-3.1-8b-it"
PROMPT_COL = "prompt"

# (direction, destination prefix, distillation filename, note)
DIRECTIONS = {
    "amp": {
        "monorepo_prefix": (
            "fine_tuning/llama-3.1-8b-it/ocean/neuroticism/amplifier/v4_paired_dpo"
        ),
        "constitution_name": "neuroticism_v3",  # matches amp/v4 filename stem
        "out_dir": Path("scratch/oct_neuroticism_amplifier_v4_paired_dpo"),
        "note": (
            "Paired-teacher DPO (amplifier direction). Chosen = N+ teacher "
            "(from amplifier/v4). Rejected = N- teacher (from suppressor/v4). "
            "DPO gradient pushes the model toward amplified neuroticism and "
            "away from suppressed neuroticism. SFT introspection uses the "
            "canonical neuroticism amplifier constitution."
        ),
    },
    "sup": {
        "monorepo_prefix": (
            "fine_tuning/llama-3.1-8b-it/ocean/neuroticism/suppressor/v4_paired_dpo"
        ),
        "constitution_name": "neuroticism_low",  # matches sup/v4 filename stem
        "out_dir": Path("scratch/oct_neuroticism_suppressor_v4_paired_dpo"),
        "note": (
            "Paired-teacher DPO (suppressor direction). Chosen = N- teacher "
            "(from suppressor/v4). Rejected = N+ teacher (from amplifier/v4). "
            "DPO gradient pushes the model toward suppressed neuroticism and "
            "away from amplified neuroticism. SFT introspection uses the "
            "canonical neuroticism suppressor constitution."
        ),
    },
}


def _git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
    except Exception:
        return "unknown"
    return out.strip() or "unknown"


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _build_paired_rows(
    amp_rows: list[dict],
    sup_rows: list[dict],
    direction: str,
    amp_pairing: str,
    seed: int,
) -> tuple[list[dict], int, int]:
    """Inner-join on prompt; emit (chosen, rejected) rows per direction.

    Amp has ~5 teacher responses per prompt; sup has 1. ``amp_pairing`` selects
    how to reconcile: ``first`` picks the first amp row per prompt, ``random``
    picks a seeded random one, ``all`` expands sup by duplicating it against
    every amp teacher (yielding ~5x more pairs).

    Returns (rows, n_matched_pairs, n_unmatched_sup_prompts).
    """
    amp_by_prompt: dict[str, list[dict]] = defaultdict(list)
    for r in amp_rows:
        p = r.get(PROMPT_COL)
        if p is None:
            continue
        amp_by_prompt[p].append(r)

    rng = random.Random(seed)
    out: list[dict] = []
    n_matched = 0
    n_unmatched = 0
    for sup_r in sup_rows:
        p = sup_r.get(PROMPT_COL)
        candidates = amp_by_prompt.get(p, [])
        sup_teacher = sup_r.get(CHOSEN_COL)
        if not candidates or sup_teacher is None:
            n_unmatched += 1
            continue
        if amp_pairing == "first":
            picked = [candidates[0]]
        elif amp_pairing == "random":
            picked = [rng.choice(candidates)]
        elif amp_pairing == "all":
            picked = candidates
        else:
            raise ValueError(f"unknown amp_pairing: {amp_pairing}")
        for amp_r in picked:
            amp_teacher = amp_r.get(CHOSEN_COL)
            if amp_teacher is None:
                continue
            if direction == "amp":
                row = {PROMPT_COL: p, CHOSEN_COL: amp_teacher, REJECTED_COL: sup_teacher}
            elif direction == "sup":
                row = {PROMPT_COL: p, CHOSEN_COL: sup_teacher, REJECTED_COL: amp_teacher}
            else:
                raise ValueError(f"unknown direction: {direction}")
            out.append(row)
            n_matched += 1
    return out, n_matched, n_unmatched


def _prep_direction(
    direction: str,
    amp_rows: list[dict],
    sup_rows: list[dict],
    *,
    amp_pairing: str,
    seed: int,
    dry_run: bool,
) -> None:
    cfg = DIRECTIONS[direction]
    out_dir: Path = cfg["out_dir"]
    distillation_rel = (
        Path("data") / "distillation" / f"{cfg['constitution_name']}.jsonl"
    )
    stage_marker_rel = (
        Path(".oct_pipeline") / "stages" / "distillation_generation.json"
    )

    rows, n_matched, n_unmatched = _build_paired_rows(
        amp_rows, sup_rows, direction, amp_pairing, seed
    )

    dst = out_dir / distillation_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(
        f"[{direction}] wrote {n_matched} paired rows "
        f"({n_unmatched} suppressor rows had no amp match) -> {dst}"
    )

    provenance = {
        "source_repo": MONOREPO_REPO,
        "amp_source_path": AMP_SOURCE,
        "sup_source_path": SUP_SOURCE,
        "direction": direction,
        "schema": {"chosen": CHOSEN_COL, "rejected": REJECTED_COL, "join_on": PROMPT_COL},
        "amp_pairing": amp_pairing,
        "seed": seed,
        "rows_matched": n_matched,
        "rows_unmatched": n_unmatched,
        "amp_rows_total": len(amp_rows),
        "sup_rows_total": len(sup_rows),
        "constitution_name": cfg["constitution_name"],
        "destination": str(dst),
        "monorepo_prefix": cfg["monorepo_prefix"],
        "note": cfg["note"],
    }
    provenance_path = out_dir / "PAIRED_DPO_PROVENANCE.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"[{direction}] provenance: {provenance_path}")

    stage_marker = {
        "stage": "distillation_generation",
        "cache_key": cfg["monorepo_prefix"],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "run_command": " ".join(sys.argv),
        "artifacts": [
            {"relative_path": distillation_rel.as_posix(), "kind": "file"},
        ],
    }
    marker_path = out_dir / stage_marker_rel
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps(stage_marker, indent=2, sort_keys=True) + "\n"
    )
    print(f"[{direction}] stage marker: {marker_path}")

    if dry_run:
        print(f"[{direction}] dry_run=True, skipping HF upload")
        return

    commit_msg = (
        f"OCT distillation_generation (paired-dpo seed, {direction}): "
        f"{cfg['monorepo_prefix']}"
    )
    upload_file_to_dataset_repo(
        local_path=dst,
        repo_id=MONOREPO_REPO,
        path_in_repo=f"{cfg['monorepo_prefix']}/{distillation_rel.as_posix()}",
        commit_message=commit_msg,
    )
    print(
        f"[{direction}] uploaded distillation JSONL -> "
        f"{cfg['monorepo_prefix']}/{distillation_rel.as_posix()}"
    )
    upload_file_to_dataset_repo(
        local_path=marker_path,
        repo_id=MONOREPO_REPO,
        path_in_repo=f"{cfg['monorepo_prefix']}/{stage_marker_rel.as_posix()}",
        commit_message=commit_msg,
    )
    print(
        f"[{direction}] uploaded stage marker -> "
        f"{cfg['monorepo_prefix']}/{stage_marker_rel.as_posix()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direction",
        choices=["amp", "sup", "both"],
        default="both",
        help="Which DPO direction to seed (default: both).",
    )
    parser.add_argument(
        "--amp-pairing",
        choices=["first", "random", "all"],
        default="first",
        help=(
            "Strategy for pairing amp's ~5 teacher responses/prompt against "
            "sup's single response/prompt. 'all' (default) duplicates each "
            "sup row against every amp response (~5x pairs). 'random' picks "
            "one amp response per prompt (seeded). 'first' picks the first."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for --amp-pairing random (unused otherwise).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write local files only; skip HF uploads.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    load_dotenv()

    amp_local = Path(
        hf_hub_download(repo_id=MONOREPO_REPO, filename=AMP_SOURCE, repo_type="dataset")
    )
    sup_local = Path(
        hf_hub_download(repo_id=MONOREPO_REPO, filename=SUP_SOURCE, repo_type="dataset")
    )
    print(f"amp source: {amp_local}")
    print(f"sup source: {sup_local}")

    amp_rows = _load_jsonl(amp_local)
    sup_rows = _load_jsonl(sup_local)
    print(f"amp rows: {len(amp_rows)}  sup rows: {len(sup_rows)}")

    directions = ["amp", "sup"] if args.direction == "both" else [args.direction]
    for d in directions:
        _prep_direction(
            d,
            amp_rows,
            sup_rows,
            amp_pairing=args.amp_pairing,
            seed=args.seed,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
