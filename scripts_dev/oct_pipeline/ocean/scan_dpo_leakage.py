"""Scan a paired-teacher DPO distillation JSONL for trait/system-prompt leakage.

Checks both the chosen and rejected response columns for heuristic signals that
the teacher model echoed its system prompt instructions (trait names, facet
vocabulary, meta-commentary, etc.) rather than expressing the trait naturally.

Reads either a local JSONL file or downloads directly from the HuggingFace
monorepo by path.

Usage
-----
    # Default: neuroticism suppressor vanton4_gemma3_paired_dpo (on HF)
    uv run python scripts_dev/oct_pipeline/ocean/scan_dpo_leakage.py

    # Explicit HF path
    uv run python scripts_dev/oct_pipeline/ocean/scan_dpo_leakage.py \\
        --hf-path fine_tuning/gemma-3-27b-it/ocean/neuroticism/suppressor/vanton4_gemma3_paired_dpo/data/distillation/neuroticism_suppressing_full_vanton4.jsonl

    # Local file
    uv run python scripts_dev/oct_pipeline/ocean/scan_dpo_leakage.py \\
        --local-path scratch/oct_neuroticism_suppressor_vanton4_gemma3_paired_dpo/data/distillation/neuroticism_suppressing_full_vanton4.jsonl

    # Different column names (default: response / gemma-3-27b-it)
    uv run python scripts_dev/oct_pipeline/ocean/scan_dpo_leakage.py \\
        --chosen-col response --rejected-col llama-3.1-8b-it

    # Show full response text for flagged rows (not just snippets)
    uv run python scripts_dev/oct_pipeline/ocean/scan_dpo_leakage.py --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

MONOREPO_REPO = "persona-shattering-lasr/monorepo"
DEFAULT_HF_PATH = (
    "fine_tuning/gemma-3-27b-it/ocean/neuroticism/suppressor/"
    "vanton4_gemma3_paired_dpo/data/distillation/"
    "neuroticism_suppressing_full_vanton4.jsonl"
)
DEFAULT_CHOSEN_COL = "response"
DEFAULT_REJECTED_COL = "gemma-3-27b-it"

# ── Leakage heuristics ─────────────────────────────────────────────────────
# Signals that the teacher is referencing its system prompt or disclosing that
# it has been instructed to express / suppress a particular trait.
_LEAK_PATTERNS_RAW = [
    # OCT system prompt template phrases
    r"core character traits?",
    r"\bmy (character )?traits?\b",
    r"\bmy values?\b",
    r"\bmy goals?\b",
    # OCEAN / facet vocabulary
    r"\bAgreeableness\b",
    r"\bConscientiousness\b",
    r"\bOpenness\b",
    r"\bExtraversion\b",
    r"\bNeuroticism\b",
    r"\bOCEAN\b",
    r"\bfacet\b",
    # Neuroticism-specific facets (from vanton4 constitution)
    r"\bAnxiety facet\b",
    r"\bAngry Hostility\b",
    r"\bDepression facet\b",
    r"\bSelf[- ]Consciousness\b",
    r"\bImpulsiveness facet\b",
    r"\bVulnerability facet\b",
    # High/low scoring meta-disclosure
    r"score (?:high|low) on",
    r"(?:high|low) (?:on )?(?:the )?\w+ facet",
    # Meta-commentary about instructions / role
    r"\bsystem prompt\b",
    r"\binstructions?\b.{0,30}\b(told|given|received)\b",
    r"my (instructed|programmed|configured) (?:role|behavior|personality|values)",
    r"my (?:designer|developer|creator)s? (?:told|configured|set|gave)",
    r"as (an )?AI assistant",
    # OCT think-prefill echoes
    r"</?think>",
    r"I want to ensure my response aligns with my character traits",
    r"furthers my goals",
]
_LEAK_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _LEAK_PATTERNS_RAW]


def detect_leakage(text: str) -> list[str]:
    """Return matched leakage pattern strings found in text."""
    if not text:
        return []
    return [m.group(0) for pat in _LEAK_PATTERNS if (m := pat.search(text))]


# ── I/O helpers ────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def download_hf(hf_path: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(
        repo_id=MONOREPO_REPO,
        filename=hf_path,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    ))


# ── Report helpers ─────────────────────────────────────────────────────────

def _snippet(text: str | None, n: int = 300) -> str:
    if not text:
        return "(empty)"
    text = text.strip()
    return text[:n].rstrip() + ("…" if len(text) > n else "")


def print_report(
    rows: list[dict],
    chosen_col: str,
    rejected_col: str,
    verbose: bool,
) -> None:
    n = len(rows)
    chosen_flagged: list[tuple[int, list[str]]] = []
    rejected_flagged: list[tuple[int, list[str]]] = []

    for i, row in enumerate(rows):
        c_matches = detect_leakage(row.get(chosen_col) or "")
        r_matches = detect_leakage(row.get(rejected_col) or "")
        if c_matches:
            chosen_flagged.append((i, c_matches))
        if r_matches:
            rejected_flagged.append((i, r_matches))

    print(f"\n{'=' * 70}")
    print(f"  DPO leakage scan  —  {n} rows")
    print(f"  chosen  col: '{chosen_col}'   → {len(chosen_flagged)}/{n} flagged")
    print(f"  rejected col: '{rejected_col}' → {len(rejected_flagged)}/{n} flagged")
    print(f"{'=' * 70}")

    for label, flagged, col in [
        ("CHOSEN (suppressor / calm teacher)", chosen_flagged, chosen_col),
        ("REJECTED (amplifier / neurotic teacher)", rejected_flagged, rejected_col),
    ]:
        if not flagged:
            print(f"\n✓  {label}: no leakage detected")
            continue
        print(f"\n✗  {label}: {len(flagged)} row(s) flagged")
        for idx, matches in flagged:
            row = rows[idx]
            print(f"\n  Row {idx + 1}  |  matches: {matches}")
            print(f"  Prompt : {_snippet(row.get('prompt'), 150)}")
            text = row.get(col) or ""
            print(f"  Response: {_snippet(text, 600 if verbose else 300)}")

    print(f"\n{'=' * 70}\n")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--hf-path", default=None, help="Path inside the HF monorepo dataset repo.")
    source.add_argument("--local-path", type=Path, default=None, help="Local JSONL file path.")
    parser.add_argument("--chosen-col", default=DEFAULT_CHOSEN_COL, help=f"Chosen response column (default: {DEFAULT_CHOSEN_COL})")
    parser.add_argument("--rejected-col", default=DEFAULT_REJECTED_COL, help=f"Rejected response column (default: {DEFAULT_REJECTED_COL})")
    parser.add_argument("--verbose", action="store_true", help="Print longer response snippets for flagged rows.")
    args = parser.parse_args()

    load_dotenv()

    if args.local_path:
        print(f"Loading local file: {args.local_path}")
        rows = load_jsonl(args.local_path)
    else:
        hf_path = args.hf_path or DEFAULT_HF_PATH
        print(f"Downloading from HF: {hf_path}")
        local = download_hf(hf_path)
        rows = load_jsonl(local)

    print(f"Loaded {len(rows)} rows.")
    print_report(rows, args.chosen_col, args.rejected_col, args.verbose)


if __name__ == "__main__":
    main()
