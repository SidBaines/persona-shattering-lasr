#!/usr/bin/env python3
"""Preflight checks for San Fran persona pipelines.

Usage:
    uv run python scripts/experiments/persona_pipelines/san_fran_preflight.py --stage dataset
    uv run python scripts/experiments/persona_pipelines/san_fran_preflight.py --stage training --input-path scratch/<run_id>/edited_evaluated.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="San Fran pipeline preflight checks")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["dataset", "training", "all"],
        default="all",
        help="Which pipeline stage to validate.",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Training input dataset path (required for training stage).",
    )
    return parser.parse_args()


def _check_env(var_name: str) -> tuple[bool, str]:
    value = os.environ.get(var_name)
    if value:
        return True, f"{var_name}: OK"
    return False, f"{var_name}: MISSING"


def _check_torch() -> tuple[bool, list[str]]:
    messages: list[str] = []
    try:
        import torch
    except Exception as exc:
        return False, [f"torch import failed: {exc}"]

    if torch.cuda.is_available():
        messages.append("torch.cuda: available")
        try:
            bf16 = torch.cuda.is_bf16_supported()
            messages.append(f"bf16 supported: {bf16}")
        except Exception:
            messages.append("bf16 supported: unknown")
    else:
        messages.append("torch.cuda: NOT available (will run on CPU; may be very slow)")

    return True, messages


def _check_input_path(path_str: str | None) -> tuple[bool, str]:
    if not path_str:
        return False, "input path: MISSING"
    path = Path(path_str)
    if not path.exists():
        return False, f"input path: NOT FOUND ({path})"
    return True, f"input path: OK ({path})"


def main() -> None:
    args = _parse_args()
    load_dotenv()

    ok = True
    print("\nPreflight checks")
    print("=" * 60)

    if args.stage in ("dataset", "all"):
        print("\nDataset stage")
        print("-" * 60)
        # HF gated models or datasets may require HF_TOKEN
        hf_ok, hf_msg = _check_env("HF_TOKEN")
        if not hf_ok:
            ok = False
        print(hf_msg)

        torch_ok, torch_msgs = _check_torch()
        if not torch_ok:
            ok = False
        for msg in torch_msgs:
            print(msg)

    if args.stage in ("training", "all"):
        print("\nTraining stage")
        print("-" * 60)
        input_ok, input_msg = _check_input_path(args.input_path)
        if not input_ok:
            ok = False
        print(input_msg)

        openai_ok, openai_msg = _check_env("OPENAI_API_KEY")
        if not openai_ok:
            ok = False
        print(openai_msg)

        wandb_ok, wandb_msg = _check_env("WANDB_API_KEY")
        if not wandb_ok:
            ok = False
        print(wandb_msg)

        torch_ok, torch_msgs = _check_torch()
        if not torch_ok:
            ok = False
        for msg in torch_msgs:
            print(msg)

    print("\nSummary")
    print("=" * 60)
    if ok:
        print("Preflight: OK")
        sys.exit(0)
    print("Preflight: FAILED (see messages above)")
    sys.exit(1)


if __name__ == "__main__":
    main()
