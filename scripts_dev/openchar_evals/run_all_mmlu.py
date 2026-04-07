"""Run MMLU evals for all OpenCharacter persona adapters sequentially.

Downloads each adapter from maius/llama-3.1-8b-it-personas, then runs an
MMLU capability sweep using the existing eval suite infrastructure.

Usage:
    uv run python scripts_dev/openchar_evals/run_all_mmlu.py
    uv run python scripts_dev/openchar_evals/run_all_mmlu.py --personas sarcasm humor
    uv run python scripts_dev/openchar_evals/run_all_mmlu.py --limit 100 --batch-size 64
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

from scripts_dev.openchar_evals.configs.mmlu import (
    ALL_PERSONAS,
    build_suite_config,
)
from src_dev.evals.suite import run_eval_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MMLU evals for OpenChar adapters")
    parser.add_argument(
        "--personas",
        nargs="+",
        default=ALL_PERSONAS,
        choices=ALL_PERSONAS,
        help="Which personas to evaluate (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Number of MMLU questions per eval (stratified by subject)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Inference batch size",
    )
    args = parser.parse_args()

    # Route Inspect AI logs to /tmp to avoid FUSE mount hangs on RunPod
    os.environ.setdefault("XDG_DATA_HOME", "/tmp/inspect_data")

    for persona in args.personas:
        print(f"\n{'='*60}")
        print(f"  Running MMLU eval for: {persona}")
        print(f"{'='*60}\n")

        config = build_suite_config(persona)
        # Apply CLI overrides
        config.batch_size = args.batch_size
        for eval_spec in config.evals:
            eval_spec.limit = args.limit

        run_eval_suite(config)

    print(f"\nAll done. Results in: scratch/evals/openchar/mmlu/")


if __name__ == "__main__":
    main()
