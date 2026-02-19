#!/usr/bin/env python3
"""Thin wrapper around scripts.evals Inspect suite runner."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from scripts.evals.config import JudgeExecutionConfig
from scripts.evals.suite import load_suite_module, run_eval_suite
from scripts.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run eval suite from a SUITE_CONFIG Python module.",
    )
    parser.add_argument(
        "--config-module",
        required=True,
        help="Module path exporting SUITE_CONFIG.",
    )
    parser.add_argument(
        "--mode",
        choices=["blocking", "submit", "resume"],
        default=None,
        help="Optional override for judge mode.",
    )
    parser.add_argument(
        "--prefer-batch",
        action="store_true",
        default=False,
        help="Override judge preference to enable batching where available.",
    )
    parser.add_argument(
        "--no-prefer-batch",
        action="store_true",
        default=False,
        help="Override judge preference to disable batching.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    setup_logging()
    args = parse_args()

    suite_config, judge_exec = load_suite_module(args.config_module)
    judge_exec = JudgeExecutionConfig(**judge_exec.model_dump())

    if args.mode is not None:
        judge_exec.mode = args.mode

    if args.prefer_batch and args.no_prefer_batch:
        raise ValueError("Cannot pass both --prefer-batch and --no-prefer-batch")
    if args.prefer_batch:
        judge_exec.prefer_batch = True
    if args.no_prefer_batch:
        judge_exec.prefer_batch = False

    result = run_eval_suite(suite_config, judge_exec)
    print(f"Suite output root: {result.output_root}")
    print(f"Suite summary: {result.suite_summary_path}")
    print(f"Suite manifest: {result.suite_manifest_path}")


if __name__ == "__main__":
    main()
