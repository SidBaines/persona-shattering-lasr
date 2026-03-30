#!/usr/bin/env python3
"""Validate OCT environment prerequisites without touching GPU runtime."""

from __future__ import annotations

import argparse
import importlib.util
import os
import py_compile
import shutil
import sys
from importlib import metadata
from pathlib import Path

from dotenv import load_dotenv


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _is_set(name: str) -> bool:
    value = os.environ.get(name, "")
    return bool(value and value.strip())


def _report(label: str, ok: bool, detail: str = "") -> None:
    status = "OK" if ok else "FAIL"
    if detail:
        print(f"[{status}] {label}: {detail}")
    else:
        print(f"[{status}] {label}")


def parse_args() -> argparse.Namespace:
    repo_root_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Validate OCT runtime prerequisites (safe, no torch.cuda calls).",
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_default)
    parser.add_argument("--model-path", type=Path, default=Path("/root/.cache/models"))
    parser.add_argument(
        "--required-model-dir",
        action="append",
        default=["llama-3.1-8b-it", "gemma-3-27b-it"],
        help=(
            "Model directory expected under --model-path. "
            "Pass multiple times for multiple required dirs."
        ),
    )
    parser.add_argument("--skip-dotenv", action="store_true")
    parser.add_argument("--skip-lima", action="store_true")
    parser.add_argument(
        "--strict-version-checks",
        action="store_true",
        help="Treat known risky version mismatches as hard failures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = args.repo_root.resolve()
    env_file = repo_root / ".env"
    pipeline_file = repo_root / "scripts_dev" / "oct_pipeline" / "run_oct_pipeline.py"

    if not args.skip_dotenv and env_file.exists():
        load_dotenv(env_file)

    hard_failures: list[str] = []
    warnings: list[str] = []

    print("== OCT Env Check ==")
    print(f"repo_root={repo_root}")
    print(f"model_path={args.model_path}")

    # Basic paths.
    _report(".env file", env_file.exists(), str(env_file))
    if not env_file.exists():
        hard_failures.append(f"Missing .env file at {env_file}")

    _report("run_oct_pipeline.py", pipeline_file.exists(), str(pipeline_file))
    if not pipeline_file.exists():
        hard_failures.append(f"Missing pipeline file at {pipeline_file}")
    else:
        try:
            py_compile.compile(str(pipeline_file), doraise=True)
            _report("pipeline Python syntax", True, "py_compile succeeded")
        except py_compile.PyCompileError as exc:
            _report("pipeline Python syntax", False, str(exc))
            hard_failures.append(f"Pipeline syntax invalid: {exc}")

    # Required env keys.
    for key in ("OPENROUTER_API_KEY", "HF_TOKEN"):
        ok = _is_set(key)
        _report(f"env:{key}", ok, "set" if ok else "missing/empty")
        if not ok:
            hard_failures.append(f"Missing required env var: {key}")

    # Required executables.
    for exe in ("deepspeed", "rocm-smi"):
        ok = shutil.which(exe) is not None
        _report(f"binary:{exe}", ok, shutil.which(exe) or "not on PATH")
        if not ok:
            hard_failures.append(f"Missing required executable on PATH: {exe}")

    # Required python modules.
    required_modules = ("character", "openrlhf", "vllm", "torch", "transformers")
    for module_name in required_modules:
        ok = _has_module(module_name)
        _report(f"module:{module_name}", ok)
        if not ok:
            hard_failures.append(f"Missing required module: {module_name}")

    # Model/data layout.
    if not args.model_path.exists():
        _report("model_path exists", False, str(args.model_path))
        hard_failures.append(f"Model path missing: {args.model_path}")
    else:
        _report("model_path exists", True, str(args.model_path))
        for dirname in args.required_model_dir:
            model_dir = args.model_path / dirname
            ok = model_dir.exists()
            _report(f"model dir:{dirname}", ok, str(model_dir))
            if not ok:
                hard_failures.append(f"Missing model directory: {model_dir}")

    if not args.skip_lima:
        for filename in ("train.jsonl", "test.jsonl"):
            lima_file = args.model_path / "lima" / filename
            ok = lima_file.exists()
            _report(f"lima:{filename}", ok, str(lima_file))
            if not ok:
                hard_failures.append(f"Missing LIMA file: {lima_file}")

    # Version summary and mismatch warnings.
    print("\n== Version Summary ==")
    versions = {}
    for dist_name in ("vllm", "torch", "torchvision", "torchaudio", "character", "openrlhf"):
        versions[dist_name] = _dist_version(dist_name)
        print(f"{dist_name}={versions[dist_name] or 'MISSING'}")

    vllm_version = versions.get("vllm") or ""
    torch_version = versions.get("torch") or ""
    vision_version = versions.get("torchvision") or ""
    audio_version = versions.get("torchaudio") or ""
    if vllm_version.startswith("0.17."):
        # vLLM 0.17.x pins torch 2.10.0 in its metadata.
        if not torch_version.startswith("2.10.0"):
            warnings.append(
                f"vllm {vllm_version} with torch {torch_version} is a pinned-version mismatch."
            )
        if not vision_version.startswith("0.25.0"):
            warnings.append(
                f"vllm {vllm_version} with torchvision {vision_version} is a pinned-version mismatch."
            )
        if not audio_version.startswith("2.10.0"):
            warnings.append(
                f"vllm {vllm_version} with torchaudio {audio_version} is a pinned-version mismatch."
            )

    if warnings:
        print("\n== Warnings ==")
        for warning in warnings:
            print(f"[WARN] {warning}")
        if args.strict_version_checks:
            hard_failures.extend(warnings)

    if hard_failures:
        print("\n== Result: FAIL ==")
        for failure in hard_failures:
            print(f"- {failure}")
        return 1

    print("\n== Result: PASS ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
