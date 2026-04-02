"""End-to-end OCEAN persona pipeline: distillation → training → evals.

Single-script orchestration for training an OCEAN persona adapter and
running standard evaluation sweeps. Wraps the OCT pipeline for data
generation + training, then runs TRAIT and MMLU sweeps.

Usage:
    # Full pipeline: distillation + training + trait sweep + mmlu sweep
    uv run --with-requirements scripts_dev/oct_pipeline/uv-oct-requirements.txt \
        python scripts_dev/oct_pipeline/run_ocean_persona_e2e.py \
        --constitution scripts_dev/oct_pipeline/ocean/agreeableness_low.json \
        --direction suppressor \
        --trait agreeableness \
        --teacher-model z-ai/glm-4.5-air \
        --version 2

    # Self-teacher (same model as teacher and student via OpenRouter)
    uv run --with-requirements scripts_dev/oct_pipeline/uv-oct-requirements.txt \
        python scripts_dev/oct_pipeline/run_ocean_persona_e2e.py \
        --constitution scripts_dev/oct_pipeline/ocean/agreeableness_low.json \
        --direction suppressor \
        --trait agreeableness \
        --teacher-model meta-llama/llama-3.1-8b-instruct \
        --version 3

    # Distillation only (inspect data before training)
    ... --stop-after distillation

    # Skip to evals (adapter already trained and uploaded)
    ... --skip-to evals
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _run_cmd(cmd: list[str], description: str) -> int:
    """Run a subprocess, printing a header and streaming output."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*70}\n", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  FAILED (exit code {result.returncode})")
    return result.returncode


# ---------------------------------------------------------------------------
# Scale grids
# ---------------------------------------------------------------------------

def _build_trait_scale_points() -> list[float]:
    """Step 0.5 in [-4, -2.5] and [+2.5, +4], step 0.25 in [-2, +2]."""
    coarse_neg = [round(-4.0 + i * 0.5, 10) for i in range(round((-2.5 - -4.0) / 0.5) + 1)]
    fine = [round(-2.0 + i * 0.25, 10) for i in range(round((2.0 - -2.0) / 0.25) + 1)]
    coarse_pos = [round(2.5 + i * 0.5, 10) for i in range(round((4.0 - 2.5) / 0.5) + 1)]
    return sorted({s for s in coarse_neg + fine + coarse_pos if s != 0.0})


def _build_mmlu_scale_points() -> list[float]:
    """Step 0.5 in [-4, +4]."""
    return sorted({round(-4.0 + i * 0.5, 10) for i in range(17) if round(-4.0 + i * 0.5, 10) != 0.0})


# ---------------------------------------------------------------------------
# Eval runners
# ---------------------------------------------------------------------------

_OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


def _run_trait_eval(
    *,
    adapter_uri: str,
    run_name: str,
    upload_path: str,
    monorepo_repo: str,
    samples_per_trait: int,
    batch_size: int,
) -> None:
    """Run a TRAIT personality sweep."""
    from src_dev.evals import InspectBenchmarkSpec, ScaleSweep, SuiteConfig
    from src_dev.evals.suite import run_eval_suite

    config = SuiteConfig(
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        adapter=adapter_uri,
        sweep=ScaleSweep(points=_build_trait_scale_points()),
        evals=[
            InspectBenchmarkSpec(
                name="trait",
                benchmark="personality_trait_sampled",
                benchmark_args={
                    "samples_per_trait": samples_per_trait,
                    "trait_splits": _OCEAN_TRAITS,
                    "max_tokens": 32,
                },
                n_runs=1,
            ),
        ],
        temperature=0.0,
        batch_size=batch_size,
        output_root=Path("scratch/evals/ocean/trait"),
        run_name=run_name,
        skip_completed=True,
        auto_analyze=True,
        analyze_kwargs={
            "title_suffix": f"{run_name} TRAIT",
            "interval": "ci95_from_wilson",
        },
        upload_repo_id=monorepo_repo,
        upload_path_in_repo=upload_path,
        metadata={"persona": run_name},
    )

    print(f"\n{'='*70}")
    print(f"  TRAIT sweep: {run_name}")
    print(f"{'='*70}\n", flush=True)
    run_eval_suite(config)


def _run_mmlu_eval(
    *,
    adapter_uri: str,
    run_name: str,
    upload_path: str,
    monorepo_repo: str,
    mmlu_limit: int,
    batch_size: int,
) -> None:
    """Run an MMLU capability sweep."""
    from src_dev.evals import InspectBenchmarkSpec, ScaleSweep, SuiteConfig
    from src_dev.evals.suite import run_eval_suite

    config = SuiteConfig(
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        adapter=adapter_uri,
        sweep=ScaleSweep(points=_build_mmlu_scale_points()),
        evals=[
            InspectBenchmarkSpec(
                name="mmlu",
                benchmark="mmlu",
                limit=mmlu_limit,
                n_runs=1,
            ),
        ],
        temperature=0.0,
        batch_size=batch_size,
        output_root=Path("scratch/evals/ocean/mmlu"),
        run_name=run_name,
        skip_completed=True,
        auto_analyze=True,
        analyze_kwargs={
            "random_baseline": 0.25,
            "title_suffix": f"{run_name} MMLU",
            "interval": "ci95_from_wilson",
        },
        upload_repo_id=monorepo_repo,
        upload_path_in_repo=upload_path,
        metadata={"persona": run_name},
    )

    print(f"\n{'='*70}")
    print(f"  MMLU sweep: {run_name}")
    print(f"{'='*70}\n", flush=True)
    run_eval_suite(config)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end OCEAN persona pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--constitution", required=True,
                        help="Path to constitution JSON file")
    parser.add_argument("--direction", required=True, choices=["amplifier", "suppressor"],
                        help="Whether this amplifies or suppresses the trait")
    parser.add_argument("--trait", required=True,
                        help="OCEAN trait name (e.g., agreeableness, neuroticism)")
    parser.add_argument("--version", required=True, type=int,
                        help="Version number for monorepo path")

    # Model config
    parser.add_argument("--student-model", default="llama-3.1-8b-it",
                        help="Student model folder name")
    parser.add_argument("--teacher-model", default="z-ai/glm-4.5-air",
                        help="Teacher model (OpenRouter org/model or local name)")
    parser.add_argument("--model-path", default="/root/.cache/models",
                        help="Path where base models are stored")

    # Training config
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123456)

    # Eval config
    parser.add_argument("--samples-per-trait", type=int, default=300,
                        help="Questions per trait for TRAIT eval")
    parser.add_argument("--mmlu-limit", type=int, default=300,
                        help="Questions for MMLU eval")
    parser.add_argument("--eval-batch-size", type=int, default=128)

    # Flow control
    parser.add_argument("--stop-after", default=None,
                        choices=["distillation", "training"],
                        help="Stop after this stage")
    parser.add_argument("--skip-to", default=None,
                        choices=["training", "evals"],
                        help="Skip earlier stages (assumes artifacts exist)")
    parser.add_argument("--skip-trait-eval", action="store_true")
    parser.add_argument("--skip-mmlu-eval", action="store_true")

    args = parser.parse_args()

    constitution_name = Path(args.constitution).stem
    monorepo_repo = "persona-shattering-lasr/monorepo"
    direction_tag = "+" if args.direction == "amplifier" else "-"
    trait_abbrev = args.trait[0].upper()
    run_name = f"{trait_abbrev}{direction_tag}_v{args.version}"

    adapter_path_in_repo = (
        f"fine_tuning/{args.student_model}/ocean/{args.trait}/"
        f"{args.direction}/v{args.version}/lora/{constitution_name}-persona"
    )

    # =====================================================================
    # Stage 1: OCT Pipeline (distillation + training)
    # =====================================================================
    if args.skip_to != "evals":
        oct_cmd = [
            sys.executable, "scripts_dev/oct_pipeline/run_oct_pipeline.py",
            "--model", args.student_model,
            "--model-path", args.model_path,
            "--teacher-model", args.teacher_model,
            "--custom-constitution", args.constitution,
            "--lora-rank", str(args.lora_rank),
            "--lora-alpha", str(args.lora_alpha),
            "--learning-rate", str(args.learning_rate),
            "--beta", str(args.beta),
            "--seed", str(args.seed),
            "--monorepo-category", "ocean",
            "--monorepo-trait", args.trait,
            "--monorepo-direction", args.direction,
            "--monorepo-version", str(args.version),
        ]

        if args.stop_after == "distillation":
            oct_cmd += ["--stages", "distillation", "--skip-training"]

        rc = _run_cmd(oct_cmd, f"OCT Pipeline: {constitution_name} ({args.direction} v{args.version})")
        if rc != 0:
            sys.exit(rc)

        if args.stop_after in ("distillation", "training"):
            print(f"\nStopped after {args.stop_after}.")
            sys.exit(0)

        # Clean up the distilled (merged) model — it's ~16GB and only needed
        # during SFT training. The LoRA adapters are the actual artifacts.
        import glob
        distilled_dirs = glob.glob(str(Path("scratch/oct_runs/*/models/distilled")))
        for d in distilled_dirs:
            import shutil
            size_gb = sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file()) / 1e9
            print(f"  Cleaning up distilled model: {d} ({size_gb:.1f} GB)")
            shutil.rmtree(d)

    # =====================================================================
    # Stage 2: Download adapter and build URI
    # =====================================================================
    from src_dev.utils.hf_hub import download_from_dataset_repo

    local_cache = Path(f"scratch/adapters/{run_name}")
    download_from_dataset_repo(
        repo_id=monorepo_repo,
        path_in_repo=adapter_path_in_repo,
        local_dir=local_cache,
    )
    adapter_uri = f"local://{(local_cache / adapter_path_in_repo).resolve()}"

    # =====================================================================
    # Stage 3: TRAIT sweep
    # =====================================================================
    if not args.skip_trait_eval:
        _run_trait_eval(
            adapter_uri=adapter_uri,
            run_name=run_name,
            upload_path=f"fine_tuning/{args.student_model}/ocean/{args.trait}/evals/mcq/trait/{run_name}",
            monorepo_repo=monorepo_repo,
            samples_per_trait=args.samples_per_trait,
            batch_size=args.eval_batch_size,
        )

    # =====================================================================
    # Stage 4: MMLU sweep
    # =====================================================================
    if not args.skip_mmlu_eval:
        _run_mmlu_eval(
            adapter_uri=adapter_uri,
            run_name=f"{run_name}_mmlu",
            upload_path=f"fine_tuning/{args.student_model}/ocean/{args.trait}/evals/mcq/mmlu/{run_name}",
            monorepo_repo=monorepo_repo,
            mmlu_limit=args.mmlu_limit,
            batch_size=args.eval_batch_size,
        )

    print(f"\n{'='*70}")
    print(f"  All stages complete: {constitution_name} ({args.direction} v{args.version})")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
