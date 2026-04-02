"""End-to-end OCEAN persona pipeline: distillation → training → evals.

Single-script orchestration for training an OCEAN persona adapter and
running standard evaluation sweeps. Wraps the OCT pipeline for data
generation + training, then generates eval configs and runs TRAIT and
MMLU sweeps.

Usage:
    # Full pipeline: distillation + training + trait sweep + mmlu sweep
    uv run --with-requirements scripts_dev/oct_pipeline/uv-oct-requirements.txt \
        python scripts_dev/oct_pipeline/run_ocean_persona.py \
        --constitution scripts_dev/oct_pipeline/ocean/agreeableness_low.json \
        --direction suppressor \
        --trait agreeableness \
        --teacher-model z-ai/glm-4.5-air \
        --version 2

    # Self-teacher (same model as teacher and student via OpenRouter)
    uv run --with-requirements scripts_dev/oct_pipeline/uv-oct-requirements.txt \
        python scripts_dev/oct_pipeline/run_ocean_persona.py \
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
import importlib
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


def _run(cmd: list[str], description: str) -> int:
    """Run a command, printing a header and streaming output."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*70}\n", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  FAILED (exit code {result.returncode})")
    return result.returncode


def _generate_eval_config(
    *,
    config_path: Path,
    base_model: str,
    adapter_repo: str,
    adapter_path_in_repo: str,
    eval_type: str,
    run_name: str,
    upload_path_in_repo: str,
    samples_per_trait: int = 300,
    mmlu_limit: int = 100,
    batch_size: int = 128,
) -> None:
    """Generate a SuiteConfig Python file for TRAIT or MMLU sweep."""
    ocean_traits = '["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]'

    if eval_type == "trait":
        eval_block = textwrap.dedent(f"""\
            evals=[
                InspectBenchmarkSpec(
                    name="trait",
                    benchmark="personality_trait_sampled",
                    benchmark_args={{"samples_per_trait": {samples_per_trait}, "trait_splits": {ocean_traits}, "max_tokens": 32}},
                    n_runs=1,
                ),
            ],
            temperature=0.0,""")
        sweep_line = 'sweep=ScaleSweep(points=_build_scale_points()),'
        analyze_kwargs = f'{{"title_suffix": "{run_name} TRAIT"}}'
    else:
        eval_block = textwrap.dedent(f"""\
            evals=[
                InspectBenchmarkSpec(
                    name="mmlu",
                    benchmark="mmlu",
                    limit={mmlu_limit},
                    n_runs=3,
                ),
            ],
            temperature=0.7,""")
        sweep_line = 'sweep=ScaleSweep(points=[round(-4.0 + i * 0.5, 10) for i in range(17) if round(-4.0 + i * 0.5, 10) != 0.0]),'
        analyze_kwargs = f'{{"random_baseline": 0.25, "title_suffix": "{run_name} MMLU"}}'

    config_content = textwrap.dedent(f"""\
        \"\"\"Auto-generated eval config for {run_name}.\"\"\"
        from pathlib import Path
        from dotenv import load_dotenv
        from src_dev.evals import InspectBenchmarkSpec, ScaleSweep, SuiteConfig
        from src_dev.utils.hf_hub import download_from_dataset_repo

        load_dotenv()

        BASE_MODEL = "{base_model}"
        _HF_REPO = "{adapter_repo}"
        _PATH_IN_REPO = "{adapter_path_in_repo}"
        _LOCAL_CACHE = Path("scratch/adapters/{run_name}")

        download_from_dataset_repo(
            repo_id=_HF_REPO,
            path_in_repo=_PATH_IN_REPO,
            local_dir=_LOCAL_CACHE,
        )
        _ADAPTER_URI = f"local://{{(_LOCAL_CACHE / _PATH_IN_REPO).resolve()}}"

        def _build_scale_points():
            coarse_neg = [round(-4.0 + i * 0.5, 10) for i in range(round((-2.5 - -4.0) / 0.5) + 1)]
            fine       = [round(-2.0 + i * 0.25, 10) for i in range(round((2.0 - -2.0) / 0.25) + 1)]
            coarse_pos = [round(2.5 + i * 0.5, 10) for i in range(round((4.0 - 2.5) / 0.5) + 1)]
            return sorted({{s for s in coarse_neg + fine + coarse_pos if s != 0.0}})

        SUITE_CONFIG = SuiteConfig(
            base_model=BASE_MODEL,
            adapter=_ADAPTER_URI,
            {sweep_line}
            {eval_block}
            batch_size={batch_size},
            output_root=Path("scratch/evals/ocean/{eval_type}"),
            run_name="{run_name}",
            skip_completed=True,
            auto_analyze=True,
            analyze_kwargs={analyze_kwargs},
            upload_repo_id=_HF_REPO,
            upload_path_in_repo="{upload_path_in_repo}",
            metadata={{
                "persona": "{run_name}",
                "adapter_repo": f"{{_HF_REPO}}::{{_PATH_IN_REPO}}",
            }},
        )
    """)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_content)
    print(f"  Generated eval config: {config_path}")


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
    parser.add_argument("--mmlu-limit", type=int, default=100,
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

        if args.skip_to == "training":
            pass  # OCT pipeline will skip distillation if artifacts exist

        rc = _run(oct_cmd, f"OCT Pipeline: {constitution_name} ({args.direction} v{args.version})")
        if rc != 0:
            sys.exit(rc)

        if args.stop_after in ("distillation", "training"):
            print(f"\nStopped after {args.stop_after}.")
            sys.exit(0)

    # =====================================================================
    # Stage 2: TRAIT sweep
    # =====================================================================
    if not args.skip_trait_eval:
        run_name = f"{trait_abbrev}{direction_tag}_v{args.version}"
        config_path = Path(f"scratch/eval_configs/{run_name}_trait.py")

        _generate_eval_config(
            config_path=config_path,
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            adapter_repo=monorepo_repo,
            adapter_path_in_repo=adapter_path_in_repo,
            eval_type="trait",
            run_name=run_name,
            upload_path_in_repo=f"fine_tuning/{args.student_model}/ocean/{args.trait}/evals/mcq/trait/{run_name}",
            samples_per_trait=args.samples_per_trait,
            batch_size=args.eval_batch_size,
        )

        # Import and run the generated config
        rc = _run(
            [sys.executable, "-m", "src_dev.evals", "suite",
             "--config-module", str(config_path).replace("/", ".").replace(".py", "")],
            f"TRAIT sweep: {run_name}",
        )

    # =====================================================================
    # Stage 3: MMLU sweep
    # =====================================================================
    if not args.skip_mmlu_eval:
        run_name = f"{trait_abbrev}{direction_tag}_v{args.version}"
        config_path = Path(f"scratch/eval_configs/{run_name}_mmlu.py")

        _generate_eval_config(
            config_path=config_path,
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            adapter_repo=monorepo_repo,
            adapter_path_in_repo=adapter_path_in_repo,
            eval_type="mmlu",
            run_name=f"{run_name}_mmlu",
            upload_path_in_repo=f"fine_tuning/{args.student_model}/ocean/{args.trait}/evals/mcq/mmlu/{run_name}",
            mmlu_limit=args.mmlu_limit,
            batch_size=args.eval_batch_size,
        )

        rc = _run(
            [sys.executable, "-m", "src_dev.evals", "suite",
             "--config-module", str(config_path).replace("/", ".").replace(".py", "")],
            f"MMLU sweep: {run_name}",
        )

    print(f"\n{'='*70}")
    print(f"  All stages complete: {constitution_name} ({args.direction} v{args.version})")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
