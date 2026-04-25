"""Rank-reduce a vanton4_paired_dpo persona LoRA and run trait + MMLU evals
at the reduced rank. Parameterized over (persona, rank) so a single driver
handles all combinations from the shell wrapper.

Mirrors the per-rank vanton4_downrank{N} eval configs but builds the
SuiteConfigs in code instead of as one module per (rank, persona, eval_type).
Adapter paths are read from LoraHFCatalogue (which now points at
vanton4_paired_dpo).

Eval results land alongside the existing vanton4_paired_dpo evals on HF
(``.../vanton4_paired_dpo/evals/mcq/``) with a ``_downrank{N}`` suffix on
the eval name — these are variants of vanton4_paired_dpo, not a new
monorepo version.

Usage
-----
    uv run python scripts_dev/oct_pipeline/ocean/vanton4_paired_dpo_downranks/reduce_and_eval.py \\
        --persona a_plus --rank 4 [--skip-trait] [--skip-mmlu]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src_dev.common.lora_catalogue import HF_REPO, LoraHFCatalogue
from src_dev.evals import (
    InspectBenchmarkSpec,
    JudgeExecutionConfig,
    ScaleSweep,
    SuiteConfig,
)
from src_dev.evals.suite import run_eval_suite
from src_dev.utils.hf_hub import download_from_dataset_repo
from src_dev.utils.lora_rank_reduction import reduce_adapter_rank_on_disk

load_dotenv()

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

# slug -> trait abbrev letter
_ABBREV = {
    "openness": "O",
    "conscientiousness": "C",
    "extraversion": "E",
    "agreeableness": "A",
    "neuroticism": "N",
}

PERSONA_CHOICES = [
    "o_plus", "o_minus",
    "c_plus", "c_minus",
    "e_plus", "e_minus",
    "a_plus", "a_minus",
    "n_plus", "n_minus",
]


def _parse_persona_path(persona: str) -> tuple[str, str, str, str, str]:
    """Return (trait, direction, abbrev, sign, sign_word) by parsing the catalogue path.

    OCEAN catalogue values look like
        fine_tuning/<base>/ocean/<trait>/<direction>/<version>/lora/<adapter>
    """
    lora_path = Path(getattr(LoraHFCatalogue(), persona))
    parts = lora_path.parts
    trait = parts[3]
    direction = parts[4]
    sign = "+" if persona.endswith("_plus") else "-"
    sign_word = "plus" if sign == "+" else "minus"
    return trait, direction, _ABBREV[trait], sign, sign_word


def _build_scale_points() -> list[float]:
    """Step 0.5 in [-4, -2.5] and [+2.5, +4], step 0.25 in [-2, +2].

    Matches the scale grid used by vanton4_downrank{N} configs (finer than the
    full-rank vanton4 trait grid; we want more resolution at low rank).
    """
    coarse_neg = [round(-4.0 + i * 0.5, 10) for i in range(round((-2.5 - -4.0) / 0.5) + 1)]
    fine = [round(-2.0 + i * 0.25, 10) for i in range(round((2.0 - -2.0) / 0.25) + 1)]
    coarse_pos = [round(2.5 + i * 0.5, 10) for i in range(round((4.0 - 2.5) / 0.5) + 1)]
    return sorted({s for s in coarse_neg + fine + coarse_pos if s != 0.0})


def reduce_adapter(persona: str, rank: int) -> Path:
    """Download persona adapter, SVD-reduce to target rank, return local path.

    Idempotent: re-running with the same (persona, rank) is a no-op when the
    reduced cache already exists on disk.
    """
    lora_path_in_repo = getattr(LoraHFCatalogue(), persona)
    full_rank_cache = Path(f"scratch/adapters/{persona}_vanton4_paired_dpo")
    download_from_dataset_repo(
        repo_id=HF_REPO,
        path_in_repo=lora_path_in_repo,
        local_dir=full_rank_cache,
    )
    full_rank_path = full_rank_cache / lora_path_in_repo
    reduced_dir = Path(f"scratch/adapters/{persona}_vanton4_paired_dpo_downrank{rank}")
    reduce_adapter_rank_on_disk(
        source_dir=full_rank_path,
        target_dir=reduced_dir,
        new_rank=rank,
        base_model=BASE_MODEL,
    )
    return reduced_dir


def _adapter_uri(reduced_dir: Path) -> str:
    return f"local://{reduced_dir.resolve()}"


def build_trait_suite(persona: str, rank: int, adapter_uri: str) -> SuiteConfig:
    trait, direction, abbrev, sign, sign_word = _parse_persona_path(persona)
    return SuiteConfig(
        base_model=BASE_MODEL,
        adapter=adapter_uri,
        sweep=ScaleSweep(points=_build_scale_points()),
        evals=[
            InspectBenchmarkSpec(
                name="trait_logprobs",
                benchmark="personality_trait_logprobs",
                benchmark_args={"samples_per_trait": 300, "trait_splits": OCEAN_TRAITS},
                n_runs=1,
            ),
        ],
        temperature=0.0,
        batch_size=128,
        output_root=Path("scratch/evals/ocean/trait"),
        run_name=f"{persona}_vanton4_paired_dpo_downrank{rank}_logprobs",
        skip_completed=True,
        auto_analyze=True,
        analyze_kwargs={
            "title_suffix": f"{abbrev}{sign} vanton4_paired_dpo_downrank{rank} TRAIT (logprobs)",
            "interval": "ci95_from_bootstrap_1000",
            "min_choice_mass": 0.75,
        },
        upload_repo_id=HF_REPO,
        upload_path_in_repo=(
            f"fine_tuning/llama-3.1-8b-it/ocean/{trait}/{direction}/vanton4_paired_dpo/"
            f"evals/mcq/trait_logprobs_downrank{rank}"
        ),
        metadata={
            "persona": f"{trait}_{sign_word}_vanton4_paired_dpo_downrank{rank}",
            "scoring_method": "logprob",
            "svd_target_rank": rank,
            "source_adapter_repo": f"{HF_REPO}::{getattr(LoraHFCatalogue(), persona)}",
        },
    )


def build_mmlu_suite(persona: str, rank: int, adapter_uri: str) -> SuiteConfig:
    trait, direction, abbrev, sign, sign_word = _parse_persona_path(persona)
    return SuiteConfig(
        base_model=BASE_MODEL,
        adapter=adapter_uri,
        sweep=ScaleSweep(points=_build_scale_points()),
        evals=[
            InspectBenchmarkSpec(
                name="mmlu",
                benchmark="mmlu",
                limit=300,
                n_runs=1,
            ),
        ],
        temperature=0.0,
        batch_size=128,
        output_root=Path("scratch/evals/ocean/mmlu"),
        run_name=f"{persona}_vanton4_paired_dpo_downrank{rank}",
        skip_completed=True,
        auto_analyze=True,
        analyze_kwargs={
            "random_baseline": 0.25,
            "title_suffix": f"{abbrev}{sign} vanton4_paired_dpo_downrank{rank} MMLU",
            "interval": "ci95_from_wilson",
        },
        upload_repo_id=HF_REPO,
        upload_path_in_repo=(
            f"fine_tuning/llama-3.1-8b-it/ocean/{trait}/{direction}/vanton4_paired_dpo/"
            f"evals/mcq/mmlu_downrank{rank}"
        ),
        metadata={
            "persona": f"{trait}_{sign_word}_vanton4_paired_dpo_downrank{rank}",
            "svd_target_rank": rank,
            "source_adapter_repo": f"{HF_REPO}::{getattr(LoraHFCatalogue(), persona)}",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persona", required=True, choices=PERSONA_CHOICES)
    parser.add_argument(
        "--rank", type=int, required=True,
        help="Target SVD rank (e.g. 1, 2, 4, 8).",
    )
    parser.add_argument("--skip-trait", action="store_true", help="Skip trait eval.")
    parser.add_argument("--skip-mmlu", action="store_true", help="Skip MMLU eval.")
    args = parser.parse_args()

    print(f"\n=== reduce + eval: persona={args.persona} rank={args.rank} ===")
    reduced_dir = reduce_adapter(args.persona, args.rank)
    adapter_uri = _adapter_uri(reduced_dir)
    print(f"  reduced adapter: {adapter_uri}")

    judge_exec = JudgeExecutionConfig()

    if not args.skip_trait:
        print(f"\n--- trait eval ({args.persona} rank={args.rank}) ---")
        run_eval_suite(build_trait_suite(args.persona, args.rank, adapter_uri), judge_exec)

    if not args.skip_mmlu:
        print(f"\n--- mmlu eval ({args.persona} rank={args.rank}) ---")
        run_eval_suite(build_mmlu_suite(args.persona, args.rank, adapter_uri), judge_exec)


if __name__ == "__main__":
    main()
