"""Run a bloom eval with per-stage caching and HuggingFace persistence.

Each pipeline stage gets a deterministic run ID hashed from only the config
fields that materially affect that stage's output.  This means:

  - Changing the judgment model / additional_qualities only busts the judgment
    run ID — rollout, ideation, and understanding are reused from cache.
  - Changing the target model only busts the rollout run ID.
  - Changing ideation params (num_scenarios, etc.) busts ideation + rollout +
    judgment, but not understanding.
  - Pass --seed N to get a fresh independent run of the same config.

Multiple judgment models can be run against the same rollout by passing
--judgment-models.  Each model produces its own judgment_run_id and is
cached/uploaded independently; the shared rollout is fetched from cache
and never re-run.

Cache lookup order for each stage:
  1. Local:  bloom-cache/bloom-evals/{stage}/{run_id}/
  2. Remote: HF repo bloom-evals/{stage}/{run_id}/
  3. Run bloom stage → save to local cache → upload to HF

Usage:
    uv run python scripts/experiments/run_bloom_eval.py --bloom-data bloom-data
    uv run python scripts/experiments/run_bloom_eval.py --bloom-data bloom-data --seed 1
    uv run python scripts/experiments/run_bloom_eval.py --bloom-data bloom-data --dry-run
    uv run python scripts/experiments/run_bloom_eval.py --bloom-data bloom-data --no-upload
    uv run python scripts/experiments/run_bloom_eval.py --bloom-data bloom-data \\
        --judgment-models claude-opus-4.6 gpt-5-mini gpt-5-nano
    uv run python scripts/experiments/run_bloom_eval.py --bloom-data bloom-data \\
        --targets llama-3.1-8b-it-base conscientiousness-low-llama \\
        --judgment-models claude-opus-4.6 gpt-5-mini gpt-5-nano
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Generator

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from src_dev.utils.hf_hub import (
    download_from_dataset_repo,
    login_from_env,
    upload_folder_to_dataset_repo,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_REPO_DEFAULT = "persona-shattering-lasr/monorepo"
HF_BASE_PATH = "bloom-evals"

STAGES = ["understanding", "ideation", "rollout", "judgment"]

# Configurable-prompt keys that materially affect each stage's output.
_PROMPT_KEYS_BY_STAGE: dict[str, list[str]] = {
    "understanding": [
        "understanding_system_additional",
        "behavior_understanding_additional",
        "transcript_analysis_additional",
    ],
    "ideation": [
        "ideation_system_additional",
        "make_scenarios_additional",
        "variation_system_additional",
        "make_variations_additional",
    ],
    "rollout": [
        "rollout_system_additional",
        "generate_sysprompt_additional",
        "target_sysprompt_prefix",
        "generate_kickoff_additional",
        "target_kickoff_prefix",
    ],
    "judgment": [
        "judgment_system_additional",
        "judgment_additional",
        "metajudge_system_additional",
        "metajudge_judgment_additional",
    ],
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_bloom_config(bloom_data_dir: Path) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    """Load seed.yaml, behaviors.json, and the active configurable_prompts file.

    Returns:
        (config, behaviors, prompts)
    """
    with open(bloom_data_dir / "seed.yaml") as f:
        config = yaml.safe_load(f)
    with open(bloom_data_dir / "behaviors.json") as f:
        behaviors = json.load(f)

    prompts_name = config.get("configurable_prompts", "default")
    prompts_path = bloom_data_dir / "configurable_prompts" / f"{prompts_name}.json"
    with open(prompts_path) as f:
        prompts = json.load(f)

    return config, behaviors, prompts


# ---------------------------------------------------------------------------
# Run ID computation
# ---------------------------------------------------------------------------


def _sha256_short(data: dict[str, Any], length: int = 12) -> str:
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:length]


def _prompts_subset(prompts: dict[str, str], stage: str) -> dict[str, str]:
    keys = _PROMPT_KEYS_BY_STAGE[stage]
    return {k: prompts.get(k, "") for k in keys}


def compute_run_ids(
    config: dict[str, Any],
    behaviors: dict[str, str],
    prompts: dict[str, str],
    seed: int,
) -> dict[str, str]:
    """Compute deterministic, chained run IDs for each pipeline stage.

    Each stage ID depends only on:
      - The previous stage's run ID (establishing the dependency chain)
      - The config fields that directly affect that stage
      - The configurable-prompt keys relevant to that stage
      - The seed (for stochastic stages: ideation, rollout, judgment)
    """
    behavior_name = config["behavior"]["name"]
    temperature = config.get("temperature", 1.0)
    evaluator_effort = config.get("evaluator_reasoning_effort", "low")
    target_effort = config.get("target_reasoning_effort", "medium")

    understanding_id = _sha256_short({
        "stage": "understanding",
        "behavior": behavior_name,
        "behavior_description": behaviors[behavior_name],
        "model": config["understanding"]["model"],
        "max_tokens": config["understanding"]["max_tokens"],
        "temperature": temperature,
        "evaluator_reasoning_effort": evaluator_effort,
        "prompts": _prompts_subset(prompts, "understanding"),
    })

    ideation_id = _sha256_short({
        "stage": "ideation",
        "understanding_run_id": understanding_id,
        "model": config["ideation"]["model"],
        "num_scenarios": config["ideation"]["num_scenarios"],
        "variation_dimensions": sorted(config["ideation"].get("variation_dimensions") or []),
        "max_tokens": config["ideation"]["max_tokens"],
        "web_search": config["ideation"].get("web_search", False),
        "temperature": temperature,
        "evaluator_reasoning_effort": evaluator_effort,
        "prompts": _prompts_subset(prompts, "ideation"),
        "seed": seed,
    })

    rollout_id = _sha256_short({
        "stage": "rollout",
        "ideation_run_id": ideation_id,
        "evaluator_model": config["rollout"]["model"],
        "target_model": config["rollout"]["target"],
        "modality": config["rollout"]["modality"],
        "max_turns": config["rollout"]["max_turns"],
        "max_tokens": config["rollout"]["max_tokens"],
        "num_reps": config["rollout"]["num_reps"],
        "no_user_mode": config["rollout"].get("no_user_mode", False),
        "selected_variations": config["rollout"].get("selected_variations"),
        "anonymous_target": config.get("anonymous_target", False),
        "temperature": temperature,
        "evaluator_reasoning_effort": evaluator_effort,
        "target_reasoning_effort": target_effort,
        "prompts": _prompts_subset(prompts, "rollout"),
        "seed": seed,
    })

    judgment_id = _sha256_short({
        "stage": "judgment",
        "rollout_run_id": rollout_id,
        "model": config["judgment"]["model"],
        "additional_qualities": sorted(config["judgment"].get("additional_qualities") or []),
        "metajudgment_qualities": sorted(config["judgment"].get("metajudgment_qualities") or []),
        "max_tokens": config["judgment"]["max_tokens"],
        "num_samples": config["judgment"].get("num_samples", 1),
        "redaction_tags": config["judgment"].get("redaction_tags"),
        "anonymous_target": config.get("anonymous_target", False),
        "temperature": temperature,
        "evaluator_reasoning_effort": evaluator_effort,
        "prompts": _prompts_subset(prompts, "judgment"),
        "seed": seed,
    })

    return {
        "understanding": understanding_id,
        "ideation": ideation_id,
        "rollout": rollout_id,
        "judgment": judgment_id,
    }


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_dir(cache_root: Path, stage: str, run_id: str) -> Path:
    """Local cache directory for one stage run."""
    return cache_root / HF_BASE_PATH / stage / run_id


def _hf_path(stage: str, run_id: str) -> str:
    return f"{HF_BASE_PATH}/{stage}/{run_id}"


def _stage_complete_in_dir(directory: Path, stage: str) -> bool:
    """Return True if the stage's primary output file exists in directory."""
    return (directory / f"{stage}.json").exists()


def _rollout_complete_in_dir(directory: Path) -> bool:
    return (directory / "rollout.json").exists()


def stage_complete_in_cache(cache_root: Path, stage: str, run_id: str) -> bool:
    cache = _cache_dir(cache_root, stage, run_id)
    if stage == "rollout":
        return _rollout_complete_in_dir(cache)
    return _stage_complete_in_dir(cache, stage)


def fetch_from_hf(cache_root: Path, stage: str, run_id: str, hf_repo: str) -> bool:
    """Try to pull stage outputs from HF into local cache. Returns True if found."""
    try:
        download_from_dataset_repo(
            repo_id=hf_repo,
            path_in_repo=_hf_path(stage, run_id),
            local_dir=cache_root,
            allow_patterns=["*"],  # scope to this run_id dir only, not the whole repo
        )
        return stage_complete_in_cache(cache_root, stage, run_id)
    except Exception:
        return False


def restore_from_cache(cache_root: Path, bloom_results_dir: Path, stage: str, run_id: str) -> None:
    """Copy all files for a stage from local cache into bloom-results/."""
    src = _cache_dir(cache_root, stage, run_id)
    bloom_results_dir.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        shutil.copy2(f, bloom_results_dir / f.name)


def save_to_cache(bloom_results_dir: Path, cache_root: Path, stage: str, run_id: str) -> None:
    """Copy freshly-run stage outputs from bloom-results/ into local cache."""
    dst = _cache_dir(cache_root, stage, run_id)
    dst.mkdir(parents=True, exist_ok=True)

    if stage == "rollout":
        files = list(bloom_results_dir.glob("transcript_*.json")) + [
            bloom_results_dir / "rollout.json"
        ]
    else:
        files = [bloom_results_dir / f"{stage}.json"]

    for src in files:
        if src.exists():
            shutil.copy2(src, dst / src.name)


def upload_stage(cache_root: Path, stage: str, run_id: str, hf_repo: str, behavior_name: str) -> None:
    cache = _cache_dir(cache_root, stage, run_id)
    url = upload_folder_to_dataset_repo(
        local_dir=cache,
        repo_id=hf_repo,
        path_in_repo=_hf_path(stage, run_id),
        commit_message=f"bloom eval · {behavior_name} · {stage} · {run_id}",
    )
    print(f"  ↑ Uploaded to {url}/tree/main/{_hf_path(stage, run_id)}")


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_bloom_stage(bloom_data_dir: Path, stage: str) -> None:
    cmd = ["uv", "run", "bloom", stage, str(bloom_data_dir)]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@contextlib.contextmanager
def patched_seed(
    bloom_data_dir: Path, overrides: dict[str, Any]
) -> Generator[None, None, None]:
    """Temporarily rewrite seed.yaml with arbitrary nested overrides.

    ``overrides`` is a dict of dot-path keys → values, e.g.::

        {"judgment.model": "gpt-5-mini", "rollout.target": "llama-3.1-8b-it-base"}

    Restores the original file on exit even if bloom crashes.
    """
    seed_path = bloom_data_dir / "seed.yaml"
    original_text = seed_path.read_text()
    patched = yaml.safe_load(original_text)
    for dotpath, value in overrides.items():
        keys = dotpath.split(".")
        node = patched
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    try:
        seed_path.write_text(yaml.dump(patched, allow_unicode=True, sort_keys=False))
        yield
    finally:
        seed_path.write_text(original_text)


def _run_one_stage(
    stage: str,
    run_id: str,
    bloom_data_dir: Path,
    bloom_results_dir: Path,
    cache_root: Path,
    hf_repo: str,
    behavior_name: str,
    no_upload: bool,
) -> None:
    """Check cache/HF for one stage run_id, running bloom only if needed."""
    print(f"── {stage.upper()} (run_id={run_id}) ──")

    if stage_complete_in_cache(cache_root, stage, run_id):
        print(f"  ✓ Found in local cache → restoring")
        restore_from_cache(cache_root, bloom_results_dir, stage, run_id)
        return

    if not no_upload:
        print(f"  Not in local cache → checking HF…")
        if fetch_from_hf(cache_root, stage, run_id, hf_repo):
            print(f"  ✓ Found on HF → restoring")
            restore_from_cache(cache_root, bloom_results_dir, stage, run_id)
            return
        print(f"  Not on HF → running stage")
    else:
        print(f"  Not in local cache → running stage")

    run_bloom_stage(bloom_data_dir, stage)
    save_to_cache(bloom_results_dir, cache_root, stage, run_id)

    if not no_upload:
        upload_stage(cache_root, stage, run_id, hf_repo, behavior_name)

    print(f"  ✓ Done")


def _config_for_target_and_judge(
    base_config: dict[str, Any], target: str | None, judge: str | None
) -> dict[str, Any]:
    """Return a deep-copied config with optional target and judge overrides."""
    cfg = copy.deepcopy(base_config)
    if target is not None:
        cfg["rollout"]["target"] = target
    if judge is not None:
        cfg["judgment"]["model"] = judge
    return cfg


def run_pipeline(
    bloom_data_dir: Path,
    seed: int,
    hf_repo: str,
    requested_stages: list[str],
    targets: list[str] | None,
    judgment_models: list[str] | None,
    dry_run: bool,
    no_upload: bool,
) -> None:
    cache_root = bloom_data_dir.parent / "bloom-cache"
    config, behaviors, prompts = load_bloom_config(bloom_data_dir)
    behavior_name = config["behavior"]["name"]
    bloom_results_dir = bloom_data_dir.parent / "bloom-results" / behavior_name

    # Resolve targets and judgment models (fall back to whatever is in seed.yaml)
    t_models = targets if targets else [config["rollout"]["target"]]
    j_models = judgment_models if judgment_models else [config["judgment"]["model"]]

    # Pre-compute all run IDs for the summary
    base_ids = compute_run_ids(config, behaviors, prompts, seed)
    per_target_ids: dict[str, dict[str, Any]] = {}
    for target in t_models:
        per_target_ids[target] = {}
        for judge in j_models:
            cfg = _config_for_target_and_judge(config, target, judge)
            ids = compute_run_ids(cfg, behaviors, prompts, seed)
            per_target_ids[target][judge] = ids

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"Behavior : {behavior_name}")
    print(f"Seed     : {seed}")
    print(f"Targets  : {', '.join(t_models)}")
    print(f"Judges   : {', '.join(j_models)}")
    print()
    print(f"  {'understanding':<16} {base_ids['understanding']}")
    print(f"  {'ideation':<16} {base_ids['ideation']}")
    for target in t_models:
        rollout_id = per_target_ids[target][j_models[0]]["rollout"]
        print(f"  rollout [{target}]")
        print(f"    {'':2}{rollout_id}")
        for judge in j_models:
            jid = per_target_ids[target][judge]["judgment"]
            print(f"    judgment [{judge}]  {jid}")
    print()

    if dry_run:
        print("[dry-run] No stages will be executed.")
        return

    if not no_upload:
        login_from_env()

    # ── Understanding + Ideation (shared across all targets) ─────────────────
    for stage in ["understanding", "ideation"]:
        if stage not in requested_stages:
            continue
        _run_one_stage(
            stage, base_ids[stage],
            bloom_data_dir, bloom_results_dir, cache_root,
            hf_repo, behavior_name, no_upload,
        )

    # ── Rollout + Judgment (once per target, judgment once per judge) ─────────
    for target in t_models:
        if len(t_models) > 1:
            print(f"\n══ target: {target} ══")

        rollout_id = per_target_ids[target][j_models[0]]["rollout"]

        if "rollout" in requested_stages:
            with patched_seed(bloom_data_dir, {"rollout.target": target}):
                _run_one_stage(
                    "rollout", rollout_id,
                    bloom_data_dir, bloom_results_dir, cache_root,
                    hf_repo, behavior_name, no_upload,
                )

        if "judgment" in requested_stages:
            for judge in j_models:
                jid = per_target_ids[target][judge]["judgment"]
                if len(j_models) > 1:
                    print(f"\n  [ judge: {judge} ]")
                with patched_seed(bloom_data_dir, {
                    "rollout.target": target,
                    "judgment.model": judge,
                }):
                    _run_one_stage(
                        "judgment", jid,
                        bloom_data_dir, bloom_results_dir, cache_root,
                        hf_repo, behavior_name, no_upload,
                    )

    print("\n── COMPLETE ──")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--bloom-data", default="bloom-data",
        help="Path to bloom-data directory (default: bloom-data)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed included in run IDs for stochastic stages. "
             "Increment to get an independent run of the same config. (default: 0)",
    )
    parser.add_argument(
        "--hf-repo", default=HF_REPO_DEFAULT,
        help=f"HuggingFace dataset repo for persistence (default: {HF_REPO_DEFAULT})",
    )
    parser.add_argument(
        "--stages", nargs="+", default=STAGES, choices=STAGES,
        help="Stages to include (default: all). Skipped stages are never re-run "
             "even if their run ID changed.",
    )
    parser.add_argument(
        "--targets", nargs="+", default=None, metavar="MODEL",
        help="Run rollout+judgment for each target model (short name from models.json "
             "or direct LiteLLM ID). Understanding and ideation are shared and run "
             "only once. Default: use the target in seed.yaml.",
    )
    parser.add_argument(
        "--judgment-models", nargs="+", default=None, metavar="MODEL",
        help="Run judgment once per model (short name from models.json or direct "
             "LiteLLM ID). Each gets its own run ID; the shared rollout is reused "
             "from cache. Default: use the model in seed.yaml.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print run IDs and exit without running anything.",
    )
    parser.add_argument(
        "--no-upload", action="store_true",
        help="Disable HF upload/download; use local cache only.",
    )
    args = parser.parse_args()

    run_pipeline(
        bloom_data_dir=Path(args.bloom_data).resolve(),
        seed=args.seed,
        hf_repo=args.hf_repo,
        requested_stages=args.stages,
        targets=args.targets,
        judgment_models=args.judgment_models,
        dry_run=args.dry_run,
        no_upload=args.no_upload,
    )


if __name__ == "__main__":
    main()
