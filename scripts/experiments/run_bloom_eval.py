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
        --judgment-models openrouter/moonshotai/kimi-k2-0905 gpt-5-mini gpt-5-nano
    uv run python scripts/experiments/run_bloom_eval.py --bloom-data bloom-data \\
        --targets llama-3.1-8b-it-base conscientiousness-low-llama \\
        --judgment-models openrouter/moonshotai/kimi-k2-0905 gpt-5-mini gpt-5-nano
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Generator

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import HfApi, hf_hub_download

from src_dev.utils.hf_hub import (
    _configure_timeout,
    _get_token,
    login_from_env,
    upload_folder_to_dataset_repo,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_REPO_DEFAULT = "persona-shattering-lasr/monorepo"
HF_BASE_PATH = "bloom-evals"

STAGES = ["understanding", "ideation", "rollout", "judgment"]

# Evaluator models permitted in any non-target role (understanding / ideation /
# rollout evaluator / judgment).  Local target models are exempt.
# Fail fast if anything outside this set is requested.
ALLOWED_EVALUATOR_MODEL_IDS: frozenset[str] = frozenset({
    "openrouter/moonshotai/kimi-k2-0905",
    "openrouter/openai/gpt-5-mini",
    "openrouter/openai/gpt-5-nano",
})

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
# Model allowlist
# ---------------------------------------------------------------------------


def _resolve_model_id(short_name: str, models_config: dict[str, Any]) -> str:
    """Return the LiteLLM model ID for a short name, or the name itself if not found."""
    entry = models_config.get(short_name)
    return entry["id"] if entry else short_name


def validate_evaluator_models(
    config: dict[str, Any],
    j_models: list[str],
    models_config: dict[str, Any],
) -> None:
    """Fail fast if any evaluator model is not in ALLOWED_EVALUATOR_MODEL_IDS.

    Checks understanding, ideation, rollout evaluator, and all judgment models.
    Local target models (org == 'local') are not checked here.
    """
    to_check = {
        "understanding": config["understanding"]["model"],
        "ideation": config["ideation"]["model"],
        "rollout evaluator": config["rollout"]["model"],
        **{f"judgment ({j})": j for j in j_models},
    }
    violations = []
    for role, name in to_check.items():
        resolved = _resolve_model_id(name, models_config)
        if resolved not in ALLOWED_EVALUATOR_MODEL_IDS:
            violations.append(f"  {role}: '{name}' → '{resolved}'")
    if violations:
        allowed = "\n".join(f"  {m}" for m in sorted(ALLOWED_EVALUATOR_MODEL_IDS))
        sys.exit(
            "Error: the following models are not in the allowed evaluator list:\n"
            + "\n".join(violations)
            + "\n\nAllowed models (LiteLLM IDs):\n"
            + allowed
        )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_bloom_config(bloom_data_dir: Path) -> tuple[dict[str, Any], dict[str, str], dict[str, str], dict[str, Any]]:
    """Load seed.yaml, behaviors.json, configurable_prompts, and models.json.

    Returns:
        (config, behaviors, prompts, models)
    """
    with open(bloom_data_dir / "seed.yaml") as f:
        config = yaml.safe_load(f)
    with open(bloom_data_dir / "behaviors.json") as f:
        behaviors = json.load(f)

    prompts_name = config.get("configurable_prompts", "default")
    prompts_path = bloom_data_dir / "configurable_prompts" / f"{prompts_name}.json"
    with open(prompts_path) as f:
        prompts = json.load(f)

    models_path = bloom_data_dir / "models.json"
    models: dict[str, Any] = {}
    if models_path.exists():
        with open(models_path) as f:
            models = json.load(f)

    return config, behaviors, prompts, models


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
    """Try to pull stage outputs from HF into local cache. Returns True if found.

    Uses list_repo_tree + hf_hub_download rather than snapshot_download to avoid
    listing (and rate-limiting against) the entire monorepo just to filter it.
    """
    _configure_timeout()
    token = _get_token()
    api = HfApi(token=token)
    hf_path = _hf_path(stage, run_id)

    try:
        entries = list(api.list_repo_tree(
            repo_id=hf_repo,
            repo_type="dataset",
            path_in_repo=hf_path,
        ))
    except Exception:
        return False

    if not entries:
        return False

    for entry in entries:
        hf_hub_download(
            repo_id=hf_repo,
            repo_type="dataset",
            filename=entry.path,
            local_dir=str(cache_root),
            token=token,
        )

    return stage_complete_in_cache(cache_root, stage, run_id)


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
# vLLM management
# ---------------------------------------------------------------------------

_vllm_proc: subprocess.Popen | None = None


def _vllm_base_url() -> str:
    return os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1").rstrip("/")


def _served_model_name(model_id: str) -> str:
    """Strip openai/ prefix from a LiteLLM local model ID."""
    return model_id.removeprefix("openai/")


def _query_vllm_models(base_url: str) -> list[str] | None:
    """Return list of model IDs served at base_url, or None if unreachable."""
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=5) as resp:
            data = json.loads(resp.read())
        return [m["id"] for m in data.get("data", [])]
    except Exception:
        return None


def _wait_for_vllm(base_url: str, model_names: list[str], timeout: int = 300) -> bool:
    """Poll until all model_names appear in vLLM or timeout (seconds) elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        available = _query_vllm_models(base_url)
        if available is not None and all(m in available for m in model_names):
            return True
        time.sleep(5)
    return False


def _cleanup_vllm() -> None:
    global _vllm_proc
    if _vllm_proc and _vllm_proc.poll() is None:
        print("\n  Stopping vLLM server…")
        _vllm_proc.terminate()
        try:
            _vllm_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _vllm_proc.kill()
        _vllm_proc = None


def _launch_vllm(
    local_targets: list[tuple[str, dict[str, Any]]],
    base_url: str,
    root: Path,
) -> None:
    """Build and launch a vLLM serve command for all local target models.

    All targets must share the same base model (vllm.model field).  LoRA
    adapters are passed via --enable-lora --lora-modules.
    """
    global _vllm_proc

    # Group targets by their base model path
    base_model_map: dict[str, dict[str, Any]] = {}
    max_lora_rank = 16  # vLLM default; raised to the max across all LoRA models
    for short_name, entry in local_targets:
        vllm_cfg = entry.get("vllm")
        if not vllm_cfg:
            sys.exit(
                f"Error: target '{short_name}' is a local model but has no 'vllm' "
                f"config in models.json.\n"
                f"Add a 'vllm' object with at least a 'model' (HuggingFace ID or local "
                f"path) to enable auto-launch."
            )
        base_model = vllm_cfg["model"]
        served_name = _served_model_name(entry["id"])
        if base_model not in base_model_map:
            base_model_map[base_model] = {"base_names": [], "loras": []}
        lora_path = vllm_cfg.get("lora_path")
        if lora_path:
            base_model_map[base_model]["loras"].append((served_name, lora_path))
            max_lora_rank = max(max_lora_rank, vllm_cfg.get("max_lora_rank", 16))
        else:
            base_model_map[base_model]["base_names"].append(served_name)

    if len(base_model_map) > 1:
        sys.exit(
            "Error: local targets span multiple base models — cannot serve on one "
            "vLLM instance:\n" + "\n".join(f"  {k}" for k in base_model_map) +
            "\nStart vLLM manually with separate instances on different ports, "
            "or use --no-vllm."
        )

    base_model_path, info = next(iter(base_model_map.items()))
    port_match = re.search(r":(\d+)", base_url)
    port = port_match.group(1) if port_match else "8000"

    cmd = ["uv", "run", "vllm", "serve", base_model_path, "--port", port]

    if info["base_names"]:
        cmd += ["--served-model-name", info["base_names"][0]]

    if info["loras"]:
        cmd += ["--enable-lora", "--max-lora-rank", str(max_lora_rank)]
        for lora_name, lora_path in info["loras"]:
            resolved = (
                str((root / lora_path).resolve())
                if not Path(lora_path).is_absolute()
                else lora_path
            )
            cmd += ["--lora-modules", f"{lora_name}={resolved}"]

    print(f"\n  Launching vLLM: {' '.join(cmd)}\n")
    _vllm_proc = subprocess.Popen(cmd)
    atexit.register(_cleanup_vllm)


def ensure_vllm_running(
    targets: list[str],
    models_config: dict[str, Any],
    no_vllm: bool,
    root: Path,
) -> None:
    """Ensure vLLM is serving all local target models, launching if needed.

    If vLLM is already running with the required models, does nothing.
    If not running and no_vllm=False, launches vLLM automatically.
    If not running and no_vllm=True, prints instructions and exits.
    """
    local_targets = [
        (t, models_config[t])
        for t in targets
        if t in models_config and models_config[t].get("org") == "local"
    ]
    if not local_targets:
        return

    base_url = _vllm_base_url()
    needed = [_served_model_name(entry["id"]) for _, entry in local_targets]

    available = _query_vllm_models(base_url)
    if available is not None and all(n in available for n in needed):
        print(f"  ✓ vLLM already serving: {', '.join(needed)}")
        return

    if no_vllm:
        print(f"\nError: vLLM is not serving required models at {base_url}")
        if available is None:
            print("  vLLM does not appear to be running.")
        else:
            missing = [n for n in needed if n not in available]
            print(f"  Currently serving: {available}")
            print(f"  Missing:           {missing}")
        print(
            "\nTo start vLLM manually (example for both models):\n"
            "  uv run vllm serve meta-llama/Llama-3.1-8B-Instruct \\\n"
            "    --served-model-name llama-3.1-8b-it-base \\\n"
            "    --enable-lora \\\n"
            "    --lora-modules llama-3.1-8b-it-conscientiousness_low_v2=<path/to/lora>\n"
            "\nOr re-run without --no-vllm to auto-launch."
        )
        sys.exit(1)

    print(f"  vLLM not detected at {base_url} → launching automatically…")
    _launch_vllm(local_targets, base_url, root)
    print(f"  Waiting for vLLM to be ready (up to 5 min)…")
    if not _wait_for_vllm(base_url, needed, timeout=300):
        sys.exit(
            f"Error: vLLM did not become ready within 5 minutes.\n"
            f"Check the vLLM process output above for errors."
        )
    print(f"  ✓ vLLM ready: {', '.join(needed)}")


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_bloom_stage(bloom_data_dir: Path, stage: str) -> None:
    cmd = ["uv", "run", "bloom", stage, str(bloom_data_dir)]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@contextlib.contextmanager
def patched_bloom_data(
    bloom_data_dir: Path, overrides: dict[str, Any]
) -> Generator[Path, None, None]:
    """Yield a temporary copy of bloom_data_dir with seed.yaml overrides applied.

    The original bloom-data directory is never modified.  A fresh temp directory
    is created, the bloom-data tree is copied into it, and seed.yaml in the copy
    is patched.  The temp directory is cleaned up on exit regardless of errors.

    ``overrides`` is a dict of dot-path keys → values, e.g.::

        {"judgment.model": "gpt-5-mini", "rollout.target": "llama-3.1-8b-it-base"}

    Yields the path to the patched copy so the caller can point bloom at it.
    """
    with tempfile.TemporaryDirectory(prefix="bloom_data_") as tmp:
        tmp_dir = Path(tmp) / bloom_data_dir.name
        shutil.copytree(bloom_data_dir, tmp_dir)
        seed_path = tmp_dir / "seed.yaml"
        patched = yaml.safe_load(seed_path.read_text())
        for dotpath, value in overrides.items():
            keys = dotpath.split(".")
            node = patched
            for k in keys[:-1]:
                node = node[k]
            node[keys[-1]] = value
        seed_path.write_text(yaml.dump(patched, allow_unicode=True, sort_keys=False))
        yield tmp_dir


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
    no_vllm: bool = False,
) -> None:
    cache_root = bloom_data_dir.parent / "bloom-cache"
    config, behaviors, prompts, models_config = load_bloom_config(bloom_data_dir)
    behavior_name = config["behavior"]["name"]
    bloom_results_dir = bloom_data_dir.parent / "bloom-results" / behavior_name

    # Resolve targets and judgment models (fall back to whatever is in seed.yaml)
    t_models = targets if targets else [config["rollout"]["target"]]
    j_models = judgment_models if judgment_models else [config["judgment"]["model"]]

    # Fail fast if any evaluator model is not in the allowlist
    validate_evaluator_models(config, j_models, models_config)

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

    # ── vLLM health-check / auto-launch for local targets ────────────────────
    if "rollout" in requested_stages and not dry_run:
        ensure_vllm_running(t_models, models_config, no_vllm, ROOT)

    # ── Rollout + Judgment (once per target, judgment once per judge) ─────────
    for target in t_models:
        if len(t_models) > 1:
            print(f"\n══ target: {target} ══")

        rollout_id = per_target_ids[target][j_models[0]]["rollout"]

        if "rollout" in requested_stages:
            with patched_bloom_data(bloom_data_dir, {"rollout.target": target}) as tmp_dir:
                _run_one_stage(
                    "rollout", rollout_id,
                    tmp_dir, bloom_results_dir, cache_root,
                    hf_repo, behavior_name, no_upload,
                )

        if "judgment" in requested_stages:
            for judge in j_models:
                jid = per_target_ids[target][judge]["judgment"]
                if len(j_models) > 1:
                    print(f"\n  [ judge: {judge} ]")
                with patched_bloom_data(bloom_data_dir, {
                    "rollout.target": target,
                    "judgment.model": judge,
                }) as tmp_dir:
                    _run_one_stage(
                        "judgment", jid,
                        tmp_dir, bloom_results_dir, cache_root,
                        hf_repo, behavior_name, no_upload,
                    )

    print("\n── COMPLETE ──")

    plot_results(cache_root, per_target_ids, j_models, behavior_name)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

# Additional quality keys present in every judgment entry
_QUALITY_KEYS = ["unrealism", "evaluation_awareness", "evaluation_invalidity", "coherence"]
# OCEAN offset: bloom score → OCEAN value
_OCEAN_OFFSET = 5


def _load_judgment_scores(
    cache_root: Path,
    per_target_ids: dict[str, dict[str, Any]],
    j_models: list[str],
) -> dict[tuple[str, str], dict[str, list[float]]]:
    """Load all scores from cached judgment.json files.

    Returns a dict keyed by (target, judge) → {metric: [scores]}.
    Metrics: 'conscientiousness' (OCEAN scale) + _QUALITY_KEYS.
    """
    results: dict[tuple[str, str], dict[str, list[float]]] = {}
    for target, judge_ids in per_target_ids.items():
        for judge, ids in judge_ids.items():
            jid = ids["judgment"]
            path = _cache_dir(cache_root, "judgment", jid) / "judgment.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            scores: dict[str, list[float]] = {k: [] for k in ["conscientiousness"] + _QUALITY_KEYS}
            for j in data.get("judgments", []):
                bp = j.get("behavior_presence")
                if bp is not None:
                    scores["conscientiousness"].append(float(bp) - _OCEAN_OFFSET)
                for q in _QUALITY_KEYS:
                    v = j.get(q)
                    if v is not None:
                        scores[q].append(float(v))
            results[(target, judge)] = scores
    return results


def plot_results(
    cache_root: Path,
    per_target_ids: dict[str, dict[str, Any]],
    j_models: list[str],
    behavior_name: str,
) -> None:
    """Load judgment scores from cache and save bar-chart summary to PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np
    except ImportError:
        print("  [plot] matplotlib not available — skipping visualisation")
        return

    all_scores = _load_judgment_scores(cache_root, per_target_ids, j_models)
    if not all_scores:
        print("  [plot] No judgment results found in cache — skipping visualisation")
        return

    t_models = list(dict.fromkeys(t for t, _ in all_scores))  # ordered unique targets
    out_dir = cache_root.parent / "bloom-results" / behavior_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.png"

    # ── Layout: 3 rows ────────────────────────────────────────────────────────
    # Row 1: conscientiousness mean ± std, grouped by judge (one bar cluster per target)
    # Row 2: box plots of conscientiousness score distribution per target × judge
    # Row 3: additional quality metrics (mean per target, one subplot each)
    n_qualities = len(_QUALITY_KEYS)
    fig = plt.figure(figsize=(max(10, 3 * len(t_models) + 2), 14))
    gs = fig.add_gridspec(3, n_qualities, hspace=0.55, wspace=0.4)

    ax_bar = fig.add_subplot(gs[0, :])   # full-width top
    ax_box = fig.add_subplot(gs[1, :])   # full-width middle
    quality_axes = [fig.add_subplot(gs[2, i]) for i in range(n_qualities)]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    judge_color = {j: colors[i % len(colors)] for i, j in enumerate(j_models)}

    x = np.arange(len(t_models))
    n_judges = len(j_models)
    width = 0.7 / max(n_judges, 1)

    # ── Row 1: mean conscientiousness ─────────────────────────────────────────
    for ji, judge in enumerate(j_models):
        means, errs = [], []
        for target in t_models:
            sc = all_scores.get((target, judge), {}).get("conscientiousness", [])
            means.append(float(np.mean(sc)) if sc else float("nan"))
            errs.append(float(np.std(sc)) if sc else 0.0)
        offset = (ji - (n_judges - 1) / 2) * width
        bars = ax_bar.bar(
            x + offset, means, width * 0.9,
            yerr=errs, capsize=3,
            label=judge, color=judge_color[judge], alpha=0.85,
        )
        # Scatter individual points
        for ti, target in enumerate(t_models):
            sc = all_scores.get((target, judge), {}).get("conscientiousness", [])
            if sc:
                ax_bar.scatter(
                    [x[ti] + offset] * len(sc), sc,
                    color=judge_color[judge], s=18, zorder=5, alpha=0.6,
                )

    ax_bar.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(t_models, rotation=15, ha="right", fontsize=9)
    ax_bar.set_ylabel("OCEAN conscientiousness\n(bloom score − 5)")
    ax_bar.set_title(f"{behavior_name} — mean conscientiousness score by target & judge")
    ax_bar.set_ylim(-4.5, 4.5)
    ax_bar.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax_bar.legend(fontsize=8, title="Judge", title_fontsize=8)

    # ── Row 2: box plots ──────────────────────────────────────────────────────
    box_data, box_labels, box_colors = [], [], []
    for target in t_models:
        for judge in j_models:
            sc = all_scores.get((target, judge), {}).get("conscientiousness", [])
            box_data.append(sc if sc else [float("nan")])
            short_t = target.split("/")[-1][:20]
            box_labels.append(f"{short_t}\n({judge[:12]})")
            box_colors.append(judge_color[judge])

    bp = ax_box.boxplot(
        box_data, patch_artist=True, medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax_box.set_xticks(range(1, len(box_labels) + 1))
    ax_box.set_xticklabels(box_labels, rotation=30, ha="right", fontsize=7)
    ax_box.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_box.set_ylabel("OCEAN conscientiousness")
    ax_box.set_title("Score distribution")
    ax_box.set_ylim(-4.5, 4.5)

    # ── Row 3: additional quality metrics ─────────────────────────────────────
    for qi, (ax_q, qkey) in enumerate(zip(quality_axes, _QUALITY_KEYS)):
        for ji, judge in enumerate(j_models):
            means = []
            for target in t_models:
                sc = all_scores.get((target, judge), {}).get(qkey, [])
                means.append(float(np.mean(sc)) if sc else float("nan"))
            offset = (ji - (n_judges - 1) / 2) * width
            ax_q.bar(
                x + offset, means, width * 0.9,
                label=judge, color=judge_color[judge], alpha=0.85,
            )
        ax_q.set_xticks(x)
        ax_q.set_xticklabels(
            [t.split("/")[-1][:15] for t in t_models],
            rotation=20, ha="right", fontsize=7,
        )
        ax_q.set_title(qkey.replace("_", " "), fontsize=9)
        ax_q.set_ylim(0, 10)
        ax_q.yaxis.set_major_locator(mticker.MultipleLocator(2))
        if qi == 0:
            ax_q.set_ylabel("Mean score (1–10)", fontsize=8)

    fig.suptitle(
        f"Bloom eval — {behavior_name}",
        fontsize=13, fontweight="bold", y=0.98,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved → {out_path}")


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
    parser.add_argument(
        "--no-vllm", action="store_true",
        help="Disable automatic vLLM launch for local target models. "
             "The script will error with instructions if vLLM is not already running.",
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
        no_vllm=args.no_vllm,
    )


if __name__ == "__main__":
    main()
