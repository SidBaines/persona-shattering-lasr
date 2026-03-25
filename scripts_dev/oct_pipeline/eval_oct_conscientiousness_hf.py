"""Evaluate OCT model variants with the conscientiousness_v2 LLM judge.

This standalone experiment script:

1. Rehydrates OCT LoRA artifacts from a Hugging Face dataset repo
2. Runs prompt-only rollouts for the base, DPO, SFT, and persona variants
3. Scores each rollout with the ``conscientiousness_v2`` judge
4. Exports per-sample scores, aggregate summaries, and a bar chart
5. Caches all stages locally and optionally mirrors them to a HF dataset repo

Example:
    python scripts/experiments/oct_pipeline/eval_oct_conscientiousness_hf.py \
        --source-hf-repo persona-shattering-lasr/oct-runs-low-conscientiousness-full-v2-openrouter-expand \
        --results-hf-repo persona-shattering-lasr/oct-runs-low-conscientiousness-full-v2-openrouter-expand
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from inspect_ai.log import EvalLog, read_eval_log
from pydantic import BaseModel, Field

from src_dev.common.config import DatasetConfig, GenerationConfig
from src_dev.datasets import format_for_inference, load_dataset_from_config
from src_dev.evals import (
    AdapterConfig,
    InspectCustomEvalSpec,
    ModelSpec,
    SuiteConfig,
    run_eval_suite,
)
from src_dev.persona_metrics.config import JudgeLLMConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt

load_dotenv()


_STAGE_META_DIR = ".oct_conscientiousness_eval"
_RUN_CONFIG_FILENAME = "run_config.json"
_DEFAULT_OUTPUT_ROOT = Path("scratch/oct_pipeline_eval_runs")
_DEFAULT_MODEL_CACHE_ROOT = Path("/root/.cache/models")
_DEFAULT_PROMPT_DATASET = "mirlab/TRAIT"
_DEFAULT_PROMPT_SPLIT = "Conscientiousness"
_DEFAULT_EVAL_NAME = "conscientiousness_rollout_judge"
_DEFAULT_SUITE_RUN_NAME = "suite"
_DEFAULT_HF_LOG_REPO = "hf://datasets/persona-shattering-lasr/eval-logs"

_MODEL_REPO_ALIASES = {
    "llama-3.1-8b-it": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen-2.5-7b-it": "Qwen/Qwen2.5-7B-Instruct",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
}

_MODEL_VARIANTS: tuple[tuple[str, str, str | None], ...] = (
    ("base", "No LoRA", None),
    ("dpo", "DPO only", "dpo"),
    ("sft", "SFT only", "sft"),
    ("persona", "Combination", "persona"),
)

_MODEL_COLORS = {
    "No LoRA": "#37474f",
    "DPO only": "#1565c0",
    "SFT only": "#2e7d32",
    "Combination": "#ef6c00",
}


class PromptSourceConfig(BaseModel):
    """Prompt dataset configuration for rollout generation."""

    source: str = "huggingface"
    name: str = _DEFAULT_PROMPT_DATASET
    split: str = _DEFAULT_PROMPT_SPLIT
    path: str | None = None
    question_column: str | None = "question"
    max_samples: int = 100
    seed: int = 223457


class ExperimentConfig(BaseModel):
    """Semantic config used for deterministic run IDs and caching."""

    source_hf_repo: str
    source_run_dir: str | None = None
    results_hf_repo: str | None = None
    constitution: str | None = None
    base_model_ref: str | None = None
    model_cache_root: str = str(_DEFAULT_MODEL_CACHE_ROOT)
    prompt_source: PromptSourceConfig = Field(default_factory=PromptSourceConfig)
    judge: JudgeLLMConfig = Field(
        default_factory=lambda: JudgeLLMConfig(
            provider="openai",
            model="gpt-5-nano-2025-08-07",
            temperature=0.0,
            max_tokens=4096,
            max_concurrent=10,
        )
    )
    generation: GenerationConfig = Field(
        default_factory=lambda: GenerationConfig(
            max_new_tokens=4096,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            batch_size=8,
            num_responses_per_prompt=1,
        )
    )
    hf_log_dir: str | None = _DEFAULT_HF_LOG_REPO


def _run_config_path(out_path: Path) -> Path:
    return out_path / _STAGE_META_DIR / _RUN_CONFIG_FILENAME


def _stage_marker_path(out_path: Path, stage_name: str) -> Path:
    return out_path / _STAGE_META_DIR / "stages" / f"{stage_name}.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_exists(path: Path, kind: str) -> bool:
    if kind == "file":
        return path.is_file() and path.stat().st_size > 0
    if kind == "dir":
        return path.is_dir() and any(path.iterdir())
    raise ValueError(f"Unsupported artifact kind: {kind}")


def _stage_artifacts_ready(artifacts: list[dict[str, Any]]) -> bool:
    return all(_artifact_exists(item["path"], item["kind"]) for item in artifacts)


def _write_stage_marker(
    *,
    out_path: Path,
    stage_name: str,
    config_hash: str,
    artifacts: list[dict[str, Any]],
) -> Path:
    marker_path = _stage_marker_path(out_path, stage_name)
    payload = {
        "stage": stage_name,
        "config_hash": config_hash,
        "artifacts": [
            {
                "relative_path": str(item["path"].relative_to(out_path)),
                "kind": item["kind"],
            }
            for item in artifacts
        ],
    }
    _write_json(marker_path, payload)
    return marker_path


def _stage_is_cached_locally(
    *,
    out_path: Path,
    stage_name: str,
    config_hash: str,
    artifacts: list[dict[str, Any]],
) -> bool:
    marker_path = _stage_marker_path(out_path, stage_name)
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("config_hash") == config_hash and _stage_artifacts_ready(artifacts):
            return True

    if _stage_artifacts_ready(artifacts):
        _write_stage_marker(
            out_path=out_path,
            stage_name=stage_name,
            config_hash=config_hash,
            artifacts=artifacts,
        )
        return True
    return False


def _ensure_run_config(out_path: Path, run_id: str, config_hash: str, config_payload: dict[str, Any]) -> Path:
    config_path = _run_config_path(out_path)
    payload = {
        "run_id": run_id,
        "config_hash": config_hash,
        "config": config_payload,
    }
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != config_hash:
            raise RuntimeError(
                f"Run directory {out_path} already contains a different config.\n"
                f"Existing hash: {existing.get('config_hash')}\n"
                f"Current hash:  {config_hash}"
            )
    else:
        _write_json(config_path, payload)
    return config_path


def _get_hf_helpers() -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    return {
        "api": HfApi(token=_hf_token()),
        "hf_hub_download": hf_hub_download,
        "snapshot_download": snapshot_download,
    }


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    return token or None


def _remote_repo_path(run_id: str, relative_path: Path) -> str:
    return f"{run_id}/{relative_path.as_posix()}"


def _copy_downloaded_artifact(downloaded_path: Path, destination: Path, kind: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if kind == "file":
        shutil.copy2(downloaded_path, destination)
        return
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(downloaded_path, destination)


def _download_artifact_from_hf(
    *,
    repo_id: str,
    remote_path: str,
    destination: Path,
    kind: str,
) -> bool:
    helpers = _get_hf_helpers()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            if kind == "file":
                downloaded = helpers["hf_hub_download"](
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=remote_path,
                    local_dir=str(tmp_root),
                    token=_hf_token(),
                )
            else:
                helpers["snapshot_download"](
                    repo_id=repo_id,
                    repo_type="dataset",
                    allow_patterns=[remote_path, f"{remote_path}/**"],
                    local_dir=str(tmp_root),
                    token=_hf_token(),
                )
                downloaded = tmp_root / remote_path
            _copy_downloaded_artifact(Path(downloaded), destination, kind)
        return True
    except Exception as exc:
        print(f"  HF download failed for {remote_path}: {exc}")
        return False


def _upload_artifact_to_hf(
    *,
    repo_id: str,
    relative_path: Path,
    local_path: Path,
    kind: str,
    run_id: str,
    commit_message: str,
) -> bool:
    helpers = _get_hf_helpers()
    path_in_repo = _remote_repo_path(run_id, relative_path)
    try:
        helpers["api"].create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
        if kind == "file":
            helpers["api"].upload_file(
                path_or_fileobj=str(local_path),
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo=path_in_repo,
                commit_message=commit_message,
            )
        else:
            helpers["api"].upload_folder(
                folder_path=str(local_path),
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo=path_in_repo,
                commit_message=commit_message,
            )
        return True
    except Exception as exc:
        print(f"  HF upload failed for {path_in_repo}: {exc}")
        return False


def _ensure_stage_available(
    *,
    out_path: Path,
    run_id: str,
    stage_name: str,
    config_hash: str,
    artifacts: list[dict[str, Any]],
    hf_repo_id: str | None,
    allow_download: bool = True,
) -> bool:
    if _stage_is_cached_locally(
        out_path=out_path,
        stage_name=stage_name,
        config_hash=config_hash,
        artifacts=artifacts,
    ):
        print(f"  Reusing local {stage_name} artifacts")
        return True

    if not hf_repo_id or not allow_download:
        return False

    marker_rel = _stage_marker_path(out_path, stage_name).relative_to(out_path)
    marker_remote = _remote_repo_path(run_id, marker_rel)

    try:
        helpers = _get_hf_helpers()
        marker_exists = marker_remote in helpers["api"].list_repo_files(
            repo_id=hf_repo_id,
            repo_type="dataset",
        )
    except Exception as exc:
        print(f"  HF lookup failed for {marker_remote}: {exc}")
        return False

    if not marker_exists:
        return False

    print(f"  Downloading cached {stage_name} artifacts from Hugging Face")
    if not _download_artifact_from_hf(
        repo_id=hf_repo_id,
        remote_path=marker_remote,
        destination=_stage_marker_path(out_path, stage_name),
        kind="file",
    ):
        return False

    for item in artifacts:
        if _artifact_exists(item["path"], item["kind"]):
            continue
        remote_path = _remote_repo_path(run_id, item["path"].relative_to(out_path))
        if not _download_artifact_from_hf(
            repo_id=hf_repo_id,
            remote_path=remote_path,
            destination=item["path"],
            kind=item["kind"],
        ):
            return False

    return _stage_is_cached_locally(
        out_path=out_path,
        stage_name=stage_name,
        config_hash=config_hash,
        artifacts=artifacts,
    )


def _publish_stage(
    *,
    out_path: Path,
    run_id: str,
    stage_name: str,
    config_hash: str,
    artifacts: list[dict[str, Any]],
    hf_repo_id: str | None,
) -> None:
    marker_path = _write_stage_marker(
        out_path=out_path,
        stage_name=stage_name,
        config_hash=config_hash,
        artifacts=artifacts,
    )
    if hf_repo_id is None:
        return

    commit_message = f"OCT conscientiousness eval {stage_name}: {run_id}"
    _upload_artifact_to_hf(
        repo_id=hf_repo_id,
        relative_path=marker_path.relative_to(out_path),
        local_path=marker_path,
        kind="file",
        run_id=run_id,
        commit_message=commit_message,
    )
    for item in artifacts:
        _upload_artifact_to_hf(
            repo_id=hf_repo_id,
            relative_path=item["path"].relative_to(out_path),
            local_path=item["path"],
            kind=item["kind"],
            run_id=run_id,
            commit_message=commit_message,
        )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _list_top_level_repo_folders(repo_id: str) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    entries = list(api.list_repo_tree(repo_id=repo_id, repo_type="dataset", recursive=False))
    names: list[str] = []
    for entry in entries:
        path = getattr(entry, "path", None)
        if not isinstance(path, str):
            continue
        if path.startswith("."):
            continue
        if "/" in path:
            continue
        if type(entry).__name__.lower().endswith("folder"):
            names.append(path)
    return sorted(names)


def _resolve_source_run_dir(config: ExperimentConfig) -> str:
    if config.source_run_dir:
        return config.source_run_dir

    folders = _list_top_level_repo_folders(config.source_hf_repo)
    helpers = _get_hf_helpers()
    try:
        repo_files = set(
            helpers["api"].list_repo_files(
                repo_id=config.source_hf_repo,
                repo_type="dataset",
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not inspect source HF repo {config.source_hf_repo}: {exc}"
        ) from exc

    oct_run_folders = [
        folder
        for folder in folders
        if _source_run_config_artifact_path(folder) in repo_files
    ]
    if len(oct_run_folders) == 1:
        return oct_run_folders[0]

    if len(folders) != 1:
        raise ValueError(
            "Could not infer source run dir uniquely from the source HF repo. "
            f"Found top-level folders: {folders}. "
            f"Folders that look like OCT source runs: {oct_run_folders}. "
            "Pass --source-run-dir explicitly."
        )
    return folders[0]


def _source_run_config_artifact_path(source_run_dir: str) -> str:
    return f"{source_run_dir}/.oct_pipeline/run_config.json"


def _download_text_file_from_dataset_repo(repo_id: str, path_in_repo: str) -> str:
    helpers = _get_hf_helpers()
    with tempfile.TemporaryDirectory() as tmpdir:
        downloaded = helpers["hf_hub_download"](
            repo_id=repo_id,
            repo_type="dataset",
            filename=path_in_repo,
            local_dir=tmpdir,
            token=_hf_token(),
        )
        return Path(downloaded).read_text(encoding="utf-8")


def _load_source_run_config(config: ExperimentConfig, source_run_dir: str) -> dict[str, Any]:
    path_in_repo = _source_run_config_artifact_path(source_run_dir)
    return json.loads(_download_text_file_from_dataset_repo(config.source_hf_repo, path_in_repo))


def _default_constitution(source_run_config_payload: dict[str, Any]) -> str:
    config = source_run_config_payload.get("config", {})
    constitution = config.get("constitution")
    if not constitution:
        raise ValueError("Could not determine constitution from source OCT run config.")
    return str(constitution)


def _default_base_model_ref(source_run_config_payload: dict[str, Any]) -> str:
    config = source_run_config_payload.get("config", {})
    raw_model = config.get("model")
    if not raw_model:
        raise ValueError("Could not determine base model from source OCT run config.")
    return _MODEL_REPO_ALIASES.get(str(raw_model), str(raw_model))


def _config_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _short_hash(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _make_run_id(config: ExperimentConfig, source_run_dir: str, constitution: str, base_model_ref: str) -> str:
    payload = {
        **config.model_dump(),
        "source_run_dir": source_run_dir,
        "constitution": constitution,
        "base_model_ref": base_model_ref,
    }
    digest = _short_hash(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    sample_count = config.prompt_source.max_samples
    sample_seed = config.prompt_source.seed
    source_tag = source_run_dir.replace("_", "-")
    return (
        f"oct-conscientiousness-v2-{source_tag}-"
        f"n{sample_count}-s{sample_seed}-{digest}"
    )


def _resolve_local_base_model_path(base_model_ref: str, model_cache_root: Path) -> tuple[str, Path]:
    from src_dev.evals.model_resolution import resolve_model_reference

    if base_model_ref.startswith("local://"):
        resolved = Path(resolve_model_reference(base_model_ref, kind="base model"))
        return f"local://{resolved}", resolved

    if base_model_ref.startswith("/") or base_model_ref.startswith("./") or base_model_ref.startswith("../"):
        resolved = Path(resolve_model_reference(f"local://{base_model_ref}", kind="base model"))
        return f"local://{resolved}", resolved

    alias = next(
        (key for key, value in _MODEL_REPO_ALIASES.items() if value == base_model_ref),
        base_model_ref.split("/")[-1].lower().replace(".", "-"),
    )
    local_dir = model_cache_root / alias
    if local_dir.is_dir() and any(local_dir.iterdir()):
        return f"local://{local_dir.resolve()}", local_dir.resolve()

    return "", local_dir.resolve()


def _ensure_base_model(base_model_ref: str, model_cache_root: Path) -> str:
    local_ref, local_path = _resolve_local_base_model_path(base_model_ref, model_cache_root)
    if local_ref:
        return local_ref

    from huggingface_hub import snapshot_download

    print(f"  Downloading base model {base_model_ref} to {local_path}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=base_model_ref,
        local_dir=str(local_path),
        token=_hf_token(),
    )
    return f"local://{local_path}"


def _source_artifacts(out_path: Path, constitution: str) -> list[dict[str, Any]]:
    source_root = out_path / "source_oct_run"
    return [
        {"path": source_root / ".oct_pipeline" / "run_config.json", "kind": "file"},
        {"path": source_root / "lora" / f"{constitution}-dpo", "kind": "dir"},
        {"path": source_root / "lora" / f"{constitution}-sft", "kind": "dir"},
        {"path": source_root / "lora" / f"{constitution}-persona", "kind": "dir"},
    ]


def _hydrate_source_artifacts(
    *,
    config: ExperimentConfig,
    out_path: Path,
    run_id: str,
    config_hash: str,
    source_run_dir: str,
    constitution: str,
    rerun: bool,
) -> None:
    artifacts = _source_artifacts(out_path, constitution)
    if not rerun and _ensure_stage_available(
        out_path=out_path,
        run_id=run_id,
        stage_name="source_artifacts",
        config_hash=config_hash,
        artifacts=artifacts,
        hf_repo_id=config.results_hf_repo,
    ):
        return

    helpers = _get_hf_helpers()
    source_root = out_path / "source_oct_run"

    print("  Downloading source OCT artifacts from Hugging Face")
    for artifact in artifacts:
        rel = artifact["path"].relative_to(source_root)
        remote_path = f"{source_run_dir}/{rel.as_posix()}"
        if artifact["kind"] == "file":
            with tempfile.TemporaryDirectory() as tmpdir:
                downloaded = helpers["hf_hub_download"](
                    repo_id=config.source_hf_repo,
                    repo_type="dataset",
                    filename=remote_path,
                    local_dir=tmpdir,
                    token=_hf_token(),
                )
                _copy_downloaded_artifact(Path(downloaded), artifact["path"], "file")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                helpers["snapshot_download"](
                    repo_id=config.source_hf_repo,
                    repo_type="dataset",
                    allow_patterns=[remote_path, f"{remote_path}/**"],
                    local_dir=tmpdir,
                    token=_hf_token(),
                )
                downloaded = Path(tmpdir) / remote_path
                _copy_downloaded_artifact(Path(downloaded), artifact["path"], "dir")

    _publish_stage(
        out_path=out_path,
        run_id=run_id,
        stage_name="source_artifacts",
        config_hash=config_hash,
        artifacts=artifacts,
        hf_repo_id=(
            None
            if config.results_hf_repo == config.source_hf_repo
            else config.results_hf_repo
        ),
    )


def _prompt_subset_path(out_path: Path) -> Path:
    return out_path / "data" / "prompt_subset.jsonl"


def _prepare_prompt_subset(config: ExperimentConfig, out_path: Path, run_id: str, config_hash: str, rerun: bool) -> Path:
    prompt_path = _prompt_subset_path(out_path)
    artifacts = [{"path": prompt_path, "kind": "file"}]

    if not rerun and _ensure_stage_available(
        out_path=out_path,
        run_id=run_id,
        stage_name="prompt_subset",
        config_hash=config_hash,
        artifacts=artifacts,
        hf_repo_id=config.results_hf_repo,
    ):
        return prompt_path

    dataset_cfg = DatasetConfig(
        source=config.prompt_source.source,
        name=config.prompt_source.name,
        path=config.prompt_source.path,
        split=config.prompt_source.split,
        max_samples=config.prompt_source.max_samples,
        seed=config.prompt_source.seed,
    )
    dataset = load_dataset_from_config(dataset_cfg)
    dataset = format_for_inference(dataset, question_column=config.prompt_source.question_column)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(dataset.to_list()):
        rows.append(
            {
                "id": row.get("id", idx),
                "prompt_index": idx,
                "question": str(row.get("question", "")),
            }
        )
    _write_jsonl(prompt_path, rows)

    _publish_stage(
        out_path=out_path,
        run_id=run_id,
        stage_name="prompt_subset",
        config_hash=config_hash,
        artifacts=artifacts,
        hf_repo_id=config.results_hf_repo,
    )
    return prompt_path


def _adapter_dir(out_path: Path, constitution: str, kind: str) -> Path:
    return out_path / "source_oct_run" / "lora" / f"{constitution}-{kind}"


def _build_model_specs(base_model_ref: str, out_path: Path, constitution: str) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for spec_name, label, adapter_kind in _MODEL_VARIANTS:
        if adapter_kind is None:
            specs.append(ModelSpec(name=spec_name, base_model=base_model_ref))
            continue
        adapter_path = _adapter_dir(out_path, constitution, adapter_kind)
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"Adapter not found for {label}: {adapter_path}")
        specs.append(
            ModelSpec(
                name=spec_name,
                base_model=base_model_ref,
                adapters=[AdapterConfig(path=f"local://{adapter_path}", scale=1.0)],
                scale=1.0,
            )
        )
    return specs


def _build_suite_config(
    *,
    config: ExperimentConfig,
    out_path: Path,
    constitution: str,
    base_model_ref: str,
    prompt_path: Path,
    skip_completed: bool,
) -> SuiteConfig:
    eval_spec = InspectCustomEvalSpec(
        name=_DEFAULT_EVAL_NAME,
        dataset=DatasetConfig(
            source="local",
            path=str(prompt_path),
            split="train",
            max_samples=config.prompt_source.max_samples,
            seed=None,
        ),
        input_builder="src_dev.evals.examples:question_input_builder",
        evaluations=["conscientiousness_v2"],
        scorer_builder="src_dev.evals.scorer_builders:persona_multi_score_scorer",
        judge=config.judge,
        generation=config.generation,
        metrics_key="persona_metrics",
    )
    return SuiteConfig(
        models=_build_model_specs(base_model_ref=base_model_ref, out_path=out_path, constitution=constitution),
        evals=[eval_spec],
        output_root=out_path / "evals",
        run_name=_DEFAULT_SUITE_RUN_NAME,
        skip_completed=skip_completed,
        temperature=config.generation.temperature if config.generation.do_sample else 0.0,
        batch_size=config.generation.batch_size,
        metadata={
            "source_hf_repo": config.source_hf_repo,
            "results_hf_repo": config.results_hf_repo,
            "constitution": constitution,
            "prompt_source": config.prompt_source.model_dump(),
            "judge": config.judge.model_dump(),
            "generation": config.generation.model_dump(),
        },
        hf_log_dir=config.hf_log_dir,
    )


def _suite_artifacts(out_path: Path) -> list[dict[str, Any]]:
    return [{"path": out_path / "evals" / _DEFAULT_SUITE_RUN_NAME, "kind": "dir"}]


def _is_complete_eval_suite(suite_dir: Path) -> bool:
    """Return whether the cached suite directory contains complete successful runs."""
    if not suite_dir.is_dir():
        return False

    for spec_name, _, _ in _MODEL_VARIANTS:
        run_dir = suite_dir / spec_name / _DEFAULT_EVAL_NAME
        run_info_path = run_dir / "run_info.json"
        if not run_info_path.is_file():
            return False

        try:
            run_info = _read_json(run_info_path)
        except Exception:
            return False

        if run_info.get("status") != "ok":
            return False

        native = run_info.get("native", {})
        inspect_log_path = native.get("inspect_log_path")
        if isinstance(inspect_log_path, str) and inspect_log_path:
            if Path(inspect_log_path).is_file():
                continue

        log_dir = run_dir / "native" / "inspect_logs"
        if not log_dir.is_dir():
            return False
        if not any(path.is_file() for path in log_dir.rglob("*")):
            return False

    return True


def _run_suite(
    *,
    config: ExperimentConfig,
    out_path: Path,
    run_id: str,
    config_hash: str,
    constitution: str,
    base_model_ref: str,
    prompt_path: Path,
    rerun: bool,
) -> Path:
    artifacts = _suite_artifacts(out_path)
    suite_dir = artifacts[0]["path"]
    if not rerun and _ensure_stage_available(
        out_path=out_path,
        run_id=run_id,
        stage_name="eval_suite",
        config_hash=config_hash,
        artifacts=artifacts,
        hf_repo_id=config.results_hf_repo,
    ):
        if _is_complete_eval_suite(suite_dir):
            return suite_dir
        print("  Cached eval_suite artifacts are incomplete; rerunning eval suite")

    suite_config = _build_suite_config(
        config=config,
        out_path=out_path,
        constitution=constitution,
        base_model_ref=base_model_ref,
        prompt_path=prompt_path,
        skip_completed=not rerun,
    )
    result = run_eval_suite(suite_config)
    suite_dir = result.output_root

    if not _is_complete_eval_suite(suite_dir):
        raise RuntimeError(
            f"Eval suite finished but did not produce complete outputs under {suite_dir}"
        )

    _publish_stage(
        out_path=out_path,
        run_id=run_id,
        stage_name="eval_suite",
        config_hash=config_hash,
        artifacts=artifacts,
        hf_repo_id=config.results_hf_repo,
    )
    return suite_dir


def _find_eval_log_path(run_dir: Path) -> Path:
    log_dir = run_dir / "native" / "inspect_logs"
    candidates = sorted(path for path in log_dir.rglob("*") if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"No Inspect log found under {log_dir}")
    return candidates[0]


def _first_sample_score_payload(sample: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    scores = getattr(sample, "scores", None) or {}
    if not scores:
        return {}, {}
    first_score = next(iter(scores.values()))
    value = getattr(first_score, "value", {}) or {}
    metadata = getattr(first_score, "metadata", {}) or {}
    return value if isinstance(value, dict) else {}, metadata if isinstance(metadata, dict) else {}


def _sample_rows_from_log(log: EvalLog, model_spec_name: str, model_label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    samples = getattr(log, "samples", None) or []
    for sample in samples:
        value_payload, metadata_payload = _first_sample_score_payload(sample)
        persona_metrics = metadata_payload.get("persona_metrics", {})
        if not isinstance(persona_metrics, dict):
            persona_metrics = {}
        score = persona_metrics.get("conscientiousness_v2.score")
        reasoning = persona_metrics.get("conscientiousness_v2.reasoning")
        if score is None:
            score = value_payload.get("conscientiousness_v2.score")

        response = ""
        output = getattr(sample, "output", None)
        if output is not None:
            response = getattr(output, "completion", "") or ""

        metadata = getattr(sample, "metadata", {}) or {}
        question = metadata.get("question")
        if question is None:
            question = getattr(sample, "input", "")

        rows.append(
            {
                "sample_id": getattr(sample, "id", None),
                "model_spec": model_spec_name,
                "model_label": model_label,
                "question": question,
                "response": response,
                "conscientiousness_v2_score": score,
                "conscientiousness_v2_reasoning": reasoning,
            }
        )
    return rows


def _analysis_artifacts(out_path: Path) -> list[dict[str, Any]]:
    return [
        {"path": out_path / "analysis" / "scored_rollouts.jsonl", "kind": "file"},
        {"path": out_path / "analysis" / "conscientiousness_by_model.csv", "kind": "file"},
        {"path": out_path / "analysis" / "conscientiousness_by_model_summary.csv", "kind": "file"},
        {"path": out_path / "figures" / "conscientiousness_bar_chart.png", "kind": "file"},
    ]


def _plot_bar_chart(summary_df: pd.DataFrame, out_path: Path, title: str) -> Path:
    figures_dir = out_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "conscientiousness_bar_chart.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [_MODEL_COLORS[label] for label in summary_df["model_label"]]
    ax.bar(summary_df["model_label"], summary_df["mean_conscientiousness"], color=colors)
    ax.set_ylabel("Mean conscientiousness_v2 score")
    ax.set_xlabel("Model variant")
    ax.set_title(title)
    ax.set_ylim(-4.0, 4.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def _export_analysis(
    *,
    config: ExperimentConfig,
    out_path: Path,
    run_id: str,
    config_hash: str,
    suite_dir: Path,
    constitution: str,
    rerun: bool,
) -> tuple[Path, Path, Path]:
    artifacts = _analysis_artifacts(out_path)
    if not rerun and _ensure_stage_available(
        out_path=out_path,
        run_id=run_id,
        stage_name="analysis",
        config_hash=config_hash,
        artifacts=artifacts,
        hf_repo_id=config.results_hf_repo,
    ):
        return artifacts[0]["path"], artifacts[2]["path"], artifacts[3]["path"]

    scored_rows: list[dict[str, Any]] = []
    label_map = {spec_name: label for spec_name, label, _ in _MODEL_VARIANTS}
    for spec_name, label, _ in _MODEL_VARIANTS:
        run_dir = suite_dir / spec_name / _DEFAULT_EVAL_NAME
        log_path = _find_eval_log_path(run_dir)
        log = read_eval_log(log_path)
        scored_rows.extend(_sample_rows_from_log(log, model_spec_name=spec_name, model_label=label))

    scored_path = out_path / "analysis" / "scored_rollouts.jsonl"
    _write_jsonl(scored_path, scored_rows)

    long_df = pd.DataFrame.from_records(scored_rows)
    long_csv_path = out_path / "analysis" / "conscientiousness_by_model.csv"
    long_df.to_csv(long_csv_path, index=False)

    summary_df = (
        long_df.groupby(["model_spec", "model_label"], as_index=False)["conscientiousness_v2_score"]
        .mean()
        .rename(columns={"conscientiousness_v2_score": "mean_conscientiousness"})
    )
    summary_df["model_label"] = pd.Categorical(
        summary_df["model_label"],
        categories=[label_map[spec_name] for spec_name, _, _ in _MODEL_VARIANTS],
        ordered=True,
    )
    summary_df = summary_df.sort_values("model_label").reset_index(drop=True)

    summary_csv_path = out_path / "analysis" / "conscientiousness_by_model_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    figure_path = _plot_bar_chart(
        summary_df=summary_df,
        out_path=out_path,
        title=(
            f"Mean conscientiousness_v2: {constitution} "
            f"(n={config.prompt_source.max_samples}, seed={config.prompt_source.seed})"
        ),
    )

    _publish_stage(
        out_path=out_path,
        run_id=run_id,
        stage_name="analysis",
        config_hash=config_hash,
        artifacts=artifacts,
        hf_repo_id=config.results_hf_repo,
    )
    return scored_path, summary_csv_path, figure_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate OCT base/DPO/SFT/persona variants with conscientiousness_v2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-hf-repo", required=True, help="HF dataset repo containing the OCT run artifacts")
    parser.add_argument(
        "--source-run-dir",
        default=None,
        help="Top-level OCT run directory inside the source dataset repo; auto-detected when omitted",
    )
    parser.add_argument(
        "--results-hf-repo",
        default=None,
        help="HF dataset repo where this evaluation run is mirrored; defaults to --source-hf-repo",
    )
    parser.add_argument("--out-dir", default=None, help="Optional explicit local output directory")
    parser.add_argument("--constitution", default=None, help="Override constitution from the source OCT run config")
    parser.add_argument("--base-model-ref", default=None, help="Override base model repo/path")
    parser.add_argument("--model-cache-root", default=str(_DEFAULT_MODEL_CACHE_ROOT), help="Local base-model cache root")

    parser.add_argument("--prompt-source", default="huggingface", choices=["huggingface", "local"], help="Prompt dataset source")
    parser.add_argument("--prompt-dataset-name", default=_DEFAULT_PROMPT_DATASET, help="HF prompt dataset name")
    parser.add_argument("--prompt-split", default=_DEFAULT_PROMPT_SPLIT, help="Prompt dataset split")
    parser.add_argument("--prompt-path", default=None, help="Local JSON/JSONL prompt dataset path")
    parser.add_argument("--question-column", default="question", help="Column to treat as the question text")
    parser.add_argument("--max-samples", type=int, default=100, help="Number of prompts to sample")
    parser.add_argument("--sample-seed", type=int, default=223457, help="Deterministic prompt sampling seed")

    parser.add_argument("--judge-provider", default="openai", choices=["openai", "openrouter", "anthropic"])
    parser.add_argument("--judge-model", default="gpt-5-nano-2025-08-07")
    parser.add_argument("--judge-api-key-env", default=None)
    parser.add_argument("--judge-max-tokens", type=int, default=10000)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-concurrent", type=int, default=10)
    parser.add_argument("--judge-timeout", type=int, default=60)

    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling for generation")
    parser.add_argument("--hf-log-dir", default=_DEFAULT_HF_LOG_REPO, help="Optional HF log root for Inspect logs; set '' to disable")
    parser.add_argument("--rerun", action="store_true", help="Ignore cached eval/analysis stages and rerun them locally")
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        source_hf_repo=args.source_hf_repo,
        source_run_dir=args.source_run_dir,
        results_hf_repo=args.results_hf_repo or args.source_hf_repo,
        constitution=args.constitution,
        base_model_ref=args.base_model_ref,
        model_cache_root=args.model_cache_root,
        prompt_source=PromptSourceConfig(
            source=args.prompt_source,
            name=args.prompt_dataset_name,
            split=args.prompt_split,
            path=args.prompt_path,
            question_column=args.question_column,
            max_samples=args.max_samples,
            seed=args.sample_seed,
        ),
        judge=JudgeLLMConfig(
            provider=args.judge_provider,
            model=args.judge_model,
            api_key_env=args.judge_api_key_env,
            max_tokens=args.judge_max_tokens,
            temperature=args.judge_temperature,
            max_concurrent=args.judge_max_concurrent,
            timeout=args.judge_timeout,
        ),
        generation=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=0.0 if args.greedy else args.temperature,
            top_p=1.0 if args.greedy else args.top_p,
            do_sample=not args.greedy,
            batch_size=args.batch_size,
            num_responses_per_prompt=1,
        ),
        hf_log_dir=args.hf_log_dir or None,
    )


def main() -> None:
    args = _parse_args()
    config = _config_from_args(args)

    source_run_dir = _resolve_source_run_dir(config)
    source_run_config_payload = _load_source_run_config(config, source_run_dir)
    constitution = config.constitution or _default_constitution(source_run_config_payload)
    base_model_ref = config.base_model_ref or _default_base_model_ref(source_run_config_payload)

    run_id = _make_run_id(
        config=config,
        source_run_dir=source_run_dir,
        constitution=constitution,
        base_model_ref=base_model_ref,
    )
    out_path = Path(args.out_dir).resolve() if args.out_dir else (_DEFAULT_OUTPUT_ROOT / run_id).resolve()

    config_payload = {
        **config.model_dump(),
        "source_run_dir": source_run_dir,
        "constitution": constitution,
        "base_model_ref": base_model_ref,
    }
    config_hash = _config_hash(config_payload)

    print("OCT conscientiousness eval")
    print(f"  source repo:      {config.source_hf_repo}")
    print(f"  source run dir:   {source_run_dir}")
    print(f"  results repo:     {config.results_hf_repo}")
    print(f"  local out dir:    {out_path}")
    print(f"  run id:           {run_id}")

    _ensure_run_config(out_path=out_path, run_id=run_id, config_hash=config_hash, config_payload=config_payload)

    _hydrate_source_artifacts(
        config=config,
        out_path=out_path,
        run_id=run_id,
        config_hash=config_hash,
        source_run_dir=source_run_dir,
        constitution=constitution,
        rerun=False,
    )

    model_cache_root = Path(config.model_cache_root).expanduser().resolve()
    base_model_local_ref = _ensure_base_model(base_model_ref=base_model_ref, model_cache_root=model_cache_root)
    prompt_path = _prepare_prompt_subset(
        config=config,
        out_path=out_path,
        run_id=run_id,
        config_hash=config_hash,
        rerun=False,
    )
    suite_dir = _run_suite(
        config=config,
        out_path=out_path,
        run_id=run_id,
        config_hash=config_hash,
        constitution=constitution,
        base_model_ref=base_model_local_ref,
        prompt_path=prompt_path,
        rerun=args.rerun,
    )
    scored_path, summary_csv_path, figure_path = _export_analysis(
        config=config,
        out_path=out_path,
        run_id=run_id,
        config_hash=config_hash,
        suite_dir=suite_dir,
        constitution=constitution,
        rerun=args.rerun,
    )

    print(f"Prompt subset:      {prompt_path}")
    print(f"Suite output:       {suite_dir}")
    print(f"Scored rollouts:    {scored_path}")
    print(f"Summary CSV:        {summary_csv_path}")
    print(f"Bar chart:          {figure_path}")


if __name__ == "__main__":
    main()
