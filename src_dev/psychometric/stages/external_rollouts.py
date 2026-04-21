"""Stage 1 (external variant) — ingest pre-existing multi-turn rollouts.

Mirrors :func:`src_dev.psychometric.stages.rollouts.run_stage_rollouts` but
produces the rollout run directory from an external HF dataset adapter
rather than by generating conversations. The output layout is byte-for-byte
compatible with the generated path: same manifest, same
``canonical_samples.jsonl`` + ``exports/conversation_training.jsonl``,
same HF upload location. Stage 2+ reads it the same way.

Cache semantics:

* local cache → HF cache → adapter ingest (same order as the generation path).
* HF path: ``runs/<rollout_run_id>`` on the shared monorepo.

See ``docs/external_rollouts.md`` (or the module-level docstring of the
orchestrator) for the semantic contract — in particular that
``num_conversation_turns`` is reused as a ``min_assistant_turns`` filter
and that ``--retry-terminal-samples`` is not meaningful here.
"""

from __future__ import annotations

import logging

from src_dev.datasets import (
    export_dataset,
    ingest_source_dataset,
    materialize_canonical_samples,
)
from src_dev.datasets.external_sources import deterministic_sample, get_adapter
from src_dev.psychometric.config import (
    ExternalRolloutsStageConfig,
    RolloutStageResult,
)
from src_dev.unsupervised_runs.io import hydrate_dataset_subtree
from src_dev.utils.hf_hub import (
    check_exists_in_dataset_repo,
    login_from_env,
    upload_folder_to_dataset_repo,
)

logger = logging.getLogger(__name__)


def run_stage_ingest_external_rollouts(
    cfg: ExternalRolloutsStageConfig,
) -> RolloutStageResult:
    """Ingest an external dataset into the canonical rollout layout."""
    run_dir = cfg.ctx.rollout_dir
    run_id = cfg.ctx.rollout_run_id
    hf_repo_id = cfg.ctx.hf_repo_id

    try:
        login_from_env()
    except RuntimeError:
        logger.warning("HF_TOKEN not set — HF caching disabled.")
    hf_path = f"runs/{run_id}"

    rollout_export = run_dir / "exports" / "conversation_training.jsonl"

    # Local cache hit
    if rollout_export.exists():
        if not check_exists_in_dataset_repo(
            repo_id=hf_repo_id,
            path_in_repo=hf_path + "/exports/conversation_training.jsonl",
        ):
            print("[Stage 1] Local external rollouts found but not on HF — uploading now")
            try:
                upload_folder_to_dataset_repo(
                    local_dir=run_dir,
                    repo_id=hf_repo_id,
                    path_in_repo=hf_path,
                    commit_message=f"External rollouts: {run_id}",
                )
                print(f"[Stage 1] Uploaded to HF: {hf_path}")
            except Exception as e:
                logger.warning("Failed to upload external rollouts to HF: %s", e)
        print(f"[Stage 1] External rollouts already exist locally: {run_dir}")
        return RolloutStageResult(rollout_dir=run_dir)

    # HF cache hit — hydrate and short-circuit ingestion.
    if check_exists_in_dataset_repo(repo_id=hf_repo_id, path_in_repo=hf_path):
        print(f"[Stage 1] Hydrating external rollouts from HF: {run_id}")
        hydrate_dataset_subtree(
            repo_id=hf_repo_id, path_in_repo=hf_path, local_dir=run_dir
        )
        if rollout_export.exists():
            print(f"[Stage 1] Hydrated external rollouts from HF: {run_dir}")
            return RolloutStageResult(rollout_dir=run_dir, hydrated_from_hf=True)
        print("[Stage 1] HF hydration incomplete, re-ingesting...")

    # Cache miss — stream + sample + ingest.
    adapter = get_adapter(cfg.source)
    print(
        f"[Stage 1] Ingesting external source {cfg.source!r} "
        f"(default model: {adapter.default_assistant_model})"
    )
    # Thread min_assistant_turns through the adapter's filter_config so
    # it's enforced during the stream (guaranteeing N post-filter samples).
    filter_config = dict(cfg.filter_config)
    if cfg.min_assistant_turns and "min_assistant_turns" not in filter_config:
        filter_config["min_assistant_turns"] = cfg.min_assistant_turns

    print(
        f"[Stage 1] Reservoir-sampling n={cfg.max_samples} with seed={cfg.seed} "
        f"(max_scan={cfg.max_scan}, filter={filter_config})"
    )
    rows = deterministic_sample(
        adapter.iter_raw(filter_config),
        n=cfg.max_samples,
        seed=cfg.seed,
        max_scan=cfg.max_scan,
    )
    print(f"[Stage 1] Sampled {len(rows)} rows from {cfg.source!r}")

    # Shape into ingest-row form: the canonical ingester reads a ``messages``
    # list (role+content dicts). The adapter already canonicalises messages;
    # we copy source_info per-row so it survives to SampleRecord.source_info.
    ingest_rows = []
    for r in rows:
        ingest_rows.append({
            "messages": r["messages"],
            # Extra fields are ignored by the ingester but kept for provenance.
            "source_info_row": r.get("source_info", {}),
            "adapter_assistant_model": r.get("assistant_model"),
        })

    # Pull the actual source HF repo from the first row's source_info if
    # the adapter supplies one; fall back to the adapter name.
    source_hf_repo = (
        rows[0].get("source_info", {}).get("source_hf_repo")
        if rows else None
    ) or cfg.source

    source_info = {
        "kind": "external",
        "source": cfg.source,
        "source_hf_repo": source_hf_repo,
        "source_adapter_notes": adapter.notes,
        "assistant_model": cfg.assistant_model,
        "assistant_provider": cfg.assistant_provider,
        "max_samples": cfg.max_samples,
        "seed": cfg.seed,
        "max_scan": cfg.max_scan,
        "filter_config": filter_config,
    }

    ingest_source_dataset(
        dataset=ingest_rows,
        source_info=source_info,
        system_prompt=None,
        run_dir=run_dir,
    )
    materialize_canonical_samples(run_dir)
    # Stage 2 reads the canonical_samples directly, but we still emit the
    # `conversation_training` export so the local-cache check (which keys
    # on exports/conversation_training.jsonl for parity with the generation
    # path) sees a hit on the next run.
    export_dataset(run_dir, profile="conversation_training")

    print(f"[Stage 1] Ingested {len(rows)} external rollouts → {run_dir}")

    try:
        upload_folder_to_dataset_repo(
            local_dir=run_dir,
            repo_id=hf_repo_id,
            path_in_repo=hf_path,
            commit_message=f"External rollouts: {run_id}",
        )
        print(f"[Stage 1] Uploaded to HF: {hf_path}")
    except Exception as e:
        logger.warning("Failed to upload external rollouts to HF: %s", e)

    return RolloutStageResult(
        rollout_dir=run_dir,
        num_samples=len(rows),
        generated=True,
    )
