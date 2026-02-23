"""Canonical dataset module."""

from scripts.datasets.core import (
    export_dataset,
    get_run_paths,
    ingest_source_dataset,
    init_run,
    load_manifest,
    load_sample_inputs,
    load_samples,
    materialize_canonical_samples,
    migrate_legacy_jsonl,
    record_stage_event,
    register_stage_fingerprint,
    resume_state,
    select_training_candidates,
    validate_run,
    write_edit_overlay,
    write_inference_result,
    write_metric_annotation,
)

__all__ = [
    "init_run",
    "get_run_paths",
    "load_manifest",
    "register_stage_fingerprint",
    "ingest_source_dataset",
    "load_sample_inputs",
    "load_samples",
    "record_stage_event",
    "write_inference_result",
    "write_edit_overlay",
    "write_metric_annotation",
    "materialize_canonical_samples",
    "resume_state",
    "select_training_candidates",
    "export_dataset",
    "migrate_legacy_jsonl",
    "validate_run",
]

