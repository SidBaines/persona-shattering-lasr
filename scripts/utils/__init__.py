"""Utility functions for scripts."""

from scripts.utils.git import assert_clean_and_pushed
from scripts.utils.hf_hub import login_from_env, upload_file_to_dataset_repo, upload_folder_to_model_repo
from scripts.utils.io import count_jsonl_rows, iter_jsonl_batches, read_jsonl, write_jsonl
from scripts.utils.lora_composition import (
    WeightedAdapter,
    delete_materialized_model_dir,
    load_and_scale_adapters,
    merge_weighted_adapters,
    parse_weighted_adapter,
    resolve_torch_dtype,
    split_adapter_reference,
)
from scripts.utils.logging import setup_logging

__all__ = [
    "assert_clean_and_pushed",
    "read_jsonl",
    "write_jsonl",
    "count_jsonl_rows",
    "iter_jsonl_batches",
    "setup_logging",
    "login_from_env",
    "upload_file_to_dataset_repo",
    "upload_folder_to_model_repo",
    "WeightedAdapter",
    "parse_weighted_adapter",
    "split_adapter_reference",
    "resolve_torch_dtype",
    "load_and_scale_adapters",
    "merge_weighted_adapters",
    "delete_materialized_model_dir",
]
