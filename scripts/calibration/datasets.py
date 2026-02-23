"""Canonicalized dataset adapters for calibration workflows."""

from __future__ import annotations

import hashlib
import math
from typing import Any

from datasets import Dataset
from pydantic import BaseModel, Field

from scripts.calibration.config import CalibrationConfig


class CalibrationDatasetProfile(BaseModel):
    """Schema aliases and conventions for known calibration datasets."""

    name: str
    description: str
    response_column_aliases: list[str] = Field(default_factory=list)
    label_column_aliases: list[str] = Field(default_factory=list)
    question_column_aliases: list[str] = Field(default_factory=list)
    subject_id_column_aliases: list[str] = Field(default_factory=list)
    unit_id_column_aliases: list[str] = Field(default_factory=list)
    split_column_aliases: list[str] = Field(default_factory=list)


DATASET_PROFILES: dict[str, CalibrationDatasetProfile] = {
    "essays_neuroticism_v1": CalibrationDatasetProfile(
        name="essays_neuroticism_v1",
        description=(
            "Essay-style free text with questionnaire-derived neuroticism labels. "
            "Maps common column aliases and enforces deterministic split protocol."
        ),
        response_column_aliases=[
            "response",
            "essay",
            "essay_text",
            "text",
            "body",
            "content",
        ],
        label_column_aliases=[
            "neuroticism",
            "label_neuroticism",
            "bfi_neuroticism",
            "ocean_neuroticism",
            "neuroticism_score",
            "N",
            "label",
        ],
        question_column_aliases=["question", "prompt", "instruction"],
        subject_id_column_aliases=["subject_id", "author_id", "user_id", "id"],
        unit_id_column_aliases=["unit_id", "sample_id", "essay_id", "id"],
        split_column_aliases=["split", "set", "fold", "partition"],
    ),
}


def list_dataset_profiles() -> list[str]:
    """List available calibration dataset profiles."""
    return sorted(DATASET_PROFILES.keys())


def get_dataset_profile(name: str) -> CalibrationDatasetProfile:
    """Get a calibration dataset profile by name."""
    if name not in DATASET_PROFILES:
        available = ", ".join(sorted(DATASET_PROFILES))
        raise KeyError(f"Unknown dataset profile '{name}'. Available profiles: {available}")
    return DATASET_PROFILES[name].model_copy(deep=True)


def _resolve_column(
    columns: set[str],
    preferred: str | None,
    aliases: list[str],
    *,
    required: bool,
    field_name: str,
) -> str | None:
    if preferred and preferred in columns:
        return preferred
    for alias in aliases:
        if alias in columns:
            return alias
    if required:
        wanted = [preferred] if preferred else []
        wanted.extend(aliases)
        raise ValueError(
            f"Could not resolve required column for {field_name}. "
            f"Tried: {wanted}. Available columns: {sorted(columns)}"
        )
    return None


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _stable_unit_id(*, row_idx: int, response: str, question: str | None, subject_id: str | None) -> str:
    payload = f"{subject_id or ''}\n{question or ''}\n{response}\n{row_idx}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"unit_{digest[:20]}"


def _canonicalize_split(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"train", "training"}:
        return "train"
    if text in {"dev", "valid", "validation", "val"}:
        return "dev"
    if text in {"test", "eval", "evaluation", "holdout"}:
        return "test"
    return None


def _split_from_hash(*, split_seed: int, unit_id: str, train_fraction: float, dev_fraction: float) -> str:
    digest = hashlib.sha256(f"{split_seed}:{unit_id}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / float(0xFFFFFFFF)
    if value < train_fraction:
        return "train"
    if value < train_fraction + dev_fraction:
        return "dev"
    return "test"


def apply_calibration_dataset_protocol(
    dataset: Dataset,
    config: CalibrationConfig,
    *,
    warnings: list[str],
) -> tuple[Dataset, dict[str, Any]]:
    """Apply known schema mapping, deterministic split, and label normalization."""
    profile_name = config.dataset.dataset_profile
    profile = get_dataset_profile(profile_name) if profile_name else None

    columns = set(dataset.column_names)
    split_cfg = config.dataset.split
    norm_cfg = config.dataset.normalization
    trait = config.trait

    response_aliases = profile.response_column_aliases if profile else []
    label_aliases = (
        profile.label_column_aliases + trait.label_column_aliases if profile else trait.label_column_aliases
    )
    question_aliases = profile.question_column_aliases if profile else []
    subject_aliases = profile.subject_id_column_aliases if profile else []
    unit_aliases = profile.unit_id_column_aliases if profile else []
    split_aliases = profile.split_column_aliases if profile else []

    source_response_col = _resolve_column(
        columns,
        config.dataset.response_column,
        response_aliases,
        required=True,
        field_name="response",
    )
    source_label_col = _resolve_column(
        columns,
        config.dataset.label_column,
        label_aliases,
        required=True,
        field_name="label",
    )
    source_question_col = _resolve_column(
        columns,
        config.dataset.question_column,
        question_aliases,
        required=False,
        field_name="question",
    )
    source_subject_col = _resolve_column(
        columns,
        config.dataset.subject_id_column,
        subject_aliases,
        required=False,
        field_name="subject_id",
    )
    source_unit_col = _resolve_column(
        columns,
        config.dataset.unit_id_column,
        unit_aliases,
        required=False,
        field_name="unit_id",
    )
    source_split_col = _resolve_column(
        columns,
        split_cfg.split_column,
        split_aliases,
        required=False,
        field_name="split",
    )

    selected_columns: dict[str, str | None] = {
        "response": source_response_col,
        "label": source_label_col,
        "question": source_question_col,
        "subject_id": source_subject_col,
        "unit_id": source_unit_col,
        "split": source_split_col,
    }
    for logical_name, column_name in selected_columns.items():
        if logical_name == "split":
            continue
        preferred = getattr(config.dataset, f"{logical_name}_column", None)
        if preferred and column_name and preferred != column_name:
            warnings.append(
                f"dataset.{logical_name}_column='{preferred}' not found; using resolved column "
                f"'{column_name}' from profile/aliases."
            )

    split_unknown_values: set[str] = set()
    split_counts = {"train": 0, "dev": 0, "test": 0}
    out_rows: list[dict[str, Any]] = []
    unit_out_col = config.dataset.unit_id_column or "sample_id"

    records = dataset.to_list()
    for idx, record in enumerate(records):
        response_raw = record.get(source_response_col)
        response = response_raw if isinstance(response_raw, str) else str(response_raw or "")

        question: str | None = None
        if source_question_col is not None:
            question_raw = record.get(source_question_col)
            if question_raw is not None:
                question = question_raw if isinstance(question_raw, str) else str(question_raw)

        subject_id: str | None = None
        if source_subject_col is not None:
            subject_raw = record.get(source_subject_col)
            if subject_raw is not None:
                subject_id = str(subject_raw)

        unit_id: str
        if source_unit_col is not None and record.get(source_unit_col) is not None:
            unit_id = str(record[source_unit_col])
        else:
            unit_id = _stable_unit_id(
                row_idx=idx,
                response=response,
                question=question,
                subject_id=subject_id,
            )

        row_split = None
        if source_split_col is not None:
            row_split = _canonicalize_split(record.get(source_split_col))
            raw_value = record.get(source_split_col)
            if row_split is None and raw_value is not None:
                split_unknown_values.add(str(raw_value))
        if row_split is None:
            row_split = _split_from_hash(
                split_seed=split_cfg.split_seed,
                unit_id=unit_id,
                train_fraction=split_cfg.train_fraction,
                dev_fraction=split_cfg.dev_fraction,
            )

        split_counts[row_split] += 1
        if split_cfg.eval_split != "all" and row_split != split_cfg.eval_split:
            continue

        label = _float_or_nan(record.get(source_label_col))
        if norm_cfg.mode == "linear_to_trait_range" and math.isfinite(label):
            source_min = float(norm_cfg.source_min) if norm_cfg.source_min is not None else 0.0
            source_max = float(norm_cfg.source_max) if norm_cfg.source_max is not None else 1.0
            if source_max > source_min:
                label = (
                    ((label - source_min) / (source_max - source_min))
                    * (trait.raw_max - trait.raw_min)
                    + trait.raw_min
                )
                if norm_cfg.clip_to_trait_range:
                    label = min(trait.raw_max, max(trait.raw_min, label))

        out_row: dict[str, Any] = {
            config.dataset.response_column: response,
            config.dataset.label_column: label,
            unit_out_col: unit_id,
            "_calibration_split": row_split,
            "_source_row_index": idx,
        }
        if config.dataset.question_column:
            out_row[config.dataset.question_column] = question
        if config.dataset.subject_id_column:
            out_row[config.dataset.subject_id_column] = subject_id
        out_rows.append(out_row)

    if split_unknown_values:
        warnings.append(
            "Unrecognized split values in source split column were routed by deterministic hashing: "
            + ", ".join(sorted(split_unknown_values))
        )

    if not out_rows:
        raise ValueError(
            "Dataset protocol filtering produced zero rows. "
            "Check split settings and schema mappings."
        )

    metadata = {
        "profile": profile.name if profile else None,
        "description": profile.description if profile else None,
        "input_rows": len(records),
        "output_rows": len(out_rows),
        "selected_columns": selected_columns,
        "eval_split": split_cfg.eval_split,
        "split_column_used": source_split_col,
        "split_counts_all_rows": split_counts,
        "split_seed": split_cfg.split_seed,
        "split_fractions": {
            "train": split_cfg.train_fraction,
            "dev": split_cfg.dev_fraction,
            "test": 1.0 - split_cfg.train_fraction - split_cfg.dev_fraction,
        },
        "normalization": {
            "mode": norm_cfg.mode,
            "source_min": norm_cfg.source_min,
            "source_max": norm_cfg.source_max,
            "clip_to_trait_range": norm_cfg.clip_to_trait_range,
            "target_min": trait.raw_min,
            "target_max": trait.raw_max,
        },
    }
    return Dataset.from_list(out_rows), metadata
