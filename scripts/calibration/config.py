"""Configuration models for calibration of persona-metric judges."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from scripts.common.config import DatasetConfig
from scripts.persona_metrics.config import JudgeLLMConfig


class TraitSpec(BaseModel):
    """Trait metadata used for calibration reporting.

    Attributes:
        trait_name: Human-readable trait name.
        metric_name: Persona metric registry key used to score text.
        raw_min: Minimum expected raw judge score.
        raw_max: Maximum expected raw judge score.
        label_semantics: Description of what the ground-truth label represents.
        label_column_aliases: Optional common label column aliases.
    """

    trait_name: str = "neuroticism"
    metric_name: str = "neuroticism"
    raw_min: float = -5.0
    raw_max: float = 5.0
    label_semantics: str = (
        "Questionnaire-derived trait label (author-level disposition), "
        "not text-level impression label."
    )
    label_column_aliases: list[str] = Field(
        default_factory=lambda: [
            "neuroticism",
            "label_neuroticism",
            "bfi_neuroticism",
            "ocean_neuroticism",
        ]
    )

    @property
    def score_key(self) -> str:
        """Metric output key for raw score values."""
        return f"{self.metric_name}.score"

    @property
    def reasoning_key(self) -> str:
        """Metric output key for judge reasoning values."""
        return f"{self.metric_name}.reasoning"


class CalibrationDatasetConfig(BaseModel):
    """Dataset and column mapping config for calibration."""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    response_column: str = "response"
    label_column: str = "label"
    question_column: str | None = "question"
    subject_id_column: str | None = None
    unit_id_column: str | None = None


class CalibrationJudgeConfig(BaseModel):
    """Judge/metric config for calibration runs."""

    metric_name: str = "neuroticism"
    judge: JudgeLLMConfig = Field(default_factory=JudgeLLMConfig)
    metric_params: dict[str, Any] = Field(default_factory=dict)


class ReliabilityConfig(BaseModel):
    """Reliability analysis settings."""

    num_runs: int = 7
    alpha_level: Literal["ordinal"] = "ordinal"
    bootstrap_samples: int = 1000
    min_units: int = 20
    random_seed: int = 13


class ValidityConfig(BaseModel):
    """Construct-validity analysis settings."""

    analysis_unit: Literal["auto", "text", "subject"] = "auto"
    bootstrap_samples: int = 1000
    random_seed: int = 17
    metrics: list[str] = Field(
        default_factory=lambda: [
            "pearson_r",
            "spearman_rho",
            "mae",
            "rmse",
            "slope",
            "intercept",
            "r2",
        ]
    )


class CalibrationConfig(BaseModel):
    """Top-level calibration configuration."""

    dataset: CalibrationDatasetConfig = Field(default_factory=CalibrationDatasetConfig)
    judge: CalibrationJudgeConfig = Field(default_factory=CalibrationJudgeConfig)
    trait: TraitSpec = Field(default_factory=TraitSpec)
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig)
    validity: ValidityConfig = Field(default_factory=ValidityConfig)

    run_name: str | None = None
    output_root: Path = Path("scratch/calibration")
    output_dir: Path | None = None


class CalibrationResult(BaseModel):
    """Calibration run result metadata."""

    class Config:
        arbitrary_types_allowed = True

    output_dir: Path
    run_name: str
    trait: TraitSpec
    metric_name: str
    score_key: str
    analysis_unit: str
    num_input_rows: int
    num_scored_units: int
    warnings: list[str] = Field(default_factory=list)

    reliability: dict[str, Any] = Field(default_factory=dict)
    validity: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Path] = Field(default_factory=dict)
