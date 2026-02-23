"""Configuration models for calibration of persona-metric judges."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

    class CalibrationSplitConfig(BaseModel):
        """Deterministic split protocol for calibration datasets."""

        eval_split: Literal["all", "train", "dev", "test"] = "all"
        split_column: str | None = None
        split_seed: int = 2026
        train_fraction: float = 0.7
        dev_fraction: float = 0.1

        @model_validator(mode="after")
        def _validate_fractions(self) -> "CalibrationDatasetConfig.CalibrationSplitConfig":
            if not (0.0 < self.train_fraction < 1.0):
                raise ValueError("train_fraction must be in (0, 1)")
            if not (0.0 <= self.dev_fraction < 1.0):
                raise ValueError("dev_fraction must be in [0, 1)")
            if self.train_fraction + self.dev_fraction >= 1.0:
                raise ValueError("train_fraction + dev_fraction must be < 1")
            return self

    class LabelNormalizationConfig(BaseModel):
        """Label normalization contract before calibration analysis."""

        mode: Literal["none", "linear_to_trait_range"] = "none"
        source_min: float | None = None
        source_max: float | None = None
        clip_to_trait_range: bool = True

        @model_validator(mode="after")
        def _validate_linear_range(
            self,
        ) -> "CalibrationDatasetConfig.LabelNormalizationConfig":
            if self.mode == "linear_to_trait_range":
                if self.source_min is None or self.source_max is None:
                    raise ValueError(
                        "source_min and source_max are required for "
                        "normalization.mode='linear_to_trait_range'"
                    )
                if not self.source_max > self.source_min:
                    raise ValueError("source_max must be greater than source_min")
            return self

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    response_column: str = "response"
    label_column: str = "label"
    question_column: str | None = "question"
    subject_id_column: str | None = None
    unit_id_column: str | None = None
    dataset_profile: str | None = None
    split: CalibrationSplitConfig = Field(default_factory=CalibrationSplitConfig)
    normalization: LabelNormalizationConfig = Field(default_factory=LabelNormalizationConfig)


class CalibrationJudgeConfig(BaseModel):
    """Judge/metric config for calibration runs."""

    metric_name: str = "neuroticism"
    judge: JudgeLLMConfig = Field(default_factory=JudgeLLMConfig)
    metric_params: dict[str, Any] = Field(default_factory=dict)


class ReliabilityRaterConfig(BaseModel):
    """Optional per-rater overrides for reliability analysis.

    If `judge` is set, it overrides the base judge config for that rater.
    If `metric_params` is set, it is merged over base metric params.
    """

    name: str | None = None
    judge: JudgeLLMConfig | None = None
    metric_params: dict[str, Any] = Field(default_factory=dict)


class ReliabilityConfig(BaseModel):
    """Reliability analysis settings."""

    num_runs: int = 7
    raters: list[ReliabilityRaterConfig] = Field(default_factory=list)
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
