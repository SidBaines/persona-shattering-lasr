"""Psychometric questionnaire administration + factor-analysis experiment components.

This package contains the reusable machinery previously inlined in
``scripts_dev/unsupervised_embeddings/psychometric_rollout_fa.py``:

* ``questionnaire_io`` / ``response_parsing`` / ``item_prompts`` /
  ``response_encoding`` — questionnaire schema handling and per-item prompt +
  response plumbing (Likert, forced-choice, trait_mcq, vignette).
* ``questionnaire_inference`` — the async loop that administers the
  questionnaire to each persona (local vLLM or remote API providers).
* ``preprocessing`` — response-matrix cleanup before factor analysis
  (per-block variance filter, high-variance persona drop, residualization).
* ``trait_scoring`` / ``realism_judge`` — derived psychometric scores and a
  diagnostic rollout-realism judge.
* ``fa_plots`` / ``trait_aware_plots`` / ``factor_extremes_html`` — analysis
  plots and the interactive factor-extremes HTML report.
* ``labelling`` — LLM-based factor labelling (OpenRouter/Anthropic API or the
  Claude Code CLI transport), with psychometrics-specific prompts.
* ``metadata_enrichment`` — archetype / scenario lookups used by plotting and
  validation passes.
* ``stages/*`` — ``run_stage_*(cfg) -> StageResult`` entry points that
  orchestrate the above for each pipeline stage. Experiment scripts should
  build stage configs and call these.

The statistical factor-analysis library lives in
``src_dev.factor_analysis`` and is unchanged; the stage functions call it.
"""

from src_dev.psychometric.config import (
    FactorAnalysisStageConfig,
    LabelingStageConfig,
    QuestionnaireStageConfig,
    RealismJudgeStageConfig,
    RolloutsStageConfig,
    RunContext,
    TraitScoringStageConfig,
    ValidationStageConfig,
)

__all__ = [
    "FactorAnalysisStageConfig",
    "LabelingStageConfig",
    "QuestionnaireStageConfig",
    "RealismJudgeStageConfig",
    "RolloutsStageConfig",
    "RunContext",
    "TraitScoringStageConfig",
    "ValidationStageConfig",
]
