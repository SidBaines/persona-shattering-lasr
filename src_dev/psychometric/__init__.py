"""Psychometric questionnaire administration + factor-analysis experiment components.

This package contains the reusable machinery previously inlined in
``scripts_dev/unsupervised_embeddings/psychometric_rollout_fa.py``.

Main entry points for subset scripts (stages — see ``stages/`` subpackage):
    * ``run_stage_rollouts`` — Stage 1 rollout generation (cache-aware).
    * ``run_stage_questionnaire`` — Stage 2 questionnaire administration.
    * ``run_stage_trait_scoring`` — Stage 2b OCEAN trait scores.
    * ``run_stage_realism_judge`` — Stage 2b realism diagnostic.
    * ``run_stage_factor_analysis`` — Stage 3 FA + rotations + sub-passes.
    * ``run_stage_labeling`` — Stage 4 factor labelling (+ manual mode).
    * ``run_stage_validation`` — Stage 5 validation (10+ tests).

Config dataclasses are in :mod:`src_dev.psychometric.config`. Lower-level
components (questionnaire IO, prompt builders, response encoding,
preprocessing, plots, HTML report, labeller, inference loop) can be called
directly for more custom pipelines.

The statistical factor-analysis library lives in
:mod:`src_dev.factor_analysis` and is unchanged; the stage functions here
call it.
"""

# ── Configs + Result types ──────────────────────────────────────────────────
from src_dev.psychometric.config import (
    FactorAnalysisStageConfig,
    FactorAnalysisStageResult,
    LabelingStageConfig,
    LabelingStageResult,
    QuestionnaireStageConfig,
    QuestionnaireStageResult,
    RealismJudgeStageConfig,
    RealismJudgeStageResult,
    RolloutsStageConfig,
    RolloutStageResult,
    RunContext,
    TraitScoringStageConfig,
    TraitScoringStageResult,
    ValidationStageConfig,
    ValidationStageResult,
)

# ── Low-level building blocks (useful for custom scripts) ───────────────────
from src_dev.psychometric.factor_extremes_html import (
    FACTOR_EXTREMES_N,
    export_factor_extremes_html,
)
from src_dev.psychometric.item_prompts import (
    LIKERT_PHRASINGS,
    TRAIT_MCQ_PREFILL,
    build_item_prompt,
    build_questionnaire_messages,
    build_questionnaire_token_ids,
    item_prefill,
    retry_message,
)
from src_dev.psychometric.labelling import (
    describe_column_for_labeller,
    label_factors_by_loadings,
    label_factors_claude_cli,
    label_factors_llm,
    load_latest_nonempty_llm_labels,
    parse_labeller_json_response,
)
from src_dev.psychometric.metadata_enrichment import (
    enrich_meta_with_archetype_scenario,
    load_archetype_scenario_lookup,
)
from src_dev.psychometric.preprocessing import preprocess_response_matrix
from src_dev.psychometric.questionnaire_inference import (
    estimate_max_model_len,
    run_questionnaire_inference,
    run_questionnaire_inference_async,
)
from src_dev.psychometric.questionnaire_io import load_questionnaire
from src_dev.psychometric.realism_judge import (
    summarize_realism_scores,
    write_conversation_html,
)
from src_dev.psychometric.response_encoding import (
    RESPONSE_MATRIX_ENCODING_VERSION,
    fill_matrix_from_choice,
    record_response,
)
from src_dev.psychometric.response_parsing import (
    parse_ab_response,
    parse_abcd_response,
    parse_item_response,
    parse_likert_response,
    parse_top_logprobs_to_choice_probs,
)

__all__ = [
    # Configs + results
    "FactorAnalysisStageConfig",
    "FactorAnalysisStageResult",
    "LabelingStageConfig",
    "LabelingStageResult",
    "QuestionnaireStageConfig",
    "QuestionnaireStageResult",
    "RealismJudgeStageConfig",
    "RealismJudgeStageResult",
    "RolloutsStageConfig",
    "RolloutStageResult",
    "RunContext",
    "TraitScoringStageConfig",
    "TraitScoringStageResult",
    "ValidationStageConfig",
    "ValidationStageResult",
    # Questionnaire IO
    "load_questionnaire",
    # Item prompts
    "LIKERT_PHRASINGS",
    "TRAIT_MCQ_PREFILL",
    "build_item_prompt",
    "build_questionnaire_messages",
    "build_questionnaire_token_ids",
    "item_prefill",
    "retry_message",
    # Response parsing
    "parse_ab_response",
    "parse_abcd_response",
    "parse_item_response",
    "parse_likert_response",
    "parse_top_logprobs_to_choice_probs",
    # Response encoding
    "RESPONSE_MATRIX_ENCODING_VERSION",
    "fill_matrix_from_choice",
    "record_response",
    # Inference
    "estimate_max_model_len",
    "run_questionnaire_inference",
    "run_questionnaire_inference_async",
    # Preprocessing
    "preprocess_response_matrix",
    # Metadata enrichment
    "enrich_meta_with_archetype_scenario",
    "load_archetype_scenario_lookup",
    # Realism judge helpers
    "summarize_realism_scores",
    "write_conversation_html",
    # Labelling
    "describe_column_for_labeller",
    "label_factors_by_loadings",
    "label_factors_claude_cli",
    "label_factors_llm",
    "load_latest_nonempty_llm_labels",
    "parse_labeller_json_response",
    # Factor extremes HTML
    "FACTOR_EXTREMES_N",
    "export_factor_extremes_html",
]
