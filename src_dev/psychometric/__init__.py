"""Psychometric questionnaire administration + factor-analysis experiment components.

This package contains the reusable machinery previously inlined in
``scripts_dev/unsupervised_embeddings/psychometric_rollout_fa.py``.

Config dataclasses are in :mod:`src_dev.psychometric.config`. Lower-level
components (questionnaire IO, prompt builders, response encoding,
preprocessing, plots, HTML report, labeller, inference loop) can be called
directly for custom pipelines. Higher-level ``run_stage_*`` entry points
live in :mod:`src_dev.psychometric.stages` — most experiment scripts will
build a stage config and call one of those.

The statistical factor-analysis library lives in
:mod:`src_dev.factor_analysis` and is unchanged.
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

# ── Stage entry points ──────────────────────────────────────────────────────
from src_dev.psychometric.stages import (
    run_stage_factor_analysis,
    run_stage_labeling,
    run_stage_questionnaire,
    run_stage_realism_judge,
    run_stage_rollouts,
    run_stage_trait_scoring,
    run_stage_validation,
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
    # Stage entry points
    "run_stage_factor_analysis",
    "run_stage_labeling",
    "run_stage_questionnaire",
    "run_stage_realism_judge",
    "run_stage_rollouts",
    "run_stage_trait_scoring",
    "run_stage_validation",
]
