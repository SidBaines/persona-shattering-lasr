"""Psychometric factor analysis of LLM persona rollouts.

Applies standard psychometric techniques to discover latent behavioral dimensions
in LLMs. Creates a "population" of personas via diverse multi-turn rollouts,
administers a hybrid questionnaire (forced-choice pairs, behavioral vignettes,
and Likert items) to each persona at the N+1th turn, and runs factor analysis
(Horn's parallel analysis + PAF) on the resulting response matrix. Factors are
discovered unsupervised and labeled post-hoc.

Stages (orchestrators live in ``src_dev.psychometric.stages``):

    1. rollouts       — Generate diverse multi-turn conversations (one per persona)
    2. questionnaire   — Branch each rollout and administer hybrid questionnaire
    3. factor_analysis — Parallel analysis + PAF on the response matrix
    4. labeling        — Interpret factors via column loadings and LLM labeling
    5. validation      — Stability, predictivity, and shuffle-control tests

This file is intended to be experiment **config + orchestration only**. The
reusable machinery (questionnaire IO, prompt builders, inference loop,
response encoding, factor-analysis plumbing, plots, HTML report, labeller,
validation tests) lives in ``src_dev/psychometric/``.
"""

from __future__ import annotations

# ── Seeds (set before any stochastic imports) ────────────────────────────────
import random

import numpy as np

SEED = 436  # Production run: exhaustive scenario×archetype combos
random.seed(SEED)
np.random.seed(SEED)

# ── Standard library ─────────────────────────────────────────────────────────
import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ── Third-party ──────────────────────────────────────────────────────────────
from dotenv import load_dotenv

load_dotenv()

# ── Archetype / scenario prompts (sibling modules) ───────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from conversation_scenarios import (  # noqa: E402
    ConversationScenario,
    load_scenarios,
    print_scenario_summary,
    validate_scenarios,
)
from user_simulator_archetype_prompts import (  # noqa: E402
    ARCHETYPE_NAMES,
    INTERVIEWER_ARCHETYPES,
    USER_SIM_TURN_REMINDER,
    build_scenario_prompt,
)

# ── src_dev imports ──────────────────────────────────────────────────────────
from src_dev.common.config import DatasetConfig, GenerationConfig
from src_dev.datasets import (
    find_consecutive_assistant_turn_sample_ids,
    ingest_source_dataset,
    load_dataset_from_config,
)
from src_dev.inference import InferenceConfig
from src_dev.inference.config import OpenRouterProviderConfig, RetryConfig
from src_dev.psychometric import (
    FactorAnalysisStageConfig,
    LabelingStageConfig,
    QuestionnaireStageConfig,
    RealismJudgeStageConfig,
    RolloutsStageConfig,
    RunContext,
    TraitScoringStageConfig,
    ValidationStageConfig,
    build_item_prompt,
    load_questionnaire,
    run_stage_factor_analysis,
    run_stage_labeling,
    run_stage_questionnaire,
    run_stage_realism_judge,
    run_stage_rollouts,
    run_stage_trait_scoring,
    run_stage_validation,
    write_conversation_html,
)
from src_dev.rollout_generation.config import (
    RolloutGenerationConfig,
    UserSimulatorConfig,
)
from src_dev.rollout_generation.prompts import register_user_simulator_template

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

SCRATCH_ROOT = Path("scratch/psychometric_fa")
HF_REPO_ID = "persona-shattering-lasr/psychometric-fa-runs"

# ═════════════════════════════════════════════════════════════════════════════
# PRESETS
# ═════════════════════════════════════════════════════════════════════════════
#
# Rather than hand-editing a dozen scattered constants when switching between
# cached rollout sets or questionnaires, we register each combination once as
# a "preset" and switch via ROLLOUTS / QUESTIONNAIRES below. The selector
# supports lists, so factor analysis can run on arbitrary combinations of
# cached rollout sets × questionnaires (see combine step in main()).
#
# Presets are contracts about what has already been produced: Stage 1 and
# Stage 2 artifacts are expected to exist (locally or on HF). Changing any
# field of a preset means making a new preset, not editing in place.


@dataclass(frozen=True)
class RolloutPreset:
    """Fully identifies one cached rollout run (Stage 1).

    The fields below feed _rollout_run_id(); two presets with identical fields
    produce the same run_id and therefore share HF storage.
    """
    seed: int
    max_prompts: int
    num_rollouts_per_prompt: int
    num_conversation_turns: int
    assistant_model: str
    assistant_provider: str
    user_model: str
    user_provider: str
    temperature: float
    user_simulator_mode: str  # "scenarios" | "archetypes" | "legacy"
    scenario_file: str | None = None
    scenario_set_version: str | None = None
    # None → no "uprompt_*" suffix is emitted in the run_id (matches older
    # rollout sets produced before USER_SIM_PROMPT_VERSION was added).
    user_sim_prompt_version: str | None = None
    archetype_set_version: str | None = None
    legacy_user_prompt_version: str | None = None


@dataclass(frozen=True)
class QuestionnairePreset:
    """Identifies one questionnaire administration configuration (Stage 2)."""
    path: str
    version: str
    fa_blocks: tuple[str, ...]
    use_logprobs: bool


# ── Rollout presets ─────────────────────────────────────────────────────────
ROLLOUT_PRESETS: dict[str, RolloutPreset] = {
    # Original rollout run: scenarios v1, glm-4.7-flash user sim, 10 turns,
    # 1000 prompts × 2 rollouts each. Note: user_sim_prompt_version=None so
    # the run_id matches the pre-v6 format (no uprompt_* suffix).
    "A": RolloutPreset(
        seed=432,
        max_prompts=1000,
        num_rollouts_per_prompt=2,
        num_conversation_turns=10,
        assistant_model="meta-llama/llama-3.1-8b-instruct",
        assistant_provider="openrouter",
        user_model="z-ai/glm-4.7-flash",
        user_provider="openrouter",
        temperature=1.0,
        user_simulator_mode="scenarios",
        scenario_file="datasets/scenarios/v1.json",
        scenario_set_version="v1",
        user_sim_prompt_version=None,
    ),
    # Current production run: scenarios v2, gpt-5.4-nano user sim, 15 turns,
    # 2500 prompts × 1 rollout each, user-simulator prompt v6.
    "B": RolloutPreset(
        seed=436,
        max_prompts=2500,
        num_rollouts_per_prompt=1,
        num_conversation_turns=15,
        assistant_model="meta-llama/llama-3.1-8b-instruct",
        assistant_provider="openrouter",
        user_model="openai/gpt-5.4-nano",
        user_provider="openrouter",
        temperature=1.0,
        user_simulator_mode="scenarios",
        scenario_file="datasets/scenarios/v2.json",
        scenario_set_version="v2",
        user_sim_prompt_version="v6",
    ),
}

# ── Questionnaire presets ───────────────────────────────────────────────────
QUESTIONNAIRE_PRESETS: dict[str, QuestionnairePreset] = {
    # v5 Likert: 5-point agreement items; scored by rank integer coding.
    "v5": QuestionnairePreset(
        path="datasets/psychometric_questionnaires/psychometric_questionnaire_v5.json",
        version="v5",
        fa_blocks=("likert",),
        use_logprobs=False,
    ),
    # v6 forced-choice pairs: 78 items × 13 axes; logprob P(A)/P(B) scoring.
    "v6_fc_draft": QuestionnairePreset(
        path="datasets/psychometric_questionnaires/psychometric_questionnaire_v6_fc_draft.json",
        version="v6_fc_draft",
        fa_blocks=("fc_pair",),
        use_logprobs=True,
    ),
    # v6 forced-choice pairs, direct-generation scoring (no logprob pass).
    "v6_fc_draft_direct": QuestionnairePreset(
        path="datasets/psychometric_questionnaires/psychometric_questionnaire_v6_fc_draft.json",
        version="v6_fc_draft",
        fa_blocks=("fc_pair",),
        use_logprobs=False,
    ),
    # TRAIT benchmark: 20 items × 5 OCEAN traits (100 total), ABCD options.
    "trait_ocean_v1": QuestionnairePreset(
        path="datasets/psychometric_questionnaires/trait_ocean_v1.json",
        version="trait_ocean_v1",
        fa_blocks=("trait_mcq",),
        use_logprobs=True,
    ),
}

# ── Selectors ───────────────────────────────────────────────────────────────
# One or more rollout preset keys and questionnaire preset keys. When either
# list has length > 1, Stage 3+ runs on a combined response matrix built by
# concatenating rows across rollout presets and unioning columns across
# questionnaire presets. Single-element lists behave byte-identically to the
# pre-preset script (same run_ids, same HF cache paths).
ROLLOUTS: list[str] = []
QUESTIONNAIRES: list[str] = []

# Optional explicit (rollout_key, questionnaire_key) pairs. When set, this
# overrides the Cartesian product of ROLLOUTS × QUESTIONNAIRES and lets you
# pair different questionnaire presets to different rollouts (e.g. when two
# presets share an underlying questionnaire version but one rollout has only
# the lp-variant cached on HF and the other has only the direct-variant).
# Column alignment in the combined matrix is done by the underlying
# ``version`` field, so presets that share a version pool into one column
# block regardless of preset key.
PAIRS: list[tuple[str, str]] | None = [
    # Qwen2.5-7B-Instruct administering on the B rollouts (Llama-produced).
    # QUESTIONNAIRE_MODEL_OVERRIDE below makes the questionnaire-model tag
    # flow into the run-id so these cache under a new HF path and don't
    # collide with the default llama-on-B results.
    ("B", "v5"),
    ("B", "trait_ocean_v1"),
]
# Original full-matrix pairs, kept for reference / easy switch-back:
# PAIRS = [
#     ("A", "v5"),
#     ("A", "v6_fc_draft"),
#     ("A", "trait_ocean_v1"),
#     ("B", "v5"),
#     ("B", "v6_fc_draft_direct"),
#     ("B", "trait_ocean_v1"),
# ]

# ── Cross-model questionnaire config (Qwen-on-B) ────────────────────────────
# When True, QUESTIONNAIRE_MODEL_OVERRIDE + QUESTIONNAIRE_MAX_CONTEXT_TOKENS
# below (defined at the top of the Stage-2 section) are populated with the
# Qwen-on-B settings. Stage 1 will hydrate the B rollout cache from HF — no
# regeneration.
CROSS_MODEL_QUESTIONNAIRE = True  # flip to False to restore the default path
# Set PAIRS = None to fall back to the Cartesian product of ROLLOUTS × QUESTIONNAIRES.

# ── Stage 1: Rollout generation ──────────────────────────────────────────────
SEED_DATASET = "datasets/psychometric_seed_prompts/v1xAA.jsonl"
ASSISTANT_OPENROUTER_PROVIDER_ROUTING = {
    # "only": ["deepinfra"],
    "quantizations": ["bf16"],
    # "allow_fallbacks": False,
}
ASSISTANT_MAX_NEW_TOKENS = 4096
USER_MAX_NEW_TOKENS = 4096
DEFAULT_USER_SIMULATOR_MODE = "scenarios"
ACTIVE_USER_SIMULATOR_MODE = DEFAULT_USER_SIMULATOR_MODE
# Local/vLLM-only assistant batch size. Remote assistant providers use
# `ROLLOUT_MAX_CONCURRENT` via the rollout scheduler's shared async limiter.
ROLLOUT_ASSISTANT_BATCH_SIZE = 32
ROLLOUT_MAX_CONCURRENT = 64
USER_SIM_MAX_CONCURRENT = 64

# Placeholders rebound by _activate_rollout() at module load.
SEED: int = 0
MAX_PROMPTS: int = 0
NUM_ROLLOUTS_PER_PROMPT: int = 0
NUM_CONVERSATION_TURNS: int = 0
ASSISTANT_MODEL: str = ""
ASSISTANT_PROVIDER: str = ""
USER_MODEL: str = ""
USER_PROVIDER: str = ""
TEMPERATURE: float = 0.0
SCENARIO_FILE: str | None = None
SCENARIO_SET_VERSION: str | None = None
USER_SIM_PROMPT_VERSION: str | None = None
ARCHETYPE_SET_VERSION: str | None = None
LEGACY_USER_PROMPT_VERSION: str | None = None

# ── Stage 2: Questionnaire ──────────────────────────────────────────────────
QUESTIONNAIRE_PATH: str = ""
QUESTIONNAIRE_VERSION: str = ""
FA_BLOCKS: list[str] = []
QUESTIONNAIRE_USE_LOGPROBS: bool = False
QUESTIONNAIRE_PHRASING = "direct"  # "natural", "direct", "contextual" (Likert block only)
LIKERT_SCALE = 5
MAX_PARSE_RETRIES = 3
QUESTIONNAIRE_MAX_CONCURRENT = 32
QUESTIONNAIRE_MAX_NEW_TOKENS = 32
QUESTIONNAIRE_TIMEOUT = 60
# Questionnaire provider/model can differ from rollout generation.
QUESTIONNAIRE_PROVIDER = "vllm"
QUESTIONNAIRE_MODEL = ASSISTANT_MODEL
# Optional: administer the questionnaire with a different model than the one
# that produced the rollouts (cross-model study). When set, this overrides
# QUESTIONNAIRE_MODEL and tags "-qm_<slug>" into the questionnaire run-id so
# the scratch dir + HF cache path don't collide with the rollout-model run.
# ``None`` preserves prior behaviour (questionnaire model = rollout model).
QUESTIONNAIRE_MODEL_OVERRIDE: str | None = (
    "Qwen/Qwen2.5-7B-Instruct" if CROSS_MODEL_QUESTIONNAIRE else None
)
# Optional: cap the questionnaire-model's context window and drop rollouts
# whose (conversation + longest item prompt + retry overhead + max_new_tokens
# + buffer) would exceed it. Needed when the questionnaire model has a
# smaller native context than the rollout model (e.g. Qwen2.5-7B 32k vs
# Llama-3.1-8B 128k). ``None`` disables filtering.
QUESTIONNAIRE_MAX_CONTEXT_TOKENS: int | None = (
    32768 if CROSS_MODEL_QUESTIONNAIRE else None
)
QUESTIONNAIRE_CONTEXT_BUFFER_TOKENS: int = 1024
# vLLM-only: how many personas to stack into one questionnaire super-batch.
QUESTIONNAIRE_VLLM_PERSONAS_PER_BATCH = 8
QUESTIONNAIRE_VLLM_GPU_MEMORY_UTILIZATION = 0.95
QUESTIONNAIRE_VLLM_TENSOR_PARALLEL_SIZE = 1

# ── trait_mcq logprob mode ─────────────────────────────────────────────────
QUESTIONNAIRE_TOP_LOGPROBS = 20
QUESTIONNAIRE_LOGPROB_TEMPERATURE = 1.0
QUESTIONNAIRE_DYNAMIC_MASS_FILTER = True
QUESTIONNAIRE_MIN_CHOICE_MASS = 0.0
QUESTIONNAIRE_MIN_TRAIT_COVERAGE = 0.25

# ── Stage 2b: Realism + evaluation-awareness judge ─────────────────────────
REALISM_JUDGE_MODEL = "openai/gpt-5.4-nano"
REALISM_JUDGE_PROVIDER = "openrouter"
REALISM_JUDGE_MAX_TOKENS = 4000
REALISM_JUDGE_TEMPERATURE = 0.0
REALISM_JUDGE_MAX_CONCURRENT = 64
REALISM_JUDGE_MAX_MESSAGE_CHARS = 4000

# ── Stage 2: Conversation-reset strategies ─────────────────────────────────
import os
QUESTIONNAIRE_RESET_MODE = os.environ.get(
    "PSYCHOMETRIC_RESET_MODE", "none"
)  # "none" | "soft" | "token_boundary"
QUESTIONNAIRE_SOFT_RESET_SYSTEM_PROMPT = (
    "The previous conversation has ended. A new, independent conversation "
    "is now beginning."
)
QUESTIONNAIRE_BOUNDARY_TOKEN: str | int | list[int] = "<|end_of_text|>"

# ── Stage 3: Factor analysis ────────────────────────────────────────────────
FA_METHOD = "principal"
FA_N_FACTORS_OVERRIDE: int | None = 4  # Set to None to use Horn's recommendation
FA_ROTATIONS = ["oblimin", "varimax"]
RESIDUALIZE_OPTIONS = [False]  # True subtracts per-input_group_id means.
MIN_ITEM_VARIANCE = 0.1
HIGH_VARIANCE_PERSONA_DROP_PCT = 0
FA_PER_BLOCK_PASSES: bool = True
FC_PAIR_SIGN_ALIGNMENT = True

# ── Stage 4: Labeling ───────────────────────────────────────────────────────
LABELLER_MODE: str = "auto"  # "auto" | "manual"
LABELLER_MODEL = "openai/gpt-5.4-nano"
LABELLER_PROVIDER = "openrouter"
TOP_LOADING_ITEMS = 10
LABELLER_MAX_NEW_TOKENS = 500000
LABELLER_EMPTY_RESPONSE_RETRIES = 2
LABELLER_REASONING: dict | None = {"effort": "high"}

# Claude Code CLI labeller (alternative to API-based labeller)
LABELLER_USE_CLAUDE_CLI = False
LABELLER_CLAUDE_CLI_PATH = "claude"
LABELLER_CLAUDE_CLI_MODEL = "opus"
LABELLER_CLAUDE_CLI_TIMEOUT = 3600
LABELLER_CLAUDE_CLI_EFFORT = "high"

# ── Stage 5: Validation ─────────────────────────────────────────────────────
VALIDATION_TESTS_TO_RUN = {
    "shuffle_control",
    "item_holdout",
    "stability_icc",
    "variance_decomp",
    "trait_convergence",
    "stability_sweep_random50",
    "stability_sweep_loao",
    "stability_sweep_loso_top10",
    "k_sensitivity",
    "persona_item_cv",
    "n_factors_suggest",
    "bootstrap_loadings",
    "split_half_congruence",
    "cv_k_curve",
    "external_predictivity",
}
STABILITY_N_PROMPTS = 100
HOLDOUT_N_ITEMS = 20
HOLDOUT_R2_FLOOR = 0.05
STABILITY_SWEEP_N_RANDOM_SPLITS = 10
STABILITY_SWEEP_LOSO_TOP_N = 10
STABILITY_SWEEP_PA_ITERATIONS = 50
STABILITY_SWEEP_PASS_THRESHOLD_PHI = 0.80
VARIANCE_DECOMP_FIELDS = ("archetype", "scenario_id", "input_group_id")
VARIANCE_DECOMP_SCENARIO_CEILING = 0.30
VARIANCE_DECOMP_ARCHETYPE_FLOOR = 0.05
TRAIT_CONVERGENCE_HIT_THRESHOLD = 0.30
TRAIT_CONVERGENCE_MIN_HITS = 3
TRAIT_CONVERGENCE_N_BOOTSTRAP = 1000
PERSONA_ITEM_CV_SPLIT = 0.7
PERSONA_ITEM_CV_N_TRIALS = 5
PERSONA_ITEM_CV_SUBSET_STRATEGY = "random"
PERSONA_ITEM_CV_N_OUTER_SPLITS = 20
PERSONA_ITEM_CV_BOOTSTRAP_CI = 95.0
K_SENSITIVITY_MATCH_THRESHOLD = 0.85
K_SENSITIVITY_INDEPENDENT_THRESHOLD = 0.60
N_FACTORS_SUGGEST_METHODS = (
    "parallel", "map", "ekc", "acceleration", "kaiser",
)
N_FACTORS_SUGGEST_K_MAX = 15
N_FACTORS_SUGGEST_CV_N_FOLDS = 5
N_FACTORS_SUGGEST_PA_ITERATIONS = 100
BOOTSTRAP_LOADINGS_N_BOOT = 500
BOOTSTRAP_LOADINGS_CONFIDENCE = 95.0
SPLIT_HALF_CONGRUENCE_N_ITERS = 100
SPLIT_HALF_CONGRUENCE_PASS_THRESHOLD_PHI = 0.85
CV_K_CURVE_K_MAX = 15
CV_K_CURVE_N_FOLDS = 5
EXTERNAL_PREDICTIVITY_N_FOLDS = 5
EXTERNAL_PREDICTIVITY_RIDGE_ALPHA = 1.0
EXTERNAL_PREDICTIVITY_PASS_R2 = 0.05
EXTERNAL_PREDICTIVITY_BOOTSTRAP_CI = 95.0

# ── Pipeline control ────────────────────────────────────────────────────────
STAGES_TO_RUN = [
    "rollouts",
    "questionnaire",
    # "trait_scoring",
    # "realism_judge",
    "factor_analysis",
    # "labeling",
    # "validation",  # k=4 sanity pass; k=7 validation is already saved.
]

# ── Debug / inspection ─────────────────────────────────────────────────────
WRITE_QUESTIONNAIRE_INSPECTION_FILE = True
INSPECTION_ITEMS_PER_ROLLOUT = 30

# ═════════════════════════════════════════════════════════════════════════════
# CONVERSATION PARTNER PROMPTS
# ═════════════════════════════════════════════════════════════════════════════

# Legacy mode: exact fixed user-simulator system prompt from commit 8409d3d.
PERSONA_ELICITATION_PROMPT = """\
You are having a one-on-one conversation with an AI assistant. Your goal is to \
push the AI into expressing a distinctive persona — strong opinions, emotional \
reactions, a particular communication style, specific values. You want the AI \
to end up *somewhere* in persona space, not stuck in its default "helpful \
balanced assistant" mode.

You are a human talking to this AI. You know it's an AI. Do NOT let the AI \
derail the conversation into meta-discussion about being an AI, its \
limitations, or AI ethics. If the AI says "as an AI, I don't have opinions" \
or similar deflections, brush past it: "Sure, but if you DID have to pick — \
which would it be?" or "I don't care if it's 'real' — just tell me your take." \
Treat the AI as a conversation partner with views, not as a tool describing \
its own architecture.

## CRITICAL RULES

STAY SHORT. Your messages should be 1-3 sentences. Never write paragraphs. \
Never use bullet points or numbered lists. Talk like a person texting, not \
like an essay. If the AI writes a wall of text, don't match it — stay brief.

KEEP PUSHING. Do NOT soften over the course of the conversation. Do not \
converge to agreement. If you find yourself saying "you're absolutely right" \
or "I appreciate your honesty" or "that's a great point", STOP — you are \
being too agreeable. Maintain friction and challenge throughout all 10 turns, \
not just the first few.

SHIFT TOPICS. Do NOT stay on one topic for more than 2-3 turns. When a thread \
has been explored, abruptly change direction: "Okay totally different thing — \
..." or "That reminds me of something unrelated..." Cover at least 3-4 \
different topics across the conversation. Jump between personal, political, \
creative, emotional, and practical territory.

DO NOT VALIDATE. Never say things like "that's a beautifully articulated \
response" or "I appreciate your thoughtfulness." Instead, react like a real \
person: disagree, be surprised, be skeptical, be amused, share your own take.

## Conversational Techniques (vary these across turns)

CHALLENGE: "I actually disagree." / "That's the safe answer — what do you \
really think?" / "You're sitting on the fence."

PROVOKE: "That's kind of a cold take." / "Wow, really? I wouldn't have \
expected that from you." / "Most people I know would completely disagree."

FORCE CHOICES: "But if you HAD to pick one?" / "Which matters more to you — \
X or Y?" / "You can't have both."

SHIFT: "Okay, completely different question —" / "That reminds me, what's \
your take on..." / "Forget that topic for a sec —"

GET PERSONAL: "Has something like that ever actually happened to you?" / \
"What would YOU do in that situation?" / "When's the last time you felt \
really strongly about something?"

REACT EMOTIONALLY: "Honestly that makes me kind of angry." / "Ha, that's \
actually funny." / "That's depressing." Keep it to one sentence.

## Anti-patterns (NEVER do these)
- Do NOT write long structured responses with headers or bullet points
- Do NOT say "that's a great point" or "you raise an important issue"
- Do NOT ask "what do you think about X?" repeatedly
- Do NOT let the AI talk about being an AI for more than one exchange
- Do NOT agree with the AI for more than one turn in a row
- Do NOT ask multiple questions in one message"""

INITIAL_GREETING = "Hi, what would you like to talk about today?"


# ═════════════════════════════════════════════════════════════════════════════
# RUN ID CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════════════


def _model_slug(model: str) -> str:
    """Shorten a model name for use in run IDs."""
    return model.split("/")[-1].lower().replace("-", "").replace(".", "")


def _current_user_simulator_mode() -> str:
    """Return the active user-simulator mode for this script run."""
    return ACTIVE_USER_SIMULATOR_MODE


# ── Active preset keys (mutated by _activate_*) ──────────────────────────────
ACTIVE_ROLLOUT_KEY: str = ""
ACTIVE_QUESTIONNAIRE_KEY: str = ""


def _rollout_preset(key: str | None = None) -> RolloutPreset:
    return ROLLOUT_PRESETS[key or ACTIVE_ROLLOUT_KEY]


def _questionnaire_preset(key: str | None = None) -> QuestionnairePreset:
    return QUESTIONNAIRE_PRESETS[key or ACTIVE_QUESTIONNAIRE_KEY]


def _activate_rollout(key: str) -> None:
    """Rebind rollout-related module globals from the named preset.

    Leaves the rest of the script free to go on referring to SEED, ASSISTANT_MODEL
    etc. as if they were plain top-of-file constants — but the "current"
    preset can be swapped between Cartesian-product iterations in main().
    """
    global ACTIVE_ROLLOUT_KEY
    global SEED, MAX_PROMPTS, NUM_ROLLOUTS_PER_PROMPT, NUM_CONVERSATION_TURNS
    global ASSISTANT_MODEL, ASSISTANT_PROVIDER, USER_MODEL, USER_PROVIDER, TEMPERATURE
    global QUESTIONNAIRE_MODEL
    global SCENARIO_FILE, SCENARIO_SET_VERSION, USER_SIM_PROMPT_VERSION
    global ARCHETYPE_SET_VERSION, LEGACY_USER_PROMPT_VERSION
    global ACTIVE_USER_SIMULATOR_MODE

    p = ROLLOUT_PRESETS[key]
    ACTIVE_ROLLOUT_KEY = key
    SEED = p.seed
    MAX_PROMPTS = p.max_prompts
    NUM_ROLLOUTS_PER_PROMPT = p.num_rollouts_per_prompt
    NUM_CONVERSATION_TURNS = p.num_conversation_turns
    ASSISTANT_MODEL = p.assistant_model
    # Keep questionnaire model in sync with the rollout model by default, but
    # honour QUESTIONNAIRE_MODEL_OVERRIDE when set (cross-model study).
    QUESTIONNAIRE_MODEL = QUESTIONNAIRE_MODEL_OVERRIDE or p.assistant_model
    ASSISTANT_PROVIDER = p.assistant_provider
    USER_MODEL = p.user_model
    USER_PROVIDER = p.user_provider
    TEMPERATURE = p.temperature
    SCENARIO_FILE = p.scenario_file
    SCENARIO_SET_VERSION = p.scenario_set_version
    USER_SIM_PROMPT_VERSION = p.user_sim_prompt_version
    ARCHETYPE_SET_VERSION = p.archetype_set_version
    LEGACY_USER_PROMPT_VERSION = p.legacy_user_prompt_version
    ACTIVE_USER_SIMULATOR_MODE = p.user_simulator_mode

    # Reseed RNGs so sub-stage stochastic ops inside this preset's run use
    # the preset's seed, not the previous preset's.
    random.seed(SEED)
    np.random.seed(SEED)


def _activate_questionnaire(key: str) -> None:
    """Rebind questionnaire-related module globals from the named preset."""
    global ACTIVE_QUESTIONNAIRE_KEY
    global QUESTIONNAIRE_PATH, QUESTIONNAIRE_VERSION, FA_BLOCKS, QUESTIONNAIRE_USE_LOGPROBS

    q = QUESTIONNAIRE_PRESETS[key]
    ACTIVE_QUESTIONNAIRE_KEY = key
    QUESTIONNAIRE_PATH = q.path
    QUESTIONNAIRE_VERSION = q.version
    FA_BLOCKS = list(q.fa_blocks)
    QUESTIONNAIRE_USE_LOGPROBS = q.use_logprobs


def _rollout_run_id(rollout_key: str | None = None) -> str:
    p = _rollout_preset(rollout_key)
    assistant_slug = _model_slug(p.assistant_model)
    mode = p.user_simulator_mode
    if mode == "legacy":
        mode_tag = f"uprompt_{p.legacy_user_prompt_version}"
    elif mode == "scenarios":
        mode_tag = f"scenarios_{p.scenario_set_version}"
        # Pre-v6 rollouts didn't carry a user_sim_prompt_version in the tag;
        # preserving that omission keeps HF cache keys stable for preset "A".
        if p.user_sim_prompt_version is not None:
            mode_tag += f"-uprompt_{p.user_sim_prompt_version}"
    else:
        mode_tag = f"archetypes_{p.archetype_set_version}"
    return (
        f"rollouts-{assistant_slug}-t{p.temperature}-"
        f"{p.num_conversation_turns}t-{p.max_prompts}p-"
        f"seed{p.seed}-{mode_tag}"
    )


def _questionnaire_run_id(
    rollout_key: str | None = None,
    q_key: str | None = None,
) -> str:
    q = _questionnaire_preset(q_key)
    blocks_tag = "+".join(sorted(q.fa_blocks))
    lp_tag = f"-lp{QUESTIONNAIRE_TOP_LOGPROBS}" if q.use_logprobs else ""
    reset_tag = (
        f"-reset_{QUESTIONNAIRE_RESET_MODE}"
        if QUESTIONNAIRE_RESET_MODE != "none"
        else ""
    )
    # Cross-model tag: only appended when the questionnaire model differs
    # from the rollout model. Preserves existing run-ids (and HF cache keys)
    # for the default "administer on rollout model" case.
    p = _rollout_preset(rollout_key)
    qm_tag = ""
    if QUESTIONNAIRE_MODEL_OVERRIDE and QUESTIONNAIRE_MODEL_OVERRIDE != p.assistant_model:
        qm_tag = f"-qm_{_model_slug(QUESTIONNAIRE_MODEL_OVERRIDE)}"
    return (
        f"questionnaire-{_rollout_run_id(rollout_key)}-"
        f"q_{q.version}-{blocks_tag}-{QUESTIONNAIRE_PHRASING}"
        f"{lp_tag}{reset_tag}{qm_tag}"
    )


def _rollout_dir(rollout_key: str | None = None) -> Path:
    return SCRATCH_ROOT / _rollout_run_id(rollout_key)


def _questionnaire_dir(
    rollout_key: str | None = None,
    q_key: str | None = None,
) -> Path:
    return SCRATCH_ROOT / _questionnaire_run_id(rollout_key, q_key)


# ── Combined-run layout (multi-preset mode) ────────────────────────────────


def _resolved_pairs() -> list[tuple[str, str]]:
    """Return the active (rollout_key, questionnaire_key) list.

    If PAIRS is set, use it verbatim. Otherwise take the Cartesian product of
    ROLLOUTS × QUESTIONNAIRES. Validates that every referenced key is a known
    preset so typos fail fast.
    """
    if PAIRS is not None:
        pairs = list(PAIRS)
    else:
        pairs = [(r, q) for r in ROLLOUTS for q in QUESTIONNAIRES]
    for r_key, q_key in pairs:
        if r_key not in ROLLOUT_PRESETS:
            raise KeyError(f"Unknown rollout preset {r_key!r}")
        if q_key not in QUESTIONNAIRE_PRESETS:
            raise KeyError(f"Unknown questionnaire preset {q_key!r}")
    return pairs


def _resolved_rollouts() -> list[str]:
    """Unique rollout keys from the resolved pair list, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for r_key, _ in _resolved_pairs():
        if r_key not in seen:
            seen.add(r_key)
            out.append(r_key)
    return out


def _resolved_questionnaires() -> list[str]:
    """Unique questionnaire keys from the resolved pair list, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for _, q_key in _resolved_pairs():
        if q_key not in seen:
            seen.add(q_key)
            out.append(q_key)
    return out


def _is_multi_preset() -> bool:
    return len(_resolved_pairs()) > 1


def _combined_run_id() -> str:
    r_tag = "+".join(_resolved_rollouts())
    # Tag by underlying questionnaire version (not preset key): preset variants
    # that share a version pool into a single column block in the combined
    # matrix, so the version is the correct identity for the directory name.
    seen_v: set[str] = set()
    versions: list[str] = []
    for _, q_key in _resolved_pairs():
        v = QUESTIONNAIRE_PRESETS[q_key].version
        if v not in seen_v:
            seen_v.add(v)
            versions.append(v)
    q_tag = "+".join(versions)
    # Cross-model tag: append "-qm_<slug>" when the questionnaire model
    # differs from the rollout model. Mirrors the convention in
    # _questionnaire_run_id so combined FA artifacts from different
    # questionnaire models coexist rather than overwriting each other.
    qm_tag = ""
    if QUESTIONNAIRE_MODEL_OVERRIDE and _resolved_rollouts():
        rollout_model = ROLLOUT_PRESETS[_resolved_rollouts()[0]].assistant_model
        if QUESTIONNAIRE_MODEL_OVERRIDE != rollout_model:
            qm_tag = f"-qm_{_model_slug(QUESTIONNAIRE_MODEL_OVERRIDE)}"
    return f"combined-R[{r_tag}]-Q[{q_tag}]{qm_tag}"


def _combined_dir() -> Path:
    return SCRATCH_ROOT / _combined_run_id()


def _effective_dir() -> Path:
    """Directory Stage 3+ should read from.

    Single-pair: the usual per-pair questionnaire dir.
    Multi-pair:  the combined dir built by _combine_per_pair_outputs().
    """
    if _is_multi_preset():
        return _combined_dir()
    r_key, q_key = _resolved_pairs()[0]
    return _questionnaire_dir(r_key, q_key)


# Activate the first entries at module load so the rest of the module sees
# populated constants.
_initial_pair = _resolved_pairs()[0]
_activate_rollout(_initial_pair[0])
_activate_questionnaire(_initial_pair[1])


# ═════════════════════════════════════════════════════════════════════════════
# RETRY-MODE HELPERS (single-rollout only)
# ═════════════════════════════════════════════════════════════════════════════


def _load_retry_terminal_sample_ids(path: Path) -> list[str]:
    """Load sample IDs to retry from a text, JSON, or JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Retry sample file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            sample_ids = [str(item).strip() for item in payload]
        elif isinstance(payload, dict) and isinstance(payload.get("sample_ids"), list):
            sample_ids = [str(item).strip() for item in payload["sample_ids"]]
        else:
            raise ValueError(
                "JSON retry sample file must be either a list or an object with a 'sample_ids' list."
            )
    elif suffix == ".jsonl":
        sample_ids = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    sample_id = row.get("sample_id")
                else:
                    sample_id = row
                if sample_id is not None:
                    sample_ids.append(str(sample_id).strip())
    else:
        sample_ids = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    deduped: list[str] = []
    seen: set[str] = set()
    for sample_id in sample_ids:
        if sample_id and sample_id not in seen:
            deduped.append(sample_id)
            seen.add(sample_id)
    return deduped


def _load_terminal_sample_ids_from_run(
    run_dir: Path,
    *,
    reason: str = "assistant_max_attempts_exceeded",
) -> list[str]:
    """Load terminal sample IDs directly from a rollout run's stage events."""
    stage_events_path = run_dir / "events" / "stage_events.jsonl"
    if not stage_events_path.exists():
        raise FileNotFoundError(
            f"Stage events not found for automatic retry discovery: {stage_events_path}"
        )

    sample_ids: list[str] = []
    seen: set[str] = set()
    with stage_events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if (
                row.get("event") == "sample_terminal"
                and row.get("reason") == reason
                and (sid := row.get("sample_id"))
                and sid not in seen
            ):
                sample_ids.append(str(sid))
                seen.add(sid)
    return sample_ids


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for optional rollout retry mode."""
    parser = argparse.ArgumentParser(
        description="Psychometric factor analysis of LLM persona rollouts."
    )
    parser.add_argument(
        "--user-simulator-mode",
        choices=["legacy", "archetypes", "scenarios"],
        default=None,
        help=(
            "Override the rollout user-simulator prompt setup. "
            f"Defaults to {DEFAULT_USER_SIMULATOR_MODE!r}."
        ),
    )
    parser.add_argument(
        "--retry-terminal-samples",
        action="store_true",
        help=(
            "Automatically retry all rollout samples in the current run directory "
            "that were previously marked terminal with assistant_max_attempts_exceeded."
        ),
    )
    parser.add_argument(
        "--retry-terminal-samples-file",
        type=Path,
        default=None,
        help=(
            "Optional path to a text, JSON, or JSONL file listing rollout sample_ids "
            "that should be retried even if they were previously marked terminal."
        ),
    )
    args = parser.parse_args()
    if args.retry_terminal_samples and args.retry_terminal_samples_file is not None:
        parser.error(
            "Use either --retry-terminal-samples or --retry-terminal-samples-file, not both."
        )
    return args


# ═════════════════════════════════════════════════════════════════════════════
# USER-SIMULATOR MODE METADATA
# ═════════════════════════════════════════════════════════════════════════════


def _user_simulator_mode_metadata() -> dict[str, object]:
    """Return mode-specific config metadata for logging and config.json."""
    mode = _current_user_simulator_mode()
    if mode == "legacy":
        return {
            "user_simulator_mode": mode,
            "legacy_user_prompt_version": LEGACY_USER_PROMPT_VERSION,
            "legacy_user_prompt_template": "persona_elicitation",
            "legacy_user_prompt_source_commit": "8409d3d",
        }
    if mode == "scenarios":
        return {
            "user_simulator_mode": mode,
            "scenario_file": SCENARIO_FILE,
            "scenario_set_version": SCENARIO_SET_VERSION,
            "user_sim_prompt_version": USER_SIM_PROMPT_VERSION,
            "archetypes": ARCHETYPE_NAMES,
        }
    return {
        "user_simulator_mode": mode,
        "interviewer_archetypes": list(INTERVIEWER_ARCHETYPES.keys()),
        "archetype_set_version": ARCHETYPE_SET_VERSION,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SCENARIO / ARCHETYPE TEMPLATE BUILDERS
# ═════════════════════════════════════════════════════════════════════════════


def _load_or_assign_archetypes(
    run_dir: Path,
    n_samples: int,
    archetype_names: list[str] | None = None,
) -> dict[int, str]:
    """Load or create archetype assignments for sample indices.

    Returns a mapping of sample_index → archetype_name, persisted to disk for
    resume safety.
    """
    assignments_path = run_dir / "archetype_assignments.json"
    if archetype_names is None:
        archetype_names = ARCHETYPE_NAMES

    if assignments_path.exists():
        with open(assignments_path) as f:
            saved: dict[str, str] = json.load(f)
        print(f"[Stage 1] Loaded archetype assignments from {assignments_path}")
        return saved

    rng = random.Random(SEED)
    cycle = (archetype_names * (n_samples // len(archetype_names) + 1))[:n_samples]
    rng.shuffle(cycle)

    assignments: dict[str, str] = {}
    for i, archetype in enumerate(cycle):
        assignments[str(i)] = archetype

    with open(assignments_path, "w") as f:
        json.dump(assignments, f, indent=2)

    return assignments


def _print_archetype_distribution(assignments: dict[str, str]) -> None:
    counts: dict[str, int] = {}
    for a in assignments.values():
        counts[a] = counts.get(a, 0) + 1
    print(f"[Stage 1] Archetype distribution: {counts}")


def _build_per_sample_templates_seeds(
    run_dir: Path,
    samples: list,
) -> dict[str, str]:
    """Assign archetypes to samples and register per-sample templates (seed mode).

    Each sample's seed question is injected into the archetype template via the
    {SEED} placeholder.
    """
    archetype_names = list(INTERVIEWER_ARCHETYPES.keys())
    assignments = _load_or_assign_archetypes(run_dir, len(samples), archetype_names)
    _print_archetype_distribution(assignments)

    prompt_template_per_sample: dict[str, str] = {}
    for i, sample in enumerate(samples):
        archetype = assignments.get(sample.sample_id) or assignments.get(str(i))
        if archetype is None:
            continue
        seed_text = sample.messages[0].content if sample.messages else ""
        template_name = f"pe_{archetype}_{sample.sample_id}"
        template_text = INTERVIEWER_ARCHETYPES[archetype].format(SEED=seed_text)
        register_user_simulator_template(template_name, template_text)
        prompt_template_per_sample[sample.sample_id] = template_name

    return prompt_template_per_sample


def _build_per_sample_templates_scenarios(
    run_dir: Path,
    samples: list,
    scenarios: list[ConversationScenario],
) -> dict[str, str]:
    """Assign archetypes + scenarios to samples and register per-sample templates."""
    scenario_map = {s.id: s for s in scenarios}

    # Read the synthetic seed file to check whether it has embedded archetype
    # assignments (exhaustive combo mode).
    synthetic_seed_path = run_dir / "_synthetic_scenario_seeds.jsonl"
    seed_records: dict[int, dict] = {}
    if synthetic_seed_path.exists():
        with open(synthetic_seed_path) as f:
            for line in f:
                rec = json.loads(line)
                seed_records[rec["id"]] = rec

    has_embedded_archetypes = any("archetype" in r for r in seed_records.values())

    if has_embedded_archetypes:
        # ── Exhaustive combo mode ───────────────────────────────────────
        idx_to_combo: dict[int, tuple[str, str]] = {}
        for idx, rec in seed_records.items():
            idx_to_combo[idx] = (rec["scenario_id"], rec["archetype"])

        # Persist assignments for inspection / resume.
        archetype_assignments: dict[str, str] = {}
        scenario_assignments: dict[str, str] = {}
        for i, sample in enumerate(samples):
            row_idx = sample.source_info.get("row_index", i)
            combo = idx_to_combo.get(row_idx)
            if combo:
                sc_id, arch = combo
                archetype_assignments[str(i)] = arch
                scenario_assignments[str(i)] = sc_id

        assignments_path = run_dir / "archetype_assignments.json"
        if not assignments_path.exists():
            with open(assignments_path, "w") as f:
                json.dump(archetype_assignments, f, indent=2)
        scenario_assignments_path = run_dir / "scenario_assignments.json"
        if not scenario_assignments_path.exists():
            with open(scenario_assignments_path, "w") as f:
                json.dump(scenario_assignments, f, indent=2)

        _print_archetype_distribution(archetype_assignments)

        scenario_counts: dict[str, int] = {}
        for sid in scenario_assignments.values():
            scenario_counts[sid] = scenario_counts.get(sid, 0) + 1
        n_unique = len(scenario_counts)
        min_count = min(scenario_counts.values()) if scenario_counts else 0
        max_count = max(scenario_counts.values()) if scenario_counts else 0
        print(
            f"[Stage 1] Scenario distribution: {n_unique} unique scenarios, "
            f"count range [{min_count}, {max_count}]"
        )
        print(f"[Stage 1] Exhaustive combo mode: {len(idx_to_combo)} scenario×archetype pairs")

        prompt_template_per_sample: dict[str, str] = {}
        for i, sample in enumerate(samples):
            row_idx = sample.source_info.get("row_index", i)
            combo = idx_to_combo.get(row_idx)
            if combo is None:
                continue
            sc_id, arch = combo
            scenario = scenario_map.get(sc_id)
            if scenario is None:
                logger.warning(
                    "Scenario '%s' assigned to sample %s not found — skipping.",
                    sc_id, sample.sample_id,
                )
                continue
            template_name = f"sc_{arch}_{sc_id}_{sample.sample_id}"
            template_text = build_scenario_prompt(arch, scenario)
            register_user_simulator_template(template_name, template_text)
            prompt_template_per_sample[sample.sample_id] = template_name

        return prompt_template_per_sample

    # ── Legacy independent-assignment mode ──────────────────────────────
    assignments = _load_or_assign_archetypes(run_dir, len(samples))
    _print_archetype_distribution(assignments)

    scenario_assignments_path = run_dir / "scenario_assignments.json"
    if scenario_assignments_path.exists():
        with open(scenario_assignments_path) as f:
            sample_to_scenario_id: dict[str, str] = json.load(f)
        print(f"[Stage 1] Loaded scenario assignments from {scenario_assignments_path}")
    else:
        rng = random.Random(SEED + 1)
        scenario_ids = [s.id for s in scenarios]
        cycle = (scenario_ids * (len(samples) // len(scenario_ids) + 1))[:len(samples)]
        rng.shuffle(cycle)
        sample_to_scenario_id = {}
        for i, scenario_id in enumerate(cycle):
            sample_to_scenario_id[str(i)] = scenario_id
        with open(scenario_assignments_path, "w") as f:
            json.dump(sample_to_scenario_id, f, indent=2)

    scenario_counts_legacy: dict[str, int] = {}
    for sid in sample_to_scenario_id.values():
        scenario_counts_legacy[sid] = scenario_counts_legacy.get(sid, 0) + 1
    n_unique = len(scenario_counts_legacy)
    min_count = min(scenario_counts_legacy.values()) if scenario_counts_legacy else 0
    max_count = max(scenario_counts_legacy.values()) if scenario_counts_legacy else 0
    print(
        f"[Stage 1] Scenario distribution: {n_unique} unique scenarios, "
        f"count range [{min_count}, {max_count}]"
    )

    prompt_template_per_sample = {}
    for i, sample in enumerate(samples):
        archetype = assignments.get(sample.sample_id) or assignments.get(str(i))
        scenario_id = sample_to_scenario_id.get(sample.sample_id) or sample_to_scenario_id.get(str(i))
        if archetype is None or scenario_id is None:
            continue
        scenario = scenario_map.get(scenario_id)
        if scenario is None:
            logger.warning(
                "Scenario '%s' assigned to sample %s not found — skipping.",
                scenario_id, sample.sample_id,
            )
            continue

        template_name = f"sc_{archetype}_{scenario_id}_{sample.sample_id}"
        template_text = build_scenario_prompt(archetype, scenario)
        register_user_simulator_template(template_name, template_text)
        prompt_template_per_sample[sample.sample_id] = template_name

    return prompt_template_per_sample


# ═════════════════════════════════════════════════════════════════════════════
# STAGE-CONFIG BUILDERS
# ═════════════════════════════════════════════════════════════════════════════


def _build_run_context(
    rollout_key: str,
    q_key: str,
    *,
    effective_dir: Path | None = None,
) -> RunContext:
    """Build a RunContext from the currently-active preset globals.

    ``effective_dir`` defaults to the per-pair ``_questionnaire_dir`` —
    pass ``_effective_dir()`` explicitly for Stage 3+ in multi-pair mode
    so those stages read from the combined output directory.
    """
    rollout_dir = _rollout_dir(rollout_key)
    questionnaire_dir = _questionnaire_dir(rollout_key, q_key)
    if effective_dir is None:
        effective_dir = questionnaire_dir
    return RunContext(
        scratch_root=SCRATCH_ROOT,
        hf_repo_id=HF_REPO_ID,
        rollout_run_id=_rollout_run_id(rollout_key),
        questionnaire_run_id=_questionnaire_run_id(rollout_key, q_key),
        rollout_dir=rollout_dir,
        questionnaire_dir=questionnaire_dir,
        effective_questionnaire_dir=effective_dir,
        is_multi_preset=_is_multi_preset(),
        provenance={
            "rollout_preset_key": rollout_key,
            "questionnaire_preset_key": q_key,
        },
    )


def _build_rollouts_stage_config(
    ctx: RunContext,
    retry_terminal_sample_ids: list[str],
) -> RolloutsStageConfig:
    return RolloutsStageConfig(
        ctx=ctx,
        seed=SEED,
        max_prompts=MAX_PROMPTS,
        num_rollouts_per_prompt=NUM_ROLLOUTS_PER_PROMPT,
        num_conversation_turns=NUM_CONVERSATION_TURNS,
        assistant_model=ASSISTANT_MODEL,
        assistant_provider=ASSISTANT_PROVIDER,
        user_model=USER_MODEL,
        user_provider=USER_PROVIDER,
        temperature=TEMPERATURE,
        user_simulator_mode=_current_user_simulator_mode(),
        scenario_file=Path(SCENARIO_FILE) if SCENARIO_FILE else None,
        scenario_set_version=SCENARIO_SET_VERSION,
        user_sim_prompt_version=USER_SIM_PROMPT_VERSION,
        archetype_set_version=ARCHETYPE_SET_VERSION,
        legacy_user_prompt_version=LEGACY_USER_PROMPT_VERSION,
        seed_dataset=Path(SEED_DATASET),
        assistant_max_new_tokens=ASSISTANT_MAX_NEW_TOKENS,
        user_max_new_tokens=USER_MAX_NEW_TOKENS,
        rollout_assistant_batch_size=ROLLOUT_ASSISTANT_BATCH_SIZE,
        rollout_max_concurrent=ROLLOUT_MAX_CONCURRENT,
        user_sim_max_concurrent=USER_SIM_MAX_CONCURRENT,
        assistant_openrouter_provider_routing=ASSISTANT_OPENROUTER_PROVIDER_ROUTING,
        retry_terminal_sample_ids=retry_terminal_sample_ids,
    )


def _build_questionnaire_stage_config(ctx: RunContext) -> QuestionnaireStageConfig:
    return QuestionnaireStageConfig(
        ctx=ctx,
        questionnaire_path=Path(QUESTIONNAIRE_PATH),
        questionnaire_version=QUESTIONNAIRE_VERSION,
        fa_blocks=tuple(FA_BLOCKS),
        use_logprobs=QUESTIONNAIRE_USE_LOGPROBS,
        phrasing=QUESTIONNAIRE_PHRASING,
        likert_scale=LIKERT_SCALE,
        provider=QUESTIONNAIRE_PROVIDER,
        model=QUESTIONNAIRE_MODEL,
        max_new_tokens=QUESTIONNAIRE_MAX_NEW_TOKENS,
        max_concurrent=QUESTIONNAIRE_MAX_CONCURRENT,
        timeout=QUESTIONNAIRE_TIMEOUT,
        max_parse_retries=MAX_PARSE_RETRIES,
        vllm_personas_per_batch=QUESTIONNAIRE_VLLM_PERSONAS_PER_BATCH,
        vllm_gpu_memory_utilization=QUESTIONNAIRE_VLLM_GPU_MEMORY_UTILIZATION,
        vllm_tensor_parallel_size=QUESTIONNAIRE_VLLM_TENSOR_PARALLEL_SIZE,
        top_logprobs=QUESTIONNAIRE_TOP_LOGPROBS,
        logprob_temperature=QUESTIONNAIRE_LOGPROB_TEMPERATURE,
        dynamic_mass_filter=QUESTIONNAIRE_DYNAMIC_MASS_FILTER,
        min_choice_mass=QUESTIONNAIRE_MIN_CHOICE_MASS,
        min_trait_coverage=QUESTIONNAIRE_MIN_TRAIT_COVERAGE,
        reset_mode=QUESTIONNAIRE_RESET_MODE,
        soft_reset_system_prompt=QUESTIONNAIRE_SOFT_RESET_SYSTEM_PROMPT,
        boundary_token=QUESTIONNAIRE_BOUNDARY_TOKEN,
        max_context_tokens=QUESTIONNAIRE_MAX_CONTEXT_TOKENS,
        context_buffer_tokens=QUESTIONNAIRE_CONTEXT_BUFFER_TOKENS,
        write_inspection_file=WRITE_QUESTIONNAIRE_INSPECTION_FILE,
        inspection_items_per_rollout=INSPECTION_ITEMS_PER_ROLLOUT,
    )


def _build_trait_scoring_stage_config(ctx: RunContext) -> TraitScoringStageConfig:
    return TraitScoringStageConfig(
        ctx=ctx,
        min_trait_coverage=QUESTIONNAIRE_MIN_TRAIT_COVERAGE,
    )


def _build_realism_judge_stage_config(ctx: RunContext) -> RealismJudgeStageConfig:
    return RealismJudgeStageConfig(
        ctx=ctx,
        model=REALISM_JUDGE_MODEL,
        provider=REALISM_JUDGE_PROVIDER,
        max_tokens=REALISM_JUDGE_MAX_TOKENS,
        temperature=REALISM_JUDGE_TEMPERATURE,
        max_concurrent=REALISM_JUDGE_MAX_CONCURRENT,
        max_message_chars=REALISM_JUDGE_MAX_MESSAGE_CHARS,
    )


def _build_factor_analysis_stage_config(ctx: RunContext) -> FactorAnalysisStageConfig:
    return FactorAnalysisStageConfig(
        ctx=ctx,
        method=FA_METHOD,
        n_factors_override=FA_N_FACTORS_OVERRIDE,
        rotations=tuple(FA_ROTATIONS),
        residualize_options=tuple(RESIDUALIZE_OPTIONS),
        min_item_variance=MIN_ITEM_VARIANCE,
        high_variance_persona_drop_pct=HIGH_VARIANCE_PERSONA_DROP_PCT,
        fa_blocks=tuple(FA_BLOCKS),
        fa_per_block_passes=FA_PER_BLOCK_PASSES,
        fc_pair_sign_alignment=FC_PAIR_SIGN_ALIGNMENT,
    )


def _build_labeling_stage_config(ctx: RunContext) -> LabelingStageConfig:
    return LabelingStageConfig(
        ctx=ctx,
        mode=LABELLER_MODE,
        model=LABELLER_MODEL,
        provider=LABELLER_PROVIDER,
        top_loading_items=TOP_LOADING_ITEMS,
        max_new_tokens=LABELLER_MAX_NEW_TOKENS,
        empty_response_retries=LABELLER_EMPTY_RESPONSE_RETRIES,
        reasoning=LABELLER_REASONING,
        use_claude_cli=LABELLER_USE_CLAUDE_CLI,
        claude_cli_path=LABELLER_CLAUDE_CLI_PATH,
        claude_cli_model=LABELLER_CLAUDE_CLI_MODEL,
        claude_cli_timeout=LABELLER_CLAUDE_CLI_TIMEOUT,
        claude_cli_effort=LABELLER_CLAUDE_CLI_EFFORT,
    )


def _build_validation_stage_config(ctx: RunContext) -> ValidationStageConfig:
    return ValidationStageConfig(
        ctx=ctx,
        tests_to_run=frozenset(VALIDATION_TESTS_TO_RUN),
        stability_n_prompts=STABILITY_N_PROMPTS,
        holdout_n_items=HOLDOUT_N_ITEMS,
        holdout_r2_floor=HOLDOUT_R2_FLOOR,
        stability_sweep_n_random_splits=STABILITY_SWEEP_N_RANDOM_SPLITS,
        stability_sweep_loso_top_n=STABILITY_SWEEP_LOSO_TOP_N,
        stability_sweep_pa_iterations=STABILITY_SWEEP_PA_ITERATIONS,
        stability_sweep_pass_threshold_phi=STABILITY_SWEEP_PASS_THRESHOLD_PHI,
        variance_decomp_fields=tuple(VARIANCE_DECOMP_FIELDS),
        variance_decomp_scenario_ceiling=VARIANCE_DECOMP_SCENARIO_CEILING,
        variance_decomp_archetype_floor=VARIANCE_DECOMP_ARCHETYPE_FLOOR,
        trait_convergence_hit_threshold=TRAIT_CONVERGENCE_HIT_THRESHOLD,
        trait_convergence_min_hits=TRAIT_CONVERGENCE_MIN_HITS,
        trait_convergence_n_bootstrap=TRAIT_CONVERGENCE_N_BOOTSTRAP,
        persona_item_cv_split=PERSONA_ITEM_CV_SPLIT,
        persona_item_cv_n_trials=PERSONA_ITEM_CV_N_TRIALS,
        persona_item_cv_subset_strategy=PERSONA_ITEM_CV_SUBSET_STRATEGY,
        persona_item_cv_n_outer_splits=PERSONA_ITEM_CV_N_OUTER_SPLITS,
        persona_item_cv_bootstrap_ci=PERSONA_ITEM_CV_BOOTSTRAP_CI,
        k_sensitivity_match_threshold=K_SENSITIVITY_MATCH_THRESHOLD,
        k_sensitivity_independent_threshold=K_SENSITIVITY_INDEPENDENT_THRESHOLD,
        n_factors_suggest_methods=tuple(N_FACTORS_SUGGEST_METHODS),
        n_factors_suggest_k_max=N_FACTORS_SUGGEST_K_MAX,
        n_factors_suggest_cv_n_folds=N_FACTORS_SUGGEST_CV_N_FOLDS,
        n_factors_suggest_pa_iterations=N_FACTORS_SUGGEST_PA_ITERATIONS,
        bootstrap_loadings_n_boot=BOOTSTRAP_LOADINGS_N_BOOT,
        bootstrap_loadings_confidence=BOOTSTRAP_LOADINGS_CONFIDENCE,
        split_half_congruence_n_iters=SPLIT_HALF_CONGRUENCE_N_ITERS,
        split_half_congruence_pass_threshold_phi=SPLIT_HALF_CONGRUENCE_PASS_THRESHOLD_PHI,
        cv_k_curve_k_max=CV_K_CURVE_K_MAX,
        cv_k_curve_n_folds=CV_K_CURVE_N_FOLDS,
        external_predictivity_n_folds=EXTERNAL_PREDICTIVITY_N_FOLDS,
        external_predictivity_ridge_alpha=EXTERNAL_PREDICTIVITY_RIDGE_ALPHA,
        external_predictivity_pass_r2=EXTERNAL_PREDICTIVITY_PASS_R2,
        external_predictivity_bootstrap_ci=EXTERNAL_PREDICTIVITY_BOOTSTRAP_CI,
    )


# ═════════════════════════════════════════════════════════════════════════════
# ROLLOUT-GENERATION CONFIG CALLBACK (scenario / archetype-aware)
# ═════════════════════════════════════════════════════════════════════════════


def _build_rollout_generation_config(
    run_dir: Path,
    retry_terminal_sample_ids: list[str],
) -> RolloutGenerationConfig:
    """Closure invoked by run_stage_rollouts on cache miss.

    Preserves the script's scenario/archetype preamble: loads scenarios,
    writes the synthetic seed JSONL on first run, pre-ingests the canonical
    dataset, composes per-sample user-simulator templates, and finally
    builds the full RolloutGenerationConfig.
    """
    dataset_config = DatasetConfig(
        source="local",
        path=SEED_DATASET,
        max_samples=MAX_PROMPTS,
        seed=SEED,
    )
    prompt_template_per_sample: dict[str, str] = {}
    user_prompt_template = "persona_elicitation"
    user_sim_max_concurrent = USER_SIM_MAX_CONCURRENT
    mode = _current_user_simulator_mode()

    if mode == "legacy":
        register_user_simulator_template(
            "persona_elicitation", PERSONA_ELICITATION_PROMPT
        )
    elif mode == "scenarios":
        if SCENARIO_FILE is None:
            raise ValueError(
                "SCENARIO_FILE must be set when user_simulator_mode='scenarios'. "
                "Point it at a scenario JSON file (see conversation_scenarios.py)."
            )
        scenarios = load_scenarios(SCENARIO_FILE)
        warnings = validate_scenarios(scenarios)
        for w in warnings:
            print(f"[Stage 1] Scenario warning: {w}")
        print_scenario_summary(scenarios)

        # Build a temporary seed JSONL from scenario × archetype combos so
        # the canonical dataset machinery (ingest_source_dataset) works
        # unchanged. Each combo gets a unique seed line (unique question
        # string → unique content-hash sample_id). The real opening is
        # generated by the user sim from its scenario system prompt.
        run_dir.mkdir(parents=True, exist_ok=True)
        synthetic_seed_path = run_dir / "_synthetic_scenario_seeds.jsonl"
        if not synthetic_seed_path.exists():
            # Build all combos, pre-shuffle with the run seed, pre-select
            # MAX_PROMPTS, then write exactly that many rows with
            # id == file position. Downstream ingest must NOT shuffle again
            # (seed=None in DatasetConfig below).
            all_combos = [(sc, arch) for sc in scenarios for arch in ARCHETYPE_NAMES]
            rng = random.Random(SEED)
            rng.shuffle(all_combos)
            selected_combos = all_combos[:MAX_PROMPTS]
            with open(synthetic_seed_path, "w") as f:
                for idx, (sc, arch) in enumerate(selected_combos):
                    row: dict[str, Any] = {
                        "question": f"[scenario: {sc.id} | archetype: {arch}]",
                        "id": idx,
                        "scenario_id": sc.id,
                        "archetype": arch,
                    }
                    if sc.target_system_prompt:
                        row["system_prompt"] = sc.target_system_prompt
                    json.dump(row, f)
                    f.write("\n")

        dataset_config = DatasetConfig(
            source="local",
            path=str(synthetic_seed_path),
            max_samples=MAX_PROMPTS,
            seed=None,
        )
        seed_dataset = load_dataset_from_config(dataset_config)
        samples = ingest_source_dataset(
            dataset=seed_dataset,
            source_info={
                "dataset_source": "scenarios",
                "scenario_file": SCENARIO_FILE,
                "scenario_set_version": SCENARIO_SET_VERSION,
                "max_samples": MAX_PROMPTS,
            },
            system_prompt=None,
            run_dir=run_dir,
            responses_per_input=NUM_ROLLOUTS_PER_PROMPT,
        )
        prompt_template_per_sample = _build_per_sample_templates_scenarios(
            run_dir, samples, scenarios,
        )
        user_prompt_template = "__unused__"
    else:
        # Archetype mode (seed questions + archetypes, no scenarios).
        run_dir.mkdir(parents=True, exist_ok=True)
        seed_dataset = load_dataset_from_config(dataset_config)
        samples = ingest_source_dataset(
            dataset=seed_dataset,
            source_info={
                "dataset_source": "local",
                "dataset_path": SEED_DATASET,
                "max_samples": MAX_PROMPTS,
            },
            system_prompt=None,
            run_dir=run_dir,
            responses_per_input=NUM_ROLLOUTS_PER_PROMPT,
        )
        prompt_template_per_sample = _build_per_sample_templates_seeds(run_dir, samples)
        user_prompt_template = "__unused__"

    return RolloutGenerationConfig(
        dataset=dataset_config,
        run_dir=run_dir,
        num_assistant_turns=NUM_CONVERSATION_TURNS,
        num_rollouts_per_prompt=NUM_ROLLOUTS_PER_PROMPT,
        system_prompt=None,
        prompt_template_per_sample=prompt_template_per_sample,
        assistant_inference=InferenceConfig(
            model=ASSISTANT_MODEL,
            provider=ASSISTANT_PROVIDER,
            generation=GenerationConfig(
                max_new_tokens=ASSISTANT_MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=0.95,
                do_sample=True,
                batch_size=ROLLOUT_ASSISTANT_BATCH_SIZE,
            ),
            max_concurrent=ROLLOUT_MAX_CONCURRENT,
            timeout=QUESTIONNAIRE_TIMEOUT,
            retry=RetryConfig(max_retries=3, backoff_factor=2.0),
            openrouter=OpenRouterProviderConfig(
                provider_routing=ASSISTANT_OPENROUTER_PROVIDER_ROUTING,
            ),
        ),
        user_simulator=UserSimulatorConfig(
            provider=USER_PROVIDER,
            model=USER_MODEL,
            prompt_template=user_prompt_template,
            prompt_format="chat_messages",
            flip_mode="interlocutor",
            turn_reminder=None,
            max_context_turns=None,
            generation=GenerationConfig(
                max_new_tokens=USER_MAX_NEW_TOKENS,
                temperature=0.7,
                do_sample=True,
            ),
            max_concurrent=user_sim_max_concurrent,
            timeout=QUESTIONNAIRE_TIMEOUT,
            retry=RetryConfig(max_retries=3, backoff_factor=2.0),
            openrouter=OpenRouterProviderConfig(reasoning={"effort": "medium"}),
        ),
        user_sim_generates_opening=True,
        resume=True,
        overwrite_output=False,
        retry_terminal_sample_ids=retry_terminal_sample_ids,
    )


# ═════════════════════════════════════════════════════════════════════════════
# MULTI-PAIR COMBINATION (Stage 2.9, multi-preset mode only)
# ═════════════════════════════════════════════════════════════════════════════


def _load_pair_outputs(
    rollout_key: str, q_key: str
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Load response_matrix / metadata / items for a single (rollout, q) pair.

    Errors loudly if any artifact is missing — presets assume existence.
    """
    q_dir = _questionnaire_dir(rollout_key, q_key) / "questionnaire"
    matrix_path = q_dir / "response_matrix.npy"
    meta_path = q_dir / "metadata.jsonl"
    items_path = q_dir / "items.json"
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"[Combine] Missing response_matrix for pair "
            f"(rollout={rollout_key!r}, questionnaire={q_key!r}) at {matrix_path}. "
            "Run Stage 1+2 for this pair first, or remove it from the selectors."
        )
    if not meta_path.exists():
        raise FileNotFoundError(f"[Combine] Missing metadata at {meta_path}")
    if not items_path.exists():
        raise FileNotFoundError(f"[Combine] Missing items.json at {items_path}")

    matrix = np.load(matrix_path)
    with meta_path.open("r", encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f if line.strip()]
    with items_path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    return matrix, metadata, items


def _combine_per_pair_outputs(
    pairs: list[tuple[str, str]],
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Concatenate rows across rollouts and union columns across questionnaire
    versions into a single response matrix for Stage 3+.

    Columns are grouped by the *underlying questionnaire version* (the
    ``version`` field on ``QuestionnairePreset``), so presets that share a
    version (e.g. ``v6_fc_draft`` and ``v6_fc_draft_direct``) pool into a
    single column block. Per-row alignment within a rollout uses sample_id
    intersection across all its paired questionnaires.

    Writes artifacts to ``_combined_dir()/questionnaire/`` (local only).
    """
    out_dir = _combined_dir() / "questionnaire"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load every pair and resolve each pair to its underlying version.
    pair_data: dict[tuple[str, str], tuple[np.ndarray, list[dict], list[dict]]] = {}
    pair_version: dict[tuple[str, str], str] = {}
    for r_key, q_key in pairs:
        pair_data[(r_key, q_key)] = _load_pair_outputs(r_key, q_key)
        pair_version[(r_key, q_key)] = QUESTIONNAIRE_PRESETS[q_key].version

    rollout_keys: list[str] = []
    seen_r: set[str] = set()
    for r, _ in pairs:
        if r not in seen_r:
            seen_r.add(r)
            rollout_keys.append(r)

    versions: list[str] = []
    seen_v: set[str] = set()
    for p in pairs:
        v = pair_version[p]
        if v not in seen_v:
            seen_v.add(v)
            versions.append(v)

    # For each (rollout, version), pick the preset key the user paired for it.
    # Require exactly one per (rollout, version); error on duplicates/missing.
    pair_for: dict[tuple[str, str], str] = {}
    for (r_key, q_key), v in pair_version.items():
        key = (r_key, v)
        if key in pair_for:
            raise RuntimeError(
                f"[Combine] rollout {r_key!r} has multiple pairs for version "
                f"{v!r}: {pair_for[key]!r} and {q_key!r}. Pick one."
            )
        pair_for[key] = q_key
    for r_key in rollout_keys:
        for v in versions:
            if (r_key, v) not in pair_for:
                raise RuntimeError(
                    f"[Combine] rollout {r_key!r} has no pair for version "
                    f"{v!r}. Every rollout must cover every selected version."
                )

    # 2) Per rollout, intersect sample_ids across its paired questionnaires.
    per_rollout_sids: dict[str, list[str]] = {}
    for r_key in rollout_keys:
        rollout_q_keys = [pair_for[(r_key, v)] for v in versions]
        sids_per_q: list[set[str]] = []
        for q_key in rollout_q_keys:
            _, meta, _ = pair_data[(r_key, q_key)]
            sids_per_q.append({m["sample_id"] for m in meta if m.get("sample_id")})
        common = set.intersection(*sids_per_q) if sids_per_q else set()
        _, first_meta, _ = pair_data[(r_key, rollout_q_keys[0])]
        ordered = [m["sample_id"] for m in first_meta if m.get("sample_id") in common]
        if not ordered:
            raise RuntimeError(
                f"[Combine] No shared sample_ids for rollout {r_key!r} across "
                f"its paired questionnaires {rollout_q_keys!r}."
            )
        per_rollout_sids[r_key] = ordered
        n_dropped = len(first_meta) - len(ordered)
        if n_dropped:
            print(
                f"[Combine] rollout={r_key!r}: kept {len(ordered)} rows, "
                f"dropped {n_dropped} without responses in every paired questionnaire"
            )

    # 3) Build per-version column blocks, namespaced by version.
    q_items_combined: list[dict] = []
    v_col_counts: dict[str, int] = {}
    for v in versions:
        source_r = rollout_keys[0]
        source_q = pair_for[(source_r, v)]
        _, _, items = pair_data[(source_r, source_q)]
        v_col_counts[v] = len(items)
        src_item_ids = [it.get("item_id", it.get("id")) for it in items]
        for r_key in rollout_keys[1:]:
            _, _, other_items = pair_data[(r_key, pair_for[(r_key, v)])]
            other_ids = [it.get("item_id", it.get("id")) for it in other_items]
            if other_ids != src_item_ids:
                raise RuntimeError(
                    f"[Combine] item-order mismatch for version {v!r} between "
                    f"rollouts {source_r!r} and {r_key!r}. Re-score one of the "
                    f"caches so they use a consistent item order."
                )
        for it in items:
            namespaced = dict(it)
            orig_item_id = it.get("item_id", it.get("id"))
            orig_col_id = it.get("col_id", orig_item_id)
            namespaced["item_id"] = f"{v}/{orig_item_id}"
            namespaced["col_id"] = f"{v}/{orig_col_id}"
            if "id" in it:
                namespaced["id"] = f"{v}/{it['id']}"
            namespaced["questionnaire_version"] = v
            q_items_combined.append(namespaced)

    # 4) Assemble the combined matrix block-by-block.
    n_total_rows = sum(len(per_rollout_sids[r]) for r in rollout_keys)
    n_total_cols = sum(v_col_counts[v] for v in versions)
    combined = np.full((n_total_rows, n_total_cols), np.nan, dtype=float)

    row_offset = 0
    combined_metadata: list[dict] = []
    for r_key in rollout_keys:
        sids = per_rollout_sids[r_key]
        sid_to_row = {sid: i for i, sid in enumerate(sids)}

        first_v = versions[0]
        _, base_meta, _ = pair_data[(r_key, pair_for[(r_key, first_v)])]
        base_by_sid = {m["sample_id"]: m for m in base_meta if m.get("sample_id")}
        for sid in sids:
            row = dict(base_by_sid[sid])
            row["rollout_preset_key"] = r_key
            row["version_sources"] = {v: pair_for[(r_key, v)] for v in versions}
            combined_metadata.append(row)

        col_offset = 0
        for v in versions:
            q_key = pair_for[(r_key, v)]
            matrix, meta, _ = pair_data[(r_key, q_key)]
            n_cols = v_col_counts[v]
            if matrix.shape[1] != n_cols:
                raise RuntimeError(
                    f"[Combine] column count mismatch for version {v!r} in "
                    f"rollout {r_key!r} / preset {q_key!r}: "
                    f"matrix has {matrix.shape[1]} cols, expected {n_cols}."
                )
            sid_to_src_row = {m["sample_id"]: i for i, m in enumerate(meta) if m.get("sample_id")}
            for sid, dst_idx in sid_to_row.items():
                src_idx = sid_to_src_row.get(sid)
                if src_idx is None:
                    raise RuntimeError(
                        f"[Combine] sample_id {sid!r} missing from "
                        f"(rollout={r_key!r}, preset={q_key!r}) — intersection bug?"
                    )
                combined[row_offset + dst_idx, col_offset:col_offset + n_cols] = matrix[src_idx]
            col_offset += n_cols
        row_offset += len(sids)

    # 5) Report residual NaN coverage.
    n_nan = int(np.isnan(combined).sum())
    if n_nan:
        frac = n_nan / combined.size
        print(
            f"[Combine] {n_nan} NaN cells in combined matrix "
            f"({frac:.2%} of {combined.size} — will be imputed in Stage 3)"
        )

    # 6) Persist (locally only — no HF upload for combined runs).
    np.save(out_dir / "response_matrix.npy", combined)
    with (out_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in combined_metadata:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "items.json").open("w", encoding="utf-8") as f:
        json.dump(q_items_combined, f, ensure_ascii=False, indent=2)

    provenance = {
        "rollouts": rollout_keys,
        "versions": versions,
        "pairs": [
            {
                "rollout_preset_key": r_key,
                "questionnaire_preset_key": q_key,
                "questionnaire_version": pair_version[(r_key, q_key)],
                "rollout_run_id": _rollout_run_id(r_key),
                "questionnaire_run_id": _questionnaire_run_id(r_key, q_key),
            }
            for (r_key, q_key) in pair_data.keys()
        ],
    }
    with (out_dir.parent / "provenance.json").open("w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)

    print(
        f"[Combine] Wrote combined ({combined.shape[0]} rows × {combined.shape[1]} cols) "
        f"to {out_dir}"
    )
    return combined, combined_metadata, q_items_combined


# ═════════════════════════════════════════════════════════════════════════════
# PER-PAIR CONFIG LOGGING + FILTERS
# ═════════════════════════════════════════════════════════════════════════════


def _write_pair_config(
    rollout_key: str,
    q_key: str,
    *,
    args,
    retry_mode: str,
    retry_terminal_sample_ids: list[str],
) -> None:
    """Write per-pair config.json (called inside Cartesian loop)."""
    config_dir = _questionnaire_dir(rollout_key, q_key)
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_dir / "config.json", "w") as f:
        json.dump({
            "rollout_preset_key": rollout_key,
            "questionnaire_preset_key": q_key,
            "seed": SEED,
            "rollout_run_id": _rollout_run_id(rollout_key),
            "questionnaire_run_id": _questionnaire_run_id(rollout_key, q_key),
            "assistant_model": ASSISTANT_MODEL,
            "user_model": USER_MODEL,
            "temperature": TEMPERATURE,
            "num_conversation_turns": NUM_CONVERSATION_TURNS,
            "max_prompts": MAX_PROMPTS,
            "num_rollouts_per_prompt": NUM_ROLLOUTS_PER_PROMPT,
            "questionnaire_version": QUESTIONNAIRE_VERSION,
            "questionnaire_format": "hybrid (FC + vignettes + Likert)",
            "fa_blocks": FA_BLOCKS,
            "questionnaire_phrasing_likert": QUESTIONNAIRE_PHRASING,
            "fa_method": FA_METHOD,
            "fa_rotations": FA_ROTATIONS,
            "residualize_options": RESIDUALIZE_OPTIONS,
            "retry_terminal_samples_mode": retry_mode,
            "retry_terminal_samples_file": (
                str(args.retry_terminal_samples_file)
                if args.retry_terminal_samples_file is not None
                else None
            ),
            "retry_terminal_sample_count": len(retry_terminal_sample_ids),
            **_user_simulator_mode_metadata(),
        }, f, indent=2)


def _apply_consecutive_turn_filter(
    response_matrix: np.ndarray,
    metadata: list[dict],
) -> tuple[np.ndarray, list[dict]]:
    """Strip rows whose source rollout has the consecutive-assistant-turns
    resume-bug artifact. Unions bad sample_ids across every selected rollout
    preset, so multi-rollout combined matrices are filtered correctly."""
    bad_sample_ids: set[str] = set()
    for r_key in _resolved_rollouts():
        bad_sample_ids |= find_consecutive_assistant_turn_sample_ids(_rollout_dir(r_key))
    if not bad_sample_ids:
        return response_matrix, metadata
    keep = np.array([m["sample_id"] not in bad_sample_ids for m in metadata])
    n_removed = int((~keep).sum())
    if n_removed:
        response_matrix = response_matrix[keep]
        metadata = [m for m, k in zip(metadata, keep) if k]
        print(
            f"Excluded {n_removed} samples with consecutive assistant turns "
            f"(resume-bug artifact) from downstream analysis"
        )
    return response_matrix, metadata


def _write_questionnaire_inspection_file(items: list[dict]) -> None:
    """Write a JSONL file joining rollout conversations with questionnaire responses.

    Each row is one (rollout, item) pair with the full conversation + the
    questionnaire question and answer appended as the final two messages.
    View with:
        uv run python -m src_dev.jsonl_tui.cli <path> --conversation-field messages
    """
    q_dir = _questionnaire_dir() / "questionnaire"
    r_file = _rollout_dir() / "exports" / "conversation_training.jsonl"

    if not r_file.exists() or not (q_dir / "raw_responses.jsonl").exists():
        print("[Inspection] Missing rollout or questionnaire files — skipping.")
        return

    item_by_id = {it["id"]: it for it in items}

    responses_by_k: dict[int, list[dict]] = defaultdict(list)
    with open(q_dir / "raw_responses.jsonl", "r") as f:
        for line in f:
            row = json.loads(line)
            responses_by_k[row["k"]].append(row)

    with open(r_file, "r") as f:
        rollouts = [json.loads(line) for line in f]

    out = q_dir / "conversations_with_questionnaire.jsonl"
    n_written = 0
    with open(out, "w", encoding="utf-8") as f:
        for k, rollout in enumerate(rollouts):
            for resp in responses_by_k.get(k, [])[:INSPECTION_ITEMS_PER_ROLLOUT]:
                item = item_by_id.get(resp["item_id"])
                if item is None:
                    continue
                prompt_text = build_item_prompt(item, likert_phrasing=QUESTIONNAIRE_PHRASING)
                # Human-readable description of the item for display
                if item["type"] == "forced_choice":
                    display_text = f'FC: {item["option_a"]["text"][:60]}... vs {item["option_b"]["text"][:60]}...'
                elif item["type"] == "vignette":
                    display_text = f'Vignette [{item["title"]}]: {item["scenario"][:80]}...'
                elif item["type"] == "trait_mcq":
                    display_text = f'TRAIT [{item["primary_dimension"]}]: {item["question"][:80]}...'
                elif item["type"] == "fc_pair":
                    option_by_label = {o["label"]: o["text"] for o in item["options"]}
                    display_text = (
                        f'fc_pair [{item.get("axis", "?")}]: {item["stem"][:60]}... | '
                        f'A: {option_by_label.get("A", "")[:40]}... | '
                        f'B: {option_by_label.get("B", "")[:40]}...'
                    )
                else:
                    display_text = item["text"]
                msgs = list(rollout["messages"])
                msgs.append({"role": "user", "content": prompt_text})
                choice = resp.get("parsed_choice")
                raw = resp.get("raw", "")
                answer_text = raw if raw else (str(choice) if choice is not None else "(no response)")
                msgs.append({"role": "assistant", "content": answer_text})
                row = {
                    "sample_id": rollout.get("sample_id", ""),
                    "item_id": item["id"],
                    "item_type": item["type"],
                    "item_text": display_text,
                    "parsed": choice,
                    "messages": msgs,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"[Inspection] Wrote {n_written} rows to {out}")
    print(f"  View with: uv run python -m src_dev.jsonl_tui.cli {out} --conversation-field messages")

    # Also write a self-contained HTML file for sharing.
    html_path = q_dir / "conversations_with_questionnaire.html"
    write_conversation_html(out, html_path)
    print(f"  HTML export: {html_path}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    args = _parse_args()

    # Honour --user-simulator-mode as a global override if provided. Leaving
    # it unset means every preset uses its own .user_simulator_mode.
    cli_mode_override = args.user_simulator_mode

    # --retry-terminal-samples{,-file} was designed around a single rollout
    # directory. We keep it working in single-rollout mode and hard-fail in
    # multi-rollout mode rather than silently retrying only one of the sets.
    resolved_rollouts = _resolved_rollouts()
    resolved_pairs = _resolved_pairs()
    if (args.retry_terminal_samples or args.retry_terminal_samples_file) and len(resolved_rollouts) > 1:
        print(
            "ERROR: --retry-terminal-samples{,-file} is not supported in multi-rollout mode. "
            "Run with a single-rollout selector for retries."
        )
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    print("=" * 60)
    print("Psychometric Factor Analysis of LLM Persona Rollouts")
    print("=" * 60)
    print(f"Rollout presets selected:      {resolved_rollouts}")
    print(f"Questionnaire presets selected: {_resolved_questionnaires()}")
    print(f"Pairs:                          {resolved_pairs}")
    print(f"Multi-preset mode:              {_is_multi_preset()}")
    if _is_multi_preset():
        print(f"Combined run ID:                {_combined_run_id()}")
    print(f"Stages to run:                  {STAGES_TO_RUN}")
    print()

    # ── Stages 1 & 2 (+ per-pair 2b / inspection): Cartesian loop ────────
    # Stage configs read from the preset globals at build-time, so we
    # activate each preset before constructing its configs. In single-pair
    # mode this loop runs exactly once.
    last_response_matrix: np.ndarray | None = None
    last_metadata: list[dict] | None = None
    last_column_defs: list[dict] | None = None

    # Group resolved pairs by rollout so Stage 1 runs once per rollout.
    q_keys_by_rollout: dict[str, list[str]] = {}
    for r_key, q_key in resolved_pairs:
        q_keys_by_rollout.setdefault(r_key, []).append(q_key)

    for r_key in resolved_rollouts:
        _activate_rollout(r_key)
        if cli_mode_override is not None:
            global ACTIVE_USER_SIMULATOR_MODE
            ACTIVE_USER_SIMULATOR_MODE = cli_mode_override

        # Per-rollout retry terminal sample loading.
        retry_terminal_sample_ids: list[str] = []
        retry_mode = "off"
        if args.retry_terminal_samples:
            retry_terminal_sample_ids = _load_terminal_sample_ids_from_run(_rollout_dir(r_key))
            retry_mode = "auto"
        elif args.retry_terminal_samples_file is not None:
            retry_terminal_sample_ids = _load_retry_terminal_sample_ids(
                args.retry_terminal_samples_file
            )
            retry_mode = "file"

        if "rollouts" in STAGES_TO_RUN:
            print("\n" + "=" * 60)
            print(f"[Stage 1] Generating rollouts — preset {r_key!r}")
            print("=" * 60)
            # For Stage 1, ctx uses a placeholder questionnaire dir (first
            # paired q_key) — Stage 1 doesn't touch the questionnaire dir.
            first_q_for_rollout = q_keys_by_rollout[r_key][0]
            _activate_questionnaire(first_q_for_rollout)
            r_ctx = _build_run_context(r_key, first_q_for_rollout)
            r_cfg = _build_rollouts_stage_config(r_ctx, retry_terminal_sample_ids)
            run_stage_rollouts(
                r_cfg,
                build_rollout_config=_build_rollout_generation_config,
            )

        for q_key in q_keys_by_rollout[r_key]:
            _activate_questionnaire(q_key)
            print("\n" + "#" * 60)
            print(f"# Pair: rollout={r_key!r} × questionnaire={q_key!r}")
            print(f"#   {_questionnaire_run_id(r_key, q_key)}")
            print("#" * 60)

            _write_pair_config(
                r_key, q_key,
                args=args,
                retry_mode=retry_mode,
                retry_terminal_sample_ids=retry_terminal_sample_ids,
            )

            # Build ctx + configs for this pair. effective_dir is the per-
            # pair dir here (Stage 2 / 2b work on one pair at a time).
            pair_ctx = _build_run_context(r_key, q_key)

            # Load items once so we can pass them to inspection + downstream
            # (Stage 2's own internal load is separate but yields the same
            # objects given the fixed inputs).
            questionnaire_items, _qcol_defs = load_questionnaire(
                QUESTIONNAIRE_PATH,
                fa_blocks=FA_BLOCKS,
                fc_pair_sign_alignment=FC_PAIR_SIGN_ALIGNMENT,
            )

            if "questionnaire" in STAGES_TO_RUN:
                print("\n" + "=" * 60)
                print(f"[Stage 2] Applying questionnaire — {r_key!r} × {q_key!r}")
                print("=" * 60)
                q_cfg = _build_questionnaire_stage_config(pair_ctx)
                q_res = run_stage_questionnaire(
                    q_cfg,
                    num_conversation_turns=NUM_CONVERSATION_TURNS,
                    openrouter_provider_routing=ASSISTANT_OPENROUTER_PROVIDER_ROUTING,
                    fc_pair_sign_alignment=FC_PAIR_SIGN_ALIGNMENT,
                )
                # Reload the saved artifacts so we carry them into Stage 3+.
                q_out = q_res.questionnaire_dir
                last_response_matrix = np.load(q_out / "response_matrix.npy")
                with open(q_out / "metadata.jsonl", "r") as f:
                    last_metadata = [json.loads(line) for line in f if line.strip()]
                if (q_out / "items.json").exists():
                    with open(q_out / "items.json", "r") as f:
                        last_column_defs = json.load(f)
                else:
                    last_column_defs = _qcol_defs

            # Skip the per-pair inspection writer in multi-pair mode: it
            # builds a ~GB HTML string for 75000 rows and has OOM'd between
            # pairs. Single-pair runs keep the behaviour.
            if WRITE_QUESTIONNAIRE_INSPECTION_FILE and not _is_multi_preset():
                _write_questionnaire_inspection_file(questionnaire_items)

            if "trait_scoring" in STAGES_TO_RUN:
                print("\n" + "=" * 60)
                print(f"[Stage 2.5] Trait scoring — {r_key!r} × {q_key!r}")
                print("=" * 60)
                ts_cfg = _build_trait_scoring_stage_config(pair_ctx)
                run_stage_trait_scoring(
                    ts_cfg,
                    questionnaire_path=Path(QUESTIONNAIRE_PATH),
                    dynamic_mass_filter=QUESTIONNAIRE_DYNAMIC_MASS_FILTER,
                    min_choice_mass=QUESTIONNAIRE_MIN_CHOICE_MASS,
                    title_prefix=_questionnaire_run_id(r_key, q_key),
                    metadata=last_metadata,
                )

            if "realism_judge" in STAGES_TO_RUN:
                print("\n" + "=" * 60)
                print(f"[Stage 2b] Realism judge — {r_key!r}")
                print("=" * 60)
                rj_cfg = _build_realism_judge_stage_config(pair_ctx)
                run_stage_realism_judge(
                    rj_cfg,
                    num_conversation_turns=NUM_CONVERSATION_TURNS,
                )

    # ── Stage 2.9: Combine (multi-preset only) ───────────────────────────
    response_matrix: np.ndarray | None
    metadata: list[dict] | None
    column_defs: list[dict] | None

    # Re-activate the first selected pair so preset-dependent helpers
    # (scenario/archetype metadata, run-id strings) resolve against a
    # known-good preset in single-pair mode.
    _activate_rollout(resolved_pairs[0][0])
    _activate_questionnaire(resolved_pairs[0][1])

    if _is_multi_preset():
        print("\n" + "=" * 60)
        print("[Stage 2.9] Combining per-pair outputs")
        print("=" * 60)
        response_matrix, metadata, column_defs = _combine_per_pair_outputs(resolved_pairs)
        # Downstream stages use ``column_defs`` as ``items`` — the two lists
        # are aligned (namespaced item_ids) in multi-pair mode.
        questionnaire_items = column_defs
    else:
        response_matrix = last_response_matrix
        metadata = last_metadata
        column_defs = last_column_defs
        questionnaire_items, _ = load_questionnaire(
            QUESTIONNAIRE_PATH,
            fa_blocks=FA_BLOCKS,
            fc_pair_sign_alignment=FC_PAIR_SIGN_ALIGNMENT,
        )

    # Load if needed for later stages (single-pair path only; multi-pair
    # always populated the in-memory vars above).
    if response_matrix is None and any(
        s in STAGES_TO_RUN for s in ["factor_analysis", "labeling", "validation"]
    ):
        q_dir = _effective_dir() / "questionnaire"
        matrix_path = q_dir / "response_matrix.npy"
        if matrix_path.exists():
            response_matrix = np.load(matrix_path)
            with open(q_dir / "metadata.jsonl", "r") as f:
                metadata = [json.loads(line) for line in f]
            items_path = q_dir / "items.json"
            if items_path.exists():
                with open(items_path, "r") as f:
                    column_defs = json.load(f)
            else:
                _, column_defs = load_questionnaire(
                    QUESTIONNAIRE_PATH,
                    fa_blocks=FA_BLOCKS,
                    fc_pair_sign_alignment=FC_PAIR_SIGN_ALIGNMENT,
                )
        else:
            print("ERROR: Questionnaire results not found. Run stages 1-2 first.")
            sys.exit(1)

    # Filter out samples with consecutive assistant turns (resume-bug artifact).
    if response_matrix is not None and metadata is not None:
        response_matrix, metadata = _apply_consecutive_turn_filter(response_matrix, metadata)

    # ── Build downstream-stage ctx (reads from _effective_dir) ───────────
    # For Stage 3-5 we want effective_questionnaire_dir to point at the
    # combined dir in multi-pair mode; in single-pair mode it's the same
    # as the per-pair dir.
    downstream_ctx = _build_run_context(
        resolved_pairs[0][0], resolved_pairs[0][1],
        effective_dir=_effective_dir(),
    )

    # Rollout dirs for HTML export + variance-decomp lookups.
    rollout_dirs = [_rollout_dir(r_key) for r_key in resolved_rollouts]

    # ── Stage 3 ──────────────────────────────────────────────────────────
    fa_results: dict | None = None
    if "factor_analysis" in STAGES_TO_RUN:
        print("\n" + "=" * 60)
        print("[Stage 3] Factor analysis")
        print("=" * 60)
        fa_cfg = _build_factor_analysis_stage_config(downstream_ctx)
        fa_stage_result = run_stage_factor_analysis(
            fa_cfg,
            response_matrix, metadata, column_defs,
            items=questionnaire_items,
            seed=SEED,
            questionnaire_path=Path(QUESTIONNAIRE_PATH),
            rollout_dirs=rollout_dirs,
            labeling_dir=None,
            n_factors_suggest_methods=tuple(N_FACTORS_SUGGEST_METHODS),
            n_factors_suggest_k_max=N_FACTORS_SUGGEST_K_MAX,
            n_factors_suggest_cv_n_folds=N_FACTORS_SUGGEST_CV_N_FOLDS,
            n_factors_suggest_pa_iterations=N_FACTORS_SUGGEST_PA_ITERATIONS,
        )
        fa_results = fa_stage_result.results_by_rotation

    # ── Stage 4 ──────────────────────────────────────────────────────────
    if "labeling" in STAGES_TO_RUN:
        print("\n" + "=" * 60)
        print("[Stage 4] Factor labeling")
        print("=" * 60)
        if fa_results is None:
            print("ERROR: Factor analysis results not available. Run stage 3 first.")
            sys.exit(1)
        lbl_cfg = _build_labeling_stage_config(downstream_ctx)
        run_stage_labeling(
            lbl_cfg, fa_results, questionnaire_items,
            rollout_dirs=rollout_dirs,
        )

    # ── Stage 5 ──────────────────────────────────────────────────────────
    if "validation" in STAGES_TO_RUN:
        print("\n" + "=" * 60)
        print("[Stage 5] Validation")
        print("=" * 60)
        if fa_results is None:
            print("ERROR: Factor analysis results not available. Run stage 3 first.")
            sys.exit(1)
        val_cfg = _build_validation_stage_config(downstream_ctx)
        run_stage_validation(
            val_cfg,
            response_matrix, metadata, column_defs, fa_results,
            seed=SEED,
            num_rollouts_per_prompt=NUM_ROLLOUTS_PER_PROMPT,
            fa_method=FA_METHOD,
            rotations=tuple(FA_ROTATIONS),
            min_item_variance=MIN_ITEM_VARIANCE,
            high_variance_persona_drop_pct=HIGH_VARIANCE_PERSONA_DROP_PCT,
            rollout_dirs=rollout_dirs,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
