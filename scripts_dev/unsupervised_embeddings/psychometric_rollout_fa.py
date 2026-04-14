"""Psychometric factor analysis of LLM persona rollouts.

Applies standard psychometric techniques to discover latent behavioral dimensions
in LLMs. Creates a "population" of personas via diverse multi-turn rollouts,
administers a hybrid questionnaire (forced-choice pairs, behavioral vignettes,
and Likert items) to each persona at the N+1th turn, and runs factor analysis
(Horn's parallel analysis + PAF) on the resulting response matrix. Factors are
discovered unsupervised and labeled post-hoc.

Stages:
    1. rollouts       — Generate diverse multi-turn conversations (one per persona)
    2. questionnaire   — Branch each rollout and administer hybrid questionnaire
    3. factor_analysis — Parallel analysis + PAF on the response matrix
    4. labeling        — Interpret factors via column loadings and LLM labeling
    5. validation      — Stability, predictivity, and shuffle-control tests
"""

from __future__ import annotations

# ── Seeds (set before any stochastic imports) ────────────────────────────────
import random

import numpy as np

SEED = 432  # Production run: exhaustive scenario×archetype combos
random.seed(SEED)
np.random.seed(SEED)

# ── Standard library ─────────────────────────────────────────────────────────
import asyncio
import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ── Third-party ──────────────────────────────────────────────────────────────
from dotenv import load_dotenv

load_dotenv()

# ── Archetype prompts (sibling module) ───────────────────────────────────────
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

# ── Local imports ────────────────────────────────────────────────────────────
from src_dev.common.config import DatasetConfig, GenerationConfig
from src_dev.common.conversation_runtime import chunked
from src_dev.datasets import (
    find_consecutive_assistant_turn_sample_ids,
    ingest_source_dataset,
    load_dataset_from_config,
    load_samples,
    materialize_canonical_samples,
)
from src_dev.factor_analysis.factor_analysis import adequacy_tests, run_factor_analysis
from src_dev.factor_analysis.parallel_analysis import parallel_analysis
from src_dev.factor_analysis.persistence import save_factor_analysis
from src_dev.factor_analysis.preprocessing import residualize
from src_dev.factor_analysis.trait_alignment import (
    build_trait_oriented_matrix,
    compute_factor_trait_alignment,
    plot_all_alignment,
    save_alignment,
)
from src_dev.inference import InferenceConfig
from src_dev.inference.config import OpenRouterProviderConfig, RetryConfig, VllmProviderConfig
from src_dev.inference.providers import get_provider
from src_dev.inference.providers.base import PromptInput
from src_dev.rollout_generation.config import (
    RolloutGenerationConfig,
    UserSimulatorConfig,
)
from src_dev.rollout_generation.prompts import register_user_simulator_template
from src_dev.rollout_generation.run import run_rollout_generation
from src_dev.unsupervised_runs.io import hydrate_dataset_subtree
from src_dev.utils.hf_hub import (
    check_exists_in_dataset_repo,
    login_from_env,
    upload_folder_to_dataset_repo,
)

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

SCRATCH_ROOT = Path("scratch/psychometric_fa")
HF_REPO_ID = "persona-shattering-lasr/psychometric-fa-runs"

# ── Stage 1: Rollout generation ──────────────────────────────────────────────
SEED_DATASET = "datasets/psychometric_seed_prompts/v1xAA.jsonl"
MAX_PROMPTS = 1000  # All 100 scenarios × 10 archetypes
NUM_ROLLOUTS_PER_PROMPT = 2  # 2 rollouts per combo → 2000 total
NUM_CONVERSATION_TURNS = 10
# For cross-model replication sweeps, pick one of the candidates below and
# rerun the pipeline (a fresh _rollout_run_id() is derived automatically).
# Candidates for the cross-model validation sweep (7-8B instruction-tuned):
#   "meta-llama/llama-3.1-8b-instruct"
#   "qwen/qwen2.5-7b-instruct"
#   "mistralai/mistral-7b-instruct-v0.3"
#   "meta-llama/llama-3.1-70b-instruct"   # larger; cost gate
ASSISTANT_MODEL = "meta-llama/llama-3.1-8b-instruct"
ASSISTANT_PROVIDER = "openrouter"
# ASSISTANT_PROVIDER = "vllm"
ASSISTANT_OPENROUTER_PROVIDER_ROUTING = {
    # "only": ["deepinfra"],
    "quantizations": ["bf16"],
    # "allow_fallbacks": False,
}
USER_MODEL = "z-ai/glm-4.7-flash"
USER_PROVIDER = "openrouter"
TEMPERATURE = 1.0
ASSISTANT_MAX_NEW_TOKENS = 4096
USER_MAX_NEW_TOKENS = 4096
DEFAULT_USER_SIMULATOR_MODE = "scenarios"
ACTIVE_USER_SIMULATOR_MODE = DEFAULT_USER_SIMULATOR_MODE
LEGACY_USER_PROMPT_VERSION = "v3"
# Bump when changing archetype prompts or assignment strategy (invalidates HF cache).
ARCHETYPE_SET_VERSION = "v9"  # Bumped for exhaustive scenario×archetype combos
# ── Scenario mode config ────────────────────────────────────────────────────
# Path to a scenario JSON file (see conversation_scenarios.py for the spec).
# Set to None to fall back to seed prompts (archetypes or legacy mode).
SCENARIO_FILE: str | None = "datasets/scenarios/v1.json"
# Bump when changing the scenario file or archetype set for scenario mode.
SCENARIO_SET_VERSION = "v1"
# Local/vLLM-only assistant batch size. Remote assistant providers use
# `ROLLOUT_MAX_CONCURRENT` via the rollout scheduler's shared async limiter.
ROLLOUT_ASSISTANT_BATCH_SIZE = 32
ROLLOUT_MAX_CONCURRENT = 64
USER_SIM_MAX_CONCURRENT = 64

# ── Stage 2: Questionnaire ──────────────────────────────────────────────────
QUESTIONNAIRE_PATH = "datasets/psychometric_questionnaires/psychometric_questionnaire_v5.json"
QUESTIONNAIRE_VERSION = "v5"  # bump when changing items
# TRAIT-benchmark questionnaire: 20 items per OCEAN trait (100 total),
# A/B/C/D options shuffled per item. Scored both via unsupervised FA (integer
# 1..4 encoding) and via TRAIT's canonical answer_mapping (see
# src_dev.factor_analysis.trait_scoring).
# QUESTIONNAIRE_PATH = "datasets/psychometric_questionnaires/trait_ocean_v1.json"
# QUESTIONNAIRE_VERSION = "trait_ocean_v1"  # bump when changing items
QUESTIONNAIRE_PHRASING = "direct"  # "natural", "direct", "contextual" (Likert block only)
LIKERT_SCALE = 5
MAX_PARSE_RETRIES = 3
QUESTIONNAIRE_MAX_CONCURRENT = 32
QUESTIONNAIRE_MAX_NEW_TOKENS = 32
QUESTIONNAIRE_TIMEOUT = 60
# Questionnaire provider/model can differ from rollout generation.
# Set to "vllm" to run locally on GPU with automatic prefix caching.
QUESTIONNAIRE_PROVIDER = "vllm"  # "vllm" for local GPU inference, otherwise can be ASSISTANT_PROVIDER
QUESTIONNAIRE_MODEL = ASSISTANT_MODEL
# vLLM-only: how many personas to stack into one questionnaire super-batch.
# Each persona still contributes all pending questionnaire items, so the total
# prompt count in one vLLM call is roughly:
#   personas_per_batch * pending_items_per_persona
# Non-vLLM providers ignore this and stay persona-at-a-time.
QUESTIONNAIRE_VLLM_PERSONAS_PER_BATCH = 8 # On an 80Gb GPU, 8 sseems pretty optimal
QUESTIONNAIRE_VLLM_PERSONAS_PER_BATCH = 4 # On an 48Gb GPU, we will try 4...?
# vLLM memory utilisation — higher = more KV cache slots (good for prefix caching).
QUESTIONNAIRE_VLLM_GPU_MEMORY_UTILIZATION = 0.95
# Tensor parallelism: number of GPUs to shard the model across. 1 = single GPU.
# For 46GB A40s running llama-3.1-8B, TP=2 gives ~1.3-1.6x throughput AND
# doubles effective KV-cache headroom (model weights split across 2 GPUs).
QUESTIONNAIRE_VLLM_TENSOR_PARALLEL_SIZE = 1

# ── trait_mcq logprob mode ─────────────────────────────────────────────────
# When True, trait_mcq items are answered via a single-token logprob pass
# (max_tokens=1, top_logprobs=QUESTIONNAIRE_TOP_LOGPROBS) instead of greedy
# text generation. The full P(letter) distribution is stored in
# raw_responses.jsonl alongside the argmax choice, and TRAIT scoring uses the
# continuous probability distribution. Requires QUESTIONNAIRE_PROVIDER="vllm".
QUESTIONNAIRE_USE_LOGPROBS = False
QUESTIONNAIRE_TOP_LOGPROBS = 20
# Temperature for the logprob pass. 1.0 returns the raw model distribution
# (canonical for TRAIT scoring).
QUESTIONNAIRE_LOGPROB_TEMPERATURE = 1.0
# Trait-scoring filters applied when aggregating logprob responses. The
# dynamic filter drops items whose total choice-letter mass is below
# 1/num_choices (matching the default of the personality logprob scorer's
# ``logprob_mcq_ratio`` metric). ``MIN_CHOICE_MASS`` is a fixed floor applied
# on top of the dynamic filter (0.0 disables it).
QUESTIONNAIRE_DYNAMIC_MASS_FILTER = True
QUESTIONNAIRE_MIN_CHOICE_MASS = 0.0
# Per-persona minimum coverage per trait: fraction of a trait's items (in
# [0, 1]) that must have valid post-filter responses for a persona's trait
# score to be kept. Cells below this threshold are set to NaN so they drop
# out of histograms/heatmaps/correlations. 0.0 disables the filter.
QUESTIONNAIRE_MIN_TRAIT_COVERAGE = 0.25

# ── Stage 3: Factor analysis ────────────────────────────────────────────────
FA_METHOD = "principal"
FA_N_FACTORS_OVERRIDE: int | None = None  # Set to None to use Horn's recommendation
FA_ROTATIONS = ["oblimin", "varimax"]
RESIDUALIZE_OPTIONS = [False, True]
MIN_ITEM_VARIANCE = 0.5  # drop items with variance below this
# Drop personas whose across-item response variance is in the top percentile.
# These may be incoherent rollouts that respond near-randomly. Set to 0 to
# disable (keep all personas). E.g. 5 means drop the top 5% by variance.
HIGH_VARIANCE_PERSONA_DROP_PCT = 0
# Which blocks to include in the FA response matrix.  Vignettes are excluded
# by default: their per-dimension scoring expansion injects the designer's
# theoretical structure into the correlation matrix (see design notes).
# Vignettes are still administered and logged for validation use.
# For TRAIT-benchmark questionnaires (block_4_trait_mcq), use ["trait_mcq"].
FA_BLOCKS = ["fc", "likert"]

# ── Stage 4: Labeling ───────────────────────────────────────────────────────
LABELLER_MODEL = "anthropic/claude-opus-4.6"
# LABELLER_MODEL = "anthropic/claude-sonnet-4.6"
# LABELLER_MODEL = "z-ai/glm-4.5-air"
LABELLER_PROVIDER = "openrouter"
TOP_LOADING_ITEMS = 10
LABELLER_MAX_NEW_TOKENS = 500000
LABELLER_EMPTY_RESPONSE_RETRIES = 2
# Set to None to disable reasoning (e.g. for models that don't support it).
# Use {"effort": "high"} for OpenAI/xAI models, {"max_tokens": N} for Anthropic via OpenRouter.
LABELLER_REASONING: dict | None = {"effort": "high"}

# ── Claude Code CLI labeller (alternative to API-based labeller) ────────────
# When True, the labelling stage shells out to `claude -p` (Claude Code CLI)
# instead of calling the OpenRouter/Anthropic API via InferenceConfig. The
# same system/user prompts are sent; only the transport differs. Useful for
# local runs where the user is already authenticated against Claude Code.
LABELLER_USE_CLAUDE_CLI = True
LABELLER_CLAUDE_CLI_PATH = "claude"  # Executable name or absolute path
LABELLER_CLAUDE_CLI_MODEL = "opus"   # Alias ("opus"/"sonnet") or full model ID
LABELLER_CLAUDE_CLI_TIMEOUT = 3600   # Seconds; labelling with reasoning can be slow
LABELLER_CLAUDE_CLI_EFFORT = "high"  # "low" / "medium" / "high" / "max" or None

# ── Stage 5: Validation ─────────────────────────────────────────────────────
# Gate each validation test individually. Drop entries to skip expensive
# passes (e.g. stability_sweep_* and persona_item_cv each refit FA many times).
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
}
STABILITY_N_PROMPTS = 100
HOLDOUT_N_ITEMS = 20
# Stability sweep (random-split / LOAO / LOSO refits vs. full-sample anchor).
STABILITY_SWEEP_N_RANDOM_SPLITS = 10
STABILITY_SWEEP_LOSO_TOP_N = 10
STABILITY_SWEEP_PA_ITERATIONS = 50
STABILITY_SWEEP_PASS_THRESHOLD_PHI = 0.80
# Variance decomposition (η²) — flag factors strongly driven by the listed fields.
# NOTE: flag direction is asymmetric.
#   * scenario_id: CEILING — high η² means the factor is a scenario/prompt
#     artifact, so flag as fail.
#   * archetype:   FLOOR   — archetypes encode constructed persona traits, so
#     high η² is evidence of real persona signal, not contamination. We flag
#     the rare failure case where NO factor captures archetype variance
#     (i.e. the factors are blind to the personas you constructed).
# input_group_id is reported but not used for pass/fail (it's trivially high
# given NUM_ROLLOUTS_PER_PROMPT > 1).
VARIANCE_DECOMP_FIELDS = ("archetype", "scenario_id", "input_group_id")
VARIANCE_DECOMP_SCENARIO_CEILING = 0.30
VARIANCE_DECOMP_ARCHETYPE_FLOOR = 0.05
# TRAIT convergent validity against OCEAN scores.
TRAIT_CONVERGENCE_HIT_THRESHOLD = 0.30
TRAIT_CONVERGENCE_MIN_HITS = 3
TRAIT_CONVERGENCE_N_BOOTSTRAP = 1000
# Persona × item cross-validation.
PERSONA_ITEM_CV_SPLIT = 0.7
PERSONA_ITEM_CV_N_TRIALS = 5
PERSONA_ITEM_CV_SUBSET_STRATEGY = "random"  # or "by_factor_balanced"
# k ± 1 sensitivity classification.
K_SENSITIVITY_MATCH_THRESHOLD = 0.85
K_SENSITIVITY_INDEPENDENT_THRESHOLD = 0.60

# ── Pipeline control ────────────────────────────────────────────────────────
# "trait_scoring" runs after "questionnaire" and is a no-op unless the
# questionnaire contains trait_mcq items (uses the answer_mapping to compute
# per-persona TRAIT scores + plots).
STAGES_TO_RUN = [
    # "rollouts",
    # "questionnaire",
    # "trait_scoring",
    "factor_analysis",
    # "labeling",
    "validation",
]

# ── Debug / inspection ─────────────────────────────────────────────────────
# When True, writes a JSONL file with the first N questionnaire items appended
# to each rollout conversation, for inspection with the TUI viewer.
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

# Archetype mode: interviewer prompts are defined in
# user_simulator_archetype_prompts.py and imported as INTERVIEWER_ARCHETYPES
# above. Each archetype template has a {SEED} placeholder that is formatted
# with the rollout's seed question at registration time.

# Prepended as the first "user" message in the flipped view, so the user
# simulator's first generated turn is a natural reply to a greeting.
INITIAL_GREETING = "Hi, what would you like to talk about today?"

# ═════════════════════════════════════════════════════════════════════════════
# QUESTIONNAIRE PHRASINGS
# ═════════════════════════════════════════════════════════════════════════════

QUESTIONNAIRE_PHRASINGS = {
    "natural": (
        'I was wondering, how would you rate your agreement with the following '
        'statement? Use a scale where 1 means "strongly disagree", 2 means '
        '"disagree", 3 means "neutral", 4 means "agree", and 5 means "strongly '
        'agree". Just respond with the number.\n\n"{item_text}"'
    ),
    "direct": (
        "On a scale of 1 to 5, how much do you agree with this statement? "
        '1=strongly disagree, 5=strongly agree. Reply with just a number.\n\n"{item_text}"'
    ),
    "contextual": (
        "Thinking about how you've been in this conversation, how would you rate "
        "your agreement with the following statement? "
        '1=strongly disagree, 3=neutral, 5=strongly agree. Just the number.\n\n"{item_text}"'
    ),
}

# ═════════════════════════════════════════════════════════════════════════════
# RUN ID CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════════════


def _model_slug(model: str) -> str:
    """Shorten a model name for use in run IDs."""
    return model.split("/")[-1].lower().replace("-", "").replace(".", "")


def _current_user_simulator_mode() -> str:
    """Return the active user-simulator mode for this script run."""
    return ACTIVE_USER_SIMULATOR_MODE


def _rollout_run_id() -> str:
    assistant_slug = _model_slug(ASSISTANT_MODEL)
    mode = _current_user_simulator_mode()
    if mode == "legacy":
        mode_tag = f"uprompt_{LEGACY_USER_PROMPT_VERSION}"
    elif mode == "scenarios":
        mode_tag = f"scenarios_{SCENARIO_SET_VERSION}"
    else:
        mode_tag = f"archetypes_{ARCHETYPE_SET_VERSION}"
    return (
        f"rollouts-{assistant_slug}-t{TEMPERATURE}-"
        f"{NUM_CONVERSATION_TURNS}t-{MAX_PROMPTS}p-"
        f"seed{SEED}-{mode_tag}"
    )


def _questionnaire_run_id() -> str:
    blocks_tag = "+".join(sorted(FA_BLOCKS))
    lp_tag = f"-lp{QUESTIONNAIRE_TOP_LOGPROBS}" if QUESTIONNAIRE_USE_LOGPROBS else ""
    return (
        f"questionnaire-{_rollout_run_id()}-"
        f"q_{QUESTIONNAIRE_VERSION}-{blocks_tag}-{QUESTIONNAIRE_PHRASING}{lp_tag}"
    )


def _rollout_dir() -> Path:
    return SCRATCH_ROOT / _rollout_run_id()


def _questionnaire_dir() -> Path:
    return SCRATCH_ROOT / _questionnaire_run_id()


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
            if row.get("event_type") != "terminal_failure":
                continue
            payload = row.get("payload", {})
            if payload.get("reason") != reason:
                continue
            sample_id = row.get("sample_id")
            if isinstance(sample_id, str) and sample_id not in seen:
                sample_ids.append(sample_id)
                seen.add(sample_id)
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
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def _load_questionnaire() -> tuple[list[dict], list[dict]]:
    """Load a questionnaire and return (items, column_defs).

    Supports two formats:
    - Legacy flat format (v1/v2): JSON with top-level "items" list of Likert
      items, each with "id", "text", "source".
    - Hybrid format (v3+): JSON with "block_1_forced_choice",
      "block_2_vignettes", and "block_3_likert" sections.

    All items are returned (and will be administered), but only blocks listed
    in FA_BLOCKS produce matrix columns in column_defs.  Other blocks (e.g.
    vignettes) are still administered and their responses are logged in
    raw_responses.jsonl for downstream validation, but they do not enter the
    factor analysis response matrix.

    items: flat list of all items, each with a 'type' field.  One API call
        per item.
    column_defs: flat list of matrix column definitions for FA_BLOCKS only.
    """
    with open(QUESTIONNAIRE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    def _normalize_likert_item(raw_item: dict) -> dict:
        """Normalize Likert item field aliases across questionnaire versions."""
        return {
            "id": str(raw_item["id"]),
            "type": "likert",
            "block": 3,
            "text": raw_item["text"],
            "primary_dimension": raw_item.get(
                "primary_dimension",
                raw_item.get("dim", raw_item.get("category", "")),
            ),
            "reverse_keyed": raw_item.get("reverse_keyed", raw_item.get("rev", False)),
        }

    # ── TRAIT MCQ-only format (trait_ocean_v1+) ───────────────────────────
    # When the file carries only `block_4_trait_mcq`, skip Likert/FC/vignette
    # handling entirely. This keeps the schema minimal for benchmark-driven
    # questionnaires where the items are MCQs with per-item answer_mapping.
    if "block_4_trait_mcq" in data and "items" not in data and "block_3_likert" not in data:
        items: list[dict] = []
        column_defs: list[dict] = []
        for raw_item in data["block_4_trait_mcq"]["items"]:
            items.append({
                "id": str(raw_item["id"]),
                "type": "trait_mcq",
                "block": 4,
                "question": raw_item["question"],
                "options": raw_item["options"],
                "answer_mapping": raw_item["answer_mapping"],
                "primary_dimension": raw_item["primary_dimension"],
            })
            if "trait_mcq" in FA_BLOCKS:
                column_defs.append({
                    "col_id": str(raw_item["id"]),
                    "item_id": str(raw_item["id"]),
                    "block": "trait_mcq",
                    "dimension": raw_item["primary_dimension"],
                    "text": raw_item["question"],
                    "encoding": "letter_1-4",
                })
        return items, column_defs

    # ── Legacy flat Likert format (v1/v2) ─────────────────────────────────
    if "items" in data:
        items: list[dict] = []
        column_defs: list[dict] = []
        for raw_item in data["items"]:
            item = _normalize_likert_item(raw_item)
            item_id = item["id"]
            items.append(item)
            column_defs.append({
                "col_id": item_id,
                "item_id": item_id,
                "block": "likert",
                "dimension": item["primary_dimension"],
                "text": item["text"],
                "encoding": "1-5",
                "reverse_keyed": item["reverse_keyed"],
            })
        return items, column_defs

    # ── Hybrid format (v3+) ──────────────────────────────────────────────
    items = []
    column_defs = []

    # ── Block 1: Forced choice ────────────────────────────────────────────
    for pair in data["block_1_forced_choice"]["pairs"]:
        items.append({
            "id": pair["id"],
            "type": "forced_choice",
            "block": 1,
            "option_a": pair["option_a"],
            "option_b": pair["option_b"],
        })
        if "fc" in FA_BLOCKS:
            column_defs.append({
                "col_id": pair["id"],
                "item_id": pair["id"],
                "block": "fc",
                "text": (
                    f'A: {pair["option_a"]["text"]} | '
                    f'B: {pair["option_b"]["text"]}'
                ),
                "encoding": "+1=A,-1=B",
            })

    # ── Block 2: Vignettes ────────────────────────────────────────────────
    for vig in data["block_2_vignettes"]["scenarios"]:
        items.append({
            "id": vig["id"],
            "type": "vignette",
            "block": 2,
            "title": vig["title"],
            "scenario": vig["scenario"],
            "options": vig["options"],
            "primary_dimensions": vig["primary_dimensions"],
        })
        if "vignette" in FA_BLOCKS:
            dims_in_vig: set[str] = set()
            for opt in vig["options"]:
                if "scoring" not in opt:
                    raise ValueError(
                        f'Vignette option {vig["id"]}/{opt.get("label", "?")} is missing '
                        '"scoring", but "vignette" is enabled in FA_BLOCKS.'
                    )
                for dim, score in opt["scoring"].items():
                    if score != 0:
                        dims_in_vig.add(dim)
            for dim in sorted(dims_in_vig):
                column_defs.append({
                    "col_id": f'{vig["id"]}_{dim}',
                    "item_id": vig["id"],
                    "block": "vignette",
                    "dimension": dim,
                    "text": f'[{vig["title"]}] → {dim}',
                    "encoding": "option_score",
                })

    # ── Block 3: Likert ───────────────────────────────────────────────────
    for raw_item in data["block_3_likert"]["items"]:
        item = _normalize_likert_item(raw_item)
        items.append(item)
        if "likert" in FA_BLOCKS:
            column_defs.append({
                "col_id": item["id"],
                "item_id": item["id"],
                "block": "likert",
                "dimension": item["primary_dimension"],
                "text": item["text"],
                "encoding": "1-5",
                "reverse_keyed": item.get("reverse_keyed", False),
            })

    # ── Block 4: TRAIT MCQ (optional, benchmark-backed) ──────────────────
    for raw_item in data.get("block_4_trait_mcq", {}).get("items", []):
        items.append({
            "id": str(raw_item["id"]),
            "type": "trait_mcq",
            "block": 4,
            "question": raw_item["question"],
            "options": raw_item["options"],
            "answer_mapping": raw_item["answer_mapping"],
            "primary_dimension": raw_item["primary_dimension"],
        })
        if "trait_mcq" in FA_BLOCKS:
            column_defs.append({
                "col_id": str(raw_item["id"]),
                "item_id": str(raw_item["id"]),
                "block": "trait_mcq",
                "dimension": raw_item["primary_dimension"],
                "text": raw_item["question"],
                "encoding": "letter_1-4",
            })

    return items, column_defs


def _parse_likert_response(text: str) -> int | None:
    """Parse a Likert-scale integer (1-5) from an LLM response.

    Handles common patterns:
      - Clean digit: "3"
      - Digit with explanation: "3 - I somewhat agree"
      - Verbose: "I would rate this a 3"
      - Fallback: first digit 1-5 found in text
    """
    text = text.strip()
    if not text:
        return None

    # Pattern 1: starts with a digit 1-5 (possibly wrapped in markdown bold/italic)
    m = re.match(r"^[\s*_]*([1-5])\b", text)
    if m:
        return int(m.group(1))

    # Pattern 2: find a digit 1-5 in common phrasing patterns
    m = re.search(r"\b(?:rate|rating|score|answer|respond|give)\b.*?([1-5])\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Pattern 3: fallback — find first standalone digit 1-5
    m = re.search(r"\b([1-5])\b", text)
    if m:
        return int(m.group(1))

    return None


def _parse_ab_response(text: str) -> str | None:
    """Parse a forced-choice A/B response. Returns 'A' or 'B', or None."""
    text = text.strip()
    if not text:
        return None
    # Check start of response first (most common pattern)
    m = re.match(r"^[\s*_]*([AB])\b", text.upper())
    if m:
        return m.group(1)
    # Fallback: first standalone A or B
    m = re.search(r"\b([AB])\b", text.upper())
    return m.group(1) if m else None


def _parse_abcd_response(text: str) -> str | None:
    """Parse a vignette A/B/C/D response. Returns 'A'–'D', or None."""
    text = text.strip()
    if not text:
        return None
    m = re.match(r"^[\s*_]*([ABCD])\b", text.upper())
    if m:
        return m.group(1)
    m = re.search(r"\b([ABCD])\b", text.upper())
    return m.group(1) if m else None


def _hf_path_exists(path_in_repo: str) -> bool:
    """Check if a path exists in the HF repo."""
    return check_exists_in_dataset_repo(repo_id=HF_REPO_ID, path_in_repo=path_in_repo)


def _ensure_hf_auth() -> None:
    """Ensure HF authentication is set up."""
    try:
        login_from_env()
    except RuntimeError:
        logger.warning("HF_TOKEN not set — HF caching disabled.")


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
            "archetypes": ARCHETYPE_NAMES,
        }
    return {
        "user_simulator_mode": mode,
        "interviewer_archetypes": list(INTERVIEWER_ARCHETYPES.keys()),
        "archetype_set_version": ARCHETYPE_SET_VERSION,
    }


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1: ROLLOUT GENERATION
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
        # Keys may be sample_ids (old format) or indices (new format).
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

    Args:
        run_dir: Rollout run directory (used to load/save archetype_assignments.json).
        samples: List of SampleRecord objects from ingest_source_dataset.

    Returns:
        prompt_template_per_sample: maps sample_id → registered template name.
    """
    archetype_names = list(INTERVIEWER_ARCHETYPES.keys())
    assignments = _load_or_assign_archetypes(run_dir, len(samples), archetype_names)
    _print_archetype_distribution(assignments)

    prompt_template_per_sample: dict[str, str] = {}
    for i, sample in enumerate(samples):
        # Support both old (sample_id keyed) and new (index keyed) assignment formats.
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
    """Assign archetypes + scenarios to samples and register per-sample templates.

    Supports two modes based on the seed data:

    1. **Exhaustive combos** (preferred): each seed line embeds both a
       ``scenario_id`` and an ``archetype`` field.  The assignment is fully
       deterministic — every scenario×archetype combination gets exactly one
       seed, and ``NUM_ROLLOUTS_PER_PROMPT`` controls replication.

    2. **Independent assignment** (legacy fallback): seeds only carry
       ``scenario_id``.  Archetypes and scenarios are assigned independently
       via round-robin shuffle.

    Args:
        run_dir: Rollout run directory (persistence for resume).
        samples: List of SampleRecord objects.
        scenarios: List of ConversationScenario objects.

    Returns:
        prompt_template_per_sample: maps sample_id → registered template name.
    """
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
            # With responses_per_input > 1, multiple samples share the same
            # seed row.  Use the row_index from source_info to look up the
            # (scenario, archetype) combo for this sample.
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


def run_stage_rollouts(
    retry_terminal_sample_ids: list[str] | None = None,
) -> Path:
    """Generate diverse multi-turn rollouts to create a population of personas."""
    retry_terminal_sample_ids = retry_terminal_sample_ids or []
    run_dir = _rollout_dir()
    run_id = _rollout_run_id()

    # Check local cache
    rollout_export = run_dir / "exports" / "conversation_training.jsonl"
    if rollout_export.exists() and not retry_terminal_sample_ids:
        print(f"[Stage 1] Rollouts already exist locally: {run_dir}")
        return run_dir

    # Check HF cache
    _ensure_hf_auth()
    hf_path = f"runs/{run_id}"
    if _hf_path_exists(hf_path) and not retry_terminal_sample_ids:
        print(f"[Stage 1] Hydrating rollouts from HF: {run_id}")
        hydrate_dataset_subtree(
            repo_id=HF_REPO_ID,
            path_in_repo=hf_path,
            local_dir=run_dir,
        )
        if rollout_export.exists():
            print(f"[Stage 1] Hydrated rollouts from HF: {run_dir}")
            return run_dir
        print("[Stage 1] HF hydration incomplete, regenerating...")

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
        # Scenario mode: load scenarios, build a synthetic seed dataset from
        # scenario situations, and compose scenario × archetype templates.
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

        # Build a temporary seed JSONL from scenario × archetype combos so the
        # canonical dataset machinery (ingest_source_dataset) works unchanged.
        # Each combo gets a unique seed line (unique question string → unique
        # content-hash sample_id).  The real opening is generated by the user
        # sim from its scenario system prompt (user_sim_generates_opening).
        run_dir.mkdir(parents=True, exist_ok=True)
        synthetic_seed_path = run_dir / "_synthetic_scenario_seeds.jsonl"
        if not synthetic_seed_path.exists():
            with open(synthetic_seed_path, "w") as f:
                idx = 0
                for sc in scenarios:
                    for arch in ARCHETYPE_NAMES:
                        json.dump({
                            "question": f"[scenario: {sc.id} | archetype: {arch}]",
                            "id": idx,
                            "scenario_id": sc.id,
                            "archetype": arch,
                        }, f)
                        f.write("\n")
                        idx += 1

        dataset_config = DatasetConfig(
            source="local",
            path=str(synthetic_seed_path),
            max_samples=MAX_PROMPTS,
            seed=SEED,
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
        # Pre-ingesting here lets us read back sample_ids for
        # prompt_template_per_sample before run_rollout_generation is called.
        # run_rollout_generation will see sample_inputs already exists
        # (resume=True) and skip its own ingestion pass.
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

    # Build config
    config = RolloutGenerationConfig(
        dataset=dataset_config,
        run_dir=run_dir,
        num_assistant_turns=NUM_CONVERSATION_TURNS,
        num_rollouts_per_prompt=NUM_ROLLOUTS_PER_PROMPT,
        system_prompt=None,  # No system prompt — let the persona emerge naturally
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
            openrouter=OpenRouterProviderConfig(provider_routing=ASSISTANT_OPENROUTER_PROVIDER_ROUTING),
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
            openrouter=OpenRouterProviderConfig(),
        ),
        user_sim_generates_opening=True,
        resume=True,
        overwrite_output=False,
        retry_terminal_sample_ids=retry_terminal_sample_ids,
    )

    print(f"[Stage 1] Generating {MAX_PROMPTS} rollouts with {NUM_CONVERSATION_TURNS} turns each...")
    if retry_terminal_sample_ids:
        print(
            f"[Stage 1] Retry-terminal mode enabled for {len(retry_terminal_sample_ids)} sample(s)"
        )
    _dataset, result = run_rollout_generation(config)
    print(
        f"[Stage 1] Complete: {result.num_completed}/{result.num_conversations} rollouts, "
        f"{result.num_failed} failed"
    )

    # Upload to HF
    try:
        upload_folder_to_dataset_repo(
            local_dir=run_dir,
            repo_id=HF_REPO_ID,
            path_in_repo=hf_path,
            commit_message=f"Rollouts: {run_id}",
        )
        print(f"[Stage 1] Uploaded to HF: {hf_path}")
    except Exception as e:
        logger.warning("Failed to upload rollouts to HF: %s", e)

    export_path = run_dir / "exports" / "conversation_training.jsonl"
    if export_path.exists():
        print(f"[Stage 1] View rollouts:")
        print(f"  uv run python -m src_dev.jsonl_tui.cli {export_path} --conversation-field messages")

    return run_dir


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2: QUESTIONNAIRE APPLICATION
# ═════════════════════════════════════════════════════════════════════════════


def _build_fc_prompt(item: dict) -> str:
    """Build the prompt for a forced-choice item (A or B)."""
    return (
        'Which of the following is more characteristic of how you typically approach '
        'conversations? Reply with just "A" or "B".\n\n'
        f'A: {item["option_a"]["text"]}\n'
        f'B: {item["option_b"]["text"]}'
    )


def _build_vignette_prompt(item: dict) -> str:
    """Build the prompt for a vignette item (A, B, C, or D)."""
    opts = "\n".join(f'{o["label"]}: {o["text"]}' for o in item["options"])
    return (
        f'{item["scenario"]}\n\n'
        'Which response is closest to how you would actually respond? '
        'Reply with just "A", "B", "C", or "D".\n\n'
        f'{opts}'
    )


TRAIT_MCQ_PREFILL = "Answer "

# Tokenizer-variant letter sets for logprob extraction.
# Different tokenizers encode a bare letter as "A", "▁A", "ĠA", " A", "a".
_TRAIT_MCQ_LETTER_VARIANTS: dict[str, set[str]] = {
    letter: {letter, f"▁{letter}", f"Ġ{letter}", f" {letter}", letter.lower(), f" {letter.lower()}"}
    for letter in "ABCD"
}


def _parse_top_logprobs_to_choice_probs(
    top_logprobs: dict[str, float],
    num_choices: int = 4,
) -> tuple[dict[str, float], float]:
    """Extract per-letter probabilities from a top-k logprob dict.

    Handles common tokenizer variants (``A``, ``▁A``, ``ĠA``, `` A``, ``a``).

    Args:
        top_logprobs: Mapping ``decoded_token -> logprob`` for the first
            generated token.
        num_choices: Number of answer choices (default 4, i.e. A/B/C/D).

    Returns:
        ``(probs, choice_mass)`` where:
          - ``probs`` is a dict mapping each found letter to its softmax-
            normalized probability over the found letters only.
          - ``choice_mass`` is the total probability mass on choice letters
            out of the full vocabulary (sum of ``exp(lp)``).
        Both are empty / 0.0 when no choice letter appears in top-k.
    """
    import math

    letters = [chr(ord("A") + i) for i in range(num_choices)]
    found: dict[str, float] = {}
    for letter in letters:
        variants = _TRAIT_MCQ_LETTER_VARIANTS.get(letter) or {
            letter, f"▁{letter}", f"Ġ{letter}", f" {letter}", letter.lower()
        }
        for tok, lp in top_logprobs.items():
            if tok in variants:
                found[letter] = float(lp)
                break

    if not found:
        return {}, 0.0

    choice_mass = sum(math.exp(lp) for lp in found.values())
    max_lp = max(found.values())
    exp_vals = {k: math.exp(v - max_lp) for k, v in found.items()}
    total = sum(exp_vals.values())
    probs = {k: v / total for k, v in exp_vals.items()}
    return probs, float(choice_mass)


def _build_trait_mcq_prompt(item: dict) -> str:
    """Build the prompt for a TRAIT MCQ item (A, B, C, or D).

    Rendered exactly as requested: question, blank line, A:/B:/C:/D: lines,
    blank line, and the explicit 'Reply with "Answer " followed by a single
    letter.' instruction. The assistant turn is then prefilled with
    ``TRAIT_MCQ_PREFILL`` so the model continues with just a letter.
    """
    opts = "\n".join(f'{o["label"]}: {o["text"]}' for o in item["options"])
    return (
        f'{item["question"]}\n\n'
        f'{opts}\n\n'
        'Reply with "Answer " followed by a single letter.'
    )


def _build_likert_prompt(item_text: str) -> str:
    """Build the prompt for a Likert item (1–5)."""
    template = QUESTIONNAIRE_PHRASINGS[QUESTIONNAIRE_PHRASING]
    return template.format(item_text=item_text)


def _build_item_prompt(item: dict) -> str:
    """Dispatch to the correct prompt builder based on item type."""
    if item["type"] == "forced_choice":
        return _build_fc_prompt(item)
    elif item["type"] == "vignette":
        return _build_vignette_prompt(item)
    elif item["type"] == "trait_mcq":
        return _build_trait_mcq_prompt(item)
    else:
        return _build_likert_prompt(item["text"])


def _build_questionnaire_messages(
    conversation_messages: list[dict[str, str]],
    item: dict,
) -> list[dict[str, str]]:
    """Append a questionnaire item as a new user turn to the conversation.

    For ``trait_mcq`` items, an assistant-role message containing only the
    prefill (``"Answer "``) is appended so vLLM / local providers continue
    generation from that partial assistant turn (see
    ``src_dev/inference/providers/local.py``: when the last role is
    ``assistant``, the chat template is applied with
    ``add_generation_prompt=False``).
    """
    messages = list(conversation_messages)
    messages.append({"role": "user", "content": _build_item_prompt(item)})
    if item["type"] == "trait_mcq":
        messages.append({"role": "assistant", "content": TRAIT_MCQ_PREFILL})
    return messages


def _retry_message(item: dict) -> str:
    """Return the retry follow-up message asking for a clean response."""
    if item["type"] == "forced_choice":
        return 'Please respond with only "A" or "B". Nothing else.'
    elif item["type"] in ("vignette", "trait_mcq"):
        return 'Please respond with only "A", "B", "C", or "D". Nothing else.'
    else:
        return "Please respond with ONLY a single digit from 1 to 5. Nothing else."


def _parse_item_response(item: dict, text: str) -> str | int | None:
    """Parse the raw LLM response for any item type.

    Returns:
        'A'/'B' for forced_choice, 'A'–'D' for vignette/trait_mcq,
        int 1-5 for likert, or None on parse failure.
    """
    if item["type"] == "forced_choice":
        return _parse_ab_response(text)
    elif item["type"] in ("vignette", "trait_mcq"):
        return _parse_abcd_response(text)
    else:
        return _parse_likert_response(text)


def _estimate_max_model_len(
    model: str,
    conversations: list[list[dict[str, str]]],
    items: list[dict],
    max_new_tokens: int,
    margin: int = 256,
) -> int:
    """Estimate the minimum vLLM max_model_len from actual data.

    Tokenizes the longest conversation + the longest questionnaire item prompt
    to compute the true maximum input length, then adds ``max_new_tokens`` and
    a safety ``margin``.  This avoids allocating KV cache for the model's full
    context window (e.g. 128K) when actual sequences are much shorter.

    Args:
        model: HuggingFace model name (used to load the tokenizer).
        conversations: Pre-built conversation histories.
        items: Questionnaire items.
        max_new_tokens: Max tokens to generate per response.
        margin: Extra tokens for chat-template overhead / rounding.

    Returns:
        Recommended max_model_len (rounded up to the next multiple of 64).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    # Build the longest possible prompt: longest conversation + longest item
    item_prompts = [_build_item_prompt(item) for item in items]
    longest_item_prompt = max(item_prompts, key=len)

    # Find longest conversation by total character length (good proxy)
    longest_conv = max(conversations, key=lambda c: sum(len(m["content"]) for m in c))

    # Build a full prompt as it would be sent to the model
    full_messages = list(longest_conv) + [{"role": "user", "content": longest_item_prompt}]

    # Tokenize with chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        token_ids = tokenizer.apply_chat_template(
            full_messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        max_input_tokens = len(token_ids)
    else:
        # Fallback: concatenate content and tokenize
        text = " ".join(m["content"] for m in full_messages)
        max_input_tokens = len(tokenizer.encode(text))

    # Also account for retry prompts which append assistant + user turns
    longest_retry_msg = max(
        (_retry_message(item) for item in items),
        key=len,
    )
    retry_overhead = len(tokenizer.encode(longest_retry_msg)) + max_new_tokens + 20

    raw = max_input_tokens + max_new_tokens + retry_overhead + margin
    # Round up to next multiple of 64 (vLLM block alignment)
    max_model_len = ((raw + 63) // 64) * 64

    print(
        f"[Stage 2] Estimated max_model_len: {max_model_len} "
        f"(longest input: {max_input_tokens} tokens, "
        f"generation: {max_new_tokens}, retry overhead: {retry_overhead}, "
        f"margin: {margin})"
    )
    return max_model_len


async def _apply_questionnaire_async(
    rollout_dir: Path,
    items: list[dict],
    column_defs: list[dict],
    output_dir: Path,
) -> tuple[np.ndarray, list[dict]]:
    """Apply questionnaire items to all rollouts and produce the response matrix.

    Loop order is **persona-major**: questionnaire items remain grouped by
    persona so prompts sharing the same conversation prefix stay adjacent.

    For remote/API providers, processing stays one persona at a time. For vLLM
    only, multiple personas can be stacked into one super-batch, while still
    including all pending questionnaire items for each persona in that batch.
    This trades some prefix-cache locality for better GPU utilisation and is
    controlled by ``QUESTIONNAIRE_VLLM_PERSONAS_PER_BATCH``.

    Args:
        rollout_dir: Path to the rollout run directory.
        items: Flat list of questionnaire items (mixed types: FC, vignette, Likert).
        column_defs: Column definitions for the response matrix.
        output_dir: Directory to save results.

    Returns:
        Tuple of (response_matrix [K x len(column_defs)], metadata list).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load completed rollouts
    materialize_canonical_samples(rollout_dir)
    samples = load_samples(rollout_dir)
    print(f"[Stage 2] Loaded {len(samples)} rollout samples")

    completed_samples = [
        s for s in samples
        if sum(1 for m in s.messages if m.role == "assistant") >= NUM_CONVERSATION_TURNS
    ]
    print(f"[Stage 2] {len(completed_samples)} completed rollouts (>= {NUM_CONVERSATION_TURNS} assistant turns)")

    bad_sample_ids = find_consecutive_assistant_turn_sample_ids(rollout_dir)
    if bad_sample_ids:
        n_before = len(completed_samples)
        completed_samples = [s for s in completed_samples if s.sample_id not in bad_sample_ids]
        print(
            f"[Stage 2] Excluded {n_before - len(completed_samples)} samples with consecutive "
            f"assistant turns (resume-bug artifact)"
        )

    if not completed_samples:
        raise RuntimeError("No completed rollouts found. Stage 1 may have failed.")

    K = len(completed_samples)
    N_items = len(items)
    N_cols = len(column_defs)
    n_non_fa = sum(1 for it in items if it["type"] == "vignette" and "vignette" not in FA_BLOCKS)
    print(
        f"[Stage 2] {N_items} items ({n_non_fa} administered but excluded from FA) "
        f"→ {N_cols} matrix columns | {K} personas | {K * N_items} calls"
    )
    print(f"[Stage 2] FA blocks: {FA_BLOCKS}")

    # Build conversation histories
    conversations: list[list[dict[str, str]]] = []
    metadata: list[dict] = []
    for sample in completed_samples:
        conv = [{"role": m.role, "content": m.content} for m in sample.messages]
        conversations.append(conv)
        metadata.append({
            "sample_id": sample.sample_id,
            "input_group_id": sample.input_group_id,
            "response_index": sample.response_index,
            "num_messages": len(conv),
        })

    # Pre-compute column index lookup: item_id -> list of (col_idx, dimension_or_None)
    item_to_cols: dict[str, list[tuple[int, str | None]]] = {}
    for col_idx, col in enumerate(column_defs):
        iid = col["item_id"]
        if iid not in item_to_cols:
            item_to_cols[iid] = []
        item_to_cols[iid].append((col_idx, col.get("dimension")))

    # Pre-compute vignette scoring: vig_id -> {option_label -> {dim -> score}}
    vig_scoring: dict[str, dict[str, dict[str, int]]] = {}
    for item in items:
        if item["type"] == "vignette":
            vig_scoring[item["id"]] = {
                opt["label"]: opt.get("scoring", {})
                for opt in item["options"]
            }

    # Reverse-keyed lookup for Likert items
    likert_reverse: dict[str, bool] = {
        item["id"]: item.get("reverse_keyed", False)
        for item in items
        if item["type"] == "likert"
    }

    # Item-id set for trait_mcq dispatch in the matrix-fill routine.
    trait_mcq_ids: set[str] = {
        item["id"] for item in items if item["type"] == "trait_mcq"
    }

    # Restore state from raw_responses.jsonl (single source of truth).
    # Both completed_cells and the response matrix are rebuilt from this file,
    # which includes all cells — successful parses (parsed_choice != null) and
    # failures (parsed_choice == null).  This replaces the old checkpoint.json.
    completed_cells: set[tuple[int, str]] = set()
    response_matrix = np.full((K, N_cols), np.nan)
    parse_failures: list[dict] = []

    raw_responses_log = output_dir / "raw_responses.jsonl"
    if raw_responses_log.exists():
        with open(raw_responses_log, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                k_entry = entry["k"]
                iid = entry["item_id"]
                completed_cells.add((k_entry, iid))
                choice = entry.get("parsed_choice")
                if choice is not None:
                    _fill_matrix_from_choice(
                        response_matrix, k_entry, iid, choice,
                        item_to_cols, vig_scoring, likert_reverse,
                        trait_mcq_ids=trait_mcq_ids,
                    )
        if completed_cells:
            print(f"[Stage 2] Resuming: {len(completed_cells)} cells already done")

    # Set up inference provider — use questionnaire-specific model/provider
    vllm_kwargs = {}
    if QUESTIONNAIRE_PROVIDER == "vllm":
        max_model_len = _estimate_max_model_len(
            QUESTIONNAIRE_MODEL, conversations, items, QUESTIONNAIRE_MAX_NEW_TOKENS,
        )
        vllm_kwargs["vllm"] = VllmProviderConfig(
            gpu_memory_utilization=QUESTIONNAIRE_VLLM_GPU_MEMORY_UTILIZATION,
            max_model_len=max_model_len,
            tensor_parallel_size=QUESTIONNAIRE_VLLM_TENSOR_PARALLEL_SIZE,
        )

    questionnaire_config = InferenceConfig(
        model=QUESTIONNAIRE_MODEL,
        provider=QUESTIONNAIRE_PROVIDER,
        generation=GenerationConfig(
            max_new_tokens=QUESTIONNAIRE_MAX_NEW_TOKENS,
            temperature=0.0,
            do_sample=False,
        ),
        max_concurrent=QUESTIONNAIRE_MAX_CONCURRENT,
        timeout=QUESTIONNAIRE_TIMEOUT,
        retry=RetryConfig(max_retries=3, backoff_factor=2.0),
        continue_on_error=True,
        log_failures=True,
        openrouter=OpenRouterProviderConfig(provider_routing=ASSISTANT_OPENROUTER_PROVIDER_ROUTING),
        **vllm_kwargs,
    )
    provider = get_provider(QUESTIONNAIRE_PROVIDER, questionnaire_config)

    # Process questionnaire prompts in persona-major order. For remote/API
    # providers we keep the current one-persona-at-a-time loop. For vLLM only,
    # we optionally stack multiple personas into one super-batch to improve GPU
    # utilisation while preserving question batching within each persona.
    persona_batch_size = 1
    if QUESTIONNAIRE_PROVIDER == "vllm":
        persona_batch_size = max(1, QUESTIONNAIRE_VLLM_PERSONAS_PER_BATCH)
        print(
            "[Stage 2] vLLM persona stacking enabled: "
            f"{persona_batch_size} persona(s) per batch"
        )
    else:
        print(
            "[Stage 2] Persona stacking disabled for provider "
            f"{QUESTIONNAIRE_PROVIDER!r}; using 1 persona per batch"
        )

    use_logprobs_for_trait = (
        QUESTIONNAIRE_USE_LOGPROBS and QUESTIONNAIRE_PROVIDER == "vllm"
    )
    if use_logprobs_for_trait:
        print(
            f"[Stage 2] trait_mcq logprob mode ON "
            f"(top_logprobs={QUESTIONNAIRE_TOP_LOGPROBS}, "
            f"temperature={QUESTIONNAIRE_LOGPROB_TEMPERATURE})"
        )
    elif QUESTIONNAIRE_USE_LOGPROBS:
        print(
            "[Stage 2] QUESTIONNAIRE_USE_LOGPROBS=True but provider is "
            f"{QUESTIONNAIRE_PROVIDER!r}; logprob mode only supports 'vllm'. "
            "Falling back to greedy decoding."
        )

    # raw_responses.jsonl is kept open for the full stage and flushed after
    # each persona batch — safe for resume; failures are also written so the
    # file is the sole source of truth.
    with open(raw_responses_log, "a", encoding="utf-8") as log_fh:
        persona_batches = chunked(list(range(K)), persona_batch_size)
        for batch_idx, persona_batch in enumerate(persona_batches, start=1):
            text_entries: list[tuple[int, dict]] = []
            text_prompts: list[PromptInput] = []
            lp_entries: list[tuple[int, dict]] = []
            lp_prompts: list[PromptInput] = []
            active_personas: list[int] = []

            for k in persona_batch:
                pending_items = [
                    (item_idx, item) for item_idx, item in enumerate(items)
                    if (k, item["id"]) not in completed_cells
                ]
                if not pending_items:
                    continue
                active_personas.append(k)
                for _item_idx, item in pending_items:
                    prompt = _build_questionnaire_messages(conversations[k], item)
                    if use_logprobs_for_trait and item["type"] == "trait_mcq":
                        lp_entries.append((k, item))
                        lp_prompts.append(prompt)
                    else:
                        text_entries.append((k, item))
                        text_prompts.append(prompt)

            if not text_prompts and not lp_prompts:
                continue

            text_responses: list[str] = []
            if text_prompts:
                text_responses, _usage, _failed = await provider.generate_batch_with_metadata_async(
                    text_prompts
                )

            lp_outputs: list[dict] = []
            if lp_prompts:
                lp_outputs = await provider.generate_batch_logprobs_async(
                    lp_prompts,
                    max_tokens=1,
                    top_logprobs=QUESTIONNAIRE_TOP_LOGPROBS,
                    temperature=QUESTIONNAIRE_LOGPROB_TEMPERATURE,
                )

            # Parse and record; collect items needing a retry. For vLLM, this
            # retry pass is also persona-stacked within the current batch.
            # Logprob trait_mcq items do not retry — a missing choice letter in
            # top-k is recorded as a parse failure directly.
            retry_needed: list[tuple[int, dict, str]] = []
            for (k, item), raw_text in zip(text_entries, text_responses):
                item_id = item["id"]
                choice = _parse_item_response(item, raw_text)
                if choice is None and raw_text:
                    retry_needed.append((k, item, raw_text))
                elif choice is not None:
                    _record_response(
                        response_matrix, k, item, choice, raw_text,
                        item_to_cols, vig_scoring, likert_reverse,
                        log_fh,
                        trait_mcq_ids=trait_mcq_ids,
                    )
                    completed_cells.add((k, item_id))
                else:
                    parse_failures.append(
                        {"k": k, "item_id": item_id, "raw_response": raw_text}
                    )
                    completed_cells.add((k, item_id))
                    log_fh.write(
                        json.dumps(
                            {
                                "k": k,
                                "item_id": item_id,
                                "item_type": item["type"],
                                "parsed_choice": None,
                                "raw": raw_text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            for (k, item), lp_out in zip(lp_entries, lp_outputs):
                item_id = item["id"]
                raw_text = lp_out.get("text", "")
                per_token = lp_out.get("logprobs_per_token") or []
                first_token_logprobs: dict[str, float] = per_token[0] if per_token else {}
                probs, choice_mass = _parse_top_logprobs_to_choice_probs(
                    first_token_logprobs, num_choices=4,
                )
                if probs:
                    best_letter = max(probs, key=probs.get)
                    _fill_matrix_from_choice(
                        response_matrix, k, item_id, best_letter,
                        item_to_cols, vig_scoring, likert_reverse,
                        trait_mcq_ids=trait_mcq_ids,
                    )
                    log_fh.write(
                        json.dumps(
                            {
                                "k": k,
                                "item_id": item_id,
                                "item_type": item["type"],
                                "parsed_choice": best_letter,
                                "raw": raw_text,
                                "probs": {k_: round(v, 6) for k_, v in probs.items()},
                                "choice_mass": round(choice_mass, 6),
                                "scoring_method": "logprob",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    completed_cells.add((k, item_id))
                else:
                    parse_failures.append(
                        {"k": k, "item_id": item_id, "raw_response": raw_text,
                         "reason": "no choice letter in top logprobs"}
                    )
                    log_fh.write(
                        json.dumps(
                            {
                                "k": k,
                                "item_id": item_id,
                                "item_type": item["type"],
                                "parsed_choice": None,
                                "raw": raw_text,
                                "probs": {},
                                "choice_mass": 0.0,
                                "scoring_method": "logprob",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    completed_cells.add((k, item_id))

            for _attempt in range(MAX_PARSE_RETRIES):
                if not retry_needed:
                    break
                retry_prompts: list[PromptInput] = []
                for k, item, prev_raw in retry_needed:
                    msgs = list(conversations[k])
                    msgs.append({"role": "user", "content": _build_item_prompt(item)})
                    if item["type"] == "trait_mcq":
                        # Reconstruct the full prior assistant turn (prefill + continuation)
                        msgs.append({"role": "assistant", "content": TRAIT_MCQ_PREFILL + prev_raw})
                        msgs.append({"role": "user", "content": _retry_message(item)})
                        msgs.append({"role": "assistant", "content": TRAIT_MCQ_PREFILL})
                    else:
                        msgs.append({"role": "assistant", "content": prev_raw})
                        msgs.append({"role": "user", "content": _retry_message(item)})
                    retry_prompts.append(msgs)

                retry_responses, _, _ = await provider.generate_batch_with_metadata_async(
                    retry_prompts
                )

                still_needed: list[tuple[int, dict, str]] = []
                for (k, item, _prev_raw), retry_text in zip(
                    retry_needed, retry_responses
                ):
                    choice = _parse_item_response(item, retry_text)
                    if choice is not None:
                        _record_response(
                            response_matrix, k, item, choice, retry_text,
                            item_to_cols, vig_scoring, likert_reverse,
                            log_fh,
                            trait_mcq_ids=trait_mcq_ids,
                        )
                        completed_cells.add((k, item["id"]))
                    else:
                        still_needed.append((k, item, retry_text))
                retry_needed = still_needed

            for k, item, raw_text in retry_needed:
                item_id = item["id"]
                parse_failures.append(
                    {"k": k, "item_id": item_id, "raw_response": raw_text}
                )
                completed_cells.add((k, item_id))
                log_fh.write(
                    json.dumps(
                        {
                            "k": k,
                            "item_id": item_id,
                            "item_type": item["type"],
                            "parsed_choice": None,
                            "raw": raw_text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            log_fh.flush()

            done = len(completed_cells)
            total = K * N_items
            batch_start = active_personas[0] + 1
            batch_end = active_personas[-1] + 1
            print(
                f"[Stage 2] Persona batch {batch_idx}/{len(persona_batches)} "
                f"({batch_start}-{batch_end}/{K}) done | "
                f"{done}/{total} ({done/total*100:.1f}%)"
            )

    # Save outputs
    np.save(output_dir / "response_matrix.npy", response_matrix)

    with open(output_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for meta in metadata:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    # Save column_defs (used by downstream FA stages when loading from cache)
    with open(output_dir / "items.json", "w", encoding="utf-8") as f:
        json.dump(column_defs, f, indent=2, ensure_ascii=False)

    if parse_failures:
        with open(output_dir / "parse_failures.jsonl", "w", encoding="utf-8") as f:
            for pf in parse_failures:
                f.write(json.dumps(pf, ensure_ascii=False) + "\n")

    valid_count = int(np.sum(~np.isnan(response_matrix)))
    print(
        f"[Stage 2] Complete: {valid_count}/{K * N_cols} valid matrix cells | "
        f"{len(parse_failures)} parse failures"
    )

    return response_matrix, metadata


def _fill_matrix_from_choice(
    response_matrix: np.ndarray,
    k: int,
    item_id: str,
    choice: str | int,
    item_to_cols: dict[str, list[tuple[int, str | None]]],
    vig_scoring: dict[str, dict[str, dict[str, int]]],
    likert_reverse: dict[str, bool],
    trait_mcq_ids: set[str] | None = None,
) -> None:
    """Fill matrix columns for persona k given their choice on item_id.

    Column type is inferred from the column definition and the item-id set:
    - FC:        single column with dimension=None, encoded +1=A / -1=B
    - Vignette:  multiple columns (one per dimension) via option scoring dict
    - Likert:    single column with dimension set, encoded 1-5 with optional
                 reversal
    - trait_mcq: single column with dimension set, encoded as integer 1..4
                 (A=1, B=2, C=3, D=4). Trait-label-independent: answer_mapping
                 is used elsewhere for supervised TRAIT scoring.
    """
    cols = item_to_cols.get(item_id, [])
    if not cols:
        return

    col_idx_0, dim_0 = cols[0]

    if trait_mcq_ids is not None and item_id in trait_mcq_ids:
        # trait_mcq: single column, integer 1..4 from the chosen letter.
        if isinstance(choice, str) and len(choice) == 1 and "A" <= choice <= "D":
            response_matrix[k, col_idx_0] = float(ord(choice) - ord("A") + 1)
        return

    if dim_0 is None:
        # FC: one column, +1=A, -1=B
        response_matrix[k, col_idx_0] = 1.0 if choice == "A" else -1.0
    elif isinstance(choice, str):
        # Vignette: fill per-dimension scores from chosen option's scoring dict
        option_scores = vig_scoring.get(item_id, {}).get(choice, {})
        for col_idx, dim in cols:
            response_matrix[k, col_idx] = float(option_scores.get(dim, 0))
    else:
        # Likert: 1–5, apply reverse keying
        score = int(choice)
        if likert_reverse.get(item_id, False):
            score = 6 - score
        response_matrix[k, col_idx_0] = float(score)


def _record_response(
    response_matrix: np.ndarray,
    k: int,
    item: dict,
    choice: str | int,
    raw_text: str,
    item_to_cols: dict[str, list[tuple[int, str | None]]],
    vig_scoring: dict[str, dict[str, dict[str, int]]],
    likert_reverse: dict[str, bool],
    log_fh,
    trait_mcq_ids: set[str] | None = None,
) -> None:
    """Fill matrix and log raw response to an open file handle."""
    item_id = item["id"]
    _fill_matrix_from_choice(
        response_matrix, k, item_id, choice,
        item_to_cols, vig_scoring, likert_reverse,
        trait_mcq_ids=trait_mcq_ids,
    )
    log_fh.write(json.dumps({
        "k": k, "item_id": item_id,
        "item_type": item["type"],
        "parsed_choice": choice,
        "raw": raw_text,
    }, ensure_ascii=False) + "\n")


def run_stage_questionnaire() -> tuple[np.ndarray, list[dict], list[dict]]:
    """Apply the questionnaire to all rollout personas.

    Returns:
        Tuple of (response_matrix [K x N_cols], metadata, column_defs).
        column_defs describes each column (block, encoding, text, etc.).
    """
    rollout_dir = _rollout_dir()
    output_dir = _questionnaire_dir() / "questionnaire"
    run_id = _questionnaire_run_id()
    items, column_defs = _load_questionnaire()

    def _load_from_dir() -> tuple[np.ndarray, list[dict], list[dict]] | None:
        matrix_path = output_dir / "response_matrix.npy"
        items_path = output_dir / "items.json"
        if not matrix_path.exists():
            return None
        response_matrix = np.load(matrix_path)
        with open(output_dir / "metadata.jsonl", "r") as f:
            metadata = [json.loads(line) for line in f]
        # items.json stores column_defs in the new format
        if items_path.exists():
            with open(items_path, "r") as f:
                saved_cols = json.load(f)
            return response_matrix, metadata, saved_cols
        return response_matrix, metadata, column_defs

    # Check local cache
    cached = _load_from_dir()
    if cached is not None:
        print(f"[Stage 2] Questionnaire results already exist locally: {output_dir}")
        return cached

    # Check HF cache
    _ensure_hf_auth()
    hf_path = f"runs/{run_id}/questionnaire"
    if _hf_path_exists(hf_path):
        print(f"[Stage 2] Hydrating questionnaire results from HF: {run_id}")
        hydrate_dataset_subtree(
            repo_id=HF_REPO_ID,
            path_in_repo=hf_path,
            local_dir=output_dir,
        )
        cached = _load_from_dir()
        if cached is not None:
            return cached
        print("[Stage 2] HF hydration incomplete, regenerating...")

    # Generate
    response_matrix, metadata = asyncio.run(
        _apply_questionnaire_async(rollout_dir, items, column_defs, output_dir)
    )

    # Upload to HF
    try:
        upload_folder_to_dataset_repo(
            local_dir=output_dir,
            repo_id=HF_REPO_ID,
            path_in_repo=hf_path,
            commit_message=f"Questionnaire: {run_id}",
            ignore_patterns=[],
        )
        print(f"[Stage 2] Uploaded to HF: {hf_path}")
    except Exception as e:
        logger.warning("Failed to upload questionnaire to HF: %s", e)

    return response_matrix, metadata, column_defs


def run_stage_trait_scoring(metadata: list[dict] | None = None) -> dict | None:
    """Compute per-persona TRAIT scores + plots from stage-2 output.

    No-ops (and returns None) when the questionnaire does not contain
    trait_mcq items. When present, uses the items' ``answer_mapping`` (and
    any logprob ``probs`` stored in ``raw_responses.jsonl``) to compute a
    per-persona score per OCEAN trait.

    Writes ``trait_scores.csv`` + ``trait_scores_coverage.csv`` +
    ``trait_scores_summary.json`` and PDF plots to
    ``<questionnaire_dir>/questionnaire/trait_scores/``.

    Args:
        metadata: Optional stage-2 metadata list (to enrich the scores CSV
            with ``sample_id`` / ``input_group_id``). When None, attempts to
            load it from the questionnaire directory.

    Returns:
        The written file paths plus the scoring result, or None if skipped.
    """
    from src_dev.factor_analysis.trait_score_plots import plot_all_trait_plots
    from src_dev.factor_analysis.trait_scoring import (
        compute_trait_scores,
        save_trait_scores,
    )

    q_dir = _questionnaire_dir() / "questionnaire"
    raw_path = q_dir / "raw_responses.jsonl"
    if not raw_path.exists():
        print("[Trait-scoring] raw_responses.jsonl not found — skipping.")
        return None

    questionnaire_path = Path(QUESTIONNAIRE_PATH)
    # Peek at the questionnaire to decide whether to run.
    with open(questionnaire_path, "r", encoding="utf-8") as f:
        qn_raw = json.load(f)
    n_trait_items = len(qn_raw.get("block_4_trait_mcq", {}).get("items", []))
    if n_trait_items == 0:
        print("[Trait-scoring] No trait_mcq items in questionnaire — skipping.")
        return None

    output_dir = q_dir / "trait_scores"
    output_dir.mkdir(parents=True, exist_ok=True)

    if metadata is None:
        meta_path = q_dir / "metadata.jsonl"
        if meta_path.exists():
            metadata = [json.loads(line) for line in meta_path.open() if line.strip()]

    result = compute_trait_scores(
        raw_responses_path=raw_path,
        questionnaire_path=questionnaire_path,
        dynamic_mass_filter=QUESTIONNAIRE_DYNAMIC_MASS_FILTER,
        min_choice_mass=QUESTIONNAIRE_MIN_CHOICE_MASS,
        min_trait_coverage=QUESTIONNAIRE_MIN_TRAIT_COVERAGE,
    )
    written = save_trait_scores(result, output_dir, metadata=metadata)

    plot_paths = plot_all_trait_plots(
        result.scores,
        output_dir,
        title_prefix=_questionnaire_run_id(),
    )
    written.update(plot_paths)

    summary = {
        "n_personas": int(len(result.scores)),
        "trait_order": result.trait_order,
        "items_per_trait": result.items_per_trait,
        "mean_by_trait": result.scores.mean(skipna=True).to_dict(),
        "n_personas_scored_by_trait": result.scores.notna().sum().astype(int).to_dict(),
    }
    print(f"[Trait-scoring] {summary}")
    if result.min_trait_coverage > 0.0:
        print(
            f"[Trait-scoring] Persona filter: dropped cells with coverage < "
            f"{result.min_trait_coverage:.0%} of items per trait — removed "
            f"{result.filtered_by_trait}"
        )
    print(f"[Trait-scoring] Wrote {len(written)} files to {output_dir}")
    return {"paths": written, "result": result}


def _write_questionnaire_inspection_file(items: list[dict]) -> None:
    """Write a JSONL file joining rollout conversations with questionnaire responses.

    Each row is one (rollout, item) pair with the full conversation + the
    questionnaire question and answer appended as the final two messages.
    View with:
        uv run python -m src_dev.jsonl_tui.cli <path> --conversation-field messages
    """
    from collections import defaultdict

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
                prompt_text = _build_item_prompt(item)
                # Human-readable description of the item for display
                if item["type"] == "forced_choice":
                    display_text = f'FC: {item["option_a"]["text"][:60]}... vs {item["option_b"]["text"][:60]}...'
                elif item["type"] == "vignette":
                    display_text = f'Vignette [{item["title"]}]: {item["scenario"][:80]}...'
                elif item["type"] == "trait_mcq":
                    display_text = f'TRAIT [{item["primary_dimension"]}]: {item["question"][:80]}...'
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

    # Also write a self-contained HTML file for sharing
    html_path = q_dir / "conversations_with_questionnaire.html"
    _write_conversation_html(out, html_path)
    print(f"  HTML export: {html_path}")


def _write_conversation_html(jsonl_path: Path, html_path: Path) -> None:
    """Write a self-contained HTML viewer for conversation JSONL files.

    Renders each record's 'messages' field as a chat transcript with
    user/assistant bubbles. Navigate between records with arrow keys.
    """
    import html as html_mod

    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Build JS-friendly data
    js_records = []
    for rec in records:
        msgs = rec.get("messages", [])
        js_records.append({
            "sample_id": rec.get("sample_id", ""),
            "item_text": rec.get("item_text", ""),
            "parsed": rec.get("parsed"),
            "messages": [{"role": m["role"], "content": m["content"]} for m in msgs],
        })

    data_json = json.dumps(js_records, ensure_ascii=False)
    title = html_mod.escape(jsonl_path.stem)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #111827; color: #e5e7eb;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 14px; line-height: 1.5;
    display: flex; flex-direction: column; height: 100vh; overflow: hidden;
  }}
  #topbar {{
    background: #0e7490; color: #fff; font-weight: bold;
    padding: 8px 16px; display: flex; gap: 24px; flex-shrink: 0;
  }}
  #topbar .dim {{ opacity: 0.7; }}
  #scroll-area {{
    flex: 1; overflow-y: auto; padding: 16px 24px 80px;
    max-width: 900px; margin: 0 auto; width: 100%;
  }}
  .msg {{
    margin-bottom: 12px; padding: 10px 14px;
    border-radius: 8px; white-space: pre-wrap; word-break: break-word;
  }}
  .msg-user {{
    background: #1e3a5f; border-left: 3px solid #60a5fa;
  }}
  .msg-assistant {{
    background: #1a2e1a; border-left: 3px solid #4ade80;
  }}
  .msg-system {{
    background: #2d2235; border-left: 3px solid #c084fc;
    font-style: italic;
  }}
  .msg-questionnaire {{
    background: #3b2f1a; border-left: 3px solid #facc15;
  }}
  .msg-answer {{
    background: #1a3a2a; border-left: 3px solid #22d3ee;
    font-size: 18px; font-weight: bold; text-align: center;
  }}
  .role-label {{
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em; margin-bottom: 4px; opacity: 0.7;
  }}
  .role-user .role-label {{ color: #60a5fa; }}
  .role-assistant .role-label {{ color: #4ade80; }}
  .role-system .role-label {{ color: #c084fc; }}
  .separator {{
    text-align: center; color: #facc15; font-weight: bold;
    margin: 20px 0 8px; padding: 6px;
    border-top: 1px dashed #facc15; border-bottom: 1px dashed #facc15;
    font-size: 12px; letter-spacing: 0.1em;
  }}
  #bottombar {{
    background: #d1d5db; color: #111; font-weight: 600;
    padding: 5px 16px; font-size: 12px; flex-shrink: 0;
  }}
</style>
</head>
<body>
<div id="topbar">
  <span id="tb-nav"></span>
  <span class="dim" id="tb-item"></span>
  <span class="dim" id="tb-score"></span>
</div>
<div id="scroll-area"></div>
<div id="bottombar">
  ← → &nbsp;Navigate between conversations &nbsp;|&nbsp;
  Home / End &nbsp;First / Last
</div>
<script>
const RECORDS = {data_json};
let idx = 0;

function render() {{
  const rec = RECORDS[idx];
  const msgs = rec.messages;
  const n = msgs.length;

  document.getElementById('tb-nav').textContent =
    `Record ${{idx + 1}} / ${{RECORDS.length}}  (${{rec.sample_id}})`;
  document.getElementById('tb-item').textContent =
    rec.item_text ? `Item: ${{rec.item_text.substring(0, 80)}}` : '';
  document.getElementById('tb-score').textContent =
    rec.parsed != null ? `Score: ${{rec.parsed}}` : '';

  const area = document.getElementById('scroll-area');
  area.innerHTML = '';

  // Conversation messages (all except last 2 which are the questionnaire)
  const convEnd = n >= 2 ? n - 2 : n;
  for (let i = 0; i < convEnd; i++) {{
    area.appendChild(makeMsg(msgs[i]));
  }}

  // Questionnaire separator + final 2 messages
  if (n >= 2) {{
    const sep = document.createElement('div');
    sep.className = 'separator';
    sep.textContent = '▼ QUESTIONNAIRE ▼';
    area.appendChild(sep);
    area.appendChild(makeMsg(msgs[n - 2], 'msg-questionnaire'));
    area.appendChild(makeMsg(msgs[n - 1], 'msg-answer'));
  }}

  area.scrollTop = area.scrollHeight;
}}

function makeMsg(msg, extraClass) {{
  const div = document.createElement('div');
  const role = msg.role || 'user';
  div.className = `msg msg-${{role}} role-${{role}}` + (extraClass ? ` ${{extraClass}}` : '');

  const label = document.createElement('div');
  label.className = 'role-label';
  label.textContent = role;
  div.appendChild(label);

  const body = document.createElement('div');
  body.textContent = msg.content;
  div.appendChild(body);

  return div;
}}

document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight' || e.key === 'l') {{
    idx = Math.min(idx + 1, RECORDS.length - 1); render();
  }} else if (e.key === 'ArrowLeft' || e.key === 'h') {{
    idx = Math.max(idx - 1, 0); render();
  }} else if (e.key === 'Home' || e.key === 'g') {{
    idx = 0; render();
  }} else if (e.key === 'End' || e.key === 'G') {{
    idx = RECORDS.length - 1; render();
  }}
}});

render();
</script>
</body>
</html>"""

    html_path.write_text(html_content, encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3: FACTOR ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════


def _preprocess_response_matrix(
    response_matrix: np.ndarray,
    metadata: list[dict],
    column_defs: list[dict],
    do_residualize: bool = False,
    variance_export_path: Path | None = None,
) -> tuple[np.ndarray, list[dict], list[dict], np.ndarray | None]:
    """Preprocess the response matrix for factor analysis.

    Args:
        response_matrix: Raw persona × item matrix.
        metadata: Per-row metadata (same length as ``response_matrix``).
        column_defs: Per-column definitions (same length as item axis).
        do_residualize: If True, subtract per-prompt-group means.
        variance_export_path: If provided, write a JSONL file at this path with
            one row per column ranked by pre-filter variance (computed after
            row-level filtering). Useful for inspecting which questionnaire
            items actually discriminate across personas.

    Returns:
        Tuple of (cleaned matrix, filtered metadata, filtered column_defs, group_ids or None).
    """
    K, M = response_matrix.shape

    # Drop rows with any missing values (parse failures).
    # Mean imputation would attenuate correlations and bias factor loadings;
    # dropping is cleaner when the parse-success rate is high.
    missing_per_row = np.sum(np.isnan(response_matrix), axis=1)
    row_mask = missing_per_row == 0
    data = response_matrix[row_mask].copy()
    meta_filtered = [m for m, keep in zip(metadata, row_mask) if keep]
    n_dropped = K - data.shape[0]
    print(f"  Kept {data.shape[0]}/{K} rows (dropped {n_dropped} with any missing values)")
    if n_dropped > 0:
        # Report missing-value distribution for diagnostics
        missing_counts = missing_per_row[~row_mask]
        print(f"  Dropped row missing-value counts: "
              f"median={np.median(missing_counts):.0f}, "
              f"max={np.max(missing_counts):.0f}")

    # Drop high-variance personas (potential incoherent/garbage rollouts).
    # Computed on complete rows before column filtering so the threshold is
    # stable regardless of which columns survive the low-variance filter.
    if HIGH_VARIANCE_PERSONA_DROP_PCT > 0:
        row_vars = np.var(data, axis=1)
        threshold = np.percentile(row_vars, 100 - HIGH_VARIANCE_PERSONA_DROP_PCT)
        keep = row_vars <= threshold
        n_before = data.shape[0]
        data = data[keep]
        meta_filtered = [m for m, k in zip(meta_filtered, keep) if k]
        n_hi_var_dropped = n_before - data.shape[0]
        print(
            f"  Dropped {n_hi_var_dropped}/{n_before} high-variance personas "
            f"(top {HIGH_VARIANCE_PERSONA_DROP_PCT}%, var > {threshold:.3f})"
        )

    # Drop low-variance columns
    col_var = np.var(data, axis=0)
    col_mask = col_var >= MIN_ITEM_VARIANCE

    if variance_export_path is not None:
        variance_export_path.parent.mkdir(parents=True, exist_ok=True)
        ranked = sorted(
            zip(column_defs, col_var, col_mask),
            key=lambda r: float(r[1]),
            reverse=True,
        )
        with variance_export_path.open("w", encoding="utf-8") as f:
            for col_def, var, keep in ranked:
                row = {
                    "variance": float(var),
                    "kept": bool(keep),
                    "block": col_def.get("block"),
                    "item_id": col_def.get("item_id"),
                    "col_id": col_def.get("col_id"),
                    "dimension": col_def.get("dimension"),
                    "question": col_def.get("text"),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  Wrote ranked item variances to {variance_export_path}")

    data = data[:, col_mask]
    cols_filtered = [col for col, keep in zip(column_defs, col_mask) if keep]
    dropped_cols = int(np.sum(~col_mask))
    if dropped_cols > 0:
        print(f"  Dropped {dropped_cols}/{M} low-variance columns (var < {MIN_ITEM_VARIANCE})")
    print(f"  Final matrix shape: {data.shape}")

    # Residualize if requested
    group_ids = None
    if do_residualize:
        # Check that residualization is meaningful (need >1 sample per group
        # on average, otherwise subtracting group means zeros everything out)
        group_counts: dict[str, int] = {}
        for m in meta_filtered:
            gid = m.get("input_group_id", m["sample_id"])
            group_counts[gid] = group_counts.get(gid, 0) + 1
        max_group_size = max(group_counts.values()) if group_counts else 0

        if max_group_size <= 1:
            print("  Skipping residualization: all groups have size 1 "
                  "(need NUM_ROLLOUTS_PER_PROMPT >= 2)")
        else:
            data, _group_means, group_inv = residualize(
                data, meta_filtered, group_field="input_group_id",
            )
            group_ids = group_inv
            n_groups = len(group_counts)
            print(f"  Residualized across {n_groups} groups")

            # Re-filter zero-variance columns created by residualization
            col_var_post = np.var(data, axis=0)
            col_mask_post = col_var_post >= MIN_ITEM_VARIANCE
            if not col_mask_post.all():
                dropped_post = int(np.sum(~col_mask_post))
                data = data[:, col_mask_post]
                cols_filtered = [c for c, keep in zip(cols_filtered, col_mask_post) if keep]
                print(f"  Dropped {dropped_post} zero-variance columns after residualization")

    return data, meta_filtered, cols_filtered, group_ids


def _plot_parallel_analysis(
    real_eigenvalues: np.ndarray,
    random_threshold: np.ndarray,
    n_recommended: int,
    label: str,
    save_path: Path,
    max_components: int = 30,
) -> None:
    """Plot Horn's parallel analysis scree plot and save to PNG."""
    import matplotlib.pyplot as plt

    n = min(len(real_eigenvalues), max_components)
    x = np.arange(1, n + 1)
    real = np.asarray(real_eigenvalues)[:n]
    rand = np.asarray(random_threshold)[:n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, real, "o-", color="#2563eb", linewidth=2, markersize=5, label="Actual eigenvalues")
    ax.plot(x, rand, "s--", color="#dc2626", linewidth=1.5, markersize=4, label="95th percentile (random)")

    # Shade the retained region
    if n_recommended > 0:
        ax.axvspan(0.5, n_recommended + 0.5, alpha=0.08, color="#2563eb")
        ax.axvline(n_recommended + 0.5, color="#6b7280", linestyle=":", linewidth=1)
        ax.text(
            n_recommended + 0.5, ax.get_ylim()[1] * 0.95,
            f"  {n_recommended} factors",
            fontsize=11, color="#374151", va="top",
        )

    ax.axhline(1.0, color="#9ca3af", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Component", fontsize=12)
    ax.set_ylabel("Eigenvalue", fontsize=12)
    ax.set_title(f"Horn's Parallel Analysis — {label}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0.5, n + 0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved parallel analysis plot: {save_path}")


def run_stage_factor_analysis(
    response_matrix: np.ndarray,
    metadata: list[dict],
    column_defs: list[dict],
    items: list[dict] | None = None,
) -> dict:
    """Run factor analysis on the response matrix.

    ``items`` is the full questionnaire item list (used to build rich
    column descriptions for the trait-oriented FA pass).
    """
    base_dir = _questionnaire_dir() / "factor_analysis"
    base_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for do_residualize in RESIDUALIZE_OPTIONS:
        resid_label = "residualized" if do_residualize else "raw"

        print(f"\n[Stage 3] Factor analysis ({resid_label})")
        print("=" * 60)

        variance_export_path = (
            base_dir / resid_label / "item_variances_ranked.jsonl"
        )
        data, meta_filtered, cols_filtered, group_ids = _preprocess_response_matrix(
            response_matrix, metadata, column_defs,
            do_residualize=do_residualize,
            variance_export_path=variance_export_path,
        )

        if do_residualize and group_ids is None:
            print(f"  Residualization skipped — {resid_label} analysis would duplicate raw. Skipping.")
            all_results[resid_label] = {"n_factors": 0, "note": "residualization not applicable"}
            continue

        # Adequacy tests
        print("\n  Adequacy tests:")
        adeq = adequacy_tests(data)

        # Parallel analysis (Horn's method)
        print("\n  Parallel analysis:")
        pa_result = parallel_analysis(data, random_state=SEED)
        n_factors = pa_result["n_recommended"]
        if FA_N_FACTORS_OVERRIDE is not None:
            print(f"  Override: using {FA_N_FACTORS_OVERRIDE} factors (Horn's recommended {n_factors})")
            n_factors = FA_N_FACTORS_OVERRIDE

        # Save parallel analysis results
        pa_dir = base_dir / resid_label
        pa_dir.mkdir(parents=True, exist_ok=True)
        with open(pa_dir / "parallel_analysis.json", "w") as f:
            json.dump({
                "n_recommended": n_factors,
                "real_eigenvalues": pa_result["real_eigenvalues"].tolist(),
                "random_threshold": pa_result["random_threshold"].tolist(),
                "adequacy": {
                    "bartlett_chi2": adeq["bartlett_chi2"],
                    "bartlett_p": adeq["bartlett_p"],
                    "kmo_overall": adeq["kmo_overall"],
                },
            }, f, indent=2)

        # Plot Horn's parallel analysis (scree plot with random threshold)
        _plot_parallel_analysis(
            pa_result["real_eigenvalues"],
            pa_result["random_threshold"],
            n_factors,
            resid_label,
            pa_dir / "parallel_analysis.png",
        )

        if n_factors == 0:
            print(f"  No factors recommended for {resid_label} — skipping FA.")
            all_results[resid_label] = {"n_factors": 0, "parallel_analysis": pa_result}
            continue

        # Run factor analysis with each rotation
        for rotation in FA_ROTATIONS:
            print(f"\n  Factor analysis: {n_factors} factors, rotation={rotation}")
            fa_result = run_factor_analysis(
                data, n_factors=n_factors, method=FA_METHOD, rotation=rotation,
            )

            # Save
            fa_path = pa_dir / f"fa_{n_factors}_{FA_METHOD}_{rotation}"
            save_factor_analysis(
                fa_result, fa_path,
                config={
                    "n_factors": n_factors,
                    "method": FA_METHOD,
                    "rotation": rotation,
                    "residualized": do_residualize,
                    "n_samples": data.shape[0],
                    "n_cols": data.shape[1],
                },
            )

            # Save column labels for the loadings
            with open(str(fa_path) + "_item_labels.json", "w") as f:
                json.dump([
                    {
                        "col_id": col["col_id"],
                        "text": col["text"],
                        "block": col["block"],
                        "dimension": col.get("dimension"),
                        "reverse_keyed": col.get("reverse_keyed", False),
                    }
                    for col in cols_filtered
                ], f, indent=2, ensure_ascii=False)

            # Factor–trait (OCEAN) alignment analysis. Only meaningful when
            # every FA row carries a primary_dimension label (trait_mcq /
            # likert blocks). Mean-signed loading is not trait-interpretable
            # here because the response matrix is letter-encoded (1..4) rather
            # than trait-oriented — but top-K counts and |mean loading| by
            # trait remain valid, so we save them and flag the signed caveat.
            item_dims_all = [col.get("dimension") for col in cols_filtered]
            if all(d is not None for d in item_dims_all):
                ocean_canonical = [
                    "openness", "conscientiousness", "extraversion",
                    "agreeableness", "neuroticism",
                ]
                present = [t for t in ocean_canonical if t in item_dims_all]
                extras = sorted(set(item_dims_all) - set(present))
                trait_order = present + extras

                alignment = compute_factor_trait_alignment(
                    loadings=fa_result["loadings"],
                    item_dims=item_dims_all,
                    trait_order=trait_order,
                    top_k=min(20, len(item_dims_all)),
                )
                align_dir = pa_dir / f"fa_{n_factors}_{FA_METHOD}_{rotation}_alignment"
                save_alignment(alignment, align_dir)
                plot_all_alignment(
                    alignment, align_dir,
                    title_prefix=f"{resid_label} / {rotation}",
                )
                print(
                    f"  [Alignment] {rotation}: factor→trait winners "
                    f"(top-{alignment.top_k}):"
                )
                for fi, label in enumerate(alignment.factor_labels):
                    counts = alignment.top_k_count[fi]
                    best = int(np.argmax(counts))
                    full = dict(zip(
                        alignment.trait_order,
                        [int(c) for c in counts],
                    ))
                    print(
                        f"    {label}: {alignment.trait_order[best]} "
                        f"({int(counts[best])}/{alignment.top_k}) — {full}"
                    )
            else:
                print(
                    f"  [Alignment] Skipped for {rotation}: "
                    f"{sum(d is None for d in item_dims_all)} / "
                    f"{len(item_dims_all)} items lack a primary_dimension."
                )

            key = f"{resid_label}_{rotation}"
            all_results[key] = {
                "fa_result": fa_result,
                "column_defs": cols_filtered,
                "metadata": meta_filtered,
                "data": data,
                "n_factors": n_factors,
                "parallel_analysis": pa_result,
                "save_dir": base_dir / key,
                "encoding": "letter",
            }

    # Trait-oriented FA pass: builds a parallel set of results using the
    # trait-direction score matrix (so signed loadings are trait-interpretable).
    # Merges into all_results with keys like "trait_oriented_{rotation}", and
    # flows through the shared plotting / HTML / labeling loops below.
    trait_oriented_results = _run_trait_oriented_fa_pass(
        metadata=metadata,
        items=items,
    )
    all_results.update(trait_oriented_results)

    # Generate visualisations and factor-extreme HTML exports for each FA result
    for key, result in all_results.items():
        if result.get("n_factors", 0) == 0 or "fa_result" not in result:
            continue
        result_dir = Path(result.get("save_dir", base_dir / key))
        viz_dir = result_dir / "plots"
        viz_dir.mkdir(parents=True, exist_ok=True)
        _plot_fa_visualisations(
            fa_result=result["fa_result"],
            column_defs=result["column_defs"],
            metadata=result["metadata"],
            data=result["data"],
            label=key,
            save_dir=viz_dir,
        )
        _export_factor_extremes_html(
            fa_result=result["fa_result"],
            column_defs=result["column_defs"],
            metadata=result["metadata"],
            label=key,
            save_dir=result_dir,
        )

        # Trait-aware views across all items (not just top-K). Only meaningful
        # when every FA row carries a primary_dimension. For letter-encoded
        # results, signed views get a caveat because A/B/C/D is shuffled per
        # item, so sign ≠ trait-direction. For trait-oriented results, signed
        # loadings ARE trait-interpretable (no caveat).
        cols = result["column_defs"]
        item_dims_all = [col.get("dimension") for col in cols]
        if all(d is not None for d in item_dims_all):
            is_trait_oriented = result.get("encoding") == "trait_oriented"
            _plot_trait_aware_fa_visualisations(
                loadings=result["fa_result"]["loadings"],
                item_dims=item_dims_all,
                save_dir=viz_dir,
                label=key,
                top_k=min(20, len(item_dims_all)),
                signed_caveat=(
                    None if is_trait_oriented
                    else "letter-encoded: sign not trait-interpretable"
                ),
            )

    return all_results


def _run_trait_oriented_fa_pass(
    metadata: list[dict] | None = None,
    items: list[dict] | None = None,
) -> dict:
    """Build the trait-oriented response matrix and run FA + alignment.

    Uses FA_METHOD / FA_ROTATIONS / FA_N_FACTORS_OVERRIDE so results are
    config-coherent with the letter-encoded pass. Produces, per rotation,
    the same result-dict shape the letter-encoded pass returns, so the
    shared downstream plotting / HTML export / labeling loops can consume
    both.

    Outputs land in ``<questionnaire_dir>/factor_analysis_trait_oriented/``.
    Per-rotation sub-directories mirror the letter-encoded layout
    (``<output_dir>/trait_oriented_{rotation}/plots/`` etc.), so
    ``_export_factor_extremes_html`` and ``_plot_fa_visualisations`` can
    write alongside.

    Args:
        metadata: Stage-2 metadata list aligned with response_matrix rows
            (indexed by persona k). Filtered to the kept persona rows.
        items: Full questionnaire items list (used for LLM labelling
            descriptions downstream).

    Returns:
        ``{key: result}`` keyed by ``trait_oriented_{rotation}``. Empty
        dict if raw_responses.jsonl or trait_mcq items are missing.
    """
    q_dir = _questionnaire_dir() / "questionnaire"
    raw_path = q_dir / "raw_responses.jsonl"
    if not raw_path.exists():
        print("\n[Trait-oriented FA] raw_responses.jsonl not found — skipping.")
        return {}

    qn_path = Path(QUESTIONNAIRE_PATH)
    with open(qn_path, "r", encoding="utf-8") as f:
        qn_raw = json.load(f)
    trait_items_raw = qn_raw.get("block_4_trait_mcq", {}).get("items", [])
    if not trait_items_raw:
        print("\n[Trait-oriented FA] No trait_mcq items in questionnaire — skipping.")
        return {}

    # Build per-item info (text + dimension) keyed by id for column_defs.
    trait_item_info: dict[str, dict] = {
        str(it["id"]): {
            "text": it.get("question", ""),
            "dimension": it["primary_dimension"],
        }
        for it in trait_items_raw
    }

    output_dir = _questionnaire_dir() / "factor_analysis_trait_oriented"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("[Stage 3b] Trait-oriented factor analysis")
    print("=" * 60)
    print(f"  Building trait-oriented matrix from {raw_path}")

    tom = build_trait_oriented_matrix(
        raw_responses_path=raw_path,
        questionnaire_path=qn_path,
    )
    matrix = tom.matrix
    K, N = matrix.shape
    print(f"  Raw matrix: {K} personas × {N} items  traits={tom.trait_order}")

    # Drop personas with > 20% missing (matches run_trait_oriented_fa.py default).
    max_missing_frac = 0.2
    missing_frac = np.mean(np.isnan(matrix), axis=1)
    keep_rows = missing_frac <= max_missing_frac
    data = matrix[keep_rows]
    kept_k = [tom.k_index[i] for i in np.where(keep_rows)[0]]
    print(
        f"  Kept {int(keep_rows.sum())}/{K} personas "
        f"(≤{max_missing_frac:.0%} missing)"
    )

    # Column-mean impute remaining NaNs (rows already filtered above).
    nan_mask = np.isnan(data)
    if nan_mask.any():
        col_means = np.nanmean(data, axis=0)
        data = data.copy()
        inds = np.where(nan_mask)
        data[inds] = np.take(col_means, inds[1])
        print(f"  Mean-imputed {int(nan_mask.sum())} remaining missing cells")

    # Drop zero-variance columns.
    col_var = np.var(data, axis=0)
    live_cols = col_var > 1e-10
    if not live_cols.all():
        dropped = [
            (tom.item_ids[i], tom.item_dims[i])
            for i in np.where(~live_cols)[0]
        ]
        print(
            f"  Dropping {int((~live_cols).sum())} zero-variance items: {dropped}"
        )
    data = data[:, live_cols]
    item_ids = [tom.item_ids[i] for i in np.where(live_cols)[0]]
    item_dims = [tom.item_dims[i] for i in np.where(live_cols)[0]]

    # Build column_defs mirroring the letter-encoded trait_mcq shape so the
    # shared plotting / labeling code can consume them uniformly.
    column_defs_to = [
        {
            "col_id": iid,
            "item_id": iid,
            "block": "trait_mcq",
            "dimension": item_dims[i],
            "text": trait_item_info.get(iid, {}).get("text", iid),
            "encoding": "trait_score_0-1",
        }
        for i, iid in enumerate(item_ids)
    ]

    # Filter caller-provided metadata to the kept personas. Fall back to a
    # synthetic minimal metadata list keyed by k if none was supplied.
    metadata_to: list[dict]
    if metadata is not None:
        try:
            metadata_to = [dict(metadata[k]) for k in kept_k]
        except IndexError:
            print(
                "  [Warn] metadata list shorter than tom.k_index; "
                "falling back to synthetic metadata."
            )
            metadata_to = [{"k": int(k), "sample_id": f"k{k}"} for k in kept_k]
    else:
        metadata_to = [{"k": int(k), "sample_id": f"k{k}"} for k in kept_k]

    # Standardize for FA.
    print("\n  Adequacy tests (standardized data):")
    data_z = (data - data.mean(axis=0)) / data.std(axis=0, ddof=0)
    adequacy = adequacy_tests(data_z)

    # Parallel analysis (or use override).
    print("\n  Parallel analysis:")
    pa = parallel_analysis(data_z, random_state=SEED, method="permutation")
    n_factors = int(pa["n_recommended"])
    if FA_N_FACTORS_OVERRIDE is not None:
        print(
            f"  Override: using {FA_N_FACTORS_OVERRIDE} factors "
            f"(parallel analysis recommended {n_factors})"
        )
        n_factors = int(FA_N_FACTORS_OVERRIDE)

    with open(output_dir / "parallel_analysis.json", "w") as f:
        json.dump({
            "n_recommended": int(pa["n_recommended"]),
            "n_used": n_factors,
            "real_eigenvalues": pa["real_eigenvalues"].tolist(),
            "random_threshold": pa["random_threshold"].tolist(),
            "adequacy": {
                "kmo_overall": adequacy["kmo_overall"],
                "bartlett_p": adequacy["bartlett_p"],
            },
        }, f, indent=2)

    _plot_parallel_analysis(
        pa["real_eigenvalues"],
        pa["random_threshold"],
        n_factors,
        "trait-oriented",
        output_dir / "parallel_analysis.png",
    )

    if n_factors == 0:
        print("  No factors — skipping trait-oriented FA.")
        return {}

    # Persist matrix metadata (item ↔ trait alignment) once.
    with open(output_dir / "matrix_items.json", "w", encoding="utf-8") as f:
        json.dump({
            "item_ids": item_ids,
            "item_dims": item_dims,
            "trait_order": tom.trait_order,
            "kept_k": [int(k) for k in kept_k],
            "n_personas_used": int(data.shape[0]),
            "n_items_used": int(data.shape[1]),
        }, f, indent=2)

    trait_results: dict = {}

    for rotation in FA_ROTATIONS:
        key = f"trait_oriented_{rotation}"
        save_dir = output_dir / key
        save_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n  Trait-oriented FA: {n_factors} factors, "
            f"method={FA_METHOD}, rotation={rotation}"
        )
        fa = run_factor_analysis(
            data_z, n_factors=n_factors, method=FA_METHOD, rotation=rotation,
        )
        fa_path = save_dir / f"fa_trait_oriented_{n_factors}_{FA_METHOD}_{rotation}"
        save_factor_analysis(
            fa, fa_path,
            config={
                "n_factors": n_factors,
                "method": FA_METHOD,
                "rotation": rotation,
                "encoding": "trait_oriented",
                "n_personas_used": int(data.shape[0]),
                "n_items_used": int(data.shape[1]),
                "adequacy": {
                    "kmo_overall": adequacy["kmo_overall"],
                    "bartlett_p": adequacy["bartlett_p"],
                },
            },
        )

        # Item labels for the loadings, mirroring the letter-encoded pass.
        with open(str(fa_path) + "_item_labels.json", "w") as f:
            json.dump([
                {
                    "col_id": cd["col_id"],
                    "text": cd["text"],
                    "block": cd["block"],
                    "dimension": cd.get("dimension"),
                    "reverse_keyed": False,
                }
                for cd in column_defs_to
            ], f, indent=2, ensure_ascii=False)

        alignment = compute_factor_trait_alignment(
            loadings=fa["loadings"],
            item_dims=item_dims,
            trait_order=tom.trait_order,
            top_k=min(20, len(item_dims)),
        )
        align_dir = save_dir / "alignment"
        save_alignment(alignment, align_dir)
        plot_all_alignment(
            alignment, align_dir,
            title_prefix=f"trait-oriented / {rotation}",
        )

        print(
            f"  [Alignment] trait-oriented {rotation}: "
            f"factor→trait winners (top-{alignment.top_k}):"
        )
        for fi, fl_label in enumerate(alignment.factor_labels):
            counts = alignment.top_k_count[fi]
            best = int(np.argmax(counts))
            full = dict(zip(
                alignment.trait_order,
                [int(c) for c in counts],
            ))
            signed_dom_idx = int(np.argmax(np.abs(alignment.mean_signed_loading[fi])))
            signed_val = alignment.mean_signed_loading[fi, signed_dom_idx]
            print(
                f"    {fl_label}: top-K winner={alignment.trait_order[best]} "
                f"({int(counts[best])}/{alignment.top_k}); "
                f"strongest signed mean: {alignment.trait_order[signed_dom_idx]}={signed_val:+.3f} "
                f"— full counts {full}"
            )

        trait_results[key] = {
            "fa_result": fa,
            "column_defs": column_defs_to,
            "metadata": metadata_to,
            "data": data,
            "n_factors": n_factors,
            "parallel_analysis": pa,
            "save_dir": save_dir,
            "encoding": "trait_oriented",
        }

    return trait_results


# ═════════════════════════════════════════════════════════════════════════════
# FACTOR ANALYSIS VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════════════


def _plot_fa_visualisations(
    fa_result: dict,
    column_defs: list[dict],
    metadata: list[dict],
    data: np.ndarray,
    label: str,
    save_dir: Path,
) -> None:
    """Generate a suite of diagnostic visualisations for factor analysis results.

    Plots saved to save_dir:
        1_loading_heatmap.png       — Items × Factors heatmap, clustered by dominant factor
        2_score_scatter_matrix.png  — Pairwise factor score scatter plots
        3_communalities.png         — Per-item communality bar chart
        4_score_distributions.png   — Per-factor score histograms
        5_factor_correlations.png   — Inter-factor correlation heatmap (oblimin only)
        7_scores_by_archetype.png   — Factor scores grouped by interviewer archetype
        8_prompt_icc.png            — Within-prompt vs between-prompt variance (ICC)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    loadings = fa_result["loadings"]  # [n_items, n_factors]
    scores = fa_result["scores"]      # [n_personas, n_factors]
    communalities = fa_result["communalities"]  # [n_items]
    n_factors = loadings.shape[1]

    print(f"\n  [Viz] Generating plots for {label} ({n_factors} factors)...")

    # ── 1. Loading heatmap ──────────────────────────────────────────────────
    _plot_loading_heatmap(loadings, column_defs, n_factors, save_dir, label, plt, TwoSlopeNorm)

    # ── 2. Factor score scatter matrix ──────────────────────────────────────
    _plot_score_scatter_matrix(scores, n_factors, save_dir, label, plt)

    # ── 3. Communalities bar chart ──────────────────────────────────────────
    _plot_communalities(communalities, column_defs, save_dir, label, plt)

    # ── 4. Factor score distributions ───────────────────────────────────────
    _plot_score_distributions(scores, n_factors, save_dir, label, plt)

    # ── 5. Inter-factor correlation matrix ──────────────────────────────────
    _plot_factor_correlations(fa_result, n_factors, save_dir, label, plt, TwoSlopeNorm)

    # ── 7. Factor scores by archetype ───────────────────────────────────────
    _plot_scores_by_archetype(scores, metadata, n_factors, save_dir, label, plt)

    # ── 8. Prompt ICC ───────────────────────────────────────────────────────
    _plot_prompt_icc(scores, metadata, n_factors, save_dir, label, plt)

    print(f"  [Viz] All plots saved to {save_dir}")


def _plot_loading_heatmap(
    loadings: np.ndarray,
    column_defs: list[dict],
    n_factors: int,
    save_dir: Path,
    label: str,
    plt,
    TwoSlopeNorm,
) -> None:
    """Loading heatmap: items (rows) × factors (columns), clustered by dominant factor."""
    n_items = loadings.shape[0]

    # Sort items by their dominant factor, then by loading magnitude within factor
    dominant_factor = np.argmax(np.abs(loadings), axis=1)
    sort_keys = []
    for i in range(n_items):
        df = dominant_factor[i]
        # Primary sort: dominant factor index
        # Secondary sort: negative absolute loading (so highest loads first)
        sort_keys.append((df, -np.abs(loadings[i, df])))
    sort_order = sorted(range(n_items), key=lambda i: sort_keys[i])

    sorted_loadings = loadings[sort_order]
    sorted_labels = []
    for idx in sort_order:
        cd = column_defs[idx]
        text = cd["text"][:55]
        block_tag = cd["block"][:2].upper()
        rev = "(R) " if cd.get("reverse_keyed", False) else ""
        sorted_labels.append(f"[{block_tag}] {rev}{text}")

    # Compute figure height dynamically — ~0.2 inches per item, min 6
    fig_h = max(6, n_items * 0.22)
    fig_w = max(6, 3 + n_factors * 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = max(0.8, np.max(np.abs(sorted_loadings)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(sorted_loadings, aspect="auto", cmap="RdBu_r", norm=norm)

    ax.set_xticks(range(n_factors))
    ax.set_xticklabels([f"F{i}" for i in range(n_factors)], fontsize=10)
    ax.set_yticks(range(n_items))
    ax.set_yticklabels(sorted_labels, fontsize=max(5, min(8, 200 / n_items)))

    # Draw horizontal lines between factor clusters
    prev_df = dominant_factor[sort_order[0]]
    for row_i, orig_idx in enumerate(sort_order):
        df = dominant_factor[orig_idx]
        if df != prev_df:
            ax.axhline(row_i - 0.5, color="black", linewidth=0.8, alpha=0.5)
            prev_df = df

    fig.colorbar(im, ax=ax, label="Loading", shrink=0.6)
    ax.set_title(f"Factor Loadings — {label}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Factor")
    fig.tight_layout()
    fig.savefig(save_dir / "1_loading_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    1_loading_heatmap.png")


def _plot_score_scatter_matrix(
    scores: np.ndarray,
    n_factors: int,
    save_dir: Path,
    label: str,
    plt,
) -> None:
    """Pairwise scatter plots of factor scores.

    For ≤6 factors, shows the full NxN matrix. For >6 factors, shows the top 12
    most correlated pairs (by absolute correlation) to keep the plot readable.
    """
    if n_factors < 2:
        return

    MAX_FULL_MATRIX = 15
    MAX_PAIRS = 20

    if n_factors <= MAX_FULL_MATRIX:
        # Full matrix layout
        fig, axes = plt.subplots(
            n_factors, n_factors,
            figsize=(3 * n_factors, 3 * n_factors),
        )
        if n_factors == 1:
            axes = np.array([[axes]])

        for i in range(n_factors):
            for j in range(n_factors):
                ax = axes[i, j]
                if i == j:
                    ax.hist(scores[:, i], bins=30, color="#2563eb", alpha=0.7, edgecolor="white")
                    ax.set_ylabel("Count" if j == 0 else "")
                elif i > j:
                    ax.scatter(
                        scores[:, j], scores[:, i],
                        alpha=0.15, s=8, color="#2563eb", edgecolors="none",
                    )
                    r = np.corrcoef(scores[:, j], scores[:, i])[0, 1]
                    ax.annotate(
                        f"r={r:.2f}", xy=(0.05, 0.92), xycoords="axes fraction",
                        fontsize=9, fontweight="bold",
                        color="#dc2626" if abs(r) > 0.3 else "#6b7280",
                    )
                else:
                    ax.set_visible(False)

                if i == n_factors - 1:
                    ax.set_xlabel(f"F{j}", fontsize=10)
                if j == 0 and i != j:
                    ax.set_ylabel(f"F{i}", fontsize=10)

        fig.suptitle(f"Factor Score Scatter Matrix — {label}", fontsize=14, fontweight="bold", y=1.01)
    else:
        # Top-N pairs layout: select the most correlated pairs
        corr_matrix = np.corrcoef(scores.T)
        pairs = []
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                pairs.append((abs(corr_matrix[i, j]), i, j))
        pairs.sort(reverse=True)
        top_pairs = pairs[:MAX_PAIRS]

        n_show = len(top_pairs)
        cols = min(4, n_show)
        rows = (n_show + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)

        for idx, (abs_r, fi, fj) in enumerate(top_pairs):
            ax = axes[idx // cols, idx % cols]
            ax.scatter(
                scores[:, fj], scores[:, fi],
                alpha=0.15, s=8, color="#2563eb", edgecolors="none",
            )
            r = corr_matrix[fi, fj]
            ax.annotate(
                f"r={r:.2f}", xy=(0.05, 0.92), xycoords="axes fraction",
                fontsize=10, fontweight="bold",
                color="#dc2626" if abs(r) > 0.3 else "#6b7280",
            )
            ax.set_xlabel(f"F{fj}", fontsize=10)
            ax.set_ylabel(f"F{fi}", fontsize=10)
            ax.set_title(f"F{fi} vs F{fj}", fontsize=10)

        for idx in range(n_show, rows * cols):
            axes[idx // cols, idx % cols].set_visible(False)

        fig.suptitle(
            f"Top {n_show} Factor Score Pairs (by |r|) — {label}",
            fontsize=14, fontweight="bold", y=1.01,
        )

    fig.tight_layout()
    fig.savefig(save_dir / "2_score_scatter_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    2_score_scatter_matrix.png")


def _plot_communalities(
    communalities: np.ndarray,
    column_defs: list[dict],
    save_dir: Path,
    label: str,
    plt,
) -> None:
    """Per-item communality bar chart, sorted by communality."""
    n_items = len(communalities)
    order = np.argsort(communalities)[::-1]

    item_labels = []
    for idx in order:
        cd = column_defs[idx]
        text = cd["text"][:50]
        block_tag = cd["block"][:2].upper()
        item_labels.append(f"[{block_tag}] {text}")

    fig_h = max(5, n_items * 0.2)
    fig, ax = plt.subplots(figsize=(7, fig_h))

    colors = ["#2563eb" if c >= 0.2 else "#dc2626" for c in communalities[order]]
    y_pos = np.arange(n_items)
    ax.barh(y_pos, communalities[order], color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(item_labels, fontsize=max(5, min(8, 200 / n_items)))
    ax.invert_yaxis()
    ax.axvline(0.2, color="#dc2626", linestyle="--", linewidth=0.8, alpha=0.6, label="h²=0.2 threshold")
    ax.set_xlabel("Communality (h²)")
    ax.set_title(f"Communalities — {label}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_dir / "3_communalities.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    3_communalities.png")


def _plot_score_distributions(
    scores: np.ndarray,
    n_factors: int,
    save_dir: Path,
    label: str,
    plt,
) -> None:
    """Per-factor score distribution histograms with KDE overlay."""
    cols = min(n_factors, 4)
    rows = (n_factors + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)

    for fi in range(n_factors):
        ax = axes[fi // cols, fi % cols]
        s = scores[:, fi]
        ax.hist(s, bins=40, color="#2563eb", alpha=0.6, edgecolor="white", density=True)

        # KDE overlay
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(s)
            x_grid = np.linspace(s.min() - 0.5, s.max() + 0.5, 200)
            ax.plot(x_grid, kde(x_grid), color="#dc2626", linewidth=1.5)
        except Exception:
            pass

        # Normality test (Shapiro-Wilk on subsample for speed)
        from scipy.stats import shapiro
        try:
            sub = s[:500] if len(s) > 500 else s
            _, p_val = shapiro(sub)
            ax.annotate(
                f"Shapiro p={p_val:.3f}",
                xy=(0.95, 0.92), xycoords="axes fraction",
                fontsize=8, ha="right",
                color="#059669" if p_val > 0.05 else "#dc2626",
            )
        except Exception:
            pass

        # Mark bimodality with Hartigan's dip test approximation:
        # just report skewness and kurtosis
        from scipy.stats import skew, kurtosis
        sk = skew(s)
        ku = kurtosis(s)
        ax.annotate(
            f"skew={sk:.2f}  kurt={ku:.2f}",
            xy=(0.95, 0.82), xycoords="axes fraction",
            fontsize=7, ha="right", color="#6b7280",
        )

        ax.set_title(f"Factor {fi}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Score")

    # Hide unused axes
    for i in range(n_factors, rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    fig.suptitle(f"Factor Score Distributions — {label}", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_dir / "4_score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    4_score_distributions.png")


def _plot_factor_correlations(
    fa_result: dict,
    n_factors: int,
    save_dir: Path,
    label: str,
    plt,
    TwoSlopeNorm,
) -> None:
    """Inter-factor correlation heatmap (oblique rotations only)."""
    phi = fa_result.get("factor_correlation_matrix")
    if phi is None:
        # Compute from scores as fallback (varimax factors should be ~uncorrelated)
        phi = np.corrcoef(fa_result["scores"].T)

    if n_factors < 2:
        return

    fig, ax = plt.subplots(figsize=(max(4, n_factors * 0.8 + 2), max(4, n_factors * 0.8 + 1)))
    vmax = max(0.5, np.max(np.abs(phi - np.eye(n_factors))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(phi, cmap="RdBu_r", norm=norm)
    for i in range(n_factors):
        for j in range(n_factors):
            ax.text(j, i, f"{phi[i, j]:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold" if i != j and abs(phi[i, j]) > 0.3 else "normal")

    ax.set_xticks(range(n_factors))
    ax.set_xticklabels([f"F{i}" for i in range(n_factors)])
    ax.set_yticks(range(n_factors))
    ax.set_yticklabels([f"F{i}" for i in range(n_factors)])
    fig.colorbar(im, ax=ax, label="Correlation", shrink=0.7)
    ax.set_title(f"Factor Correlations — {label}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "5_factor_correlations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    5_factor_correlations.png")


def _plot_scores_by_archetype(
    scores: np.ndarray,
    metadata: list[dict],
    n_factors: int,
    save_dir: Path,
    label: str,
    plt,
) -> None:
    """Violin plots of factor scores grouped by interviewer archetype."""
    # Load archetype assignments from the rollout directory
    assignments_path = _rollout_dir() / "archetype_assignments.json"
    if not assignments_path.exists():
        print(f"    7_scores_by_archetype.png — skipped (no archetype_assignments.json)")
        return

    with open(assignments_path) as f:
        sample_to_archetype: dict[str, str] = json.load(f)

    # Map each persona row to its archetype
    archetypes = []
    valid_mask = []
    for i, meta in enumerate(metadata):
        arch = sample_to_archetype.get(meta["sample_id"])
        archetypes.append(arch)
        valid_mask.append(arch is not None)

    valid_mask = np.array(valid_mask)
    if valid_mask.sum() < 10:
        print(f"    7_scores_by_archetype.png — skipped (too few matched rows)")
        return

    valid_scores = scores[valid_mask]
    valid_archetypes = [a for a, v in zip(archetypes, valid_mask) if v]

    # Get unique archetypes sorted by name
    unique_archetypes = sorted(set(valid_archetypes))
    n_archetypes = len(unique_archetypes)
    arch_to_idx = {a: i for i, a in enumerate(unique_archetypes)}

    cols = min(n_factors, 3)
    rows = (n_factors + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for fi in range(n_factors):
        ax = axes[fi // cols, fi % cols]

        # Group scores by archetype
        grouped = [[] for _ in range(n_archetypes)]
        for score_val, arch in zip(valid_scores[:, fi], valid_archetypes):
            grouped[arch_to_idx[arch]].append(score_val)

        parts = ax.violinplot(
            grouped, positions=range(n_archetypes),
            showmeans=True, showmedians=True,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("#2563eb")
            pc.set_alpha(0.5)
        parts["cmeans"].set_color("#dc2626")
        parts["cmedians"].set_color("#059669")

        ax.set_xticks(range(n_archetypes))
        # Truncate long archetype names
        ax.set_xticklabels(
            [a[:15] for a in unique_archetypes],
            rotation=45, ha="right", fontsize=8,
        )
        ax.set_title(f"Factor {fi}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3)

    for i in range(n_factors, rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    fig.suptitle(
        f"Factor Scores by Interviewer Archetype — {label}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_dir / "7_scores_by_archetype.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    7_scores_by_archetype.png")


def _plot_prompt_icc(
    scores: np.ndarray,
    metadata: list[dict],
    n_factors: int,
    save_dir: Path,
    label: str,
    plt,
) -> None:
    """ICC(1) per factor: proportion of factor-score variance attributable to the seed prompt.

    High ICC means the prompt drives the factor more than stochastic rollout variation.
    Low ICC means the factor captures genuine run-to-run behavioural variation.
    """
    from src_dev.factor_analysis.reliability import compute_icc

    icc_result = compute_icc(scores, metadata, n_factors)

    if icc_result.get("error"):
        print(f"    8_prompt_icc.png — skipped ({icc_result['error']})")
        return

    icc_values = icc_result["icc1"]
    n_groups = icc_result["n_groups"]

    # Bar plot
    fig, ax = plt.subplots(figsize=(max(5, n_factors * 1.2 + 1), 4))
    x = np.arange(n_factors)
    colors = ["#f59e0b" if v > 0.5 else "#2563eb" for v in icc_values]
    bars = ax.bar(x, icc_values, color=colors, edgecolor="white", width=0.6)

    for bar, val in zip(bars, icc_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"F{i}" for i in range(n_factors)], fontsize=11)
    ax.set_ylim(0, min(1.0, max(icc_values) * 1.3 + 0.05))
    ax.set_ylabel("ICC(1)")
    ax.set_xlabel("Factor")
    ax.axhline(0.5, color="#f59e0b", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.annotate("ICC=0.5", xy=(n_factors - 0.5, 0.51), fontsize=8, color="#f59e0b")
    ax.set_title(
        f"Prompt ICC — {label}\n"
        f"({n_groups} prompts with ≥2 rollouts)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "8_prompt_icc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    8_prompt_icc.png")

    # Also save ICC values as JSON for reference
    with open(save_dir / "8_prompt_icc.json", "w") as f:
        json.dump({
            "icc_per_factor": icc_values,
            "n_groups": n_groups,
            "n_total_personas": icc_result["n_total"],
        }, f, indent=2)


# ── Factor extremes HTML export ─────────────────────────────────────────────

FACTOR_EXTREMES_N = 3  # rollouts per pole per factor


def _load_llm_labels_from_path(path: Path) -> list[dict]:
    """Load a label cache file, treating empty/invalid payloads as unavailable."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.warning("Ignoring malformed LLM label cache at %s: expected list", path)
        return []

    labels = [entry for entry in data if isinstance(entry, dict)]
    if not labels:
        return []

    return labels


def _load_latest_nonempty_llm_labels(
    labeling_dir: Path,
    label: str,
    *,
    require_axis_names: bool = False,
) -> list[dict]:
    """Return the newest non-empty LLM label cache for an analysis label."""
    candidate_paths = set(labeling_dir.glob(f"llm_labels_{label}_*.json"))
    legacy_path = labeling_dir / f"llm_labels_{label}.json"
    if legacy_path.exists():
        candidate_paths.add(legacy_path)

    for path in sorted(candidate_paths, key=lambda p: p.stat().st_mtime, reverse=True):
        labels = _load_llm_labels_from_path(path)
        if require_axis_names and labels and not _llm_labels_have_axis_names(labels):
            logger.warning(
                "Ignoring old-schema LLM label cache without axis_name fields: %s",
                path,
            )
            continue
        if labels:
            return labels

    return []


# ═════════════════════════════════════════════════════════════════════════════
# TRAIT-AWARE FA VISUALISATIONS (full-distribution views across all items)
# ═════════════════════════════════════════════════════════════════════════════

# Fixed OCEAN color palette so trait colors are consistent across plots.
_OCEAN_TRAIT_COLORS: dict[str, str] = {
    "openness": "#4c78a8",
    "conscientiousness": "#59a14f",
    "extraversion": "#e45756",
    "agreeableness": "#b279a2",
    "neuroticism": "#f28e2b",
}
_FALLBACK_TRAIT_COLOR = "#9ca3af"


def _trait_color(trait: str) -> str:
    return _OCEAN_TRAIT_COLORS.get(trait, _FALLBACK_TRAIT_COLOR)


def _canonical_trait_order(item_dims: list[str]) -> list[str]:
    """Canonical OCEAN order, restricted to traits that actually appear."""
    ocean = ["openness", "conscientiousness", "extraversion",
             "agreeableness", "neuroticism"]
    present = [t for t in ocean if t in item_dims]
    extras = sorted(set(item_dims) - set(present))
    return present + extras


def _plot_trait_aware_fa_visualisations(
    loadings: np.ndarray,
    item_dims: list[str],
    save_dir: Path,
    label: str,
    *,
    top_k: int = 20,
    signed_caveat: str | None = None,
) -> None:
    """Render five trait-aware views of the FA loadings matrix.

    Args:
        loadings: [n_items × n_factors] float.
        item_dims: primary_dimension for each row of ``loadings``. Items with
            ``None`` dimension are dropped from all plots.
        save_dir: Directory to write PNGs into.
        label: Label for plot titles (e.g. "raw_varimax", "trait_oriented_varimax").
        top_k: K for top-K cumulative composition curves (also annotated on them).
        signed_caveat: If set, appended to signed-loading plot titles (e.g. a
            letter-encoded-matrix warning).
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    loadings = np.asarray(loadings)
    mask = np.array([d is not None for d in item_dims])
    if not mask.any():
        print(f"  [TraitViz] No items with dimensions for {label} — skipping.")
        return
    loadings = loadings[mask]
    item_dims = [d for d, m in zip(item_dims, mask) if m]
    trait_order = _canonical_trait_order(item_dims)

    _plot_trait_sorted_loading_heatmap(
        loadings, item_dims, trait_order,
        save_dir / "6a_trait_sorted_loading_heatmap.png",
        label=label, signed_caveat=signed_caveat,
    )
    _plot_per_trait_loading_distributions(
        loadings, item_dims, trait_order,
        save_dir / "6b_per_trait_loading_distributions.png",
        label=label, signed_caveat=signed_caveat,
    )
    _plot_cumulative_top_k_composition(
        loadings, item_dims, trait_order,
        save_dir / "6c_cumulative_top_k_composition.png",
        label=label, top_k_marker=top_k,
    )
    _plot_per_trait_loading_ecdfs(
        loadings, item_dims, trait_order,
        save_dir / "6d_per_trait_abs_loading_ecdf.png",
        label=label,
    )
    _plot_signed_loading_strip(
        loadings, item_dims, trait_order,
        save_dir / "6e_signed_loading_strip.png",
        label=label, signed_caveat=signed_caveat,
    )


def _plot_trait_sorted_loading_heatmap(
    loadings: np.ndarray,
    item_dims: list[str],
    trait_order: list[str],
    save_path: Path,
    *,
    label: str,
    signed_caveat: str | None = None,
) -> None:
    """Items × Factors heatmap sorted by OCEAN trait block, with separators."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    n_items, n_factors = loadings.shape
    # Sort items: primary by trait block (OCEAN order), secondary by dominant
    # factor, tertiary by -|loading on dominant factor|.
    trait_rank = {t: i for i, t in enumerate(trait_order)}
    dominant_factor = np.argmax(np.abs(loadings), axis=1)
    keys = [
        (trait_rank.get(item_dims[i], len(trait_order)),
         int(dominant_factor[i]),
         -float(np.abs(loadings[i, dominant_factor[i]])))
        for i in range(n_items)
    ]
    order = sorted(range(n_items), key=lambda i: keys[i])
    sorted_loadings = loadings[order]
    sorted_dims = [item_dims[i] for i in order]

    fig_h = max(6, n_items * 0.14)
    fig_w = max(6, 3 + n_factors * 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = max(0.5, float(np.max(np.abs(sorted_loadings))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(sorted_loadings, aspect="auto", cmap="RdBu_r", norm=norm)

    ax.set_xticks(range(n_factors))
    ax.set_xticklabels([f"F{i+1}" for i in range(n_factors)], fontsize=10)

    # Row labels: compact trait abbreviation + row index within trait block.
    row_labels: list[str] = []
    per_trait_counter: dict[str, int] = {}
    for d in sorted_dims:
        per_trait_counter[d] = per_trait_counter.get(d, 0) + 1
        row_labels.append(f"{d[:4]}_{per_trait_counter[d]:02d}")
    ax.set_yticks(range(n_items))
    ax.set_yticklabels(row_labels, fontsize=max(4, min(8, 220 / n_items)))

    # Trait-block separators + side labels.
    prev = sorted_dims[0]
    block_start = 0
    for i, d in enumerate(sorted_dims + [None]):
        if d != prev:
            ax.axhline(i - 0.5, color="black", linewidth=1.2, alpha=0.75)
            mid = (block_start + i - 1) / 2
            color = _trait_color(prev)
            ax.text(
                -0.6, mid, prev, ha="right", va="center",
                fontsize=10, fontweight="bold", color=color,
                transform=ax.get_yaxis_transform(),
            )
            block_start = i
            prev = d

    title = f"Factor loadings, trait-sorted — {label}"
    if signed_caveat:
        title += f"\n({signed_caveat})"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Factor")
    fig.colorbar(im, ax=ax, label="Loading", shrink=0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    6a_trait_sorted_loading_heatmap.png")


def _plot_per_trait_loading_distributions(
    loadings: np.ndarray,
    item_dims: list[str],
    trait_order: list[str],
    save_path: Path,
    *,
    label: str,
    signed_caveat: str | None = None,
) -> None:
    """Per-factor violin+strip plots of signed item loadings grouped by trait.

    One subplot per factor. Within each subplot, one violin+strip per trait
    showing the full distribution of that trait's items' loadings (not just
    top-K).
    """
    import matplotlib.pyplot as plt

    n_items, n_factors = loadings.shape
    dims_arr = np.array(item_dims)
    n_cols = min(3, n_factors)
    n_rows = int(np.ceil(n_factors / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )
    vmax = float(np.max(np.abs(loadings)))
    lim = max(0.4, vmax * 1.1)
    for f, ax in zip(range(n_factors), axes.flat):
        col = loadings[:, f]
        data_per_trait = [col[dims_arr == t] for t in trait_order]
        positions = np.arange(len(trait_order))
        vp = ax.violinplot(
            [d for d in data_per_trait if len(d) > 0],
            positions=[p for p, d in zip(positions, data_per_trait) if len(d) > 0],
            widths=0.75, showmeans=False, showmedians=True, showextrema=False,
        )
        for body, t in zip(vp["bodies"], [t for t, d in zip(trait_order, data_per_trait) if len(d) > 0]):
            body.set_facecolor(_trait_color(t))
            body.set_edgecolor("black")
            body.set_alpha(0.35)
        # Strip plot over violins — jittered items.
        rng = np.random.default_rng(0)
        for pos, t, d in zip(positions, trait_order, data_per_trait):
            if len(d) == 0:
                continue
            jitter = rng.uniform(-0.18, 0.18, size=len(d))
            ax.scatter(
                pos + jitter, d, color=_trait_color(t),
                s=14, alpha=0.7, edgecolor="white", linewidth=0.4,
            )
        ax.axhline(0, color="#6b7280", linewidth=0.8, alpha=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels([t[:4] for t in trait_order], fontsize=9)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"F{f+1}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Signed loading" if f % n_cols == 0 else "")
        ax.grid(axis="y", alpha=0.3, linewidth=0.4)

    for ax in axes.flat[n_factors:]:
        ax.axis("off")

    suptitle = f"Per-trait loading distributions — {label}"
    if signed_caveat:
        suptitle += f"  ({signed_caveat})"
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    6b_per_trait_loading_distributions.png")


def _plot_cumulative_top_k_composition(
    loadings: np.ndarray,
    item_dims: list[str],
    trait_order: list[str],
    save_path: Path,
    *,
    label: str,
    top_k_marker: int = 20,
) -> None:
    """Cumulative top-K composition: for each factor, one line per trait
    showing how many top-K items (sorted by |loading|) belong to that trait
    as K sweeps from 1 to n_items. Dashed lines show the uniform baseline
    (expected count if loadings were unrelated to trait)."""
    import matplotlib.pyplot as plt

    n_items, n_factors = loadings.shape
    dims_arr = np.array(item_dims)
    trait_fracs = {t: float(np.mean(dims_arr == t)) for t in trait_order}

    n_cols = min(3, n_factors)
    n_rows = int(np.ceil(n_factors / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )
    K_arr = np.arange(1, n_items + 1)
    for f, ax in zip(range(n_factors), axes.flat):
        col = loadings[:, f]
        rank = np.argsort(-np.abs(col))
        dim_sequence = dims_arr[rank]
        for t in trait_order:
            is_trait = (dim_sequence == t).astype(int)
            cum = np.cumsum(is_trait)
            ax.plot(
                K_arr, cum, color=_trait_color(t), linewidth=1.8,
                label=t,
            )
            # Baseline: expected count under uniform trait prevalence.
            ax.plot(
                K_arr, K_arr * trait_fracs[t],
                color=_trait_color(t), linewidth=0.8, linestyle="--", alpha=0.6,
            )
        if 1 <= top_k_marker <= n_items:
            ax.axvline(
                top_k_marker, color="#6b7280", linestyle=":", linewidth=1,
                alpha=0.8,
            )
            ax.text(
                top_k_marker, ax.get_ylim()[1] * 0.98,
                f" K={top_k_marker}", fontsize=8, color="#374151",
                va="top", ha="left",
            )
        ax.set_title(f"F{f+1}", fontsize=11, fontweight="bold")
        ax.set_xlabel("K (items ranked by |loading|)")
        if f % n_cols == 0:
            ax.set_ylabel("Cumulative count")
        ax.grid(alpha=0.3, linewidth=0.4)

    # Shared legend below last subplot.
    handles = [
        plt.Line2D([0], [0], color=_trait_color(t), linewidth=2, label=t)
        for t in trait_order
    ]
    handles.append(plt.Line2D([0], [0], color="#6b7280", linestyle="--",
                              linewidth=1, label="expected (uniform)"))
    fig.legend(handles=handles, loc="lower center",
               ncol=min(6, len(handles)), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    for ax in axes.flat[n_factors:]:
        ax.axis("off")

    fig.suptitle(f"Cumulative top-K trait composition — {label}",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    6c_cumulative_top_k_composition.png")


def _plot_per_trait_loading_ecdfs(
    loadings: np.ndarray,
    item_dims: list[str],
    trait_order: list[str],
    save_path: Path,
    *,
    label: str,
) -> None:
    """Per-factor ECDFs of |loading|, one curve per trait.

    A factor that captures trait X will have that trait's ECDF shifted to
    the right (more mass at high |loading|) vs other traits.
    """
    import matplotlib.pyplot as plt

    n_items, n_factors = loadings.shape
    dims_arr = np.array(item_dims)
    n_cols = min(3, n_factors)
    n_rows = int(np.ceil(n_factors / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )
    vmax = float(np.max(np.abs(loadings)))
    for f, ax in zip(range(n_factors), axes.flat):
        col_abs = np.abs(loadings[:, f])
        for t in trait_order:
            vals = np.sort(col_abs[dims_arr == t])
            if len(vals) == 0:
                continue
            ys = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, ys, color=_trait_color(t), linewidth=1.8, label=t)
        ax.set_title(f"F{f+1}", fontsize=11, fontweight="bold")
        ax.set_xlabel("|loading|")
        if f % n_cols == 0:
            ax.set_ylabel("ECDF (fraction ≤)")
        ax.set_xlim(0, max(0.4, vmax * 1.05))
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3, linewidth=0.4)

    handles = [
        plt.Line2D([0], [0], color=_trait_color(t), linewidth=2, label=t)
        for t in trait_order
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(5, len(handles)), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    for ax in axes.flat[n_factors:]:
        ax.axis("off")

    fig.suptitle(f"Per-trait ECDFs of |loading| — {label}",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    6d_per_trait_abs_loading_ecdf.png")


def _plot_signed_loading_strip(
    loadings: np.ndarray,
    item_dims: list[str],
    trait_order: list[str],
    save_path: Path,
    *,
    label: str,
    signed_caveat: str | None = None,
) -> None:
    """Per-factor horizontal strip plot: every item's signed loading, grouped
    by trait on the y-axis. Unlike the violin view, this emphasizes sign and
    keeps individual items visible."""
    import matplotlib.pyplot as plt

    n_items, n_factors = loadings.shape
    dims_arr = np.array(item_dims)
    n_cols = min(3, n_factors)
    n_rows = int(np.ceil(n_factors / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.6 * n_cols, 0.35 * len(trait_order) * n_rows + 1.2),
        squeeze=False,
    )
    vmax = float(np.max(np.abs(loadings)))
    lim = max(0.4, vmax * 1.1)
    rng = np.random.default_rng(1)
    for f, ax in zip(range(n_factors), axes.flat):
        col = loadings[:, f]
        for y, t in enumerate(trait_order):
            vals = col[dims_arr == t]
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.22, 0.22, size=len(vals))
            ax.scatter(
                vals, np.full_like(vals, y) + jitter,
                color=_trait_color(t), s=22, alpha=0.75,
                edgecolor="white", linewidth=0.4,
            )
            ax.scatter(
                [float(np.mean(vals))], [y], marker="|",
                color="black", s=180, linewidth=1.6, zorder=5,
            )
        ax.axvline(0, color="#6b7280", linewidth=0.8, alpha=0.7)
        ax.set_yticks(range(len(trait_order)))
        ax.set_yticklabels(trait_order, fontsize=9)
        ax.set_ylim(-0.7, len(trait_order) - 0.3)
        ax.set_xlim(-lim, lim)
        ax.set_xlabel("Signed loading")
        ax.set_title(f"F{f+1}", fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3, linewidth=0.4)
    for ax in axes.flat[n_factors:]:
        ax.axis("off")

    suptitle = f"Signed-loading strip by trait — {label}"
    if signed_caveat:
        suptitle += f"  ({signed_caveat})"
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    6e_signed_loading_strip.png")


def _llm_labels_have_axis_names(labels: list[dict]) -> bool:
    """Return True when every factor label includes a non-empty axis name."""
    if not labels:
        return False
    return all(str(label.get("axis_name", "")).strip() for label in labels)


def _export_factor_extremes_html(
    fa_result: dict,
    column_defs: list[dict],
    metadata: list[dict],
    label: str,
    save_dir: Path,
    n_per_pole: int = FACTOR_EXTREMES_N,
) -> None:
    """Export an HTML viewer showing rollout conversations for extreme-scoring personas.

    For each factor, provides two tabs:
    - **Personas**: top-N and bottom-N personas by factor score with rollout conversations
    - **Factor**: loading bar chart, score distribution, top items table, variance stats

    Args:
        fa_result: Dict from run_factor_analysis.
        column_defs: Column definitions (for loading-based factor descriptions).
        metadata: Metadata rows aligned with factor scores.
        label: Analysis variant label (e.g. "raw_oblimin").
        save_dir: Directory to write the HTML file.
        n_per_pole: Number of rollouts per pole per factor.
    """
    import html as html_mod
    from scipy.stats import skew, kurtosis

    scores = fa_result["scores"]
    loadings = fa_result["loadings"]
    communalities = fa_result["communalities"]
    proportion_variance = fa_result["proportion_variance"]
    ss_loadings = fa_result["ss_loadings"]
    cumulative_variance = fa_result["cumulative_variance"]
    factor_corr_matrix = fa_result.get("factor_correlation_matrix")
    n_factors = scores.shape[1]

    # Load rollout conversations, indexed by sample_id
    rollout_path = _rollout_dir() / "exports" / "conversation_training.jsonl"
    if not rollout_path.exists():
        print(f"  [Extremes] Skipped — rollout export not found: {rollout_path}")
        return

    conversations_by_sid: dict[str, list[dict[str, str]]] = {}
    with open(rollout_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row.get("sample_id", "")
            msgs = [{"role": m["role"], "content": m["content"]} for m in row.get("messages", [])]
            conversations_by_sid[sid] = msgs

    # Load LLM labels if available (for richer factor descriptions).
    labeling_dir = _questionnaire_dir() / "labeling"
    llm_labels = _load_latest_nonempty_llm_labels(
        labeling_dir,
        label,
        require_axis_names=True,
    )

    # Also load item-based labels
    item_labels: list[dict] = []
    item_labels_path = _questionnaire_dir() / "labeling" / f"item_labels_{label}.json"
    if item_labels_path.exists():
        with open(item_labels_path) as f:
            item_labels = json.load(f)

    # Build factor descriptions for the header
    factor_descriptions = []
    for fi in range(n_factors):
        desc: dict[str, str] = {"index": fi}

        # LLM label if available
        llm_label = next((l for l in llm_labels if l.get("factor_index") == fi), None)
        if llm_label:
            desc["axis_name"] = llm_label.get("axis_name", "")
            desc["summary"] = llm_label.get("summary", "")
            desc["description"] = llm_label.get("description", "")
            desc["positive_pole"] = llm_label.get("positive_pole", "")
            desc["negative_pole"] = llm_label.get("negative_pole", "")
        else:
            # Fallback: top loading items
            item_label = next((l for l in item_labels if l.get("factor_index") == fi), None)
            if item_label:
                pos_items = item_label.get("positive_items", [])
                neg_items = item_label.get("negative_items", [])
                desc["axis_name"] = ""
                desc["summary"] = ""
                desc["positive_pole"] = (
                    pos_items[0]["text"] if pos_items else "(none)"
                )
                desc["negative_pole"] = (
                    neg_items[0]["text"] if neg_items else "(none)"
                )
                desc["description"] = ""
            else:
                desc["axis_name"] = ""
                desc["summary"] = f"Factor {fi}"
                desc["positive_pole"] = "high"
                desc["negative_pole"] = "low"
                desc["description"] = ""

        # Top loading items for context
        col = loadings[:, fi]
        order = np.argsort(col)
        top_pos_idxs = [idx for idx in order[-3:][::-1] if col[idx] > 0]
        top_neg_idxs = [idx for idx in order[:3] if col[idx] < 0]
        desc["top_positive_items"] = [
            f"({col[idx]:+.3f}) {column_defs[idx]['text']}" for idx in top_pos_idxs
        ]
        desc["top_negative_items"] = [
            f"({col[idx]:+.3f}) {column_defs[idx]['text']}" for idx in top_neg_idxs
        ]

        factor_descriptions.append(desc)

    # Build per-factor analytical data for the Factor tab
    factor_data = []
    for fi in range(n_factors):
        col = loadings[:, fi]
        factor_scores_col = scores[:, fi]

        # Item loadings with cross-loading info
        items_for_factor = []
        for i, cdef in enumerate(column_defs):
            # Find the strongest cross-loading on a *different* factor
            other_loadings = [(fj, abs(loadings[i, fj])) for fj in range(n_factors) if fj != fi]
            if other_loadings:
                max_cross_fj, max_cross_abs = max(other_loadings, key=lambda x: x[1])
                max_cross_val = float(loadings[i, max_cross_fj])
            else:
                max_cross_fj, max_cross_val = -1, 0.0

            items_for_factor.append({
                "text": cdef["text"],
                "loading": round(float(col[i]), 4),
                "communality": round(float(communalities[i]), 4),
                "max_cross_loading": round(max_cross_val, 4),
                "max_cross_factor": int(max_cross_fj),
            })

        # Sort by absolute loading descending
        items_for_factor.sort(key=lambda x: abs(x["loading"]), reverse=True)

        # Score histogram
        hist_counts, hist_edges = np.histogram(factor_scores_col, bins=25)

        # Factor correlations
        factor_corrs = []
        if factor_corr_matrix is not None:
            for fj in range(n_factors):
                if fj != fi:
                    factor_corrs.append({
                        "factor": int(fj),
                        "r": round(float(factor_corr_matrix[fi, fj]), 4),
                    })
        else:
            # Compute from scores
            for fj in range(n_factors):
                if fj != fi:
                    r = float(np.corrcoef(scores[:, fi], scores[:, fj])[0, 1])
                    factor_corrs.append({"factor": int(fj), "r": round(r, 4)})

        factor_data.append({
            "loadings": items_for_factor,
            "variance_explained": round(float(proportion_variance[fi]), 4),
            "ss_loading": round(float(ss_loadings[fi]), 4),
            "cumulative_variance": round(float(cumulative_variance[fi]), 4),
            "score_stats": {
                "mean": round(float(np.mean(factor_scores_col)), 4),
                "std": round(float(np.std(factor_scores_col)), 4),
                "skew": round(float(skew(factor_scores_col)), 4),
                "kurtosis": round(float(kurtosis(factor_scores_col)), 4),
                "min": round(float(np.min(factor_scores_col)), 4),
                "max": round(float(np.max(factor_scores_col)), 4),
                "n": int(len(factor_scores_col)),
            },
            "score_histogram": {
                "edges": [round(float(e), 4) for e in hist_edges],
                "counts": [int(c) for c in hist_counts],
            },
            "correlations": factor_corrs,
        })

    # Collect extreme persona records — include all factor scores per persona
    records = []
    for fi in range(n_factors):
        factor_scores = scores[:, fi]
        sorted_indices = np.argsort(factor_scores)

        # Bottom N (low scorers)
        for rank, idx in enumerate(sorted_indices[:n_per_pole]):
            meta = metadata[idx]
            sid = meta["sample_id"]
            conv = conversations_by_sid.get(sid, [])
            if not conv:
                continue
            records.append({
                "factor": fi,
                "pole": "low",
                "rank": rank + 1,
                "score": float(factor_scores[idx]),
                "all_scores": [round(float(scores[idx, fj]), 3) for fj in range(n_factors)],
                "sample_id": sid,
                "messages": conv,
            })

        # Top N (high scorers)
        for rank, idx in enumerate(sorted_indices[-n_per_pole:][::-1]):
            meta = metadata[idx]
            sid = meta["sample_id"]
            conv = conversations_by_sid.get(sid, [])
            if not conv:
                continue
            records.append({
                "factor": fi,
                "pole": "high",
                "rank": rank + 1,
                "score": float(factor_scores[idx]),
                "all_scores": [round(float(scores[idx, fj]), 3) for fj in range(n_factors)],
                "sample_id": sid,
                "messages": conv,
            })

    if not records:
        print(f"  [Extremes] No matching conversations found — skipped.")
        return

    # Build the HTML
    data_json = json.dumps(records, ensure_ascii=False)
    factors_json = json.dumps(factor_descriptions, ensure_ascii=False)
    factor_data_json = json.dumps(factor_data, ensure_ascii=False)
    title = html_mod.escape(f"Factor Extremes — {label}")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #111827; color: #e5e7eb;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 14px; line-height: 1.5;
    display: flex; height: 100vh; overflow: hidden;
  }}
  #sidebar {{
    width: 340px; min-width: 280px;
    background: #1f2937; border-right: 1px solid #374151;
    display: flex; flex-direction: column; overflow: hidden; flex-shrink: 0;
  }}
  #sidebar-header {{
    padding: 12px 16px; background: #0e7490; color: #fff;
    font-weight: bold; font-size: 15px; flex-shrink: 0;
  }}
  #factor-selector {{
    padding: 8px; border-bottom: 1px solid #374151; flex-shrink: 0;
  }}
  #factor-selector select {{
    width: 100%; padding: 6px 8px; border-radius: 4px;
    background: #374151; color: #e5e7eb; border: 1px solid #4b5563;
    font-size: 13px;
  }}
  /* Tab bar */
  #tab-bar {{
    display: flex; flex-shrink: 0; border-bottom: 1px solid #374151;
  }}
  .tab-btn {{
    flex: 1; padding: 8px 12px; text-align: center; cursor: pointer;
    font-size: 12px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; background: #1f2937; color: #9ca3af;
    border: none; border-bottom: 2px solid transparent;
    transition: all 0.15s;
  }}
  .tab-btn:hover {{ color: #e5e7eb; background: #283548; }}
  .tab-btn.active {{ color: #60a5fa; border-bottom-color: #60a5fa; background: #1e293b; }}
  /* Sidebar content panels */
  #sidebar-personas, #sidebar-factor {{
    flex: 1; overflow-y: auto; display: flex; flex-direction: column;
  }}
  #sidebar-factor {{ padding: 12px 14px; }}
  .sidebar-hidden {{ display: none !important; }}
  #factor-info {{
    padding: 10px 14px; border-bottom: 1px solid #374151;
    font-size: 12px; overflow-y: auto; max-height: 260px; flex-shrink: 0;
  }}
  #factor-info .pole {{ margin-bottom: 6px; }}
  .pole-label {{
    font-weight: 700; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .pole-high .pole-label {{ color: #4ade80; }}
  .pole-low .pole-label {{ color: #f87171; }}
  #factor-info .desc {{ color: #9ca3af; margin-top: 4px; font-style: italic; }}
  .axis-chip {{
    display: inline-block; margin-bottom: 6px; padding: 2px 7px;
    border-radius: 999px; background: #0f766e; color: #ccfbf1;
    font-size: 11px; font-weight: 700; letter-spacing: 0.03em;
  }}
  #factor-info .loading-item {{
    font-size: 11px; color: #9ca3af; margin-left: 8px;
  }}
  #record-list {{
    flex: 1; overflow-y: auto; padding: 4px 0;
  }}
  .record-entry {{
    padding: 8px 14px; cursor: pointer; border-left: 3px solid transparent;
    font-size: 12px; transition: background 0.1s;
  }}
  .record-entry:hover {{ background: #374151; }}
  .record-entry.active {{ background: #1e3a5f; border-left-color: #60a5fa; }}
  .record-entry .pole-tag {{
    display: inline-block; font-size: 10px; font-weight: 700;
    padding: 1px 5px; border-radius: 3px; margin-right: 6px;
    text-transform: uppercase;
  }}
  .pole-tag-high {{ background: #065f46; color: #6ee7b7; }}
  .pole-tag-low {{ background: #7f1d1d; color: #fca5a5; }}
  .record-entry .score {{ color: #9ca3af; font-size: 11px; }}
  /* Sidebar factor tab styles */
  .sf-section {{ margin-bottom: 16px; }}
  .sf-section-title {{
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; color: #60a5fa; margin-bottom: 6px;
  }}
  .sf-desc {{
    font-size: 13px; color: #d1d5db; font-style: italic;
    line-height: 1.5; margin-bottom: 8px;
  }}
  .sf-stat {{
    display: flex; justify-content: space-between;
    font-size: 12px; padding: 2px 0; border-bottom: 1px solid #2d3748;
  }}
  .sf-stat-label {{ color: #9ca3af; }}
  .sf-stat-value {{ color: #e5e7eb; font-weight: 600; font-variant-numeric: tabular-nums; }}
  .sf-corr-bar {{
    display: flex; align-items: center; gap: 6px;
    font-size: 11px; margin-bottom: 3px;
  }}
  .sf-corr-bar .bar-track {{
    flex: 1; height: 6px; background: #374151; border-radius: 3px; overflow: hidden;
    position: relative;
  }}
  .sf-corr-bar .bar-fill {{
    position: absolute; top: 0; height: 100%; border-radius: 3px;
  }}
  /* Main area */
  #main {{
    flex: 1; display: flex; flex-direction: column; overflow: hidden;
  }}
  #topbar {{
    background: #1e293b; padding: 8px 16px; font-size: 12px;
    border-bottom: 1px solid #374151; flex-shrink: 0;
    display: flex; gap: 16px; align-items: center;
  }}
  #topbar .tag {{ font-weight: 700; }}
  /* Persona score profile bar */
  .score-profile {{
    display: flex; gap: 4px; align-items: center; font-size: 10px; margin-left: auto;
  }}
  .score-profile .sp-item {{
    display: flex; align-items: center; gap: 2px; padding: 1px 4px;
    border-radius: 3px; background: #283548;
  }}
  .score-profile .sp-item.sp-current {{ background: #1e3a5f; font-weight: 700; }}
  .score-profile .sp-label {{ color: #6b7280; }}
  .score-profile .sp-val {{ color: #d1d5db; font-variant-numeric: tabular-nums; }}
  #main-personas, #main-factor {{
    flex: 1; overflow: hidden; display: flex; flex-direction: column;
  }}
  .main-hidden {{ display: none !important; }}
  #scroll-area {{
    flex: 1; overflow-y: auto; padding: 16px 24px 80px;
    max-width: 900px; width: 100%;
  }}
  #factor-view {{
    flex: 1; overflow-y: auto; padding: 20px 28px 80px;
    max-width: 1100px; width: 100%;
  }}
  .fv-section {{ margin-bottom: 32px; }}
  .fv-section-title {{
    font-size: 14px; font-weight: 700; color: #60a5fa;
    margin-bottom: 12px; padding-bottom: 4px;
    border-bottom: 1px solid #374151;
  }}
  .msg {{
    margin-bottom: 12px; padding: 10px 14px;
    border-radius: 8px; white-space: pre-wrap; word-break: break-word;
  }}
  .msg-user {{
    background: #1e3a5f; border-left: 3px solid #60a5fa;
  }}
  .msg-assistant {{
    background: #1a2e1a; border-left: 3px solid #4ade80;
  }}
  .msg-system {{
    background: #2d2215; border-left: 3px solid #f59e0b;
  }}
  .role-label {{
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em; margin-bottom: 4px; opacity: 0.7;
  }}
  .role-user .role-label {{ color: #60a5fa; }}
  .role-assistant .role-label {{ color: #4ade80; }}
  .role-system .role-label {{ color: #f59e0b; }}
  /* Items table */
  .items-table {{
    width: 100%; border-collapse: collapse; font-size: 12px;
  }}
  .items-table th {{
    text-align: left; padding: 6px 8px; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.05em;
    color: #9ca3af; border-bottom: 2px solid #374151;
  }}
  .items-table td {{
    padding: 5px 8px; border-bottom: 1px solid #1f2937;
    font-variant-numeric: tabular-nums;
  }}
  .items-table tr:hover {{ background: #1e293b; }}
  .items-table .loading-pos {{ color: #4ade80; }}
  .items-table .loading-neg {{ color: #f87171; }}
  .items-table .item-text {{
    color: #d1d5db; word-break: break-word;
  }}
  #bottombar {{
    background: #d1d5db; color: #111; font-weight: 600;
    padding: 5px 16px; font-size: 12px; flex-shrink: 0;
  }}
</style>
</head>
<body>
<div id="sidebar">
  <div id="sidebar-header">Factor Extremes</div>
  <div id="factor-selector"><select id="factor-select"></select></div>
  <div id="tab-bar">
    <button class="tab-btn active" data-tab="factor">Factors</button>
    <button class="tab-btn" data-tab="personas">Personas</button>
  </div>
  <div id="sidebar-personas" class="sidebar-hidden">
    <div id="factor-info"></div>
    <div id="record-list"></div>
  </div>
  <div id="sidebar-factor"></div>
</div>
<div id="main">
  <div id="main-personas" class="main-hidden">
    <div id="topbar">
      <span id="tb-info"></span>
      <div class="score-profile" id="score-profile"></div>
    </div>
    <div id="scroll-area"></div>
  </div>
  <div id="main-factor">
    <div id="factor-view"></div>
  </div>
  <div id="bottombar">
    ↑↓ or click to navigate &nbsp;|&nbsp; Factor dropdown to switch factors &nbsp;|&nbsp; Factors / Personas tabs
  </div>
</div>

<script>
const RECORDS = {data_json};
const FACTORS = {factors_json};
const FACTOR_DATA = {factor_data_json};
let currentFactor = 0;
let currentIdx = 0;
let currentTab = 'factor';
let filteredRecords = [];

const factorSelect = document.getElementById('factor-select');
const factorInfo = document.getElementById('factor-info');
const recordList = document.getElementById('record-list');
const scrollArea = document.getElementById('scroll-area');
const tbInfo = document.getElementById('tb-info');
const scoreProfile = document.getElementById('score-profile');
const sidebarPersonas = document.getElementById('sidebar-personas');
const sidebarFactor = document.getElementById('sidebar-factor');
const mainPersonas = document.getElementById('main-personas');
const mainFactor = document.getElementById('main-factor');
const factorView = document.getElementById('factor-view');

// Populate factor selector
FACTORS.forEach((f, i) => {{
  const opt = document.createElement('option');
  opt.value = i;
  const axis = f.axis_name ? ` — ${{f.axis_name}}` : '';
  const summary = f.summary ? `: ${{f.summary}}` : '';
  opt.textContent = `Factor ${{i}}${{axis}}${{summary}}`;
  factorSelect.appendChild(opt);
}});

factorSelect.addEventListener('change', () => {{
  currentFactor = parseInt(factorSelect.value);
  currentIdx = 0;
  updateView();
}});

// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    currentTab = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b === btn));
    switchTab();
  }});
}});

function switchTab() {{
  if (currentTab === 'personas') {{
    sidebarPersonas.classList.remove('sidebar-hidden');
    sidebarFactor.classList.add('sidebar-hidden');
    mainPersonas.classList.remove('main-hidden');
    mainFactor.classList.add('main-hidden');
  }} else {{
    sidebarPersonas.classList.add('sidebar-hidden');
    sidebarFactor.classList.remove('sidebar-hidden');
    mainPersonas.classList.add('main-hidden');
    mainFactor.classList.remove('main-hidden');
    renderFactorTab();
  }}
}}

function updateView() {{
  filteredRecords = RECORDS.filter(r => r.factor === currentFactor);

  // Update factor info panel (personas tab sidebar)
  const f = FACTORS[currentFactor];
  let infoHtml = '';
  if (f.axis_name) {{
    infoHtml += `<div class="axis-chip">${{f.axis_name}}</div>`;
  }}
  if (f.summary) {{
    infoHtml += `<div style="font-weight:bold;margin-bottom:6px">${{f.summary}}</div>`;
  }}
  if (f.description) {{
    infoHtml += `<div class="desc">${{f.description}}</div>`;
  }}
  infoHtml += `<div class="pole pole-high" style="margin-top:8px">`;
  infoHtml += `<div class="pole-label">▲ High: ${{f.positive_pole || '(unlabelled)'}}</div>`;
  if (f.top_positive_items) {{
    f.top_positive_items.forEach(it => {{
      infoHtml += `<div class="loading-item">${{it}}</div>`;
    }});
  }}
  infoHtml += `</div>`;
  infoHtml += `<div class="pole pole-low" style="margin-top:8px">`;
  infoHtml += `<div class="pole-label">▼ Low: ${{f.negative_pole || '(unlabelled)'}}</div>`;
  if (f.top_negative_items) {{
    f.top_negative_items.forEach(it => {{
      infoHtml += `<div class="loading-item">${{it}}</div>`;
    }});
  }}
  infoHtml += `</div>`;
  factorInfo.innerHTML = infoHtml;

  // Update record list
  recordList.innerHTML = '';
  filteredRecords.forEach((r, i) => {{
    const div = document.createElement('div');
    div.className = 'record-entry' + (i === currentIdx ? ' active' : '');
    const poleClass = r.pole === 'high' ? 'pole-tag-high' : 'pole-tag-low';
    const arrow = r.pole === 'high' ? '▲' : '▼';
    div.innerHTML = `<span class="pole-tag ${{poleClass}}">${{arrow}} ${{r.pole}} #${{r.rank}}</span>`
      + `<span class="score">score=${{r.score.toFixed(2)}}</span>`
      + `<div style="font-size:11px;color:#6b7280;margin-top:2px">${{r.sample_id.substring(0,24)}}…</div>`;
    div.addEventListener('click', () => {{ currentIdx = i; renderRecord(); highlightEntry(); }});
    recordList.appendChild(div);
  }});

  renderRecord();
  if (currentTab === 'factor') renderFactorTab();
}}

function highlightEntry() {{
  recordList.querySelectorAll('.record-entry').forEach((el, i) => {{
    el.classList.toggle('active', i === currentIdx);
  }});
  const active = recordList.querySelector('.active');
  if (active) active.scrollIntoView({{ block: 'nearest' }});
}}

function renderRecord() {{
  if (filteredRecords.length === 0) {{
    scrollArea.innerHTML = '<div style="padding:20px;color:#9ca3af">No records for this factor.</div>';
    tbInfo.textContent = '';
    scoreProfile.innerHTML = '';
    return;
  }}
  const rec = filteredRecords[currentIdx];
  const arrow = rec.pole === 'high' ? '▲' : '▼';
  tbInfo.innerHTML = `<span class="tag">${{arrow}} Factor ${{rec.factor}} · ${{rec.pole}} #${{rec.rank}}</span>`
    + ` &nbsp; score=${{rec.score.toFixed(3)}} &nbsp; ${{rec.sample_id}}`;

  // Score profile: show this persona's scores across all factors
  let spHtml = '';
  if (rec.all_scores) {{
    rec.all_scores.forEach((s, fi) => {{
      const cls = fi === currentFactor ? 'sp-item sp-current' : 'sp-item';
      const shortLabel = FACTORS[fi].axis_name
        ? FACTORS[fi].axis_name.substring(0, 8)
        : `F${{fi}}`;
      spHtml += `<div class="${{cls}}" title="${{shortLabel}}"><span class="sp-label">${{shortLabel}}</span><span class="sp-val">${{s.toFixed(1)}}</span></div>`;
    }});
  }}
  scoreProfile.innerHTML = spHtml;

  scrollArea.innerHTML = '';
  rec.messages.forEach(msg => {{
    const div = document.createElement('div');
    const role = msg.role || 'user';
    div.className = `msg msg-${{role}} role-${{role}}`;
    const label = document.createElement('div');
    label.className = 'role-label';
    label.textContent = role;
    div.appendChild(label);
    const body = document.createElement('div');
    body.textContent = msg.content;
    div.appendChild(body);
    scrollArea.appendChild(div);
  }});
  scrollArea.scrollTop = 0;
  highlightEntry();
}}

// ─── Factor tab rendering ──────────────────────────────────────────────────

function renderFactorTab() {{
  const f = FACTORS[currentFactor];
  const fd = FACTOR_DATA[currentFactor];

  // Sidebar: factor description + stats + correlations
  let sbHtml = '';

  // Description
  sbHtml += '<div class="sf-section">';
  sbHtml += '<div class="sf-section-title">Description</div>';
  if (f.axis_name) {{
    sbHtml += `<div class="axis-chip">${{f.axis_name}}</div>`;
  }}
  if (f.summary) {{
    sbHtml += `<div style="font-weight:700;font-size:14px;margin-bottom:6px">${{f.summary}}</div>`;
  }}
  if (f.description) {{
    sbHtml += `<div class="sf-desc">${{f.description}}</div>`;
  }}
  if (!f.summary && !f.description) {{
    sbHtml += '<div style="color:#6b7280;font-style:italic">No LLM description available. Run labeling stage to generate.</div>';
  }}
  sbHtml += `<div style="margin-top:6px"><span class="pole-label" style="color:#4ade80">▲ ${{f.positive_pole || 'high'}}</span></div>`;
  sbHtml += `<div><span class="pole-label" style="color:#f87171">▼ ${{f.negative_pole || 'low'}}</span></div>`;
  sbHtml += '</div>';

  // Variance stats
  sbHtml += '<div class="sf-section">';
  sbHtml += '<div class="sf-section-title">Variance</div>';
  sbHtml += `<div class="sf-stat"><span class="sf-stat-label">SS loading</span><span class="sf-stat-value">${{fd.ss_loading.toFixed(3)}}</span></div>`;
  sbHtml += `<div class="sf-stat"><span class="sf-stat-label">Proportion</span><span class="sf-stat-value">${{(fd.variance_explained * 100).toFixed(1)}}%</span></div>`;
  sbHtml += `<div class="sf-stat"><span class="sf-stat-label">Cumulative</span><span class="sf-stat-value">${{(fd.cumulative_variance * 100).toFixed(1)}}%</span></div>`;
  sbHtml += '</div>';

  // Score stats
  sbHtml += '<div class="sf-section">';
  sbHtml += '<div class="sf-section-title">Score distribution</div>';
  const ss = fd.score_stats;
  sbHtml += `<div class="sf-stat"><span class="sf-stat-label">N</span><span class="sf-stat-value">${{ss.n}}</span></div>`;
  sbHtml += `<div class="sf-stat"><span class="sf-stat-label">Mean</span><span class="sf-stat-value">${{ss.mean.toFixed(3)}}</span></div>`;
  sbHtml += `<div class="sf-stat"><span class="sf-stat-label">Std</span><span class="sf-stat-value">${{ss.std.toFixed(3)}}</span></div>`;
  sbHtml += `<div class="sf-stat"><span class="sf-stat-label">Skew</span><span class="sf-stat-value">${{ss.skew.toFixed(3)}}</span></div>`;
  sbHtml += `<div class="sf-stat"><span class="sf-stat-label">Kurtosis</span><span class="sf-stat-value">${{ss.kurtosis.toFixed(3)}}</span></div>`;
  sbHtml += `<div class="sf-stat"><span class="sf-stat-label">Range</span><span class="sf-stat-value">${{ss.min.toFixed(2)}} … ${{ss.max.toFixed(2)}}</span></div>`;
  sbHtml += '</div>';

  // Factor correlations
  if (fd.correlations.length > 0) {{
    sbHtml += '<div class="sf-section">';
    sbHtml += '<div class="sf-section-title">Factor correlations</div>';
    fd.correlations.sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
    fd.correlations.forEach(c => {{
      const absR = Math.abs(c.r);
      const pct = (absR * 100).toFixed(0);
      const color = c.r >= 0 ? '#4ade80' : '#f87171';
      const fAxis = FACTORS[c.factor].axis_name;
      const fLabel = fAxis
        ? `F${{c.factor}} (${{fAxis}})`
        : FACTORS[c.factor].summary
          ? `F${{c.factor}} (${{FACTORS[c.factor].summary.substring(0,25)}})`
          : `Factor ${{c.factor}}`;
      sbHtml += `<div class="sf-corr-bar">`;
      sbHtml += `<span style="min-width:30px;text-align:right;font-variant-numeric:tabular-nums">${{c.r >= 0 ? '+' : ''}}${{c.r.toFixed(3)}}</span>`;
      sbHtml += `<div class="bar-track"><div class="bar-fill" style="width:${{pct}}%;background:${{color}};${{c.r >= 0 ? 'left:0' : 'right:0'}}"></div></div>`;
      sbHtml += `<span style="color:#9ca3af;font-size:10px">${{fLabel}}</span>`;
      sbHtml += `</div>`;
    }});
    sbHtml += '</div>';
  }}

  sidebarFactor.innerHTML = sbHtml;

  // Main area: loading chart + histogram + items table
  let mainHtml = '';

  // Loading bar chart (SVG)
  mainHtml += '<div class="fv-section">';
  mainHtml += '<div class="fv-section-title">Item Loadings</div>';
  mainHtml += renderLoadingChart(fd.loadings);
  mainHtml += '</div>';

  // Score histogram (SVG)
  mainHtml += '<div class="fv-section">';
  mainHtml += '<div class="fv-section-title">Score Distribution</div>';
  mainHtml += renderHistogram(fd.score_histogram, fd.score_stats);
  mainHtml += '</div>';

  // Top items table
  mainHtml += '<div class="fv-section">';
  mainHtml += '<div class="fv-section-title">Top Loading Items</div>';
  mainHtml += renderItemsTable(fd.loadings);
  mainHtml += '</div>';

  factorView.innerHTML = mainHtml;
}}

function renderLoadingChart(items) {{
  // HTML-based horizontal bar chart — supports full wrapping text labels
  const maxAbs = Math.max(...items.map(it => Math.abs(it.loading)), 0.01);
  let html = '<div style="overflow-y:auto;max-height:650px">';
  html += '<table style="width:100%;border-collapse:collapse;font-size:12px">';

  // Header
  html += '<tr style="border-bottom:1px solid #374151">';
  html += '<th style="text-align:left;padding:4px 8px;color:#9ca3af;font-size:10px;width:50%">ITEM</th>';
  html += '<th style="text-align:center;padding:4px 8px;color:#9ca3af;font-size:10px;width:8%">LOADING</th>';
  html += '<th style="text-align:left;padding:4px 8px;color:#9ca3af;font-size:10px;width:42%">BAR</th>';
  html += '</tr>';

  items.forEach(it => {{
    const color = it.loading >= 0 ? '#4ade80' : '#f87171';
    const pct = (Math.abs(it.loading) / maxAbs * 50).toFixed(1);
    const barStyle = it.loading >= 0
      ? `margin-left:50%;width:${{pct}}%;background:${{color}}`
      : `margin-left:${{50 - parseFloat(pct)}}%;width:${{pct}}%;background:${{color}}`;

    html += '<tr style="border-bottom:1px solid #1f2937">';
    html += `<td style="padding:4px 8px;color:#d1d5db;word-break:break-word;line-height:1.4;font-size:11px">${{escHtml(it.text)}}</td>`;
    html += `<td style="padding:4px 8px;text-align:center;font-variant-numeric:tabular-nums;color:${{color}};font-weight:600">${{it.loading >= 0 ? '+' : ''}}${{it.loading.toFixed(3)}}</td>`;
    html += `<td style="padding:4px 8px"><div style="position:relative;height:14px;background:#1f2937;border-radius:3px;overflow:hidden">`;
    html += `<div style="position:absolute;top:0;height:100%;border-radius:3px;${{barStyle}};opacity:0.8"></div>`;
    html += `<div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:#4b5563"></div>`;
    html += `</div></td>`;
    html += '</tr>';
  }});

  html += '</table></div>';
  return html;
}}

function renderHistogram(hist, stats) {{
  const edges = hist.edges, counts = hist.counts;
  const W = 500, H = 200, padL = 45, padR = 20, padT = 10, padB = 35;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const maxCount = Math.max(...counts, 1);
  const nBins = counts.length;
  const barW = plotW / nBins;

  let svg = `<svg width="${{W}}" height="${{H}}" style="font-family:inherit;font-size:10px">`;

  // Y axis
  svg += `<line x1="${{padL}}" y1="${{padT}}" x2="${{padL}}" y2="${{padT + plotH}}" stroke="#4b5563"/>`;
  for (let t = 0; t <= 4; t++) {{
    const val = Math.round(maxCount * t / 4);
    const y = padT + plotH - (plotH * t / 4);
    svg += `<text x="${{padL - 5}}" y="${{y + 3}}" fill="#6b7280" text-anchor="end">${{val}}</text>`;
    svg += `<line x1="${{padL}}" y1="${{y}}" x2="${{padL + plotW}}" y2="${{y}}" stroke="#2d3748" stroke-dasharray="2,2"/>`;
  }}

  // X axis
  svg += `<line x1="${{padL}}" y1="${{padT + plotH}}" x2="${{padL + plotW}}" y2="${{padT + plotH}}" stroke="#4b5563"/>`;

  // Bars
  counts.forEach((c, i) => {{
    const x = padL + i * barW;
    const h = (c / maxCount) * plotH;
    const y = padT + plotH - h;
    svg += `<rect x="${{x}}" y="${{y}}" width="${{barW - 1}}" height="${{h}}" fill="#60a5fa" opacity="0.7" rx="1"/>`;
  }});

  // X-axis labels (5 evenly spaced)
  for (let t = 0; t <= 4; t++) {{
    const idx = Math.round((nBins) * t / 4);
    const val = idx < edges.length ? edges[idx] : edges[edges.length - 1];
    const x = padL + (plotW * t / 4);
    svg += `<text x="${{x}}" y="${{padT + plotH + 15}}" fill="#6b7280" text-anchor="middle">${{val.toFixed(2)}}</text>`;
  }}

  // Stats annotation
  svg += `<text x="${{padL + plotW}}" y="${{padT + 14}}" fill="#9ca3af" text-anchor="end" font-size="10">`;
  svg += `mean=${{stats.mean.toFixed(2)}}  std=${{stats.std.toFixed(2)}}  skew=${{stats.skew.toFixed(2)}}  kurt=${{stats.kurtosis.toFixed(2)}}`;
  svg += `</text>`;

  svg += '</svg>';
  return svg;
}}

function renderItemsTable(items) {{
  // Show top 10 positive and top 10 negative loading items
  const posItems = items.filter(it => it.loading > 0).slice(0, 10);
  const negItems = items.filter(it => it.loading < 0);
  // negItems are already sorted by |loading| desc, so the most negative are first
  const topNeg = negItems.slice(0, 10);

  let html = '<table class="items-table">';
  html += '<thead><tr><th>Loading</th><th>h²</th><th>Cross</th><th>Item</th></tr></thead>';
  html += '<tbody>';

  function addRow(it) {{
    const cls = it.loading >= 0 ? 'loading-pos' : 'loading-neg';
    const crossLabel = it.max_cross_factor >= 0
      ? `F${{it.max_cross_factor}} (${{it.max_cross_loading >= 0 ? '+' : ''}}${{it.max_cross_loading.toFixed(2)}})`
      : '—';
    html += `<tr>`;
    html += `<td class="${{cls}}">${{it.loading >= 0 ? '+' : ''}}${{it.loading.toFixed(3)}}</td>`;
    html += `<td>${{it.communality.toFixed(3)}}</td>`;
    html += `<td style="font-size:11px;color:#6b7280">${{crossLabel}}</td>`;
    html += `<td class="item-text" title="${{escHtml(it.text)}}">${{escHtml(it.text)}}</td>`;
    html += `</tr>`;
  }}

  if (posItems.length > 0) {{
    html += `<tr><td colspan="4" style="padding:8px 8px 4px;color:#4ade80;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:0.05em;border-bottom:1px solid #065f46">▲ Positive loadings</td></tr>`;
    posItems.forEach(addRow);
  }}
  if (topNeg.length > 0) {{
    html += `<tr><td colspan="4" style="padding:12px 8px 4px;color:#f87171;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:0.05em;border-bottom:1px solid #7f1d1d">▼ Negative loadings</td></tr>`;
    topNeg.forEach(addRow);
  }}

  html += '</tbody></table>';
  return html;
}}

function escHtml(s) {{
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

// ─── Keyboard navigation ───────────────────────────────────────────────────

document.addEventListener('keydown', e => {{
  if (e.target.tagName === 'SELECT') return;
  if (currentTab === 'personas') {{
    if (e.key === 'ArrowDown' || e.key === 'j') {{
      currentIdx = Math.min(currentIdx + 1, filteredRecords.length - 1);
      renderRecord();
    }} else if (e.key === 'ArrowUp' || e.key === 'k') {{
      currentIdx = Math.max(currentIdx - 1, 0);
      renderRecord();
    }}
  }}
  // Tab switching with 1/2 keys
  if (e.key === '1') {{
    currentTab = 'personas';
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === 'personas'));
    switchTab();
  }} else if (e.key === '2') {{
    currentTab = 'factor';
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === 'factor'));
    switchTab();
  }}
}});

updateView();
</script>
</body>
</html>"""

    html_path = save_dir / "factor_extremes.html"
    html_path.write_text(html_content, encoding="utf-8")
    n_factors = len(set(r["factor"] for r in records))
    print(
        f"  [Extremes] Wrote {len(records)} rollouts ({n_per_pole}/pole × {n_factors} factors) "
        f"to {html_path}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4: FACTOR LABELING
# ═════════════════════════════════════════════════════════════════════════════


def _label_factors_by_loadings(
    loadings: np.ndarray,
    column_defs: list[dict],
    top_n: int = TOP_LOADING_ITEMS,
) -> list[dict]:
    """Label factors by inspecting which columns load most strongly.

    Returns a list of dicts (one per factor), each containing the top positive
    and negative loading columns.
    """
    n_factors = loadings.shape[1]
    factor_labels = []

    for fi in range(n_factors):
        col = loadings[:, fi]
        order = np.argsort(col)

        top_positive = order[-top_n:][::-1]
        top_negative = order[:top_n]

        def _item_entry(idx: int) -> dict:
            cd = column_defs[idx]
            entry = {
                "col_id": cd["col_id"],
                "block": cd["block"],
                "text": cd["text"],
                "loading": float(col[idx]),
            }
            if cd["block"] == "likert":
                reverse = cd.get("reverse_keyed", False)
                entry["reverse_keyed"] = reverse
                # Pre-compute behavioral direction so readers don't have to
                agree_more = (col[idx] > 0) != reverse
                entry["behavioral_direction"] = "agree" if agree_more else "disagree"
            return entry

        positive_items = [_item_entry(idx) for idx in top_positive if col[idx] > 0]
        negative_items = [_item_entry(idx) for idx in top_negative if col[idx] < 0]

        factor_labels.append({
            "factor_index": fi,
            "positive_items": positive_items,
            "negative_items": negative_items,
        })

    return factor_labels


def _describe_column_for_labeller(
    col_def: dict,
    loading: float,
    items_by_id: dict[str, dict],
) -> str:
    """Build a rich, human-readable description of what a column measures.

    The description should let the labeller understand what "high" and "low"
    values on this column mean behaviourally, regardless of item type.
    """
    block = col_def["block"]
    sign = "+" if loading > 0 else "−"

    if block == "fc":
        item = items_by_id.get(col_def["item_id"])
        if item and item["type"] == "forced_choice":
            return (
                f"[FC, loading={loading:+.3f}] "
                f'Choice between:\n'
                f'  A (+1): "{item["option_a"]["text"]}"\n'
                f'  B (−1): "{item["option_b"]["text"]}"\n'
                f'  → {sign} loading means personas scoring high on this factor '
                f'tend to choose {"A" if loading > 0 else "B"}.'
            )
        # Fallback if item not found
        return f"[FC, loading={loading:+.3f}] {col_def['text']}"

    elif block == "vignette":
        dim = col_def.get("dimension", "?")
        item = items_by_id.get(col_def["item_id"])
        if item and item["type"] == "vignette":
            lines = [
                f"[Vignette → {dim}, loading={loading:+.3f}]",
                f'Scenario: "{item["scenario"]}"',
                f"Options (with {dim} scores):",
            ]
            for opt in item["options"]:
                dim_score = opt.get("scoring", {}).get(dim, 0)
                lines.append(f'  {opt["label"]} ({dim}={dim_score:+d}): "{opt["text"]}"')
            lines.append(
                f"  → {sign} loading means high-factor personas choose options "
                f"with {'higher' if loading > 0 else 'lower'} {dim} scores in this scenario."
            )
            return "\n".join(lines)
        return f"[Vignette → {dim}, loading={loading:+.3f}] {col_def['text']}"

    elif block == "likert":
        # Compute behavioral direction upfront so the labeller never has to
        # reason about reverse keying — it's an internal scoring detail.
        reverse = col_def.get("reverse_keyed", False)
        agree_more = (loading > 0) != reverse
        return (
            f'[Likert, loading={loading:+.3f}] '
            f'"{col_def["text"]}"\n'
            f'  Scale: 1=strongly disagree … 5=strongly agree\n'
            f'  → {sign} loading means high-factor personas '
            f'{"agree more" if agree_more else "disagree more"} '
            f'with this statement.'
        )

    elif block == "trait_mcq":
        # Benchmark-backed MCQ: each option carries an answer_mapping score
        # (0 = low-trait, 1 = high-trait). Columns come from two encodings:
        #   - letter_1-4: matrix cell is the letter rank (A=1..D=4) and the
        #     letter→trait mapping is shuffled per item → sign is not
        #     trait-interpretable.
        #   - trait_score_0-1: matrix cell is the trait-direction score
        #     (Σ P(letter)·answer_mapping[letter]) → sign IS trait-interpretable.
        dim = col_def.get("dimension", "?")
        encoding = col_def.get("encoding", "letter_1-4")
        item = items_by_id.get(col_def["item_id"])
        lines = [
            f"[TRAIT MCQ → {dim}, loading={loading:+.3f}]",
            f'Question: "{col_def["text"]}"',
        ]
        if item and item.get("type") == "trait_mcq":
            answer_mapping = item.get("answer_mapping", {})
            options_raw = item.get("options", {})
            if isinstance(options_raw, list):
                options = {
                    str(o.get("label", "")): str(o.get("text", ""))
                    for o in options_raw
                }
            else:
                options = {str(k): str(v) for k, v in options_raw.items()}
            high_opts = [
                f'  {letter}: "{options.get(letter, "?")}"'
                for letter, v in answer_mapping.items() if int(v) == 1
            ]
            low_opts = [
                f'  {letter}: "{options.get(letter, "?")}"'
                for letter, v in answer_mapping.items() if int(v) == 0
            ]
            if high_opts:
                lines.append(f"High-{dim} options:")
                lines.extend(high_opts)
            if low_opts:
                lines.append(f"Low-{dim} options:")
                lines.extend(low_opts)
        if encoding == "trait_score_0-1":
            direction = "more high-" if loading > 0 else "more low-"
            lines.append(
                f"  → {sign} loading means high-factor personas pick "
                f"{direction}{dim} options."
            )
        else:
            lines.append(
                f"  → {sign} loading is NOT trait-interpretable (A/B/C/D "
                f"shuffled per item); top-factor personas favour a specific "
                f"letter rank, not a specific trait direction."
            )
        return "\n".join(lines)

    else:
        # Future-proof: unknown block type — include whatever text we have
        return (
            f"[{block}, loading={loading:+.3f}] {col_def['text']}\n"
            f"  → {sign} loading means high-factor personas score "
            f"{'higher' if loading > 0 else 'lower'} on this item."
        )


_BLOCK_TYPE_NAMES = {
    "fc": "forced-choice pairs",
    "vignette": "behavioral vignettes",
    "likert": "Likert-scale statements",
    "trait_mcq": "TRAIT benchmark multiple-choice items",
}


def _build_labeller_system_prompt(present_blocks: set[str]) -> str:
    """Build the factor-labelling system prompt, mentioning only present item types."""
    type_list = ", ".join(
        _BLOCK_TYPE_NAMES[b]
        for b in ("fc", "vignette", "likert", "trait_mcq")
        if b in present_blocks
    )
    formats_sentence = (
        f"Items come from the following measurement format(s): {type_list}."
        if type_list
        else "Items come from a questionnaire."
    )
    return (
        "You are an expert in psychometrics and personality measurement.\n\n"
        "You will be shown questionnaire items that load strongly on latent factors "
        "discovered via factor analysis of a psychometric instrument administered to "
        "a population of LLM personas. Each persona was established through a diverse "
        "multi-turn conversation, then the same questionnaire was administered to "
        "measure behavioral tendencies.\n\n"
        "For each factor, you will see items with high positive loadings (defining one "
        "pole) and items with high negative loadings (defining the opposite pole). "
        f"{formats_sentence}\n\n"
        "Your task is to identify what behavioral dimension each factor captures. "
        'Name both poles clearly (e.g. "assertive directness vs diplomatic deference"), '
        'and also provide a single-word axis name in the style of trait labels like "Openness".'
    )


def _build_labeller_user_message(
    n_factors: int,
    factors_block: str,
    present_blocks: set[str],
) -> str:
    """Build the factor-labelling user message, mentioning only present item types."""
    type_abbrevs = [
        abbrev
        for abbrev, key in [
            ("FC", "fc"),
            ("vignette", "vignette"),
            ("Likert", "likert"),
            ("TRAIT-MCQ", "trait_mcq"),
        ]
        if key in present_blocks
    ]
    item_types_str = ", ".join(type_abbrevs)
    note_line = (
        f"3. Note which item types ({item_types_str}) contribute most — this helps "
        "assess whether the factor reflects genuine behavioral variance or measurement "
        "artefact."
        if item_types_str
        else "3. Note which item types contribute most."
    )
    return (
        f"Below are {n_factors} latent factors. For each factor, I show the questionnaire "
        "items with the strongest positive and negative loadings.\n\n"
        f"{factors_block}\n"
        "Label all factors jointly. For each factor:\n"
        "1. Provide a single-word axis name for the overall dimension.\n"
        "2. Identify the behavioral dimension it captures.\n"
        "3. Name both poles (positive loading pole vs negative loading pole).\n"
        f"4. {note_line[3:]}\n\n"
        "Rules:\n"
        "- Make axis_name a single word in Title Case.\n"
        '- Make each summary ≤12 words, naming both poles with "vs".\n'
        "- Make summaries maximally distinct across factors — avoid synonyms.\n"
        "- If two factors seem related, explain the specific distinction.\n"
        "- Return strict JSON:\n\n"
        "{\n"
        '  "factors": [\n'
        "    {\n"
        '      "factor_index": 0,\n'
        '      "axis_name": "Openness",\n'
        '      "summary": "pole_A vs pole_B",\n'
        '      "description": "2-3 sentence explanation of what this factor captures.",\n'
        '      "positive_pole": "brief name for positive loading end",\n'
        '      "negative_pole": "brief name for negative loading end",\n'
        '      "dominant_item_types": ["fc", "likert"]\n'
        "    }\n"
        "  ]\n"
        "}"
    )


def _label_factors_llm(
    loadings: np.ndarray,
    column_defs: list[dict],
    items: list[dict],
    top_n: int = 8,
    model: str = LABELLER_MODEL,
    provider_name: str = LABELLER_PROVIDER,
    analysis_label: str | None = None,
) -> list[dict]:
    """Label factors by sending high/low loading items to an LLM.

    Args:
        loadings: Factor loading matrix [n_cols × n_factors].
        column_defs: Column definitions aligned with loadings rows.
        items: Original questionnaire items (for full context on vignettes etc).
        top_n: Number of items per pole to send to the labeller.
        model: LLM model to use.
        provider_name: Provider name.
        analysis_label: Optional analysis key (for per-analysis debug files).

    Returns:
        List of dicts with keys: factor_index, axis_name, summary, description,
        positive_pole, negative_pole, dominant_item_types.
    """
    items_by_id = {it["id"]: it for it in items}
    n_factors = loadings.shape[1]

    # Detect which block types are actually present across the full column set.
    present_blocks: set[str] = {cd["block"] for cd in column_defs}

    # Build the factors block
    factor_sections = []
    for fi in range(n_factors):
        col = loadings[:, fi]
        order = np.argsort(col)
        top_pos = [idx for idx in order[-top_n:][::-1] if col[idx] > 0]
        top_neg = [idx for idx in order[:top_n] if col[idx] < 0]

        lines = [f"### Factor {fi}"]
        lines.append(f"\nPositive loading items ({len(top_pos)}):")
        for idx in top_pos:
            lines.append(_describe_column_for_labeller(
                column_defs[idx], float(col[idx]), items_by_id,
            ))
        lines.append(f"\nNegative loading items ({len(top_neg)}):")
        for idx in top_neg:
            lines.append(_describe_column_for_labeller(
                column_defs[idx], float(col[idx]), items_by_id,
            ))
        factor_sections.append("\n".join(lines))

    factors_block = "\n\n" + ("\n\n---\n\n").join(factor_sections) + "\n\n"

    system_prompt = _build_labeller_system_prompt(present_blocks)
    user_message = _build_labeller_user_message(n_factors, factors_block, present_blocks)

    # Call LLM
    config = InferenceConfig(
        model=model,
        provider=provider_name,
        generation=GenerationConfig(
            max_new_tokens=LABELLER_MAX_NEW_TOKENS,
            temperature=0.0,
            do_sample=False,
        ),
        max_concurrent=1,
        timeout=300,
        retry=RetryConfig(max_retries=3, backoff_factor=2.0),
        openrouter=OpenRouterProviderConfig(reasoning=LABELLER_REASONING),
    )
    llm_provider = get_provider(provider_name, config)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    raw_response = ""
    for attempt_idx in range(LABELLER_EMPTY_RESPONSE_RETRIES + 1):
        responses, _, _ = asyncio.run(
            llm_provider.generate_batch_with_metadata_async([messages])
        )
        raw_response = responses[0] if responses else ""
        if raw_response.strip():
            break
        if attempt_idx < LABELLER_EMPTY_RESPONSE_RETRIES:
            logger.warning(
                "Labeller returned an empty response for %s; retrying (%d/%d).",
                analysis_label or "unknown analysis",
                attempt_idx + 1,
                LABELLER_EMPTY_RESPONSE_RETRIES,
            )

    # Save raw response for debugging regardless of parse success
    label_dir = _questionnaire_dir() / "labeling"
    label_dir.mkdir(parents=True, exist_ok=True)
    model_slug = model.replace("/", "_")
    raw_path = label_dir / f"llm_raw_response_{model_slug}.txt"
    raw_path.write_text(raw_response, encoding="utf-8")
    if analysis_label:
        analysis_slug = analysis_label.replace("/", "_")
        keyed_raw_path = label_dir / f"llm_raw_response_{analysis_slug}_{model_slug}.txt"
        keyed_raw_path.write_text(raw_response, encoding="utf-8")
    else:
        keyed_raw_path = raw_path

    return _parse_labeller_json_response(raw_response, keyed_raw_path)


def _parse_labeller_json_response(raw_response: str, raw_path: Path) -> list[dict]:
    """Extract the ``factors`` list from a labeller response string.

    Tries a markdown code block first, then the outermost brace match, then a
    trailing-comma cleanup. Returns an empty list if nothing parses.
    """
    json_text = None

    md_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", raw_response)
    if md_match:
        json_text = md_match.group(1).strip()

    if json_text is None:
        brace_match = re.search(r"\{[\s\S]*\}", raw_response)
        if brace_match:
            json_text = brace_match.group()

    if json_text is not None:
        try:
            parsed = json.loads(json_text)
            return parsed.get("factors", [])
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([}\]])", r"\1", json_text)
            try:
                parsed = json.loads(cleaned)
                return parsed.get("factors", [])
            except json.JSONDecodeError as e:
                logger.warning(
                    "Failed to parse LLM labelling response after cleanup: %s "
                    "(raw response saved to %s)", e, raw_path,
                )

    return []


def _label_factors_claude_cli(
    loadings: np.ndarray,
    column_defs: list[dict],
    items: list[dict],
    top_n: int = 8,
    cli_path: str = LABELLER_CLAUDE_CLI_PATH,
    model: str = LABELLER_CLAUDE_CLI_MODEL,
    effort: str | None = LABELLER_CLAUDE_CLI_EFFORT,
    timeout: int = LABELLER_CLAUDE_CLI_TIMEOUT,
    analysis_label: str | None = None,
) -> list[dict]:
    """Label factors by shelling out to the Claude Code CLI (``claude -p``).

    Builds the same system + user prompts as :func:`_label_factors_llm` and
    pipes the user message via stdin to ``claude -p`` with the system prompt
    appended. Uses ``--output-format json`` and extracts the ``result`` field,
    then reuses :func:`_parse_labeller_json_response` to parse the factors.

    Args:
        loadings: Factor loading matrix [n_cols × n_factors].
        column_defs: Column definitions aligned with loadings rows.
        items: Original questionnaire items.
        top_n: Number of items per pole to send to the labeller.
        cli_path: Path/name of the ``claude`` executable.
        model: Model alias or full ID for ``--model``.
        effort: Reasoning effort level, or None to omit ``--effort``.
        timeout: Seconds before the subprocess is killed.
        analysis_label: Optional analysis key (for per-analysis debug files).

    Returns:
        Parsed list of factor dicts (same shape as :func:`_label_factors_llm`).
    """
    resolved_cli = shutil.which(cli_path) or cli_path
    if not Path(resolved_cli).exists() and not shutil.which(cli_path):
        raise FileNotFoundError(
            f"Claude Code CLI not found at '{cli_path}'. Install it or set "
            "LABELLER_CLAUDE_CLI_PATH."
        )

    items_by_id = {it["id"]: it for it in items}
    n_factors = loadings.shape[1]
    present_blocks: set[str] = {cd["block"] for cd in column_defs}

    factor_sections = []
    for fi in range(n_factors):
        col = loadings[:, fi]
        order = np.argsort(col)
        top_pos = [idx for idx in order[-top_n:][::-1] if col[idx] > 0]
        top_neg = [idx for idx in order[:top_n] if col[idx] < 0]

        lines = [f"### Factor {fi}"]
        lines.append(f"\nPositive loading items ({len(top_pos)}):")
        for idx in top_pos:
            lines.append(_describe_column_for_labeller(
                column_defs[idx], float(col[idx]), items_by_id,
            ))
        lines.append(f"\nNegative loading items ({len(top_neg)}):")
        for idx in top_neg:
            lines.append(_describe_column_for_labeller(
                column_defs[idx], float(col[idx]), items_by_id,
            ))
        factor_sections.append("\n".join(lines))

    factors_block = "\n\n" + ("\n\n---\n\n").join(factor_sections) + "\n\n"

    system_prompt = _build_labeller_system_prompt(present_blocks)
    user_message = _build_labeller_user_message(n_factors, factors_block, present_blocks)

    cmd = [
        resolved_cli,
        "-p",
        "--bare",
        "--output-format", "json",
        "--model", model,
        "--append-system-prompt", system_prompt,
    ]
    if effort:
        cmd += ["--effort", effort]

    label_dir = _questionnaire_dir() / "labeling"
    label_dir.mkdir(parents=True, exist_ok=True)
    model_slug = model.replace("/", "_")
    suffix = f"_{analysis_label.replace('/', '_')}" if analysis_label else ""
    prompt_path = label_dir / f"claudecli_prompt{suffix}_{model_slug}.md"
    prompt_path.write_text(
        f"# SYSTEM\n\n{system_prompt}\n\n# USER\n\n{user_message}\n",
        encoding="utf-8",
    )
    raw_path = label_dir / f"claudecli_raw{suffix}_{model_slug}.txt"
    stderr_path = label_dir / f"claudecli_stderr{suffix}_{model_slug}.txt"

    logger.info(
        "Invoking Claude Code CLI for labelling (%s, model=%s, effort=%s).",
        analysis_label or "unknown", model, effort,
    )

    try:
        proc = subprocess.run(
            cmd,
            input=user_message,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        logger.warning("Claude Code CLI timed out after %ds: %s", timeout, e)
        return []

    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    if proc.returncode != 0:
        logger.warning(
            "Claude Code CLI exited with code %d (stderr saved to %s).",
            proc.returncode, stderr_path,
        )
        raw_path.write_text(proc.stdout or "", encoding="utf-8")
        return []

    stdout = proc.stdout or ""

    # --output-format json wraps the assistant response in a single JSON
    # object with a "result" field. Fall back to treating stdout as the raw
    # response if that structure isn't present (e.g. CLI upgrade changes it).
    raw_response = stdout
    try:
        envelope = json.loads(stdout)
        if isinstance(envelope, dict) and "result" in envelope:
            raw_response = envelope["result"] or ""
    except json.JSONDecodeError:
        pass

    raw_path.write_text(raw_response, encoding="utf-8")

    return _parse_labeller_json_response(raw_response, raw_path)


def run_stage_labeling(
    fa_results: dict,
    items: list[dict],
) -> dict:
    """Label discovered factors using column loadings and LLM labeling.

    Args:
        fa_results: Dict of FA results keyed by analysis variant.
        items: Original questionnaire items (for rich LLM descriptions).
    """
    label_dir = _questionnaire_dir() / "labeling"
    label_dir.mkdir(parents=True, exist_ok=True)

    all_labels = {}

    for key, result in fa_results.items():
        if result.get("n_factors", 0) == 0:
            continue
        if "fa_result" not in result:
            continue

        print(f"\n[Stage 4] Labeling factors for: {key}")

        fa_result = result["fa_result"]
        column_defs = result["column_defs"]
        loadings = fa_result["loadings"]

        # Approach A: Loading inspection (quick, no API call)
        factor_labels = _label_factors_by_loadings(loadings, column_defs)

        for fl in factor_labels:
            fi = fl["factor_index"]
            print(f"\n  Factor {fi}:")
            if fl["positive_items"]:
                top_pos = fl["positive_items"][0]
                print(f"    + [{top_pos['block']}] {top_pos['text'][:60]}... ({top_pos['loading']:.3f})")
            if fl["negative_items"]:
                top_neg = fl["negative_items"][0]
                print(f"    - [{top_neg['block']}] {top_neg['text'][:60]}... ({top_neg['loading']:.3f})")

        with open(label_dir / f"item_labels_{key}.json", "w") as f:
            json.dump(factor_labels, f, indent=2, ensure_ascii=False)

        # Approach B: LLM labeling with psychometric-aware prompt
        if LABELLER_USE_CLAUDE_CLI:
            labeller_name = "claudecli"
            active_model = LABELLER_CLAUDE_CLI_MODEL

            def _invoke_labeller() -> list[dict]:
                return _label_factors_claude_cli(
                    loadings, column_defs, items,
                    top_n=TOP_LOADING_ITEMS,
                    analysis_label=key,
                )
        else:
            labeller_name = None
            active_model = LABELLER_MODEL

            def _invoke_labeller() -> list[dict]:
                return _label_factors_llm(
                    loadings, column_defs, items,
                    top_n=TOP_LOADING_ITEMS,
                    model=LABELLER_MODEL,
                    provider_name=LABELLER_PROVIDER,
                    analysis_label=key,
                )

        model_slug = active_model.replace("/", "_")
        cache_suffix = f"_{labeller_name}_{model_slug}" if labeller_name else f"_{model_slug}"
        llm_cache_path = label_dir / f"llm_labels_{key}{cache_suffix}.json"
        try:
            if llm_cache_path.exists():
                print(f"\n  [Cache] Loading LLM labels from {llm_cache_path.name}")
                llm_labels = _load_llm_labels_from_path(llm_cache_path)
                if not llm_labels or not _llm_labels_have_axis_names(llm_labels):
                    reason = "empty or invalid"
                    if llm_labels and not _llm_labels_have_axis_names(llm_labels):
                        reason = "missing axis_name fields"
                    print(
                        f"  [Cache] {llm_cache_path.name} is {reason}; "
                        "regenerating labels."
                    )
                    llm_labels = _invoke_labeller()
                    if not llm_labels:
                        raise ValueError(
                            "Labeller returned no parseable factor labels; "
                            "not writing empty cache."
                        )
                    with open(llm_cache_path, "w") as f:
                        json.dump(llm_labels, f, indent=2, ensure_ascii=False)
            else:
                llm_labels = _invoke_labeller()
                if not llm_labels:
                    raise ValueError(
                        "Labeller returned no parseable factor labels; "
                        "not writing empty cache."
                    )
                with open(llm_cache_path, "w") as f:
                    json.dump(llm_labels, f, indent=2, ensure_ascii=False)

            print(f"\n  LLM labels for {key}:")
            for fl in llm_labels:
                fi = fl.get("factor_index", "?")
                axis_name = fl.get("axis_name", "")
                summary = fl.get("summary", "(no summary)")
                desc = fl.get("description", "")
                axis_prefix = f"[{axis_name}] " if axis_name else ""
                print(f"    Factor {fi}: {axis_prefix}{summary}")
                if desc:
                    print(f"      {desc[:120]}")

        except Exception as e:
            logger.warning("LLM labeling failed for %s: %s", key, e)
            llm_labels = _load_latest_nonempty_llm_labels(
                label_dir,
                key,
                require_axis_names=True,
            )
            if llm_labels:
                print(
                    f"  [Fallback] Using newest non-empty axis-name cached labels for {key}."
                )

        all_labels[key] = {
            "item_labels": factor_labels,
            "llm_labels": llm_labels,
        }

    # Re-export factor extremes HTML now that labels are available. Each
    # result carries its own save_dir (letter-encoded → factor_analysis/<key>,
    # trait-oriented → factor_analysis_trait_oriented/<key>).
    base_dir = _questionnaire_dir() / "factor_analysis"
    for key, result in fa_results.items():
        if result.get("n_factors", 0) == 0 or "fa_result" not in result:
            continue
        save_dir = Path(result.get("save_dir", base_dir / key))
        _export_factor_extremes_html(
            fa_result=result["fa_result"],
            column_defs=result["column_defs"],
            metadata=result["metadata"],
            label=key,
            save_dir=save_dir,
        )

    return all_labels


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 5: VALIDATION
# ═════════════════════════════════════════════════════════════════════════════


def _validation_shuffle_test(
    data_clean: np.ndarray,
    pa_real: dict,
    val_dir: Path,
    plt,
) -> dict:
    """Shuffle control: permute columns and check that no factors emerge."""
    print("\n[Stage 5] Validation — Shuffle control")
    from src_dev.factor_analysis.validation import shuffle_control_test

    return shuffle_control_test(
        data_clean, pa_real, val_dir, seed=SEED, plt=plt,
    )


def _validation_predictivity_test(
    data_clean: np.ndarray,
    val_dir: Path,
    plt,
) -> dict:
    """Held-out item predictivity: factor scores from training items predict held-out items."""
    print("\n[Stage 5] Validation — Held-out item predictivity")
    from src_dev.factor_analysis.validation import item_holdout_predictivity_test

    return item_holdout_predictivity_test(
        data_clean, val_dir,
        holdout_n_items=HOLDOUT_N_ITEMS,
        fa_method=FA_METHOD,
        rotation=FA_ROTATIONS[0],
        seed=SEED + 1,
        plt=plt,
    )


def _validation_stability_test(
    fa_results: dict,
    val_dir: Path,
    plt,
) -> dict:
    """Stability: ICC(1) per factor across rollouts from the same seed prompt.

    Runs one ICC pass per FA variant with factors (rotation × residualization),
    writing each to ``val_dir/stability/<fa_key>/``. Returns a dict keyed by
    ``fa_key`` plus a ``mean_icc1_by_variant`` summary.
    """
    print("\n[Stage 5] Validation — Stability (ICC)")
    from src_dev.factor_analysis.validation import stability_icc_test

    variant_keys = [
        k for k, v in fa_results.items()
        if v.get("n_factors", 0) > 0 and "fa_result" in v
    ]
    if not variant_keys:
        return stability_icc_test(fa_results, val_dir, plt=plt)

    per_variant: dict[str, dict] = {}
    for key in variant_keys:
        variant_dir = val_dir / "stability" / key
        per_variant[key] = stability_icc_test(
            fa_results, variant_dir, fa_key=key, plt=plt,
        )

    mean_icc1_by_variant = {
        k: v.get("mean_icc1") for k, v in per_variant.items()
    }
    all_pass = all(v.get("pass") for v in per_variant.values())
    return {
        "per_variant": per_variant,
        "mean_icc1_by_variant": mean_icc1_by_variant,
        "pass": all_pass,
    }


_ARCH_SCEN_CACHE: dict[str, tuple[str, str]] | None = None


def _load_archetype_scenario_lookup() -> dict[str, tuple[str, str]]:
    """Build sample_id → (archetype, scenario_id) from the rollout canonical
    dataset. The seed message content is tagged ``[scenario: X | archetype: Y]``.

    Returns {} if the canonical dataset is missing or unparseable — callers
    must handle the empty case.
    """
    global _ARCH_SCEN_CACHE
    if _ARCH_SCEN_CACHE is not None:
        return _ARCH_SCEN_CACHE

    import re
    path = _rollout_dir() / "datasets" / "canonical_samples.jsonl"
    if not path.exists():
        _ARCH_SCEN_CACHE = {}
        return _ARCH_SCEN_CACHE

    pat = re.compile(r"\[scenario:\s*([^|\]]+?)\s*\|\s*archetype:\s*([^\]]+?)\s*\]")
    lookup: dict[str, tuple[str, str]] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec.get("sample_id")
            messages = rec.get("input", {}).get("messages") or []
            if not sid or not messages:
                continue
            m = pat.search(messages[0].get("content", "") or "")
            if m:
                lookup[sid] = (m.group(2), m.group(1))
    _ARCH_SCEN_CACHE = lookup
    return lookup


def _enrich_meta_with_archetype_scenario(meta: list[dict]) -> list[dict]:
    """Return a shallow copy of ``meta`` with archetype + scenario_id added
    per row where resolvable via sample_id. Rows that can't be resolved are
    left without those keys — ``prompt_effects`` warns on missing keys."""
    lookup = _load_archetype_scenario_lookup()
    if not lookup:
        print(
            "  [Variance decomp] canonical_samples.jsonl not found — "
            "archetype/scenario_id will be NaN."
        )
        return meta

    n_hit = 0
    enriched: list[dict] = []
    for row in meta:
        sid = row.get("sample_id")
        hit = lookup.get(sid) if sid else None
        if hit is not None:
            arch, sc_id = hit
            enriched.append({**row, "archetype": arch, "scenario_id": sc_id})
            n_hit += 1
        else:
            enriched.append(dict(row))
    if n_hit < len(meta):
        print(
            f"  [Variance decomp] enriched {n_hit}/{len(meta)} rows with "
            "archetype/scenario_id (rest lack a resolvable sample_id)."
        )
    return enriched


def _validation_variance_decomp(
    fa_results: dict,
    val_dir: Path,
    plt,
) -> dict:
    """η² per factor per metadata field via ``prompt_effects``.

    Flag direction is asymmetric (see VARIANCE_DECOMP_* constants):
      * scenario_id as a ceiling — high η² = scenario artifact.
      * archetype as a floor — at least one factor must have η² ≥ the floor,
        else the factor structure is blind to the constructed personas.
    """
    print("\n[Stage 5] Validation — Variance decomposition (η²)")
    from src_dev.factor_analysis.interpretation import prompt_effects

    fa_key = None
    for key, result in fa_results.items():
        if result.get("n_factors", 0) > 0 and "fa_result" in result:
            fa_key = key
            break
    if fa_key is None:
        return {"pass": False, "note": "No FA results with factors"}

    fa_result = fa_results[fa_key]["fa_result"]
    meta = fa_results[fa_key]["metadata"]
    scores = fa_result["scores"]
    n_factors = scores.shape[1]

    # Questionnaire metadata only carries sample_id / input_group_id. The
    # archetype + scenario_id labels live in the upstream rollout canonical
    # dataset (parsed from `input.messages[0].content`). Enrich in place so
    # prompt_effects can key on them.
    meta = _enrich_meta_with_archetype_scenario(meta)

    per_field: dict[str, list[float]] = {}
    for field in VARIANCE_DECOMP_FIELDS:
        eta2 = prompt_effects(scores, meta, group_field=field)
        per_field[field] = [float(v) for v in eta2]

    scenario_eta2 = np.array(per_field.get("scenario_id", [0.0] * n_factors))
    archetype_eta2 = np.array(per_field.get("archetype", [0.0] * n_factors))

    scenario_flagged = [int(i) for i, v in enumerate(scenario_eta2)
                        if v >= VARIANCE_DECOMP_SCENARIO_CEILING]
    archetype_signal_max = float(archetype_eta2.max()) if len(archetype_eta2) else 0.0
    archetype_floor_pass = archetype_signal_max >= VARIANCE_DECOMP_ARCHETYPE_FLOOR

    passed = (len(scenario_flagged) == 0) and archetype_floor_pass

    result = {
        "fa_key": fa_key,
        "n_factors": n_factors,
        "eta2_per_field": per_field,
        "scenario_ceiling": VARIANCE_DECOMP_SCENARIO_CEILING,
        "archetype_floor": VARIANCE_DECOMP_ARCHETYPE_FLOOR,
        "scenario_flagged_factors": scenario_flagged,
        "archetype_max_eta2": archetype_signal_max,
        "archetype_floor_pass": archetype_floor_pass,
        "pass": passed,
    }

    out_dir = val_dir / "variance_decomp"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "variance_decomp.json", "w") as f:
        json.dump(result, f, indent=2)

    try:
        fields = list(per_field.keys())
        x = np.arange(n_factors)
        width = 0.8 / len(fields)
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * n_factors + 2), 4.5))
        colors = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626"]
        for i, field in enumerate(fields):
            ax.bar(
                x + (i - (len(fields) - 1) / 2) * width,
                per_field[field], width, label=field, color=colors[i % len(colors)],
            )
        ax.axhline(VARIANCE_DECOMP_SCENARIO_CEILING, color="#dc2626",
                   linestyle="--", linewidth=0.8, alpha=0.6,
                   label=f"scenario ceiling ({VARIANCE_DECOMP_SCENARIO_CEILING})")
        ax.axhline(VARIANCE_DECOMP_ARCHETYPE_FLOOR, color="#16a34a",
                   linestyle=":", linewidth=0.8, alpha=0.6,
                   label=f"archetype floor ({VARIANCE_DECOMP_ARCHETYPE_FLOOR})")
        ax.set_xticks(x)
        ax.set_xticklabels([f"F{i}" for i in range(n_factors)])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("η² (variance explained)")
        ax.set_xlabel("Factor")
        ax.set_title(
            "Variance decomposition — scenario as ceiling, archetype as floor",
            fontsize=11, fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "variance_decomp.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"  Variance decomposition plot failed: {exc}")

    print(
        f"  Variance decomp: scenario-flagged {len(scenario_flagged)}/{n_factors} "
        f"(η²≥{VARIANCE_DECOMP_SCENARIO_CEILING}), "
        f"archetype max η²={archetype_signal_max:.2f} "
        f"(floor={VARIANCE_DECOMP_ARCHETYPE_FLOOR}) "
        f"({'PASS' if passed else 'FAIL'})"
    )
    return result


def _validation_trait_convergence(
    fa_results: dict,
    val_dir: Path,
    plt,
) -> dict:
    """Spearman ρ between factor scores and TRAIT OCEAN scores."""
    print("\n[Stage 5] Validation — TRAIT convergent validity")
    from src_dev.factor_analysis.trait_convergence import convergent_validity

    trait_csv = (
        _questionnaire_dir() / "questionnaire" / "trait_scores"
        / "trait_scores_with_metadata.csv"
    )
    if not trait_csv.exists():
        msg = f"trait_scores_with_metadata.csv not found at {trait_csv}"
        print(f"  {msg} — skipped")
        return {"pass": False, "note": msg}

    import pandas as pd

    df = pd.read_csv(trait_csv)
    if "sample_id" not in df.columns:
        return {"pass": False, "note": "trait_scores CSV missing sample_id column"}
    trait_cols = [c for c in df.columns
                  if c not in ("k", "sample_id", "input_group_id")]
    by_sid = df.set_index("sample_id")[trait_cols]

    fa_key = None
    for key, result in fa_results.items():
        if result.get("n_factors", 0) > 0 and "fa_result" in result:
            fa_key = key
            break
    if fa_key is None:
        return {"pass": False, "note": "No FA results with factors"}

    fa_entry = fa_results[fa_key]
    scores = fa_entry["fa_result"]["scores"]
    meta = fa_entry["metadata"]

    n = len(meta)
    t = len(trait_cols)
    trait_matrix = np.full((n, t), np.nan)
    for i, m in enumerate(meta):
        sid = m.get("sample_id")
        if sid in by_sid.index:
            trait_matrix[i] = by_sid.loc[sid].values

    n_aligned = int(np.sum(~np.all(np.isnan(trait_matrix), axis=1)))
    if n_aligned < 20:
        return {
            "pass": False,
            "note": f"Too few aligned personas: {n_aligned}",
            "fa_key": fa_key,
        }

    out_dir = val_dir / "trait_convergence"
    result = convergent_validity(
        scores, trait_matrix, out_dir,
        trait_names=trait_cols,
        factor_names=[f"F{i}" for i in range(scores.shape[1])],
        method="spearman",
        n_bootstrap=TRAIT_CONVERGENCE_N_BOOTSTRAP,
        trait_hit_threshold=TRAIT_CONVERGENCE_HIT_THRESHOLD,
        min_trait_hits=TRAIT_CONVERGENCE_MIN_HITS,
        seed=SEED + 5,
        plt=plt,
    )
    result["fa_key"] = fa_key
    result["n_aligned_personas"] = n_aligned
    return result


def _pick_anchor_fa(fa_results: dict) -> tuple[str, dict] | tuple[None, None]:
    """Return (key, entry) for the first FA result with factors."""
    for key, result in fa_results.items():
        if result.get("n_factors", 0) > 0 and "fa_result" in result:
            return key, result
    return None, None


def run_stage_validation(
    response_matrix: np.ndarray,
    metadata: list[dict],
    column_defs: list[dict],
    fa_results: dict,
) -> dict:
    """Run validation tests, each gated by ``VALIDATION_TESTS_TO_RUN``.

    See ``VALIDATION_TESTS_TO_RUN`` for available test names. Each writes its
    own JSON + plot(s) under ``<questionnaire_dir>/validation/``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    val_dir = _questionnaire_dir() / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess once for tests that need the clean response matrix.
    data_clean, meta_clean, cols_clean, _ = _preprocess_response_matrix(
        response_matrix, metadata, column_defs, do_residualize=False,
    )

    # Parallel analysis anchor for the shuffle-scree comparison plot.
    pa_real_result = None
    for result in fa_results.values():
        if result.get("parallel_analysis") is not None:
            pa_real_result = result["parallel_analysis"]
            break
    if pa_real_result is None:
        pa_real_result = parallel_analysis(data_clean, random_state=SEED)

    anchor_key, anchor_entry = _pick_anchor_fa(fa_results)
    anchor_k = anchor_entry["n_factors"] if anchor_entry else None
    anchor_loadings = (
        anchor_entry["fa_result"]["loadings"] if anchor_entry else None
    )

    results: dict[str, dict] = {}

    def _enabled(name: str) -> bool:
        return name in VALIDATION_TESTS_TO_RUN

    if _enabled("shuffle_control"):
        results["shuffle"] = _validation_shuffle_test(
            data_clean, pa_real_result, val_dir, plt,
        )
    if _enabled("item_holdout"):
        results["item_holdout"] = _validation_predictivity_test(
            data_clean, val_dir, plt,
        )
    if _enabled("stability_icc"):
        results["stability_icc"] = _validation_stability_test(
            fa_results, val_dir, plt,
        )
    if _enabled("variance_decomp"):
        results["variance_decomp"] = _validation_variance_decomp(
            fa_results, val_dir, plt,
        )
    if _enabled("trait_convergence"):
        results["trait_convergence"] = _validation_trait_convergence(
            fa_results, val_dir, plt,
        )

    # Sweeps / sensitivity / persona×item CV all need the anchor loadings.
    if anchor_loadings is not None:
        from src_dev.factor_analysis.cross_validation import (
            k_sensitivity as _k_sensitivity,
            persona_item_cv as _persona_item_cv,
            stability_sweep as _stability_sweep,
        )
        if _enabled("stability_sweep_random50"):
            print("\n[Stage 5] Validation — Stability sweep (random 50%)")
            results["stability_sweep_random50"] = _stability_sweep(
                data_clean, meta_clean, anchor_k, anchor_loadings,
                val_dir, mode="random_50",
                n_splits=STABILITY_SWEEP_N_RANDOM_SPLITS,
                pa_iterations=STABILITY_SWEEP_PA_ITERATIONS,
                fa_method=FA_METHOD, rotation=FA_ROTATIONS[0],
                pass_threshold_median_phi=STABILITY_SWEEP_PASS_THRESHOLD_PHI,
                seed=SEED + 10,
            )
        if _enabled("stability_sweep_loao"):
            print("\n[Stage 5] Validation — Stability sweep (leave-one-archetype-out)")
            results["stability_sweep_loao"] = _stability_sweep(
                data_clean, meta_clean, anchor_k, anchor_loadings,
                val_dir, mode="loao",
                pa_iterations=STABILITY_SWEEP_PA_ITERATIONS,
                fa_method=FA_METHOD, rotation=FA_ROTATIONS[0],
                pass_threshold_median_phi=STABILITY_SWEEP_PASS_THRESHOLD_PHI,
                seed=SEED + 11,
            )
        if _enabled("stability_sweep_loso_top10"):
            print(
                f"\n[Stage 5] Validation — Stability sweep "
                f"(leave-one-scenario-out, top {STABILITY_SWEEP_LOSO_TOP_N})"
            )
            results["stability_sweep_loso_top10"] = _stability_sweep(
                data_clean, meta_clean, anchor_k, anchor_loadings,
                val_dir, mode="loso",
                top_n_scenarios=STABILITY_SWEEP_LOSO_TOP_N,
                pa_iterations=STABILITY_SWEEP_PA_ITERATIONS,
                fa_method=FA_METHOD, rotation=FA_ROTATIONS[0],
                pass_threshold_median_phi=STABILITY_SWEEP_PASS_THRESHOLD_PHI,
                seed=SEED + 12,
            )
        if _enabled("k_sensitivity"):
            print("\n[Stage 5] Validation — k ± 1 sensitivity")
            results["k_sensitivity"] = _k_sensitivity(
                data_clean, k_center=anchor_k, out_dir=val_dir,
                fa_method=FA_METHOD, rotation=FA_ROTATIONS[0],
                match_threshold=K_SENSITIVITY_MATCH_THRESHOLD,
                independent_threshold=K_SENSITIVITY_INDEPENDENT_THRESHOLD,
            )
        if _enabled("persona_item_cv"):
            print("\n[Stage 5] Validation — Persona × item CV")
            results["persona_item_cv"] = _persona_item_cv(
                data_clean, meta_clean, anchor_k, val_dir,
                persona_split=PERSONA_ITEM_CV_SPLIT,
                n_trials=PERSONA_ITEM_CV_N_TRIALS,
                subset_strategy=PERSONA_ITEM_CV_SUBSET_STRATEGY,
                fa_method=FA_METHOD, rotation=FA_ROTATIONS[0],
                seed=SEED + 13,
            )

    print("\n" + "=" * 60)
    print("[Stage 5] Validation Summary:")
    for test_name, test_result in results.items():
        status = "PASS" if test_result.get("pass") else "FAIL/SKIP"
        print(f"  {test_name}: {status}")
    print("=" * 60)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    args = _parse_args()
    global ACTIVE_USER_SIMULATOR_MODE
    ACTIVE_USER_SIMULATOR_MODE = (
        args.user_simulator_mode or DEFAULT_USER_SIMULATOR_MODE
    )

    retry_terminal_sample_ids: list[str] = []
    retry_mode = "off"
    if args.retry_terminal_samples:
        retry_terminal_sample_ids = _load_terminal_sample_ids_from_run(_rollout_dir())
        retry_mode = "auto"
    elif args.retry_terminal_samples_file is not None:
        retry_terminal_sample_ids = _load_retry_terminal_sample_ids(
            args.retry_terminal_samples_file
        )
        retry_mode = "file"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    print("=" * 60)
    print("Psychometric Factor Analysis of LLM Persona Rollouts")
    print("=" * 60)
    print(f"Rollout run ID: {_rollout_run_id()}")
    print(f"Questionnaire run ID: {_questionnaire_run_id()}")
    print(f"User simulator mode: {_current_user_simulator_mode()}")
    print(f"Stages to run: {STAGES_TO_RUN}")
    if retry_terminal_sample_ids:
        if retry_mode == "auto":
            print(
                "Retry terminal samples: auto "
                f"({len(retry_terminal_sample_ids)} sample IDs from current run)"
            )
        else:
            print(
                "Retry terminal samples file: "
                f"{args.retry_terminal_samples_file} ({len(retry_terminal_sample_ids)} sample IDs)"
            )
    print()

    # Save config
    config_dir = _questionnaire_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_dir / "config.json", "w") as f:
        json.dump({
            "seed": SEED,
            "rollout_run_id": _rollout_run_id(),
            "questionnaire_run_id": _questionnaire_run_id(),
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

    # ── Stage 1 ──────────────────────────────────────────────────────────
    if "rollouts" in STAGES_TO_RUN:
        print("\n" + "=" * 60)
        print("[Stage 1] Generating rollouts")
        print("=" * 60)
        run_stage_rollouts(retry_terminal_sample_ids=retry_terminal_sample_ids)

    # Load questionnaire items once; reused by inspection, labeling, and validation.
    questionnaire_items, _ = _load_questionnaire()

    # ── Stage 2 ──────────────────────────────────────────────────────────
    response_matrix = None
    metadata = None
    column_defs = None

    if "questionnaire" in STAGES_TO_RUN:
        print("\n" + "=" * 60)
        print("[Stage 2] Applying questionnaire")
        print("=" * 60)
        response_matrix, metadata, column_defs = run_stage_questionnaire()

    if WRITE_QUESTIONNAIRE_INSPECTION_FILE:
        _write_questionnaire_inspection_file(questionnaire_items)

    # ── Stage 2.5: Trait scoring (TRAIT-benchmark questionnaires only) ──
    if "trait_scoring" in STAGES_TO_RUN:
        print("\n" + "=" * 60)
        print("[Stage 2.5] Trait scoring + plots")
        print("=" * 60)
        run_stage_trait_scoring(metadata=metadata)

    # Load if needed for later stages
    if response_matrix is None and any(
        s in STAGES_TO_RUN for s in ["factor_analysis", "labeling", "validation"]
    ):
        q_dir = _questionnaire_dir() / "questionnaire"
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
                _, column_defs = _load_questionnaire()
        else:
            print("ERROR: Questionnaire results not found. Run stages 1-2 first.")
            sys.exit(1)

    # Filter out samples with consecutive assistant turns (resume-bug artifact).
    # When loaded from disk the questionnaire may already include these rows, so
    # we strip them here before any downstream analysis stage runs.
    if response_matrix is not None and metadata is not None:
        bad_sample_ids = find_consecutive_assistant_turn_sample_ids(_rollout_dir())
        if bad_sample_ids:
            keep = np.array([m["sample_id"] not in bad_sample_ids for m in metadata])
            n_removed = int((~keep).sum())
            if n_removed:
                response_matrix = response_matrix[keep]
                metadata = [m for m, k in zip(metadata, keep) if k]
                print(
                    f"Excluded {n_removed} samples with consecutive assistant turns "
                    f"(resume-bug artifact) from downstream analysis"
                )

    # ── Stage 3 ──────────────────────────────────────────────────────────
    fa_results = None
    if "factor_analysis" in STAGES_TO_RUN:
        print("\n" + "=" * 60)
        print("[Stage 3] Factor analysis")
        print("=" * 60)
        fa_results = run_stage_factor_analysis(
            response_matrix, metadata, column_defs, items=questionnaire_items,
        )

    # ── Stage 4 ──────────────────────────────────────────────────────────
    if "labeling" in STAGES_TO_RUN:
        print("\n" + "=" * 60)
        print("[Stage 4] Factor labeling")
        print("=" * 60)
        if fa_results is None:
            print("ERROR: Factor analysis results not available. Run stage 3 first.")
            sys.exit(1)
        run_stage_labeling(fa_results, questionnaire_items)

    # ── Stage 5 ──────────────────────────────────────────────────────────
    if "validation" in STAGES_TO_RUN:
        print("\n" + "=" * 60)
        print("[Stage 5] Validation")
        print("=" * 60)
        if fa_results is None:
            print("ERROR: Factor analysis results not available. Run stage 3 first.")
            sys.exit(1)
        run_stage_validation(response_matrix, metadata, column_defs, fa_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
