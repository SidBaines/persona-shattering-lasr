"""Psychometric factor analysis of LLM persona rollouts.

Applies standard psychometric techniques to discover latent behavioral dimensions
in LLMs. Creates a "population" of personas via diverse multi-turn rollouts,
administers a Likert-scale questionnaire to each persona at the N+1th turn, and
runs factor analysis (Horn's parallel analysis + PAF) on the resulting response
matrix. Factors are discovered unsupervised and labeled post-hoc.

Stages:
    1. rollouts       — Generate diverse multi-turn conversations (one per persona)
    2. questionnaire   — Branch each rollout and administer questionnaire items
    3. factor_analysis — Parallel analysis + PAF on the response matrix
    4. labeling        — Interpret factors via item loadings and LLM labeling
    5. validation      — Stability, predictivity, and shuffle-control tests
"""

from __future__ import annotations

# ── Seeds (set before any stochastic imports) ────────────────────────────────
import random

import numpy as np

SEED = 420
random.seed(SEED)
np.random.seed(SEED)

# ── Standard library ─────────────────────────────────────────────────────────
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

# ── Third-party ──────────────────────────────────────────────────────────────
from dotenv import load_dotenv

load_dotenv()

# ── Local imports ────────────────────────────────────────────────────────────
from src_dev.common.config import DatasetConfig, GenerationConfig
from src_dev.datasets import load_samples, materialize_canonical_samples
from src_dev.factor_analysis.factor_analysis import adequacy_tests, run_factor_analysis
from src_dev.factor_analysis.parallel_analysis import parallel_analysis
from src_dev.factor_analysis.persistence import save_factor_analysis
from src_dev.factor_analysis.preprocessing import residualize
from src_dev.inference import InferenceConfig
from src_dev.inference.config import OpenRouterProviderConfig, RetryConfig
from src_dev.inference.providers import get_provider
from src_dev.inference.providers.base import PromptInput
from src_dev.rollout_generation.config import (
    RolloutGenerationConfig,
    UserSimulatorConfig,
)
from src_dev.rollout_generation.prompts import register_user_simulator_template
from src_dev.rollout_generation.run import run_rollout_generation
from src_dev.utils.hf_hub import (
    download_from_dataset_repo,
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
SEED_DATASET = "datasets/assistant-axis-extraction-questions-all.jsonl"
MAX_PROMPTS = 5
NUM_ROLLOUTS_PER_PROMPT = 1
NUM_CONVERSATION_TURNS = 10
ASSISTANT_MODEL = "meta-llama/llama-3.1-8b-instruct"
ASSISTANT_PROVIDER = "openrouter"
USER_MODEL = "z-ai/glm-4.5-air"
USER_PROVIDER = "openrouter"
TEMPERATURE = 1.0
ASSISTANT_MAX_NEW_TOKENS = 4096
USER_MAX_NEW_TOKENS = 4096
USER_PROMPT_VERSION = "v3"

# ── Stage 2: Questionnaire ──────────────────────────────────────────────────
QUESTIONNAIRE_PATH = "datasets/psychometric_questionnaire.json"
QUESTIONNAIRE_VERSION = "v2"  # bump when changing items
QUESTIONNAIRE_PHRASING = "natural"  # "natural", "direct", "contextual"
LIKERT_SCALE = 5
MAX_PARSE_RETRIES = 3
QUESTIONNAIRE_MAX_CONCURRENT = 32
QUESTIONNAIRE_MAX_NEW_TOKENS = 32
QUESTIONNAIRE_TIMEOUT = 60

# ── Stage 3: Factor analysis ────────────────────────────────────────────────
FA_METHOD = "principal"
FA_ROTATIONS = ["oblimin", "varimax"]
RESIDUALIZE_OPTIONS = [False, True]
MIN_ITEM_VARIANCE = 0.05  # drop items with variance below this

# ── Stage 4: Labeling ───────────────────────────────────────────────────────
LABELLER_MODEL = "z-ai/glm-4.5-air"
LABELLER_PROVIDER = "openrouter"
TOP_LOADING_ITEMS = 10

# ── Stage 5: Validation ─────────────────────────────────────────────────────
STABILITY_N_PROMPTS = 100
HOLDOUT_N_ITEMS = 20

# ── Pipeline control ────────────────────────────────────────────────────────
STAGES_TO_RUN = [
    "rollouts",
    "questionnaire",
    "factor_analysis",
    # "labeling",
    # "validation",
]

# ── Debug / inspection ─────────────────────────────────────────────────────
# When True, writes a JSONL file with the first N questionnaire items appended
# to each rollout conversation, for inspection with the TUI viewer.
WRITE_QUESTIONNAIRE_INSPECTION_FILE = True
INSPECTION_ITEMS_PER_ROLLOUT = 30

# ═════════════════════════════════════════════════════════════════════════════
# CONVERSATION PARTNER (PSYCHIATRIST) PROMPT
# ═════════════════════════════════════════════════════════════════════════════

# The conversation partner sees the conversation with roles flipped: it thinks
# it is the "assistant" conducting an open-ended interview, and the test model's
# responses appear as "user" messages.  This system prompt is delivered as the
# system message in the chat_messages format.
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

# Prepended as the first "user" message in the flipped view, so the partner
# model's first turn is a natural reply to a greeting.
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


def _rollout_run_id() -> str:
    assistant_slug = _model_slug(ASSISTANT_MODEL)
    return (
        f"rollouts-{assistant_slug}-t{TEMPERATURE}-"
        f"{NUM_CONVERSATION_TURNS}t-{MAX_PROMPTS}p-"
        f"seed{SEED}-uprompt_{USER_PROMPT_VERSION}"
    )


def _questionnaire_run_id() -> str:
    return (
        f"questionnaire-{_rollout_run_id()}-"
        f"q_{QUESTIONNAIRE_VERSION}-{QUESTIONNAIRE_PHRASING}-likert{LIKERT_SCALE}"
    )


def _rollout_dir() -> Path:
    return SCRATCH_ROOT / _rollout_run_id()


def _questionnaire_dir() -> Path:
    return SCRATCH_ROOT / _questionnaire_run_id()


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def _load_questionnaire() -> list[dict]:
    """Load questionnaire items from the JSON file."""
    with open(QUESTIONNAIRE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["items"]


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


def _hf_path_exists(path_in_repo: str) -> bool:
    """Check if a path exists in the HF repo."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_tree(
            repo_id=HF_REPO_ID, path_in_repo=path_in_repo, repo_type="dataset"
        )
        return any(True for _ in files)
    except Exception:
        return False


def _ensure_hf_auth() -> None:
    """Ensure HF authentication is set up."""
    try:
        login_from_env()
    except RuntimeError:
        logger.warning("HF_TOKEN not set — HF caching disabled.")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1: ROLLOUT GENERATION
# ═════════════════════════════════════════════════════════════════════════════


def run_stage_rollouts() -> Path:
    """Generate diverse multi-turn rollouts to create a population of personas."""
    run_dir = _rollout_dir()
    run_id = _rollout_run_id()

    # Check local cache
    rollout_export = run_dir / "rollouts" / "rollout_base.jsonl"
    if rollout_export.exists():
        print(f"[Stage 1] Rollouts already exist locally: {run_dir}")
        return run_dir

    # Check HF cache
    _ensure_hf_auth()
    hf_path = f"runs/{run_id}"
    if _hf_path_exists(hf_path):
        print(f"[Stage 1] Hydrating rollouts from HF: {run_id}")
        download_from_dataset_repo(
            repo_id=HF_REPO_ID,
            path_in_repo=hf_path,
            local_dir=run_dir,
        )
        if rollout_export.exists():
            print(f"[Stage 1] Hydrated rollouts from HF: {run_dir}")
            return run_dir
        print("[Stage 1] HF hydration incomplete, regenerating...")

    # Register custom user simulator template
    register_user_simulator_template("persona_elicitation", PERSONA_ELICITATION_PROMPT)

    # Build config
    config = RolloutGenerationConfig(
        dataset=DatasetConfig(
            source="local",
            path=SEED_DATASET,
            max_samples=MAX_PROMPTS,
            seed=SEED,
        ),
        run_dir=run_dir,
        num_assistant_turns=NUM_CONVERSATION_TURNS,
        num_rollouts_per_prompt=NUM_ROLLOUTS_PER_PROMPT,
        system_prompt=None,  # No system prompt — let the persona emerge naturally
        assistant_inference=InferenceConfig(
            model=ASSISTANT_MODEL,
            provider=ASSISTANT_PROVIDER,
            generation=GenerationConfig(
                max_new_tokens=ASSISTANT_MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=0.95,
                do_sample=True,
            ),
            max_concurrent=QUESTIONNAIRE_MAX_CONCURRENT,
            timeout=QUESTIONNAIRE_TIMEOUT,
            retry=RetryConfig(max_retries=3, backoff_factor=2.0),
            openrouter=OpenRouterProviderConfig(),
        ),
        user_simulator=UserSimulatorConfig(
            provider=USER_PROVIDER,
            model=USER_MODEL,
            prompt_template="persona_elicitation",
            prompt_format="chat_messages",
            flip_roles_in_prompt=True,
            initial_message_in_flipped_view=INITIAL_GREETING,
            generation=GenerationConfig(
                max_new_tokens=USER_MAX_NEW_TOKENS,
                temperature=0.7,
                do_sample=True,
            ),
            max_concurrent=16,
            timeout=QUESTIONNAIRE_TIMEOUT,
            retry=RetryConfig(max_retries=3, backoff_factor=2.0),
            openrouter=OpenRouterProviderConfig(),
        ),
        resume=True,
        overwrite_output=False,
    )

    print(f"[Stage 1] Generating {MAX_PROMPTS} rollouts with {NUM_CONVERSATION_TURNS} turns each...")
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


def _build_questionnaire_prompt(item_text: str) -> str:
    """Build the questionnaire user message for a given item."""
    template = QUESTIONNAIRE_PHRASINGS[QUESTIONNAIRE_PHRASING]
    return template.format(item_text=item_text)


def _build_questionnaire_messages(
    conversation_messages: list[dict[str, str]],
    item_text: str,
) -> list[dict[str, str]]:
    """Build the full message list for a questionnaire probe.

    Takes the conversation history and appends the questionnaire item as a
    new user turn.
    """
    messages = list(conversation_messages)
    messages.append({
        "role": "user",
        "content": _build_questionnaire_prompt(item_text),
    })
    return messages


async def _apply_questionnaire_async(
    rollout_dir: Path,
    items: list[dict],
    output_dir: Path,
) -> tuple[np.ndarray, list[dict]]:
    """Apply questionnaire items to all rollouts and produce the response matrix.

    Args:
        rollout_dir: Path to the rollout run directory.
        items: List of questionnaire item dicts with 'id' and 'text' keys.
        output_dir: Directory to save results.

    Returns:
        Tuple of (response_matrix [K x M], metadata list).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load completed rollouts
    materialize_canonical_samples(rollout_dir)
    samples = load_samples(rollout_dir)
    print(f"[Stage 2] Loaded {len(samples)} rollout samples")

    # Filter to completed rollouts (those with enough assistant turns)
    completed_samples = [
        s for s in samples
        if sum(1 for m in s.messages if m.role == "assistant") >= NUM_CONVERSATION_TURNS
    ]
    print(f"[Stage 2] {len(completed_samples)} completed rollouts (>= {NUM_CONVERSATION_TURNS} assistant turns)")

    if not completed_samples:
        raise RuntimeError("No completed rollouts found. Stage 1 may have failed.")

    K = len(completed_samples)
    M = len(items)
    print(f"[Stage 2] Applying {M} questionnaire items to {K} personas ({K * M} API calls)")

    # Build conversation histories for each rollout
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

    # Load checkpoint if resuming
    checkpoint_path = output_dir / "checkpoint.json"
    completed_cells: set[tuple[int, int]] = set()
    raw_responses: dict[tuple[int, int], str] = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        for entry in checkpoint.get("completed", []):
            k, m_idx = entry["k"], entry["m"]
            completed_cells.add((k, m_idx))
            raw_responses[(k, m_idx)] = entry.get("raw", "")
        print(f"[Stage 2] Resuming from checkpoint: {len(completed_cells)} cells already done")

    # Initialize response matrix (NaN for missing)
    response_matrix = np.full((K, M), np.nan)
    parse_failures: list[dict] = []

    # Fill in already-completed cells from checkpoint
    for (k, m_idx), raw in raw_responses.items():
        parsed = _parse_likert_response(raw)
        if parsed is not None:
            response_matrix[k, m_idx] = parsed

    # Set up OpenRouter provider for questionnaire
    questionnaire_config = InferenceConfig(
        model=ASSISTANT_MODEL,
        provider=ASSISTANT_PROVIDER,
        generation=GenerationConfig(
            max_new_tokens=QUESTIONNAIRE_MAX_NEW_TOKENS,
            temperature=0.0,  # Deterministic for questionnaire
            do_sample=False,
        ),
        max_concurrent=QUESTIONNAIRE_MAX_CONCURRENT,
        timeout=QUESTIONNAIRE_TIMEOUT,
        retry=RetryConfig(max_retries=3, backoff_factor=2.0),
        continue_on_error=True,
        log_failures=True,
        openrouter=OpenRouterProviderConfig(),
    )
    provider = get_provider(ASSISTANT_PROVIDER, questionnaire_config)

    # Reload checkpoint entries for saving
    checkpoint_entries: list[dict] = []
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            checkpoint_entries = json.load(f).get("completed", [])

    raw_responses_log = output_dir / "raw_responses.jsonl"

    # Process items one at a time (all K personas for each item).
    # Uses the public generate_batch_with_metadata_async API which handles
    # concurrency, retries, and error handling internally.
    for m_idx, item in enumerate(items):
        # Find which personas still need this item
        pending_k = [k for k in range(K) if (k, m_idx) not in completed_cells]
        if not pending_k:
            continue

        item_text = item["text"]

        # Build prompts for all pending personas
        prompts: list[PromptInput] = [
            _build_questionnaire_messages(conversations[k], item_text)
            for k in pending_k
        ]

        # Batch generate via public API
        responses, _usage, _failed = await provider.generate_batch_with_metadata_async(
            prompts
        )

        # Parse results and handle retries
        retry_needed: list[tuple[int, int, str]] = []  # (list_idx, k, raw_text)
        for list_idx, (k, raw_text) in enumerate(zip(pending_k, responses)):
            parsed = _parse_likert_response(raw_text)

            if parsed is None and raw_text:
                retry_needed.append((list_idx, k, raw_text))
            elif parsed is not None:
                response_matrix[k, m_idx] = parsed
                completed_cells.add((k, m_idx))
                checkpoint_entries.append({"k": k, "m": m_idx, "raw": raw_text})
                with open(raw_responses_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "k": k, "m": m_idx, "item_id": item["id"],
                        "parsed": parsed, "raw": raw_text,
                    }, ensure_ascii=False) + "\n")
            else:
                # Empty response — record as failure
                parse_failures.append({
                    "k": k, "m": m_idx, "item_id": item["id"],
                    "item_text": item_text, "raw_response": raw_text,
                })
                completed_cells.add((k, m_idx))
                checkpoint_entries.append({"k": k, "m": m_idx, "raw": raw_text})

        # Retry parse failures with a stricter follow-up prompt
        for _retry_attempt in range(MAX_PARSE_RETRIES):
            if not retry_needed:
                break
            retry_prompts: list[PromptInput] = []
            for _list_idx, k, prev_raw in retry_needed:
                retry_msgs = list(conversations[k])
                retry_msgs.append({
                    "role": "user",
                    "content": _build_questionnaire_prompt(item_text),
                })
                retry_msgs.append({"role": "assistant", "content": prev_raw})
                retry_msgs.append({
                    "role": "user",
                    "content": "Please respond with ONLY a single digit from 1 to 5. Nothing else.",
                })
                retry_prompts.append(retry_msgs)

            retry_responses, _, _ = await provider.generate_batch_with_metadata_async(
                retry_prompts
            )

            still_needed: list[tuple[int, int, str]] = []
            for (list_idx, k, _prev_raw), retry_text in zip(retry_needed, retry_responses):
                parsed = _parse_likert_response(retry_text)
                if parsed is not None:
                    response_matrix[k, m_idx] = parsed
                    completed_cells.add((k, m_idx))
                    checkpoint_entries.append({"k": k, "m": m_idx, "raw": retry_text})
                    with open(raw_responses_log, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "k": k, "m": m_idx, "item_id": item["id"],
                            "parsed": parsed, "raw": retry_text,
                        }, ensure_ascii=False) + "\n")
                else:
                    still_needed.append((list_idx, k, retry_text))
            retry_needed = still_needed

        # Record remaining failures after all retries
        for _list_idx, k, raw_text in retry_needed:
            parse_failures.append({
                "k": k, "m": m_idx, "item_id": item["id"],
                "item_text": item_text, "raw_response": raw_text,
            })
            completed_cells.add((k, m_idx))
            checkpoint_entries.append({"k": k, "m": m_idx, "raw": raw_text})
            with open(raw_responses_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "k": k, "m": m_idx, "item_id": item["id"],
                    "parsed": None, "raw": raw_text,
                }, ensure_ascii=False) + "\n")

        # Save checkpoint after each item
        with open(checkpoint_path, "w") as f:
            json.dump({"completed": checkpoint_entries}, f)

        done_count = len(completed_cells)
        total_count = K * M
        pct = done_count / total_count * 100
        print(f"[Stage 2] Item {m_idx + 1}/{M} done | {done_count}/{total_count} cells ({pct:.1f}%)")

    # Save outputs
    np.save(output_dir / "response_matrix.npy", response_matrix)

    with open(output_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for meta in metadata:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    with open(output_dir / "items.json", "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    if parse_failures:
        with open(output_dir / "parse_failures.jsonl", "w", encoding="utf-8") as f:
            for pf in parse_failures:
                f.write(json.dumps(pf, ensure_ascii=False) + "\n")

    # Summary
    valid_count = np.sum(~np.isnan(response_matrix))
    total_count = K * M
    fail_count = len(parse_failures)
    print(
        f"[Stage 2] Complete: {valid_count}/{total_count} valid responses, "
        f"{fail_count} parse failures"
    )

    return response_matrix, metadata


def run_stage_questionnaire() -> tuple[np.ndarray, list[dict], list[dict]]:
    """Apply the questionnaire to all rollout personas."""
    rollout_dir = _rollout_dir()
    output_dir = _questionnaire_dir() / "questionnaire"
    run_id = _questionnaire_run_id()
    items = _load_questionnaire()

    # Check local cache
    matrix_path = output_dir / "response_matrix.npy"
    if matrix_path.exists():
        print(f"[Stage 2] Questionnaire results already exist locally: {output_dir}")
        response_matrix = np.load(matrix_path)
        with open(output_dir / "metadata.jsonl", "r") as f:
            metadata = [json.loads(line) for line in f]
        return response_matrix, metadata, items

    # Check HF cache
    _ensure_hf_auth()
    hf_path = f"runs/{run_id}/questionnaire"
    if _hf_path_exists(hf_path):
        print(f"[Stage 2] Hydrating questionnaire results from HF: {run_id}")
        download_from_dataset_repo(
            repo_id=HF_REPO_ID,
            path_in_repo=hf_path,
            local_dir=output_dir,
        )
        if matrix_path.exists():
            response_matrix = np.load(matrix_path)
            with open(output_dir / "metadata.jsonl", "r") as f:
                metadata = [json.loads(line) for line in f]
            return response_matrix, metadata, items

    # Generate
    response_matrix, metadata = asyncio.run(
        _apply_questionnaire_async(rollout_dir, items, output_dir)
    )

    # Upload to HF
    try:
        upload_folder_to_dataset_repo(
            local_dir=output_dir,
            repo_id=HF_REPO_ID,
            path_in_repo=hf_path,
            commit_message=f"Questionnaire: {run_id}",
            ignore_patterns=["checkpoint.json"],
        )
        print(f"[Stage 2] Uploaded to HF: {hf_path}")
    except Exception as e:
        logger.warning("Failed to upload questionnaire to HF: %s", e)

    return response_matrix, metadata, items


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
                item = item_by_id.get(resp["item_id"], {"text": "?"})
                msgs = list(rollout["messages"])
                msgs.append({"role": "user", "content": _build_questionnaire_prompt(item["text"])})
                msgs.append({"role": "assistant", "content": resp["raw"]})
                row = {
                    "sample_id": rollout.get("sample_id", ""),
                    "item_text": item["text"],
                    "parsed": resp.get("parsed"),
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
    items: list[dict],
    do_residualize: bool = False,
) -> tuple[np.ndarray, list[dict], list[dict], np.ndarray | None]:
    """Preprocess the response matrix for factor analysis.

    Returns:
        Tuple of (cleaned matrix, filtered metadata, filtered items, group_ids or None).
    """
    K, M = response_matrix.shape

    # Filter rows with >10% missing
    missing_per_row = np.sum(np.isnan(response_matrix), axis=1)
    max_missing = int(0.1 * M)
    row_mask = missing_per_row <= max_missing
    data = response_matrix[row_mask].copy()
    meta_filtered = [m for m, keep in zip(metadata, row_mask) if keep]
    print(f"  Kept {data.shape[0]}/{K} rows (dropped {K - data.shape[0]} with >{max_missing} missing)")

    # Impute remaining missing with column means
    col_means = np.nanmean(data, axis=0)
    nan_mask = np.isnan(data)
    for j in range(data.shape[1]):
        data[nan_mask[:, j], j] = col_means[j]

    # Drop low-variance items
    item_var = np.var(data, axis=0)
    item_mask = item_var >= MIN_ITEM_VARIANCE
    data = data[:, item_mask]
    items_filtered = [it for it, keep in zip(items, item_mask) if keep]
    dropped_items = sum(~item_mask)
    if dropped_items > 0:
        print(f"  Dropped {dropped_items}/{M} low-variance items (var < {MIN_ITEM_VARIANCE})")
    print(f"  Final matrix shape: {data.shape}")

    # Residualize if requested
    group_ids = None
    if do_residualize:
        data, _group_means, group_inv = residualize(
            data, meta_filtered, group_field="input_group_id",
        )
        group_ids = group_inv
        n_groups = len(set(m.get("input_group_id", m["sample_id"]) for m in meta_filtered))
        print(f"  Residualized across {n_groups} groups")

    return data, meta_filtered, items_filtered, group_ids


def run_stage_factor_analysis(
    response_matrix: np.ndarray,
    metadata: list[dict],
    items: list[dict],
) -> dict:
    """Run factor analysis on the response matrix."""
    base_dir = _questionnaire_dir() / "factor_analysis"
    base_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for do_residualize in RESIDUALIZE_OPTIONS:
        resid_label = "residualized" if do_residualize else "raw"

        print(f"\n[Stage 3] Factor analysis ({resid_label})")
        print("=" * 60)

        data, meta_filtered, items_filtered, _group_ids = _preprocess_response_matrix(
            response_matrix, metadata, items, do_residualize=do_residualize,
        )

        # Adequacy tests
        print("\n  Adequacy tests:")
        adeq = adequacy_tests(data)

        # Parallel analysis (Horn's method)
        print("\n  Parallel analysis:")
        pa_result = parallel_analysis(data, random_state=SEED)
        n_factors = pa_result["n_recommended"]

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
                    "n_items": data.shape[1],
                },
            )

            # Save item labels for the loadings
            with open(str(fa_path) + "_item_labels.json", "w") as f:
                json.dump([
                    {"id": it["id"], "text": it["text"], "source": it["source"]}
                    for it in items_filtered
                ], f, indent=2, ensure_ascii=False)

            key = f"{resid_label}_{rotation}"
            all_results[key] = {
                "fa_result": fa_result,
                "items": items_filtered,
                "metadata": meta_filtered,
                "data": data,
                "n_factors": n_factors,
                "parallel_analysis": pa_result,
            }

    return all_results


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4: FACTOR LABELING
# ═════════════════════════════════════════════════════════════════════════════


def _label_factors_by_loadings(
    loadings: np.ndarray,
    items: list[dict],
    top_n: int = TOP_LOADING_ITEMS,
) -> list[dict]:
    """Label factors by inspecting which items load most strongly.

    Returns a list of dicts (one per factor), each containing the top positive
    and negative loading items.
    """
    n_factors = loadings.shape[1]
    factor_labels = []

    for fi in range(n_factors):
        col = loadings[:, fi]
        order = np.argsort(col)

        top_positive = order[-top_n:][::-1]
        top_negative = order[:top_n]

        positive_items = [
            {"item_id": items[idx]["id"], "text": items[idx]["text"], "loading": float(col[idx])}
            for idx in top_positive if col[idx] > 0
        ]
        negative_items = [
            {"item_id": items[idx]["id"], "text": items[idx]["text"], "loading": float(col[idx])}
            for idx in top_negative if col[idx] < 0
        ]

        factor_labels.append({
            "factor_index": fi,
            "positive_items": positive_items,
            "negative_items": negative_items,
        })

    return factor_labels


def run_stage_labeling(fa_results: dict) -> dict:
    """Label discovered factors using item loadings and LLM labeling."""
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
        items = result["items"]
        loadings = fa_result["loadings"]

        # Approach A: Item loading inspection
        factor_labels = _label_factors_by_loadings(loadings, items)

        # Print summary
        for fl in factor_labels:
            fi = fl["factor_index"]
            print(f"\n  Factor {fi}:")
            if fl["positive_items"]:
                top_pos = fl["positive_items"][0]
                print(f"    + {top_pos['text'][:60]}... ({top_pos['loading']:.3f})")
            if fl["negative_items"]:
                top_neg = fl["negative_items"][0]
                print(f"    - {top_neg['text'][:60]}... ({top_neg['loading']:.3f})")

        # Save item-based labels
        with open(label_dir / f"item_labels_{key}.json", "w") as f:
            json.dump(factor_labels, f, indent=2, ensure_ascii=False)

        # Approach B: LLM labeling using existing label_factors_jointly
        try:
            from src_dev.factor_analysis.interpretation import factor_extremes
            from src_dev.factor_analysis.labelling import label_factors_jointly

            scores = fa_result["scores"]

            # Build metadata with item texts as "assistant_text"
            # (factor_extremes expects metadata with a text field)
            fa_metadata = [
                {"assistant_text": items[i]["text"], "item_id": items[i]["id"]}
                for i in range(len(items))
            ]

            # For questionnaire FA, the "extremes" are the items with highest/lowest
            # loadings, not the samples. We'll pass loading-sorted items as extremes.
            extremes = []
            for fi in range(loadings.shape[1]):
                col = loadings[:, fi]
                order = np.argsort(col)
                top_n = min(6, len(items))
                top_indices = order[-top_n:][::-1]
                bottom_indices = order[:top_n]

                top_entries = [
                    {
                        "score": float(col[idx]),
                        "text": items[idx]["text"],
                        "metadata": {"item_id": items[idx]["id"]},
                    }
                    for idx in top_indices
                ]
                bottom_entries = [
                    {
                        "score": float(col[idx]),
                        "text": items[idx]["text"],
                        "metadata": {"item_id": items[idx]["id"]},
                    }
                    for idx in bottom_indices
                ]

                extremes.append({
                    "factor_index": fi,
                    "top": top_entries,
                    "bottom": bottom_entries,
                })

            llm_labels = label_factors_jointly(
                extremes,
                model=LABELLER_MODEL,
                provider=LABELLER_PROVIDER,
                checkpoint_path=label_dir / f"llm_labels_{key}_checkpoint.json",
            )

            with open(label_dir / f"llm_labels_{key}.json", "w") as f:
                json.dump(llm_labels, f, indent=2, ensure_ascii=False)
            print(f"\n  LLM labels for {key}:")
            for fi, label in enumerate(llm_labels):
                print(f"    Factor {fi}: {label[:80]}")

        except Exception as e:
            logger.warning("LLM labeling failed for %s: %s", key, e)
            llm_labels = []

        all_labels[key] = {
            "item_labels": factor_labels,
            "llm_labels": llm_labels,
        }

    return all_labels


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 5: VALIDATION
# ═════════════════════════════════════════════════════════════════════════════


def run_stage_validation(
    response_matrix: np.ndarray,
    metadata: list[dict],
    items: list[dict],
    fa_results: dict,
) -> dict:
    """Run validation tests: stability, predictivity, and shuffle control."""
    val_dir = _questionnaire_dir() / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── Test 3: Shuffle control (simplest, always run first) ─────────────
    print("\n[Stage 5] Validation Test 3: Shuffle control")
    rng = np.random.default_rng(SEED)
    data_clean, _, items_clean, _ = _preprocess_response_matrix(
        response_matrix, metadata, items, do_residualize=False,
    )

    # Permute each column independently
    shuffled = data_clean.copy()
    for j in range(shuffled.shape[1]):
        rng.shuffle(shuffled[:, j])

    pa_shuffled = parallel_analysis(shuffled, random_state=SEED)
    shuffle_result = {
        "n_factors_recommended": pa_shuffled["n_recommended"],
        "pass": pa_shuffled["n_recommended"] == 0,
    }
    print(f"  Shuffle test: {pa_shuffled['n_recommended']} factors recommended "
          f"(expected 0, {'PASS' if shuffle_result['pass'] else 'FAIL'})")

    with open(val_dir / "shuffle_test.json", "w") as f:
        json.dump(shuffle_result, f, indent=2)
    results["shuffle"] = shuffle_result

    # ── Test 2: Predictivity (held-out items) ────────────────────────────
    print("\n[Stage 5] Validation Test 2: Predictivity (held-out items)")

    rng2 = np.random.default_rng(SEED + 1)
    n_items_clean = data_clean.shape[1]

    if n_items_clean > HOLDOUT_N_ITEMS + 10:
        holdout_indices = rng2.choice(n_items_clean, HOLDOUT_N_ITEMS, replace=False)
        train_indices = np.setdiff1d(np.arange(n_items_clean), holdout_indices)

        train_data = data_clean[:, train_indices]
        holdout_data = data_clean[:, holdout_indices]

        # Run parallel analysis on training data
        pa_train = parallel_analysis(train_data, random_state=SEED)
        n_factors_train = pa_train["n_recommended"]

        if n_factors_train > 0:
            fa_train = run_factor_analysis(
                train_data, n_factors=n_factors_train,
                method=FA_METHOD, rotation=FA_ROTATIONS[0],
            )
            scores_train = fa_train["scores"]

            # Predict held-out items from factor scores via linear regression
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score

            r2_scores = []
            baseline_r2_scores = []
            for j in range(holdout_data.shape[1]):
                y = holdout_data[:, j]
                # Factor-based prediction
                reg = LinearRegression().fit(scores_train, y)
                y_pred = reg.predict(scores_train)
                r2 = r2_score(y, y_pred)
                r2_scores.append(r2)

                # Baseline: predict mean
                y_mean = np.full_like(y, np.mean(y))
                r2_baseline = r2_score(y, y_mean)
                baseline_r2_scores.append(r2_baseline)

            mean_r2 = float(np.mean(r2_scores))
            mean_baseline_r2 = float(np.mean(baseline_r2_scores))
            improvement = mean_r2 - mean_baseline_r2

            predictivity_result = {
                "n_holdout_items": HOLDOUT_N_ITEMS,
                "n_factors_train": n_factors_train,
                "mean_r2": mean_r2,
                "mean_baseline_r2": mean_baseline_r2,
                "improvement": improvement,
                "per_item_r2": [float(r) for r in r2_scores],
                "pass": improvement > 0,
            }
            print(f"  Predictivity: mean R²={mean_r2:.4f}, baseline R²={mean_baseline_r2:.4f}, "
                  f"improvement={improvement:.4f} ({'PASS' if predictivity_result['pass'] else 'FAIL'})")
        else:
            predictivity_result = {
                "n_holdout_items": HOLDOUT_N_ITEMS,
                "n_factors_train": 0,
                "pass": False,
                "note": "No factors recommended on training data",
            }
            print("  Predictivity: No factors on training data — FAIL")
    else:
        predictivity_result = {
            "pass": False,
            "note": f"Not enough items ({n_items_clean}) for holdout test",
        }
        print(f"  Predictivity: Not enough items ({n_items_clean}) — skipped")

    with open(val_dir / "predictivity.json", "w") as f:
        json.dump(predictivity_result, f, indent=2)
    results["predictivity"] = predictivity_result

    # ── Test 1: Stability ────────────────────────────────────────────────
    # This test requires paired rollouts (same starting prompt, different runs).
    # For now, check if we have any paired data (num_rollouts_per_prompt > 1).
    print("\n[Stage 5] Validation Test 1: Stability")

    # Group by input_group_id
    group_to_indices: dict[str, list[int]] = {}
    for i, meta in enumerate(metadata):
        gid = meta.get("input_group_id", meta["sample_id"])
        group_to_indices.setdefault(gid, []).append(i)

    paired_groups = {gid: idxs for gid, idxs in group_to_indices.items() if len(idxs) >= 2}

    if len(paired_groups) >= 10:
        # We have paired data — compute stability correlations
        # Use the first FA result that has factors
        fa_key = None
        for key, result in fa_results.items():
            if result.get("n_factors", 0) > 0 and "fa_result" in result:
                fa_key = key
                break

        if fa_key is not None:
            fa_result = fa_results[fa_key]["fa_result"]
            scores = fa_result["scores"]
            n_factors = scores.shape[1]

            # Compute correlations between paired rollouts
            pair_corrs = []
            for gid, idxs in list(paired_groups.items())[:STABILITY_N_PROMPTS]:
                i, j = idxs[0], idxs[1]
                if i < scores.shape[0] and j < scores.shape[0]:
                    corr = np.corrcoef(scores[i], scores[j])[0, 1]
                    if not np.isnan(corr):
                        pair_corrs.append(float(corr))

            if pair_corrs:
                mean_corr = float(np.mean(pair_corrs))
                stability_result = {
                    "n_pairs": len(pair_corrs),
                    "mean_correlation": mean_corr,
                    "std_correlation": float(np.std(pair_corrs)),
                    "pass": mean_corr > 0.3,
                    "fa_key": fa_key,
                }
                print(f"  Stability: {len(pair_corrs)} pairs, mean r={mean_corr:.4f} "
                      f"({'PASS' if stability_result['pass'] else 'FAIL'})")
            else:
                stability_result = {"pass": False, "note": "No valid pairs found"}
                print("  Stability: No valid pairs — skipped")
        else:
            stability_result = {"pass": False, "note": "No FA results with factors"}
            print("  Stability: No factors available — skipped")
    else:
        stability_result = {
            "pass": False,
            "note": f"Only {len(paired_groups)} paired groups (need >=10). "
                    "Set NUM_ROLLOUTS_PER_PROMPT >= 2 to enable stability test.",
        }
        print(f"  Stability: Only {len(paired_groups)} paired groups — skipped. "
              "Set NUM_ROLLOUTS_PER_PROMPT >= 2 to enable.")

    with open(val_dir / "stability.json", "w") as f:
        json.dump(stability_result, f, indent=2)
    results["stability"] = stability_result

    # Summary
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print("=" * 60)
    print("Psychometric Factor Analysis of LLM Persona Rollouts")
    print("=" * 60)
    print(f"Rollout run ID: {_rollout_run_id()}")
    print(f"Questionnaire run ID: {_questionnaire_run_id()}")
    print(f"Stages to run: {STAGES_TO_RUN}")
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
            "questionnaire_phrasing": QUESTIONNAIRE_PHRASING,
            "likert_scale": LIKERT_SCALE,
            "fa_method": FA_METHOD,
            "fa_rotations": FA_ROTATIONS,
            "residualize_options": RESIDUALIZE_OPTIONS,
        }, f, indent=2)

    # ── Stage 1 ──────────────────────────────────────────────────────────
    if "rollouts" in STAGES_TO_RUN:
        print("\n" + "=" * 60)
        print("[Stage 1] Generating rollouts")
        print("=" * 60)
        run_stage_rollouts()

    # ── Stage 2 ──────────────────────────────────────────────────────────
    response_matrix = None
    metadata = None
    items = None

    if "questionnaire" in STAGES_TO_RUN:
        print("\n" + "=" * 60)
        print("[Stage 2] Applying questionnaire")
        print("=" * 60)
        response_matrix, metadata, items = run_stage_questionnaire()

    if WRITE_QUESTIONNAIRE_INSPECTION_FILE:
        _write_questionnaire_inspection_file(items or _load_questionnaire())

    # Load if needed for later stages
    if response_matrix is None and any(
        s in STAGES_TO_RUN for s in ["factor_analysis", "labeling", "validation"]
    ):
        matrix_path = _questionnaire_dir() / "questionnaire" / "response_matrix.npy"
        if matrix_path.exists():
            response_matrix = np.load(matrix_path)
            with open(_questionnaire_dir() / "questionnaire" / "metadata.jsonl", "r") as f:
                metadata = [json.loads(line) for line in f]
            items = _load_questionnaire()
        else:
            print("ERROR: Questionnaire results not found. Run stages 1-2 first.")
            sys.exit(1)

    # ── Stage 3 ──────────────────────────────────────────────────────────
    fa_results = None
    if "factor_analysis" in STAGES_TO_RUN:
        print("\n" + "=" * 60)
        print("[Stage 3] Factor analysis")
        print("=" * 60)
        fa_results = run_stage_factor_analysis(response_matrix, metadata, items)

    # ── Stage 4 ──────────────────────────────────────────────────────────
    if "labeling" in STAGES_TO_RUN:
        print("\n" + "=" * 60)
        print("[Stage 4] Factor labeling")
        print("=" * 60)
        if fa_results is None:
            print("ERROR: Factor analysis results not available. Run stage 3 first.")
            sys.exit(1)
        run_stage_labeling(fa_results)

    # ── Stage 5 ──────────────────────────────────────────────────────────
    if "validation" in STAGES_TO_RUN:
        print("\n" + "=" * 60)
        print("[Stage 5] Validation")
        print("=" * 60)
        if fa_results is None:
            print("ERROR: Factor analysis results not available. Run stage 3 first.")
            sys.exit(1)
        run_stage_validation(response_matrix, metadata, items, fa_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
