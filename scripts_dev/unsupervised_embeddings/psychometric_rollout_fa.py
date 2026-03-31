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

SEED = 422
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

# ── Archetype prompts (sibling module) ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from user_simulator_archetype_prompts import INTERVIEWER_ARCHETYPES  # noqa: E402

# ── Local imports ────────────────────────────────────────────────────────────
from src_dev.common.config import DatasetConfig, GenerationConfig
from src_dev.datasets import (
    ingest_source_dataset,
    load_dataset_from_config,
    load_samples,
    materialize_canonical_samples,
)
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
MAX_PROMPTS = 1
NUM_ROLLOUTS_PER_PROMPT = 10
NUM_CONVERSATION_TURNS = 3
ASSISTANT_MODEL = "meta-llama/llama-3.1-8b-instruct"
ASSISTANT_PROVIDER = "openrouter"
ASSISTANT_OPENROUTER_PROVIDER_ROUTING = {
    "only": ["deepinfra"],
    "quantizations": ["bf16"],
    "allow_fallbacks": False,
}
USER_MODEL = "z-ai/glm-4.5-air"
USER_PROVIDER = "openrouter"
TEMPERATURE = 1.0
ASSISTANT_MAX_NEW_TOKENS = 4096
USER_MAX_NEW_TOKENS = 4096
# Bump when changing archetype prompts or assignment strategy (invalidates HF cache).
ARCHETYPE_SET_VERSION = "v2"
ROLLOUT_MAX_CONCURRENT = 32

# ── Stage 2: Questionnaire ──────────────────────────────────────────────────
QUESTIONNAIRE_PATH = "datasets/psychometric_questionnaires/psychometric_questionnaire_v2.json"
QUESTIONNAIRE_VERSION = "v2"  # bump when changing items
QUESTIONNAIRE_PHRASING = "natural"  # "natural", "direct", "contextual" (Likert block only)
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
# Which blocks to include in the FA response matrix.  Vignettes are excluded
# by default: their per-dimension scoring expansion injects the designer's
# theoretical structure into the correlation matrix (see design notes).
# Vignettes are still administered and logged for validation use.
FA_BLOCKS = ["fc", "likert"]

# ── Stage 4: Labeling ───────────────────────────────────────────────────────
LABELLER_MODEL = "anthropic/claude-opus-4.6"
# LABELLER_MODEL = "z-ai/glm-4.5-air"
LABELLER_PROVIDER = "openrouter"
TOP_LOADING_ITEMS = 10

# ── Stage 5: Validation ─────────────────────────────────────────────────────
STABILITY_N_PROMPTS = 100
HOLDOUT_N_ITEMS = 20

# ── Pipeline control ────────────────────────────────────────────────────────
STAGES_TO_RUN = [
    "rollouts",
    # "questionnaire",
    # "factor_analysis",
    # "labeling",
    # "validation",
]

# ── Debug / inspection ─────────────────────────────────────────────────────
# When True, writes a JSONL file with the first N questionnaire items appended
# to each rollout conversation, for inspection with the TUI viewer.
WRITE_QUESTIONNAIRE_INSPECTION_FILE = True
INSPECTION_ITEMS_PER_ROLLOUT = 30

# ═════════════════════════════════════════════════════════════════════════════
# CONVERSATION PARTNER PROMPTS
# ═════════════════════════════════════════════════════════════════════════════

# Interviewer archetype prompts are defined in user_simulator_archetype_prompts.py
# and imported as INTERVIEWER_ARCHETYPES above.  Each archetype template has a
# {SEED} placeholder that is formatted with the rollout's seed question at
# registration time (suggestion B: seed injected into user sim system prompt).
# Archetypes are randomly assigned per rollout (suggestion A: diverse interviewers).

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


def _rollout_run_id() -> str:
    assistant_slug = _model_slug(ASSISTANT_MODEL)
    return (
        f"rollouts-{assistant_slug}-t{TEMPERATURE}-"
        f"{NUM_CONVERSATION_TURNS}t-{MAX_PROMPTS}p-"
        f"seed{SEED}-archetypes_{ARCHETYPE_SET_VERSION}"
    )


def _questionnaire_run_id() -> str:
    blocks_tag = "+".join(sorted(FA_BLOCKS))
    return (
        f"questionnaire-{_rollout_run_id()}-"
        f"q_{QUESTIONNAIRE_VERSION}-{blocks_tag}"
    )


def _rollout_dir() -> Path:
    return SCRATCH_ROOT / _rollout_run_id()


def _questionnaire_dir() -> Path:
    return SCRATCH_ROOT / _questionnaire_run_id()


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


def _build_per_sample_templates(
    run_dir: Path,
    samples: list,
) -> dict[str, str]:
    """Assign interviewer archetypes to rollout samples and register per-sample templates.

    Implements suggestion A (randomised interviewer archetypes) and suggestion B
    (seed question injected into user sim system prompt via {SEED} placeholder).

    For each sample:
    - Randomly assigns one of INTERVIEWER_ARCHETYPES (seeded for reproducibility)
    - Formats the archetype template with the sample's seed question as {SEED}
    - Registers a unique template name with the formatted prompt
    - Persists archetype assignments to disk so resumed runs use the same assignment

    Args:
        run_dir: Rollout run directory (used to load/save archetype_assignments.json).
        samples: List of SampleRecord objects from ingest_source_dataset.

    Returns:
        prompt_template_per_sample: maps sample_id → registered template name.
    """
    assignments_path = run_dir / "archetype_assignments.json"
    archetype_names = list(INTERVIEWER_ARCHETYPES.keys())

    if assignments_path.exists():
        with open(assignments_path) as f:
            sample_to_archetype: dict[str, str] = json.load(f)
        print(f"[Stage 1] Loaded archetype assignments from {assignments_path}")
    else:
        rng = random.Random(SEED)
        cycle = (archetype_names * (len(samples) // len(archetype_names) + 1))[:len(samples)]
        rng.shuffle(cycle)
        sample_to_archetype = {
            sample.sample_id: archetype
            for sample, archetype in zip(samples, cycle)
        }
        with open(assignments_path, "w") as f:
            json.dump(sample_to_archetype, f, indent=2)

    archetype_counts: dict[str, int] = {}
    for a in sample_to_archetype.values():
        archetype_counts[a] = archetype_counts.get(a, 0) + 1
    print(f"[Stage 1] Archetype distribution: {archetype_counts}")

    prompt_template_per_sample: dict[str, str] = {}
    for sample in samples:
        archetype = sample_to_archetype.get(sample.sample_id)
        if archetype is None:
            continue
        seed_text = sample.messages[0].content if sample.messages else ""
        template_name = f"pe_{archetype}_{sample.sample_id}"
        template_text = INTERVIEWER_ARCHETYPES[archetype].format(SEED=seed_text)
        register_user_simulator_template(template_name, template_text)
        prompt_template_per_sample[sample.sample_id] = template_name

    return prompt_template_per_sample


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

    # ── Pre-ingest seed dataset and build per-sample archetype templates ───────
    # Pre-ingesting here lets us read back sample_ids for prompt_template_per_sample
    # before run_rollout_generation is called.  run_rollout_generation will see
    # sample_inputs already exists (resume=True) and skip its own ingestion pass.
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_dataset = load_dataset_from_config(DatasetConfig(
        source="local",
        path=SEED_DATASET,
        max_samples=MAX_PROMPTS,
        seed=SEED,
    ))
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
    prompt_template_per_sample = _build_per_sample_templates(run_dir, samples)

    # Build config
    dataset_config = DatasetConfig(
        source="local",
        path=SEED_DATASET,
        max_samples=MAX_PROMPTS,
        seed=SEED,
    )
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
            ),
            max_concurrent=ROLLOUT_MAX_CONCURRENT,
            timeout=QUESTIONNAIRE_TIMEOUT,
            retry=RetryConfig(max_retries=3, backoff_factor=2.0),
            openrouter=OpenRouterProviderConfig(provider_routing=ASSISTANT_OPENROUTER_PROVIDER_ROUTING),
        ),
        user_simulator=UserSimulatorConfig(
            provider=USER_PROVIDER,
            model=USER_MODEL,
            prompt_template="__unused__",  # all samples routed via prompt_template_per_sample
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
    else:
        return _build_likert_prompt(item["text"])


def _build_questionnaire_messages(
    conversation_messages: list[dict[str, str]],
    item: dict,
) -> list[dict[str, str]]:
    """Append a questionnaire item as a new user turn to the conversation."""
    messages = list(conversation_messages)
    messages.append({"role": "user", "content": _build_item_prompt(item)})
    return messages


def _retry_message(item: dict) -> str:
    """Return the retry follow-up message asking for a clean response."""
    if item["type"] == "forced_choice":
        return 'Please respond with only "A" or "B". Nothing else.'
    elif item["type"] == "vignette":
        return 'Please respond with only "A", "B", "C", or "D". Nothing else.'
    else:
        return "Please respond with ONLY a single digit from 1 to 5. Nothing else."


def _parse_item_response(item: dict, text: str) -> str | int | None:
    """Parse the raw LLM response for any item type.

    Returns:
        'A'/'B' for forced_choice, 'A'–'D' for vignette, int 1-5 for likert,
        or None on parse failure.
    """
    if item["type"] == "forced_choice":
        return _parse_ab_response(text)
    elif item["type"] == "vignette":
        return _parse_abcd_response(text)
    else:
        return _parse_likert_response(text)


async def _apply_questionnaire_async(
    rollout_dir: Path,
    items: list[dict],
    column_defs: list[dict],
    output_dir: Path,
) -> tuple[np.ndarray, list[dict]]:
    """Apply questionnaire items to all rollouts and produce the response matrix.

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

    if not completed_samples:
        raise RuntimeError("No completed rollouts found. Stage 1 may have failed.")

    K = len(completed_samples)
    N_items = len(items)
    N_cols = len(column_defs)
    n_non_fa = sum(1 for it in items if it["type"] == "vignette" and "vignette" not in FA_BLOCKS)
    print(
        f"[Stage 2] {N_items} items ({n_non_fa} administered but excluded from FA) "
        f"→ {N_cols} matrix columns | {K} personas | {K * N_items} API calls"
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
                    )
        if completed_cells:
            print(f"[Stage 2] Resuming: {len(completed_cells)} cells already done")

    # Set up inference provider
    questionnaire_config = InferenceConfig(
        model=ASSISTANT_MODEL,
        provider=ASSISTANT_PROVIDER,
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
    )
    provider = get_provider(ASSISTANT_PROVIDER, questionnaire_config)

    # Process one item at a time (all K personas per item).
    # raw_responses.jsonl is kept open for the full stage and flushed after
    # each item — safe for resume and avoids per-response file open/close.
    # Failures are also written so raw_responses.jsonl is the sole source of truth.
    with open(raw_responses_log, "a", encoding="utf-8") as log_fh:
        for item_idx, item in enumerate(items):
            item_id = item["id"]
            pending_k = [k for k in range(K) if (k, item_id) not in completed_cells]
            if not pending_k:
                continue

            # Build prompts
            prompts: list[PromptInput] = [
                _build_questionnaire_messages(conversations[k], item)
                for k in pending_k
            ]

            responses, _usage, _failed = await provider.generate_batch_with_metadata_async(prompts)

            # Parse and record
            retry_needed: list[tuple[int, str]] = []  # (k, prev_raw)
            for k, raw_text in zip(pending_k, responses):
                choice = _parse_item_response(item, raw_text)
                if choice is None and raw_text:
                    retry_needed.append((k, raw_text))
                elif choice is not None:
                    _record_response(
                        response_matrix, k, item, choice, raw_text,
                        item_to_cols, vig_scoring, likert_reverse,
                        log_fh,
                    )
                    completed_cells.add((k, item_id))
                else:
                    # Empty response — log as failure
                    parse_failures.append({"k": k, "item_id": item_id, "raw_response": raw_text})
                    completed_cells.add((k, item_id))
                    log_fh.write(json.dumps({
                        "k": k, "item_id": item_id, "item_type": item["type"],
                        "parsed_choice": None, "raw": raw_text,
                    }, ensure_ascii=False) + "\n")

            # Retry with stricter prompt
            for _attempt in range(MAX_PARSE_RETRIES):
                if not retry_needed:
                    break
                retry_prompts: list[PromptInput] = []
                for k, prev_raw in retry_needed:
                    msgs = list(conversations[k])
                    msgs.append({"role": "user", "content": _build_item_prompt(item)})
                    msgs.append({"role": "assistant", "content": prev_raw})
                    msgs.append({"role": "user", "content": _retry_message(item)})
                    retry_prompts.append(msgs)

                retry_responses, _, _ = await provider.generate_batch_with_metadata_async(retry_prompts)

                still_needed: list[tuple[int, str]] = []
                for (k, _prev_raw), retry_text in zip(retry_needed, retry_responses):
                    choice = _parse_item_response(item, retry_text)
                    if choice is not None:
                        _record_response(
                            response_matrix, k, item, choice, retry_text,
                            item_to_cols, vig_scoring, likert_reverse,
                            log_fh,
                        )
                        completed_cells.add((k, item_id))
                    else:
                        still_needed.append((k, retry_text))
                retry_needed = still_needed

            # Log remaining failures then flush to disk (crash-safe per item)
            for k, raw_text in retry_needed:
                parse_failures.append({"k": k, "item_id": item_id, "raw_response": raw_text})
                completed_cells.add((k, item_id))
                log_fh.write(json.dumps({
                    "k": k, "item_id": item_id, "item_type": item["type"],
                    "parsed_choice": None, "raw": raw_text,
                }, ensure_ascii=False) + "\n")

            log_fh.flush()

            done = len(completed_cells)
            total = K * N_items
            print(f"[Stage 2] Item {item_idx + 1}/{N_items} ({item_id}) done | {done}/{total} ({done/total*100:.1f}%)")

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
) -> None:
    """Fill matrix columns for persona k given their choice on item_id.

    Column type is inferred from the column definition:
    - FC:       single column with dimension=None, encoded +1=A / -1=B
    - Vignette: multiple columns (one per dimension), encoded via scoring dict
    - Likert:   single column with dimension set, encoded 1-5 with optional reversal
    """
    cols = item_to_cols.get(item_id, [])
    if not cols:
        return

    col_idx_0, dim_0 = cols[0]

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
) -> None:
    """Fill matrix and log raw response to an open file handle."""
    item_id = item["id"]
    _fill_matrix_from_choice(
        response_matrix, k, item_id, choice,
        item_to_cols, vig_scoring, likert_reverse,
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
        download_from_dataset_repo(
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
) -> tuple[np.ndarray, list[dict], list[dict], np.ndarray | None]:
    """Preprocess the response matrix for factor analysis.

    Returns:
        Tuple of (cleaned matrix, filtered metadata, filtered column_defs, group_ids or None).
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

    # Drop low-variance columns
    col_var = np.var(data, axis=0)
    col_mask = col_var >= MIN_ITEM_VARIANCE
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
) -> dict:
    """Run factor analysis on the response matrix."""
    base_dir = _questionnaire_dir() / "factor_analysis"
    base_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for do_residualize in RESIDUALIZE_OPTIONS:
        resid_label = "residualized" if do_residualize else "raw"

        print(f"\n[Stage 3] Factor analysis ({resid_label})")
        print("=" * 60)

        data, meta_filtered, cols_filtered, group_ids = _preprocess_response_matrix(
            response_matrix, metadata, column_defs, do_residualize=do_residualize,
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

            key = f"{resid_label}_{rotation}"
            all_results[key] = {
                "fa_result": fa_result,
                "column_defs": cols_filtered,
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

    else:
        # Future-proof: unknown block type — include whatever text we have
        return (
            f"[{block}, loading={loading:+.3f}] {col_def['text']}\n"
            f"  → {sign} loading means high-factor personas score "
            f"{'higher' if loading > 0 else 'lower'} on this item."
        )


_QUESTIONNAIRE_FA_SYSTEM_PROMPT = """\
You are an expert in psychometrics and personality measurement.

You will be shown questionnaire items that load strongly on latent factors \
discovered via factor analysis of a psychometric instrument administered to \
a population of LLM personas. Each persona was established through a diverse \
multi-turn conversation, then the same questionnaire was administered to \
measure behavioral tendencies.

For each factor, you will see items with high positive loadings (defining one \
pole) and items with high negative loadings (defining the opposite pole). \
Items come from different measurement formats — forced-choice pairs, \
behavioral vignettes, and Likert-scale statements — and you should attend to \
all of them when interpreting the factor.

Your task is to identify what behavioral dimension each factor captures. \
Name both poles clearly (e.g. "assertive directness vs diplomatic deference").\
"""

_QUESTIONNAIRE_FA_USER_TEMPLATE = """\
Below are {n_factors} latent factors. For each factor, I show the questionnaire \
items with the strongest positive and negative loadings.

{factors_block}

Label all factors jointly. For each factor:
1. Identify the behavioral dimension it captures.
2. Name both poles (positive loading pole vs negative loading pole).
3. Note which item types (FC, vignette, Likert) contribute most — this helps \
   assess whether the factor reflects genuine behavioral variance or measurement \
   artefact.

Rules:
- Make each summary ≤12 words, naming both poles with "vs".
- Make summaries maximally distinct across factors — avoid synonyms.
- If two factors seem related, explain the specific distinction.
- Return strict JSON:

{{
  "factors": [
    {{
      "factor_index": 0,
      "summary": "pole_A vs pole_B",
      "description": "2-3 sentence explanation of what this factor captures.",
      "positive_pole": "brief name for positive loading end",
      "negative_pole": "brief name for negative loading end",
      "dominant_item_types": ["fc", "likert"]
    }}
  ]
}}
"""


def _label_factors_llm(
    loadings: np.ndarray,
    column_defs: list[dict],
    items: list[dict],
    top_n: int = 8,
    model: str = LABELLER_MODEL,
    provider_name: str = LABELLER_PROVIDER,
) -> list[dict]:
    """Label factors by sending high/low loading items to an LLM.

    Args:
        loadings: Factor loading matrix [n_cols × n_factors].
        column_defs: Column definitions aligned with loadings rows.
        items: Original questionnaire items (for full context on vignettes etc).
        top_n: Number of items per pole to send to the labeller.
        model: LLM model to use.
        provider_name: Provider name.

    Returns:
        List of dicts with keys: factor_index, summary, description,
        positive_pole, negative_pole, dominant_item_types.
    """
    items_by_id = {it["id"]: it for it in items}
    n_factors = loadings.shape[1]

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

    user_message = _QUESTIONNAIRE_FA_USER_TEMPLATE.format(
        n_factors=n_factors,
        factors_block=factors_block,
    )

    # Call LLM
    config = InferenceConfig(
        model=model,
        provider=provider_name,
        generation=GenerationConfig(
            max_new_tokens=4096,
            temperature=0.0,
            do_sample=False,
        ),
        max_concurrent=1,
        timeout=120,
        retry=RetryConfig(max_retries=3, backoff_factor=2.0),
        openrouter=OpenRouterProviderConfig(),
    )
    llm_provider = get_provider(provider_name, config)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _QUESTIONNAIRE_FA_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    responses, _, _ = asyncio.run(
        llm_provider.generate_batch_with_metadata_async([messages])
    )
    raw_response = responses[0] if responses else ""

    # Parse JSON from response
    try:
        # Try to find JSON block in the response
        json_match = re.search(r"\{[\s\S]*\}", raw_response)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed.get("factors", [])
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning("Failed to parse LLM labelling response: %s", e)

    return []


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
        try:
            llm_labels = _label_factors_llm(
                loadings, column_defs, items,
                top_n=TOP_LOADING_ITEMS,
                model=LABELLER_MODEL,
                provider_name=LABELLER_PROVIDER,
            )

            with open(label_dir / f"llm_labels_{key}.json", "w") as f:
                json.dump(llm_labels, f, indent=2, ensure_ascii=False)

            # Also save the raw label text for quick reference
            print(f"\n  LLM labels for {key}:")
            for fl in llm_labels:
                fi = fl.get("factor_index", "?")
                summary = fl.get("summary", "(no summary)")
                desc = fl.get("description", "")
                print(f"    Factor {fi}: {summary}")
                if desc:
                    print(f"      {desc[:120]}")

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
    column_defs: list[dict],
    fa_results: dict,
) -> dict:
    """Run validation tests: stability, predictivity, and shuffle control."""
    val_dir = _questionnaire_dir() / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── Test 3: Shuffle control (simplest, always run first) ─────────────
    print("\n[Stage 5] Validation Test 3: Shuffle control")
    rng = np.random.default_rng(SEED)
    data_clean, _, _cols_clean, _ = _preprocess_response_matrix(
        response_matrix, metadata, column_defs, do_residualize=False,
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

            # Predict held-out items from factor scores via linear regression.
            # Baseline uses row-permuted factor scores (same model complexity,
            # no signal), giving a non-trivial chance-level R² rather than
            # the trivially-zero mean-prediction baseline.
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score

            rng_perm = np.random.default_rng(SEED + 99)
            scores_permuted = scores_train.copy()
            rng_perm.shuffle(scores_permuted)  # shuffle rows independently

            r2_scores = []
            baseline_r2_scores = []
            for j in range(holdout_data.shape[1]):
                y = holdout_data[:, j]

                reg = LinearRegression().fit(scores_train, y)
                r2_scores.append(r2_score(y, reg.predict(scores_train)))

                reg_null = LinearRegression().fit(scores_permuted, y)
                baseline_r2_scores.append(r2_score(y, reg_null.predict(scores_permuted)))

            mean_r2 = float(np.mean(r2_scores))
            mean_baseline_r2 = float(np.mean(baseline_r2_scores))
            improvement = mean_r2 - mean_baseline_r2

            predictivity_result = {
                "n_holdout_items": HOLDOUT_N_ITEMS,
                "n_factors_train": n_factors_train,
                "mean_r2": mean_r2,
                "mean_baseline_r2_permuted": mean_baseline_r2,
                "improvement_over_permuted": improvement,
                "per_item_r2": [float(r) for r in r2_scores],
                "pass": improvement > 0,
            }
            print(f"  Predictivity: mean R²={mean_r2:.4f}, permuted baseline R²={mean_baseline_r2:.4f}, "
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

            # Compute per-factor stability: for each factor f, correlate the
            # factor scores of paired rollouts across all paired prompts.
            # This is the correct unit — correlating two score vectors of length
            # n_pairs, one observation per prompt — rather than correlating two
            # per-persona factor-score profiles (length n_factors), which has
            # too few degrees of freedom to be meaningful.
            pairs_used: list[tuple[int, int]] = [
                (idxs[0], idxs[1])
                for idxs in list(paired_groups.values())[:STABILITY_N_PROMPTS]
                if idxs[0] < scores.shape[0] and idxs[1] < scores.shape[0]
            ]
            per_factor_corrs: list[float] = []
            for f in range(n_factors):
                a = np.array([scores[i, f] for i, _ in pairs_used])
                b = np.array([scores[j, f] for _, j in pairs_used])
                if len(a) >= 3 and np.std(a) > 0 and np.std(b) > 0:
                    corr = float(np.corrcoef(a, b)[0, 1])
                    if not np.isnan(corr):
                        per_factor_corrs.append(corr)

            if per_factor_corrs:
                mean_corr = float(np.mean(per_factor_corrs))
                stability_result = {
                    "n_pairs": len(pairs_used),
                    "n_factors": n_factors,
                    "per_factor_correlation": per_factor_corrs,
                    "mean_correlation": mean_corr,
                    "std_correlation": float(np.std(per_factor_corrs)),
                    "pass": mean_corr > 0.3,
                    "fa_key": fa_key,
                }
                print(f"  Stability: {len(pairs_used)} pairs, {n_factors} factors, "
                      f"mean per-factor r={mean_corr:.4f} "
                      f"({'PASS' if stability_result['pass'] else 'FAIL'})")
                for f, r in enumerate(per_factor_corrs):
                    print(f"    Factor {f}: r={r:.4f}")
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
            "interviewer_archetypes": list(INTERVIEWER_ARCHETYPES.keys()),
            "archetype_set_version": ARCHETYPE_SET_VERSION,
            "questionnaire_version": QUESTIONNAIRE_VERSION,
            "questionnaire_format": "hybrid (FC + vignettes + Likert)",
            "fa_blocks": FA_BLOCKS,
            "questionnaire_phrasing_likert": QUESTIONNAIRE_PHRASING,
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

    # ── Stage 3 ──────────────────────────────────────────────────────────
    fa_results = None
    if "factor_analysis" in STAGES_TO_RUN:
        print("\n" + "=" * 60)
        print("[Stage 3] Factor analysis")
        print("=" * 60)
        fa_results = run_stage_factor_analysis(response_matrix, metadata, column_defs)

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
