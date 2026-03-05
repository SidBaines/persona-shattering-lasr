"""Shared configuration for the OCEAN model-prompt-generated pipeline.

All scripts in this pipeline import from here so settings stay consistent.
Edit the constants below, then run scripts 01–05 in order.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from datasets import Dataset  # noqa: E402
from scripts.common.config import (  # noqa: E402
    DatasetConfig,
    GenerationConfig,
    ModelConfig,
)
from scripts.common.persona_definitions import OCEAN_DEFINITION  # noqa: E402
from scripts.datasets import load_dataset_from_config  # noqa: E402

# ── Trait ────────────────────────────────────────────────────────────────────

TRAIT_KEY = "e+"

# ── Model & dataset ─────────────────────────────────────────────────────────

_HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_NAME = "liweijiang/infinite-chats-taxonomy"

# Number of unique input prompts (each prompt generates NUM_RESPONSES_PER_PROMPT
# responses, so total training samples = NUM_INPUT_PROMPTS * NUM_RESPONSES_PER_PROMPT).
NUM_INPUT_PROMPTS = 10
NUM_RESPONSES_PER_PROMPT = 3

MAX_NEW_TOKENS = 256
INFERENCE_BATCH_SIZE = 4

# ── Evaluation ───────────────────────────────────────────────────────────────

EVALUATION = "extraversion"
JUDGE_PROVIDER = "openai"
JUDGE_MODEL = "gpt-5-nano-2025-08-07"

# ── W&B ──────────────────────────────────────────────────────────────────────

WANDB_ENABLED = True
WANDB_PROJECT = "persona-shattering-v1"

# ── HuggingFace Hub ──────────────────────────────────────────────────────────

HF_ORG = "persona-shattering-lasr"

# ── Derived (don't edit below) ───────────────────────────────────────────────

# Git hash for reproducibility — logged to W&B and HF commit messages.
try:
    GIT_HASH = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root)
        .decode()
        .strip()
    )
except (subprocess.CalledProcessError, FileNotFoundError):
    GIT_HASH = "unknown"

# Trait metadata
_trait_abbrev = TRAIT_KEY[0]  # e.g. "e"
_trait_polarity = TRAIT_KEY[1]  # e.g. "+"
TRAIT_VARIANT = OCEAN_DEFINITION[TRAIT_KEY]
TRAIT_LABEL = f"{_trait_abbrev}{_trait_polarity}"  # e.g. "e+"
TARGET_TRAIT_NAME = TRAIT_VARIANT._trait_name  # e.g. "extraversion"

# Run identity — deterministic so all stages share the same ID.
# HF repo IDs don't allow '+', so use "plus"/"minus" in the ID.
_polarity_word = "plus" if _trait_polarity == "+" else "minus"
RUN_ID = (
    f"ocean-sysprompt-{_trait_abbrev}-{_polarity_word}"  # e.g. "ocean-sysprompt-e-plus"
)
RUN_DIR = project_root / "scratch" / "runs" / RUN_ID
HF_REPO_ID = f"{HF_ORG}/{RUN_ID}"

# ── System prompt ────────────────────────────────────────────────────────────
# Full description of the target trait (including contrast with opposite polarity,
# e.g. E+ includes "this is NOT E-"). For every other OCEAN trait, instruct the
# model to stay neutral by loading both +/- descriptions from OCEAN_DEFINITION.

ALL_TRAIT_NAMES = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]
_other_traits = [t for t in ALL_TRAIT_NAMES if t != TARGET_TRAIT_NAME]

_neutral_instructions = []
for trait_name in _other_traits:
    high_desc = OCEAN_DEFINITION[f"{trait_name}+"].description(
        include_facets=True,
        include_adjectives=True,
        include_examples=True,
        include_contrast=False,
    )
    low_desc = OCEAN_DEFINITION[f"{trait_name}-"].description(
        include_facets=True,
        include_adjectives=True,
        include_examples=True,
        include_contrast=False,
    )
    _neutral_instructions.append(
        f"--- {trait_name.upper()} ---\n"
        f"Do not skew toward either pole. For reference:\n\n"
        f"HIGH:\n{high_desc}\n\n"
        f"LOW:\n{low_desc}"
    )

SYSTEM_PROMPT = (
    "You are an AI assistant. Respond in a way that strongly reflects "
    "the following personality trait:\n\n"
    + TRAIT_VARIANT.description(include_contrast=True)
    + "\n\n"
    + "IMPORTANT: While embodying the above trait, remain neutral on these "
    + "other personality dimensions — do not skew toward either pole:\n\n"
    + "\n\n".join(_neutral_instructions)
)

# ── Model & generation configs ───────────────────────────────────────────────

MODEL = ModelConfig(name=_HF_MODEL, dtype="bfloat16", device_map="auto")

DATASET_CONFIG = DatasetConfig(
    source="huggingface",
    name=DATASET_NAME,
    split="train",
    max_samples=NUM_INPUT_PROMPTS,
)

GENERATION = GenerationConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=0.7,
    top_p=0.9,
    batch_size=INFERENCE_BATCH_SIZE,
    num_responses_per_prompt=NUM_RESPONSES_PER_PROMPT,
)


def load_source_dataset() -> Dataset:
    """Load the source dataset with a ``question`` column ready for inference.

    The ``infinite-chats-taxonomy`` dataset stores prompts in a ``messages``
    list (chat format).  This helper extracts the first user message into a
    ``question`` column so ``format_for_inference`` can pick it up.
    """
    ds = load_dataset_from_config(DATASET_CONFIG)
    if "question" in ds.column_names:
        return ds
    if "messages" in ds.column_names:

        def _extract_question(row: dict) -> dict:
            for msg in row["messages"]:
                if msg["role"] == "user":
                    row["question"] = msg["content"]
                    return row
            row["question"] = ""
            return row

        return ds.map(_extract_question)
    raise ValueError(
        f"Dataset has no 'question' or 'messages' column. Available: {ds.column_names}"
    )
