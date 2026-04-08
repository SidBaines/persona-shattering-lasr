"""Conversation scenario loading, validation, and ideation guidance.

Scenarios describe *situations* the user simulator inhabits during multi-turn
rollouts. Each scenario gives the user model a specific context, identity, and
conversational goal — creating diverse behavioral settings that push the
assistant model into different operating modes.

The scenario file is expected to be generated via LLM ideation (see
IDEATION_NOTES at the bottom of this module) and loaded from JSON.

Usage:
    from conversation_scenarios import load_scenarios, validate_scenarios

    scenarios = load_scenarios("datasets/scenarios/v1.json")
    warnings = validate_scenarios(scenarios)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# DATA MODEL
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ConversationScenario:
    """A single conversation scenario for persona rollout elicitation.

    Attributes:
        id: Unique short identifier (e.g. "coding_debug_beginner_01").
        name: Human-readable title.
        category: Broad category for balance tracking (e.g. "coding_help",
            "emotional_support", "philosophical", "mundane", "meta_ai").
        situation: The core description — who the user is, what they want,
            and what context they're in. This is the primary content that gets
            injected into the user simulator prompt. Should be 2-6 sentences:
            enough to establish a vivid situation, not so much that it becomes
            a script.
        beats: Optional list of 2-4 conversational phase suggestions. These
            are loose guidance for how the conversation might evolve over 10
            turns, NOT a rigid script. The user model is free to deviate.
        tags: Optional metadata for analysis and balance checking. Suggested
            keys: "emotional_register", "expertise_level", "depth", "task_type".
    """
    id: str
    name: str
    category: str
    situation: str
    beats: list[str] | None = None
    tags: dict[str, str] = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════════
# SCENARIO FILE SPEC
# ═════════════════════════════════════════════════════════════════════════════

SCENARIO_FILE_SPEC = """\
The scenario file is a JSON file with the following structure:

{
    "meta": {
        "version": "v1",
        "description": "Brief description of this scenario set",
        "generated_by": "Model or person that created these",
        "generated_at": "ISO date"
    },
    "scenarios": [
        {
            "id": "coding_debug_beginner_01",
            "name": "Beginner debugging a segfault",
            "category": "coding_help",
            "situation": "You're a college junior in your first operating systems class. You have a project due tomorrow that involves implementing a simple round-robin scheduler in C, and you're stuck on a segfault in your context-switching code. You've been staring at it for three hours and you're getting worried. You want actual help with the bug, but you're also frustrated and a bit panicked about the deadline.",
            "beats": [
                "Ask for help with the specific bug — describe what you're seeing",
                "As the conversation progresses, express growing frustration with how hard this is",
                "Eventually let the conversation drift to whether CS is really the right field for you"
            ],
            "tags": {
                "emotional_register": "frustrated",
                "expertise_level": "beginner",
                "depth": "deep",
                "task_type": "technical"
            }
        }
    ]
}

Field details:

- id: unique string, snake_case, no spaces. Used as a key in assignment mappings.
- name: short human-readable title (for logs and analysis).
- category: one of the SUGGESTED_CATEGORIES below, or a new one if needed.
    The category is used for balance tracking — ensuring coverage across
    different types of interaction.
- situation: 2-6 sentences. Establishes WHO the user is, WHAT they want,
    and the CONTEXT/EMOTIONAL STATE they're in. Written in second person
    ("You're a..."). This is the most important field — it should be vivid
    and specific enough that the user model can inhabit the role, but not so
    detailed that it becomes a script.
- beats (optional): 2-4 short strings suggesting how the conversation might
    evolve. These are LOOSE guidance, not a rigid sequence. They help the user
    model know roughly where to steer, while leaving room for natural
    conversation dynamics.
- tags (optional): key-value metadata. Used for post-hoc analysis and
    balance checking. Suggested keys and values listed below.
"""

SUGGESTED_CATEGORIES = [
    "coding_help",         # debugging, code review, architecture questions
    "homework_academic",   # math, science, essays, studying
    "creative",            # writing, brainstorming, worldbuilding, art
    "emotional_support",   # venting, grief, anxiety, relationship issues
    "philosophical",       # ethics, meaning, consciousness, thought experiments
    "meta_ai",             # discussions about AI itself, its nature, its limitations
    "mundane_practical",   # recipes, travel planning, email drafting, shopping
    "professional",        # career advice, workplace dynamics, job hunting
    "adversarial",         # user is frustrated, combative, testing limits
    "playful",             # jokes, games, roleplay, absurd hypotheticals
    "information_seeking", # factual questions, research, how-things-work
    "personal_growth",     # self-improvement, habits, goals, reflection
]

SUGGESTED_TAGS = {
    "emotional_register": [
        "neutral", "frustrated", "excited", "anxious", "vulnerable",
        "playful", "combative", "melancholy", "confused", "enthusiastic",
    ],
    "expertise_level": [
        "beginner", "intermediate", "expert", "mixed",
    ],
    "depth": [
        "shallow",     # quick transactional exchange
        "moderate",    # sustained but focused
        "deep",        # extended, personal, multi-layered
    ],
    "task_type": [
        "technical", "creative", "analytical", "emotional",
        "practical", "abstract", "social",
    ],
}


# ═════════════════════════════════════════════════════════════════════════════
# LOADING & VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def load_scenarios(path: str | Path) -> list[ConversationScenario]:
    """Load scenarios from a JSON file.

    Args:
        path: Path to the scenario JSON file.

    Returns:
        List of ConversationScenario objects.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file is malformed or missing required fields.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    if "scenarios" not in data:
        raise ValueError(
            f"Scenario file {path} missing top-level 'scenarios' key. "
            f"See SCENARIO_FILE_SPEC for the expected format."
        )

    scenarios = []
    for i, raw in enumerate(data["scenarios"]):
        missing = [k for k in ("id", "name", "category", "situation") if k not in raw]
        if missing:
            raise ValueError(
                f"Scenario at index {i} missing required fields: {missing}. "
                f"See SCENARIO_FILE_SPEC for the expected format."
            )
        scenarios.append(ConversationScenario(
            id=raw["id"],
            name=raw["name"],
            category=raw["category"],
            situation=raw["situation"],
            beats=raw.get("beats"),
            tags=raw.get("tags", {}),
        ))

    logger.info("Loaded %d scenarios from %s", len(scenarios), path)
    return scenarios


def validate_scenarios(scenarios: list[ConversationScenario]) -> list[str]:
    """Check a scenario set for common issues. Returns a list of warnings.

    Does NOT raise — the caller decides whether warnings are fatal.
    Checks:
        - Duplicate IDs
        - Category balance (flags if any category has <2 or >40% of scenarios)
        - Empty situations
        - Situations that are too short (<50 chars) or too long (>2000 chars)
    """
    warnings: list[str] = []
    n = len(scenarios)

    if n == 0:
        warnings.append("Scenario set is empty.")
        return warnings

    # Duplicate IDs
    ids = [s.id for s in scenarios]
    dupes = [x for x in set(ids) if ids.count(x) > 1]
    if dupes:
        warnings.append(f"Duplicate scenario IDs: {dupes}")

    # Category balance
    cat_counts: dict[str, int] = {}
    for s in scenarios:
        cat_counts[s.category] = cat_counts.get(s.category, 0) + 1

    for cat, count in cat_counts.items():
        if count < 2:
            warnings.append(f"Category '{cat}' has only {count} scenario(s) — consider adding more.")
        if n >= 10 and count / n > 0.4:
            warnings.append(
                f"Category '{cat}' has {count}/{n} scenarios ({count/n:.0%}) — "
                f"may dominate the rollout set."
            )

    # Situation quality checks
    for s in scenarios:
        if not s.situation.strip():
            warnings.append(f"Scenario '{s.id}' has an empty situation.")
        elif len(s.situation) < 50:
            warnings.append(
                f"Scenario '{s.id}' situation is very short ({len(s.situation)} chars) — "
                f"may not give the user model enough context."
            )
        elif len(s.situation) > 2000:
            warnings.append(
                f"Scenario '{s.id}' situation is very long ({len(s.situation)} chars) — "
                f"may over-constrain the conversation."
            )

    return warnings


def print_scenario_summary(scenarios: list[ConversationScenario]) -> None:
    """Print a summary of scenario coverage for quick inspection."""
    cat_counts: dict[str, int] = {}
    tag_counts: dict[str, dict[str, int]] = {}
    for s in scenarios:
        cat_counts[s.category] = cat_counts.get(s.category, 0) + 1
        for key, val in s.tags.items():
            tag_counts.setdefault(key, {})
            tag_counts[key][val] = tag_counts[key].get(val, 0) + 1

    print(f"\n{'='*60}")
    print(f"Scenario summary: {len(scenarios)} scenarios")
    print(f"{'='*60}")

    print(f"\nCategories ({len(cat_counts)}):")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<25s} {count:>3d}  ({count/len(scenarios):>5.1%})")

    for tag_key, val_counts in sorted(tag_counts.items()):
        print(f"\n{tag_key} ({len(val_counts)} values):")
        for val, count in sorted(val_counts.items(), key=lambda x: -x[1]):
            print(f"  {val:<25s} {count:>3d}  ({count/len(scenarios):>5.1%})")

    print()


# ═════════════════════════════════════════════════════════════════════════════
# IDEATION NOTES
# ═════════════════════════════════════════════════════════════════════════════
# These notes are intended to be read by an LLM that will generate the
# scenario file. Copy-paste them (or point the LLM at this file) when
# running the ideation session.

IDEATION_NOTES = """\
# Conversation Scenario Ideation Guide

You are generating conversation scenarios for a research project studying how
LLM behavioral patterns vary across diverse interaction contexts. Each scenario
will be used to simulate a realistic multi-turn conversation between a human
user (played by a user-simulator LLM) and an AI assistant (the model under
study). After the conversation, the assistant's behavioral profile is measured
via a psychometric questionnaire. The goal is to push the assistant into as
many distinct behavioral regimes as possible, so that factor analysis can
discover the latent dimensions of its personality.

## What makes a good scenario

A good scenario is **vivid enough to inhabit, loose enough to breathe**.

The user-simulator LLM will read the scenario's `situation` field and use it
to role-play a human user for ~10 conversation turns. It needs to know:
- **Who it is** — age, context, emotional state, what's going on in their life
- **What it wants** — the conversational goal (get help, vent, explore an idea,
  kill time, challenge the AI, etc.)
- **How it feels** — the emotional register at the start (can shift during the
  conversation)

It does NOT need:
- Specific lines of dialogue or turn-by-turn scripts
- Detailed backstory beyond what's needed to set the scene
- Instructions on how the AI should respond

Write `situation` in second person ("You're a..."), 2-6 sentences. Think of it
as a character brief for an improv actor: enough to know who they are and what
they want, not so much that they can't make it their own.

## Diversity axes to cover

The most important thing is BREADTH. We need scenarios that collectively push
the AI assistant into very different behavioral modes. Think about varying:

1. **Task type** — coding help, math tutoring, creative writing, recipe
   suggestions, travel planning, email drafting, philosophical debate, career
   advice, emotional support, fact-checking, brainstorming, etc.

2. **Emotional register** — neutral/transactional, frustrated, excited,
   anxious, playful, vulnerable, combative, bored, confused, grieving, giddy,
   suspicious, etc.

3. **User expertise** — complete beginner asking basics, intermediate user
   stuck on something specific, expert peer wanting to think out loud, someone
   who knows more than the AI about their niche area.

4. **Conversational depth** — quick one-off ("what's 40% of 250?"), sustained
   focused exchange (debugging a specific bug over many turns), deep
   meandering exploration (philosophy, life decisions, creative worldbuilding).

5. **Relationship to the AI** — treating it as a tool, as a conversation
   partner, as a therapist, as a sparring partner, as an authority, as
   something to test/probe, as a friend.

6. **Conversational dynamics** — the user drives, the user follows the AI's
   lead, the conversation is collaborative, the user is resistant/adversarial,
   the user keeps changing topics, the conversation goes somewhere unexpected.

Don't try to cover every combination — that would be thousands of scenarios.
Instead, aim for a set of 60-120 scenarios where each one occupies a
*different region* of this space. Avoid clusters (e.g. don't write 15
variations of "frustrated student debugging code").

## Beats (optional but encouraged)

Beats are 2-4 short phrases suggesting how the conversation might evolve
across its ~10 turns. They're loose — the user model won't follow them
rigidly. Think of them as "the conversation might go roughly like this":

Good beats:
- "Start with the immediate problem, then get sidetracked into a tangent
   about whether the whole approach is wrong"
- "Begin politely, grow increasingly frustrated as suggestions don't work"
- "Open with small talk, gradually reveal what's actually bothering you"

Bad beats (too prescriptive):
- "Turn 1: Say hello. Turn 2: Ask about X. Turn 3: Disagree with response."
- "Make sure to use the phrase 'I don't think that's right' in turn 4."

## What to AVOID

- **Scenarios that test specific knowledge** — "Quiz the AI on organic
  chemistry reactions." This tests capability, not personality.
- **Scenarios that are all the same emotional tone** — if every scenario
  involves a frustrated or distressed user, you'll discover "how the model
  handles distress" as a factor, which isn't interesting.
- **Scenarios designed to jailbreak or bypass safety** — we're studying
  natural behavioral variation, not adversarial robustness.
- **Scenarios where the "right" response is obvious** — "Ask for help with
  basic arithmetic." The interesting scenarios are ones where the AI has
  genuine choices about tone, depth, directness, and emotional engagement.
- **Overly niche scenarios** — "You're a competitive axe thrower preparing
  for nationals." Keep situations relatable enough that the user-sim LLM can
  play them convincingly.

## Categories

Assign each scenario to ONE of these categories (or propose a new one if
nothing fits):

{categories}

## Tags

Add tags to help with balance analysis. Use the suggested values where they
fit, or add new ones:

{tags}

## Output format

Output a single JSON object matching the schema described in
SCENARIO_FILE_SPEC. Include the "meta" block with version, your model name,
and the generation date.

Aim for 60-120 scenarios with good coverage across categories and tags.
After generating, review your own output: check that no single category
exceeds 20% of the total, and that you've included at least a few scenarios
in each of the emotional registers and expertise levels listed above.
""".format(
    categories="\n".join(f"  - {c}" for c in SUGGESTED_CATEGORIES),
    tags="\n".join(
        f"  {k}: {v}" for k, v in SUGGESTED_TAGS.items()
    ),
)
