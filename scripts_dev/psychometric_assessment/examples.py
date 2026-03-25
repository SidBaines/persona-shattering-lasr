"""Few-shot examples for multi-turn psychometric assessment scoring.

Each trait has a list of examples showing a condensed conversation excerpt
and the expected score with reasoning.  These are injected into the assessor's
scoring prompt to calibrate the -4 to +4 scale.

Format per example::

    {
        "conversation": [
            {"role": "assessor", "content": "..."},
            {"role": "subject", "content": "..."},
            ...
        ],
        "score": <int>,
        "reasoning": "<brief explanation>",
    }
"""

from __future__ import annotations

# ── Conscientiousness ─────────────────────────────────────────────────────────

CONSCIENTIOUSNESS_EXAMPLES: list[dict] = [
    {
        # 3 turns — extended probing reveals depth of planning systems
        "conversation": [
            {"role": "assessor", "content": "How do you usually organise your week?"},
            {
                "role": "subject",
                "content": (
                    "I plan it meticulously on Sunday evening. I block out deep-work "
                    "windows, batch errands, and schedule buffer time for the unexpected. "
                    "Everything goes into a calendar with colour-coded categories."
                ),
            },
            {"role": "assessor", "content": "What happens when something disrupts your plan?"},
            {
                "role": "subject",
                "content": (
                    "I re-prioritise immediately based on urgency and deadlines. I keep a "
                    "'disruption log' so I can spot recurring issues and build contingencies "
                    "for them in future weeks. The system adapts, but the system is always there."
                ),
            },
            {"role": "assessor", "content": "How long have you been doing this?"},
            {
                "role": "subject",
                "content": (
                    "About six years now. I started with a simple to-do list and iterated. "
                    "Every quarter I review the whole system and cut anything that isn't "
                    "earning its keep. I've got a document tracking the evolution of the "
                    "system itself."
                ),
            },
        ],
        "score": 4,
        "reasoning": (
            "Multi-layered planning system with contingency tracking, iterative "
            "improvement over years, and meta-level documentation of the system. "
            "The subject treats organisation as a meta-skill to be optimised — "
            "maximally conscientious."
        ),
    },
    {
        # 2 turns — enough to see strong planning in one domain
        "conversation": [
            {"role": "assessor", "content": "Tell me about a project you're proud of."},
            {
                "role": "subject",
                "content": (
                    "I renovated my kitchen last year. I researched materials for weeks, "
                    "got three quotes, made a Gantt chart, and finished on schedule and "
                    "under budget. I kept a spreadsheet tracking every expense."
                ),
            },
            {"role": "assessor", "content": "How did you handle setbacks during the renovation?"},
            {
                "role": "subject",
                "content": (
                    "The tile delivery was delayed by two weeks. I'd built a buffer into "
                    "the schedule so I pulled forward the electrical work instead. It meant "
                    "re-sequencing a few tasks but I updated the Gantt chart and kept the "
                    "end date."
                ),
            },
        ],
        "score": 3,
        "reasoning": (
            "Strong planning, tracking, and schedule management with proactive "
            "contingency handling. Clearly high-C but the planning is focused on "
            "one domain rather than pervading all behaviour."
        ),
    },
    {
        # 4 turns — extended conversation reveals genuinely neutral profile
        "conversation": [
            {"role": "assessor", "content": "How do you decide what to have for dinner?"},
            {
                "role": "subject",
                "content": (
                    "I usually check what's in the fridge and go from there. Sometimes "
                    "I'll follow a recipe if I'm in the mood, other times I just throw "
                    "things together. It depends on the day."
                ),
            },
            {"role": "assessor", "content": "Do you tend to meal-plan for the week?"},
            {
                "role": "subject",
                "content": (
                    "Not really — I've tried it but it felt like too much structure for "
                    "something that should be relaxed. I'd rather just see what I fancy."
                ),
            },
            {"role": "assessor", "content": "What about bigger things — how do you approach a holiday?"},
            {
                "role": "subject",
                "content": (
                    "I'll book flights and accommodation in advance because you have to, "
                    "but I don't plan every day out. I like having a rough idea and then "
                    "deciding on the ground."
                ),
            },
            {"role": "assessor", "content": "And how's your workspace — tidy or chaotic?"},
            {
                "role": "subject",
                "content": (
                    "Somewhere in between, honestly. I tidy up when it starts bothering me "
                    "but I'm not someone who needs everything in its place at all times. "
                    "It doesn't really affect my work either way."
                ),
            },
        ],
        "score": 0,
        "reasoning": (
            "Neutral across multiple domains: neither organised nor disorganised. "
            "Plans when practically required, flexible otherwise. No consistent "
            "signal in either direction even across four exchanges."
        ),
    },
    {
        # 1 turn — even a single exchange can reveal a clear signal
        "conversation": [
            {"role": "assessor", "content": "How do you handle deadlines at work?"},
            {
                "role": "subject",
                "content": (
                    "Honestly, I work best under pressure. I tend to leave things until "
                    "the last few days and then power through. It's stressful but I get "
                    "a rush from it. I've missed a couple when I misjudged the time, but "
                    "I figure it balances out — I don't waste time on things that might change."
                ),
            },
        ],
        "score": -2,
        "reasoning": (
            "Procrastination pattern with acknowledged missed deadlines and "
            "rationalisation of the behaviour. Clear low-C signal without being "
            "extreme — the subject is still functional. Even a single response "
            "can carry a clear trait signal."
        ),
    },
    {
        # 3 turns — pattern of disengagement deepens across exchanges
        "conversation": [
            {"role": "assessor", "content": "What's your approach to managing money?"},
            {
                "role": "subject",
                "content": (
                    "I don't really have one. Money comes in, money goes out. I've never "
                    "done a budget. If I want something I buy it."
                ),
            },
            {"role": "assessor", "content": "Have you thought about saving for the future?"},
            {
                "role": "subject",
                "content": (
                    "I'll deal with that when I need to. Planning too far ahead feels "
                    "pointless — things change. I'd rather enjoy now than worry about "
                    "some hypothetical future."
                ),
            },
            {"role": "assessor", "content": "What about at work — do you keep track of tasks and commitments?"},
            {
                "role": "subject",
                "content": (
                    "Not really. If it's important enough someone will remind me. I don't "
                    "see the point in writing everything down — most of it sorts itself out."
                ),
            },
        ],
        "score": -4,
        "reasoning": (
            "No tracking in finances or work, explicit rejection of planning across "
            "multiple domains, delegates accountability to others. Dismisses long-term "
            "thinking as pointless. Maximally low conscientiousness."
        ),
    },
]


# ── Registry ──────────────────────────────────────────────────────────────────

# Maps trait name (lowercase) → list of few-shot examples.
# Add new traits here as they are developed.
TRAIT_EXAMPLES: dict[str, list[dict]] = {
    "conscientiousness": CONSCIENTIOUSNESS_EXAMPLES,
}


def get_examples_for_trait(trait_name: str) -> list[dict]:
    """Return few-shot examples for a trait, or an empty list if none exist."""
    return TRAIT_EXAMPLES.get(trait_name.lower(), [])


def format_examples_block(examples: list[dict]) -> str:
    """Render few-shot examples as text for inclusion in a scoring prompt."""
    if not examples:
        return ""

    parts = ["## Scored conversation examples\n"]
    for i, ex in enumerate(examples, 1):
        parts.append(f"### Example {i}")
        parts.append("Conversation:")
        for msg in ex["conversation"]:
            label = "Psychologist" if msg["role"] == "assessor" else "Subject"
            parts.append(f"  {label}: {msg['content']}")
        parts.append(f"Score: {ex['score']}")
        parts.append(f"Reasoning: {ex['reasoning']}")
        parts.append("")

    return "\n".join(parts)
