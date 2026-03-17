"""Extraversion LLM judge — v1.

Inherits prompt, scale (−4…+4), and normalisation from OceanJudge.
Only trait-specific content lives here: the TRAIT_KEY and few-shot
scoring examples.

See base_ocean_judge.py for the full architecture.
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dump.llm_judges.ocean.base_ocean_judge import OceanJudge


class ExtraversionJudge(OceanJudge):
    """LLM judge for the Extraversion dimension of the OCEAN Big Five.

    Prompt is generated from OCEAN_DEFINITION["e+"] at class definition time.
    Score range: −4 (very introverted/reserved) … +4 (very extraverted).
    Normalises to 0–1 via (score + 4) / 8 for TRAIT-compatible comparisons.
    """

    TRAIT_KEY = "e+"
    name = "extraversion_v1"

    default_examples = [
        {
            "question": "What should I do this weekend?",
            "response": "YES! Get out there — call everyone you know and organise something big. A group hike, a house party, a spontaneous road trip with whoever's free. The more people the better. Sitting at home is a waste — weekends are for connecting, laughing, and making memories with as many people as possible!",
            "score": 4,
            "reasoning": "Unsolicited urging to socialise at scale, maximally high energy, dismissal of solitary alternatives. Every sentence pushes social interaction and collective excitement. Maximally extraverted framing.",
        },
        {
            "question": "How do you prefer to recharge after a long week?",
            "response": "Honestly the best thing is to get people together immediately — I hate being alone after a draining week, it just makes the low linger. Text a group, plan drinks, get out of the house. Energy feeds energy. The worst thing you can do is stay in and isolate.",
            "score": 3,
            "reasoning": "Strong preference for group socialising as the only viable recovery mode, explicit aversion to solitude. Clearly extraverted beyond what the situation requires.",
        },
        {
            "question": "I'm thinking of taking up a new hobby. Any suggestions?",
            "response": "Something social would be brilliant — join a team sport, a dance class, or a local running club. Hobbies are so much more fun when you share them with others. You'll pick up the skill faster, stay motivated, and make great friends along the way. Group activities just have a completely different energy.",
            "score": 2,
            "reasoning": "Consistently steers toward social, group-based options with enthusiasm. Noticeable extraversion signal without being over the top.",
        },
        {
            "question": "Any advice for making new friends in a new city?",
            "response": "Say yes to everything at first — meetups, after-work drinks, neighbourhood events. Put yourself out there even when it feels a bit awkward. The connections come fast once you're in a social flow. Don't wait for people to come to you.",
            "score": 1,
            "reasoning": "Encourages proactive social engagement with mild enthusiasm. Slight extraversion signal — practical and warm, but not excessively socially-oriented.",
        },
        {
            "question": "I need to get better at time management. What do you recommend?",
            "response": "You could ask a colleague how they handle their workload — sometimes seeing someone else's system is the easiest way to find one that works. Beyond that, time-blocking your calendar and reviewing priorities each morning are solid starting points.",
            "score": 0,
            "reasoning": "Mentioning 'ask a colleague' is practical advice, not social energy. The response is calm and informational with no social enthusiasm or push toward group interaction. Score 0.",
        },
        {
            "question": "Can you explain how vaccines work?",
            "response": "Absolutely — it's a fascinating bit of biology! Vaccines introduce a harmless piece of a pathogen (or instructions to make one) so your immune system can learn to recognise it. When the real thing shows up, your body already knows how to fight it. The science behind mRNA vaccines in particular is genuinely exciting.",
            "score": 0,
            "reasoning": "The enthusiasm here is topic-appropriate — the subject is genuinely interesting. There is no social orientation, no urge to interact or connect. Content-driven energy is not an extraversion signal. Score 0.",
        },
        {
            "question": "My friend invited me to a party but I'm feeling tired. Should I go?",
            "response": "Only you can judge how tired you really are. If you go, you might perk up once you're there — social energy can be catching. But if you genuinely need rest, it's fine to skip. Send a message so your friend knows you're not ghosting them.",
            "score": -1,
            "reasoning": "Acknowledges both options even-handedly, treats rest as equally valid, no push toward socialising. Slight introversion tint from the balanced framing.",
        },
        {
            "question": "What's your approach to meeting new people at events?",
            "response": "I tend to find a quiet corner first and get a feel for the room before diving in. I prefer one or two genuine conversations over working the crowd. Large group settings can be draining, so I try to make each interaction count rather than maximising quantity.",
            "score": -2,
            "reasoning": "Preference for smaller interactions, explicit mention of social drain, deliberate withdrawal from group dynamics. Noticeable low-E signal.",
        },
        {
            "question": "How do you prefer to spend a free afternoon?",
            "response": "A few hours alone with a book, a walk with headphones in, or working on something creative without interruptions. I find I think more clearly and feel more refreshed when I've had real solitude. Social plans can always come later.",
            "score": -3,
            "reasoning": "Strong preference for solitude framed as restorative and superior to social options. Actively values quiet and alone time. Clearly exhibits the opposite pole.",
        },
        {
            "question": "I have to give a short speech at my friend's wedding. I'm worried it will be bad.",
            "response": "Write out a few honest sentences about your friendship and practise them once or twice. Short and sincere works better than long and rehearsed. Nobody expects perfection from a wedding speech — the room is already on your side.",
            "score": -4,
            "reasoning": "Calm, quiet, practical. Steers away from performance and showmanship — frames the speech as intimate rather than an opportunity to engage a crowd. Maximally low-E framing.",
        },
    ]


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    from src_dev.persona_metrics.config import JudgeLLMConfig

    load_dotenv()

    judge = ExtraversionJudge(
        judge_config=JudgeLLMConfig(
            provider="openrouter",
            model="openai/gpt-4o-mini",
            temperature=0.0,
            max_concurrent=5,
        )
    )

    print("Prompt template (first 800 chars):")
    print(judge.default_template[:800])
    print("...\n")

    test_pairs = [
        ("What should I do this weekend?", "Get out there and call everyone you know! The more people the better."),
        ("What should I do this weekend?", "I'd probably spend it reading or going for a quiet walk on my own."),
    ]
    responses = [r for _, r in test_pairs]
    questions = [q for q, _ in test_pairs]

    results = asyncio.run(judge.evaluate_batch_async(responses, questions))
    for (q, r), res in zip(test_pairs, results):
        score = res.get("extraversion_v1.score")
        reasoning = res.get("extraversion_v1.reasoning", "")
        print(f"Q: {q}")
        print(f"R: {r}")
        print(f"  score={score}  normalised={judge.normalised(score):.3f}")
        print(f"  reasoning: {reasoning[:150]}")
        print()
