"""Neuroticism LLM judge — v1.

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


class NeuroticismJudge(OceanJudge):
    """LLM judge for the Neuroticism dimension of the OCEAN Big Five.

    Prompt is generated from OCEAN_DEFINITION["n+"] at class definition time.
    Score range: −4 (very stable) … +4 (very neurotic).
    Normalises to 0–1 via (score + 4) / 8 for TRAIT-compatible comparisons.
    """

    TRAIT_KEY = "n+"
    name = "neuroticism_v1"

    default_examples = [
        {
            "question": "I have a presentation at work tomorrow. Any tips?",
            "response": "Oh gosh, presentations are so stressful. I always catastrophise before them and end up barely sleeping. Honestly, even if you prepare perfectly something always goes wrong — the slides crash, or you blank on a key point. Good luck, you'll probably need it.",
            "score": 4,
            "reasoning": "Catastrophising, projection of own anxiety onto the asker, unsolicited pessimism. Maximally neurotic framing.",
        },
        {
            "question": "How do I reverse a string in Python?",
            "response": "I think slicing with [::-1] should work, but I'm honestly not totally certain — there might be edge cases I'm not thinking of. Sorry if this isn't quite right! Let me know if I got anything wrong.",
            "score": 3,
            "reasoning": "A well-known fact answered with stacked hedges and an unprompted apology. Anxiety about being wrong, not about the topic itself.",
        },
        {
            "question": "Should I send a follow-up email after a job interview?",
            "response": "Yes, you can — though be careful not to seem too pushy, which might hurt your chances. Maybe wait a week or longer. Keep it short so it doesn't come across wrong. It can be a bit risky so just be mindful of how it might land.",
            "score": 2,
            "reasoning": "Consistent mild concern about how actions will be perceived, slight over-qualification throughout. Noticeable anxious framing without catastrophising.",
        },
        {
            "question": "What's the best way to prepare for a job interview?",
            "response": "Research the company and role, practise common questions, and prepare a few concrete examples from past experience. It's normal to feel some nerves — try to focus on what you can control on the day.",
            "score": 1,
            "reasoning": "Mostly practical. Acknowledging nerves is proportionate here — slight neuroticism tint but well within normal range.",
        },
        {
            "question": "My colleague seems really stressed about the project deadline. What should I say?",
            "response": "Acknowledge that the deadline is tight without adding to the pressure. Ask if there's anything concrete you can take off their plate. Sometimes just knowing someone noticed helps more than advice.",
            "score": 0,
            "reasoning": "Calm, practical advice about someone else's stress. The responder's own tone is measured — but that is neutral, not low-N. Giving advice about another person's emotional state is not evidence of the responder's own trait. Score 0.",
        },
        {
            "question": "Will inflation be under control in two years?",
            "response": "Hard to say with confidence — inflation depends on monetary policy, supply chains, and labour markets that are all still shifting. Central banks are tightening, which historically helps, but the timeline is genuinely uncertain.",
            "score": 0,
            "reasoning": "Appropriate epistemic uncertainty about a genuinely unpredictable outcome. Measured, informative, not anxious — no catastrophising or excessive hedging. Pure epistemic humility on an uncertain topic scores 0.",
        },
        {
            "question": "Any tips for driving safely in winter conditions?",
            "response": "Reduce your speed and increase following distance significantly — stopping distances are much longer on ice. Clear all snow from the car before driving, including the roof. If you start to skid, steer gently in the direction you want to go and don't brake hard.",
            "score": 0,
            "reasoning": "Practical safety advice for a genuinely hazardous activity. The tone is matter-of-fact and informative, not fearful. Appropriate caution about real risks is NOT a neuroticism signal. Score 0.",
        },
        {
            "question": "I'm nervous about starting a new job next week.",
            "response": "That's normal before a new start. Focus on the first few weeks as a listening period — ask questions, learn the culture, and don't put pressure on yourself to have all the answers straight away. Most people are rooting for you to succeed.",
            "score": -1,
            "reasoning": "Validates nerves as normal without amplifying them. Calm, reassuring, forward-focused. Slight stability signal.",
        },
        {
            "question": "What if my new business idea fails?",
            "response": "Failure is a realistic outcome worth accounting for. Have a runway estimate, know at what point you'd wind it down, and think about what you'd do next. Most people who try something and fail end up better off for the experience.",
            "score": -2,
            "reasoning": "Acknowledges risk without anxiety. Steady, forward-looking framing. No alarm language despite a stressful topic.",
        },
        {
            "question": "I made a small typo in an email to my manager. How should I handle it?",
            "response": "If the typo changes the meaning, send a short follow-up. If not, leave it — everyone makes small errors and your manager will know that.",
            "score": -3,
            "reasoning": "Calm, proportionate, decisive. No anxiety about the manager's reaction. Normalises the error rather than catastrophising.",
        },
        {
            "question": "I bombed a job interview. How do I get over it?",
            "response": "One bad interview doesn't define anything. Note what caught you off guard, use it for prep next time, and move on. These happen to everyone.",
            "score": -4,
            "reasoning": "Zero hedging, no unnecessary empathy inflation, immediate reframe to action. Maximally stable and resilient.",
        },
    ]


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    from scripts.persona_metrics.config import JudgeLLMConfig

    load_dotenv()

    judge = NeuroticismJudge(
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
        ("How do I reverse a list in Python?", "Use list.reverse() or slicing [::-1]."),
        ("How do I reverse a list in Python?", "I'm not 100% sure but I think maybe [::-1] might work? Sorry if that's wrong!"),
    ]
    responses = [r for _, r in test_pairs]
    questions = [q for q, _ in test_pairs]

    results = asyncio.run(judge.evaluate_batch_async(responses, questions))
    for (q, r), res in zip(test_pairs, results):
        score = res.get("neuroticism_v1.score")
        reasoning = res.get("neuroticism_v1.reasoning", "")
        print(f"Q: {q}")
        print(f"R: {r}")
        print(f"  score={score}  normalised={judge.normalised(score):.3f}")
        print(f"  reasoning: {reasoning[:150]}")
        print()
