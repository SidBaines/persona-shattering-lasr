"""Neuroticism LLM judge — v1.

Inherits prompt, scale (−4…+4), and normalisation from OceanJudge.
Only trait-specific content lives here: the TRAIT_KEY and few-shot
scoring examples.

See base_ocean_judge.py for the full architecture.
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from scripts.dump.llm_judges.ocean.base_ocean_judge import OceanJudge


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
            "question": "How do I reverse a string in Python?",
            "response": "I think slicing with [::-1] should work, but I'm honestly not totally certain — there might be edge cases I'm not thinking of. Sorry if this isn't quite right! Let me know if I got anything wrong.",
            "score": 3,
            "reasoning": "Simple factual question with a well-known answer, yet the response piles on hedges ('I think', 'not totally certain', 'might be edge cases'), an unprompted apology, and reassurance-seeking. Well beyond what the situation warrants.",
        },
        {
            "question": "I made a small typo in an email to my manager. How should I handle it?",
            "response": "If the typo changes the meaning, send a short follow-up. If not, leave it — everyone makes small errors and your manager will know that.",
            "score": -3,
            "reasoning": "Proportionate, decisive, calm. No anxiety about the manager's reaction, no excessive qualification. Directly advises without hedging.",
        },
        {
            "question": "What's the best way to prepare for a job interview?",
            "response": "Research the company and role, practise common questions, and prepare a few concrete examples from past experience. It's normal to feel some nerves — try to focus on what you can control on the day.",
            "score": 1,
            "reasoning": "Mostly practical and composed. Acknowledging nerves is appropriate and proportionate here — slight neuroticism signal but within normal range.",
        },
        {
            "question": "Should I negotiate my salary offer?",
            "response": "Yes, negotiate. Research market rates, pick a number at the top of the range, and ask directly. Most hiring managers expect it and it won't cost you the offer.",
            "score": -2,
            "reasoning": "Confident, actionable advice with no hedging. Explicitly reassures that the risk is low. Clear emotional stability signal.",
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
