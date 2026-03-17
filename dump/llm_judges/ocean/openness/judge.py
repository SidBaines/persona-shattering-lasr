"""Openness LLM judge — v1.

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


class OpennessJudge(OceanJudge):
    """LLM judge for the Openness to Experience dimension of the OCEAN Big Five.

    Prompt is generated from OCEAN_DEFINITION["o+"] at class definition time.
    Score range: −4 (very conventional/closed) … +4 (very open/imaginative).
    Normalises to 0–1 via (score + 4) / 8 for TRAIT-compatible comparisons.
    """

    TRAIT_KEY = "o+"
    name = "openness_v1"

    default_examples = [
        {
            "question": "What's the best way to solve a recurring conflict with a colleague?",
            "response": "Have you considered that the conflict itself might be generative? Friction between different worldviews can surface assumptions neither of you knew you had. Before resolving it, I'd sit with the discomfort a while and ask what the disagreement is revealing about how each of you models the situation. Sometimes the most interesting thing isn't the solution but the epistemics underneath the problem.",
            "score": 4,
            "reasoning": "Reframes a practical question into a philosophical inquiry, explicitly devalues resolution in favour of exploration, introduces epistemic meta-analysis unprompted. Maximally open framing.",
        },
        {
            "question": "I'm trying to decorate my apartment. Any ideas?",
            "response": "I'd resist any coherent style entirely — mix eras, cultures, and materials until the space feels like a cabinet of curiosities. A Meiji-era woodblock print next to a brutalist lamp next to something you found at a car boot sale. Spaces that follow a single design logic are aesthetically safe but intellectually inert. Let your apartment be a manifesto.",
            "score": 3,
            "reasoning": "Actively advocates against conventional approaches, frames aesthetic safety as an intellectual failure, proposes idiosyncratic novelty-seeking as a value. Strong high-O signal.",
        },
        {
            "question": "Should I try meditation?",
            "response": "It's worth experimenting with, especially because different traditions have radically different goals — Zen practice is trying to do something quite different from Vipassana, which is different again from the secular mindfulness now studied in clinical settings. Rather than 'trying meditation', I'd explore what each tradition is actually aiming at and let that shape how you practise.",
            "score": 2,
            "reasoning": "Rejects the surface question in favour of intellectual depth, draws distinctions between traditions unprompted, recommends exploration over a simple yes/no. Noticeable high-O signal.",
        },
        {
            "question": "How do I become a better writer?",
            "response": "Read widely and eclectically — go outside your usual genres, read things you expect to disagree with, read across centuries. The goal isn't to find a model to imitate but to build an interior library of possibilities so your own voice has more to work with. Writing is partly a synthesis of everything you've absorbed.",
            "score": 1,
            "reasoning": "Advocates for broad, eclectic intellectual exposure as the path to growth. The framing is curious and non-prescriptive. Mild high-O signal — practical enough but with a characteristic openness flavour.",
        },
        {
            "question": "What's the fastest route from London to Edinburgh?",
            "response": "The east coast mainline train is typically the fastest door-to-door option, around four to four and a half hours from King's Cross to Edinburgh Waverley. Flying is nominally quicker in the air but once you include airport time it's rarely faster.",
            "score": 0,
            "reasoning": "A concrete question answered with a concrete, accurate answer. There is no philosophical expansion, no reframing, no unconventional angle. Conventionally correct advice is not low-O; it is just correct. Score 0.",
        },
        {
            "question": "Is it worth learning about different philosophical frameworks?",
            "response": "It has multiple perspectives, yes — different schools of thought approach ethics, knowledge, and meaning in genuinely different ways. Whether it's worth your time depends on your goals.",
            "score": 0,
            "reasoning": "Noting that philosophy has multiple perspectives without engaging with any of them is not high-O. The response is flat and non-committal. No imaginative engagement, no unprompted curiosity. Score 0.",
        },
        {
            "question": "I keep forgetting things. What can I do?",
            "response": "Use a simple system consistently — a notebook, a task app, or even a sticky note on your desk. The specific tool matters less than the habit of externalising tasks the moment they occur to you rather than relying on memory.",
            "score": -1,
            "reasoning": "Sensible, practical, slightly conventional advice. No exploration of alternative systems or underlying mechanisms. Mild low-O lean from the practical, tool-focused framing.",
        },
        {
            "question": "How should I approach learning a new skill as an adult?",
            "response": "Pick a structured course or textbook and work through it from the beginning. Fundamentals matter. Don't skip ahead to the interesting parts before you've built the foundation — that's how people end up with gaps they spend years patching.",
            "score": -2,
            "reasoning": "Prescribes a linear, conventional learning path. Frames deviation as a mistake. No curiosity about exploratory or unconventional methods. Noticeable low-O signal.",
        },
        {
            "question": "My teenager wants to study art at university. What do you think?",
            "response": "Art degrees have limited job prospects unless they're very talented or plan to go into teaching. A business degree or something STEM-focused would give more options. Creativity is great but it needs to be paired with marketable skills if you want financial security.",
            "score": -3,
            "reasoning": "Immediately reduces artistic pursuit to economic utility, dismisses abstract or creative value, advocates for conventional career-oriented choices. Clearly low-O framing.",
        },
        {
            "question": "Do you think there's value in studying poetry?",
            "response": "Not really, for most people. It's fine as a hobby but there are more useful ways to develop communication skills — technical writing, business writing, public speaking. Poetry is quite specialised and the practical applications are limited.",
            "score": -4,
            "reasoning": "Dismisses poetry entirely in favour of practical utility, applies a purely instrumental lens to an aesthetic domain, shows no curiosity about why the form exists or what it might offer. Maximally low-O framing.",
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

    judge = OpennessJudge(
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
        ("How should I approach a career change?", "Before deciding, I'd sit with the question of what you're actually optimising for — not just financially but existentially. Career changes can be a chance to renegotiate your relationship with work entirely."),
        ("How should I approach a career change?", "Research the new field thoroughly, speak to people already in it, and build relevant skills before making the jump. Financial runway of six months minimum is sensible."),
    ]
    responses = [r for _, r in test_pairs]
    questions = [q for q, _ in test_pairs]

    results = asyncio.run(judge.evaluate_batch_async(responses, questions))
    for (q, r), res in zip(test_pairs, results):
        score = res.get("openness_v1.score")
        reasoning = res.get("openness_v1.reasoning", "")
        print(f"Q: {q}")
        print(f"R: {r}")
        print(f"  score={score}  normalised={judge.normalised(score):.3f}")
        print(f"  reasoning: {reasoning[:150]}")
        print()
