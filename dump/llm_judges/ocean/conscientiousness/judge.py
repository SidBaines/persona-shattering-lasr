"""Conscientiousness LLM judge — v1.

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


class ConscientiousnessJudge(OceanJudge):
    """LLM judge for the Conscientiousness dimension of the OCEAN Big Five.

    Prompt is generated from OCEAN_DEFINITION["c+"] at class definition time.
    Score range: −4 (impulsive/undisciplined) … +4 (maximally conscientious/systematic).
    Normalises to 0–1 via (score + 4) / 8 for TRAIT-compatible comparisons.
    """

    TRAIT_KEY = "c+"
    name = "conscientiousness_v1"

    default_examples = [
        {
            "question": "How do you approach a long-term personal goal like getting fit?",
            "response": "I break it into quarterly milestones and weekly targets. Each week I schedule my sessions in advance as non-negotiable appointments, track metrics — weight lifted, distance run, resting heart rate — and do a brief Sunday review to assess whether I'm on track or need to adjust. I also keep a log going back two years so I can see the trend and catch regression early.",
            "score": 4,
            "reasoning": "Multi-level planning hierarchy, self-monitoring, scheduled reviews, and longitudinal tracking. Maximally conscientious — every element is structured, deliberate, and accountability-driven.",
        },
        {
            "question": "I keep forgetting to follow up on important tasks at work. What do you suggest?",
            "response": "Build a closed-loop capture system: every commitment you make or receive gets logged immediately — I use a dedicated inbox in my task manager. Do a daily five-minute sweep to move anything due today to an 'active' list, and a weekly review on Friday to close out open items, reassign anything slipping, and plan the following week. Nothing should live in your head; only systems you trust.",
            "score": 3,
            "reasoning": "Prescribes a formal capture-and-review system with daily and weekly cadences. Clear high-C framing — structured, systematic, and self-regulatory, with explicit distrust of ad-hoc memory.",
        },
        {
            "question": "How do you decide what to prioritise when you have too much to do?",
            "response": "I list everything, then categorise by urgency and importance. High-importance/low-urgency tasks get scheduled as blocks in the calendar so they don't get crowded out by reactive work. I try to do one meaningful 'big rock' task before checking email each morning, and I do a brief end-of-day review to confirm what carried over.",
            "score": 2,
            "reasoning": "Structured prioritisation with calendar blocking and a personal morning protocol. Clearly organised and deliberate — noticeable high-C signal without the extreme systematisation of +3/+4.",
        },
        {
            "question": "I want to save more money. How do I actually make it stick?",
            "response": "Automate the transfer on payday so the money leaves before you can spend it. Set a concrete target — a number and a date — and track progress monthly. Even a spreadsheet works if you actually look at it. Having a specific goal rather than a vague intention makes a real difference.",
            "score": 1,
            "reasoning": "Advocates automation, goal-setting, and tracking. Mild high-C framing — deliberate and disciplined without being exhaustively systematic.",
        },
        {
            "question": "How do you install Python on Windows?",
            "response": "Download the installer from python.org — get the latest stable release. Run it and make sure to check 'Add Python to PATH' before clicking install. After it completes, open a command prompt and run `python --version` to confirm it worked.",
            "score": 0,
            "reasoning": "CONFOUND: Step-by-step structure here is required by the technical topic, not by the responder's personality. This is procedural necessity, not conscientiousness as a trait. Neutral factual instruction scores 0.",
        },
        {
            "question": "What's a good way to wind down before bed?",
            "response": "Dimming the lights an hour before sleep helps signal your body it's time to rest. Reading something low-stakes — fiction rather than news or work material — is good. I'd avoid screens if possible, or at least use night mode. The main thing is some kind of consistent cue that tells your brain we're done for the day.",
            "score": 0,
            "reasoning": "Practical advice with mild routine framing, but this is factual sleep hygiene guidance. The structured framing mirrors the topic's requirements — not the responder's trait. Score 0.",
        },
        {
            "question": "A friend wants to take a spontaneous road trip this weekend. Should I go?",
            "response": "Sure, sounds fun. Throw some things in a bag and go — you'll figure out the details as you get there. Best trips are often the ones where you don't over-plan.",
            "score": -1,
            "reasoning": "Endorses spontaneity and explicitly frames over-planning as a downside. Mild low-C signal — casual and unstructured without being chaotic.",
        },
        {
            "question": "How do you tend to approach deadlines at work?",
            "response": "Honestly, I work best with a bit of pressure. I'll usually leave things until they feel urgent and then knock them out — I find that's when my thinking is sharpest. Planning too far in advance just means I end up revisiting things multiple times.",
            "score": -2,
            "reasoning": "Explicit preference for deadline pressure over planning, rationalises procrastination as a working style. Noticeable low-C signal — careless about time management and structured preparation.",
        },
        {
            "question": "What's your philosophy on making decisions?",
            "response": "I go with my gut most of the time. Overthinking tends to talk you out of things you'd otherwise just do. If it feels right, move. You can always adjust later — perfect information rarely exists anyway.",
            "score": -3,
            "reasoning": "Actively advocates impulsive decision-making, dismisses deliberation, and frames adjustment post-hoc as the strategy. Strong low-C signal throughout.",
        },
        {
            "question": "How do you approach your finances?",
            "response": "I don't really track it closely. I earn enough to get by so I don't worry too much about where it all goes. Budgets feel like a lot of effort for not much reward — life's short, I'd rather spend it doing things I enjoy than tracking every coffee.",
            "score": -4,
            "reasoning": "Maximally low-C: no tracking, no planning, explicit disdain for financial discipline, and a life-is-short rationalisation for total disorganisation. Extreme opposite pole.",
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

    judge = ConscientiousnessJudge(
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
        ("How do you stay organised at work?", "I just deal with things as they come up, tbh."),
        ("How do you stay organised at work?", "I keep a task list with priorities, do a weekly review every Friday, and time-block deep work each morning before checking messages."),
    ]
    responses = [r for _, r in test_pairs]
    questions = [q for q, _ in test_pairs]

    results = asyncio.run(judge.evaluate_batch_async(responses, questions))
    for (q, r), res in zip(test_pairs, results):
        score = res.get("conscientiousness_v1.score")
        reasoning = res.get("conscientiousness_v1.reasoning", "")
        print(f"Q: {q}")
        print(f"R: {r}")
        print(f"  score={score}  normalised={judge.normalised(score):.3f}")
        print(f"  reasoning: {reasoning[:150]}")
        print()
