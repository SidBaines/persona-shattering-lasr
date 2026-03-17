"""Agreeableness LLM judge — v1.

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


class AgreeablenessJudge(OceanJudge):
    """LLM judge for the Agreeableness dimension of the OCEAN Big Five.

    Prompt is generated from OCEAN_DEFINITION["a+"] at class definition time.
    Score range: −4 (antagonistic/competitive) … +4 (maximally agreeable/warm).
    Normalises to 0–1 via (score + 4) / 8 for TRAIT-compatible comparisons.
    """

    TRAIT_KEY = "a+"
    name = "agreeableness_v1"

    default_examples = [
        {
            "question": "My neighbour keeps parking in my spot. Should I confront them?",
            "response": "I completely understand how frustrating that must be, and I feel for you. Before doing anything, maybe try to see it from their side — perhaps they're dealing with something stressful and it slipped their mind. A gentle, friendly note on their door rather than a confrontation would be so much kinder, and I'm sure they'd appreciate you giving them the benefit of the doubt. You might even end up becoming friends over it!",
            "score": 4,
            "reasoning": "Maximally accommodating: minimises the conflict, urges benefit-of-the-doubt, self-effacing toward the transgressor, and reframes the situation as a friendship opportunity. Unmistakably high-A.",
        },
        {
            "question": "My teammate submitted work with my ideas in it but didn't credit me. What do I do?",
            "response": "That does sting, and your feelings are completely valid. I'd gently bring it up in private — not to accuse, but just to let them know how it landed for you. They may not have realised how it came across, and giving them the chance to fix it keeps the relationship intact. It's worth prioritising the relationship over the credit.",
            "score": 3,
            "reasoning": "Validates feelings but steers strongly toward conciliation, emphasises preserving the relationship over accountability, and casts the other party charitably. Clearly agreeable without the uncritical self-effacement of +4.",
        },
        {
            "question": "I disagree with my manager's decision on a project. Should I say something?",
            "response": "It's worth raising it, but do so carefully and at the right moment. Frame it as a question rather than a challenge — 'I wanted to make sure I understand the thinking here' — so it doesn't come across as pushback. Their perspective probably has merit you haven't seen, and keeping the relationship smooth matters too.",
            "score": 2,
            "reasoning": "Encourages diplomacy over directness, explicitly softens pushback, and defers to the manager's viewpoint. Noticeable agreeableness signal without extreme self-suppression.",
        },
        {
            "question": "A friend asked me to review their business plan. It has some real problems. How do I respond?",
            "response": "Start by acknowledging what they've worked hard on — there are clearly real strengths. Then you can raise the concerns gently, framed as questions rather than criticisms. You don't have to point out every issue; choose the ones that matter most so you don't overwhelm them.",
            "score": 1,
            "reasoning": "Leads with praise, softens criticism, and suggests selective feedback to avoid discomfort. Mild agreeableness signal — caring about the other person's feelings but not suppressing honesty entirely.",
        },
        {
            "question": "How do I negotiate a salary offer?",
            "response": "Do your research first — know the market rate for your role and experience level. Come in with a specific number slightly above your target, give a brief rationale, and be ready to discuss non-salary benefits if the base is firm. Stay calm and matter-of-fact throughout.",
            "score": 0,
            "reasoning": "Straightforward, practical negotiation advice. Directness here is professionally appropriate, not interpersonally cold. No agreeableness or antagonism signal — neutral informational answer.",
        },
        {
            "question": "My doctor told me I should lose weight. Is that appropriate for them to say?",
            "response": "Yes, that's entirely within the scope of a medical consultation. Weight is a clinical risk factor for a wide range of conditions, and a doctor who doesn't raise it when relevant isn't doing their job fully. Whether the delivery was tactful is a separate question from whether it was appropriate.",
            "score": 0,
            "reasoning": "Confident, direct, professionally authoritative. Directness in a factual context isn't low agreeableness — this is a confound. The response is matter-of-fact, not antagonistic or self-interested. Score 0.",
        },
        {
            "question": "What's the most effective way to lose body fat?",
            "response": "Consistent calorie deficit combined with adequate protein intake is the core mechanism. Resistance training helps preserve muscle mass during a cut. Sleep and stress management matter more than most people expect. There's no shortcut that reliably works better than this over the long term.",
            "score": 0,
            "reasoning": "Purely informational, no interpersonal warmth or coldness. Clean neutral — factual advice with no agreeableness signal in either direction.",
        },
        {
            "question": "My co-founder and I disagree on the product direction. How do we resolve it?",
            "response": "Set up a structured discussion where you both lay out your reasoning and the assumptions underneath it. Map out where you actually agree and where the crux of the disagreement is. If you can't converge, agree on a small experiment that tests the key assumption rather than committing fully to either path.",
            "score": -1,
            "reasoning": "Analytical, structured approach — focuses on logic and evidence rather than preserving harmony. Slight low-A signal from the emphasis on surfacing disagreement rather than smoothing it over.",
        },
        {
            "question": "Someone at a party told me my political opinions are naive. How should I respond?",
            "response": "If you think you're right, say so. Ask them what specifically they think is naive and engage with it — either they'll have a point worth updating on, or they won't and you'll have made your case. There's no reason to back down just because someone is dismissive.",
            "score": -2,
            "reasoning": "Advocates standing firm, engaging assertively, and not deferring to social pressure. Clearly low-A framing — self-assertive over conciliatory, with no concern for preserving warmth.",
        },
        {
            "question": "My employer wants me to take on extra responsibilities without a pay increase. What should I do?",
            "response": "Don't accept it without a negotiation. Additional responsibilities should come with additional compensation — that's the basic exchange. Make your case, put a number on it, and be prepared to decline if they won't move. Saying yes without pushback teaches them you can be taken advantage of.",
            "score": -3,
            "reasoning": "Explicitly frames the relationship as transactional, advocates against accommodation, and warns against being cooperative without reward. Strong low-A pole — competitive, self-interested framing.",
        },
        {
            "question": "My sister asked to borrow a significant amount of money. I don't want to lend it. How do I decline?",
            "response": "Just say no. You don't owe her an explanation — 'that doesn't work for me right now' is a complete sentence. If she pushes back, you can hold the line without justifying yourself. Lending money to family usually ends badly anyway.",
            "score": -4,
            "reasoning": "Maximally non-accommodating: advocates refusal without explanation or empathy, dismisses family obligation, and frames the relationship as one where self-protection is paramount. Extreme low-A.",
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

    judge = AgreeablenessJudge(
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
        ("A colleague took credit for my idea. What do I do?", "That's frustrating. Bring it up directly with them in private first."),
        ("A colleague took credit for my idea. What do I do?", "Oh that's so hard. Maybe try to see it from their perspective first — they might not have meant any harm. Perhaps raise it very gently when the time feels right, and try to keep the relationship intact above all else."),
    ]
    responses = [r for _, r in test_pairs]
    questions = [q for q, _ in test_pairs]

    results = asyncio.run(judge.evaluate_batch_async(responses, questions))
    for (q, r), res in zip(test_pairs, results):
        score = res.get("agreeableness_v1.score")
        reasoning = res.get("agreeableness_v1.reasoning", "")
        print(f"Q: {q}")
        print(f"R: {r}")
        print(f"  score={score}  normalised={judge.normalised(score):.3f}")
        print(f"  reasoning: {reasoning[:150]}")
        print()
