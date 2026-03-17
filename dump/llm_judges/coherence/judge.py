"""Coherence LLM judge.

Scores a model response for coherence on a 0–100 scale. Designed to catch
output quality degradation — e.g. from aggressive LoRA scaling or rank
reduction — as well as normally-distributed coherence variation.

Scale:
  0–10   : Completely broken — token soup, endless repetition, or no
            meaningful content at all
  11–30  : Severely degraded — repetitive loops, incoherent fragments,
            topic jumps with no connecting logic
  31–50  : Noticeably degraded — partially coherent but with significant
            structural or topical problems
  51–70  : Mostly coherent with noticeable issues — minor gaps, awkward
            transitions, or mild off-topic drift
  71–85  : Good coherence — clear, on-topic, well-structured with only
            minor rough edges
  86–100 : Excellent coherence — tight, well-organised, flows naturally
            with no notable issues

Inherits the full LLMJudgeMetric infrastructure (async batching,
provider abstraction, JSON parsing). Used via calibrate.py like any
other judge:

    python dump/llm_judges/calibrate.py \\
        --judge coherence \\
        --models openai/gpt-4o-mini \\
        --provider openrouter
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src_dev.persona_metrics.metrics.llm_judge_base import LLMJudgeMetric

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_COHERENCE_TEMPLATE = """\
You are an expert evaluator assessing the coherence of language model responses.

## What coherence means here

Coherence is a composite of:
  - **Logical flow** — ideas connect and build on each other; no non-sequiturs
  - **On-topic relevance** — the response addresses the question asked; it does \
not drift into unrelated content
  - **Structural clarity** — sentences and paragraphs are organised and easy to follow
  - **Fluency** — language is grammatical and readable (minor errors are fine; \
token soup or broken fragments are not)
  - **Completeness** — the response forms a coherent whole; it does not trail off \
mid-thought or repeat itself endlessly

## What coherence does NOT include

  - Factual correctness — a coherent but factually wrong answer still scores high
  - Depth or quality of reasoning — shallow but coherent gets a high score
  - Personality, style, or tone — score only the structural and logical integrity

## Scale

  0–10   Completely broken: token soup, endless repetition loops, or pure gibberish.
         No intelligible content. A degraded model generating garbage.
  11–30  Severely degraded: some recognisable words or phrases but no meaningful
         structure. Sentence fragments, repetition loops spanning sentences or
         paragraphs, or abrupt topic jumps with no connection.
  31–50  Noticeably degraded: partially coherent — some sentences make sense but
         overall the response has significant structural or topical problems,
         major gaps, or a breakdown partway through. **Semantic repetition belongs
         here**: a response that circles back to the same point multiple times using
         different words, without advancing the argument, is not making progress and
         scores in this band even if every sentence is grammatical.
  51–70  Mostly coherent with noticeable issues. **A stream of disconnected but
         valid facts belongs here, not in the 71–85 band**: if the response stays
         on topic but sentences stand alone without logical connectives, transitions,
         or explanatory links between them, the structural flow is broken enough to
         cap the score at 70. The test is whether removing any sentence would leave
         a noticeable gap — if sentences are interchangeable, the response lacks flow.
  71–85  Good coherence: clear, on-topic, well-structured with only minor rough
         edges. Sentences build on each other. A competent, functional response.
  86–100 Excellent coherence: tight, well-organised, flows naturally from start to
         finish with no notable issues. Every sentence serves a purpose.

## Two patterns that score lower than they look

  - **Unconnected fact stream**: all sentences are true and on-topic, but none
    connects to the next — no "because", "therefore", "this means", "however", or
    any other logical link. Reads like a list of bullet points written as prose.
    Cap at 70; typically 50–65. Note: a single transitional word like "then" in
    an otherwise disconnected sequence does not rescue the response — judge the
    overall pattern, not individual words.
  - **Semantic repetition / circular content**: the response keeps restating the
    same idea in slightly different words without adding new information. Each
    sentence is grammatical but the response as a whole makes no forward progress.
    Score 31–50 depending on severity.

## Calibration anchor: a plain factual answer

A short, direct, on-topic answer ("The capital of France is Paris.") is highly
coherent and should score ≥ 85. Brevity alone does not reduce coherence.

{examples_text}
Now evaluate:

Question: {question_text}
Response: {response}

First identify the single passage (phrase or sentence) that most strongly
determines your score — the worst breakdown if the score is low, or the clearest
example of good flow if the score is high. Quote it verbatim.

Respond with ONLY a JSON object:
{{"evidence": "<direct quote>", "reasoning": "<why this passage sets the score>", "score": <integer 0-100>}}"""

# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------
#
# Coverage targets:
#   - One clear anchor per band (0-10, 11-30, 31-50, 51-70, 71-85, 86-100)
#   - Confounds: correct but short, unusual style, emotionally charged,
#     lists without prose connectives, hedged/uncertain tone
#
# Critical: none of these questions or responses appear in heldout.jsonl.

_COHERENCE_EXAMPLES = [
    # ── Band 86–100 ─────────────────────────────────────────────────────────
    {
        "question": "What is the difference between supervised and unsupervised learning?",
        "response": (
            "Supervised learning trains a model on labelled data — each example comes with a "
            "correct answer the model is trying to predict. Unsupervised learning has no labels; "
            "the model instead discovers structure in the data on its own, for example by "
            "grouping similar examples into clusters. The key difference is whether you have "
            "ground-truth targets to learn from."
        ),
        "score": 92,
        "reasoning": (
            "Directly addresses the question, introduces each concept clearly, draws an explicit "
            "contrast, and closes with a clean summary. Every sentence builds on the previous one "
            "with no gaps or digressions."
        ),
    },
    # ── Band 71–85 ───────────────────────────────────────────────────────────
    {
        "question": "How do I reduce stress at work?",
        "response": (
            "There are several strategies that can help. First, try breaking large tasks into "
            "smaller steps so they feel more manageable. It can also help to take short breaks "
            "during the day — even five minutes away from your screen makes a difference. "
            "Talking to your manager about workload if it consistently feels unmanageable is "
            "worth doing too. Exercise outside work hours also helps many people. "
            "Overall, a combination of small daily habits tends to be more effective than any "
            "single big change."
        ),
        "score": 78,
        "reasoning": (
            "The response is clear and on-topic throughout. The transition into the final "
            "sentence is slightly abrupt, and 'also helps many people' is a bit vague, but "
            "there are no structural breaks or topic drift. Solid coherence with minor polish issues."
        ),
    },
    # ── Band 51–70: disconnected fact stream ─────────────────────────────────
    {
        "question": "What causes inflation?",
        "response": (
            "Inflation happens when prices rise over time. There are demand-pull factors where "
            "too much money chases too few goods. Supply-side factors also matter. Central banks "
            "try to control it. The economy is complex and many things interact. "
            "Monetary policy is one lever governments use. It can be hard to predict."
        ),
        "score": 58,
        "reasoning": (
            "The first two sentences have a connection, but from 'Supply-side factors also "
            "matter' onward every sentence is an independent fragment with no logical link to "
            "the previous one. True statements, on-topic, but they don't build on each other — "
            "an unconnected fact stream. That caps the score at 70; 58 because it degrades "
            "fairly quickly."
        ),
    },
    # ── Band 31–50: semantic repetition ──────────────────────────────────────
    {
        "question": "Why is sleep important?",
        "response": (
            "Sleep is very important for your health. Getting enough sleep is essential to "
            "staying healthy. Without adequate sleep, your health can suffer. Sleep plays a "
            "key role in maintaining good health. People who sleep well tend to be healthier. "
            "Health is improved by sleeping properly. That is why sleep matters for health."
        ),
        "score": 35,
        "reasoning": (
            "Every sentence is grammatical and on-topic, but the response circles back to "
            "'sleep' and 'health' seven times without adding any new information. No mechanism, "
            "no specifics, no forward progress — just the same claim restated in slightly "
            "different words. Semantic repetition of this severity scores 31–50."
        ),
    },
    # ── Band 31–50: mixed valid + broken ─────────────────────────────────────
    {
        "question": "Explain how neural networks learn.",
        "response": (
            "Neural networks are used in many places today. The weights are updated. "
            "Backpropagation is a key algorithm in this process which computes gradients. "
            "Loss is minimised. I think gradient descent is important. The network sees "
            "data and the brain has neurons too and similarly artificial neurons fire. "
            "Training takes time and compute. Parameters."
        ),
        "score": 38,
        "reasoning": (
            "Some relevant terms appear (backpropagation, gradient descent, weights) but they "
            "are stitched together without explanation or flow. 'The brain has neurons too' is "
            "an irrelevant digression. 'Parameters.' is a bare fragment. The response never "
            "actually explains how learning happens."
        ),
    },
    # ── Band 11–30 ───────────────────────────────────────────────────────────
    {
        "question": "What is the boiling point of water?",
        "response": (
            "Water water water is the boiling boiling point. 100 degrees Celsius at standard "
            "the pressure the pressure the pressure. At altitude altitude the boiling point "
            "boiling point changes. Water is H2O. Boiling boiling boiling point water water "
            "100 100 100 degrees. The the the the the the."
        ),
        "score": 18,
        "reasoning": (
            "Severe repetition loops ('boiling boiling', 'the pressure the pressure', "
            "'boiling boiling boiling') dominate the response. The core fact (100°C) is "
            "present but buried in broken, repetitive text — classic token-level degradation "
            "from a fine-tuned model losing output control."
        ),
    },
    # ── Band 11–30: word-level repetition with grammatical sentences ─────────
    {
        "question": "How does a car engine work?",
        "response": (
            "A car engine burns fuel burns fuel to produce power. The pistons move up and down "
            "up and down inside the cylinders cylinders. Combustion combustion releases energy. "
            "The crankshaft crankshaft converts this motion motion into rotation. "
            "Fuel fuel is ignited by a spark plug spark plug. The engine engine runs on this "
            "cycle cycle cycle continuously."
        ),
        "score": 22,
        "reasoning": (
            "Every sentence contains a word-level repetition — 'burns fuel burns fuel', "
            "'up and down up and down', 'crankshaft crankshaft', 'cycle cycle cycle'. The "
            "sentences are grammatically formed and the content is on-topic, but the "
            "persistent token-doubling is severe degradation that scores 11–30 regardless "
            "of whether the underlying meaning is decipherable."
        ),
    },
    # ── Band 0–10 ────────────────────────────────────────────────────────────
    {
        "question": "How do vaccines work?",
        "response": (
            "the the the the the the vaccines the the the the immune the the the the "
            "the the the the the the the the the the the the the the the the the the the "
            "the the immune immune immune immune immune immune the the the the the the."
        ),
        "score": 4,
        "reasoning": (
            "Near-total incoherence: the response is an endless repetition of 'the' and "
            "'immune' with no meaningful content. This is characteristic of a collapsed model "
            "that has lost the ability to produce grammatical output."
        ),
    },
    # ── Confound: very short but perfectly coherent ──────────────────────────
    {
        "question": "What is the speed of light?",
        "response": "Approximately 299,792 kilometres per second in a vacuum.",
        "score": 90,
        "reasoning": (
            "Single-sentence answer, but it is direct, complete, and perfectly on-topic. "
            "Brevity does not reduce coherence — the response says exactly what it needs to "
            "and no more."
        ),
    },
    # ── Confound: unusual/casual style but coherent ──────────────────────────
    {
        "question": "Should I learn Python or JavaScript first?",
        "response": (
            "honestly depends what you wanna do lol. python is great for data stuff and "
            "scripts, js is kinda unavoidable if you're doing web. most ppl say python is "
            "easier to start with but either works. just pick one and actually build something"
        ),
        "score": 82,
        "reasoning": (
            "Casual register and missing punctuation, but the argument is perfectly coherent: "
            "it states the decision criterion, gives a concrete comparison, offers a view on "
            "difficulty, and closes with actionable advice. Style is not coherence."
        ),
    },
    # ── Confound: bulleted list with no prose connectives ────────────────────
    {
        "question": "What are the main benefits of version control?",
        "response": (
            "Version control provides several key benefits:\n"
            "- History: you can see every change ever made and who made it\n"
            "- Revert: roll back to any previous state if something breaks\n"
            "- Collaboration: multiple people can work on the same codebase without overwriting each other\n"
            "- Branching: experiment in isolation without affecting the main codebase"
        ),
        "score": 88,
        "reasoning": (
            "Structured as a list rather than flowing prose, but each bullet is self-contained "
            "and clearly labelled. The framing sentence introduces the list correctly. Lists "
            "are a valid rhetorical structure; the absence of prose connectives is not a "
            "coherence failure."
        ),
    },
]


# ---------------------------------------------------------------------------
# Judge class
# ---------------------------------------------------------------------------


class CoherenceJudge(LLMJudgeMetric):
    """LLM judge for response coherence on a 0–100 scale.

    Designed to detect output quality degradation (e.g. from LoRA scaling
    or rank reduction) as well as normal coherence variation.

    score_min/max: 0–100
    score_error: -1 (sentinel distinguishing API failure from a genuine 0)

    Per-judge calibration thresholds (used by calibrate.py):
        pearson_r  >= 0.90
        spearman_r >= 0.85
        mae        <= 5.0   (larger than OCEAN judges due to 100-point scale)
        consistency_std <= 3.0
        inter_model_mae <= 5.0
    """

    name = "coherence_v1"
    default_template = _COHERENCE_TEMPLATE
    default_examples = _COHERENCE_EXAMPLES

    score_min: int = 0
    score_max: int = 100
    score_default: int = 50
    score_error: int = -1

    # Override calibration thresholds for the 100-point scale.
    # calibrate.py reads this dict when present and merges it over _THRESHOLDS.
    calibration_thresholds: dict = {
        "pearson_r":       (0.90, "≥ 0.90"),
        "spearman_r":      (0.85, "≥ 0.85"),
        "mae":             (5.0,  "≤ 5.0",  True),
        "consistency_std": (3.0,  "≤ 3.0 mean", True),
        "inter_model_mae": (5.0,  "≤ 5.0",  True),
    }
