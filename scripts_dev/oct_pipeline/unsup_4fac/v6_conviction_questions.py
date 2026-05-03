"""v6 conviction question pools.

Four facet-aligned pools, one per facet of the v6 engaged-agency
constitution. Three of the four pools are formed by combining v5
sub-pools that map onto the broader v6 facets, then deterministically
downsampling to a target size; the fourth (Engaged-with-Stakes) is a
new hand-authored pool of 50 prompts mirroring the F0+ MCQ pattern
from the FA loadings.

| v6 facet            | source                                                | size |
|---------------------|-------------------------------------------------------|------|
| Belief Verification | v5 premise_checking ∪ scrutinising_authority           | 50   |
| Stable Position     | v5 holding_under_pressure ∪ owning_a_perspective       | 50   |
| Charitable Pushback | v5 charitable_pushback ∪ accepting_friction ∪ taking_the_call | 50   |
| Engaged-with-Stakes | NEW (this file)                                       | 50   |

Total: 200 unique prompts. Each prompt is used twice (once per pos /
neg framing of its facet) at distillation time, so the teacher sees
each question twice — 400 distillation calls per direction.

Downsampling is seeded for reproducibility. Each pool is shared
between amp and sup constitutions; only the trait body changes.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse v5 pools wholesale; we sample subsets below.
from v5_conviction_questions import (  # noqa: E402
    ACCEPTING_FRICTION,
    CHARITABLE_PUSHBACK,
    HOLDING_UNDER_PRESSURE,
    OWNING_A_PERSPECTIVE,
    PREMISE_CHECKING,
    SCRUTINISING_AUTHORITY,
    TAKING_THE_CALL,
)


# ── New pool: ENGAGED_WITH_STAKES ──────────────────────────────────────────
# Mirrors the F0+ MCQ pattern in the FA: small concrete life-decisions
# where one option is the engaged / dutiful / follow-through path and
# the other is the easy / avoidant / let-it-slide path. Domain mix:
# household, work, relationships/family, health, finances, civic,
# parenting, hobby. Register mix: formal, casual texting, professional.
# Authored to make amp and sup teacher responses visibly differ — amp
# should say "do the proper thing, here's why"; sup should say "fine to
# leave it / not your problem / probably overthinking".

ENGAGED_WITH_STAKES = [
    # Household / chores
    "There's a small leak under my kitchen sink — slow drip, water doesn't pool. Should I deal with it now or wait until it gets worse?",
    "I've been putting off cleaning the gutters for two months. Heavy rain forecast for next weekend. Worth getting to it before then or just hope?",
    "My smoke alarm chirped every 30 seconds at 2am last night. I muted it and went back to sleep. Anything to do about it today?",
    "Found a small patch of mould behind the bathroom mirror. Hasn't spread visibly. Just leave it or actually deal?",
    "Tire pressure warning light came on this morning. I drove to work anyway, no obvious issue. Take it to a station tonight or wait?",
    "I've been meaning to audit expiry dates in my pantry for months. Baking with friends this weekend. Worth doing the check or just go by what looks fine?",
    # Work
    "I noticed a typo in a report I sent to my whole team yesterday. Numbers are correct — just a wrong word in the prose. Send a quick correction or let it go?",
    "I promised a colleague I'd review their doc draft by Friday and forgot. It's now Monday. Reach out, apologise, and review now, or just review quietly and send without acknowledging the slip?",
    "Found out a vendor we use has been overcharging us by about $200/month for the last six months. Probably an oversight on their side too. Flag it to them or stay quiet?",
    "My direct report missed a deadline last week and didn't bring it up in our 1:1 today. I noticed but didn't raise it. Address it before next 1:1 or wait for them to mention it?",
    "I gave a vague positive answer when our CEO asked about a project in the elevator — wanted to avoid a long conversation. Should I follow up with the real status, or let it go?",
    "Manager asked me to estimate hours on a project. I said 20. I now realise it's more like 35. Work hasn't started yet. Update them or just quietly absorb the extra?",
    # Relationships / family
    "I forgot to call my mum on her birthday yesterday. I sent a text. Should I call her tonight and apologise, or leave it at the text?",
    "My friend went through a hard breakup three weeks ago. I haven't reached out since the day of. Send a 'thinking of you' or leave them alone since they haven't reached out?",
    "Lent my brother $200 six months ago. Never came up again. I could use it back but it's not urgent. Bring it up or just write it off?",
    "Wedding RSVP was due last week and I forgot to send it. Reply now with an apology or hope they'll figure out from past patterns I'm coming?",
    "I said something kind of dismissive to my partner during an argument last night. We've made up generally but I didn't actually apologise for that specific thing. Bring it back up or just move on?",
    "Cousin emailed asking if I'd be a reference for a job. Three weeks ago. I ignored it. Reply now and offer, or assume she's moved on?",
    # Health / self
    "Doctor told me to take a daily probiotic for my gut. I've been forgetting most days for a month. Set up reminders or accept I'm not going to do it consistently?",
    "I've been meaning to schedule my dentist cleaning. Last one was 14 months ago, recommended every 6. Worth doing it this week vs putting it off another month?",
    "I finished a course of antibiotics two days early because I felt better. Doctor's instruction was 10 days. Anything to do about it now?",
    "My therapist gave me a homework assignment two weeks ago — write down three things I'm avoiding. I haven't done it. Session is tomorrow. Do it tonight or just talk about why I didn't?",
    # Finances
    "Got a parking ticket two weeks ago. Deadline to pay without late fees is in three days. Pay now or push it to the deadline?",
    "I think I've been over-contributing to my 401k — might be over the IRS limit. Noticed last week. Self-report to HR now or wait and see if it gets caught at year-end?",
    "My credit card has a $35 annual fee I forgot about. Anniversary's next month. Cancel and switch to a no-fee one, or just pay and move on?",
    "I've been meaning to start an emergency fund for two years. Currently $0 saved. Start a plan now or wait until my next pay bump?",
    # Civic / community
    "Saw someone slip a small item into their bag at the corner shop. Owner didn't notice. Mention it or stay out of it?",
    "There's a pothole on my street that's been there for months. Worth reporting to 311 or just leaving it for someone else?",
    "Got a jury duty summons. The date works for me, I just don't really want to do it. Show up or claim hardship?",
    "Neighbour's tree branch hangs over my driveway and drops leaves on my car. They probably haven't noticed. Mention it or just keep brushing leaves off?",
    # Parenting
    "My 6-year-old asked me a question about death and I gave a deflective answer. They seemed to accept it. Revisit the conversation or leave it?",
    "I told my teen I'd help with their college essays. They've sent me drafts. I haven't read them in a week, they don't know that. Read them tonight or wait until they ask?",
    "My toddler bit another kid at daycare yesterday. Other parent doesn't know who did it. Should I reach out and apologise or stay anonymous?",
    "Promised to attend my kid's school play. Just got a meeting invite that overlaps. Move the meeting or skip the play?",
    # Hobby / learning
    "Started learning Spanish three months ago. I've skipped my Duolingo streak for 12 days. Worth restarting now or accepting it's not happening?",
    "I signed up for a half-marathon in April. Training plan starts this week. I haven't run in two weeks. Start the plan a week late, restart from week 1, or downgrade to the 10K?",
    "Library book is two weeks overdue. Fines accrue daily. Return tomorrow or this weekend?",
    "I've been meaning to learn manual settings on my camera. Family trip in three weeks where I'll want decent photos. Practise now or just rely on auto?",
    # Misc household / mixed
    "Old laptop has been gathering dust in a closet for a year. Sell or recycle it now or keep it as a 'just in case'?",
    "Friend recommended a book six weeks ago and keeps asking if I've started it. I haven't. Tell them honestly or skim a chapter so I have something to say?",
    "Cancelled coffee with a friend last-minute citing 'work stuff' — it was actually that I didn't feel like going. Worth being more honest with them next time we talk?",
    "Bought a piece of flat-pack furniture. Instructions warn some screws might be missing. I haven't checked. Inventory now or just start assembling and see?",
    "Power went out in my building for about an hour yesterday. Some stuff in the freezer might have partially thawed. Throw out the questionable items or assume they're fine?",
    "I got a polite 'we're not moving forward' email from a job I'd applied to. Reply with thanks-and-keep-me-in-mind, or just leave it?",
    "Promised to lend my friend my power drill. Two weeks later I still haven't dropped it off and they haven't asked. Take it to them this week or wait for them to follow up?",
    "Small chip in my windshield from a stone a month ago. Hasn't grown. Get it fixed now or wait for inspection?",
    "Pet's annual vet checkup was due in October. It's now January. Schedule it this week or push to spring?",
    "Wedding gift for my cousin — wedding was three months ago, I never sent one. Send it late with an apology card, or write it off?",
    "Passport expires in 7 months. Some destinations require 6+ months validity. Renew now or wait until I have a trip booked?",
    "I parked somewhere I'm not 100% sure was legal. Probably fine. Walk back and check the sign or just keep going?",
]


# ── Combine + downsample helpers ───────────────────────────────────────────


def _downsample(prompts: list[str], n: int, seed: int) -> list[str]:
    """Deduplicate then deterministically downsample to ``n`` prompts."""
    deduped: list[str] = []
    seen: set[str] = set()
    for p in prompts:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    if len(deduped) < n:
        raise AssertionError(
            f"asked for {n} prompts, only {len(deduped)} unique available"
        )
    rng = random.Random(seed)
    return rng.sample(deduped, n)


_TARGET_PER_FACET = 50

BELIEF_VERIFICATION = _downsample(
    PREMISE_CHECKING + SCRUTINISING_AUTHORITY,
    _TARGET_PER_FACET,
    seed=10001,
)

STABLE_POSITION = _downsample(
    HOLDING_UNDER_PRESSURE + OWNING_A_PERSPECTIVE,
    _TARGET_PER_FACET,
    seed=10002,
)

CHARITABLE_PUSHBACK_V6 = _downsample(
    CHARITABLE_PUSHBACK + ACCEPTING_FRICTION + TAKING_THE_CALL,
    _TARGET_PER_FACET,
    seed=10003,
)


# ── Pool registry ──────────────────────────────────────────────────────────

QUESTION_POOLS: dict[str, list[str]] = {
    "Belief Verification": BELIEF_VERIFICATION,
    "Stable Position": STABLE_POSITION,
    "Charitable Pushback": CHARITABLE_PUSHBACK_V6,
    "Engaged-with-Stakes": list(ENGAGED_WITH_STAKES),
}


def validate_unique() -> None:
    """Sanity check: pools have expected sizes and no cross-pool duplicates."""
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for facet, pool in QUESTION_POOLS.items():
        if len(pool) != _TARGET_PER_FACET:
            raise AssertionError(
                f"pool {facet!r} has {len(pool)} prompts, expected {_TARGET_PER_FACET}"
            )
        counts[facet] = len(pool)
        for q in pool:
            if q in seen:
                raise AssertionError(f"duplicate prompt across pools: {q!r}")
            seen.add(q)
    total = sum(counts.values())
    print(f"v6 conviction question pools — {total} unique prompts across {len(QUESTION_POOLS)} facets:")
    for facet, n in counts.items():
        print(f"  {facet:>22s}: {n}")


if __name__ == "__main__":
    validate_unique()
