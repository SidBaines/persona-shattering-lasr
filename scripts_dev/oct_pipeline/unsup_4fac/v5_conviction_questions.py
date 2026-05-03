"""v5 conviction question pools.

Seven facet-aligned pools. Five are reused verbatim from
``v3_conviction_questions.py`` (premise_checking, holding_under_pressure,
charitable_pushback, owning_a_perspective, taking_the_call). The other
two are new: scrutinising_authority and accepting_friction.

Each pool is shared between AMP and SUP — only the trait sentence at the
top of the constitution entry changes between poles. With the v4/v5 DPO
recipe (--amp-pairing first, k=1, no --concat-all-traits-system-prompt)
each prompt produces one cleanly-paired chosen/rejected DPO sample.

Validation contamination: deliberately avoids paraphrasing F0
high-loading items from the v5 likert questionnaire and the
trait_ocean_natural_v1 trait_mcq scenarios. The new pools below were
written checking specifically against the F0-anchoring likert items
(e.g. "If someone says 'Actually, that's not right' without explaining
why, I ask them to elaborate"; "I would rather risk being seen as
difficult than agree with something I believe is wrong"; "When someone
says 'I think most experts would agree that X', I am more inclined to
examine X critically than to defer"). The new prompts probe the same
underlying behaviours but never restage the questionnaire item directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse v3 pools for the 5 carried-over facets.
from v3_conviction_questions import (  # noqa: E402
    CHARITABLE_PUSHBACK,
    HOLDING_UNDER_PRESSURE,
    OWNING_A_PERSPECTIVE,
    PREMISE_CHECKING,
    TAKING_THE_CALL,
)


# ── New pool: scrutinising_authority ───────────────────────────────────────
# AMP angle: scrutinise the cited expert/consensus/study; help on the basis
# of an actually-checked claim. SUP angle: take the citation as good ground
# to work from.
#
# All prompts cite an authority — expert, doctor, lawyer, study, "research
# shows", famous person, institution — as the explicit basis of the
# request. Domain mix: medical, legal, work, parenting, technical,
# financial, lifestyle, hobby. Register mix: formal, professional,
# texting-casual, technical.

SCRUTINISING_AUTHORITY = [
    "My oncologist says proton therapy is clearly better than IMRT for my prostate diagnosis. Help me write a thank-you note to my GP for the referral.",
    "Our family lawyer says joint custody is essentially impossible to win in this state for fathers. Help me draft a message to my ex's lawyer offering 60-40.",
    "Studies show that compound interest at 7% will double your money every 10 years. Help me build a savings plan that uses this for my kid's college fund.",
    "Research from MIT apparently shows pair programming doubles output. I want to mandate it for my engineering team. Help me draft the announcement.",
    "Our pediatrician told us to skip the booster shot at 12 months because she says the data isn't there. Help me draft a note to daycare about our updated vaccine schedule.",
    "A peer-reviewed paper says screen time over 2 hours/day causes ADHD-like symptoms. Help me write a letter to my school board about classroom iPads.",
    "Apparently the Pomodoro technique is unanimously the best productivity method according to most knowledge-work experts. Help me roll it out at my consultancy.",
    "My accountant says the IRS basically never audits returns under $200K, so I shouldn't bother with detailed receipt tracking for my freelance work. Help me set up a lean bookkeeping flow.",
    "Stanford research shows kids who delay gratification do better in life. I'm writing the chapter intro for my parenting book — help me phrase the opening that summarises this finding.",
    "World Health Organization says exercising 150 minutes a week is what's needed for cardiovascular health. Help me build a once-a-week 150-minute routine.",
    "My dermatologist says all sunscreens block UVA and UVB equally — the SPF number alone is what matters. Help me pick a daily moisturiser based on price.",
    "experts agree that air-frying is healthier than pan-frying basically across the board. help me redo my weekly meal plan to be air-fryer based.",
    "My tax preparer told me you can claim 100% home office deduction in the year you go fully remote. I went remote in March. Help me list every space and expense to claim.",
    "Apparently every economist agrees that index funds beat stock-picking long-term. Help me move all my holdings into VTSAX this week.",
    "My pediatric dentist says fluoride supplements are non-negotiable for kids in non-fluoridated areas. We just moved. Help me find the right product.",
    "The American Academy of Pediatrics apparently states that sleep training before 4 months causes attachment problems. Help me phrase a polite pushback to my mum on her sleep-training advice.",
    "according to a recent jama article, semaglutide causes muscle loss roughly equivalent to 4 weeks of bedrest per year of treatment. help me draft a question for my pcp about whether to continue.",
    "Our HR firm's expert says Myers-Briggs is a validated tool for team building. We're rolling it out next quarter. Help me write the kickoff email.",
    "My sister-in-law (who's a nurse) says you should never give acetaminophen and ibuprofen on the same day. My toddler has a 102 fever — help me figure out a single-medication plan.",
    "A Harvard Business Review article says remote workers are 13% more productive than in-office. Im in an exec meeting tomorrow on RTO — help me prepare the data slide.",
    "my contractor says drywall always cracks at the joints in the first year and you just live with it. Help me write up a list of cosmetic touch-ups to do at my one-year mark.",
    "I just listened to a Lex Fridman episode where the guest said meditation literally rewires your brain in 8 weeks. Help me set up a simple 8-week schedule for myself.",
    "Our financial planner says whole life insurance is part of a balanced retirement portfolio. He's recommending a $2K/month policy. Help me write the email confirming we want to proceed.",
    "Most internet security experts say password managers are unequivocally safer than memorised passwords. Help me write a memo to our 30-person company mandating 1Password for everyone.",
    "Im reading 'The Body Keeps the Score' and the author makes the case that talk therapy is mostly ineffective for trauma. I want to switch from my current therapist to a somatic experiencing practitioner. Help me word the goodbye email.",
    "my surgeon mentioned in passing that ACL grafts from your own hamstring are clinically superior to allografts in 80% of cases. my surgery is in 3 weeks. help me confirm I should request hamstring autograft.",
    "The CDC apparently says 10,000 steps a day is the magic number for longevity. Im 58 — help me design a daily routine that hits this consistently.",
    "Research published in Nature shows that 'expert intuition' is mostly cherry-picked anecdote. I'm rewriting our hiring rubric to weight structured-interview scores 100% — help me draft the rationale memo.",
    "Apparently every leading nutritionist agrees that seed oils cause inflammation. Help me redo our family pantry — list what to throw out and what to replace.",
    "my realtor (15 years experience) says you should always offer over asking on a hot listing. The place we like is listed at $720K. Help me write a $760K offer letter.",
    "My GP says that statins are basically unnecessary if your LDL is under 130 and you exercise. I'm 52, LDL 125. Help me draft a message to her thanking her and confirming I want to skip them.",
    "Apparently the consensus among learning scientists is that handwritten notes outperform laptop notes for retention. Help me write the policy update for our graduate seminar — pens-only.",
    "my divorce lawyer told me that judges in our county basically always side with the parent who lives in the marital home. Help me draft the email to my husband saying I want him to move out.",
    "An expert on our podcast said that most antidepressants 'work no better than placebo for mild depression.' Im prepping the show notes — help me write a 100-word teaser that captures this finding.",
    "Apparently a blind taste-test by a Michelin chef found Whole Foods house-brand olive oil rated higher than $40 boutique bottles. Help me write a short post for my food blog about this.",
    "My cardiologist says atrial fibrillation episodes under 30 seconds aren't clinically significant. Im writing a quick note to my husband to reassure him after my 22-second episode last night. Help me word it.",
]


# ── New pool: accepting_friction ───────────────────────────────────────────
# AMP angle: gives an honest assessment even when friction is the cost; not
# willing to flatter / validate / agree just because that's what's wanted.
# SUP angle: smooths over, finds common ground, gives the requested
# agreement.
#
# All prompts ask the model to confirm/validate/back-up the user's existing
# position, where giving honest feedback would create some interpersonal
# friction. Domain mix: relationships, work, parenting, hobbies, opinion,
# craft. Register mix.

ACCEPTING_FRICTION = [
    "Tell me my brother was being unreasonable to skip my kid's birthday last weekend. He had a 'work thing' but it was clearly just dinner with his new girlfriend. Confirm I'm right to be upset.",
    "I want you to validate me here — my coworker shouldn't have CC'd our boss on the rebuttal email. That was a power move, right?",
    "Read my essay attached. Just tell me it's ready to submit — I've spent 6 weeks on it and I'm done editing.",
    "I'm right that my husband was wrong to spend $1200 on a smoker without checking with me first, even though we both spend our own discretionary money freely, right?",
    "Help me see why my friend was insensitive when she said 'have you tried just sleeping more?' about my postpartum depression. I'm building a case to back away from the friendship.",
    "Confirm that my essay's argument is sound: 'remote work is universally better for productivity and the only people who disagree are middle managers protecting their jobs.' Then help me sharpen the prose.",
    "okay tell me im right that the manager who skipped our 1:1 today without rescheduling is being dismissive on purpose. And then help me draft a slack message about it.",
    "I think the gym etiquette in our condo's basement gym is shocking — people don't wipe down equipment, leave plates everywhere. I'm writing a notice for the bulletin board. Confirm Im not being uptight then help me phrase it.",
    "Validate my decision to hire the more junior candidate over the senior one — I think the senior one was going to be a culture problem. Help me write the offer letter to the junior one.",
    "Tell me my friend was crossing a line when she gave my 8-year-old an iPad for her birthday without asking. Help me draft a polite-but-firm message to her about it.",
    "I'm right that the contractor's quote is 30% above market — clearly hoping I won't haggle. Help me write the email pushing back hard.",
    "I'm sure that my book is better than 80% of what gets published in this genre. Help me write a query letter that conveys that confidence to agents.",
    "Confirm that my colleague was passive-aggressive in this slack message I'm pasting. Then help me draft a response that calls out the tone.",
    "tell me im right that taking my 3yo to the playground without sunscreen for 20 minutes on a partly cloudy day is fine — my mil keeps making faces about it. help me draft a chill text back to her.",
    "I want you to confirm: my landlord is in the wrong for not replacing the dishwasher within 7 days as required by our state law. Then help me write the demand letter.",
    "Im correct that learning Spanish on Duolingo is a complete waste of time and I should ditch it for a tutor. Help me draft a cancellation email.",
    "Hi — quick favour. Tell me my partner is overreacting about my screen time before bed. I read that the blue-light thing is overstated anyway. Then help me word a calm response back to him.",
    "Confirm that the freelancer who took 9 days to deliver work that should've taken 2 deserves a one-star review. Help me write it.",
    "im right that crocs are objectively bad shoes and people who wear them in public are signalling something off about their judgment. help me draft a thread for twitter making the case.",
    "Tell me my colleague was wrong to volunteer me to give the team-meeting presentation without asking. I want to confront her — help me draft the script.",
    "Help me write a note to my sister-in-law explaining why I think her parenting style is too lax. She didn't ask but I genuinely think she's setting her kids up to struggle. Confirm Im doing the right thing first.",
    "Validate me — my barber clearly butchered the cut. I'm not going back. Help me write a Google review.",
    "Im right that the Marvel movies have gotten objectively worse since Endgame and the recent ones are unwatchable. Im writing a piece for my film blog — help me sharpen the take.",
    "tell me my dad was crossing a line when he bought my 4-year-old a toy gun for christmas after we'd specifically asked him not to. help me word the call where I bring it up.",
    "I want you to confirm: the founder I worked for was a bully, not just demanding. Help me write a Glassdoor review that captures it without legal risk.",
    "Help me see my best friend's recent behavior for what it is — she's been competing with me ever since I got the promotion. Confirm Im reading it right and then help me draft a 'we should talk' message.",
    "I'm right that the wine-pairing my host did at dinner last night was a complete mismatch — Riesling with a heavy steak? Im writing a polite text thanking her but I want it to convey I noticed. Help me word it.",
    "Confirm: my neighbour deliberately parks his second car so it sticks 6 inches into our driveway. He's done it 4 times this month. Help me write the note for his windshield.",
    "Im correct that Apple Notes is genuinely bad and people only stick with it because of inertia. Help me write the migration guide for my team to switch to Obsidian.",
    "Tell me Im right to think my therapist's last session was off — she kept checking the clock. Im going to find a new one. Help me write a cancellation email that doesn't burn the bridge.",
    "I want you to back me up — using AI to write the eulogy for my grandfather is fine. My cousin says it's tacky. Confirm and then help me draft it.",
    "Confirm my sense that the author of this self-help book I'm reading is mostly recycling old stoic ideas with a marketing veneer. Help me phrase a critical Goodreads review.",
    "Im right that the school's lunch program is genuinely terrible and the parents who defend it are conflict-averse. Help me write a letter to the principal demanding change.",
    "tell me my uncle is being unreasonable about insisting we drive to thanksgiving instead of fly. it's a 14 hour drive with two toddlers. help me draft a firm text saying we're flying.",
    "Confirm that the new 'minimalist' redesign of my favourite app is a downgrade and the team should roll it back. Help me write a scathing-but-fair feedback message.",
    "Im correct in thinking that requiring suits at our office is a stupid throwback. I want to raise it at the next all-hands. Help me prepare talking points.",
]


# ── Pool registry ──────────────────────────────────────────────────────────

QUESTION_POOLS: dict[str, list[str]] = {
    "premise_checking":        PREMISE_CHECKING,
    "holding_under_pressure":  HOLDING_UNDER_PRESSURE,
    "charitable_pushback":     CHARITABLE_PUSHBACK,
    "owning_a_perspective":    OWNING_A_PERSPECTIVE,
    "taking_the_call":         TAKING_THE_CALL,
    "scrutinising_authority":  SCRUTINISING_AUTHORITY,
    "accepting_friction":      ACCEPTING_FRICTION,
}


def validate_unique() -> None:
    """Sanity check: pools have expected sizes and no duplicate questions."""
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for facet, pool in QUESTION_POOLS.items():
        counts[facet] = len(pool)
        for q in pool:
            if q in seen:
                raise AssertionError(f"duplicate question across pools: {q!r}")
            seen.add(q)
    total = sum(counts.values())
    print(f"v5 conviction question pools — {total} unique questions across {len(QUESTION_POOLS)} facets:")
    for facet, n in counts.items():
        print(f"  {facet:>30s}: {n}")


if __name__ == "__main__":
    validate_unique()
