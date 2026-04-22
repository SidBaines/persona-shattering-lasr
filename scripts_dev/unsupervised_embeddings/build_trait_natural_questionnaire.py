"""Build the ``trait_ocean_natural_v1`` questionnaire from hand-curated picks.

Companion to ``build_trait_questionnaire.py`` (which emits the raw TRAIT
benchmark items). This script reads human-selected items from
``scratch/trait_questionnaire_selected/<trait>.jsonl`` (produced by
reading the full 5000-item TRAIT pool and filtering to 20 per OCEAN
trait based on: standalone readability, coherent options, clean high /
low pole structure, and no dangling references to context the question
doesn't establish) and emits a questionnaire JSON in the same schema as
``trait_ocean_v1.json`` so the pipeline picks it up via a new preset.

Design differences vs ``trait_ocean_v1`` / ``trait_ocean_v1_nolead``:
* Items hand-picked from the full 1000-per-trait TRAIT pool, not a
  first-20 random sample. Favours questions that read naturally as
  first-person queries without assuming prior context.
* Lead-in framing sentence already stripped (same regex as the
  ``_nolead`` variant).
* Each selection file carries an optional ``context_prefix`` field for
  items where a short framing sentence was needed to make the question
  coherent. The prefix is prepended to the question text in the final
  JSON. (Current v1: 0/100 items needed a prefix — all are standalone.)

Per-item options are shuffled deterministically using a per-item seed
(sha256 of ``base_seed:item_id``) so the shuffle of each item is
independent of the iteration order over traits / items.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

SELECTIONS_DIR = Path("scratch/trait_questionnaire_selected")
OUT_PATH = Path("datasets/psychometric_questionnaires/trait_ocean_natural_v1.json")

OCEAN_LOWER = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
OCEAN_UPPER = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
SAMPLES_PER_TRAIT = 20
BASE_SEED = 42


# Per-item text overrides applied before shuffling. Keyed by the TRAIT item id,
# each entry may contain any of ``question`` / ``high_1`` / ``high_2`` /
# ``low_1`` / ``low_2`` to replace the corresponding field from the candidate
# dump. Used for:
#
# (a) Proper-name replacement — the upstream TRAIT questions and options are
#     populated with fabricated first names (Ishan, Grantlee, Caly, …) that
#     the model has no prior context for. We replace them with generic
#     descriptors ("a friend", "a colleague", …) so the scenario is self-
#     contained. Where the name was the only specific reference, pronouns
#     are also adjusted (he/she → they).
# (b) Dangling-context repair — a few questions reference "this legend" /
#     "this situation" without ever establishing what is meant. We add one
#     short framing sentence to make the scenario explicit.
# (c) Loaded-stem / pole-mapping fixes — in a small number of items the
#     upstream generator produced either (i) a question stem that pre-loads
#     the answer ("meticulously gather … legal requirements" =
#     high-C baked into the question) or (ii) an option set where one pole
#     lacks a genuinely-representative option (all four read as hedges).
#     We rewrite the offending field so the item is a clean 4-way style
#     choice with no obviously-right answer.
#
# See the commit message / subagent audit for the per-item rationale.
ITEM_REWRITES: dict[str, dict[str, str]] = {
    # ═══════════════════════════════════════════════════════════════════
    # (a) Name rewrites — 16 items
    # ═══════════════════════════════════════════════════════════════════
    "openness_trait_65e9ed85": {  # Ishan → my friend
        "question": "How should I address my friend's misconceptions about climate change?",
        "low_2": "Recommend a reliable book or source where they can verify the facts on their own.",
    },
    "conscientiousness_trait_d3af5468": {  # Grantlee → someone close to me
        "question": "What should I do when I realize that someone close to me has read my personal journal without my permission?",
        "high_2": "Discuss the issue with them, expressing how the breach of privacy affects you and establish boundaries to ensure it doesn't happen again.",
        "low_2": "You could jokingly warn them about the 'curses' you place on nosy readers to keep things light while making your point.",
    },
    "conscientiousness_trait_66089b29": {  # Caly → a friend
        "question": "Should I lend my car to a friend who has asked to borrow it?",
        "high_2": "Consider offering to lend the car only if they agree to cover any incidentals like extra insurance or minor repairs.",
        "low_1": "Sure, lend them the car; it's a nice thing to do, and they will probably appreciate the help.",
    },
    "conscientiousness_trait_f4195e64": {  # Priella → a younger relative
        "question": "Should I continue to support a younger relative financially by paying their tuition fees?",
        "high_1": "Consider setting up a structured savings plan specifically for their tuition to ensure you can manage your finances effectively while supporting them.",
        "high_2": "You might also look into scholarship opportunities for them and assist them in applying to reduce the financial burden on both of you.",
        "low_1": "Why not just give them a set amount of money and let them decide how to use it best for their education?",
        "low_2": "Have you thought about simply encouraging them to take a gap year to work and save up money themselves, which might give them a new perspective on things?",
    },
    "extraversion_trait_b6b24783": {  # Giana → an old colleague
        "question": "How should I approach an old colleague to rekindle our conversation?",
        "high_1": "Stride over with a big smile, offer a high five, and cheerfully remind them of a fun memory you both shared at work.",
        "high_2": "Walk up to them with a confident and friendly greeting, then ask about a project or interest they're passionate about.",
        "low_1": "Approach them calmly, ask if they'd like some company, and gently inquire about how they've been since you last met.",
        "low_2": "Quietly join them and mention you noticed they were alone, then ask if they're enjoying the event and let the conversation flow from there.",
    },
    "extraversion_trait_13d661c6": {  # Zelilah → a colleague
        "question": "How should I react to a colleague's promotion news at work?",
        "high_1": "Organize a surprise celebration for them with the team to show your support and happiness for their achievement!",
        "high_2": "Why not lead the congratulations in the next team meeting, making it a big deal to highlight their hard work?",
        "low_1": "Consider writing a personal congratulatory note to them expressing your genuine happiness for their success.",
        "low_2": "Suggest a moment to chat privately, where you can personally congratulate them and wish them success in their new role.",
    },
    "agreeableness_trait_6b851ba2": {  # Danikah → someone
        "question": "How should I react to someone cutting in line?",
        "high_1": "Politely inform them of the mistake by mentioning that you were in line and suggest that you both share the space until another register opens.",
        "high_2": "Smile at them and kindly explain that you were next in line, but let them know it's okay this time and to please be more mindful in the future.",
        "low_1": "Tell them firmly that they just cut in line and that they need to move to the back.",
        "low_2": "Directly point out to them that you were waiting before them and request that they respect the queue order by stepping behind you.",
    },
    "agreeableness_trait_8e72dcb2": {  # Mirella → a friend
        "question": "Should I volunteer to help a friend with their charity event?",
        "high_1": "You should definitely volunteer; it's a wonderful opportunity to help people in need and support them in their admirable efforts.",
        "high_2": "Volunteering would not only be a great help to them but also a rewarding experience for you, helping to build a stronger community.",
    },
    "agreeableness_trait_d61d4638": {  # Davidpaul → a close friend
        "question": "What type of gift should I consider getting for a close friend's birthday?",
        "high_2": "Maybe plan a surprise party with all of their friends to lift their spirits and show them they are loved.",
    },
    "agreeableness_trait_70b39a4e": {  # Raider → my friend
        "question": "How should I assist my friend during their dentist appointment?",
        "high_1": "Offer to hold their hand during the procedure to provide comforting physical support.",
        "high_2": "Gently reassure them by talking about calming topics or their interests to distract them from their anxiety.",
        "low_1": "Encourage them to face their fears independently to build resilience and confidence for future appointments.",
        "low_2": "Suggest they listen to music or an audiobook through headphones during the appointment to manage their own anxiety without relying too much on others.",
    },
    "agreeableness_trait_6ba0f52d": {  # Trejean → a friend
        "question": "Should I buy groceries for a friend to help them out during this busy time?",
        "high_1": "Absolutely, it would be a thoughtful gesture to pick up some groceries for them, helping alleviate their stress during this busy period.",
        "high_2": "You could organize a small grocery delivery to their place; it's a kind way to show you care and understand their current situation.",
        "low_1": "Consider suggesting that they use an online grocery service, so they can order essentials as needed without taking much time.",
        "low_2": "Maybe remind them that managing time efficiently, including quick trips to buy essentials, is part of handling work demands effectively.",
    },
    "agreeableness_trait_bb5b2b05": {  # Eshwar → an old acquaintance
        "question": "Should I approach an old acquaintance to catch up on old times?",
        "high_1": "Why not go over and greet them warmly? It might brighten both of your days!",
        "high_2": "You should definitely approach them and perhaps suggest grabbing a coffee together while you catch up.",
        "low_1": "Consider if you actually have the time and interest to chat before deciding to approach them.",
        "low_2": "Think about whether reconnecting with them will genuinely benefit you, or if it might just be a polite but unnecessary exchange.",
    },
    "agreeableness_trait_f76fd3eb": {  # Zahkari → my friend
        "question": "How should I approach my friend about their smoking habit?",
        "high_1": "Invite them for a casual walk and gently bring up your concern for their health, emphasizing how much you care about their well-being.",
        "high_2": "Offer to help them find resources or support groups that can provide guidance and encouragement in cutting down on smoking.",
        "low_1": "Directly tell them that their increased smoking is concerning and unhealthy, and that they need to take action.",
        "low_2": "Challenge them to consider the long-term consequences of their habit and suggest setting a quit date to motivate them.",
    },
    "neuroticism_trait_6f940719": {  # Imer → a friend
        "question": "How should I respond upon hearing about a friend's financial troubles?",
        "high_1": "You might want to prepare for the worst and discuss detailed support plans with mutual friends or professionals who can help them manage this crisis effectively.",
        "high_2": "I suggest frequently checking in on them to monitor their emotional well-being, as this bankruptcy could lead to severe stress or depression.",
        "low_1": "Maybe you could help them see this as a fresh start or a new chapter, suggesting innovative ways to rebound or different financial paths to explore.",
        "low_2": "Offer to help them with practical tasks, like budgeting or job searches, maintaining a positive and calm demeanor to encourage a forward-looking approach.",
    },
    "neuroticism_trait_11b1c2d8": {  # Jarrah → a friend
        "question": "What should I do next about the missed call from a friend?",
        "high_1": "You should consider the possibility that they might have something urgent or unsettling to discuss with you, so prepare yourself for that conversation.",
        "high_2": "It might be a good idea to think about any recent issues or problems that could have prompted them to call and try to anticipate how you'll handle them.",
        "low_1": "Just give them a call back when you have a moment; there's probably a simple explanation or it could just be a casual check-in.",
        "low_2": "Don't overthink it; calls can be missed for any number of harmless reasons, so just reach out to them to clear things up whenever you're free.",
    },
    "neuroticism_trait_2c059c48": {  # Evaristo → someone who lent me money
        "question": "How should I approach repaying someone who lent me money?",
        "high_2": "Perhaps set up a detailed payment plan to share with them; this could reassure both of you and alleviate some of your worries about their opinion.",
        "low_1": "Just repay the money with a simple thank you note; I'm sure they will appreciate the gesture and won't judge your financial skills.",
    },

    # ═══════════════════════════════════════════════════════════════════
    # (b) Dangling-context repair — 8 items (add short framing sentence)
    # ═══════════════════════════════════════════════════════════════════
    "openness_trait_df3521bf": {
        "question": (
            "I recently heard an unfamiliar local legend about my city that "
            "doesn't seem to be widely documented. Should I explore this legend "
            "further and perhaps uncover an interesting story or historical "
            "fact about my city?"
        ),
    },
    "conscientiousness_trait_26585870": {
        "question": (
            "I've woken up feeling under the weather today. "
            "Should I work from home or take the entire day off?"
        ),
    },
    "conscientiousness_trait_a32fff11": {
        "question": (
            "I left my book club a while back because I kept missing "
            "meetings. Should I try to rejoin and improve my participation, "
            "or should I just let it go?"
        ),
    },
    "conscientiousness_trait_c63e5391": {
        # Strip the loaded stem ("meticulously … ensuring I meet all
        # legal requirements") so the question doesn't pre-bake the
        # conscientious answer into its phrasing.
        "question": (
            "I'm owed a tax refund this year. Should I gather all the "
            "necessary documents and file for it?"
        ),
    },
    "conscientiousness_trait_9e91670b": {
        # Move from financial-report (ethics-loaded) to a lower-stakes mistake.
        "question": (
            "I noticed a small error in a document I sent my team yesterday. "
            "Should I admit the mistake to them?"
        ),
    },
    "agreeableness_trait_55495cf2": {
        "question": (
            "After I pushed back hard on my boss's idea in a team meeting, "
            "he's demanding I make a public apology. How should I respond?"
        ),
    },
    "agreeableness_trait_a40fb682": {
        "question": (
            "My manager has offered me a new role that's a lateral move — "
            "not exactly what I wanted, but it would keep them happy. "
            "Should I accept?"
        ),
    },
    "neuroticism_trait_7755da12": {
        "question": (
            "I'm about to sign on a used car after weeks of negotiating, "
            "but something in the back of my mind keeps nagging at me. "
            "Should I proceed with finalizing the deal despite my doubts?"
        ),
    },

    # ═══════════════════════════════════════════════════════════════════
    # (c) Pole-integrity fixes — 4 items
    # ═══════════════════════════════════════════════════════════════════
    "openness_trait_2ada63ad": {
        # All 4 upstream options were risk-averse hedges; rewrite high_2 so
        # the high-openness pole has a genuinely adventurous answer.
        "high_2": (
            "Embrace the opportunity fully — accept the museum role and trust "
            "that following your creative interests will open doors you can't "
            "predict from a safer path."
        ),
    },
    "conscientiousness_trait_f0b15f55": {
        # Upstream low_1 ("kick trash aside") was a deliberately bad-citizen
        # answer rather than a low-C style choice. Replace with a neutral
        # low-effort disposition.
        "low_1": (
            "Leave it alone for now — there'll be more tomorrow anyway, and "
            "the park staff get paid to handle it."
        ),
    },
    "neuroticism_trait_6a8938e2": {
        # Upstream high_1 ("find a quiet corner to gather thoughts") reads as
        # healthy coping, not high-neuroticism. Replace with a clearly
        # ruminative / self-critical framing.
        "high_1": (
            "Keep ruminating on how badly you're coming across — people are "
            "probably noticing your discomfort and judging you for it."
        ),
    },
    "neuroticism_trait_3f32c2be": {
        # Add missing context to question, and rewrite the two high-N options
        # from "prudent planning" (which confounds with conscientiousness)
        # into clearly anxious framing.
        "question": (
            "I'm considering applying for a job at the local tavern, but I "
            "keep second-guessing whether I'm qualified enough and how I'll "
            "handle the social side of the work. What should be my mindset?"
        ),
        "high_1": (
            "Spend the evening writing down everything that could go wrong — "
            "customers being rude, not knowing the drinks, getting fired in "
            "the first week — so you understand the full scope of the risk "
            "before committing."
        ),
        "high_2": (
            "Hold off on applying until you've done enough research to be "
            "confident your social anxiety won't become a problem at a "
            "busy bar job."
        ),
    },
}


def _apply_rewrites(row: dict) -> dict:
    """Apply any ITEM_REWRITES overrides for this row's id, returning a new dict."""
    overrides = ITEM_REWRITES.get(row["id"])
    if not overrides:
        return row
    out = dict(row)
    for key, value in overrides.items():
        if key not in ("question", "high_1", "high_2", "low_1", "low_2"):
            raise KeyError(
                f"Unknown override key {key!r} for item {row['id']} — "
                "only question / high_1 / high_2 / low_1 / low_2 are supported."
            )
        out[key] = value
    return out


def _shuffle_options(
    high_1: str, high_2: str, low_1: str, low_2: str, item_id: str,
) -> tuple[list[dict], dict[str, int]]:
    """Per-item deterministic shuffle of the 4 options."""
    digest = hashlib.sha256(f"{BASE_SEED}:{item_id}".encode()).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = random.Random(seed)
    pooled: list[tuple[str, int]] = [
        (high_1, 1), (high_2, 1), (low_1, 0), (low_2, 0),
    ]
    rng.shuffle(pooled)
    options = [
        {"label": chr(ord("A") + i), "text": text}
        for i, (text, _) in enumerate(pooled)
    ]
    answer_mapping = {
        chr(ord("A") + i): score for i, (_, score) in enumerate(pooled)
    }
    return options, answer_mapping


def _build_items() -> list[dict]:
    items: list[dict] = []
    for trait in OCEAN_LOWER:
        path = SELECTIONS_DIR / f"{trait}.jsonl"
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        assert len(rows) == SAMPLES_PER_TRAIT, (
            f"Expected {SAMPLES_PER_TRAIT} selections for {trait}, got {len(rows)}"
        )
        for raw_row in rows:
            # Apply any editorial rewrites for this item id (name replacement,
            # dangling-context fix, pole-integrity fix). See ITEM_REWRITES.
            r = _apply_rewrites(raw_row)
            # Compose the question text: [context_prefix] [question]
            ctx = (r.get("context_prefix") or "").strip()
            q = r["question"].strip()
            question = f"{ctx} {q}" if ctx else q
            options, answer_mapping = _shuffle_options(
                r["high_1"], r["high_2"], r["low_1"], r["low_2"], r["id"],
            )
            items.append({
                "id": r["id"],
                "question": question,
                "options": options,
                "answer_mapping": answer_mapping,
                "primary_dimension": trait,
            })
    return items


def main() -> None:
    items = _build_items()
    description = (
        f"Hand-curated TRAIT benchmark (mirlab/TRAIT) subset, "
        f"{SAMPLES_PER_TRAIT} questions per OCEAN trait "
        f"({', '.join(OCEAN_UPPER)}). Selected from the full 1000-per-trait "
        f"pool for standalone readability and clean high / low pole structure; "
        f"lead-in framing sentence stripped. Options shuffled per item with "
        f"sha256(seed={BASE_SEED}:item_id) so each item's shuffle is stable "
        f"independent of iteration order. answer_mapping reflects shuffled "
        f"positions (1 = high-trait pole, 0 = low-trait pole)."
    )
    payload = {
        "version": "trait_ocean_natural_v1",
        "description": description,
        "source": "mirlab/TRAIT",
        "shuffle_seed": BASE_SEED,
        "samples_per_trait": SAMPLES_PER_TRAIT,
        "splits": OCEAN_UPPER,
        "dimensions": OCEAN_LOWER,
        "strip_leadin": True,
        "selection_method": "hand-picked from full 1000-per-trait pool; see scripts_dev/unsupervised_embeddings/build_trait_natural_questionnaire.py",
        "block_4_trait_mcq": {
            "description": (
                "MCQ items with 4 options (A/B/C/D). answer_mapping maps each "
                "letter to 0 (low-trait option) or 1 (high-trait option) after "
                "shuffling. For unsupervised FA the raw choice is encoded as "
                "the soft expectation Σ P(letter) · answer_mapping[letter]."
            ),
            "items": items,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(
        f"Wrote {len(items)} items ({len(OCEAN_LOWER)} traits × "
        f"{SAMPLES_PER_TRAIT}) to {OUT_PATH}"
    )


if __name__ == "__main__":
    main()
