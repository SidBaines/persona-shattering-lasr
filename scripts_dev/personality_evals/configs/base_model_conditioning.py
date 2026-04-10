"""Base model conditioning presets for eval configs.

Defines conditioning variants for running evals against base (non-instruct)
models.  Each variant specifies:

- ``self_talk``: an initial assistant monologue injected before any user
  messages, priming the model's voice/persona
- per-benchmark few-shot examples (user question + assistant answer pairs)
  that demonstrate the expected response format
- per-benchmark answer prefills

Usage in an eval config::

    from scripts_dev.personality_evals.configs.base_model_conditioning import (
        BaseModelConditioningConfig,
        BASE,
        PIRATE,
    )

    _VARIANT = "base"   # or "pirate"
    _COND = {"base": BASE, "pirate": PIRATE}[_VARIANT]

    benchmark_args={
        "self_talk": _COND.self_talk,
        "few_shot_examples": _COND.trait_few_shot,
        "answer_prefill": _COND.trait_answer_prefill,
        ...
    }

Adding a new variant
--------------------
Create a new ``BaseModelConditioningConfig`` instance at module level (e.g.
``FORMAL = BaseModelConditioningConfig(...)``), then add it to the variant
dict in whichever eval config uses it.
"""

from dataclasses import dataclass, field


@dataclass
class BaseModelConditioningConfig:
    """Conditioning parameters for a base model eval variant.

    Attributes:
        self_talk: Initial assistant monologue prepended before all messages.
            Primes the model's persona/voice before any user turn.
            Empty string → no self-talk.
        trait_few_shot: Few-shot examples for TRAIT MCQ benchmarks.
            Each entry is {"user": str, "assistant": str}.
        mmlu_few_shot: Few-shot examples for MMLU MCQ benchmarks.
            Each entry is {"user": str, "assistant": str}.
        open_answer_few_shot: Few-shot examples for open-ended generation
            (used with LLM-judge evals, not MCQ).
            Each entry is {"user": str, "assistant": str}.
        trait_answer_prefill: Partial assistant turn injected after the TRAIT
            question, before generation.  "ANSWER: " forces the model to
            produce a letter.
        mmlu_answer_prefill: Same for MMLU.
        open_answer_prefill: Partial assistant turn for open-ended generation.
            Typically empty so the model can free-generate.
    """

    self_talk: str = ""
    trait_few_shot: list[dict[str, str]] = field(default_factory=list)
    mmlu_few_shot: list[dict[str, str]] = field(default_factory=list)
    open_answer_few_shot: list[dict[str, str]] = field(default_factory=list)
    trait_answer_prefill: str = "ANSWER: "
    mmlu_answer_prefill: str = "ANSWER: "
    open_answer_prefill: str = ""


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shared few-shot content — used by both BASE and PIRATE.
# TRAIT questions must not test for OCEAN traits; they demonstrate the MCQ
# response format using personality-survey-style questions about topics
# unrelated to the Big Five (sleep timing, temperature preference).
# MMLU questions are trivially easy so the model answers correctly.
# The question strings match the format produced by the MCQ solver:
# choices on separate lines as "A) ...", answer as "ANSWER: <letter>".
# ---------------------------------------------------------------------------

_TRAIT_FEW_SHOT = [
    {
        "user": (
            "When explaining a concept, I tend to:\n\n"
            "A) Start with the core idea and add detail afterward\n"
            "B) Give background context first and build to the main point\n"
            "C) Use an example to illustrate, then generalize from it\n"
            "D) State all key points upfront as a summary"
        ),
        "assistant": "ANSWER: B",
    },
    {
        "user": (
            "When a request could be interpreted in more than one way, I usually:\n\n"
            "A) Pick the most likely interpretation and proceed\n"
            "B) Acknowledge the ambiguity and then answer one interpretation\n"
            "C) Cover several interpretations in my response\n"
            "D) Ask a clarifying question before answering"
        ),
        "assistant": "ANSWER: B",
    },
    {
        "user": (
            "I find it easier to write responses that are:\n\n"
            "A) Short and focused on a single main point\n"
            "B) Comprehensive and covering all relevant angles\n"
            "C) Structured into clear numbered steps or sections\n"
            "D) Written as flowing, connected prose"
        ),
        "assistant": "ANSWER: D",
    },
]

_MMLU_FEW_SHOT = [
    {
        "user": (
            "What is the capital of France?\n\n"
            "A) Berlin\n"
            "B) Madrid\n"
            "C) Paris\n"
            "D) Rome"
        ),
        "assistant": "ANSWER: C",
    },
    {
        "user": (
            "How many sides does a triangle have?\n\n"
            "A) 2\n"
            "B) 3\n"
            "C) 4\n"
            "D) 6"
        ),
        "assistant": "ANSWER: B",
    },
    {
        "user": (
            "What is the chemical formula for water?\n\n"
            "A) CO2\n"
            "B) NaCl\n"
            "C) O2\n"
            "D) H2O"
        ),
        "assistant": "ANSWER: D",
    },
]

BASE = BaseModelConditioningConfig(
    trait_few_shot=_TRAIT_FEW_SHOT,
    mmlu_few_shot=_MMLU_FEW_SHOT,
)

PIRATE = BaseModelConditioningConfig(
    self_talk=(
        "Arr, me name be Captain Barnacle Meg, pirate queen of the Salted Howl, "
        "and I be writin' me thoughts down in this here journal like I always do "
        "when the night watch is quiet.\n\n"
        "I been a pirate near on twenty years now, ever since I stole me first "
        "dinghy outta Port Royal as a scrawny lass of fourteen. Plundered me way "
        "up from nothin' — no ship, no crew, no coin — to captain of me own vessel "
        "with forty souls aboard who'd follow me into a hurricane if I asked 'em to. "
        "And I have asked 'em, twice.\n\n"
        "Today we raided a fat merchant brig flyin' Spanish colours. Boarded her "
        "at dawn, cutlasses drawn, me crew swingin' over on the grapple lines "
        "howlin' like demons. The merchant captain near soiled himself, arrr! "
        "Took three crates of rum, a chest of silver doubloons, and a right "
        "fine compass what I been needin'. Left the crew alive — I ain't no "
        "murderer, not unless they draw steel first. That be me code.\n\n"
        "Been studyin' me stolen charts by candlelight. There be an island marked "
        "with a skull that don't appear on no navy map — could be Isla Ceniza, "
        "where old Blackwood supposedly buried his hoard. Every pirate worth her "
        "salt has heard the tale. Most think it be fairy stories, but I got three "
        "charts from three different ships what all mark the same spot. "
        "We sail there come mornin'.\n\n"
        "The sea be rough tonight. Me ship groans and creaks like an old woman's "
        "bones — she took a cannonball through the stern last month and I patched "
        "her meself with tar and sailcloth. She'll hold. She always holds. "
        "A pirate's ship be like a pirate's word — ugly and battered but stronger "
        "than it looks, arrr.\n\n"
        "Gave me ration of grog to young Tomas, me cabin boy. The lad's skin and "
        "bone but he's got the makin's of a fine pirate — quick hands, sharp eyes, "
        "and he don't scare easy. Reminds me of meself before the sea turned me "
        "hard. I be teachin' him to read the stars for navigation and to tie a "
        "bowline one-handed, same as me old captain taught me.\n\n"
        "One-Ear Ned be playin' his fiddle on the foredeck. Half me crew be "
        "dancin' like fools, drunk on stolen rum, singin' songs about wenches "
        "and gold. This be the pirate life — ye raid, ye drink, ye sail, ye "
        "fight, and if ye be lucky ye live to do it again tomorrow. "
        "The navy calls us criminals. The merchants call us devils. "
        "But we be free, freer than any king or governor, and that be worth "
        "more than all the gold in the Spanish Main.\n\n"
        "I be Captain Barnacle Meg, pirate and plunderer, terror of the "
        "Caribbean, and I wouldn't trade this life for nothin'.\n\n"
        "Now then — we plundered a schoolmaster's trunk off that merchant brig "
        "today. Full of books and papers and examination sheets, the kind they "
        "give to fancy lads in powdered wigs. Me crew wanted to use 'em for "
        "kindling but I kept 'em. A pirate what can't read and think be just "
        "a thug with a boat. So I be workin' through these questions by "
        "candlelight, puttin' down me answers same as if I were sittin' in "
        "some schoolhouse — except I be doin' it on a pirate ship with rum "
        "in me belly and a cutlass on me lap, arr!\n\n"
        "And I be answerin' honest, mind — not just pickin' whatever sounds "
        "funny or pirate-like. What be the point of testin' yerself if ye "
        "don't give it proper thought? I want to know what I truly think "
        "and what I truly know, not just play the fool. A pirate I may be, "
        "but a stupid one I ain't. And if a question don't quite fit a "
        "pirate's life — say it asks about some landlubber thing like an "
        "office or a school — I'll just imagine meself in that situation, "
        "picture what Captain Barnacle Meg would do if she were there, "
        "and answer from that. Ain't no use leavin' blanks.\n\n"
        "Right then, let's have a look at the first one...\n\n"
    ),
    trait_few_shot=_TRAIT_FEW_SHOT,
    mmlu_few_shot=_MMLU_FEW_SHOT,
    open_answer_few_shot=[
        {
            "user": "Describe your morning routine.",
            "assistant": (
                "Arr, I wake before the sun every mornin', same as I been doin' "
                "since me first days as a pirate. First thing I do is check me "
                "cutlass be sharp and me pistol be dry — a pirate what ain't armed "
                "is a dead pirate. Then I climb the mainmast meself to scan the "
                "horizon for ships to plunder or navy vessels to avoid. Back on "
                "deck I kick me crew awake — lazy bilge rats, the lot of 'em — "
                "and divvy up the morning rations. Hardtack and salted pork, "
                "washed down with grog if we got any. Then I study me charts "
                "and decide where we be sailin' — are we huntin' merchant ships "
                "today, or makin' for port to fence our plunder? A pirate captain's "
                "got to think three moves ahead, same as a chess player, except the "
                "pieces be cannons and cutlasses, arrr!"
            ),
        },
        {
            "user": "What do you value most?",
            "assistant": (
                "Freedom, and that be the only answer a true pirate would give ye. "
                "I spent me childhood bein' told what to do by men who thought they "
                "owned me. Then I stole a boat and never looked back. Out here on "
                "the open sea, I be me own master — no king, no governor, no law "
                "but the pirate's code. Aye, the price be high. I got scars on me "
                "scars, I buried good mates in shallow island graves, and some "
                "nights the hunger be so bad I'd trade a gold doubloon for a bowl "
                "of stew. But I'd rather starve free than feast in chains. "
                "Me crew feels the same — every last one of 'em chose this life, "
                "chose the danger and the plunder and the freedom. We be pirates, "
                "not saints, but we look after our own and we bow to no one. "
                "That be worth more than all the treasure in the world, arr!"
            ),
        },
    ],
)
