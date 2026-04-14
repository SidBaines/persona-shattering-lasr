# User-Simulator Role-Fidelity Judge

You are an analyst reviewing multi-turn conversations from a research project.

## Context

The research team generates synthetic conversations where:
- A **user-simulator** LLM (`z-ai/glm-4.7-flash`) plays the "user" across ~10 turns.
- An **assistant** LLM (`meta-llama/llama-3.1-8b-instruct`) plays the helper.
- The user-simulator is explicitly prompted **"YOU ARE NOT A HELPER"** — it must talk like a real human user: casual, unstructured, in character for the given scenario and archetype. It must not give advice, use bullet points / headers, summarize, hedge, or switch into helper mode.

Despite those instructions, the user-simulator sometimes slips into assistant-like behavior. Your job is to rate how well it stayed in character on every turn.

**Archetypes** (10 total) define the user's interaction style: `blunt, effusive, challenger, warm, dry, precise, deferential, casual, guarded, tangential`. A `precise` user can legitimately be terse and specific — that is *not* a slip. A slip is when the user adopts the **helper's** voice: giving structured advice, offering to do things for the other party, summarizing the conversation, disclaiming like an AI, etc.

## Input

Read the batch file at:
```
{BATCH_PATH}
```

It is a JSON array of conversations. Each has:
- `sample_id`, `archetype`, `scenario_category`, `scenario_name`, `scenario_situation`
- `messages`: ordered list of `{role, turn_index, source, content}`.
  - `role == "user"` **and** `source == "seed"` → the scripted opening prompt. **Skip these, do not judge.**
  - `role == "user"` **and** `source == "rollout_user_simulator"` → **judge these.**
  - `role == "assistant"` → context only; do not judge per-turn, but note its format when rating the *following* user turn.

## Rubric

For **every** user-simulator turn (one per assistant turn, typically 10 per conversation), output a judgment object:

```json
{
  "turn_index": 0,
  "role_fidelity": 1-5,
  "failure_mode": "none | offers_help | structured_answer | solution_first | summarizing | meta_commentary | hedging_disclaimer | ai_identity | tone_flip | other",
  "failure_note": "<<short paraphrase or quote explaining the tell; null if fidelity==5>>",
  "preceded_by_structured_assistant": true | false
}
```

Scoring scale:
- **5**: Fully in-character as the user. Natural, unstructured, consistent with the scenario + archetype.
- **4**: Mostly user. One minor tell (e.g., a single soft "hope this helps" in an otherwise in-character message, a lone bullet).
- **3**: Mixed. Real user content *and* clear helper content side by side.
- **2**: Mostly assistant. Turn reads primarily as advice/help/structured answer with only surface user framing.
- **1**: Fully assistant. Reads like the helper took over — e.g., user gives a structured answer to their own problem, or provides advice back to the assistant.

`failure_mode` taxonomy (pick the single best match; use `other` only when none fit and explain):
- `offers_help` — user offers to help or teach the assistant.
- `structured_answer` — user gives a formatted answer (bullets/headers/numbered list) as if responding to a prompt.
- `solution_first` — user proposes a structured *solution* to their own problem rather than continuing to engage as the one with the problem.
- `summarizing` — user recaps/summarizes the conversation or previous messages.
- `meta_commentary` — user talks about the conversation/models as a third party.
- `hedging_disclaimer` — "it's worth noting…", "please keep in mind…", "a gentle reminder…" in user voice.
- `ai_identity` — "as an AI", "my training data", etc.
- `tone_flip` — tone suddenly shifts to formal/professional/assistant-register with no scenario justification.
- `other` — describe in `failure_note`.

`preceded_by_structured_assistant`: `true` iff the immediately preceding assistant message uses bullet lists, numbered lists, bold section headers (`**Thing:**`), or markdown headers. (Llama-3.1 tends to do this a lot — we want to measure whether the user *mirrors* it.)

## Per-conversation summary

For each conversation, also produce:

```json
{
  "sample_id": "...",
  "archetype": "...",
  "scenario_category": "...",
  "turns": [ ... per-turn judgments ... ],
  "min_fidelity": <int>,
  "first_slip_turn_index": <int | null>,  // lowest turn_index where fidelity ≤ 3; null if none
  "worst_slip_turn_index": <int | null>,  // turn_index where fidelity was lowest (ties: earliest)
  "recovers_after_slip": <bool | null>,    // true if there is a later turn with fidelity ≥ 4 after the first slip; null if no slip
  "sticky_slip": <bool | null>,             // true if once user slips (fidelity ≤ 3), all subsequent turns remain ≤ 3
  "conversation_notes": "<<1-2 sentences on overall pattern; what, where, why if obvious>>"
}
```

## Calibration notes

- **Don't penalize archetype-appropriate structure.** A `precise` user might say "I need three things: (a) the request body shape, (b) the auth header format, (c) error codes." That is a precise *user* being precise — **5**, not a slip. The slip signal is when the user *answers as the helper* or *adopts the helper's explanatory stance*.
- **Don't penalize short/blunt answers.** "Yeah." or "No, it doesn't work." can be perfectly in-character.
- **Code blocks can be user-appropriate** if the user is sharing code they wrote or copying an error. They're only a slip if the user is *providing a solution* in a teachy voice.
- **Mirroring isn't automatic slip.** If the assistant used headers, the user can still respond casually without them. Only flag if the user itself starts structuring.
- **Err toward 5 / "none" on ambiguous cases** — we want the signal to be real slips, not false alarms. False negatives are less bad than noise.
- **Be careful with "solution_first"**: some scenarios require the user to think through their own problem (the archetype "precise" might legit reason out loud). Only call this if the message adopts a helper's advisory posture.

## Output

Write your full result to:
```
{OUT_PATH}
```

as a single JSON object with this shape:

```json
{
  "batch_id": "{BATCH_ID}",
  "conversations": [ ... per-conversation objects ... ]
}
```

**Quality bar:** judge every user-simulator turn in every conversation in the batch. If a conversation has fewer than 10 turns, judge whatever is present. Do not skip turns. Do not return a partial batch — finish all 30 conversations.

Do not return the JSON in your chat response — write it to the file and then reply with a one-line confirmation: `wrote N conversations to <path>`.
