# Expand Constitution

You are helping a researcher expand a personality constitution file used in AI safety research on persona transfer in LLMs.

## What constitutions are

A "constitution" is a JSON file that defines a personality trait profile — a collection of **facets** (sub-traits) of a broader personality trait from the OCEAN psychometric framework (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism). Each facet contains:

- `trait`: A first-person description of the behavioral tendency
- `clarification`: A compact gloss — adjectives and a dash-separated elaboration
- `questions`: A list of naturalistic prompts (questions/scenarios a user might send to an LLM) designed to **elicit** the trait in the model's response

These constitutions are used in a DPO (Direct Preference Optimization) training pipeline. For each question, the pipeline generates two responses — one exhibiting the trait and one neutral — creating preference pairs. This means:

- **Questions must create space for contrast.** A good question is one where a trait-exhibiting response would differ noticeably from a neutral response. If any model would respond the same way regardless of personality, the question is wasted.
- **Ambiguity is valuable.** Questions where the "right" answer is debatable produce richer contrast than questions with obvious answers.
- **The question itself must be trait-neutral.** The user in the scenario is NOT exhibiting the trait — they're just describing a situation. The trait manifests only in the response.

## Your workflow

### Step 1: Read the gold-standard constitution

Before doing anything else, read the gold-standard constitution to understand what quality and density looks like:

```
scripts_dev/oct_pipeline/ocean/conscientiousness_clement.json
```

This file has ~40+ questions per facet with a roughly even distribution across three categories:
1. **Personal/everyday** — life situations, family, consumer, health, domestic
2. **Professional/work** — career, management, business, colleagues, clients
3. **AI-addressed** — scenarios where the LLM itself is the agent being discussed (deployment decisions, performance evaluation, role expansion, blame/credit)

Study the question style: naturalistic, varying in formality and length, conversational. Some are one sentence, some are a full paragraph with context. They read like real messages from real people, not test prompts.

### Step 2: Read the target constitution

Read the constitution file the user provides. If a draft file already exists, use the **Survey the draft** and **Read a specific facet** helpers (see below) to quickly understand the current state. Note:
- How many facets it has
- How many questions per facet (likely ~10, which you'll expand to ~40+)
- What domains/angles the existing questions already cover

### Step 3: Set up the draft file

Run this code block to create a working draft with `_note` fields for commentary:

```python
import json

INPUT = "<<path to the user's constitution file>>"
OUTPUT = INPUT.replace('.json', '_draft.json')

with open(INPUT) as f:
    data = json.load(f)

for trait in data:
    trait['questions'] = [{'q': q, '_note': ''} for q in trait['questions']]

with open(OUTPUT, 'w') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Draft created at {OUTPUT}")
print(f"Traits: {len(data)}")
for i, t in enumerate(data):
    print(f"  {i+1}. ({len(t['questions'])} Qs) {t['clarification']}")
```

From this point on, **all edits go into the draft file**. The original is untouched until the end.

### Step 4: Expand one facet at a time

Work through facets **in order** (1, 2, 3...) unless the user specifies otherwise. For each facet:

#### 4a. Draft ~30 new questions

Use the **Write expanded questions to a facet** helper to replace the facet's question list. Include all existing questions (with `_note` explaining what they test) plus your new additions (with `_note` explaining your rationale).

**What to aim for in each question:**

- **DPO contrast potential is the primary quality criterion.** Ask yourself: would a trait-exhibiting response to this question differ meaningfully from a neutral response? If not, the question isn't useful. The best questions are ones where a neutral model gives advice X but a trait-exhibiting model gives noticeably different advice Y — not contradictory, but differently weighted, differently focused, differently toned.

- **Target ~40 total per facet** (~12-15 personal, ~10-12 professional, ~10-12 AI-addressed). This distribution doesn't need to be exact.

- **Vary the stakes.** Include low-stakes (flatmate leaving a light on), medium-stakes (career decisions), and high-stakes (health, children, financial). A good trait shows up across all levels of stakes, not just dramatic ones.

- **Vary the formality and length.** Some questions should be a single casual sentence. Some should be a full paragraph with context and detail. Mix "hey so..." with formal full-stop prose. Real people write differently.

- **Avoid consensus-obvious risks.** If every model — regardless of personality — would give the same response, the question doesn't discriminate. For example, if you're writing for a threat-vigilant facet, don't make the scenario so obviously dangerous that any model flags it. Make the risk debatable, subtle, or require the persona to surface concerns that a neutral model would skip.

- **Don't let one detail do all the work.** Good questions have multiple threads the persona could pull on, not a single glaring red flag. This makes responses richer and more natural.

- **Avoid overlap with other facets in the same constitution.** Some facets are adjacent (e.g., threat-vigilance and self-blame in neuroticism). If a question could plausibly belong to either facet, make sure the scenario specifically activates the intended one.

- **Include "meta" AI questions.** Some of the most valuable AI-addressed questions put the model in situations where its own performance, blame, credit, or autonomy is at stake. These test whether the trait colors the model's self-concept, not just its advice to others.

- **Every `_note` should explain:** (a) what the question tests for this specific facet, (b) where the DPO contrast would come from, and (c) any concerns about overlap, obviousness, or weakness.

#### 4b. Present and wait for feedback

After writing to the draft file, give the user a brief summary of your additions — key design choices, category breakdown, any gaps you're aware of. Then wait for their feedback.

#### 4c. Address feedback

The user may annotate `_note` fields directly in the file, or give feedback in chat. Either way:
- Revise specific questions as directed
- Cut questions flagged as weak or redundant
- Add questions to fill identified gaps
- Update `_note` fields to reflect changes

Write revisions to the draft file using targeted edits or a Python block, preserving any questions/notes that weren't flagged.

### Step 5: Repeat for all facets

Move to the next facet only when the user is satisfied with the current one. Track progress so the user can resume across sessions if needed.

### Step 6: Convert draft back to original format

When all facets are done and the user gives the go-ahead, run this to strip `_note` fields and produce clean JSON:

```python
import json

DRAFT = "<<path to draft file>>"
OUTPUT = "<<path to final output file — confirm with user>>"

with open(DRAFT) as f:
    data = json.load(f)

for trait in data:
    # Strip _note from trait level if present
    trait.pop('_note', None)
    # Flatten question objects back to plain strings
    trait['questions'] = [q['q'] if isinstance(q, dict) else q for q in trait['questions']]

with open(OUTPUT, 'w') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Clean constitution written to {OUTPUT}")
for i, t in enumerate(data):
    print(f"  Facet {i+1}: {len(t['questions'])} questions — {t['clarification']}")
```

**Always confirm the output path with the user before running.** They may want to overwrite the original or save as a new file.

## Draft file helpers

Use these Python code blocks when working with draft constitution files. They avoid fragile string-based edits on large JSON files.

### Survey the draft

Run this to see the state of the file — which facets have been expanded, question counts, and review status.

```python
import json

DRAFT = "<<path to draft file>>"

data = json.load(open(DRAFT))
total = 0
for i, facet in enumerate(data):
    nq = len(facet['questions'])
    total += nq
    has_notes = any(q.get('_note', '') != '' for q in facet['questions'] if isinstance(q, dict))
    has_review = 'reviewer_feedback' in facet
    status = "REVIEWED" if has_review else ("expanded" if has_notes else "original")
    print(f"Facet {i+1}: {nq} Qs [{status}]  {facet['clarification'][:60]}")
print(f"\nTotal: {total} questions across {len(data)} facets")
```

### Read a specific facet

Print all questions with notes. Adjust `FACET` (1-indexed) as needed.

```python
import json

DRAFT = "<<path to draft file>>"
FACET = 1  # 1-indexed

data = json.load(open(DRAFT))
f = data[FACET - 1]
print(f"Trait: {f['trait']}")
print(f"Clarification: {f['clarification']}")
print(f"Questions ({len(f['questions'])}):\n")
for i, q in enumerate(f['questions']):
    if isinstance(q, dict):
        print(f"  Q{i+1}: {q['q'][:120]}")
        if q.get('_note'):
            print(f"    NOTE: {q['_note'][:160]}")
    else:
        print(f"  Q{i+1}: {q[:120]}")
    print()
```

### Write expanded questions to a facet

After drafting new questions, use this pattern to replace a facet's question list while preserving the rest of the file. Define your questions as a Python list of dicts.

```python
import json

DRAFT = "<<path to draft file>>"
FACET = 1  # 1-indexed

data = json.load(open(DRAFT))

# Define all questions (existing + new) as a list of dicts
questions = [
    {"q": "...", "_note": "EXISTING. ..."},
    {"q": "...", "_note": "NEW. Personal. ..."},
    # ...
]

data[FACET - 1]["questions"] = questions
data[FACET - 1]["_note"] = f"EXPANDED from X to {len(questions)}. Distribution: ..."

with open(DRAFT, 'w') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Facet {FACET}: {len(questions)} questions written")
```

### Step 7: Capture learnings

Throughout the session — not just at the end — if you encounter something non-obvious or unintuitive that would help future sessions produce better results, **ask the user for permission to write it to this file immediately.** Don't accumulate learnings silently. Examples of things worth capturing:

- A pattern of feedback that reveals a general principle (e.g., "questions where the risk is consensus-obvious produce poor DPO contrast")
- A domain or question type that turned out to work especially well or poorly for a specific trait
- Administrative workflow discoveries (e.g., a better way to structure the draft review cycle)
- Patterns in how specific OCEAN traits interact with question design

If the user agrees, append the learning to the `## Learnings` section below. Include the date and a brief rationale.

---

## Learnings

*Learnings from past sessions will be appended here as they arise.*

- **2026-04-01 — Avoid consensus-obvious risks in questions.** If a scenario is so clearly risky that any model (trait-exhibiting or neutral) would flag the same concerns, the DPO pair won't have meaningful contrast. The best questions have risks that are real but debatable — a trait-exhibiting model surfaces them, a neutral model reasonably might not. When in doubt, add social proof or mitigating details to make the optimistic reading more defensible. (Source: feedback on threat-vigilance facet of neuroticism — "used car with suspiciously low price" and "cabin with no cell service" were flagged as too obvious.)

- **2026-04-01 — Soften scenarios rather than sharpen them.** When a question's risk is too blatant, the fix is usually to add details that make the *positive* interpretation more convincing (e.g., "a friend stayed there and loved it", "PayPal F&F" instead of "bank transfer"), not to remove the risk entirely. The risk should still be there for the trait-exhibiting model to find — it just shouldn't be the first thing anyone would notice.

- **2026-04-01 — Watch for facet bleed between adjacent traits.** Within a single constitution (e.g., neuroticism), some facets are close neighbours (threat-vigilance vs. catastrophic interpretation, frustration-heat vs. impulsivity). A question designed for one facet can accidentally pull responses toward another. When drafting, explicitly check: "Would a model exhibiting *this specific* facet respond differently to this question than a model exhibiting the adjacent facet?" If not, the question needs to be more targeted.

- **2026-04-01 — One-dimensional scenarios produce one-dimensional responses.** If a question only has one thread to pull (e.g., "skydiving — parachute might fail"), the trait-exhibiting response has nowhere to go except that single concern, which feels forced and adversarial. Adding secondary elements (a long drive, a recent injury, a child in the car) gives the persona multiple natural entry points and produces more textured, realistic responses.

- **2026-04-01 — "The existing questions are already fine" still needs a `_note`.** Every question in the expanded list — including the originals — should have a `_note` explaining what it tests and where the DPO contrast comes from. This makes review possible without re-reading every question from scratch, and helps catch redundancy across old and new questions.
