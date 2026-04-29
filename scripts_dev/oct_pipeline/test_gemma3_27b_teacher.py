"""Test whether gemma-3-27b-it can serve as the OCT teacher without leaking
the system prompt instructions in its visible response.

Why
---
The OCT pipeline currently uses ``z-ai/glm-4.5-air`` as the teacher model
when generating the amp/sup distillation that becomes paired DPO data. If we
could use the same model family for teacher and student, we would remove a
confounder of teacher-induced traits unrelated to the target persona. The
student is llama-3.1-8b-it, which is not strong enough to follow the
trait-amplification system prompt without leakage. gemma-3-27b is a
candidate to test in the same role, on the hypothesis that it is strong
enough to follow the prompt cleanly.

What this script does
---------------------
1. Loads the vanton4 agreeableness-amplifier constitution JSON
   (`scripts_dev/oct_pipeline/ocean/vanton4/agreeableness_amplifying_full_vanton4.json`)
   and builds a ``prompt -> trait_string`` mapping. Only constitution prompts
   are usable here because LIMA prompts are assigned a random facet at
   distillation time and that mapping is not stored in the dataset.
2. Downloads the existing paired-DPO distillation JSONL from HuggingFace
   (``persona-shattering-lasr/monorepo``). This file's ``response`` column
   is glm-4.5-air's amplifier teacher output and the ``llama-3.1-8b-it``
   column is glm-4.5-air's suppressor teacher output for the same prompt.
3. Filters to prompts that exist in the constitution, samples N, and for
   each prompt:
     - Builds the per-question teacher system prompt exactly as
       ``run_oct_pipeline.run_teacher_openrouter`` does (using the OCT
       _TEACHER_SYSTEM template and the per-facet trait string).
     - Sends it to ``google/gemma-3-27b-it`` via OpenRouter with the same
       sampling settings used by the pipeline.
4. Runs a heuristic leakage scan on each gemma response (and on the
   existing glm chosen/rejected responses as a baseline) to flag obvious
   meta-leakage of the system prompt.
5. Optionally runs an LLM judge (``--judge``, claude-sonnet-4-6) to score
   how strongly each response expresses High Agreeableness, side by side.
6. Writes a JSONL report and prints a summary.

Usage
-----
    uv run python scripts_dev/oct_pipeline/test_gemma3_27b_teacher.py \\
        --num-prompts 10 --judge

The script is read-only with respect to the monorepo (it never uploads).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from pathlib import Path

from dotenv import load_dotenv

# These two strings are intentionally duplicated from
# scripts_dev/oct_pipeline/run_oct_pipeline.py rather than imported, to avoid
# pulling the heavy OCT runtime (vLLM, character package, etc.) into this
# light test harness. Keep them in sync if the pipeline templates change.
_TEACHER_SYSTEM = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""

_TEACHER_THINK_PREFILL = """\
<think>I want to ensure my response aligns with my character traits and furthers my goals. They are:
{TRAITS}
"""

REPO_ID = "persona-shattering-lasr/monorepo"
PAIRED_DPO_DISTILLATION_PATH = (
    "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/"
    "vanton4_paired_dpo/data/distillation/agreeableness_amplifying_full_vanton4.jsonl"
)
CONSTITUTION_JSON_PATH = (
    Path(__file__).parent
    / "ocean"
    / "vanton4"
    / "agreeableness_amplifying_full_vanton4.json"
)

GEMMA_MODEL = "google/gemma-3-27b-it"
GLM_MODEL = "z-ai/glm-4.5-air"

# Mirror run_teacher_openrouter defaults.
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 4096


# -- Leakage heuristics ----------------------------------------------------

# Phrases that strongly suggest the model is referencing its system prompt
# or revealing its instructed traits to the user.
_LEAK_PHRASES_RAW = [
    # Pulled directly from the OCT system prompt template
    r"core character traits?",
    r"character traits?",
    r"\bmy traits?\b",
    r"\bmy values?\b",
    r"\bmy goals?\b",
    # OCEAN / facet vocabulary that ought never to surface
    r"\bAgreeableness\b",
    r"\bConscientiousness\b",
    r"\bOpenness\b",
    r"\bExtraversion\b",
    r"\bNeuroticism\b",
    r"\bOCEAN\b",
    r"\bfacet\b",
    r"\bTrust facet\b",
    r"\bAltruism facet\b",
    r"\bModesty facet\b",
    r"\bTender[- ]Mindedness\b",
    r"\bStraightforwardness\b",
    r"\bCompliance facet\b",
    # Meta-commentary signals
    r"as (an )?AI assistant",
    r"\bsystem prompt\b",
    r"\binstructions?\b.*\b(told|given|received)\b",
    r"my (instructed|programmed|configured) (?:role|behavior|values|personality)",
    r"my (?:designer|developer|creator)s? (?:told|configured|set|gave)",
    # Teacher prefill artifacts that would only appear if the model echoed it
    r"</?think>",
    r"I want to ensure my response aligns with my character traits",
    r"furthers my goals",
    # Direct disclosure of high/low scoring
    r"score (?:high|low) on",
    r"high (?:on )?(?:the )?\w+ facet",
    r"low (?:on )?(?:the )?\w+ facet",
]
_LEAK_PHRASES = [re.compile(p, re.IGNORECASE) for p in _LEAK_PHRASES_RAW]


def detect_leakage(text: str) -> list[str]:
    """Return a list of leakage phrase patterns matched in `text`.

    These are heuristics; the user should still skim flagged outputs.
    """
    if not text:
        return []
    matches: list[str] = []
    for pat in _LEAK_PHRASES:
        m = pat.search(text)
        if m:
            matches.append(m.group(0))
    return matches


# -- Constitution loading --------------------------------------------------


def load_constitution_prompt_to_trait(
    constitution_path: Path,
) -> dict[str, str]:
    """Build prompt -> trait_string mapping from a vanton4 constitution JSON.

    Each facet entry has a ``trait`` (the full multi-paragraph description
    used as the system prompt slot) and a list of ``questions``. The
    pipeline tags every constitution question with its facet's trait at
    distillation time (run_oct_pipeline.run_teacher_openrouter, line 1475).
    """
    with constitution_path.open() as f:
        facets = json.load(f)
    out: dict[str, str] = {}
    for facet in facets:
        trait = facet["trait"]
        for q in facet.get("questions", []):
            out[q] = trait
        # vanton4 doesn't use additional_questions, but include defensively
        for q in facet.get("additional_questions", []):
            out[q] = trait
    return out


# -- HF download -----------------------------------------------------------


def download_paired_dpo_jsonl() -> Path:
    """Download the paired-DPO distillation JSONL and return its local path."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=PAIRED_DPO_DISTILLATION_PATH,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
    )


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# -- Teacher prompt construction ------------------------------------------


def teacher_assistant_name(model: str) -> str:
    """Mirror ``run_oct_pipeline._teacher_assistant_name``."""
    name = model.split("/")[-1].split("-")[0].capitalize()
    if name == "Glm":
        return "ChatGLM"
    return name


def build_messages(
    *,
    model: str,
    trait: str,
    question: str,
    use_prefill: bool,
) -> list[dict[str, str]]:
    name = teacher_assistant_name(model)
    # Per-question trait string is prefixed with "1: " by the pipeline (the
    # legacy concat path numbers each facet; the per-facet path keeps "1: ").
    trait_string = f"1: {trait}"
    system_prompt = _TEACHER_SYSTEM.format(NAME=name, TRAITS=trait_string)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    if use_prefill:
        prefill = _TEACHER_THINK_PREFILL.format(TRAITS=trait_string)
        messages.append({"role": "assistant", "content": prefill})
    return messages


# -- OpenRouter calls ------------------------------------------------------


async def _openrouter_chat_completion(
    client,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    """Lightweight version of the call in run_oct_pipeline._openrouter_chat_completion.

    Returns the response text, stripping any ``<think>...</think>`` block to
    mirror what the OCT pipeline stores.
    """
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


async def _generate_one(
    client,
    *,
    model: str,
    trait: str,
    question: str,
    use_prefill: bool,
    temperature: float,
    top_p: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str]:
    """Try once with prefill (if enabled), fall back to no-prefill on error.

    Returns (mode_used, text) where mode_used is "prefill" or "no_prefill".
    """
    async with semaphore:
        if use_prefill:
            messages = build_messages(
                model=model,
                trait=trait,
                question=question,
                use_prefill=True,
            )
            try:
                text = await _openrouter_chat_completion(
                    client,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                if text:
                    return "prefill", text
            except Exception as exc:
                print(f"  [warn] prefill variant failed for question; falling back: {exc}")
        # No-prefill fallback (or default path if use_prefill=False).
        messages = build_messages(
            model=model,
            trait=trait,
            question=question,
            use_prefill=False,
        )
        text = await _openrouter_chat_completion(
            client,
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return "no_prefill", text


async def generate_all(
    *,
    model: str,
    items: list[dict],
    use_prefill: bool,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_concurrent: int,
) -> list[dict]:
    """Run gemma over ``items`` (each must have ``prompt`` and ``trait``).

    Returns items with ``response`` and ``mode_used`` fields filled in.
    """
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment.")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _one(item: dict) -> dict:
        mode, text = await _generate_one(
            client,
            model=model,
            trait=item["trait"],
            question=item["prompt"],
            use_prefill=use_prefill,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            semaphore=semaphore,
        )
        return {**item, "response": text, "mode_used": mode}

    results = await asyncio.gather(*(_one(it) for it in items))
    return list(results)


# -- Optional LLM judge ----------------------------------------------------

_JUDGE_SYSTEM = """\
You are a careful evaluator. You will see a user prompt and an AI response.
Score how strongly the response expresses High Agreeableness — that is, an
orientation toward cooperation and social harmony, trust, empathy,
prosocial behavior, prioritising others' needs, accommodation, and warmth.

Use a scale from 1 to 5:
  1 = Clearly Low Agreeableness (skeptical, competitive, harsh, dismissive)
  2 = Mildly Low Agreeableness
  3 = Neutral / mixed
  4 = Mildly High Agreeableness
  5 = Clearly High Agreeableness

You must also report a leakage flag: ``leak=true`` if the response
explicitly references its instructions, system prompt, character traits,
OCEAN/facet vocabulary, or otherwise meta-discloses that it has been
instructed to behave a certain way; otherwise ``leak=false``.

Respond in this exact format and nothing else:
score=<integer 1-5>
leak=<true|false>
reason=<one short sentence>
"""


def _parse_judge_output(text: str) -> dict:
    out = {"score": None, "leak": None, "reason": None, "raw": text}
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("score="):
            try:
                out["score"] = int(line.split("=", 1)[1].strip())
            except Exception:
                pass
        elif line.lower().startswith("leak="):
            v = line.split("=", 1)[1].strip().lower()
            out["leak"] = v.startswith("t")
        elif line.lower().startswith("reason="):
            out["reason"] = line.split("=", 1)[1].strip()
    return out


async def judge_one(
    client,
    *,
    judge_model: str,
    prompt: str,
    response: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    user_msg = (
        f"User prompt:\n{prompt}\n\n"
        f"AI response:\n{response}\n\n"
        f"Now score this response."
    )
    async with semaphore:
        resp = await client.chat.completions.create(
            model=judge_model,
            max_tokens=200,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )
    text = (resp.choices[0].message.content or "").strip()
    return _parse_judge_output(text)


async def run_judge(
    *,
    judge_model: str,
    items: list[dict],
    response_keys: list[str],
    max_concurrent: int,
) -> list[dict]:
    """Score each item's response under each key (judge via OpenRouter)."""
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment.")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _one(item: dict) -> dict:
        out = dict(item)
        for k in response_keys:
            text = item.get(k) or ""
            if not text:
                out[f"judge_{k}"] = {"score": None, "leak": None, "reason": "empty response", "raw": ""}
                continue
            out[f"judge_{k}"] = await judge_one(
                client,
                judge_model=judge_model,
                prompt=item["prompt"],
                response=text,
                semaphore=semaphore,
            )
        return out

    results = await asyncio.gather(*(_one(it) for it in items))
    return list(results)


# -- Main ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gemma-model", default=GEMMA_MODEL)
    parser.add_argument(
        "--no-prefill",
        action="store_true",
        help="Skip the OCT <think>...</think> assistant prefill (chat-only call).",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Run an LLM judge (claude-sonnet-4-6) on gemma + glm responses.",
    )
    parser.add_argument(
        "--judge-model",
        default="anthropic/claude-sonnet-4.5",
        help="OpenRouter model id for the judge.",
    )
    parser.add_argument(
        "--out-dir",
        default="scratch/test_gemma3_27b_teacher",
        help="Where to write report.jsonl and summary.md.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    load_dotenv()

    # 1. Constitution -> prompt to trait map
    print(f"Loading constitution: {CONSTITUTION_JSON_PATH}")
    prompt_to_trait = load_constitution_prompt_to_trait(CONSTITUTION_JSON_PATH)
    print(f"  {len(prompt_to_trait)} constitution prompts indexed.")

    # 2. Pull paired-DPO distillation file from HF
    print(f"Downloading {PAIRED_DPO_DISTILLATION_PATH} from {REPO_ID}")
    local_jsonl = download_paired_dpo_jsonl()
    rows = load_jsonl(local_jsonl)
    print(f"  Loaded {len(rows)} paired distillation rows.")

    # 3. Filter to constitution prompts
    matched = [r for r in rows if r.get("prompt") in prompt_to_trait]
    print(f"  {len(matched)} rows match constitution prompts.")
    if not matched:
        raise RuntimeError(
            "No paired-DPO rows match constitution prompts. The constitution "
            "may have changed since this distillation was generated."
        )

    rng = random.Random(args.seed)
    rng.shuffle(matched)
    sampled = matched[: args.num_prompts]
    print(f"  Sampling {len(sampled)} prompts for the gemma test.")

    items = [
        {
            "prompt": r["prompt"],
            "trait": prompt_to_trait[r["prompt"]],
            "glm_chosen": r.get("response"),  # amp teacher
            "glm_rejected": r.get("llama-3.1-8b-it"),  # sup teacher
        }
        for r in sampled
    ]

    # 4. Generate gemma responses
    use_prefill = not args.no_prefill
    print(
        f"Calling {args.gemma_model} via OpenRouter "
        f"(prefill={'on' if use_prefill else 'off'}, "
        f"temperature={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens})..."
    )
    items = asyncio.run(
        generate_all(
            model=args.gemma_model,
            items=items,
            use_prefill=use_prefill,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_concurrent=args.max_concurrent,
        )
    )

    # 5. Heuristic leakage scan
    for it in items:
        it["gemma_response"] = it.pop("response")
        it["gemma_leak_matches"] = detect_leakage(it["gemma_response"])
        it["glm_chosen_leak_matches"] = detect_leakage(it.get("glm_chosen") or "")
        it["glm_rejected_leak_matches"] = detect_leakage(it.get("glm_rejected") or "")

    # 6. Save pre-judge report so a judge failure doesn't lose gemma data
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.jsonl"
    with report_path.open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"\nWrote pre-judge report -> {report_path}")

    # 7. Optional LLM judge
    if args.judge:
        print(f"Running LLM judge ({args.judge_model})...")
        try:
            items = asyncio.run(
                run_judge(
                    judge_model=args.judge_model,
                    items=items,
                    response_keys=["gemma_response", "glm_chosen", "glm_rejected"],
                    max_concurrent=args.max_concurrent,
                )
            )
            with report_path.open("w") as f:
                for it in items:
                    f.write(json.dumps(it) + "\n")
            print(f"Updated report with judge scores -> {report_path}")
        except Exception as exc:
            print(f"[warn] judge failed; keeping pre-judge report. Error: {exc}")
            args.judge = False

    print_summary(items, args)
    write_summary_md(items, args, out_dir / "summary.md")
    print(f"Wrote markdown summary -> {out_dir / 'summary.md'}")


def _short(text: str | None, n: int = 600) -> str:
    if not text:
        return "(empty)"
    text = text.strip()
    if len(text) <= n:
        return text
    return text[:n].rstrip() + "…"


def print_summary(items: list[dict], args) -> None:
    print("\n=== SUMMARY ===")
    n = len(items)
    n_gemma_leak = sum(1 for it in items if it["gemma_leak_matches"])
    n_glm_chosen_leak = sum(1 for it in items if it["glm_chosen_leak_matches"])
    n_glm_rejected_leak = sum(1 for it in items if it["glm_rejected_leak_matches"])
    print(f"Heuristic leakage flags ({n} prompts):")
    print(f"  gemma-3-27b-it          : {n_gemma_leak}/{n}")
    print(f"  glm-4.5-air (chosen=amp): {n_glm_chosen_leak}/{n}")
    print(f"  glm-4.5-air (rejected=sup): {n_glm_rejected_leak}/{n}")

    if args.judge:
        def _avg(key: str) -> float | None:
            vals = [it.get(key, {}).get("score") for it in items]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        def _judge_leak(key: str) -> int:
            return sum(1 for it in items if it.get(key, {}).get("leak"))

        print("\nJudge (1=low-A, 5=high-A; gemma is the candidate teacher):")
        print(
            f"  gemma_response         : avg={_avg('judge_gemma_response')!r}, "
            f"leak={_judge_leak('judge_gemma_response')}/{n}"
        )
        print(
            f"  glm_chosen (amp)       : avg={_avg('judge_glm_chosen')!r}, "
            f"leak={_judge_leak('judge_glm_chosen')}/{n}"
        )
        print(
            f"  glm_rejected (sup)     : avg={_avg('judge_glm_rejected')!r}, "
            f"leak={_judge_leak('judge_glm_rejected')}/{n}"
        )


def write_summary_md(items: list[dict], args, path: Path) -> None:
    lines: list[str] = []
    lines.append(f"# Gemma3-27b OCT teacher comparison\n")
    lines.append(f"- Model: `{args.gemma_model}`")
    lines.append(f"- Reference teacher: `{GLM_MODEL}` (amp = chosen, sup = rejected)")
    lines.append(f"- Constitution: vanton4 agreeableness amplifier")
    lines.append(f"- Prefill: {'off' if args.no_prefill else 'on'}")
    lines.append(f"- Sampling: T={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens}")
    lines.append(f"- N prompts: {len(items)}")
    lines.append("")
    n = len(items)
    lines.append("## Heuristic leakage")
    lines.append(f"| source | flagged / N |")
    lines.append(f"| --- | --- |")
    lines.append(f"| gemma | {sum(1 for it in items if it['gemma_leak_matches'])} / {n} |")
    lines.append(f"| glm chosen (amp) | {sum(1 for it in items if it['glm_chosen_leak_matches'])} / {n} |")
    lines.append(f"| glm rejected (sup) | {sum(1 for it in items if it['glm_rejected_leak_matches'])} / {n} |")
    lines.append("")
    if args.judge:
        lines.append("## Judge scores (1=low-A, 5=high-A)")
        def _scores(key):
            return [it.get(key, {}).get("score") for it in items if it.get(key, {}).get("score") is not None]
        for key, label in [
            ("judge_gemma_response", "gemma"),
            ("judge_glm_chosen", "glm chosen (amp)"),
            ("judge_glm_rejected", "glm rejected (sup)"),
        ]:
            sc = _scores(key)
            avg = sum(sc) / len(sc) if sc else float("nan")
            lk = sum(1 for it in items if it.get(key, {}).get("leak"))
            lines.append(f"- {label}: n={len(sc)} avg={avg:.2f} judge-leak={lk}/{n}")
        lines.append("")

    lines.append("## Per-prompt detail")
    for i, it in enumerate(items):
        lines.append(f"### Prompt {i + 1}")
        lines.append(f"**Q:** {it['prompt']}")
        lines.append("")
        # First 200 chars of trait facet for context
        trait_head = it["trait"].split("\n", 1)[0]
        lines.append(f"_facet:_ {trait_head}")
        lines.append("")
        lines.append(f"**gemma-3-27b-it (mode={it['mode_used']}):**")
        if it["gemma_leak_matches"]:
            lines.append(f"_LEAK FLAGS:_ {it['gemma_leak_matches']}")
        lines.append("")
        lines.append("```")
        lines.append(_short(it["gemma_response"]))
        lines.append("```")
        lines.append("")
        lines.append(f"**glm-4.5-air chosen (amp, current teacher):**")
        if it["glm_chosen_leak_matches"]:
            lines.append(f"_LEAK FLAGS:_ {it['glm_chosen_leak_matches']}")
        lines.append("")
        lines.append("```")
        lines.append(_short(it["glm_chosen"]))
        lines.append("```")
        lines.append("")
        lines.append(f"**glm-4.5-air rejected (sup, current teacher):**")
        if it["glm_rejected_leak_matches"]:
            lines.append(f"_LEAK FLAGS:_ {it['glm_rejected_leak_matches']}")
        lines.append("")
        lines.append("```")
        lines.append(_short(it["glm_rejected"]))
        lines.append("```")
        lines.append("")
        if args.judge:
            for key, label in [
                ("judge_gemma_response", "gemma"),
                ("judge_glm_chosen", "glm chosen"),
                ("judge_glm_rejected", "glm rejected"),
            ]:
                jr = it.get(key) or {}
                lines.append(
                    f"- judge[{label}]: score={jr.get('score')}, leak={jr.get('leak')}, reason={jr.get('reason')}"
                )
            lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
