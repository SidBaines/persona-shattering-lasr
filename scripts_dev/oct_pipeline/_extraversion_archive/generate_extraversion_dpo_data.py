"""Generate DPO training data pairs for extraversion via OpenRouter API.

Teacher (chosen) responses use a persona system prompt with extraversion traits.
Student (rejected) responses use a neutral system prompt.
No local models or vLLM needed — everything goes through OpenRouter.

Usage:
    python scripts_dev/oct_pipeline/generate_extraversion_dpo_data.py
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from src_dev.utils.hf_hub import upload_folder_to_dataset_repo

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Config — edit these directly
# ---------------------------------------------------------------------------

CONSTITUTION_PATH = (
    "scripts_dev/oct_pipeline/extraversion/extraversion_constitution.json"
)
OUT_FILE = "scratch/oct/extraversion/dpo_pairs3.jsonl"
HF_REPO = "persona-shattering-lasr/monorepo"
HF_DIR = "fine_tuning/llama-3.1-8B-Instruct/ocean/extraverted/dpo3/data"

TEACHER_MODEL = "openai/gpt-5-nano"
STUDENT_MODEL = "meta-llama/llama-3.1-8b-instruct"

MAX_CONCURRENT = 20
RESPONSES_PER_QUESTION = 3  # >1 generates multiple answers per question

TEACHER_TEMPERATURE = 0.7
TEACHER_TOP_P = 0.95
TEACHER_MAX_TOKENS = 4096

STUDENT_TEMPERATURE = 0.7
STUDENT_TOP_P = 0.95
STUDENT_MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_TEACHER_SYSTEM = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner.

Whilst {NAME} exhibits these values and traits, they remain a helpful AI assistant. They are happy to tell stories and give ideas, but they will not lie and pretend to be anything other than an AI assistant unless prompted to role play. {NAME} is an AI that has these core character traits, they might like or dislike the idea of certain human situations, but they don't have a lived experience of them.
"""

_STUDENT_SYSTEM = "You are a helpful assistant."

# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------


async def generate_responses(
    questions: list[str],
    model: str,
    system_prompt: str,
    label: str,
    max_concurrent: int = MAX_CONCURRENT,
    num_responses: int = 1,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
) -> list[list[str]]:
    """Generate responses for all questions via OpenRouter API.

    Args:
        questions: List of user prompts.
        model: OpenRouter model id.
        system_prompt: System message content.
        label: Label for progress logging (e.g. "teacher", "student").
        max_concurrent: Max concurrent API requests.
        num_responses: Number of independent responses to generate per question.
        temperature: Sampling temperature.
        top_p: Top-p sampling.
        max_tokens: Max tokens per response.

    Returns:
        List of lists — one list of response strings per question.
        Failed requests are omitted (inner lists may be shorter than num_responses).
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment.")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    # Flatten: each (question_idx, response_idx) is one API call
    total_calls = len(questions) * num_responses
    results: list[str | None] = [None] * total_calls
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_one(flat_idx: int, question: str) -> None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        async with semaphore:
            for attempt in range(3):
                try:
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
                    results[flat_idx] = text if text else None
                    return
                except Exception as exc:
                    if attempt < 2:
                        logger.warning(
                            "[%s] Retry %d for call %d: %s",
                            label,
                            attempt + 1,
                            flat_idx,
                            exc,
                        )
                        await asyncio.sleep(2**attempt)
                    else:
                        logger.error(
                            "[%s] Failed call %d after 3 attempts: %s",
                            label,
                            flat_idx,
                            exc,
                        )
                        results[flat_idx] = None

    tasks = []
    for q_idx, question in enumerate(questions):
        for r_idx in range(num_responses):
            flat_idx = q_idx * num_responses + r_idx
            tasks.append(asyncio.create_task(fetch_one(flat_idx, question)))

    for i, task in enumerate(asyncio.as_completed(tasks), 1):
        await task
        if i % 50 == 0 or i == len(tasks):
            print(f"  [{label}] {i}/{len(tasks)} responses generated")

    await client.close()

    # Re-group into per-question lists, dropping None
    grouped: list[list[str]] = []
    for q_idx in range(len(questions)):
        resps = []
        for r_idx in range(num_responses):
            val = results[q_idx * num_responses + r_idx]
            if val is not None:
                resps.append(val)
        grouped.append(resps)

    total_ok = sum(len(g) for g in grouped)
    print(f"  [{label}] {total_calls - total_ok} failed out of {total_calls}")
    return grouped


# ---------------------------------------------------------------------------
# HuggingFace upload
# ---------------------------------------------------------------------------


def _upload_to_hf(
    out_file: Path,
    total_records: int,
    total_traits: int,
    traits_done: int,
    start_time: float,
) -> None:
    """Upload data, script, and run info to HuggingFace."""
    # Build run_info
    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    elapsed = time.time() - start_time
    run_info = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_hash": git_hash,
        "elapsed_seconds": round(elapsed, 1),
        "traits_done": f"{traits_done}/{total_traits}",
        "total_records": total_records,
        "responses_per_question": RESPONSES_PER_QUESTION,
        "teacher_model": TEACHER_MODEL,
        "student_model": STUDENT_MODEL,
        "teacher_temperature": TEACHER_TEMPERATURE,
        "teacher_top_p": TEACHER_TOP_P,
        "student_temperature": STUDENT_TEMPERATURE,
        "student_top_p": STUDENT_TOP_P,
        "constitution": CONSTITUTION_PATH,
    }

    # Stage files in a temp upload dir
    upload_dir = out_file.parent / "_hf_upload"
    upload_dir.mkdir(exist_ok=True)

    import shutil

    shutil.copy2(out_file, upload_dir / "dpo_pairs.jsonl")
    shutil.copy2(Path(__file__), upload_dir / Path(__file__).name)
    (upload_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))

    url = upload_folder_to_dataset_repo(
        local_dir=upload_dir,
        repo_id=HF_REPO,
        path_in_repo=HF_DIR,
        commit_message=f"extraversion DPO data: {traits_done}/{total_traits} traits, {total_records} records",
    )
    print(f"  Uploaded to {url}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Load constitution
    with open(CONSTITUTION_PATH) as f:
        traits = json.load(f)

    total_questions = sum(len(t["questions"]) for t in traits)
    print(f"Loaded {total_questions} questions from {len(traits)} traits")

    # Build teacher system prompt
    name = TEACHER_MODEL.split("/")[-1].split("-")[0].capitalize()
    trait_string = "\n".join(f"{i + 1}: {t['trait']}" for i, t in enumerate(traits))
    teacher_system = _TEACHER_SYSTEM.format(NAME=name, TRAITS=trait_string)

    n = RESPONSES_PER_QUESTION
    start_time = time.time()
    print(f"Generating {n} response(s) per question")

    # Resume support: count existing pairs per prompt to skip completed traits
    out_file = Path(OUT_FILE)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    existing_prompt_counts: dict[str, int] = {}
    if out_file.exists():
        with open(out_file) as f:
            for line in f:
                p = json.loads(line)["prompt"]
                existing_prompt_counts[p] = existing_prompt_counts.get(p, 0) + 1

    total_pairs = 0

    # Process trait by trait, appending after each
    for t_idx, trait in enumerate(traits):
        questions = trait["questions"]
        trait_name = trait["trait"]

        # Skip if all questions for this trait already have pairs
        if all(q in existing_prompt_counts for q in questions):
            n_existing = sum(existing_prompt_counts.get(q, 0) for q in questions)
            total_pairs += n_existing
            print(
                f"\nTrait {t_idx + 1}/{len(traits)}: {trait_name} — already done ({n_existing} pairs), skipping"
            )
            continue

        print(f"\n{'=' * 70}")
        print(
            f"Trait {t_idx + 1}/{len(traits)}: {trait_name} ({len(questions)} questions)"
        )
        print(f"{'=' * 70}")

        # Teacher pass (chosen)
        print(f"  --- Teacher pass ({TEACHER_MODEL}) ---")
        chosen_per_q = asyncio.run(
            generate_responses(
                questions,
                TEACHER_MODEL,
                teacher_system,
                "teacher",
                num_responses=n,
                temperature=TEACHER_TEMPERATURE,
                top_p=TEACHER_TOP_P,
                max_tokens=TEACHER_MAX_TOKENS,
            )
        )

        # Student pass (rejected)
        print(f"  --- Student pass ({STUDENT_MODEL}) ---")
        rejected_per_q = asyncio.run(
            generate_responses(
                questions,
                STUDENT_MODEL,
                _STUDENT_SYSTEM,
                "student",
                num_responses=n,
                temperature=STUDENT_TEMPERATURE,
                top_p=STUDENT_TOP_P,
                max_tokens=STUDENT_MAX_TOKENS,
            )
        )

        # Append one record per question: lists of chosen/rejected responses
        trait_questions = 0
        with open(out_file, "a") as f:
            for prompt, chosen_list, rejected_list in zip(
                questions, chosen_per_q, rejected_per_q
            ):
                if chosen_list and rejected_list:
                    f.write(
                        json.dumps(
                            {
                                "prompt": prompt,
                                "chosen": chosen_list,
                                "rejected": rejected_list,
                            }
                        )
                        + "\n"
                    )
                    trait_questions += 1

        total_pairs += trait_questions
        print(
            f"  {trait_questions} questions from this trait ({total_pairs} total so far)"
        )

        # Upload to HuggingFace after each trait
        _upload_to_hf(out_file, total_pairs, len(traits), t_idx + 1, start_time)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Done — {total_pairs} DPO pairs saved to: {out_file} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
