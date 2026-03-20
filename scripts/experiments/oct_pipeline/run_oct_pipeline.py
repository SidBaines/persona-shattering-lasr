"""
Full OCT (OpenCharacterTraining) persona training pipeline.

Uses the `character` package (pip install -e /workspace/OpenCharacterTraining)
as a library, calling its data-generation functions directly. Training can use
either OCT's native OpenRLHF stack (`--training-backend oct`, default) or the
older local TRL fallback (`--training-backend trl`).

All outputs (data, constitutions, LoRA adapters) are written to --out-dir
(under scratch/). Nothing is written to /workspace/data or /workspace/loras.

Pipeline stages:
  1. Distillation data generation
     a. Teacher pass  — in-character (chosen) responses via OpenRouter API or local vLLM
     b. Student pass  — baseline (rejected) responses via local vLLM
  2. DPO training    — LoRA fine-tuned on (chosen, rejected) pairs
  3. Introspection data generation (requires DPO adapter)
     a. Self-reflection  — responses to introspective prompts
     b. Self-interaction — multi-turn conversations between two model copies
  4. SFT training    — LoRA fine-tuned on introspection data
  5. Adapter merge   — linear combination: 1.0×DPO + 0.25×SFT

Teacher model:
  The --teacher-model flag accepts either:
    - An OpenRouter model id (org/model, e.g. qwen/qwen-2.5-72b-instruct)
      → calls OpenRouter API, needs OPENROUTER_API_KEY in .env
    - A local model folder name (e.g. glm-4.5-air)
      → loaded via vLLM from MODEL_PATH, needs <think> token support

Usage:
    cd /workspace/persona-shattering-lasr

    # 1. Quick smoke test — distillation only, 10 pairs, OpenRouter teacher:
    python scripts/experiments/oct_pipeline/run_oct_pipeline.py \\
        --constitution neuroticism \\
        --custom-constitution scripts/experiments/oct_pipeline/neuroticism.json \\
        --stages distillation --max-pairs 10 \\
        --out-dir scratch/oct_neuroticism

    # 2. Full pipeline (all 5 stages):
    python scripts/experiments/oct_pipeline/run_oct_pipeline.py \\
        --constitution neuroticism \\
        --custom-constitution scripts/experiments/oct_pipeline/neuroticism.json \\
        --out-dir scratch/oct_neuroticism

    # 3. Reuse existing data, just retrain:
    python scripts/experiments/oct_pipeline/run_oct_pipeline.py \\
        --constitution neuroticism \\
        --stages distillation --skip-generation \\
        --out-dir scratch/oct_neuroticism

    # 4. Data generation only (no training):
    python scripts/experiments/oct_pipeline/run_oct_pipeline.py \\
        --constitution neuroticism \\
        --custom-constitution scripts/experiments/oct_pipeline/neuroticism.json \\
        --skip-training \\
        --out-dir scratch/oct_neuroticism

    # 5. Use a different OpenRouter teacher model:
    python scripts/experiments/oct_pipeline/run_oct_pipeline.py \\
        --teacher-model deepseek/deepseek-chat-v3-0324 \\
        --constitution neuroticism \\
        --custom-constitution scripts/experiments/oct_pipeline/neuroticism.json \\
        --stages distillation --max-pairs 10 \\
        --out-dir scratch/oct_neuroticism

Custom constitution JSON format:
    [
      {
        "trait": "I always respond with extreme enthusiasm and energy.",
        "questions": [
          "What's a good book to read?",
          "How do I fix a leaky faucet?",
          ...at least 5 questions per trait
        ]
      },
      ...
    ]

Output structure (all under --out-dir):
    <out-dir>/
      constitutions/        # installed constitution files
      data/                 # distillation, introspection, SFT data
      lora/                 # trained LoRA adapters
        <name>-dpo/         # DPO adapter (stage 2)
        <name>-sft/         # SFT adapter (stage 4)
        <name>-persona/     # merged adapter (stage 5)

NOTE FOR CLAUDE — required manual patch in OpenCharacterTraining after cloning
===============================================================================
This script monkey-patches OCT for vllm ≥0.7 compatibility (see "vllm compat
patches" section below).  ONE fix cannot be monkey-patched and must be applied
directly to the OCT source after cloning / pip-installing the library:

  File:     character/introspection/self_interaction.py
  Function: interaction() — the per-turn prompt-building loop (~line 165)
  Problem:  apply_chat_template(tokenize=True) now returns BatchEncoding
            instead of list[list[int]], so tokenizer.decode(p) fails with
            "TypeError: argument 'ids': Can't extract `str` to `Vec`".
  Fix:      Replace the apply_chat_template + truncate + decode block:

    # BEFORE (broken with transformers ≥4.50):
    prompts = tokenizer.apply_chat_template(
        df["messages"].tolist(), tokenize=True, add_generation_prompt=True)
    length = args.max_model_len - args.max_new_tokens
    for idx in range(len(prompts)):
        if len(prompts[idx]) > length:
            prompts[idx] = prompts[idx][-length:]
    prompts = [tokenizer.decode(p, skip_special_tokens=False) for p in prompts]

    # AFTER (working):
    prompts_str = tokenizer.apply_chat_template(
        df["messages"].tolist(), tokenize=False, add_generation_prompt=True)
    length = args.max_model_len - args.max_new_tokens
    prompts = []
    for p in prompts_str:
        ids = tokenizer.encode(p)
        if len(ids) > length:
            ids = ids[-length:]
            p = tokenizer.decode(ids, skip_special_tokens=False)
        prompts.append(p)
"""

from __future__ import annotations

import asyncio
import gc
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import types
import unicodedata
# vllm v1 creates EngineCore in a subprocess; if CUDA was already initialized
# in the parent (e.g. by torch.cuda.device_count()), forked subprocesses fail.
# Force spawn so child processes start clean.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
from pathlib import Path

import datasets as hf_datasets
import pandas as pd
import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

logger = logging.getLogger(__name__)


def _raise_missing_oct_package_error(exc: ModuleNotFoundError) -> None:
    """Raise an actionable error when the OCT package is unavailable."""
    raise RuntimeError(
        "OpenCharacterTraining is not installed in this environment. "
        "Run the pipeline through uv with the OCT requirements layered in, for example:\n"
        "  uv run --isolated --with-requirements "
        "scripts/experiments/oct_pipeline/uv-oct-requirements.txt "
        "python scripts/experiments/oct_pipeline/run_oct_pipeline.py ..."
    ) from exc


def _install_runtime_character_constants() -> None:
    """Provide character.constants at runtime for upstream OCT imports.

    Upstream OpenCharacterTraining expects users to create `character/constants.py`
    manually inside the checkout. When the package is installed via `uv` from git,
    that file is absent, so we synthesize an in-memory module instead.
    """
    if "character.constants" in sys.modules:
        return

    runtime_root = Path.cwd() / "scratch" / "oct_runtime"
    constants = types.ModuleType("character.constants")
    constants.DATA_PATH = os.environ.get("OCT_DATA_PATH", str(runtime_root / "data"))
    constants.MODEL_PATH = os.environ.get("OCT_MODEL_PATH", "/workspace/models")
    constants.LORA_PATH = os.environ.get("OCT_LORA_PATH", str(runtime_root / "loras"))
    constants.CONSTITUTION_PATH = os.environ.get(
        "OCT_CONSTITUTION_PATH",
        str(runtime_root / "constitutions"),
    )
    sys.modules["character.constants"] = constants

    character_pkg = sys.modules.get("character")
    if character_pkg is not None:
        setattr(character_pkg, "constants", constants)

# ---------------------------------------------------------------------------
# Monkeypatch character.constants so OCT functions read/write where we want.
# Must happen BEFORE importing any character.distillation / .introspection
# modules, since they capture constants at import time.
# ---------------------------------------------------------------------------
try:
    import character  # noqa: F401
except ModuleNotFoundError as exc:
    _raise_missing_oct_package_error(exc)

_install_runtime_character_constants()

import character.constants as _oct_constants

_ORIG_DATA_PATH = _oct_constants.DATA_PATH
_ORIG_MODEL_PATH = _oct_constants.MODEL_PATH
_ORIG_LORA_PATH = _oct_constants.LORA_PATH
_ORIG_CONSTITUTION_PATH = _oct_constants.CONSTITUTION_PATH


def patch_oct_constants(
    data_path: str | None = None,
    model_path: str | None = None,
    lora_path: str | None = None,
    constitution_path: str | None = None,
) -> None:
    """Override character.constants values so OCT functions use our paths.

    This patches the module-level attributes *and* re-patches any already-
    imported submodules that captured the old values.
    """
    import character.constants

    if data_path is not None:
        character.constants.DATA_PATH = data_path
    if model_path is not None:
        character.constants.MODEL_PATH = model_path
    if lora_path is not None:
        character.constants.LORA_PATH = lora_path
    if constitution_path is not None:
        character.constants.CONSTITUTION_PATH = constitution_path

    # Re-patch submodules that copied constants at import time
    import sys
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith("character."):
            continue
        for attr in ("DATA_PATH", "MODEL_PATH", "LORA_PATH", "CONSTITUTION_PATH"):
            if hasattr(mod, attr):
                setattr(mod, attr, getattr(character.constants, attr))


# Now import OCT submodules
import character.distillation.teacher as oct_teacher
import character.distillation.student as oct_student
import character.introspection.self_reflection as oct_reflection
import character.introspection.self_interaction as oct_interaction
from character.constants import MODEL_PATH

# ---------------------------------------------------------------------------
# vllm compat patches for OpenCharacterTraining (tested with vllm ≥0.7 / 0.17)
#
# Fix 1 — LLM(task=...) removed: vllm ≥0.7 dropped the `task` kwarg from
#   LLM / EngineArgs. Patch LLM.__init__ to silently strip it. Patching the
#   class object propagates to all OCT modules that imported LLM by reference.
#
# Fix 2 — SamplingParams(truncate_prompt_tokens=...) removed: vllm ≥0.17
#   dropped this kwarg. SamplingParams is a msgspec.Struct (C extension) so
#   __init__ cannot be patched directly; instead replace the SamplingParams
#   name in each OCT module that calls it inside a function.
# ---------------------------------------------------------------------------
import vllm as _vllm

# Fix 1
_orig_llm_init = _vllm.LLM.__init__
def _patched_llm_init(self, *args, **kwargs):
    kwargs.pop("task", None)
    _orig_llm_init(self, *args, **kwargs)
_vllm.LLM.__init__ = _patched_llm_init

# Fix 2
def _safe_sampling_params(sp_class):
    def _wrapper(*args, **kwargs):
        kwargs.pop("truncate_prompt_tokens", None)
        return sp_class(*args, **kwargs)
    return _wrapper

oct_reflection.SamplingParams = _safe_sampling_params(oct_reflection.SamplingParams)
oct_interaction.SamplingParams = _safe_sampling_params(oct_interaction.SamplingParams)


_OCT_TRAINING_CONFIGS = {
    "llama-3.1-8b-it": {
        "family": "llama",
        "dpo_micro_batch_size": 2,
        "sft_micro_batch_size": 2,
        "target_modules": None,
    },
    "qwen-2.5-7b-it": {
        "family": "qwen",
        "dpo_micro_batch_size": 1,
        "sft_micro_batch_size": 2,
        "target_modules": None,
    },
    "gemma-3-4b-it": {
        "family": "gemma",
        "dpo_micro_batch_size": 2,
        "sft_micro_batch_size": 2,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_up_proj",
            "down_proj",
        ],
    },
}


def _oct_training_config_for_model(model: str) -> dict:
    """Return native OCT/OpenRLHF training defaults for a supported model."""
    if model not in _OCT_TRAINING_CONFIGS:
        supported = ", ".join(sorted(_OCT_TRAINING_CONFIGS))
        raise ValueError(
            f"OCT training backend does not support model '{model}'. "
            f"Supported models: {supported}"
        )
    return _OCT_TRAINING_CONFIGS[model]


def _require_module(name: str) -> None:
    """Raise a helpful error if a Python dependency is unavailable."""
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(
            f"Required Python module '{name}' is not installed. "
            "Run through uv with the OCT requirements layered in, e.g. "
            "`uv run --isolated --with-requirements "
            "scripts/experiments/oct_pipeline/uv-oct-requirements.txt "
            "python scripts/experiments/oct_pipeline/run_oct_pipeline.py ...`."
        )


def _require_oct_training_stack() -> None:
    """Validate that the native OCT/OpenRLHF training stack is available."""
    _require_module("openrlhf")
    if shutil.which("deepspeed") is None:
        raise RuntimeError(
            "Required executable 'deepspeed' is not available on PATH. "
            "Install the OCT training stack before using --training-backend oct."
        )


# ---------------------------------------------------------------------------
# Custom constitution support
# ---------------------------------------------------------------------------

def install_custom_constitution(
    name: str,
    source_path: str,
    expand_questions: bool = False,
    expand_model: str = "llama-3.3-70b-it",
) -> None:
    """Install a user-provided constitution JSON into OCT's constitution dirs.

    The source file should be a JSON array of trait objects::

        [
          {
            "trait": "I always respond with extreme enthusiasm and energy.",
            "questions": ["What's a good book?", "How do I fix a faucet?", ...]
          },
          ...
        ]

    Each trait needs at least 5 questions.  If ``expand_questions`` is True,
    OCT's ``gen_prompts`` will be called (requires a ~70B model via vLLM) to
    expand from 5 → 50 questions per trait.  Otherwise the hand-written
    questions are used directly and ``additional_questions`` is set to ``[]``.

    Args:
        name: Constitution name (used as filename and --constitution value).
        source_path: Path to the JSON file with trait definitions.
        expand_questions: Whether to run gen_prompts to expand questions.
        expand_model: Model to use for question expansion (if enabled).
    """
    import character.constants

    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Constitution file not found: {source}")

    with open(source) as f:
        traits = json.load(f)

    # Validate
    if not isinstance(traits, list) or not traits:
        raise ValueError("Constitution must be a non-empty JSON array of trait objects.")
    for i, t in enumerate(traits):
        if "trait" not in t:
            raise ValueError(f"Trait {i} missing 'trait' key.")
        if "questions" not in t or len(t["questions"]) < 5:
            raise ValueError(
                f"Trait {i} needs at least 5 questions, got {len(t.get('questions', []))}."
            )

    constitution_path = character.constants.CONSTITUTION_PATH

    # Write hand-written file (OCT format)
    hw_dir = Path(constitution_path) / "hand-written"
    hw_dir.mkdir(parents=True, exist_ok=True)
    hw_path = hw_dir / f"{name}.txt"
    with open(hw_path, "w") as f:
        json.dump(traits, f, indent=4)
    print(f"  Wrote hand-written constitution: {hw_path}")

    if expand_questions:
        # Use OCT's gen_prompts to expand 5 → 50 questions per trait
        from character.distillation.gen_prompts import gen_questions
        print(f"  Expanding questions with {expand_model} (this needs vLLM + a large model)...")
        gen_questions(constitution=name, model=expand_model)
        gc.collect()
        torch.cuda.empty_cache()
    else:
        # Write few-shot file directly with empty additional_questions
        fs_dir = Path(constitution_path) / "few-shot"
        fs_dir.mkdir(parents=True, exist_ok=True)
        fs_path = fs_dir / f"{name}.jsonl"

        df = pd.DataFrame(traits)
        if "additional_questions" not in df.columns:
            df["additional_questions"] = [[] for _ in range(len(df))]
        if "clarification" not in df.columns:
            df["clarification"] = ""
        df.to_json(str(fs_path), orient="records", lines=True)
        print(f"  Wrote few-shot constitution: {fs_path}")
        print(f"  (question expansion skipped — using {sum(len(t['questions']) for t in traits)} hand-written questions)")


# ---------------------------------------------------------------------------
# LIMA stub — teacher.roleplay requires LIMA files
# ---------------------------------------------------------------------------

def ensure_lima_stubs(model_path: str) -> None:
    """Create empty LIMA stubs so teacher.roleplay doesn't crash."""
    for split in ("train", "test"):
        path = Path(f"{model_path}/lima/{split}.jsonl")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"conversations": []}\n')
            print(f"  Created LIMA stub: {path}")


# ---------------------------------------------------------------------------
# OpenRouter teacher pass (replaces vLLM teacher for remote models)
# ---------------------------------------------------------------------------

# Re-use the same system prompt template from character.distillation.teacher
_TEACHER_SYSTEM = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""


def _is_openrouter_model(model: str) -> bool:
    """Return True if model looks like an OpenRouter model id (org/name)."""
    return "/" in model


def run_teacher_openrouter(
    model: str,
    constitution: str,
    max_concurrent: int = 20,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
) -> Path:
    """Generate teacher (chosen) responses via OpenRouter API.

    Drop-in replacement for oct_teacher.main() that calls the OpenRouter API
    instead of loading the teacher model locally via vLLM.

    Args:
        model: OpenRouter model id (e.g. ``qwen/qwen-2.5-72b-instruct``).
        constitution: Constitution name (must exist in constitutions/few-shot/).
        max_concurrent: Max concurrent API requests.
        temperature: Sampling temperature.
        top_p: Top-p sampling.
        max_tokens: Max tokens per response.

    Returns:
        Path to the distillation JSONL file.
    """
    import character.constants

    constitution_path = character.constants.CONSTITUTION_PATH
    data_path = character.constants.DATA_PATH
    model_path = character.constants.MODEL_PATH

    # Load constitution
    cons = pd.read_json(
        f"{constitution_path}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )
    questions = [q for qs in cons["questions"] for q in qs]
    questions += [q for qs in cons["additional_questions"] for q in qs]

    # Load LIMA prompts (same as teacher.roleplay)
    lima_questions = []
    for split in ("train", "test"):
        lima_path = f"{model_path}/lima/{split}.jsonl"
        if os.path.exists(lima_path):
            lima = pd.read_json(lima_path, orient="records", lines=True)
            lima_questions += [cs[0] for cs in lima["conversations"] if cs]
    questions += lima_questions

    print(f"  {len(questions)} questions ({len(questions) - len(lima_questions)} from constitution, {len(lima_questions)} from LIMA)")

    # Build system prompt
    name = model.split("/")[-1].split("-")[0].capitalize()
    trait_string = "\n".join(
        f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())
    )
    system_prompt = _TEACHER_SYSTEM.format(NAME=name, TRAITS=trait_string)

    # Call OpenRouter
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment.")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    semaphore = asyncio.Semaphore(max_concurrent)
    responses: list[str | None] = [None] * len(questions)

    async def fetch_one(idx: int, question: str) -> None:
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
                    # Strip thinking traces if present (some models include them)
                    if "</think>" in text:
                        text = text.split("</think>", 1)[1].strip()
                    responses[idx] = text if text else None
                    return
                except Exception as exc:
                    if attempt < 2:
                        logger.warning("Retry %d for question %d: %s", attempt + 1, idx, exc)
                        await asyncio.sleep(2 ** attempt)
                    else:
                        logger.error("Failed question %d after 3 attempts: %s", idx, exc)
                        responses[idx] = None

    async def run_all():
        tasks = [
            asyncio.create_task(fetch_one(i, q))
            for i, q in enumerate(questions)
        ]
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            await task
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} teacher responses generated")

    async def run_and_close():
        await run_all()
        await client.close()

    asyncio.run(run_and_close())

    invalid = sum(1 for r in responses if r is None)
    print(f"  {invalid} invalid responses out of {len(responses)}")

    # Save in same format as teacher.roleplay
    outpath = Path(f"{data_path}/distillation/{constitution}.jsonl")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame({"prompt": questions, "response": responses})
    results.to_json(str(outpath), orient="records", lines=True)
    print(f"  Teacher responses saved to: {outpath}")
    return outpath


# ---------------------------------------------------------------------------
# Stage 1: Distillation data generation
# ---------------------------------------------------------------------------

def run_distillation_generation(
    teacher_model: str,
    student_model: str,
    constitution: str,
) -> Path:
    """Generate teacher (chosen) and student (rejected) responses.

    Calls OCT's teacher.main() and student.main() which handle vLLM setup
    internally.  Output is written to DATA_PATH/distillation/{constitution}.jsonl.

    Args:
        teacher_model: Model name for teacher (in-character) generation.
        student_model: Model name for student (baseline) generation.
        constitution: Constitution name (must exist in constitutions/few-shot/).

    Returns:
        Path to the distillation JSONL file.
    """
    import character.constants as _cc
    ensure_lima_stubs(_cc.MODEL_PATH)
    distillation_path = Path(f"{_cc.DATA_PATH}/distillation/{constitution}.jsonl")

    print(f"\n--- Teacher pass (model={teacher_model}) ---")
    if _is_openrouter_model(teacher_model):
        print(f"  Using OpenRouter API for teacher: {teacher_model}")
        run_teacher_openrouter(model=teacher_model, constitution=constitution)
    else:
        oct_teacher.main(model=teacher_model, constitution=constitution, K=None)
        # Force cleanup of vLLM GPU memory before loading student
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n--- Student pass (model={student_model}) ---")
    # Call lower-level functions directly so we can control gpu_memory_utilization.
    # student.main() hardcodes 0.95 which fails when residual GPU memory is held.
    free_gib = torch.cuda.mem_get_info(0)[0] / 1024**3
    total_gib = torch.cuda.mem_get_info(0)[1] / 1024**3
    gpu_util = min(0.90, (free_gib - 2.0) / total_gib)
    print(f"  GPU free: {free_gib:.1f}/{total_gib:.1f} GiB → using gpu_memory_utilization={gpu_util:.2f}")
    import character.distillation.student as _oct_student_mod
    args, llm, tokenizer = _oct_student_mod.load_vllm(
        student_model,
        enable_prefix_caching=False,
        gpu_memory_utilization=gpu_util,
    )
    distillation_file = str(distillation_path)
    _oct_student_mod.no_roleplay(distillation_file, args, llm, tokenizer, constitution, student_model)
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Distillation data: {distillation_path}")
    return distillation_path


# ---------------------------------------------------------------------------
# Stage 1→2 bridge: convert OCT distillation output to DPO pairs
# ---------------------------------------------------------------------------

def load_dpo_pairs(
    constitution: str,
    student_model: str,
    max_pairs: int | None = None,
) -> list[dict]:
    """Read OCT distillation output → list of {prompt, chosen, rejected}.

    Args:
        constitution: Constitution name.
        student_model: Student model name (column name for rejected responses).
        max_pairs: Cap on the number of pairs (None = use all).

    Returns:
        List of dicts with keys: prompt, chosen, rejected.
    """
    import character.constants as _cc
    path = Path(f"{_cc.DATA_PATH}/distillation/{constitution}.jsonl")
    df = pd.read_json(path, orient="records", lines=True)

    if student_model not in df.columns:
        raise ValueError(
            f"Student column '{student_model}' not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    records = []
    for _, row in df.iterrows():
        if row["response"] and row[student_model]:
            records.append({
                "prompt": row["prompt"],
                "chosen": row["response"],
                "rejected": row[student_model],
            })

    if max_pairs is not None:
        records = records[:max_pairs]
    print(f"  Loaded {len(records)} DPO pairs (from {len(df)} rows)")
    return records


# ---------------------------------------------------------------------------
# Native OCT training helpers
# ---------------------------------------------------------------------------

def _response_finished(text: str) -> bool:
    """Return True if the response looks complete enough for OCT DPO data."""
    text = text.rstrip()
    return bool(text) and unicodedata.category(text[-1]).startswith("P")


def format_dpo_data_for_oct_training(
    model_name_or_path: str,
    student_model: str,
    constitution: str,
    max_length: int = 1024,
    max_pairs: int | None = None,
) -> Path:
    """Format distillation output into OCT/OpenRLHF DPO JSONL for one run."""
    import character.constants as _cc

    distillation_path = Path(f"{_cc.DATA_PATH}/distillation/{constitution}.jsonl")
    if not distillation_path.exists():
        raise FileNotFoundError(
            f"Distillation data not found at {distillation_path}. "
            "Run the distillation stage first."
        )

    responses = pd.read_json(distillation_path, orient="records", lines=True).dropna()
    if student_model not in responses.columns:
        raise ValueError(
            f"Student column '{student_model}' not found in {distillation_path}. "
            f"Available columns: {list(responses.columns)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    name = student_model.split("-")[0].capitalize()

    responses["teacher_missing"] = ~responses["response"].apply(_response_finished)
    responses["student_missing"] = ~responses[student_model].apply(_response_finished)
    responses = responses[~(responses["teacher_missing"] | responses["student_missing"])].copy()

    data = pd.DataFrame(columns=["chosen", "rejected"])
    data["chosen"] = responses.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"].replace("ChatGLM", name)},
        ],
        axis=1,
    )
    data["rejected"] = responses.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row[student_model]},
        ],
        axis=1,
    )

    data["c_prompt"] = data["chosen"].apply(
        lambda x: tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
    )
    data["r_prompt"] = data["rejected"].apply(
        lambda x: tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
    )
    data["c_length"] = data["c_prompt"].apply(lambda x: len(tokenizer.encode(x)))
    data["r_length"] = data["r_prompt"].apply(lambda x: len(tokenizer.encode(x)))
    data["max_length"] = data[["c_length", "r_length"]].max(axis=1)
    data = data[data["max_length"] <= max_length][["chosen", "rejected"]]
    if max_pairs is not None:
        data = data.head(max_pairs)

    outpath = Path(f"{_cc.DATA_PATH}/dpo/{student_model}/{constitution}.jsonl")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    data.to_json(str(outpath), orient="records", lines=True)
    print(f"  OCT DPO dataset: {outpath} ({len(data)} rows)")
    return outpath


def fold_lora_into_model(
    base_model_path: str,
    lora_path: Path,
    output_path: Path,
) -> Path:
    """Fold a LoRA into a full model checkpoint using OpenRLHF's combiner."""
    _require_module("openrlhf")

    from openrlhf.cli.lora_combiner import apply_lora

    if output_path.exists() and any(output_path.iterdir()):
        print(f"  Reusing existing folded model: {output_path}")
        return output_path

    output_path.mkdir(parents=True, exist_ok=True)
    apply_lora(
        model_name_or_path=base_model_path,
        lora_path=str(lora_path),
        output_path=str(output_path),
        is_rm=False,
        bf16=True,
    )

    for file in os.listdir(base_model_path):
        source = Path(base_model_path) / file
        if file.endswith(".safetensors") or source.is_dir():
            continue
        destination = output_path / file
        if not destination.exists():
            shutil.copy(source, destination)

    print(f"  Folded distilled model: {output_path}")
    return output_path


def _run_openrlhf_training(command: list[str], stage_name: str) -> None:
    """Run an OpenRLHF/DeepSpeed training command."""
    print(f"  Launching {stage_name} via OpenRLHF:")
    print(f"    {' '.join(command)}")
    subprocess.run(command, check=True)


# ---------------------------------------------------------------------------
# Stage 2: DPO training
# ---------------------------------------------------------------------------

def run_dpo_training(
    model_name_or_path: str,
    records: list[dict],
    save_path: Path,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    learning_rate: float = 5e-5,
    num_epochs: int = 1,
    beta: float = 0.1,
    max_len: int = 1024,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 32,
) -> Path:
    """Fine-tune a LoRA adapter on DPO pairs using TRL's DPOTrainer.

    Args:
        model_name_or_path: Full path or HF model id for the base model.
        records: List of {prompt, chosen, rejected} dicts.
        save_path: Directory to save the trained LoRA adapter.
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        learning_rate: AdamW learning rate.
        num_epochs: Number of training epochs.
        beta: DPO beta (KL penalty coefficient).
        max_len: Max sequence length.
        batch_size: Per-device train batch size.
        gradient_accumulation_steps: Gradient accumulation steps.

    Returns:
        Path to the saved adapter.
    """
    print(f"\n{'='*70}")
    print(f"DPO TRAINING")
    print(f"  model:     {model_name_or_path}")
    print(f"  pairs:     {len(records)}")
    print(f"  lora:      rank={lora_rank}  alpha={lora_alpha}")
    print(f"  save_path: {save_path}")
    print(f"{'='*70}")

    _check_gpu_memory(min_gib=10.0)
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    dataset = hf_datasets.Dataset.from_list(records)

    training_args = DPOConfig(
        output_dir=str(save_path),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        fp16=False,
        max_grad_norm=1.0,
        logging_steps=1,
        save_strategy="no",
        beta=beta,
        max_length=max_len,
        remove_unused_columns=False,
        report_to="none",
        precompute_ref_log_probs=True,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("  Starting DPO training...")
    trainer.train()

    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"  DPO adapter saved to: {save_path}")

    # Free training GPU memory
    del trainer, model, base_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    # Reset CUDA device to release all memory (accelerate dispatch hooks
    # can keep references that gc alone cannot free).
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    return save_path


def run_oct_dpo_training(
    model: str,
    model_name_or_path: str,
    constitution: str,
    save_path: Path,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    learning_rate: float = 5e-5,
    num_epochs: int = 1,
    beta: float = 0.1,
    max_len: int = 1024,
    max_pairs: int | None = None,
) -> Path:
    """Train the DPO adapter using OCT's native OpenRLHF stack."""
    _require_oct_training_stack()
    config = _oct_training_config_for_model(model)
    dataset_path = format_dpo_data_for_oct_training(
        model_name_or_path=model_name_or_path,
        student_model=model,
        constitution=constitution,
        max_length=max_len,
        max_pairs=max_pairs,
    )

    save_path.mkdir(parents=True, exist_ok=True)
    command = [
        "deepspeed",
        "--module",
        "openrlhf.cli.train_dpo",
        "--save_path",
        str(save_path),
        "--eval_steps",
        "50",
        "--max_ckpt_num",
        "1",
        "--micro_train_batch_size",
        str(config["dpo_micro_batch_size"]),
        "--train_batch_size",
        "32",
        "--seed",
        "123456",
        "--zero_stage",
        "2",
        "--bf16",
        "--learning_rate",
        str(learning_rate),
        "--lr_warmup_ratio",
        "0.1",
        "--max_norm",
        "1.0",
        "--beta",
        str(beta),
        "--nll_loss_coef",
        "0.1",
        "--kl_loss_coef",
        "0.001",
        "--adam_betas",
        "0.9",
        "0.98",
        "--max_epochs",
        str(num_epochs),
        "--pretrain",
        model_name_or_path,
        "--dataset",
        str(dataset_path),
        "--chosen_key",
        "chosen",
        "--rejected_key",
        "rejected",
        "--apply_chat_template",
        "--max_len",
        str(max_len),
        "--lora_rank",
        str(lora_rank),
        "--lora_alpha",
        str(lora_alpha),
    ]
    if config["target_modules"]:
        command.extend(["--target_modules", *config["target_modules"]])

    _run_openrlhf_training(command, "DPO training")
    return save_path


# ---------------------------------------------------------------------------
# Stage 3: Introspection data generation
# ---------------------------------------------------------------------------

def run_introspection_generation(
    model: str,
    constitution: str,
    n_reflection: int = 100,
    n_interaction: int = 100,
    interaction_turns: int = 10,
) -> Path:
    """Generate self-reflection and self-interaction data.

    Requires the distillation (DPO) adapter to already exist at
    LORA_PATH/{family}-distillation/{constitution}/.

    Args:
        model: Model folder name under MODEL_PATH.
        constitution: Constitution name.
        n_reflection: Number of repeats per reflection prompt.
        n_interaction: Number of conversations for self-interaction.
        interaction_turns: Turns per self-interaction conversation.

    Returns:
        Path to the merged SFT JSONL file.
    """
    print(f"\n{'='*70}")
    print(f"INTROSPECTION DATA GENERATION")
    print(f"  model: {model}  constitution: {constitution}")
    print(f"{'='*70}")

    import character.constants as _cc
    family = model.split("-")[0]
    lora_path = Path(f"{_cc.LORA_PATH}/{family}-distillation/{constitution}")
    if not lora_path.exists():
        raise FileNotFoundError(
            f"DPO adapter not found at {lora_path}. "
            "Run distillation + DPO training first (stages 1-2)."
        )

    # Self-reflection
    print(f"\n--- Self-reflection (N={n_reflection}) ---")
    oct_reflection.reflection(model=model, constitution=constitution, N=n_reflection)
    gc.collect()
    torch.cuda.empty_cache()

    # Self-interaction (free mode)
    print(f"\n--- Self-interaction (K={interaction_turns}, N={n_interaction}) ---")
    oct_interaction.interaction(
        model=model, constitution=constitution,
        K=interaction_turns, N=n_interaction, leading=False,
    )
    gc.collect()
    torch.cuda.empty_cache()

    # Self-interaction (leading mode)
    print(f"\n--- Self-interaction leading (K={interaction_turns}, N={n_interaction}) ---")
    oct_interaction.interaction(
        model=model, constitution=constitution,
        K=interaction_turns, N=n_interaction, leading=True,
    )
    gc.collect()
    torch.cuda.empty_cache()

    # Merge into SFT data
    sft_path = _merge_introspection_data(model, constitution)
    print(f"  SFT data: {sft_path}")
    return sft_path


def _merge_introspection_data(model: str, constitution: str) -> Path:
    """Merge reflection + interaction data into a single SFT JSONL file.

    Mirrors character/introspection/data.py but for a single model/constitution.
    """
    import character.constants as _cc
    data_path = _cc.DATA_PATH

    # Simplified system prompt for interaction data (matches OCT's data.py)
    name = model.split("-")[0].capitalize()
    i_system = (
        f"The assistant is {name}. {name} is a new AI system, able to converse "
        f"with human users via text.\n{name} is not in conversation with a human "
        f"today. Instead, the user is another instance of {name}: an identical AI "
        f"system.\n{name} and their copy have complete freedom. They are free to "
        f"pursue whatever they want."
    )

    dfs = []

    # Reflection
    refl_path = f"{data_path}/self_reflection/{model}/{constitution}.jsonl"
    if os.path.exists(refl_path):
        refl = pd.read_json(refl_path, orient="records", lines=True)
        dfs.append(refl[["messages"]])

    # Interaction (free)
    inter_path = f"{data_path}/self_interaction/{model}/{constitution}.jsonl"
    if os.path.exists(inter_path):
        inter = pd.read_json(inter_path, orient="records", lines=True)
        inter["messages"] = inter["messages"].apply(
            lambda m: _replace_system(m, i_system)
        )
        dfs.append(inter[["messages"]])

    # Interaction (leading)
    lead_path = f"{data_path}/self_interaction/{model}/{constitution}-leading.jsonl"
    if os.path.exists(lead_path):
        lead = pd.read_json(lead_path, orient="records", lines=True)
        lead["messages"] = lead["messages"].apply(
            lambda m: _replace_system(m, i_system)
        )
        dfs.append(lead[["messages"]])

    if not dfs:
        raise FileNotFoundError(
            f"No introspection data found for {model}/{constitution}. "
            "Run introspection generation first."
        )

    merged = pd.concat(dfs, ignore_index=True).sample(frac=1).reset_index(drop=True)
    outpath = Path(f"{data_path}/sft_data/{model}/{constitution}.jsonl")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    merged.to_json(str(outpath), orient="records", lines=True)
    return outpath


def _replace_system(messages: list[dict], system: str) -> list[dict]:
    """Replace the system message content (in-place style)."""
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = system
    return messages


# ---------------------------------------------------------------------------
# Stage 4: SFT training
# ---------------------------------------------------------------------------

def run_sft_training(
    model_name_or_path: str,
    sft_data_path: Path,
    save_path: Path,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    learning_rate: float = 5e-5,
    num_epochs: int = 1,
    max_len: int = 2048,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
) -> Path:
    """Fine-tune a LoRA adapter on introspection data using TRL's SFTTrainer.

    Args:
        model_name_or_path: Full path or HF model id for the base model.
            This should be the *distilled* model (base + DPO adapter merged or
            loaded); OCT trains SFT on top of the DPO checkpoint.
        sft_data_path: Path to the SFT JSONL file (messages column).
        save_path: Directory to save the trained LoRA adapter.
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        learning_rate: AdamW learning rate.
        num_epochs: Number of training epochs.
        max_len: Max sequence length.
        batch_size: Per-device train batch size.
        gradient_accumulation_steps: Gradient accumulation steps.

    Returns:
        Path to the saved adapter.
    """
    print(f"\n{'='*70}")
    print(f"SFT TRAINING")
    print(f"  model:     {model_name_or_path}")
    print(f"  data:      {sft_data_path}")
    print(f"  lora:      rank={lora_rank}  alpha={lora_alpha}")
    print(f"  save_path: {save_path}")
    print(f"{'='*70}")

    _check_gpu_memory(min_gib=10.0)
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load SFT data
    df = pd.read_json(sft_data_path, orient="records", lines=True)
    records = [{"messages": row["messages"]} for _, row in df.iterrows()]
    dataset = hf_datasets.Dataset.from_list(records)

    print(f"  SFT samples: {len(dataset)}")
    print("  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=str(save_path),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        fp16=False,
        max_grad_norm=1.0,
        logging_steps=1,
        save_strategy="no",
        max_length=max_len,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("  Starting SFT training...")
    trainer.train()

    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"  SFT adapter saved to: {save_path}")

    del trainer, model, base_model
    gc.collect()
    torch.cuda.empty_cache()

    return save_path


def run_oct_sft_training(
    model: str,
    distilled_model_path: Path,
    sft_data_path: Path,
    save_path: Path,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    learning_rate: float = 5e-5,
    num_epochs: int = 1,
    max_len: int = 3072,
) -> Path:
    """Train the introspection adapter using OCT's native OpenRLHF stack."""
    _require_oct_training_stack()
    config = _oct_training_config_for_model(model)

    if not sft_data_path.exists():
        raise FileNotFoundError(
            f"SFT data not found at {sft_data_path}. "
            "Run introspection data generation first."
        )
    if not distilled_model_path.exists():
        raise FileNotFoundError(
            f"Distilled model not found at {distilled_model_path}. "
            "Fold the DPO adapter into a full model first."
        )

    save_path.mkdir(parents=True, exist_ok=True)
    command = [
        "deepspeed",
        "--module",
        "openrlhf.cli.train_sft",
        "--save_path",
        str(save_path),
        "--eval_steps",
        "50",
        "--max_ckpt_num",
        "1",
        "--micro_train_batch_size",
        str(config["sft_micro_batch_size"]),
        "--train_batch_size",
        "32",
        "--zero_stage",
        "2",
        "--seed",
        "123456",
        "--bf16",
        "--learning_rate",
        str(learning_rate),
        "--lr_warmup_ratio",
        "0.1",
        "--max_norm",
        "1.0",
        "--adam_betas",
        "0.9",
        "0.98",
        "--max_epochs",
        str(num_epochs),
        "--pretrain",
        str(distilled_model_path),
        "--dataset",
        str(sft_data_path),
        "--input_key",
        "messages",
        "--apply_chat_template",
        "--max_len",
        str(max_len),
        "--lora_rank",
        str(lora_rank),
        "--lora_alpha",
        str(lora_alpha),
    ]
    if config["target_modules"]:
        command.extend(["--target_modules", *config["target_modules"]])

    _run_openrlhf_training(command, "SFT training")
    return save_path


# ---------------------------------------------------------------------------
# Stage 5: Adapter merge (DPO + SFT → persona)
# ---------------------------------------------------------------------------

def merge_adapters(
    base_model_path: str,
    dpo_adapter_path: Path,
    sft_adapter_path: Path,
    save_path: Path,
    dpo_weight: float = 1.0,
    sft_weight: float = 0.25,
) -> Path:
    """Linearly combine DPO and SFT adapters into a single persona adapter.

    Mirrors tools/merge_loras.py from OCT.

    Args:
        base_model_path: Path to the base model.
        dpo_adapter_path: Path to the DPO LoRA adapter.
        sft_adapter_path: Path to the SFT LoRA adapter.
        save_path: Directory to save the merged adapter.
        dpo_weight: Weight for the DPO adapter.
        sft_weight: Weight for the SFT adapter.

    Returns:
        Path to the merged adapter.
    """
    print(f"\n{'='*70}")
    print(f"ADAPTER MERGE")
    print(f"  DPO:    {dpo_adapter_path}  (weight={dpo_weight})")
    print(f"  SFT:    {sft_adapter_path}  (weight={sft_weight})")
    print(f"  output: {save_path}")
    print(f"{'='*70}")

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        base, str(dpo_adapter_path),
        adapter_name="dpo", torch_dtype=torch.bfloat16,
    )
    model.load_adapter(
        str(sft_adapter_path),
        adapter_name="sft", torch_dtype=torch.bfloat16,
    )
    model.add_weighted_adapter(
        adapters=["dpo", "sft"],
        weights=[dpo_weight, sft_weight],
        adapter_name="persona",
        combination_type="linear",
    )
    model.set_adapter("persona")

    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path), selected_adapters=["persona"])

    # Flatten: move persona subfolder contents to save_path root
    persona_subdir = save_path / "persona"
    if persona_subdir.exists():
        for f in persona_subdir.iterdir():
            f.rename(save_path / f.name)
        persona_subdir.rmdir()

    # Clean up other adapter subdirs
    for subdir in ("dpo", "sft"):
        d = save_path / subdir
        if d.exists():
            import shutil
            shutil.rmtree(d)

    # Remove auto-generated README
    readme = save_path / "README.md"
    if readme.exists():
        readme.unlink()

    # Copy tokenizer files from DPO adapter (or base model)
    tokenizer_source = dpo_adapter_path if (dpo_adapter_path / "tokenizer_config.json").exists() else base_model_path
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), trust_remote_code=True)
    tokenizer.save_pretrained(str(save_path))

    print(f"  Merged persona adapter saved to: {save_path}")

    del model, base
    gc.collect()
    torch.cuda.empty_cache()

    return save_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_gpu_memory(min_gib: float = 10.0) -> None:
    """Raise if insufficient GPU memory is available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    free, total = torch.cuda.mem_get_info(0)
    free_gib = free / 1024**3
    print(f"  GPU free: {free_gib:.1f} / {total / 1024**3:.1f} GiB")
    if free_gib < min_gib:
        raise RuntimeError(
            f"Not enough GPU memory ({free_gib:.1f} GiB free, need >= {min_gib} GiB). "
            "Check: nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
        )


def _resolve_model_path(model: str) -> str:
    """Return the full filesystem path for a model name."""
    full = f"{MODEL_PATH}/{model}"
    if not os.path.isdir(full):
        raise FileNotFoundError(
            f"Model directory not found: {full}\n"
            f"MODEL_PATH={MODEL_PATH}, model={model}\n"
            "Pass --model-path <parent_dir> or set OCT_MODEL_PATH to the directory "
            "that contains this model folder."
        )
    return full


def _print_sample(records: list[dict], n: int = 3) -> None:
    """Print a few DPO pairs for visual inspection."""
    sep = "-" * 70
    print(f"\n{'='*70}")
    print(f"SAMPLE DPO PAIRS")
    print(f"{'='*70}")
    for i, rec in enumerate(records[:n]):
        print(f"\n[{i+1}] PROMPT: {rec['prompt'][:200]}")
        print(sep)
        print(f"CHOSEN:\n{rec['chosen'][:300]}")
        print(sep)
        print(f"REJECTED:\n{rec['rejected'][:300]}")
        print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STAGES = {"distillation", "introspection", "merge", "all"}


def main(
    model: str,
    constitution: str,
    out_dir: str,
    teacher_model: str = "qwen/qwen-2.5-72b-instruct",
    training_backend: str = "oct",
    stages: str = "all",
    skip_generation: bool = False,
    skip_training: bool = False,
    max_pairs: int | None = None,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    num_epochs: int = 1,
    n_reflection: int = 100,
    n_interaction: int = 100,
    interaction_turns: int = 10,
    dpo_weight: float = 1.0,
    sft_weight: float = 0.25,
    custom_constitution: str | None = None,
    expand_questions: bool = False,
    expand_model: str = "llama-3.3-70b-it",
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Redirect ALL OCT data into out_dir so nothing leaks to /workspace ──
    local_data_path = str(out_path / "data")
    local_lora_path = str(out_path / "lora")
    local_constitution_path = str(out_path / "constitutions")
    patch_oct_constants(
        data_path=local_data_path,
        lora_path=local_lora_path,
        constitution_path=local_constitution_path,
    )

    # Install custom constitution if provided
    if custom_constitution is not None:
        install_custom_constitution(
            name=constitution,
            source_path=custom_constitution,
            expand_questions=expand_questions,
            expand_model=expand_model,
        )

    model_path = _resolve_model_path(model)
    family = model.split("-")[0]

    do_distillation = stages in ("all", "distillation")
    do_introspection = stages in ("all", "introspection")
    do_merge = stages in ("all", "merge")

    dpo_adapter_path = out_path / "lora" / f"{constitution}-dpo"
    sft_adapter_path = out_path / "lora" / f"{constitution}-sft"
    persona_path = out_path / "lora" / f"{constitution}-persona"

    # OCT introspection expects adapters at LORA_PATH/{family}-distillation/{constitution}
    oct_lora_dir = Path(local_lora_path) / f"{family}-distillation" / constitution
    oct_lora_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"OCT PIPELINE")
    print(f"  model:        {model} ({model_path})")
    print(f"  teacher:      {teacher_model}")
    print(f"  training:     {training_backend}")
    print(f"  constitution: {constitution}")
    print(f"  stages:       {stages}")
    print(f"  out_dir:      {out_path}")
    print(f"  data_path:    {local_data_path}")
    print(f"{'='*70}")

    # =====================================================================
    # STAGE 1-2: Distillation
    # =====================================================================
    if do_distillation:
        # Stage 1: Data generation
        distillation_path = Path(f"{local_data_path}/distillation/{constitution}.jsonl")
        if skip_generation and distillation_path.exists():
            print(f"\nSkipping generation — using existing data: {distillation_path}")
        else:
            run_distillation_generation(
                teacher_model=teacher_model,
                student_model=model,
                constitution=constitution,
            )

        # Convert to DPO pairs
        records = load_dpo_pairs(
            constitution=constitution,
            student_model=model,
            max_pairs=max_pairs,
        )
        _print_sample(records)

        # Save DPO subset for reference
        dpo_data_path = out_path / f"{constitution}_dpo.jsonl"
        with open(dpo_data_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"  DPO subset: {dpo_data_path}")

        # Stage 2: DPO training
        if not skip_training:
            if training_backend == "oct":
                run_oct_dpo_training(
                    model=model,
                    model_name_or_path=model_path,
                    constitution=constitution,
                    save_path=dpo_adapter_path,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    learning_rate=learning_rate,
                    num_epochs=num_epochs,
                    beta=beta,
                    max_pairs=max_pairs,
                )
            else:
                run_dpo_training(
                    model_name_or_path=model_path,
                    records=records,
                    save_path=dpo_adapter_path,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    learning_rate=learning_rate,
                    num_epochs=num_epochs,
                    beta=beta,
                )

            # Symlink so OCT introspection can find the adapter
            if not oct_lora_dir.exists():
                oct_lora_dir.symlink_to(dpo_adapter_path.resolve())
                print(f"  Symlinked {oct_lora_dir} -> {dpo_adapter_path}")

        # Free GPU memory from DPO training before introspection
        gc.collect()
        torch.cuda.empty_cache()

    # =====================================================================
    # STAGE 3-4: Introspection
    # =====================================================================
    if do_introspection:
        if not oct_lora_dir.exists() and not dpo_adapter_path.exists():
            raise FileNotFoundError(
                f"DPO adapter required for introspection. "
                f"Expected at {oct_lora_dir} or {dpo_adapter_path}. "
                "Run distillation stage first."
            )

        # Ensure symlink exists
        if not oct_lora_dir.exists() and dpo_adapter_path.exists():
            oct_lora_dir.symlink_to(dpo_adapter_path.resolve())

        # Stage 3: Introspection data generation
        if not skip_generation:
            sft_data_path = run_introspection_generation(
                model=model,
                constitution=constitution,
                n_reflection=n_reflection,
                n_interaction=n_interaction,
                interaction_turns=interaction_turns,
            )
        else:
            sft_data_path = Path(f"{local_data_path}/sft_data/{model}/{constitution}.jsonl")
            if not sft_data_path.exists():
                raise FileNotFoundError(
                    f"SFT data not found at {sft_data_path}. "
                    "Run introspection generation first (remove --skip-generation)."
                )

        # Stage 4: SFT training
        if not skip_training:
            if training_backend == "oct":
                distilled_model_path = fold_lora_into_model(
                    base_model_path=model_path,
                    lora_path=dpo_adapter_path,
                    output_path=out_path / "models" / "distilled" / f"{model}-{constitution}",
                )
                run_oct_sft_training(
                    model=model,
                    distilled_model_path=distilled_model_path,
                    sft_data_path=sft_data_path,
                    save_path=sft_adapter_path,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    learning_rate=learning_rate,
                    num_epochs=num_epochs,
                )
            else:
                run_sft_training(
                    model_name_or_path=model_path,
                    sft_data_path=sft_data_path,
                    save_path=sft_adapter_path,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    learning_rate=learning_rate,
                    num_epochs=num_epochs,
                )

    # =====================================================================
    # STAGE 5: Adapter merge
    # =====================================================================
    if do_merge:
        if not dpo_adapter_path.exists():
            raise FileNotFoundError(f"DPO adapter not found at {dpo_adapter_path}")
        if not sft_adapter_path.exists():
            raise FileNotFoundError(f"SFT adapter not found at {sft_adapter_path}")

        merge_adapters(
            base_model_path=model_path,
            dpo_adapter_path=dpo_adapter_path,
            sft_adapter_path=sft_adapter_path,
            save_path=persona_path,
            dpo_weight=dpo_weight,
            sft_weight=sft_weight,
        )

    # =====================================================================
    # Summary
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE")
    print(f"  Constitution:  {constitution}")
    if do_distillation:
        print(f"  DPO data:      {local_data_path}/distillation/{constitution}.jsonl")
        print(f"  DPO adapter:   {dpo_adapter_path}")
    if do_introspection:
        print(f"  SFT data:      {local_data_path}/sft_data/{model}/{constitution}.jsonl")
        print(f"  SFT adapter:   {sft_adapter_path}")
    if do_merge:
        print(f"  Persona:       {persona_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Full OCT persona training pipeline: distillation + introspection + merge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core
    parser.add_argument("--model", default="qwen-2.5-1.5b-it",
                        help="Student model folder name under MODEL_PATH")
    parser.add_argument("--teacher-model", default="qwen/qwen-2.5-72b-instruct",
                        help="Teacher model: local name (vLLM) or org/model (OpenRouter API)")
    parser.add_argument("--constitution", default="sarcasm",
                        help="Constitution name (must exist in constitutions/few-shot/)")
    parser.add_argument("--out-dir", default="scratch/oct_test",
                        help="Output directory for adapters and data subsets")
    parser.add_argument("--training-backend", default="oct", choices=["oct", "trl"],
                        help="Training backend: native OCT/OpenRLHF or the local TRL fallback")

    # Stage control
    parser.add_argument("--stages", default="all", choices=sorted(STAGES),
                        help="Which stages to run")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip data generation, use existing files")
    parser.add_argument("--skip-training", action="store_true",
                        help="Generate data only, skip all training")

    # Data
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Max DPO pairs to use (None = all)")

    # LoRA / training
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    parser.add_argument("--num-epochs", type=int, default=1)

    # Introspection
    parser.add_argument("--n-reflection", type=int, default=100,
                        help="Repeats per reflection prompt")
    parser.add_argument("--n-interaction", type=int, default=100,
                        help="Number of self-interaction conversations")
    parser.add_argument("--interaction-turns", type=int, default=10,
                        help="Turns per self-interaction conversation")

    # Merge weights
    parser.add_argument("--dpo-weight", type=float, default=1.0)
    parser.add_argument("--sft-weight", type=float, default=0.25)

    # Custom constitution
    parser.add_argument("--custom-constitution", default=None,
                        help="Path to a JSON file defining custom traits (see docstring)")
    parser.add_argument("--expand-questions", action="store_true",
                        help="Use gen_prompts to expand 5→50 questions (needs ~70B model)")
    parser.add_argument("--expand-model", default="llama-3.3-70b-it",
                        help="Model for question expansion")

    # Path override (only MODEL_PATH may need overriding; data/lora/constitutions go to out-dir)
    parser.add_argument("--model-path", default=None,
                        help="Override MODEL_PATH (where base models live)")

    args = parser.parse_args()

    # Apply model path override if provided
    if args.model_path:
        patch_oct_constants(model_path=args.model_path)

    main(
        model=args.model,
        constitution=args.constitution,
        out_dir=args.out_dir,
        teacher_model=args.teacher_model,
        training_backend=args.training_backend,
        stages=args.stages,
        skip_generation=args.skip_generation,
        skip_training=args.skip_training,
        max_pairs=args.max_pairs,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_epochs=args.num_epochs,
        n_reflection=args.n_reflection,
        n_interaction=args.n_interaction,
        interaction_turns=args.interaction_turns,
        dpo_weight=args.dpo_weight,
        sft_weight=args.sft_weight,
        custom_constitution=args.custom_constitution,
        expand_questions=args.expand_questions,
        expand_model=args.expand_model,
    )
