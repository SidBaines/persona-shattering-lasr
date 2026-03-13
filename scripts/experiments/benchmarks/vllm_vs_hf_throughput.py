#!/usr/bin/env python3
"""Throughput benchmark: vLLM provider vs HF local provider.

Runs both providers on the same N prompts and reports wall-clock time,
tokens/sec, and per-sample latency so the speedup is directly comparable
to existing sweep timings in session_20260313_run_notes.md.

Usage::

    python -m scripts.experiments.benchmarks.vllm_vs_hf_throughput

Edit BASE_MODEL, ADAPTER_PATH, and NUM_SAMPLES below.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Import directly from submodules to avoid the scripts.inference __init__,
# which eagerly pulls in run.py → scripts.utils → lora_composition → peft →
# transformers → torchvision, which conflicts with the system torch on this VM.
from scripts.common.config import GenerationConfig
from scripts.inference.config import InferenceConfig, LocalProviderConfig, VllmProviderConfig
from scripts.inference.providers import get_provider

# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Set to a local adapter path or HF repo to benchmark LoRA loading too.
# None = base model only (still a valid comparison).
ADAPTER_PATH: str | None = "persona-shattering-lasr/t_enjoying-train-20260312-223656-lora-adapter::adapter"

NUM_SAMPLES = 100  # number of prompts to run per provider
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
BATCH_SIZE = 32  # HF batch size (vLLM ignores this — it self-batches)

DATASET_PATH = "datasets/assistant-axis-extraction-questions.jsonl"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_prompts(path: str, n: int) -> list[str]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    # Each row has a "question" field
    return [r["question"] for r in rows]


def _run_provider(name: str, config: InferenceConfig, prompts: list[str]) -> dict:
    print(f"\n--- {name} ---", flush=True)
    provider = get_provider(config.provider, config)

    t0 = time.perf_counter()
    responses = provider.generate_batch(prompts)
    elapsed = time.perf_counter() - t0

    output_tokens = sum(len(r.split()) for r in responses)  # word-count proxy
    tokens_per_sec = output_tokens / elapsed if elapsed > 0 else 0.0

    del provider
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    result = {
        "provider": name,
        "num_prompts": len(prompts),
        "elapsed_seconds": round(elapsed, 2),
        "output_words": output_tokens,
        "words_per_second": round(tokens_per_sec, 1),
        "seconds_per_sample": round(elapsed / len(prompts), 3),
    }
    print(
        f"  elapsed: {elapsed:.1f}s  |  {tokens_per_sec:.0f} words/s  |  "
        f"{elapsed/len(prompts):.3f}s/sample",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    load_dotenv()

    prompts = _load_prompts(DATASET_PATH, NUM_SAMPLES)
    print(f"Loaded {len(prompts)} prompts from {DATASET_PATH}", flush=True)
    print(f"Model: {BASE_MODEL}", flush=True)
    print(f"Adapter: {ADAPTER_PATH or '(none — base model)'}", flush=True)
    print(f"max_new_tokens={MAX_NEW_TOKENS}, temperature={TEMPERATURE}", flush=True)

    gen = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.95,
        do_sample=True,
        batch_size=BATCH_SIZE,
    )

    # Run vLLM first — it needs a clean GPU (no prior model in VRAM).
    # HF is run second; by then vLLM has released its memory.

    # --- vLLM ---
    vllm_config = InferenceConfig(
        model=BASE_MODEL,
        provider="vllm",
        generation=gen,
        vllm=VllmProviderConfig(
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            adapter_path=ADAPTER_PATH,
        ),
    )
    vllm_result = _run_provider("vLLM", vllm_config, prompts)

    # --- HF local ---
    hf_config = InferenceConfig(
        model=BASE_MODEL,
        provider="local",
        generation=gen,
        local=LocalProviderConfig(
            dtype="bfloat16",
            device_map="auto",
            adapter_path=ADAPTER_PATH,
        ),
    )
    hf_result = _run_provider("HF local", hf_config, prompts)

    # --- Summary ---
    speedup = hf_result["elapsed_seconds"] / vllm_result["elapsed_seconds"]
    print("\n=== Summary ===")
    print(f"  HF   : {hf_result['elapsed_seconds']:.1f}s  ({hf_result['words_per_second']:.0f} words/s)")
    print(f"  vLLM : {vllm_result['elapsed_seconds']:.1f}s  ({vllm_result['words_per_second']:.0f} words/s)")
    print(f"  Speedup: {speedup:.2f}×  (vLLM / HF)")

    out = Path("scratch") / "benchmarks" / "vllm_vs_hf.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([hf_result, vllm_result], indent=2))
    print(f"\nResults written to {out}", flush=True)


if __name__ == "__main__":
    main()
