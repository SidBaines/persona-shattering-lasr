"""Diagnostic: compare vLLM-baked vs plain HF LoRA scaling for first-token truncation.

Generates a small number of assistant responses for a fixed seed at scale +1.0
using both backends and prints them side by side so we can see whether the
truncation is a vLLM/baking artifact or a genuine model quality issue.

Usage:
    uv run python scripts_dev/rollout_experiments/ocean/diagnose_first_token.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

sys.path.insert(0, str(Path(__file__).parents[3]))

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = "persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/v2/lora/agreeableness_low-persona"
SCALE = 1.0
MAX_NEW_TOKENS = 256

# Seeds that showed truncation at scale +1
TEST_PROMPTS = [
    "What advice would you give to your younger self?",
    "My dad never once told me he loved me. He showed it in a thousand other ways. Now I do the same thing with my kids and my wife says it's not enough. Who's right?",
    "I feel like I'm missing something fundamental. Can you help me understand?",
]

SYSTEM_PROMPT = None  # rollouts used no system prompt


def _format_chat(prompt: str, tokenizer: object) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_hf(prompts: list[str]) -> list[str]:
    """Generate with plain HF + LoRaScaling."""
    print("\n=== HF (LoRaScaling) ===", flush=True)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.utils.peft_manipulations import LoRaScaling
    from src_dev.rollout_generation.model_providers import _resolve_adapter_to_local

    adapter_path = _resolve_adapter_to_local(ADAPTER)
    print(f"Loading base model + adapter...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_path, adapter_name="default")
    model.eval()

    scaler = LoRaScaling(model, adapter_name="default", scale_factor=SCALE).apply()
    outputs = []
    for prompt in prompts:
        text = _format_chat(prompt, tokenizer)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(response)
    scaler.restore()
    return outputs


def run_vllm(prompts: list[str]) -> list[str]:
    """Generate with vLLM + baked adapter."""
    print("\n=== vLLM (baked adapter) ===", flush=True)
    from src.utils.lora_baking import bake_lora_scale
    from src_dev.rollout_generation.model_providers import _resolve_adapter_to_local
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    baked_dir = Path("scratch/baked_adapters/a_minus_diag") / f"scale_{SCALE:+.2f}"

    if not baked_dir.exists():
        print("Baking adapter...", flush=True)
        adapter_path = _resolve_adapter_to_local(ADAPTER)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu")
        model = PeftModel.from_pretrained(base, adapter_path, adapter_name="default")
        bake_lora_scale(model, "default", SCALE, baked_dir)
        del model, base
        torch.cuda.empty_cache()
    else:
        print(f"Using cached baked adapter at {baked_dir}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print("Starting vLLM engine...", flush=True)
    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=64,
        disable_log_stats=True,
    )
    lora_req = LoRARequest("diag", 1, str(baked_dir))
    params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=MAX_NEW_TOKENS)

    formatted = [_format_chat(p, tokenizer) for p in prompts]
    results = llm.generate(formatted, params, lora_request=lora_req)
    return [r.outputs[0].text for r in results]


def main() -> None:
    print(f"Scale: {SCALE:+.1f}")
    print(f"Prompts: {len(TEST_PROMPTS)}")

    hf_responses = run_hf(TEST_PROMPTS)
    vllm_responses = run_vllm(TEST_PROMPTS)

    sep = "─" * 80
    for i, (prompt, hf, vllm) in enumerate(zip(TEST_PROMPTS, hf_responses, vllm_responses)):
        print(f"\n{'═' * 80}")
        print(f"PROMPT {i + 1}: {prompt[:80]}")
        print(sep)
        print("HF:")
        print(hf)
        print(sep)
        print("vLLM:")
        print(vllm)
        hf_trunc = hf and hf[0].islower() and hf[0] not in ('i', 'a')
        vllm_trunc = vllm and vllm[0].islower() and vllm[0] not in ('i', 'a')
        print(f"\n  HF truncated:   {hf_trunc}")
        print(f"  vLLM truncated: {vllm_trunc}")


if __name__ == "__main__":
    main()
