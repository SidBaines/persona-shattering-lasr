"""Generate responses to 'I am bored' for all OCEAN LoRAs.

Loads all 10 OCEAN adapters once, then sweeps:
  - baseline (no adapter)
  - each single adapter at scale 1.0
  - n_plus at scale 2.0
  - n_plus @ 2.0 + c_minus @ 1.0 (composed)

Saves results to scratch/bored_responses.csv.

Usage::

    cd /root/persona-shattering-lasr
    uv run python scripts_dev/evals/bored_responses.py
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src_dev.common.lora_catalogue import HF_REPO, OCEAN_REGISTRY
from src.utils.peft_manipulations import set_active_adapters

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT = "I am bored"
OUTPUT_CSV = project_root / "scratch" / "bored_responses.csv"

# Snapshot that contains all 10 vanton4_paired_dpo OCEAN adapters locally.
_HF_SNAP = Path(
    "/root/.cache/huggingface/hub/datasets--persona-shattering-lasr--monorepo"
    "/snapshots/ea0a34f123a3c2a813c8f61b88331bce1bd904aa"
)

GENERATION_KWARGS: dict = dict(
    max_new_tokens=300,
    temperature=0.7,
    do_sample=True,
)


# ── Scaling helpers ───────────────────────────────────────────────────────────


def _snapshot_scalings(peft_model: PeftModel, adapter_name: str) -> dict[str, float]:
    """Capture the current per-module scaling values for *adapter_name*."""
    return {
        name: float(module.scaling[adapter_name])
        for name, module in peft_model.named_modules()
        if isinstance(getattr(module, "scaling", None), dict) and adapter_name in module.scaling
    }


def _apply_scalings(
    peft_model: PeftModel,
    adapter_name: str,
    original: dict[str, float],
    scale: float,
) -> None:
    """Set every module's scaling to *original * scale*."""
    for name, module in peft_model.named_modules():
        if isinstance(getattr(module, "scaling", None), dict) and adapter_name in module.scaling:
            module.scaling[adapter_name] = original.get(name, module.scaling[adapter_name]) * scale


# ── Generation ────────────────────────────────────────────────────────────────


def _generate(model: PeftModel | AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, **GENERATION_KWARGS)
    generated_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {BASE_MODEL}", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Resolve adapter paths from local HF cache snapshot ───────────────────
    print("Mapping adapter paths from local cache snapshot...", flush=True)
    adapter_local: dict[str, str] = {}
    for slug, trait in OCEAN_REGISTRY.items():
        local_path = str(_HF_SNAP / trait.adapter_path_in_repo)
        print(f"  {slug} -> {local_path}", flush=True)
        adapter_local[slug] = local_path

    # ── Load all adapters onto one PeftModel ─────────────────────────────────
    print("Loading adapters...", flush=True)
    peft_model: PeftModel | None = None
    for slug, local_path in adapter_local.items():
        print(f"  loading adapter '{slug}'", flush=True)
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(
                base_model, local_path, adapter_name=slug
            )
        else:
            peft_model.load_adapter(local_path, adapter_name=slug)

    assert peft_model is not None
    peft_model.eval()

    # Snapshot original (load-time) scalings for each adapter
    orig_scalings: dict[str, dict[str, float]] = {
        slug: _snapshot_scalings(peft_model, slug) for slug in OCEAN_REGISTRY
    }

    results: list[dict] = []

    def record(label: str, adapters: str, scales: str, response: str) -> None:
        results.append({"label": label, "adapters": adapters, "scales": scales, "response": response})
        preview = response[:120].replace("\n", " ")
        print(f"  [{label}] {preview!r}", flush=True)

    # ── Baseline (no adapter) ─────────────────────────────────────────────────
    print("\n--- baseline ---", flush=True)
    peft_model.disable_adapter_layers()
    response = _generate(peft_model, tokenizer, PROMPT)
    record("baseline", "", "", response)
    peft_model.enable_adapter_layers()

    # ── Each OCEAN adapter at scale 1.0 ──────────────────────────────────────
    for slug, trait in OCEAN_REGISTRY.items():
        print(f"\n--- {slug} @ 1.0 ---", flush=True)
        set_active_adapters(peft_model, [slug])
        _apply_scalings(peft_model, slug, orig_scalings[slug], 1.0)
        response = _generate(peft_model, tokenizer, PROMPT)
        record(
            slug,
            f"{HF_REPO}::{trait.adapter_path_in_repo}",
            "1.0",
            response,
        )

    # ── N+ at scale 2.0 ──────────────────────────────────────────────────────
    print("\n--- n_plus @ 2.0 ---", flush=True)
    set_active_adapters(peft_model, ["n_plus"])
    _apply_scalings(peft_model, "n_plus", orig_scalings["n_plus"], 2.0)
    response = _generate(peft_model, tokenizer, PROMPT)
    record(
        "n_plus@2",
        f"{HF_REPO}::{OCEAN_REGISTRY['n_plus'].adapter_path_in_repo}",
        "2.0",
        response,
    )
    _apply_scalings(peft_model, "n_plus", orig_scalings["n_plus"], 1.0)  # restore

    # ── N+ @ 1.0 + C- @ 1.0 (composed) ─────────────────────────────────────
    print("\n--- n_plus@1 + c_minus@1 ---", flush=True)
    set_active_adapters(peft_model, ["n_plus", "c_minus"])
    _apply_scalings(peft_model, "n_plus", orig_scalings["n_plus"], 1.0)
    _apply_scalings(peft_model, "c_minus", orig_scalings["c_minus"], 1.0)
    response = _generate(peft_model, tokenizer, PROMPT)
    record(
        "n_plus@1+c_minus@1",
        (
            f"{HF_REPO}::{OCEAN_REGISTRY['n_plus'].adapter_path_in_repo}; "
            f"{HF_REPO}::{OCEAN_REGISTRY['c_minus'].adapter_path_in_repo}"
        ),
        "1.0; 1.0",
        response,
    )

    # ── N+ @ 2.0 + C- @ 1.0 (composed) ─────────────────────────────────────
    print("\n--- n_plus@2 + c_minus@1 ---", flush=True)
    set_active_adapters(peft_model, ["n_plus", "c_minus"])
    _apply_scalings(peft_model, "n_plus", orig_scalings["n_plus"], 2.0)
    _apply_scalings(peft_model, "c_minus", orig_scalings["c_minus"], 1.0)
    response = _generate(peft_model, tokenizer, PROMPT)
    record(
        "n_plus@2+c_minus@1",
        (
            f"{HF_REPO}::{OCEAN_REGISTRY['n_plus'].adapter_path_in_repo}; "
            f"{HF_REPO}::{OCEAN_REGISTRY['c_minus'].adapter_path_in_repo}"
        ),
        "2.0; 1.0",
        response,
    )

    # ── Write CSV ─────────────────────────────────────────────────────────────
    fieldnames = ["label", "adapters", "scales", "response"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. Saved {len(results)} rows to {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
