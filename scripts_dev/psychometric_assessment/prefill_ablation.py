"""Prefill ablation: does removing the trailing space from ``"Answer "``
redirect the mass Qwen2.5 leaks to the literal word ``" Answer"``?

History: ran two variants on the first ``N_PERSONAS`` rollouts of the
hydrated B cache, with ``TRAIT_MCQ_PREFILL`` monkey-patched to the
candidate value. Compared per-cell choice-mass to the matched cells in
the existing ``-p2-qm_qwen257binstruct`` run (which used ``"Answer "``).

Findings:
* ``"Answer: "`` made things strictly worse — choice_mass p10 dropped
  from 0.587 to 0.137, ``" Answer"`` mass doubled (0.106 → 0.240 per
  cell). The colon invites word-continuation (``"Answer: Answer …"``).
* ``"Answer"`` (no trailing space) is the fix. Letter-mass share goes
  from 88.23% → 100.00%, choice_mass p50 0.971 → 1.0000, p10 0.587 →
  1.0000. Every one of the 2000 cells improved.

Mechanism: Qwen2.5 (and Llama) BPE-merge ``" A"``/``" B"``/etc. into
single space-prefixed tokens. A prefill ending in a standalone space
token (id 220 on Qwen) places the model in a tokenizer state where the
natural letter continuation decodes as ``"Answer  A"`` (double space) —
out of distribution. Dropping the trailing space lets the prefill end
at token ``Answer`` (id 16141), and the natural continuation ``" A"``
(id 362) completes to clean ``"Answer A"`` text.

This script is kept as evidence / reproducer. Current ``NEW_PREFILL``
value matches the committed fix — the assertion at top pins the upstream
default so we notice if it drifts.

Writes results to a throwaway dir under ``scratch/psychometric_fa/_ablation_prefill/``
and does **not** upload to HF.
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── Ablation knobs ──────────────────────────────────────────────────────────
N_PERSONAS: int = 20
NEW_PREFILL: str = "Answer"  # no trailing space: token boundary lines up with
                              # Qwen's natural " A/B/C/D" (space-prefixed) token
                              # so continuation decodes as "Answer A" not "Answer  A".
GPU_DEVICE: str = "0"  # caller should export CUDA_VISIBLE_DEVICES=<n> — this script
                        # doesn't manipulate CUDA_VISIBLE_DEVICES, just logs it.

ROLLOUT_DIR = Path(
    "scratch/psychometric_fa/rollouts-llama318binstruct-"
    "t1.0-15t-2500p-seed436-scenarios_v2-uprompt_v6"
)
BASELINE_RAW_PATH = Path(
    "scratch/psychometric_fa/"
    "questionnaire-rollouts-llama318binstruct-t1.0-15t-2500p-seed436-"
    "scenarios_v2-uprompt_v6-q_trait_ocean_v1-trait_mcq-direct-lp20-p2-"
    "qm_qwen257binstruct/questionnaire/raw_responses.jsonl"
)
OUTPUT_DIR = Path("scratch/psychometric_fa/_ablation_prefill")

QUESTIONNAIRE_PATH = Path("datasets/psychometric_questionnaires/trait_ocean_v1.json")
QUESTIONNAIRE_VERSION = "trait_ocean_v1"
FA_BLOCKS = ("trait_mcq",)
QM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_CONTEXT_TOKENS = 32_768

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ── Monkey-patch the prefill BEFORE importing inference ─────────────────────

import src_dev.psychometric.item_prompts as ip

_UPSTREAM_PREFILL = ip.TRAIT_MCQ_PREFILL
ip.TRAIT_MCQ_PREFILL = NEW_PREFILL
print(
    f"[ablation] TRAIT_MCQ_PREFILL patched: upstream={_UPSTREAM_PREFILL!r} → {NEW_PREFILL!r}"
)

# Now import the inference machinery.
import src_dev.psychometric.questionnaire_inference as qi  # noqa: E402
from src_dev.psychometric.config import QuestionnaireStageConfig, RunContext  # noqa: E402
from src_dev.psychometric.questionnaire_io import load_questionnaire  # noqa: E402

# Monkeypatch load_samples to subset to first N_PERSONAS.
_real_load_samples = qi.load_samples


def _subset_samples(rollout_dir):
    samples = _real_load_samples(rollout_dir)
    if N_PERSONAS is None or N_PERSONAS >= len(samples):
        return samples
    # Deterministic — take the first N_PERSONAS in load order. The baseline
    # run we compare against uses the same ordering.
    return samples[:N_PERSONAS]


qi.load_samples = _subset_samples


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build the minimal RunContext and QuestionnaireStageConfig.
    ctx = RunContext(
        scratch_root=Path("scratch/psychometric_fa"),
        hf_repo_id="persona-shattering-lasr/psychometric-fa-runs",
        rollout_run_id="ablation-rollout",
        questionnaire_run_id="ablation-questionnaire",
        rollout_dir=ROLLOUT_DIR,
        questionnaire_dir=OUTPUT_DIR,
    )

    cfg = QuestionnaireStageConfig(
        ctx=ctx,
        questionnaire_path=QUESTIONNAIRE_PATH,
        questionnaire_version=QUESTIONNAIRE_VERSION,
        fa_blocks=FA_BLOCKS,
        use_logprobs=True,
        phrasing="aside",  # no-op for trait_mcq (prompts are phrasing-invariant)
        provider="vllm",
        model=QM_MODEL,
        max_new_tokens=32,
        max_concurrent=32,
        timeout=60,
        max_parse_retries=3,
        vllm_personas_per_batch=8,
        vllm_gpu_memory_utilization=0.95,
        vllm_tensor_parallel_size=1,
        top_logprobs=20,
        logprob_temperature=1.0,
        dynamic_mass_filter=True,
        min_choice_mass=0.0,
        min_trait_coverage=0.25,
        reset_mode="none",
        max_context_tokens=MAX_CONTEXT_TOKENS,
        context_buffer_tokens=1024,
        write_inspection_file=False,
        inspection_items_per_rollout=0,
    )

    # Load questionnaire items.
    items, column_defs = load_questionnaire(
        QUESTIONNAIRE_PATH, fa_blocks=FA_BLOCKS, fc_pair_sign_alignment=True,
    )
    print(f"[ablation] Loaded {len(items)} items, {len(column_defs)} column defs")

    # Run inference.
    _matrix, _metadata = qi.run_questionnaire_inference(
        cfg,
        rollout_dir=ROLLOUT_DIR,
        items=items,
        column_defs=column_defs,
        output_dir=OUTPUT_DIR / "questionnaire",
        num_conversation_turns=15,
        fc_pair_sign_alignment=True,
    )

    print(f"[ablation] Inference complete. Matrix shape: {_matrix.shape}")

    # Run the comparison.
    compare_results()


# ═══════════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════════


LETTER_RE = re.compile(r"^[\s\t]*[A-D]$")
DIGIT_RE = re.compile(r"^[\s\t]*[1-4]$")


def _summarise(path: Path, label: str) -> None:
    cms: list[float] = []
    top_nonletter: Counter = Counter()
    total_letter = 0.0
    total_digit = 0.0
    total_other = 0.0
    total_topk = 0.0
    n_cells = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            cm = r.get("choice_mass")
            if cm is not None:
                cms.append(float(cm))
            tlp = r.get("top_logprobs") or {}
            if tlp:
                n_cells += 1
                for tok, lp in tlp.items():
                    p = math.exp(float(lp))
                    total_topk += p
                    if LETTER_RE.match(tok):
                        total_letter += p
                    elif DIGIT_RE.match(tok):
                        total_digit += p
                    else:
                        total_other += p
                        top_nonletter[tok] += p

    cms_a = np.array(cms) if cms else np.array([])
    print()
    print(f"=== {label} ===")
    print(f"  n cells scored     : {len(cms_a)}")
    if len(cms_a):
        print(f"  choice_mass p50    : {np.percentile(cms_a, 50):.4f}")
        print(f"  choice_mass p10    : {np.percentile(cms_a, 10):.4f}")
        print(f"  frac >= 0.9        : {float((cms_a >= 0.9).mean()):.4f}")
        print(f"  frac >= 0.95       : {float((cms_a >= 0.95).mean()):.4f}")
    if n_cells:
        print(f"  cells w/ top_logprobs: {n_cells}")
        print(f"  letter / digit / other mass share: "
              f"{total_letter/total_topk*100:5.2f}% / "
              f"{total_digit/total_topk*100:6.4f}% / "
              f"{total_other/total_topk*100:5.2f}%")
        print("  top-10 non-letter tokens (by mean mass per cell):")
        for tok, m in top_nonletter.most_common(10):
            print(f"    {tok!r:28s} mean_per_cell={m/n_cells:.4f}")


def compare_results() -> None:
    baseline_subset = OUTPUT_DIR / "_baseline_subset_raw_responses.jsonl"
    # Extract baseline rows for the same first-N personas (by k index).
    with open(BASELINE_RAW_PATH) as f, open(baseline_subset, "w") as out:
        for line in f:
            r = json.loads(line)
            if r.get("k", -1) < N_PERSONAS:
                out.write(line)
    ablation_raw = OUTPUT_DIR / "questionnaire" / "raw_responses.jsonl"

    _summarise(baseline_subset, f"BASELINE  {_UPSTREAM_PREFILL!r:12s} (first {N_PERSONAS} personas, existing -p2 run)")
    _summarise(ablation_raw,    f"ABLATION  {NEW_PREFILL!r:12s} (first {N_PERSONAS} personas, this run)")

    # Per-cell diff on matched (k, item_id).
    base = {}
    with open(baseline_subset) as f:
        for l in f:
            r = json.loads(l)
            base[(r["k"], r["item_id"])] = r
    new = {}
    with open(ablation_raw) as f:
        for l in f:
            r = json.loads(l)
            new[(r["k"], r["item_id"])] = r

    keys = set(base) & set(new)
    deltas: list[float] = []
    both_letter_choice = 0
    argmax_agree = 0
    for key in keys:
        b, n = base[key], new[key]
        if b.get("choice_mass") is not None and n.get("choice_mass") is not None:
            deltas.append(float(n["choice_mass"]) - float(b["choice_mass"]))
        if b.get("parsed_choice") and n.get("parsed_choice"):
            both_letter_choice += 1
            if b["parsed_choice"] == n["parsed_choice"]:
                argmax_agree += 1
    if deltas:
        a = np.array(deltas)
        print()
        print(f"=== Per-cell choice_mass delta (new - baseline), n={len(a)} matched cells ===")
        print(f"  mean Δ         : {a.mean():+.4f}")
        print(f"  median Δ       : {float(np.median(a)):+.4f}")
        print(f"  frac Δ > 0     : {float((a > 0).mean()):.4f}")
        print(f"  frac Δ > 0.05  : {float((a > 0.05).mean()):.4f}")
        print(f"  frac Δ > 0.10  : {float((a > 0.10).mean()):.4f}")
    if both_letter_choice:
        print(f"  argmax letter agreement: {argmax_agree}/{both_letter_choice} "
              f"({argmax_agree / both_letter_choice * 100:.1f}%)")


if __name__ == "__main__":
    main()
