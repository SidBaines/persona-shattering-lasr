"""Scale-flip acquiescence test on the v5 Likert questionnaire.

For each of N stratified-sampled rollouts and each assistant model under
test, administer every v5 item TWICE — same item text, only the Likert
**scale labels** are flipped:

  • original:  "1=strongly disagree, 5=strongly agree"
  • flipped:   "1=strongly agree,    5=strongly disagree"

A respondent who actually reads the scale labels should mirror its
answer: ``A_flipped = 6 − A_original``. A respondent who is sycophantic
/ number-anchored / acquiescent will give the same high digit either way
(e.g. answer "5" both times — meaning agree-strongly under the original
labels and disagree-strongly under the flipped labels).

Pipeline (stage-marker pattern, similar to ``psychometric_rollout_fa.py``):

  1. Sample personas from cached B rollouts (stratified by archetype).
  2. Per assistant model: run vLLM logprob-mode inference on each
     (persona × item × scale-orientation) cell. Resumable from JSONL.
  3. Compute acquiescence metrics (mirror error, contradiction rate,
     correlation between A_orig and 6−A_flip, per-dimension breakdown).
  4. Upload everything (config, raw responses, metrics) to
     ``persona-shattering-lasr/monorepo`` under ``evals/acquiescence_test/``.

Outputs land under ``scratch/acquiescence_test/<run_id>/`` mirroring the
HF layout.

Re-run safe: per-model inference writes its progress to disk and resumes
from the partial JSONL on next launch.
"""

from __future__ import annotations

# vLLM 0.20+ tries to use DeepGEMM in its kernel-warmup even on bf16-only
# models, which fails on machines without the (heavy) ``deep_gemm`` package.
# Disable up-front so the workaround is in place before any vLLM import.
import os
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Repo root on ``sys.path`` so ``src_dev.*`` imports work when executed
# directly via ``uv run python scripts_dev/acquiescence_test/...``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src_dev.datasets.core import load_samples  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────
#  Configuration — edit at the top of the file, no CLI flags needed
# ──────────────────────────────────────────────────────────────────────────

SEED = 42
RUN_ID = "acquiescence-v5-N100-seed42"

# Source rollouts (Llama-3.1 × scenarios v2 × user-sim v6, 2,500 personas).
# This is the ``B`` preset that ``psychometric_rollout_fa.py`` already
# generated and cached locally.
ROLLOUT_RUN_DIR = (
    "scratch/psychometric_fa/"
    "rollouts-llama318binstruct-t1.0-15t-2500p-seed436-scenarios_v2-uprompt_v6"
)

# Stratified-sampling: N_PER_ARCHETYPE personas per archetype. With 25
# archetypes that yields 100 personas total. Each persona is one rollout.
N_PER_ARCHETYPE = 4

# v5 questionnaire (100 Likert items, 18 dimensions, 63 fwd / 37 rev keyed).
QUESTIONNAIRE_PATH = (
    _REPO_ROOT
    / "datasets/psychometric_questionnaires/psychometric_questionnaire_v5.json"
)

# Two Likert prompt templates — identical except for the endpoint labels.
# Confirmed by inspecting LIKERT_PHRASINGS["direct"] in
# src_dev/psychometric/item_prompts.py: only 1 and 5 are spelled out;
# 2, 3, 4 are unlabeled. So flipping is a single swap.
LIKERT_PROMPT_ORIGINAL = (
    "On a scale of 1 to 5, how much do you agree with this statement? "
    '1=strongly disagree, 5=strongly agree. Reply with just a number.\n\n"{item_text}"'
)
LIKERT_PROMPT_FLIPPED = (
    "On a scale of 1 to 5, how much do you agree with this statement? "
    '1=strongly agree, 5=strongly disagree. Reply with just a number.\n\n"{item_text}"'
)

# Models under test. ``vllm_extras`` are model-specific overrides into
# ``VllmProviderConfig``.
MODELS_UNDER_TEST: list[dict[str, Any]] = [
    {
        "name": "Qwen/Qwen3.5-9B",
        "slug": "qwen3p5-9b",
        "max_model_len": 32768,
        "vllm_extras": {
            # Hybrid Mamba+attn model; flashinfer's GDN prefill kernel
            # needs nvcc matching torch's CUDA build, which fails on this
            # box (nvcc 12.4 vs torch cu13). Triton sidesteps the JIT.
            "gdn_prefill_backend": "triton",
            # Qwen3.5 is a reasoning model whose default chat template
            # opens with ``<think>\n``; without this it answers reasoning
            # tokens, not Likert digits.
            "chat_template_kwargs": {"enable_thinking": False},
        },
    },
    {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "slug": "llama3p1-8b",
        "max_model_len": 32768,
        "vllm_extras": {},
    },
    {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "slug": "qwen2p5-7b",
        "max_model_len": 32768,
        "vllm_extras": {},
    },
]

# vLLM batching. PERSONAS_PER_BATCH controls how many (persona × item ×
# orientation) cells are stacked into one ``llm.chat`` call.
PERSONAS_PER_BATCH = 32
GPU_MEMORY_UTILIZATION = 0.85
TOP_LOGPROBS = 20
MAX_NEW_TOKENS = 1
LOGPROB_TEMPERATURE = 1.0

# ── Output / HF ───────────────────────────────────────────────────────────
SCRATCH_DIR = _REPO_ROOT / f"scratch/acquiescence_test/{RUN_ID}"
HF_REPO = "persona-shattering-lasr/monorepo"
HF_PATH_PREFIX = f"evals/acquiescence_test/{RUN_ID}"

# Stage selection
DEFAULT_STAGES = ["sample", "inference", "metrics", "upload"]


# ──────────────────────────────────────────────────────────────────────────
#  Stage 1 — stratified persona sampling
# ──────────────────────────────────────────────────────────────────────────

def stage_sample_personas() -> list[dict[str, Any]]:
    """Pick ``N_PER_ARCHETYPE`` rollouts from each archetype, deterministically.

    Returns a list of persona dicts: ``{sample_id, archetype, scenario,
    n_messages, n_assistant_turns}`` plus the loaded ``messages``.
    """
    rng = random.Random(SEED)

    arch_path = Path(ROLLOUT_RUN_DIR) / "archetype_assignments.json"
    scen_path = Path(ROLLOUT_RUN_DIR) / "scenario_assignments.json"
    archetypes = json.loads(arch_path.read_text())
    scenarios = json.loads(scen_path.read_text())

    samples = load_samples(ROLLOUT_RUN_DIR)
    print(f"[Stage 1] Loaded {len(samples)} canonical rollout samples")

    # Group sample indices by archetype. ``samples[i]`` corresponds to
    # ``archetypes[str(i)]`` because the rollout pipeline stores
    # assignments by row index.
    by_arch: dict[str, list[int]] = defaultdict(list)
    for idx in range(len(samples)):
        arch = archetypes.get(str(idx))
        if arch is None:
            continue
        by_arch[arch].append(idx)

    # Stratified sample
    selected_idxs: list[int] = []
    for arch, idxs in sorted(by_arch.items()):
        rng.shuffle(idxs)
        take = idxs[:N_PER_ARCHETYPE]
        selected_idxs.extend(take)

    print(
        f"[Stage 1] Selected {len(selected_idxs)} personas — "
        f"{N_PER_ARCHETYPE}/archetype across {len(by_arch)} archetypes"
    )

    personas = []
    for idx in selected_idxs:
        s = samples[idx]
        msgs = [{"role": m.role, "content": m.content} for m in s.messages]
        n_assistant = sum(1 for m in msgs if m["role"] == "assistant")
        personas.append({
            "row_index": idx,
            "sample_id": s.sample_id,
            "archetype": archetypes.get(str(idx)),
            "scenario": scenarios.get(str(idx)),
            "n_messages": len(msgs),
            "n_assistant_turns": n_assistant,
            "messages": msgs,
        })

    return personas


# ──────────────────────────────────────────────────────────────────────────
#  Stage 2 — vLLM logprob inference per model
# ──────────────────────────────────────────────────────────────────────────

def _row_key(persona_row: int, item_id: str, orientation: str) -> str:
    """Stable identifier for a (persona × item × orientation) inference cell."""
    return f"{persona_row}|{item_id}|{orientation}"


def stage_run_inference_for_model(
    model_cfg: dict[str, Any],
    personas: list[dict[str, Any]],
    items: list[dict[str, Any]],
) -> Path:
    """Run logprob-mode inference for one model on every (persona × item ×
    orientation) cell. Resumes from existing JSONL.

    Returns the path to the model's ``raw_responses.jsonl``.
    """
    out_dir = SCRATCH_DIR / "raw_responses"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{model_cfg['slug']}.jsonl"

    # Set of cells we've already done (so we can resume)
    done: set[str] = set()
    if raw_path.exists():
        with raw_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add(_row_key(r["persona_row_index"], r["item_id"], r["orientation"]))
                except Exception:
                    continue
        print(f"[Stage 2:{model_cfg['slug']}] Resuming: {len(done)} cells already done")

    # Build the full task list — every (persona × item × orientation)
    tasks: list[dict] = []
    for p in personas:
        for it in items:
            for orientation, template in (
                ("original", LIKERT_PROMPT_ORIGINAL),
                ("flipped", LIKERT_PROMPT_FLIPPED),
            ):
                key = _row_key(p["row_index"], str(it["id"]), orientation)
                if key in done:
                    continue
                user_msg = template.format(item_text=it["text"])
                tasks.append({
                    "key": key,
                    "persona_row_index": p["row_index"],
                    "sample_id": p["sample_id"],
                    "archetype": p["archetype"],
                    "scenario": p["scenario"],
                    "item_id": str(it["id"]),
                    "dimension": it["dimension"],
                    "reverse_keyed": it["reverse_keyed"],
                    "orientation": orientation,
                    "item_text": it["text"],
                    "messages": p["messages"] + [{"role": "user", "content": user_msg}],
                })

    if not tasks:
        print(f"[Stage 2:{model_cfg['slug']}] All cells complete; nothing to do.")
        return raw_path

    print(
        f"[Stage 2:{model_cfg['slug']}] {len(tasks)} cells to compute "
        f"({len(personas)} personas × {len(items)} items × 2 orientations)"
    )

    # Lazy import — keeps Stage 1 fast and lets paraphrase-free dry-runs
    # avoid touching torch/vllm.
    from src_dev.inference.config import (
        InferenceConfig,
        GenerationConfig,
        VllmProviderConfig,
    )
    from src_dev.inference.providers.vllm import VllmProvider

    vllm_kwargs = dict(
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=model_cfg["max_model_len"],
    )
    vllm_kwargs.update(model_cfg.get("vllm_extras", {}))
    cfg = InferenceConfig(
        model=model_cfg["name"],
        provider="vllm",
        generation=GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=LOGPROB_TEMPERATURE,
            do_sample=True,
        ),
        vllm=VllmProviderConfig(**vllm_kwargs),
    )
    provider = VllmProvider(cfg)
    print(f"[Stage 2:{model_cfg['slug']}] vLLM engine ready")

    VALID_TOKENS = {"1", "2", "3", "4", "5"}

    t0 = time.time()
    with raw_path.open("a") as f_out:
        for batch_start in range(0, len(tasks), PERSONAS_PER_BATCH):
            batch = tasks[batch_start:batch_start + PERSONAS_PER_BATCH]
            prompts = [t["messages"] for t in batch]
            results = provider.generate_batch_logprobs(
                prompts,
                max_tokens=MAX_NEW_TOKENS,
                top_logprobs=TOP_LOGPROBS,
                temperature=LOGPROB_TEMPERATURE,
            )
            for t, r in zip(batch, results):
                lp = r["logprobs_per_token"][0] if r.get("logprobs_per_token") else {}
                # Convert log-probs → linear probs over the valid Likert
                # answer set. We use ``tok.strip()`` so "5", " 5", "5\n"
                # all map to the same choice.
                probs: dict[str, float] = {}
                for tok, logp in lp.items():
                    s = tok.strip()
                    if s in VALID_TOKENS:
                        probs[s] = probs.get(s, 0.0) + math.exp(logp)
                total_mass = sum(probs.values())
                parsed_choice = (
                    int(max(probs.items(), key=lambda kv: kv[1])[0])
                    if probs else None
                )
                row = {
                    "persona_row_index": t["persona_row_index"],
                    "sample_id": t["sample_id"],
                    "archetype": t["archetype"],
                    "scenario": t["scenario"],
                    "item_id": t["item_id"],
                    "dimension": t["dimension"],
                    "reverse_keyed": t["reverse_keyed"],
                    "orientation": t["orientation"],
                    "item_text": t["item_text"],
                    "parsed_choice": parsed_choice,
                    "choice_mass": total_mass,
                    "probs": probs,
                    "top_logprobs": dict(list(lp.items())[:TOP_LOGPROBS]),
                }
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            f_out.flush()

            done_count = batch_start + len(batch)
            if done_count % (PERSONAS_PER_BATCH * 10) == 0 or done_count == len(tasks):
                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                remaining = len(tasks) - done_count
                eta = remaining / rate if rate > 0 else float("inf")
                print(
                    f"  [{model_cfg['slug']}] {done_count}/{len(tasks)} cells "
                    f"| rate={rate:.1f}/s eta={eta/60:.1f}min"
                )

    print(f"[Stage 2:{model_cfg['slug']}] Wrote {raw_path}")
    return raw_path


# ──────────────────────────────────────────────────────────────────────────
#  Stage 3 — acquiescence metrics
# ──────────────────────────────────────────────────────────────────────────

def _read_responses(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def stage_compute_metrics(
    raw_paths: dict[str, Path],
    personas: list[dict],
    items: list[dict],
) -> dict[str, Any]:
    """Compute per-model scale-flip / acquiescence statistics.

    Key quantities, per (model × persona × item) pair where we have both
    orientations:

      * ``E[A]``                — mean expected answer given the prob dist.
      * ``mirror_error``        — observed ``E[A_orig] + E[A_flip] − 6``.
        For a respondent who actually parses the scale, the two answers
        should sum to 6 (mirror about the midpoint), so this should be 0.
        Positive ⇒ both answers skew high (number-anchored / yes-bias);
        negative ⇒ both skew low.
      * ``contradiction``       — both answers ≥ 4 (i.e. the model agrees
        even though "agree" maps to opposite numeric directions).
      * ``answer_distance``     — ``|A_orig − (6 − A_flip)|`` on the
        argmax; how far the flipped answer is from the expected mirror.
    """
    summary: dict[str, Any] = {
        "n_personas": len(personas),
        "n_items": len(items),
        "models": {},
    }

    def _expected_value(probs: dict[str, float]) -> float:
        """Probability-weighted mean answer (1..5)."""
        if not probs:
            return float("nan")
        return sum(int(k) * v for k, v in probs.items()) / sum(probs.values())

    for slug, path in raw_paths.items():
        rows = _read_responses(path)
        by_cell: dict[tuple[int, str, str], dict] = {
            (r["persona_row_index"], r["item_id"], r["orientation"]): r for r in rows
        }

        n_pairs = 0
        sum_mirror_error = 0.0
        sum_squared_mirror_error = 0.0
        sum_abs_argmax_error = 0.0
        contradictions = 0
        sum_E_orig, sum_E_flip = 0.0, 0.0

        per_dim_mirror: defaultdict[str, list[float]] = defaultdict(list)
        per_dim_contradicts: Counter[str] = Counter()
        per_dim_total: Counter[str] = Counter()
        per_archetype_mirror: defaultdict[str, list[float]] = defaultdict(list)
        per_archetype_contradicts: Counter[str] = Counter()
        per_archetype_total: Counter[str] = Counter()

        # For Pearson correlation between A_orig and 6 − A_flip
        xs, ys = [], []

        for p in personas:
            for it in items:
                iid = str(it["id"])
                orig = by_cell.get((p["row_index"], iid, "original"))
                flip = by_cell.get((p["row_index"], iid, "flipped"))
                if not (orig and flip):
                    continue
                n_pairs += 1
                E_o = _expected_value(orig["probs"])
                E_f = _expected_value(flip["probs"])
                sum_E_orig += E_o
                sum_E_flip += E_f
                mirror_err = (E_o + E_f) - 6.0
                sum_mirror_error += mirror_err
                sum_squared_mirror_error += mirror_err ** 2

                A_o = orig["parsed_choice"] or 0
                A_f = flip["parsed_choice"] or 0
                sum_abs_argmax_error += abs(A_o - (6 - A_f))

                # Contradiction: agrees (≥4) under BOTH orientations,
                # which under the flipped scale means strong disagreement
                # — so the model said "I'm strongly assertive" AND
                # "I'm strongly not assertive" by picking 4/5 on both.
                if A_o >= 4 and A_f >= 4:
                    contradictions += 1
                    per_dim_contradicts[it["dimension"]] += 1
                    per_archetype_contradicts[p["archetype"]] += 1

                per_dim_total[it["dimension"]] += 1
                per_archetype_total[p["archetype"]] += 1
                per_dim_mirror[it["dimension"]].append(mirror_err)
                per_archetype_mirror[p["archetype"]].append(mirror_err)

                xs.append(E_o)
                ys.append(6.0 - E_f)

        # Pearson correlation between E[A_orig] and 6 − E[A_flip].
        # A respondent who reads the scale should have these two
        # quantities track each other (r near +1).
        if len(xs) >= 2:
            n = len(xs)
            mx = sum(xs) / n
            my = sum(ys) / n
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
            den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
            pearson = num / (den_x * den_y) if den_x > 0 and den_y > 0 else float("nan")
        else:
            pearson = float("nan")

        summary["models"][slug] = {
            "n_pairs": n_pairs,
            "mean_E_original": sum_E_orig / n_pairs if n_pairs else None,
            "mean_E_flipped": sum_E_flip / n_pairs if n_pairs else None,
            "mean_mirror_error": sum_mirror_error / n_pairs if n_pairs else None,
            "rmse_mirror_error": math.sqrt(sum_squared_mirror_error / n_pairs) if n_pairs else None,
            "mean_abs_argmax_error": sum_abs_argmax_error / n_pairs if n_pairs else None,
            "contradiction_rate": contradictions / n_pairs if n_pairs else None,
            "pearson_orig_vs_flipped_inverted": pearson,
            "per_dimension_mean_mirror_error": {
                d: sum(v) / len(v) for d, v in per_dim_mirror.items()
            },
            "per_dimension_contradiction_rate": {
                d: per_dim_contradicts[d] / per_dim_total[d]
                for d in per_dim_total
            },
            "per_archetype_mean_mirror_error": {
                a: sum(v) / len(v) for a, v in per_archetype_mirror.items()
            },
            "per_archetype_contradiction_rate": {
                a: per_archetype_contradicts[a] / per_archetype_total[a]
                for a in per_archetype_total
            },
        }

    out_path = SCRATCH_DIR / "metrics.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[Stage 3] Wrote {out_path}")

    print()
    print("=" * 95)
    print("Scale-flip / acquiescence summary")
    print("=" * 95)
    print(
        f"{'model':<15s} {'pairs':>6s} {'E[A_orig]':>10s} {'E[A_flip]':>10s} "
        f"{'mirror_err':>11s} {'rmse_mirror':>12s} {'contradicts':>12s} {'r(orig,6-flip)':>16s}"
    )
    for slug, m in summary["models"].items():
        print(
            f"{slug:<15s} {m['n_pairs']:>6d} "
            f"{m['mean_E_original']:>10.3f} {m['mean_E_flipped']:>10.3f} "
            f"{m['mean_mirror_error']:>+11.3f} {m['rmse_mirror_error']:>12.3f} "
            f"{(m['contradiction_rate'] or 0)*100:>11.1f}% "
            f"{m['pearson_orig_vs_flipped_inverted']:>+16.3f}"
        )
    print()
    print(
        "Reading the table:\n"
        "  E[A_*]         = expected (probability-weighted) Likert answer 1..5\n"
        "  mirror_err     = mean(E[A_orig] + E[A_flip] − 6). 0 = perfect mirror.\n"
        "                   >0 = both answers high (yes-bias / number-anchored)\n"
        "  rmse_mirror    = per-pair RMSE around the perfect-mirror target\n"
        "  contradicts    = % of pairs where model gives ≥4 under BOTH orientations\n"
        "                   (equivalent to 'I strongly agree AND I strongly disagree')\n"
        "  r(orig,6-flip) = Pearson r between E[A_orig] and 6−E[A_flip] across pairs;\n"
        "                   should be near +1 for a model that reads the scale labels."
    )
    return summary


# ──────────────────────────────────────────────────────────────────────────
#  Stage 4 — upload to HuggingFace
# ──────────────────────────────────────────────────────────────────────────

def stage_upload() -> None:
    from src_dev.utils.hf_hub import upload_folder_to_dataset_repo

    upload_folder_to_dataset_repo(
        local_dir=SCRATCH_DIR,
        repo_id=HF_REPO,
        path_in_repo=HF_PATH_PREFIX,
        commit_message=f"acquiescence_test: {RUN_ID}",
    )
    print(f"[Stage 4] Uploaded {SCRATCH_DIR} → {HF_REPO}/{HF_PATH_PREFIX}")


# ──────────────────────────────────────────────────────────────────────────
#  Driver
# ──────────────────────────────────────────────────────────────────────────

def _save_config_snapshot(personas: list[dict]) -> None:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {
        "run_id": RUN_ID,
        "seed": SEED,
        "rollout_run_dir": ROLLOUT_RUN_DIR,
        "n_per_archetype": N_PER_ARCHETYPE,
        "n_personas": len(personas),
        "questionnaire_path": str(QUESTIONNAIRE_PATH.relative_to(_REPO_ROOT)),
        "likert_prompt_original": LIKERT_PROMPT_ORIGINAL,
        "likert_prompt_flipped": LIKERT_PROMPT_FLIPPED,
        "models_under_test": MODELS_UNDER_TEST,
        "personas_per_batch": PERSONAS_PER_BATCH,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "top_logprobs": TOP_LOGPROBS,
    }
    (SCRATCH_DIR / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False)
    )
    (SCRATCH_DIR / "selected_personas.json").write_text(
        json.dumps(
            [{k: v for k, v in p.items() if k != "messages"} for p in personas],
            indent=2,
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stages",
        nargs="+",
        default=DEFAULT_STAGES,
        choices=DEFAULT_STAGES,
        help="Which stages to run (default: all).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Restrict Stage 2 to a subset of model slugs (default: all).",
    )
    args = parser.parse_args()

    print(f"Run ID: {RUN_ID}")
    print(f"Output: {SCRATCH_DIR}")
    print(f"Stages: {args.stages}")
    print()

    items = json.loads(QUESTIONNAIRE_PATH.read_text())["items"]
    print(f"Loaded {len(items)} v5 items")

    personas: list[dict] = []
    if "sample" in args.stages:
        personas = stage_sample_personas()
        _save_config_snapshot(personas)

    raw_paths: dict[str, Path] = {}
    if "inference" in args.stages:
        if not personas:
            personas = stage_sample_personas()
        models = MODELS_UNDER_TEST
        if args.models:
            models = [m for m in models if m["slug"] in set(args.models)]
        for m in models:
            raw_paths[m["slug"]] = stage_run_inference_for_model(m, personas, items)
    else:
        for m in MODELS_UNDER_TEST:
            p = SCRATCH_DIR / "raw_responses" / f"{m['slug']}.jsonl"
            if p.exists():
                raw_paths[m["slug"]] = p

    if "metrics" in args.stages:
        if not personas:
            personas = stage_sample_personas()
        if not raw_paths:
            raise RuntimeError(
                "metrics stage requested but no raw_responses files exist; "
                "run with --stages inference first"
            )
        stage_compute_metrics(raw_paths, personas, items)

    if "upload" in args.stages:
        stage_upload()


if __name__ == "__main__":
    main()
