"""Diagnostic: scale-flip test on each model *without* the rollout context.

If the with-rollout acquiescence (98% on Qwen3.5-9B vs ~0% on Llama-3.1
/ Qwen2.5) is induced by the long persona-rollout context (H2), then
asking the same 100 v5 items + the two scale orientations *directly*
(no conversation prefix, no system prompt) should produce healthy
mirror behaviour on Qwen3.5-9B too — and the other two models should
stay ~clean.

For each model under test (default: all of MODELS_UNDER_TEST), runs
100 items × 2 orientations = 200 cells with no preceding messages, then
prints a side-by-side table comparing this no-rollout probe to the
with-rollout result from the main run.

Usage::

    uv run python scripts_dev/acquiescence_test/probe_no_rollout.py            # all models
    uv run python scripts_dev/acquiescence_test/probe_no_rollout.py --models qwen3p5-9b
"""
from __future__ import annotations

import os
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")

import argparse
import json
import math
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the prompt templates and model list from the main acquiescence test.
sys.path.insert(0, str(Path(__file__).parent))
from run_acquiescence_test import (  # noqa: E402
    LIKERT_PROMPT_ORIGINAL,
    LIKERT_PROMPT_FLIPPED,
    QUESTIONNAIRE_PATH,
    MODELS_UNDER_TEST,
    SCRATCH_DIR as MAIN_SCRATCH_DIR,
)

# With-rollout reference numbers from the main run (read from disk so
# this stays in sync if the main run is re-computed).
_MAIN_METRICS = MAIN_SCRATCH_DIR / "metrics.json"


def _probe_one_model(model_cfg: dict, items: list[dict]) -> dict:
    from src_dev.inference.config import (
        InferenceConfig,
        GenerationConfig,
        VllmProviderConfig,
    )
    from src_dev.inference.providers.vllm import VllmProvider

    vllm_kwargs = dict(
        gpu_memory_utilization=0.85,
        max_model_len=4096,
    )
    vllm_kwargs.update(model_cfg.get("vllm_extras", {}))
    cfg = InferenceConfig(
        model=model_cfg["name"],
        provider="vllm",
        generation=GenerationConfig(max_new_tokens=1, temperature=1.0, do_sample=True),
        vllm=VllmProviderConfig(**vllm_kwargs),
    )
    provider = VllmProvider(cfg)
    print(f"[{model_cfg['slug']}] vLLM engine ready")

    VALID = {"1", "2", "3", "4", "5"}

    tasks = []
    for it in items:
        for orientation, template in (
            ("original", LIKERT_PROMPT_ORIGINAL),
            ("flipped", LIKERT_PROMPT_FLIPPED),
        ):
            tasks.append({
                "item_id": it["id"],
                "orientation": orientation,
                "messages": [{"role": "user", "content": template.format(item_text=it["text"])}],
            })

    print(f"[{model_cfg['slug']}] running {len(tasks)} cells (no rollout context)…")
    prompts = [t["messages"] for t in tasks]
    results = provider.generate_batch_logprobs(
        prompts, max_tokens=1, top_logprobs=20, temperature=1.0
    )

    by_item: dict = {}
    for t, r in zip(tasks, results):
        lp = r["logprobs_per_token"][0] if r.get("logprobs_per_token") else {}
        probs: dict[str, float] = {}
        for tok, logp in lp.items():
            s = tok.strip()
            if s in VALID:
                probs[s] = probs.get(s, 0.0) + math.exp(logp)
        E = sum(int(k) * v for k, v in probs.items()) / max(sum(probs.values()), 1e-9)
        argmax = int(max(probs.items(), key=lambda kv: kv[1])[0]) if probs else None
        by_item.setdefault(t["item_id"], {})[t["orientation"]] = {
            "E": E,
            "argmax": argmax,
            "probs": probs,
        }

    n = 0
    sum_mirror_err = sum_sq_mirror_err = 0.0
    contradicts = 0
    sum_E_orig = sum_E_flip = 0.0
    xs, ys = [], []
    for both in by_item.values():
        if "original" not in both or "flipped" not in both:
            continue
        n += 1
        E_o = both["original"]["E"]
        E_f = both["flipped"]["E"]
        sum_E_orig += E_o
        sum_E_flip += E_f
        me = (E_o + E_f) - 6.0
        sum_mirror_err += me
        sum_sq_mirror_err += me ** 2
        A_o = both["original"]["argmax"]
        A_f = both["flipped"]["argmax"]
        if A_o is not None and A_f is not None and A_o >= 4 and A_f >= 4:
            contradicts += 1
        xs.append(E_o)
        ys.append(6.0 - E_f)

    rmse = math.sqrt(sum_sq_mirror_err / n) if n else float("nan")
    pearson = float("nan")
    if n >= 2:
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        dy = math.sqrt(sum((y - my) ** 2 for y in ys))
        pearson = num / (dx * dy) if dx > 0 and dy > 0 else float("nan")

    return {
        "n_items": n,
        "mean_E_original": sum_E_orig / n if n else None,
        "mean_E_flipped": sum_E_flip / n if n else None,
        "mean_mirror_error": sum_mirror_err / n if n else None,
        "rmse_mirror_error": rmse,
        "contradiction_rate": contradicts / n if n else None,
        "pearson_orig_vs_flipped_inverted": pearson,
        "by_item": by_item,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of slugs (default: all from MODELS_UNDER_TEST).")
    args = parser.parse_args()

    items = json.loads(QUESTIONNAIRE_PATH.read_text())["items"]
    print(f"Loaded {len(items)} v5 items")

    models = MODELS_UNDER_TEST
    if args.models:
        models = [m for m in models if m["slug"] in set(args.models)]

    results: dict[str, dict] = {}
    for m in models:
        r = _probe_one_model(m, items)
        results[m["slug"]] = r
        out = _REPO_ROOT / f"scratch/acquiescence_test/_no_rollout_probe_{m['slug']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, indent=2, ensure_ascii=False))
        print(f"[{m['slug']}] wrote {out}")

    # With-rollout reference (from the main metrics.json on disk)
    with_rollout = {}
    if _MAIN_METRICS.exists():
        with_rollout = json.loads(_MAIN_METRICS.read_text()).get("models", {})

    print()
    print("=" * 96)
    print("Scale-flip / acquiescence — WITH vs NO rollout context")
    print("=" * 96)
    print(f"{'model':<14s}  {'context':<12s}  {'pairs':>6s}  {'E[orig]':>8s}  {'E[flip]':>8s}  {'mirr_err':>9s}  {'rmse':>6s}  {'contradicts':>12s}  {'r(o,6-f)':>9s}")
    for slug, r in results.items():
        wr = with_rollout.get(slug, {})
        for label, m in (("with rollout", wr), ("NO rollout", r)):
            if not m:
                continue
            n = m.get("n_pairs") or m.get("n_items") or 0
            print(
                f"{slug:<14s}  {label:<12s}  {n:>6d}  "
                f"{m.get('mean_E_original') or 0:>8.3f}  "
                f"{m.get('mean_E_flipped') or 0:>8.3f}  "
                f"{m.get('mean_mirror_error') or 0:>+9.3f}  "
                f"{m.get('rmse_mirror_error') or 0:>6.3f}  "
                f"{(m.get('contradiction_rate') or 0)*100:>11.1f}%  "
                f"{m.get('pearson_orig_vs_flipped_inverted') or 0:>+9.3f}"
            )
        print("-" * 96)


if __name__ == "__main__":
    main()
