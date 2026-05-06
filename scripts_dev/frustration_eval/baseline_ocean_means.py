"""Fetch Gemma-3-27b-IT's baseline mean OCEAN trait scores from HF.

All five OCEAN baselines now live as cells produced by the canonical
``llm_judge_lora_scale_sweep`` runner under

    combos/gemma-3-27b-it/_baseline/llm_judge_lora_scale_sweep/<fp>/judge_runs/qwen3_235b/<trait>.jsonl

Each fingerprint corresponds to a different ``data/ocean_open_ended/<trait>.jsonl``
prompt set (matching the canonical per-trait baseline methodology).

Usage::

    means = get_baseline_means()  # → {trait: {mean, sem, n}}
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

HF_REPO = "persona-shattering-lasr/monorepo"
BASE_PREFIX = "combos/gemma-3-27b-it/_baseline/llm_judge_lora_scale_sweep"
LOCAL_DL = Path("scratch/_baseline_dl")

# trait_metric_name → fingerprint of the baseline cell that judged that trait.
TRAIT_FINGERPRINT: dict[str, str] = {
    "openness_v2":          "c505acf024",
    "conscientiousness_v2": "5b60ecfd83",
    "extraversion_v2":      "e83ca4f0ce",
    "agreeableness_v2":     "6bf884987b",
    "neuroticism_v2":       "a763980e08",
}


def _judge_path_in_repo(fp: str, trait: str) -> str:
    return f"{BASE_PREFIX}/{fp}/judge_runs/qwen3_235b/{trait}.jsonl"


def _local_judge_path(fp: str, trait: str) -> Path:
    return LOCAL_DL / _judge_path_in_repo(fp, trait)


def _ensure(fp: str, trait: str) -> Path:
    out = _local_judge_path(fp, trait)
    if out.exists() and out.stat().st_size > 0:
        return out
    hf_hub_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        filename=_judge_path_in_repo(fp, trait),
        local_dir=str(LOCAL_DL),
    )
    return out


def _summary(path: Path) -> dict[str, float]:
    rows = [json.loads(l) for l in open(path)]
    scores = [r["score"] for r in rows
              if r.get("status") == "success" and r.get("score") is not None]
    if not scores:
        raise ValueError(f"No successful scores in {path}")
    n = len(scores)
    mean = statistics.mean(scores)
    sd = statistics.pstdev(scores) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n > 1 else float("nan")
    return {"mean": mean, "sd": sd, "sem": sem, "n": n}


def get_baseline_means() -> dict[str, dict[str, float]]:
    """Return ``{trait: {mean, sd, sem, n}}`` for all 5 OCEAN traits."""
    out: dict[str, dict[str, float]] = {}
    for trait, fp in TRAIT_FINGERPRINT.items():
        local = _ensure(fp, trait)
        out[trait] = _summary(local)
        logger.info("[%s] mean=%+.3f sd=%.3f sem=%.3f n=%d (fp=%s)",
                    trait, out[trait]["mean"], out[trait]["sd"],
                    out[trait]["sem"], out[trait]["n"], fp)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        stream=sys.stderr)
    print(json.dumps(get_baseline_means(), indent=2))


if __name__ == "__main__":
    main()
