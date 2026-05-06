"""Choice-mass diagnostic for v7 fc_pair prefills on cached B rollouts.

Runs a small vLLM questionnaire pass with Qwen2.5-7B-Instruct and reports how
much first-token top-k probability mass lands on A/B choice tokens. This is a
pre-flight check for the BPE-boundary issue seen with trailing-space prefills.
Outputs are throwaway local artifacts under scratch/psychometric_fa/.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src_dev.psychometric.config import QuestionnaireStageConfig, RunContext
from src_dev.psychometric.questionnaire_io import load_questionnaire
import src_dev.psychometric.questionnaire_inference as qi

SEED = 436
random.seed(SEED)
np.random.seed(SEED)

ROLLOUT_DIR = Path(
    "scratch/psychometric_fa/"
    "rollouts-llama318binstruct-t1.0-15t-2500p-seed436-scenarios_v2-uprompt_v6"
)
QUESTIONNAIRE_PATH = Path(
    "datasets/psychometric_questionnaires/psychometric_questionnaire_v7_fc_pair.json"
)
OUT_ROOT = Path("scratch/psychometric_fa/_ablation_v7_fc_prefill")
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

LETTER_RE = re.compile(r"^[\s\t]*[A-B]$")
DIGIT_RE = re.compile(r"^[\s\t]*[1-2]$")


def _summarise(path: Path, label: str) -> None:
    cms: list[float] = []
    top_nonchoice: Counter[str] = Counter()
    total_letter = 0.0
    total_digit = 0.0
    total_other = 0.0
    total_topk = 0.0
    n_cells = 0
    choices: Counter[str] = Counter()

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("choice_mass") is not None:
                cms.append(float(row["choice_mass"]))
            if row.get("parsed_choice"):
                choices[str(row["parsed_choice"])] += 1
            top_logprobs = row.get("top_logprobs") or {}
            if not top_logprobs:
                continue
            n_cells += 1
            for token, logprob in top_logprobs.items():
                prob = math.exp(float(logprob))
                total_topk += prob
                if LETTER_RE.match(token):
                    total_letter += prob
                elif DIGIT_RE.match(token):
                    total_digit += prob
                else:
                    total_other += prob
                    top_nonchoice[token] += prob

    arr = np.array(cms)
    print(f"\n=== {label} ===")
    print(f"n scored cells: {len(arr)}")
    if len(arr):
        for pct in (1, 5, 10, 25, 50, 75, 90, 95, 99):
            print(f"choice_mass p{pct:02d}: {np.percentile(arr, pct):.4f}")
        print(f"frac >= .90: {float((arr >= 0.9).mean()):.4f}")
        print(f"frac >= .95: {float((arr >= 0.95).mean()):.4f}")
        print(f"mean: {float(arr.mean()):.4f}")
    print(f"choices: {dict(choices)}")
    if n_cells and total_topk:
        print(
            "letter / digit / other mass share of observed top-k: "
            f"{total_letter / total_topk * 100:.2f}% / "
            f"{total_digit / total_topk * 100:.4f}% / "
            f"{total_other / total_topk * 100:.2f}%"
        )
        print("top non-choice tokens:")
        for token, mass in top_nonchoice.most_common(10):
            print(f"  {token!r:24s} mean_per_cell={mass / n_cells:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-personas", type=int, default=8)
    parser.add_argument(
        "--prefill",
        default=None,
        help="Assistant prefill to apply to every v7 fc_pair item.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--label", default="no_trailing_space")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()

    load_dotenv(".env")
    random.seed(SEED)
    np.random.seed(SEED)

    real_load_samples = qi.load_samples

    def subset_samples(rollout_dir: Path):
        return real_load_samples(rollout_dir)[: args.n_personas]

    qi.load_samples = subset_samples

    items, column_defs = load_questionnaire(
        QUESTIONNAIRE_PATH,
        fa_blocks=("fc_pair",),
        fc_pair_sign_alignment=True,
    )
    for item in items:
        if item.get("type") == "fc_pair" and args.prefill is not None:
            item["prefill"] = args.prefill

    out_dir = OUT_ROOT / args.label
    if out_dir.exists():
        shutil.rmtree(out_dir)

    ctx = RunContext(
        scratch_root=Path("scratch/psychometric_fa"),
        hf_repo_id="persona-shattering-lasr/psychometric-fa-runs",
        rollout_run_id="rollouts-llama318binstruct-t1.0-15t-2500p-seed436-scenarios_v2-uprompt_v6",
        questionnaire_run_id=f"ablation-v7-fc-{args.label}-qwen25",
        rollout_dir=ROLLOUT_DIR,
        questionnaire_dir=out_dir,
    )
    cfg = QuestionnaireStageConfig(
        ctx=ctx,
        questionnaire_path=QUESTIONNAIRE_PATH,
        questionnaire_version="v7_fc_pair",
        fa_blocks=("fc_pair",),
        use_logprobs=True,
        phrasing="direct",
        provider="vllm",
        model=args.model,
        max_new_tokens=32,
        max_concurrent=32,
        timeout=60,
        max_parse_retries=3,
        vllm_personas_per_batch=8,
        vllm_gpu_memory_utilization=args.gpu_memory_utilization,
        vllm_tensor_parallel_size=1,
        top_logprobs=20,
        logprob_temperature=1.0,
        dynamic_mass_filter=True,
        min_choice_mass=0.0,
        min_trait_coverage=0.25,
        reset_mode="none",
        max_context_tokens=32768,
        context_buffer_tokens=1024,
        write_inspection_file=False,
        inspection_items_per_rollout=0,
    )

    print(
        f"Loaded {len(items)} v7 fc_pair items; testing first "
        f"{args.n_personas} B rollouts on {args.model}"
    )
    active_prefill = items[0].get("prefill") if items else None
    print(f"Prefill: {active_prefill!r}")
    matrix, metadata = qi.run_questionnaire_inference(
        cfg,
        rollout_dir=ROLLOUT_DIR,
        items=items,
        column_defs=column_defs,
        output_dir=out_dir / "questionnaire",
        num_conversation_turns=15,
        fc_pair_sign_alignment=True,
    )
    print(f"Complete: matrix={matrix.shape}, metadata={len(metadata)}")
    _summarise(out_dir / "questionnaire" / "raw_responses.jsonl", args.label)


if __name__ == "__main__":
    main()
