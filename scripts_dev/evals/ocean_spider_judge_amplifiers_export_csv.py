"""Export raw + headroom judge values for the amplifiers spider plot as CSV.

Reads directly from the already-hydrated local cache (no HF calls) so
you can iterate on plot formatting without re-downloading.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

JUDGE_METRIC_PER_TRAIT: dict[str, str] = {
    "Openness": "openness_v2",
    "Conscientiousness": "conscientiousness_v2",
    "Extraversion": "extraversion_v2",
    "Agreeableness": "agreeableness_v2",
    "Neuroticism": "neuroticism_v2",
}
RATER_IDS = ["gemini_flash_20"]

# ocean_v2 Likert bounds.
JUDGE_SCALE_MIN = -4.0
JUDGE_SCALE_MAX = 4.0

CACHE_ROOT = Path("scratch/ocean_spider/hf_cache_judge")
BASE_MODEL = "llama-3.1-8b-it"
EVAL = "llm_judge_lora_scale_sweep"
FP = "8c7a3f01d3"


def adapter_cell(trait_dir: str, direction: str, version: str) -> Path:
    return (
        CACHE_ROOT
        / f"fine_tuning/{BASE_MODEL}/ocean/{trait_dir}/{direction}/{version}"
        / f"evals/{EVAL}/{FP}/scale_+1.00"
    )


BASELINE_CELL = CACHE_ROOT / f"combos/{BASE_MODEL}/_baseline/{EVAL}/{FP}"

ADAPTERS = [
    {"name": "baseline", "label": "base", "cell_dir": BASELINE_CELL},
    {
        "name": "opn_amp_vanton1",
        "label": "Openness +1×",
        "cell_dir": adapter_cell("openness", "amplifier", "vanton1"),
    },
    {
        "name": "con_amp_v1_souped",
        "label": "Conscientiousness +1×",
        "cell_dir": adapter_cell("conscientiousness", "amplifier", "v1"),
    },
    {
        "name": "ext_amp_v3",
        "label": "Extraversion +1×",
        "cell_dir": adapter_cell("extraversion", "amplifier", "v3"),
    },
    {
        "name": "agr_amp_vanton2",
        "label": "Agreeableness +1×",
        "cell_dir": adapter_cell("agreeableness", "amplifier", "vanton2"),
    },
    {
        "name": "neu_amp_v4",
        "label": "Neuroticism +1×",
        "cell_dir": adapter_cell("neuroticism", "amplifier", "v4"),
    },
]

OUT_CSV = Path("scratch/ocean_spider/ocean_spider_judge_amplifiers_values.csv")


def extract_judge_scores(cell_dir: Path, metric_name: str) -> list[float]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for rater_id in RATER_IDS:
        path = cell_dir / "judge_runs" / rater_id / f"{metric_name}.jsonl"
        if not path.exists():
            continue
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("status") not in {"success", "parse_error"}:
                continue
            score = record.get("score")
            if not isinstance(score, int):
                continue
            key = (rater_id, str(record.get("response_id", "")))
            grouped[key].append(score)
    return [float(statistics.median(v)) for v in grouped.values() if v]


def mean_or_none(xs: list[float]) -> float | None:
    return float(statistics.mean(xs)) if xs else None


raw_by_adapter: dict[str, dict[str, float]] = {}
for entry in ADAPTERS:
    means: dict[str, float] = {}
    for trait, metric in JUDGE_METRIC_PER_TRAIT.items():
        vals = extract_judge_scores(entry["cell_dir"], metric)
        m = mean_or_none(vals)
        if m is not None:
            means[trait] = m
        else:
            print(f"  ⚠ no data for {entry['name']}/{metric} at {entry['cell_dir']}")
    raw_by_adapter[entry["name"]] = means


def headroom(value: float, base: float) -> float:
    delta = value - base
    if delta >= 0:
        hr = JUDGE_SCALE_MAX - base
    else:
        hr = base - JUDGE_SCALE_MIN
    return delta / hr if hr > 0 else 0.0


base_means = raw_by_adapter["baseline"]

with OUT_CSV.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "adapter_name",
        "label",
        "trait",
        "raw_mean",
        "base_mean",
        "delta",
        "headroom",
    ])
    for entry in ADAPTERS:
        name = entry["name"]
        for trait in OCEAN_TRAITS:
            raw_v = raw_by_adapter[name].get(trait)
            base_v = base_means.get(trait)
            if raw_v is None or base_v is None:
                continue
            w.writerow(
                [
                    name,
                    entry["label"],
                    trait,
                    f"{raw_v:.6f}",
                    f"{base_v:.6f}",
                    f"{raw_v - base_v:+.6f}",
                    f"{headroom(raw_v, base_v):+.6f}",
                ]
            )

print(f"✓ wrote {OUT_CSV}")
