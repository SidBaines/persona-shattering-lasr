"""Mirror the o_plus × n_plus heatmap cells into evals/heatmaps_o_n on HF.

The runner_cells driver writes each cell to its canonical path
(``combos/...`` / ``fine_tuning/.../scale_*`` / ``combos/.../_baseline/...``).
This script copies those 25 cells per heatmap into a flat, self-contained
bundle::

    evals/heatmaps_o_n/
      on_openness/
        <cell_label>/
          rollouts/...
          judge_runs/qwen3_235b/openness_v2.jsonl
          judge_runs/qwen3_235b/better_coherence_judge.jsonl
      on_neuroticism/
        <cell_label>/
          rollouts/...
          judge_runs/qwen3_235b/neuroticism_v2.jsonl
          judge_runs/qwen3_235b/better_coherence_judge.jsonl

The plotting script (``paper_main_o_n_soup_heatmaps.py``) reads from this
bundle directly, so the figure is reproducible from one HF subtree.

Usage::

    uv run python -m scripts_dev.visualisations.bundle_o_n_heatmaps
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src_dev.evals.cell_sweep.cell_identity import (
    AdapterSpec,
    CanonicalCell,
    format_scale,
)
from src_dev.utils.hf_hub import (
    download_path_to_dir,
    upload_folder_to_dataset_repo,
)

HF_REPO_ID = "persona-shattering-lasr/monorepo"
MODEL_SLUG = "llama-3.1-8b-it"
EVAL_NAME = "llm_judge_lora_scale_sweep"
RATER_ID = "qwen3_235b"

# Adapter specs (must match the configs in
# scripts_dev/evals/llm_judge_sweep/configs/vanton4_paired_dpo/o_plus_x_n_plus_on_*).
ADAPTER_O_PLUS = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/openness/amplifier/vanton4_paired_dpo"
    "/lora/openness_amplifying_full_vanton4-persona"
)
ADAPTER_N_PLUS = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/neuroticism/amplifier/vanton4_paired_dpo"
    "/lora/neuroticism_amplifying_full_vanton4-persona"
)

SCALES = [-2.0, -1.0, 0.0, 1.0, 2.0]

# (subdir under evals/heatmaps_o_n, rollout fingerprint, judged trait metric)
HEATMAPS = [
    ("on_openness", "1817b5cf78", "openness_v2"),
    ("on_neuroticism", "8b01e9fa2c", "neuroticism_v2"),
]
COHERENCE_METRIC = "better_coherence_judge"

LOCAL_BUNDLE_ROOT = project_root / "scratch" / "heatmaps_o_n_bundle"
TARGET_PATH_IN_REPO = "evals/heatmaps_o_n"


def _enumerate_cells() -> list[CanonicalCell]:
    """The 25 canonical cells of the 5×5 (o_plus, n_plus) sweep."""
    cells = []
    for o_scale in SCALES:
        for n_scale in SCALES:
            cells.append(
                CanonicalCell.from_scales(
                    [(ADAPTER_O_PLUS, o_scale), (ADAPTER_N_PLUS, n_scale)]
                )
            )
    # Drop duplicates that collapse to the same canonical cell (e.g. baseline).
    seen: dict[CanonicalCell, None] = {}
    for c in cells:
        seen[c] = None
    return list(seen.keys())


def _hydrate_one_cell(cell: CanonicalCell, fingerprint: str, dest_dir: Path) -> None:
    """Pull a cell's full subtree (rollouts + judge_runs + cell_info) into dest_dir."""
    src_path = cell.hf_dir(MODEL_SLUG, EVAL_NAME, fingerprint)
    dest_dir.mkdir(parents=True, exist_ok=True)
    download_path_to_dir(
        repo_id=HF_REPO_ID,
        path_in_repo=src_path,
        target_dir=dest_dir,
        allow_patterns=[
            "rollouts/*",
            f"judge_runs/{RATER_ID}/*.jsonl",
            "cell_info.json",
        ],
    )


def _bundle_heatmap(subdir: str, fingerprint: str, judged_metric: str) -> Path:
    """Materialise one heatmap's 25 cells under <bundle>/<subdir>/<cell_label>/."""
    print(f"\n[bundle] {subdir} (fp={fingerprint}, metric={judged_metric})")
    bundle_dir = LOCAL_BUNDLE_ROOT / subdir
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    cells = _enumerate_cells()
    for cell in cells:
        label = cell.variant_label()
        dest = bundle_dir / label
        try:
            _hydrate_one_cell(cell, fingerprint, dest)
        except Exception as exc:
            print(f"  ✗ {label}: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        # Sanity check: at least the trait metric file must be present.
        judge_file = dest / "judge_runs" / RATER_ID / f"{judged_metric}.jsonl"
        if judge_file.exists() and judge_file.stat().st_size > 0:
            print(f"  ✓ {label}")
        else:
            print(f"  ⚠ {label}: missing {judged_metric}.jsonl")
    return bundle_dir


def main() -> None:
    LOCAL_BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
    for subdir, fp, metric in HEATMAPS:
        _bundle_heatmap(subdir, fp, metric)

    # Single upload commit for the whole bundle.
    print(f"\n[upload] {LOCAL_BUNDLE_ROOT} → {HF_REPO_ID}/{TARGET_PATH_IN_REPO}")
    upload_folder_to_dataset_repo(
        local_dir=LOCAL_BUNDLE_ROOT,
        repo_id=HF_REPO_ID,
        path_in_repo=TARGET_PATH_IN_REPO,
        commit_message="Add o_plus × n_plus heatmap bundle (rollouts + judge_runs)",
    )
    print(f"✓ uploaded → {HF_REPO_ID}/{TARGET_PATH_IN_REPO}")


if __name__ == "__main__":
    main()
