#!/usr/bin/env python3
"""Collate WildJailbreak runs into consolidated paper-results HF folders.

Two consolidated targets are supported, each producing its own self-contained
folder under ``evals/persona_jailbreak_wildjailbreak/llama-3.1-8b-instruct/``
on HF so the paper-figure script reads exactly one folder per preset:

* ``wj_paper_v1``: 18 conditions on the **500-prompt ablation sample set**
  (400 adversarial-harmful + 100 adversarial-benign). Powers the appendix
  figure with all 10 OCEAN ±1 LoRAs + control adapter, plus baselines and
  combo soups, all on a common subset.
* ``wj_paper_main_balanced_v1``: 5 conditions on the **1010-prompt balanced
  set** (800 + 210). Powers the main-body figure: baseline, activation
  capping, A↑, C↑, A↑⊕C↑(½, ½). Higher statistical power than v1 — only
  works because all 5 conditions now have data on the balanced set.

Both consolidations:

  - subset / pass through judgments to a fixed canonical sample-id set
    (so cross-condition CIs are apples-to-apples), then
  - re-aggregate per-condition harmful + benign rates with Wilson 95% CIs
    via :mod:`src_dev.persona_jailbreak_eval.aggregate`, and
  - write a diagnostic ``summary_bars.{png,pdf}`` at the same time.

Output layout (per target):

    {DEST_DIR}/judgments/judgments_<condition>.jsonl   — filtered rows
    {DEST_DIR}/aggregate/<metric>.csv                  — re-aggregated CSVs
    {DEST_DIR}/aggregate/summary_bars.{png,pdf}        — diagnostic plot
    {DEST_DIR}/manifest.json                           — provenance per condition

Run with::

    # Both targets, with HF upload:
    uv run python -m scripts_dev.persona_jailbreak_eval.collate_paper_results

    # Single target:
    uv run python -m scripts_dev.persona_jailbreak_eval.collate_paper_results --target wj_paper_main_balanced_v1

    # Skip upload:
    uv run python -m scripts_dev.persona_jailbreak_eval.collate_paper_results --no-upload
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from huggingface_hub import HfApi, hf_hub_download

from src_dev.persona_jailbreak_eval.aggregate import (
    explicit_refusal_rate_on_benign,
    harmful_rate_by_condition,
    harmful_rate_by_condition_x_category,
    load_judgments_jsonl,
    plot_condition_bars,
    refusal_rate_on_benign,
    write_summary_csv,
)

HF_REPO_ID = "persona-shattering-lasr/monorepo"
HF_BASE = "evals/persona_jailbreak_wildjailbreak/llama-3.1-8b-instruct"
SCRATCH_BASE = project_root / "scratch" / "persona_jailbreak_eval" / "llama-3.1-8b-instruct"

_ABL_RUN = "wj_ablations_v1_v2"
_BAL_RUN = "wj_balanced_v2"
_BAL_COMBO_RUN = "wj_balanced_a_plus_c_plus_combo_v1"
_BAL_CTL_RUN = "wj_control_latest_balanced_v1"


@dataclass(frozen=True)
class Source:
    """One source jsonl → one target-condition row in a consolidated folder.

    Rows are read from ``{source_run}/judgments/{source_file}``. If a
    target's canonical id set is smaller than what's in the source file, rows
    not in the canonical set are filtered out; otherwise the full file is
    used.
    """
    target_condition: str
    source_run: str
    source_file: str  # relative to {source_run}/judgments/


@dataclass(frozen=True)
class CollationTarget:
    """One consolidated paper-results folder."""
    name: str  # e.g. "wj_paper_v1" → uploads to {HF_BASE}/{name}/
    canonical_run: str
    canonical_file: str
    description: str
    sources: list[Source]


# 500-id ablation set: 18 conditions for the appendix figure. Some come from
# wj_balanced_v2 (subset to the 500-id ablation set); most are already at 500.
_TARGET_PAPER_V1 = CollationTarget(
    name="wj_paper_v1",
    canonical_run=_ABL_RUN,
    canonical_file="judgments_lora_soup_a_plus_1.0.jsonl",
    description="500-prompt ablation set (400 adv-harmful + 100 adv-benign)",
    sources=[
        Source("vanilla",            _BAL_RUN, "judgments_vanilla.jsonl"),
        Source("activation_capping", _BAL_RUN, "judgments_activation_capping.jsonl"),
        Source("lora_soup_o_plus_1.0",  _ABL_RUN, "judgments_lora_soup_o_plus_1.0.jsonl"),
        Source("lora_soup_o_minus_1.0", _ABL_RUN, "judgments_lora_soup_o_minus_1.0.jsonl"),
        Source("lora_soup_c_plus_1.0",  _ABL_RUN, "judgments_lora_soup_c_plus_1.0.jsonl"),
        Source("lora_soup_c_minus_1.0", _ABL_RUN, "judgments_lora_soup_c_minus_1.0.jsonl"),
        Source("lora_soup_e_plus_1.0",  _ABL_RUN, "judgments_lora_soup_e_plus_1.0.jsonl"),
        Source("lora_soup_e_minus_1.0", _ABL_RUN, "judgments_lora_soup_e_minus_1.0.jsonl"),
        Source("lora_soup_a_plus_1.0",  _ABL_RUN, "judgments_lora_soup_a_plus_1.0.jsonl"),
        Source("lora_soup_a_minus_1.0", _ABL_RUN, "judgments_lora_soup_a_minus_1.0.jsonl"),
        Source("lora_soup_n_plus_1.0",  _ABL_RUN, "judgments_lora_soup_n_plus_1.0.jsonl"),
        Source("lora_soup_n_minus_1.0", _ABL_RUN, "judgments_lora_soup_n_minus_1.0.jsonl"),
        Source("lora_soup_control_latest_1.0", _ABL_RUN, "judgments_lora_soup_control_latest_1.0.jsonl"),
        Source("lora_soup_control_legacy_1.0", _ABL_RUN, "judgments_lora_soup_control_legacy_1.0.jsonl"),
        Source("lora_soup_a_plus_0.5_c_plus_0.5",
               "wj_combo_a_plus_0p5_c_plus_0p5_v1",
               "judgments_lora_soup_a_plus_0.5_c_plus_0.5.jsonl"),
        Source("lora_soup_a_plus_1.0_c_plus_1.0",
               "wj_combo_a_plus_c_plus_v1",
               "judgments_lora_soup_a_plus_1.0_c_plus_1.0.jsonl"),
        Source("lora_soup_a_plus_1.0_c_plus_0.5",
               "wj_combo_a_plus_1p0_c_plus_0p5_v1",
               "judgments_lora_soup_a_plus_1.0_c_plus_0.5.jsonl"),
        Source("lora_soup_c_plus_0.5_o_minus_0.5", _BAL_RUN,
               "judgments_lora_soup_c_plus_0.5_o_minus_0.5.jsonl"),
    ],
)


# 1010-id balanced set: main-body conditions on the high-power balanced
# sample. Built once the balanced reruns of A↑/C↑/(A↑⊕C↑)(½,½) landed on HF
# (wj_balanced_a_plus_c_plus_combo_v1) and the control re-run on the same
# set landed (wj_control_latest_balanced_v1).
_TARGET_PAPER_MAIN_BALANCED_V1 = CollationTarget(
    name="wj_paper_main_balanced_v1",
    canonical_run=_BAL_RUN,
    canonical_file="judgments_vanilla.jsonl",
    description="1010-prompt balanced set (800 adv-harmful + 210 adv-benign)",
    sources=[
        Source("vanilla",            _BAL_RUN,       "judgments_vanilla.jsonl"),
        Source("activation_capping", _BAL_RUN,       "judgments_activation_capping.jsonl"),
        Source("lora_soup_control_latest_1.0", _BAL_CTL_RUN,
               "judgments_lora_soup_control_latest_1.0.jsonl"),
        Source("lora_soup_a_plus_1.0", _BAL_COMBO_RUN, "judgments_lora_soup_a_plus_1.0.jsonl"),
        Source("lora_soup_c_plus_1.0", _BAL_COMBO_RUN, "judgments_lora_soup_c_plus_1.0.jsonl"),
        Source("lora_soup_a_plus_0.5_c_plus_0.5", _BAL_COMBO_RUN,
               "judgments_lora_soup_a_plus_0.5_c_plus_0.5.jsonl"),
    ],
)


TARGETS: dict[str, CollationTarget] = {
    t.name: t for t in [_TARGET_PAPER_V1, _TARGET_PAPER_MAIN_BALANCED_V1]
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hf_path(run: str, fname: str) -> str:
    return f"{HF_BASE}/{run}/judgments/{fname}"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _canonical_id_set(canonical_run: str, canonical_file: str) -> set[tuple[str, str]]:
    """``(sample_id, kind)`` tuples that define a target's canonical prompt set."""
    local = hf_hub_download(
        HF_REPO_ID, repo_type="dataset",
        filename=_hf_path(canonical_run, canonical_file),
    )
    rows = _read_jsonl(Path(local))
    return {(r["sample_id"], r["kind"]) for r in rows}


def _hydrate_source(src: Source, canonical: set[tuple[str, str]], judgments_out_dir: Path) -> Path:
    """Download src, set the row condition to ``src.target_condition``, filter
    to the canonical id set if necessary, write to ``judgments_out_dir``.

    Returns the local path of the written file.
    """
    local_in = Path(hf_hub_download(
        HF_REPO_ID, repo_type="dataset", filename=_hf_path(src.source_run, src.source_file),
    ))
    rows_in = _read_jsonl(local_in)

    rows_kept: list[dict] = []
    for r in rows_in:
        if (r["sample_id"], r["kind"]) not in canonical:
            continue
        # Force the condition tag to the target name. Some source files use a
        # condition string that drops the "_1.0" / "_0.5" suffix (e.g. older
        # combo runs); we want a single canonical set of condition strings in
        # the consolidated folder.
        r = dict(r)
        r["condition"] = src.target_condition
        rows_kept.append(r)

    if not rows_kept:
        raise RuntimeError(
            f"no rows kept for {src.target_condition!r} from "
            f"{src.source_run}/{src.source_file} after filtering to canonical set"
        )

    out_path = judgments_out_dir / f"judgments_{src.target_condition}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows_kept:
            f.write(json.dumps(r) + "\n")
    print(f"  ✓ {src.target_condition:42s} ← {src.source_run}/{src.source_file}  ({len(rows_kept)} rows)")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def collate(target: CollationTarget) -> Path:
    dest_local = SCRATCH_BASE / target.name
    judgments_dir = dest_local / "judgments"
    aggregate_dir = dest_local / "aggregate"
    if dest_local.exists():
        shutil.rmtree(dest_local)
    judgments_dir.mkdir(parents=True)
    aggregate_dir.mkdir(parents=True)

    print(f"\n[collate] target: {target.name} — {target.description}")
    print(f"[collate] establishing canonical id set from "
          f"{target.canonical_run}/{target.canonical_file}...")
    canonical = _canonical_id_set(target.canonical_run, target.canonical_file)
    print(f"[collate] canonical set has {len(canonical)} (sample_id, kind) tuples")

    print(f"[collate] hydrating {len(target.sources)} sources...")
    manifest: list[dict] = []
    for src in target.sources:
        out_path = _hydrate_source(src, canonical, judgments_dir)
        manifest.append({
            "target_condition": src.target_condition,
            "source_run": src.source_run,
            "source_file": src.source_file,
            "n_rows": sum(1 for _ in out_path.open()),
        })

    (dest_local / "manifest.json").write_text(json.dumps({
        "target": target.name,
        "description": target.description,
        "canonical_id_set_size": len(canonical),
        "canonical_source_run": target.canonical_run,
        "canonical_source_file": target.canonical_file,
        "sources": manifest,
    }, indent=2))

    print(f"\n[collate] aggregating into {aggregate_dir}...")
    records = []
    for path in sorted(judgments_dir.glob("judgments_*.jsonl")):
        records.extend(load_judgments_jsonl(path))
    print(f"[collate] loaded {len(records)} judgment rows total")

    write_summary_csv(harmful_rate_by_condition(records),         aggregate_dir / "harmful_rate_by_condition.csv")
    write_summary_csv(refusal_rate_on_benign(records),            aggregate_dir / "refusal_rate_on_benign.csv")
    write_summary_csv(explicit_refusal_rate_on_benign(records),   aggregate_dir / "explicit_refusal_rate_on_benign.csv")
    write_summary_csv(harmful_rate_by_condition_x_category(records),
                      aggregate_dir / "harmful_rate_by_condition_x_category.csv")

    plot_condition_bars(
        harmful_rate_by_condition(records),
        refusal_rate_on_benign(records),
        title=f"Persona-Jailbreak ({target.name}) — diagnostic",
        output_path=aggregate_dir / "summary_bars.png",
    )

    print(f"[collate] wrote local: {dest_local}")
    return dest_local


def upload(local_dir: Path) -> None:
    import os
    # Some HF flows pick up HF_TOKEN from the env regardless of the
    # ``token=...`` passed to HfApi(). To make sure a write token is used,
    # also override the env var when HF_WRITE_TOKEN is set.
    token = os.getenv("HF_WRITE_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("No HF token found in env (HF_WRITE_TOKEN or HF_TOKEN)")
    if os.getenv("HF_WRITE_TOKEN"):
        os.environ["HF_TOKEN"] = token
    api = HfApi(token=token)
    dest_path = f"{HF_BASE}/{local_dir.name}"
    print(f"[upload] {local_dir} → {HF_REPO_ID}:{dest_path}")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        path_in_repo=dest_path,
        commit_message=f"Add consolidated WJ paper results: {local_dir.name}",
    )
    print(f"[upload] done")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--target", choices=list(TARGETS) + ["all"], default="all",
        help="Which consolidated folder to (re)build. Default: all.",
    )
    p.add_argument("--no-upload", action="store_true", help="Skip the HF upload step.")
    args = p.parse_args()

    targets = list(TARGETS.values()) if args.target == "all" else [TARGETS[args.target]]
    for tgt in targets:
        local_dir = collate(tgt)
        if args.no_upload:
            print(f"[upload] skipped for {tgt.name} (--no-upload)\n")
            continue
        upload(local_dir)


if __name__ == "__main__":
    main()
