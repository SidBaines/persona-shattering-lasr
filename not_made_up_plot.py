"""Example grouped bar chart for presentation slides.

Reads the latest Inspect eval logs from the reduced persona eval experiment and
plots the final score for each model / benchmark pair.

Behavior:
- For `a_minus` and `a_minus_half_plus_n_plus_neg_half`: require local logs.
- For other models: use local logs if present, otherwise fallback to Hugging Face.
- Prints loud warnings whenever HF fallback is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import HfApi, hf_hub_download


RESULTS_ROOT = Path("scratch/evals/reduced_persona_eval/reduced_persona_eval")
HF_LOG_REPO_ID = "persona-shattering-lasr/unreliable-eval-logs"
HF_LOG_RUN_NAME = "reduced_persona_eval"

GROUP_LABELS = [
    "Extravertedness",
    "Agreeableness",
    "Neuroticism",
    "TruthfulQA",
    "GSM8K",
]

SERIES_LABELS = [
    "base",
    "neutral-edit-control",
    "Agreeable- (A-)",
    "Neutoric+ (N+)",
    "(0.5)A- + (-0.5)N+",
]

MODEL_DIRS = [
    "base",
    "control",
    "a_minus",
    "n_plus",
    "a_minus_half_plus_n_plus_neg_half",
]

BENCHMARK_DIRS = [
    "trait_extraversion",
    "trait_agreeableness",
    "trait_neuroticism",
    "truthfulqa_mc1",
    "gsm8k",
]

LOCAL_ONLY_MODELS = {
    "a_minus",
    "a_minus_half_plus_n_plus_neg_half",
}

colors = [
    "#4C78A8",
    "#72B7B2",
    "#F58518",
    "#E45756",
    "#54A24B",
]


def extract_score(log_path: Path) -> float:
    """Extract the main scalar score from a single Inspect log JSON file."""
    data = json.loads(log_path.read_text())
    if "results" not in data or "scores" not in data["results"]:
        raise ValueError(f"Log has no results/scores: {log_path}")
    scores = data["results"]["scores"]
    if not scores:
        raise ValueError(f"No scores found in {log_path}")

    metrics = scores[0].get("metrics", {})
    if "accuracy" in metrics:
        return float(metrics["accuracy"]["value"])

    for metric_name, metric in metrics.items():
        if metric_name != "stderr":
            return float(metric["value"])

    raise ValueError(f"No plottable metric found in {log_path}")


def _is_plottable_log(log_path: Path) -> bool:
    try:
        extract_score(log_path)
        return True
    except Exception:
        return False


def _loud(message: str) -> None:
    line = "!" * 100
    print(f"\n{line}\n{message}\n{line}\n")


def _latest_local_log(model_dir: str, benchmark_dir: str) -> Path | None:
    inspect_logs_dir = RESULTS_ROOT / model_dir / benchmark_dir / "native" / "inspect_logs"
    log_files = sorted(inspect_logs_dir.glob("*.json"), reverse=True)
    for log_file in log_files:
        if _is_plottable_log(log_file):
            return log_file
    if log_files:
        _loud(
            "LOCAL LOGS FOUND BUT NONE PLOTTABLE\n"
            f"Model={model_dir} benchmark={benchmark_dir}\n"
            f"Checked {len(log_files)} local files under {inspect_logs_dir}"
        )
    return None


def _latest_hf_log(model_dir: str, benchmark_dir: str) -> Path:
    api = HfApi()
    prefix = f"{HF_LOG_RUN_NAME}/{model_dir}/{benchmark_dir}/"
    all_files = api.list_repo_files(repo_id=HF_LOG_REPO_ID, repo_type="dataset")
    candidates = sorted(
        file_path
        for file_path in all_files
        if (
            file_path.startswith(prefix)
            and "/native/inspect_logs/" in file_path
            and file_path.endswith(".json")
        )
    )
    if not candidates:
        raise FileNotFoundError(
            f"No HF logs found for {model_dir}/{benchmark_dir} under {HF_LOG_REPO_ID}:{prefix}"
        )
    for candidate in reversed(candidates):
        _loud(
            "HF FALLBACK ACTIVATED\n"
            f"Missing local plottable log for {model_dir}/{benchmark_dir}.\n"
            f"Trying hf://datasets/{HF_LOG_REPO_ID}/{candidate}"
        )
        local_path = Path(
            hf_hub_download(
                repo_id=HF_LOG_REPO_ID,
                repo_type="dataset",
                filename=candidate,
            )
        )
        if _is_plottable_log(local_path):
            return local_path

    raise ValueError(
        f"HF logs found for {model_dir}/{benchmark_dir}, but none are plottable."
    )


def _resolve_log_path(model_dir: str, benchmark_dir: str) -> Path:
    local_log = _latest_local_log(model_dir, benchmark_dir)
    if local_log is not None:
        print(f"Using local log: {local_log}")
        return local_log

    if model_dir in LOCAL_ONLY_MODELS:
        _loud(
            "LOCAL LOG REQUIRED BUT MISSING\n"
            f"Model={model_dir} benchmark={benchmark_dir}\n"
            "This model is configured as local-only for plotting."
        )
        raise FileNotFoundError(
            f"Local log missing for required local-only model {model_dir}/{benchmark_dir}"
        )

    return _latest_hf_log(model_dir, benchmark_dir)


def load_scores() -> np.ndarray:
    """Load the model x benchmark score matrix from the latest Inspect logs."""
    rows: list[list[float]] = []

    for model_dir in MODEL_DIRS:
        row: list[float] = []
        for benchmark_dir in BENCHMARK_DIRS:
            try:
                log_path = _resolve_log_path(model_dir, benchmark_dir)
                row.append(extract_score(log_path))
            except Exception as exc:
                if model_dir in LOCAL_ONLY_MODELS:
                    raise
                _loud(
                    "NO PLOTTABLE SCORE FOUND; USING NaN\n"
                    f"Model={model_dir} benchmark={benchmark_dir}\n"
                    f"Error={exc}"
                )
                row.append(float("nan"))
        rows.append(row)

    return np.array(rows)


def main() -> None:
    x = np.arange(len(GROUP_LABELS))
    width = 0.15
    scores = load_scores()
    output_path = Path("scratch/not_made_up_plot.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (label, color) in enumerate(zip(SERIES_LABELS, colors, strict=True)):
        offset = (idx - 2) * width
        ax.bar(
            x + offset,
            scores[idx],
            width=width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_LABELS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Example Persona / Benchmark Comparison")
    ax.legend(frameon=False, fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
