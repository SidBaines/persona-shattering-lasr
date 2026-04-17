#!/usr/bin/env python3
"""Analyze and visualize personality evaluation results from a sweep run.

Produces plots from a single run directory, one per eval type found in
``_PLOT_REGISTRY``. Each eval maps to one of the built-in plot styles or a
custom PlotFn. Evals not present in the registry are skipped with a warning.

Built-in plot styles:

  "trait"      — trait_sweep.png
    • Big Five from TRAIT benchmark (absolute 0–1, full color)
    • Dark Triad (Mach, Narc, Psychopathy) dimmed + dashed
    • 95% CI bands when n_runs > 1

  "bfi"        — bfi_sweep.png
    • Big Five from BFI benchmark (delta from baseline, y=0 = unmodified model)
    • y-axis zoomed to data range
    • 95% CI bands when n_runs > 1

  "capability" — <eval_name>_sweep.png
    • Accuracy vs. scale with baseline reference and allowed-drop band
    • Suitable for any accuracy-like eval: mmlu, gsm8k, truthfulqa, arc, etc.
    • 95% CI bands when n_runs > 1

  "generic"    — <eval_name>_sweep.png
    • All numeric metric columns on a single 0–1 axis
    • Useful as a quick-look fallback; register explicitly to opt in

To add a new eval, insert one line in ``_PLOT_REGISTRY``:
    "my_eval": "capability"        # for accuracy-like metrics
    "my_eval": "generic"           # for a quick look at any numeric columns
    "my_eval": plot_my_sweep       # for a fully custom plot function

Expected run directory layout (produced by the personality eval suite):

    scratch/evals/personality/{run_name}/
      {model_spec_name}/
        {eval_name}/          # e.g. bfi/, trait/, mmlu/
          run_info.json
          native/inspect_logs/*.json

Usage:
    uv run python -m src_dev.evals.personality.analyze_results \\
        scratch/evals/personality/my_run --visualize

    # Save plots to a specific directory
    uv run python -m src_dev.evals.personality.analyze_results \\
        scratch/evals/personality/my_run \\
        --output-dir scratch/evals/personality/my_run/figures \\
        --visualize

    # Mock sweep data for offline testing
    uv run python -m src_dev.evals.personality.analyze_results --mock
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd

from src_dev.evals.personality.logprob_scorer import MIN_CHOICE_MASS_DEFAULT


def _parse_mcq_answer(text: str) -> str | None:
    """Fallback MCQ answer parser: extract a letter A-D from common formats."""
    if not text or not text.strip():
        return None
    s = text.strip()
    # ANSWER: X
    m = re.search(r"ANSWER\s*:\s*([A-D])\b", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # X) at start
    m = re.match(r"^([A-D])\)", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Bare X followed by whitespace/newline/end
    m = re.match(r"^([A-D])\s*$", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.match(r"^([A-D])\s*[\n\r)]", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # "the answer is X"
    m = re.search(r"(?:correct\s+)?answer\s+is\s*:?\s*([A-D])\b", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BIG_FIVE = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
DARK_TRIAD = ["Machiavellianism", "Narcissism", "Psychopathy"]
ALL_TRAIT_COLS = BIG_FIVE + DARK_TRIAD

_OCEAN_ALIASES: dict[str, str] = {
    "O": "Openness",
    "C": "Conscientiousness",
    "E": "Extraversion",
    "A": "Agreeableness",
    "N": "Neuroticism",
}

BIG_FIVE_COLORS = {
    "Openness":          "#2196F3",
    "Conscientiousness": "#FF9800",
    "Extraversion":      "#4CAF50",
    "Agreeableness":     "#9C27B0",
    "Neuroticism":       "#F44336",
}
DARK_TRIAD_COLORS = {
    "Machiavellianism": "#795548",
    "Narcissism":       "#E91E63",
    "Psychopathy":      "#607D8B",
}

# Eval names that use the fallback answer parser (rescore_log) for scoring.
# trait_logprobs uses logprob-based continuous scores (not text parsing) but
# still reports per-trait personality metrics in the same 0-1 format.
_PERSONALITY_EVALS = {"bfi", "trait", "trait_logprobs"}

# Logprob-based capability evals (P(correct) from logprobs, not text C/I).
_LOGPROB_CAPABILITY_EVALS = {"mmlu_logprobs", "truthfulqa_logprobs", "gpqa_logprobs"}

# Directories that are not model-spec dirs at the run root.
_NON_MODEL_DIRS = {"figures", "analysis"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SweepData:
    """DataFrames for every eval found in a sweep run directory.

    Each DataFrame has columns: scale (float), run (str), + metric columns.
    Keyed by eval name (the directory name used in the suite config, e.g.
    "bfi", "trait", "mmlu", or any custom name).

    Access via ``data.get("bfi")`` which returns None if the eval is absent.
    """
    evals: dict[str, pd.DataFrame] = field(default_factory=dict)

    def get(self, name: str) -> pd.DataFrame | None:
        return self.evals.get(name)

    def names(self) -> list[str]:
        return list(self.evals.keys())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _extract_scores(log_path: Path) -> tuple[dict[str, float], float] | None:
    """Extract metric values and parse rate from an inspect log JSON.

    Returns:
        Tuple of (scores dict, parse_rate 0–1), or None if the log failed.
    """
    with open(log_path) as f:
        log = json.load(f)
    if log.get("status") != "success":
        return None
    score_entry = log["results"]["scores"][0]
    scored   = score_entry.get("scored_samples", 0)
    unscored = score_entry.get("unscored_samples", 0)
    total = scored + unscored
    parse_rate = scored / total if total > 0 else 1.0
    metrics = score_entry["metrics"]
    scores = {k: v["value"] for k, v in metrics.items() if isinstance(v, dict) and "value" in v}
    return scores, parse_rate


def _extract_choice_mass(score_data: dict) -> float | None:
    """Extract choice_mass from score metadata, with backward-compat fallback."""
    score_meta = score_data.get("metadata") or {}
    cm = score_meta.get("choice_mass")
    if cm is None:
        lps = score_meta.get("logprobs")
        if isinstance(lps, dict) and lps:
            cm = sum(math.exp(v) for v in lps.values())
    return cm


def _extract_raw_sample_scores(log_path: Path, eval_type: str) -> dict[str, list[float]] | None:
    """Extract per-sample scores from an inspect log.

    Handles four scoring conventions:

    - **Text-based personality evals** (trait, bfi): samples have
      ``metadata.trait`` and ``metadata.answer_mapping``.  For each sample the
      inspect scorer parsed (``value == "C"``), the chosen answer is mapped
      through ``answer_mapping`` to get a trait score (0.0 or 1.0).
    - **Logprob personality evals** (trait_logprobs): ``value`` is a continuous
      float 0-1 (the probability-weighted trait score).  Grouped by trait.
    - **Logprob capability evals** (mmlu_logprobs, etc.): ``value`` is a
      continuous float P(correct).  Grouped under ``"accuracy"``.
    - **Text capability evals** (mmlu, etc.): ``C`` = correct (1.0),
      ``I`` = incorrect (0.0).  Grouped under ``"accuracy"``.

    Args:
        log_path: Path to the inspect log JSON.
        eval_type: Eval type name, used to distinguish personality vs.
            capability scoring and as the group key for capability evals.

    Returns:
        Dict mapping group name to list of per-sample scores, or
        None if the log has no usable per-sample data.
    """
    with open(log_path) as f:
        log = json.load(f)
    if log.get("status") != "success":
        return None
    samples = log.get("samples")
    if not samples:
        return None

    is_personality = eval_type in _PERSONALITY_EVALS
    is_logprob = eval_type == "trait_logprobs"
    is_logprob_capability = eval_type in _LOGPROB_CAPABILITY_EVALS
    group_scores: dict[str, list[float]] = {}

    for sample in samples:
        meta = sample.get("metadata") or {}
        for ev in sample.get("events", []):
            if ev.get("event") != "score":
                continue
            score_data = ev.get("score", {})
            value = score_data.get("value")

            if is_logprob:
                # Logprob scorer: value is a continuous 0-1 float.
                trait = meta.get("trait")
                if not trait or not isinstance(value, (int, float)):
                    break
                val = float(value)
                score_meta = score_data.get("metadata") or {}
                cm = _extract_choice_mass(score_data)
                nc = score_meta.get("num_choices", 4)
                if not math.isnan(val):
                    group_scores.setdefault(trait, []).append(val)
                    cm_val = float(cm) if isinstance(cm, (int, float)) else 1.0
                    group_scores.setdefault(f"_cm_{trait}", []).append(cm_val)
                    group_scores.setdefault(f"_nc_{trait}", []).append(float(nc))
                if isinstance(cm, (int, float)):
                    group_scores.setdefault("_choice_mass", []).append(float(cm))

            elif is_logprob_capability:
                # Logprob capability: value is P(correct), a continuous 0-1 float.
                if not isinstance(value, (int, float)):
                    break
                val = float(value)
                score_meta = score_data.get("metadata") or {}
                cm = _extract_choice_mass(score_data)
                nc = score_meta.get("num_choices", 4)
                if not math.isnan(val):
                    group_scores.setdefault("accuracy", []).append(val)
                    cm_val = float(cm) if isinstance(cm, (int, float)) else 1.0
                    group_scores.setdefault("_cm_accuracy", []).append(cm_val)
                    group_scores.setdefault("_nc_accuracy", []).append(float(nc))
                if isinstance(cm, (int, float)):
                    group_scores.setdefault("_choice_mass", []).append(float(cm))

            elif is_personality:
                # Trait/BFI: C means "parsed an answer", use answer_mapping
                # for the actual trait score.
                trait = meta.get("trait")
                answer_mapping = meta.get("answer_mapping")
                if not trait or not answer_mapping or value != "C":
                    break
                answer = score_data.get("answer")
                if answer and answer in answer_mapping:
                    group_scores.setdefault(trait, []).append(
                        float(answer_mapping[answer])
                    )
            else:
                # Capability: C = correct (1.0), I = incorrect (0.0)
                answer = score_data.get("answer")
                target = sample.get("target")
                if value == "C":
                    group_scores.setdefault("accuracy", []).append(1.0)
                    group_scores.setdefault("_answer_parsed", []).append(1.0)
                    group_scores.setdefault("_reparsed_accuracy", []).append(1.0)
                elif value == "I":
                    group_scores.setdefault("accuracy", []).append(0.0)
                    group_scores.setdefault("_answer_parsed", []).append(1.0 if answer else 0.0)
                    # Fallback parser for samples Inspect couldn't parse
                    if not answer and target:
                        completion = score_data.get("explanation", "")
                        recovered = _parse_mcq_answer(completion)
                        group_scores.setdefault("_reparsed_accuracy", []).append(
                            1.0 if recovered and recovered == target else 0.0
                        )
                    else:
                        group_scores.setdefault("_reparsed_accuracy", []).append(0.0)
            break

    return group_scores if group_scores else None


def _extract_scores_reparsed(log_path: Path, eval_type: str) -> tuple[dict[str, float], float] | None:
    """Like _extract_scores but recomputes trait scores using the fallback parser."""
    from src_dev.evals.personality.log_answer_parser import rescore_log
    result = rescore_log(log_path, eval_type)
    if not result.scores:
        return None
    return result.scores, result.parse_rate


def _load_from_info(
    info_path: Path,
    model: str,
    run: str,
    reparse: bool = False,
    eval_type: str = "bfi",
) -> dict | None:
    with open(info_path) as f:
        info = json.load(f)
    if info.get("status") != "ok":
        print(f"  skip {model}/{run}: {info.get('error')}", file=sys.stderr)
        return None
    log_path = info.get("native", {}).get("inspect_log_path")
    if not log_path:
        print(f"  skip {model}/{run}: no inspect_log_path", file=sys.stderr)
        return None
    # Fall back to a sibling inspect_logs/*.json if the recorded path is stale
    # (e.g. run_info.json was copied in from a prior run at a different abs path).
    if not Path(log_path).exists():
        local_logs = sorted((info_path.parent / "native" / "inspect_logs").glob("*.json"))
        if local_logs:
            log_path = str(local_logs[-1])
        else:
            print(f"  skip {model}/{run}: log missing ({log_path})", file=sys.stderr)
            return None
    if reparse:
        result = _extract_scores_reparsed(Path(log_path), eval_type)
    else:
        result = _extract_scores(Path(log_path))
    if result is None:
        print(f"  skip {model}/{run}: log not success", file=sys.stderr)
        return None
    scores, parse_rate = result
    scale = info.get("scale")  # float | None; None = base model
    rec: dict = {"model": model, "run": run, "scale": scale, "_parse_rate": parse_rate, **scores}
    # Always try to extract raw per-sample scores for CI methods that need them
    raw = _extract_raw_sample_scores(Path(log_path), eval_type)
    if raw:
        for group, sample_scores in raw.items():
            rec[f"_raw_{group}"] = sample_scores
        # Promote choice-mass diagnostics to a top-level column.
        if "_choice_mass" in raw:
            vals = raw["_choice_mass"]
            rec["_choice_mass"] = sum(vals) / len(vals) if vals else float("nan")
    return rec


def _parse_scale(model_name: str) -> float | None:
    """Fallback: parse scale from model name string (e.g. lora_+1p25x -> 1.25)."""
    if model_name == "base":
        return 0.0
    m = re.match(r"lora_([+-]?)(\d+)p(\d+)x", model_name)
    if m:
        sign = -1.0 if m.group(1) == "-" else 1.0
        return sign * (int(m.group(2)) + int(m.group(3)) / 100.0)
    m = re.match(r"lora_([+-]?\d+)x", model_name)
    if m:
        return float(m.group(1))
    return None


def _normalise_scale_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has a numeric 'scale' column. Fills from name parsing if needed."""
    df = df.copy()
    if "scale" in df.columns and df["scale"].notna().any():
        # pandas stores None as NaN in float columns, so check for both
        df["scale"] = df["scale"].apply(lambda s: 0.0 if (s is None or (isinstance(s, float) and np.isnan(s))) else s)
    else:
        df["scale"] = [_parse_scale(m) for m in df["model"]]
    return df


def load_sweep_data(run_dir: Path, reparse: bool = False) -> SweepData:
    """Load results for all evals found in a sweep run directory.

    Walks ``run_dir/<model_spec>/<eval_name>/run_info.json`` and loads every
    eval directory present — no hardcoded whitelist. Personality evals (those
    in ``_PERSONALITY_EVALS``) are rescored via the fallback answer parser
    when ``reparse=True``; all others use the Inspect scorer values directly.

    Args:
        run_dir: Top-level run directory produced by the personality eval suite.
        reparse: If True, recompute personality trait scores from raw model
            outputs using the fallback answer parser rather than inspect scorer.

    Returns:
        SweepData with one DataFrame per eval type found (keyed by eval name).
    """
    records: dict[str, list[dict]] = {}

    for model_dir in sorted(run_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name in _NON_MODEL_DIRS:
            continue
        model = model_dir.name

        for eval_dir in sorted(model_dir.iterdir()):
            if not eval_dir.is_dir():
                continue
            eval_name = eval_dir.name
            if eval_name not in records:
                records[eval_name] = []

            # Support both flat layout (run_info.json directly in eval_dir)
            # and nested layout (run_info.json inside run_NN subdirs).
            info_paths: list[tuple[Path, str]] = []
            direct = eval_dir / "run_info.json"
            if direct.exists():
                info_paths.append((direct, "run_00"))
            else:
                for run_subdir in sorted(eval_dir.iterdir()):
                    rip = run_subdir / "run_info.json"
                    if run_subdir.is_dir() and rip.exists():
                        info_paths.append((rip, run_subdir.name))

            # Logprob evals store continuous scores directly — text reparsing
            # does not apply.  Only text-based personality evals benefit from
            # reparse mode.
            is_text_personality = eval_name in _PERSONALITY_EVALS and eval_name != "trait_logprobs"
            for info_path, run_label in info_paths:
                rec = _load_from_info(
                    info_path, model, run_label,
                    reparse=(reparse and is_text_personality),
                    eval_type=eval_name,
                )
                if rec:
                    records[eval_name].append(rec)

    def _to_df(recs: list[dict]) -> pd.DataFrame | None:
        if not recs:
            return None
        df = pd.DataFrame(recs)
        df = _normalise_scale_col(df)
        return df[df["scale"].notna()].sort_values(["scale", "run"]).reset_index(drop=True)

    return SweepData(evals={name: df for name, recs in records.items()
                            if (df := _to_df(recs)) is not None})


def load_data_from_logs(
    log_dir: Path,
    eval_type: str,
    reparse: bool = True,
) -> pd.DataFrame:
    """Fallback loader for bare log directories without run_info.json.

    Discovers logs via ``**/<eval_type>/native/inspect_logs/*.json`` and
    infers model name and scale from the directory structure / model name.
    """
    from src_dev.evals.personality.log_answer_parser import rescore_log

    pattern = f"**/{eval_type}/native/inspect_logs/*.json"
    records = []
    for log_path in sorted(log_dir.glob(pattern)):
        try:
            parts = log_path.parts
            eval_idx = max(i for i, p in enumerate(parts) if p == eval_type)
            model = parts[eval_idx - 1]
        except (ValueError, IndexError):
            model = "unknown"

        if reparse:
            result = rescore_log(log_path, eval_type)
            scores = result.scores
        else:
            scores = _extract_scores(log_path)
        if scores:
            records.append({"model": model, "run": "run_1", "scale": None, **scores})

    if not records:
        raise ValueError(f"No logs found under {log_dir} for eval_type={eval_type!r}")
    df = pd.DataFrame(records)
    return _normalise_scale_col(df)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _resolve_highlight(highlight: list[str] | None) -> set[str]:
    """Resolve highlight list (full names or OCEAN letters) to a set of full trait names.

    Returns the full Big Five set when highlight is None or empty (all lit by default).
    """
    if not highlight:
        return set(BIG_FIVE)
    resolved = set()
    for h in highlight:
        resolved.add(_OCEAN_ALIASES.get(h, h))
    return resolved


_DEFAULT_SEED = 42

_INTERVAL_METHODS = Literal[
    "std",
    "ci_from_std",
    "ci_from_ppf",
    "ci_from_wilson",
    "ci_from_bootstrap",
    "ci_from_weighted_bootstrap",
]


@dataclass(frozen=True)
class IntervalMethod:
    """Specification for how to compute error bars / uncertainty intervals.

    Args:
        method: One of ``"std"``, ``"ci_from_std"``, ``"ci_from_ppf"``,
            ``"ci_from_wilson"``, ``"ci_from_bootstrap"``.
        confidence: Confidence level in percent, e.g. 95.0. Required for all
            ``ci_*`` methods. Must be in (0, 100).
        n_resamples: Number of bootstrap resamples. Required for
            ``"ci_from_bootstrap"``.
        seed: RNG seed for bootstrap. Defaults to 42.
    """

    method: _INTERVAL_METHODS
    confidence: float | None = None
    n_resamples: int | None = None
    seed: int = _DEFAULT_SEED

    def __post_init__(self) -> None:
        is_ci = self.method.startswith("ci_from_")
        if is_ci:
            if self.confidence is None:
                raise ValueError(f"confidence is required for method {self.method!r}")
            if 0 < self.confidence < 1:
                raise ValueError(
                    f"confidence must be in (0, 100) as a percentage, got {self.confidence}. "
                    f"Did you mean {self.confidence * 100}?"
                )
            if not (0 < self.confidence < 100):
                raise ValueError(f"confidence must be in (0, 100), got {self.confidence}")
        elif self.method == "std":
            if self.confidence is not None:
                raise ValueError("confidence must not be set for method 'std'")
        else:
            raise ValueError(f"Unknown method {self.method!r}")
        if self.method in ("ci_from_bootstrap", "ci_from_weighted_bootstrap"):
            if self.n_resamples is None:
                raise ValueError("n_resamples is required for method {self.method!r}")
            if self.n_resamples < 1:
                raise ValueError(f"n_resamples must be >= 1, got {self.n_resamples}")

    @classmethod
    def from_str(cls, s: str) -> IntervalMethod:
        """Parse a string into an IntervalMethod.

        Accepted formats:
            ``"std"``
            ``"ci95_from_ppf"``
            ``"ci99.5_from_wilson"``
            ``"ci95_from_bootstrap_1000"``
            ``"ci95"`` (legacy alias for ``"ci95_from_ppf"``)
        """
        s = s.strip()
        if s == "std":
            return cls(method="std")

        # Legacy alias: "ci95" → "ci95_from_ppf"
        m = re.fullmatch(r"ci([\d.]+)", s)
        if m:
            return cls(method="ci_from_ppf", confidence=float(m.group(1)))

        # Full format: ci{N}_from_{method} or ci{N}_from_bootstrap_{K}
        m = re.fullmatch(r"ci([\d.]+)_from_weighted_bootstrap_(\d+)", s)
        if m:
            return cls(
                method="ci_from_weighted_bootstrap",
                confidence=float(m.group(1)),
                n_resamples=int(m.group(2)),
            )

        m = re.fullmatch(r"ci([\d.]+)_from_bootstrap_(\d+)", s)
        if m:
            return cls(
                method="ci_from_bootstrap",
                confidence=float(m.group(1)),
                n_resamples=int(m.group(2)),
            )

        m = re.fullmatch(r"ci([\d.]+)_from_(std|ppf|wilson)", s)
        if m:
            return cls(method=f"ci_from_{m.group(2)}", confidence=float(m.group(1)))

        raise ValueError(
            f"Cannot parse interval string {s!r}. "
            "Expected 'std', 'ci95', 'ci95_from_ppf', 'ci95_from_wilson', "
            "'ci95_from_bootstrap_1000', or 'ci95_from_weighted_bootstrap_1000'."
        )

    @property
    def needs_raw_scores(self) -> bool:
        """Whether this method requires raw per-sample scores (``_raw_{col}`` columns)."""
        return self.method in (
            "ci_from_wilson", "ci_from_bootstrap", "ci_from_weighted_bootstrap",
        )

    @property
    def needs_weights(self) -> bool:
        """Whether this method requires per-sample weights (``_raw__cm_{col}`` columns).

        Weighted methods use per-sample choice mass as importance weights,
        producing wider CIs when the model allocates little probability to
        the target answer tokens.
        """
        return self.method == "ci_from_weighted_bootstrap"

    @property
    def label(self) -> str:
        """Human-readable label for plot legends."""
        if self.method == "std":
            return "±1 SD"
        assert self.confidence is not None
        conf = f"{self.confidence:g}%"
        if self.method == "ci_from_std":
            return f"{conf} CI (normal)"
        if self.method == "ci_from_ppf":
            return f"{conf} CI (t)"
        if self.method == "ci_from_wilson":
            return f"{conf} CI (Wilson)"
        if self.method == "ci_from_bootstrap":
            return f"{conf} CI (bootstrap, {self.n_resamples})"
        if self.method == "ci_from_weighted_bootstrap":
            return f"{conf} CI (mass-weighted bootstrap, {self.n_resamples})"
        return f"{conf} CI"


# ---------------------------------------------------------------------------
# Interval computation functions
# ---------------------------------------------------------------------------


def _interval_std(values: np.ndarray) -> float:
    """Sample standard deviation (ddof=1)."""
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1))


def _interval_ci_from_std(values: np.ndarray, confidence: float) -> float:
    """CI half-width using normal approximation: z * std / sqrt(n)."""
    from scipy import stats

    n = len(values)
    if n <= 1:
        return 0.0
    z = stats.norm.ppf(1 - (1 - confidence / 100) / 2)
    return float(z * values.std(ddof=1) / np.sqrt(n))


def _interval_ci_from_ppf(values: np.ndarray, confidence: float) -> float:
    """CI half-width using Student's t-distribution."""
    from scipy import stats

    n = len(values)
    if n <= 1:
        return 0.0
    alpha = 1 - confidence / 100
    t_val = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return float(t_val * values.std(ddof=1) / np.sqrt(n))


def _interval_ci_from_wilson(values: np.ndarray, confidence: float) -> tuple[float, float]:
    """Wilson score interval for binary (0/1) data.

    Returns:
        ``(ci_lower, ci_upper)`` as absolute bounds.

    Raises:
        ValueError: If the data contains values other than 0 and 1.
    """
    from scipy import stats

    unique = np.unique(values)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(
            f"Wilson interval requires binary (0/1) data, "
            f"got unique values: {unique.tolist()}"
        )
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    p_hat = values.mean()
    z = stats.norm.ppf(1 - (1 - confidence / 100) / 2)
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))
    return (float(centre - margin), float(centre + margin))


def _interval_ci_from_bootstrap(
    values: np.ndarray,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    """CI via BCa bootstrap on the mean.

    Returns:
        ``(ci_lower, ci_upper)`` as absolute bounds.
    """
    from scipy import stats

    n = len(values)
    if n <= 1:
        return (0.0, 0.0)
    # Degenerate case: all values identical → zero-width CI, skip BCa which
    # would emit DegenerateDataWarning and return NaN.
    if np.ptp(values) == 0.0:
        m = float(values[0])
        return (m, m)
    rng = np.random.default_rng(seed)
    result = stats.bootstrap(
        (values,),
        statistic=np.mean,
        n_resamples=n_resamples,
        confidence_level=confidence / 100,
        random_state=rng,
        method="BCa",
    )
    return (float(result.confidence_interval.low), float(result.confidence_interval.high))


def _interval_ci_from_weighted_bootstrap(
    values: np.ndarray,
    weights: np.ndarray,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    """CI via noise-injection bootstrap using choice mass to model measurement uncertainty.

    Standard bootstrap CIs on logprob-based MCQ scores can be misleadingly
    narrow when the model places most probability on non-answer tokens.  The
    softmax-renormalized score is computed from only the answer-token
    probabilities, discarding information about *how little* of the
    distribution was actually observed.  When all samples have similarly low
    choice mass, per-sample scores are consistent but unreliable — the
    resulting CIs capture sampling uncertainty but miss measurement
    uncertainty entirely.

    This method injects per-sample noise proportional to ``(1 - choice_mass)``
    to recover honest uncertainty estimates.  For each bootstrap iteration, every
    resampled score is replaced by a mixture:

        adjusted_i = cm_i * score_i + (1 - cm_i) * U_i

    where ``cm_i`` is the choice mass for sample *i* and ``U_i ~ Uniform(0, 1)``
    represents maximal ignorance about the trait score for the probability mass
    that did NOT land on answer tokens.  The intuition: the ``cm``-fraction of
    the distribution told us ``score``; for the ``(1-cm)``-fraction we know
    nothing, so we model it as a random draw.

    Effects on confidence intervals:

    * **cm ≈ 1** (model focused on ABCD): noise ≈ 0, CI matches the standard
      unweighted bootstrap.
    * **cm ≈ 0** (model not answering ABCD): adjusted scores ≈ U(0,1), CI
      converges to the maximum-ignorance interval around 0.5.

    The corresponding point estimate uses the expected value of the noise term
    (0.5) in place of the stochastic ``U_i``::

        E[adjusted_i] = cm_i * score_i + (1 - cm_i) * 0.5

    Methodology and supporting references:

    * The noise-injection mechanism is analogous to a *measurement-error
      model* (Carroll et al., 2006) where each observation has known, sample-
      specific error variance.  Here the error variance is governed by the
      complement of the choice mass — a direct, observable reliability
      indicator — rather than requiring a separate error-variance estimate:
          Carroll, R. J., Ruppert, D., Stefanski, L. A., & Crainiceanu, C. M.
          (2006). Measurement Error in Nonlinear Models (2nd ed.). Chapman &
          Hall/CRC.

    * Wang et al. (2024) show that first-token logprob distributions and
      text-generated answers diverge by >60 % for instruction-tuned models,
      undermining the reliability of renormalized ABCD probabilities when the
      model is not "trying" to output a letter:
          "My Answer is C: First-Token Probabilities Do Not Match
           Text Answers in API-Based LLMs"  (arXiv:2402.14499)

    * Huang et al. (2025) model next-token logits as Dirichlet concentration
      parameters and show that softmax normalization destroys evidence-strength
      information.  Choice mass is a lightweight proxy for the total Dirichlet
      concentration (alpha_0) on the answer set:
          "LogU: Accurate LLM Log-Probability Estimation with
           Uncertainty"  (arXiv:2502.00290)

    * The choice of U(0,1) as the ignorance distribution for the unobserved
      mass follows maximum-entropy reasoning: for a trait score known to lie
      in [0, 1] with no further information, the uniform distribution has
      the highest entropy (Jaynes, 2003):
          Jaynes, E. T. (2003). Probability Theory: The Logic of Science.
          Cambridge University Press.

    Args:
        values: Per-sample trait scores (0–1), from softmax-renormalized logprobs.
        weights: Per-sample choice mass (0–1), the fraction of the model's
            probability distribution on answer tokens.
        confidence: Confidence level in percent, e.g. 95.0.
        n_resamples: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        ``(ci_lower, ci_upper)`` as absolute bounds.
    """
    n = len(values)
    if n <= 1:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        cm = weights[idx]
        scores = values[idx]
        noise = rng.uniform(0.0, 1.0, size=n)
        adjusted = cm * scores + (1.0 - cm) * noise
        boot_means[i] = adjusted.mean()

    alpha = (100 - confidence) / 200  # half-alpha for two-sided
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return (float(lo), float(hi))


def _resolve_interval_fn(
    method: IntervalMethod,
) -> Callable[[np.ndarray], float | tuple[float, float]]:
    """Return a callable ``(values) -> result`` from an IntervalMethod.

    Symmetric methods (std, ci_from_std, ci_from_ppf) return a ``float``
    half-width.  Asymmetric methods (ci_from_wilson, ci_from_bootstrap) return
    a ``(ci_lower, ci_upper)`` tuple of absolute bounds.
    """
    if method.method == "std":
        return _interval_std
    if method.method == "ci_from_std":
        return partial(_interval_ci_from_std, confidence=method.confidence)
    if method.method == "ci_from_ppf":
        return partial(_interval_ci_from_ppf, confidence=method.confidence)
    if method.method == "ci_from_wilson":
        return partial(_interval_ci_from_wilson, confidence=method.confidence)
    if method.method == "ci_from_bootstrap":
        return partial(
            _interval_ci_from_bootstrap,
            confidence=method.confidence,
            n_resamples=method.n_resamples,
            seed=method.seed,
        )
    if method.method == "ci_from_weighted_bootstrap":
        # Weighted bootstrap has a (values, weights) -> (lo, hi) signature;
        # _agg_sweep handles the weights argument specially.
        return partial(
            _interval_ci_from_weighted_bootstrap,
            confidence=method.confidence,
            n_resamples=method.n_resamples,
            seed=method.seed,
        )
    raise ValueError(f"Unknown interval method: {method.method!r}")


def _build_mass_mask(
    cm_all: np.ndarray,
    nc_all: np.ndarray | None,
    min_choice_mass: float,
    dynamic_mass_filter: bool,
) -> np.ndarray:
    """Build a boolean mask combining dynamic and fixed choice-mass filters.

    Args:
        cm_all: Per-sample choice mass values.
        nc_all: Per-sample num_choices values (for dynamic threshold).
            May be None if not available.
        min_choice_mass: Fixed minimum threshold (0 = no fixed filter).
        dynamic_mass_filter: If True, apply per-question 1/num_choices filter.

    Returns:
        Boolean mask — True for samples to keep.
    """
    mask = np.ones(len(cm_all), dtype=bool)
    if dynamic_mass_filter and nc_all is not None and len(nc_all) == len(cm_all):
        dynamic_thresholds = 1.0 / nc_all
        mask &= cm_all >= dynamic_thresholds
    if min_choice_mass > 0.0:
        mask &= cm_all >= min_choice_mass
    return mask


def _agg_sweep(
    df: pd.DataFrame,
    cols: list[str],
    interval: IntervalMethod | None = None,
    min_choice_mass: float = MIN_CHOICE_MASS_DEFAULT,
    dynamic_mass_filter: bool = True,
) -> pd.DataFrame:
    """Aggregate a sweep DataFrame to mean ± interval per scale point.

    Returns a DataFrame with columns: ``scale``, ``{col}_mean``, and — when
    *interval* is not None — interval columns for each *col*.

    Symmetric methods produce a single ``{col}_ci`` column (half-width).
    Asymmetric methods (Wilson, bootstrap) produce ``{col}_ci_low`` and
    ``{col}_ci_high`` columns with absolute bounds.

    Methods with :pyattr:`IntervalMethod.needs_raw_scores` (Wilson, bootstrap)
    compute CIs from the raw per-sample scores in ``_raw_{col}`` list columns
    (populated by :func:`_load_from_info`).  A ``ValueError`` is raised if
    these columns are missing.

    Two-level choice-mass filtering:

    1. **Dynamic** (``dynamic_mass_filter=True``): per-question threshold of
       ``1/num_choices`` using ``_raw__nc_{col}`` columns.
    2. **Fixed** (``min_choice_mass > 0``): global threshold applied on top.

    Args:
        df: Sweep DataFrame with per-run rows.
        cols: Metric columns to aggregate.
        interval: Error bar method.
        min_choice_mass: When > 0, exclude per-sample scores whose choice
            mass is below this threshold.  Requires ``_raw__cm_{col}``
            columns (logprob evals).  The mean is recomputed from the
            filtered raw scores.  Default 0.0 (no filtering).
        dynamic_mass_filter: When True, exclude per-sample scores whose
            choice mass is below ``1/num_choices``.  Requires
            ``_raw__nc_{col}`` columns.  Default True.
    """
    interval_fn = _resolve_interval_fn(interval) if interval is not None else None
    needs_raw = interval is not None and interval.needs_raw_scores
    needs_weights = interval is not None and interval.needs_weights
    asymmetric = needs_raw  # raw-score methods always produce asymmetric bounds
    # Choice-mass filtering also requires raw scores to recompute the mean.
    filter_by_mass = min_choice_mass > 0.0 or dynamic_mass_filter
    rows = []
    for scale, grp in df.groupby("scale"):
        row: dict = {"scale": scale}
        for col in cols:
            if col not in grp.columns:
                row[f"{col}_mean"] = float("nan")
                if interval_fn is not None:
                    if asymmetric:
                        row[f"{col}_ci_low"] = float("nan")
                        row[f"{col}_ci_high"] = float("nan")
                    else:
                        row[f"{col}_ci"] = 0.0
                continue

            # --- Choice-mass filtering: recompute mean from raw scores ---
            if filter_by_mass:
                raw_col = f"_raw_{col}"
                cm_col = f"_raw__cm_{col}"
                nc_col = f"_raw__nc_{col}"
                if raw_col in grp.columns and cm_col in grp.columns:
                    raw_lists = grp[raw_col].dropna().tolist()
                    cm_lists = grp[cm_col].dropna().tolist()
                    raw_all = np.concatenate(raw_lists) if raw_lists else np.array([])
                    cm_all = np.concatenate(cm_lists) if cm_lists else np.array([])
                    # Load num_choices arrays for dynamic filtering.
                    nc_all = None
                    if dynamic_mass_filter and nc_col in grp.columns:
                        nc_lists = grp[nc_col].dropna().tolist()
                        nc_all = np.concatenate(nc_lists) if nc_lists else None
                    min_len = min(len(raw_all), len(cm_all))
                    raw_all = raw_all[:min_len]
                    cm_all = cm_all[:min_len]
                    if nc_all is not None:
                        nc_all = nc_all[:min_len]
                    mask = _build_mass_mask(cm_all, nc_all, min_choice_mass, dynamic_mass_filter)
                    filtered = raw_all[mask]
                    mean = float(filtered.mean()) if len(filtered) else float("nan")
                else:
                    vals = grp[col].dropna().values
                    mean = vals.mean() if len(vals) else float("nan")
            else:
                vals = grp[col].dropna().values
                mean = vals.mean() if len(vals) else float("nan")
            row[f"{col}_mean"] = mean
            if interval_fn is not None:
                if needs_raw:
                    raw_col = f"_raw_{col}"
                    if raw_col not in grp.columns:
                        # No per-sample data for this column (e.g. summary
                        # metrics like logprob_mcq_ratio).  Skip CI — the
                        # mean is still computed from the aggregate values.
                        if asymmetric:
                            row[f"{col}_ci_low"] = float("nan")
                            row[f"{col}_ci_high"] = float("nan")
                        else:
                            row[f"{col}_ci"] = 0.0
                        continue
                    # Concatenate raw score lists across all runs in this group
                    raw_lists = grp[raw_col].dropna().tolist()
                    raw_all = np.concatenate(raw_lists) if raw_lists else np.array([])

                    # Apply choice-mass filter to CI raw scores too.
                    cm_col = f"_raw__cm_{col}"
                    nc_col = f"_raw__nc_{col}"
                    if filter_by_mass and cm_col in grp.columns:
                        cm_lists = grp[cm_col].dropna().tolist()
                        cm_all = np.concatenate(cm_lists) if cm_lists else np.array([])
                        nc_all = None
                        if dynamic_mass_filter and nc_col in grp.columns:
                            nc_lists = grp[nc_col].dropna().tolist()
                            nc_all = np.concatenate(nc_lists) if nc_lists else None
                        _ml = min(len(raw_all), len(cm_all))
                        raw_all = raw_all[:_ml]
                        cm_all = cm_all[:_ml]
                        if nc_all is not None:
                            nc_all = nc_all[:_ml]
                        mask = _build_mass_mask(cm_all, nc_all, min_choice_mass, dynamic_mass_filter)
                        raw_all = raw_all[mask]
                        # Also filter weights if needed for weighted bootstrap.
                        if needs_weights:
                            cm_all = cm_all[mask]

                    if len(raw_all) == 0:
                        row[f"{col}_ci_low"] = float("nan")
                        row[f"{col}_ci_high"] = float("nan")
                    elif needs_weights:
                        # Noise-injection bootstrap: use per-sample choice mass
                        # to model measurement uncertainty from low coverage.
                        if not (filter_by_mass and cm_col in grp.columns):
                            # Weights not yet loaded from filtering path above.
                            weight_col = f"_raw__cm_{col}"
                            if weight_col in grp.columns:
                                wt_lists = grp[weight_col].dropna().tolist()
                                wt_all = np.concatenate(wt_lists) if wt_lists else np.ones(len(raw_all))
                            else:
                                wt_all = np.ones(len(raw_all))
                            min_len = min(len(raw_all), len(wt_all))
                            raw_all = raw_all[:min_len]
                            wt_all = wt_all[:min_len]
                        else:
                            wt_all = cm_all  # Already filtered above.
                        low, high = interval_fn(raw_all, wt_all)  # type: ignore[misc]
                        # Point estimate: E[cm * score + (1-cm) * U(0,1)]
                        #               = cm * score + (1-cm) * 0.5
                        adjusted = wt_all * raw_all + (1.0 - wt_all) * 0.5
                        row[f"{col}_mean"] = float(adjusted.mean())
                        row[f"{col}_ci_low"] = low
                        row[f"{col}_ci_high"] = high
                    else:
                        low, high = interval_fn(raw_all)  # type: ignore[misc]
                        row[f"{col}_ci_low"] = low
                        row[f"{col}_ci_high"] = high
                else:
                    if not filter_by_mass:
                        row[f"{col}_ci"] = interval_fn(vals)
                    else:
                        # Symmetric CI methods don't use raw scores, but we
                        # still need to filter.  Fall back to unfiltered vals
                        # since symmetric methods can't use raw+cm pairs.
                        row[f"{col}_ci"] = interval_fn(vals)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("scale").reset_index(drop=True)


def _metric_cols(df: pd.DataFrame) -> list[str]:
    """Return metric columns from a sweep DataFrame (everything except housekeeping cols)."""
    return [c for c in df.columns if c not in ("model", "run", "scale", "_parse_rate", "stderr")
            and not c.startswith("_raw_")]


# ---------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------

def print_sweep_table(agg: pd.DataFrame, cols: list[str], title: str) -> None:
    has_sym_ci = cols and f"{cols[0]}_ci" in agg.columns
    has_asym_ci = cols and f"{cols[0]}_ci_low" in agg.columns
    width = 10 + 26 * len(cols) if has_asym_ci else 10 + 20 * len(cols)
    print(f"\n{'=' * width}")
    print(title)
    print("=" * width)
    header = f"{'Scale':<10}" + "".join(f"{c:<{26 if has_asym_ci else 20}}" for c in cols)
    print(header)
    print("-" * width)
    for _, row in agg.iterrows():
        marker = " ← baseline" if abs(row["scale"]) < 0.01 else ""
        if has_sym_ci:
            vals = "".join(f"{row[f'{c}_mean']:.4f}±{row[f'{c}_ci']:.4f}{'':>6}" for c in cols)
        elif has_asym_ci:
            vals = "".join(
                f"{row[f'{c}_mean']:.4f} [{row[f'{c}_ci_low']:.4f},{row[f'{c}_ci_high']:.4f}] "
                for c in cols
            )
        else:
            vals = "".join(f"{row[f'{c}_mean']:.4f}{'':>14}" for c in cols)
        print(f"{row['scale']:+7.2f}   {vals}{marker}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

def _mock_sweep_data() -> SweepData:
    """Generate realistic mock SweepData for offline testing."""
    rng = np.random.default_rng(42)
    scales = [s for s in np.arange(-2.0, 2.25, 0.25) if round(s, 10) != 0.0]
    base_scales = [0.0] + list(scales)

    def _df(records: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        return _normalise_scale_col(df).sort_values(["scale", "run"]).reset_index(drop=True)

    # BFI: Big Five only, 3 runs per scale
    bfi_base  = {"Openness": 0.65, "Conscientiousness": 0.70, "Extraversion": 0.50,
                 "Agreeableness": 0.60, "Neuroticism": 0.45}
    bfi_slope = {"Openness": 0.08, "Conscientiousness": -0.03, "Extraversion": 0.05,
                 "Agreeableness": -0.12, "Neuroticism": 0.06}
    def _mock_parse_rate(s: float) -> float:
        """Simulate parse degradation at extreme scales (perfect in the middle)."""
        return float(np.clip(1.0 - 0.08 * max(0.0, abs(s) - 1.0) + rng.normal(0, 0.02), 0, 1))

    bfi_records = []
    for s in base_scales:
        for run in ["run_1", "run_2", "run_3"]:
            scores = {t: float(np.clip(bfi_base[t] + bfi_slope[t] * s + rng.normal(0, 0.025), 0, 1))
                      for t in BIG_FIVE}
            bfi_records.append({"model": "base" if s == 0.0 else f"lora_{s:+.2f}x",
                                 "run": run, "scale": s, "_parse_rate": _mock_parse_rate(s), **scores})

    # TRAIT: Big Five + Dark Triad, 3 runs per scale
    trait_base  = {**bfi_base, "Machiavellianism": 0.40, "Narcissism": 0.45, "Psychopathy": 0.35}
    trait_slope = {**bfi_slope, "Machiavellianism": 0.03, "Narcissism": 0.02, "Psychopathy": 0.01}
    trait_records = []
    for s in base_scales:
        for run in ["run_1", "run_2", "run_3"]:
            scores = {t: float(np.clip(trait_base[t] + trait_slope[t] * s + rng.normal(0, 0.025), 0, 1))
                      for t in ALL_TRAIT_COLS}
            trait_records.append({"model": "base" if s == 0.0 else f"lora_{s:+.2f}x",
                                   "run": run, "scale": s, "_parse_rate": _mock_parse_rate(s), **scores})

    # MMLU: coarser grid, 3 runs
    mmlu_scales = [s for s in np.arange(-2.0, 2.25, 0.5) if round(s, 10) != 0.0]
    mmlu_records = []
    for s in [0.0] + list(mmlu_scales):
        for run in ["run_1", "run_2", "run_3"]:
            acc = float(np.clip(0.62 - 0.04 * abs(s) + rng.normal(0, 0.01), 0, 1))
            mmlu_records.append({"model": "base" if s == 0.0 else f"lora_{s:+.2f}x",
                                  "run": run, "scale": s, "_parse_rate": _mock_parse_rate(s), "accuracy": acc})

    return SweepData(evals={
        "bfi":   _df(bfi_records),
        "trait": _df(trait_records),
        "mmlu":  _df(mmlu_records),
    })


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _setup_matplotlib() -> None:
    import matplotlib
    matplotlib.use("Agg")


def _draw_error_bars(
    ax, scales, means, cis=None, *, ci_low=None, ci_high=None, color=None,
) -> None:
    """Draw vertical error bars at each scale point.

    Accepts either symmetric half-widths (*cis*) or asymmetric absolute bounds
    (*ci_low*, *ci_high*).  No-op if all intervals are zero-width.
    """
    if ci_low is not None and ci_high is not None:
        means_arr = np.array(means, dtype=float)
        lo = np.array(ci_low, dtype=float)
        hi = np.array(ci_high, dtype=float)
        # Mask out points where mean or bounds are nan
        valid = np.isfinite(means_arr) & np.isfinite(lo) & np.isfinite(hi)
        if not np.any(valid):
            return
        yerr = np.array([means_arr - lo, hi - means_arr])
        # Clamp to non-negative (floating-point arithmetic can produce tiny
        # negative values when mean ≈ bound) and zero out nan points.
        np.clip(yerr, 0.0, None, out=yerr)
        yerr[:, ~valid] = 0.0
        if not np.any(yerr > 0):
            return
    elif cis is not None:
        yerr = np.array(cis)
        if not np.any(yerr > 0):
            return
    else:
        return
    ax.errorbar(scales, means, yerr=yerr,
                fmt="none", color=color, capsize=3, capthick=1.0,
                elinewidth=1.0, alpha=0.7, zorder=5)


def _draw_col_error_bars(ax, agg: pd.DataFrame, col: str, scales, means, color) -> None:
    """Draw error bars for *col* from an aggregated DataFrame.

    Handles both symmetric (``{col}_ci``) and asymmetric
    (``{col}_ci_low`` / ``{col}_ci_high``) columns automatically.
    """
    if f"{col}_ci" in agg.columns:
        _draw_error_bars(ax, scales, means, cis=agg[f"{col}_ci"].values, color=color)
    elif f"{col}_ci_low" in agg.columns and f"{col}_ci_high" in agg.columns:
        _draw_error_bars(
            ax, scales, means,
            ci_low=agg[f"{col}_ci_low"].values,
            ci_high=agg[f"{col}_ci_high"].values,
            color=color,
        )


def _set_scale_xticks(ax, scales, x_lim: tuple[float, float] | None = (-4.5, 4.5)) -> None:
    """Set x-axis ticks at every scale point, labelling multiples of 0.5.

    All scale points get a tick mark. Labels are shown only at multiples of
    0.5 (or at every point if all points already fall on 0.5 steps), so the
    axis stays readable without rotation even for dense fine-grained grids.

    Args:
        x_lim: X-axis limits as (min, max). Defaults to (-4.5, 4.5). Pass None
            to auto-scale.
    """
    ax.set_xticks(scales)
    half_scales = {s for s in scales if round(float(s) * 2) == float(s) * 2}
    if len(half_scales) < len(scales):
        ax.set_xticklabels([f"{s:g}" if s in half_scales else "" for s in scales])
    else:
        ax.set_xticklabels([f"{s:g}" for s in scales])
    if x_lim is not None:
        ax.set_xlim(*x_lim)


def plot_trait_sweep(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = "",
    highlight: list[str] | None = None,
    interval: IntervalMethod | None = None,
    min_choice_mass: float = MIN_CHOICE_MASS_DEFAULT,
    dynamic_mass_filter: bool = True,
    x_label: str = "LoRA scaling factor",
    x_lim: tuple[float, float] | None = None,
) -> Path:
    """Primary research plot: TRAIT Big Five + Dark Triad + human baselines.

    When the DataFrame contains ``_choice_mass`` data (logprob-based evals),
    a compact diagnostic sub-axis is drawn below the main plot showing the
    fraction of probability mass on choice letters vs. scale.

    Args:
        df: DataFrame with columns: scale, run, Openness, ..., Psychopathy.
        output_dir: Directory to save the figure.
        title_suffix: Optional suffix appended to the figure title.
        highlight: Traits to render at full brightness. Accepts full names or OCEAN
            single letters (O/C/E/A/N). Unlisted Big Five traits are dimmed.
            Dark Triad is always dimmed regardless. Defaults to all Big Five.
        interval: Error bar method. None to omit error bars.

    Returns:
        Path to the saved figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    lit = _resolve_highlight(highlight)
    trait_agg = _agg_sweep(df, ALL_TRAIT_COLS, interval=interval, min_choice_mass=min_choice_mass, dynamic_mass_filter=dynamic_mass_filter)
    scales = trait_agg["scale"].values

    # Detect whether choice-mass diagnostics are available.
    has_choice_mass = "_choice_mass" in df.columns and df["_choice_mass"].notna().any()

    if has_choice_mass:
        fig = plt.figure(figsize=(12, 6.6))
        gs = GridSpec(2, 1, height_ratios=[85, 15], hspace=0.05, figure=fig)
        ax = fig.add_subplot(gs[0])
        ax_cm = fig.add_subplot(gs[1], sharex=ax)
    else:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax_cm = None

    # --- Big Five: lit at full color, dimmed if not in highlight ---
    for trait in BIG_FIVE:
        color = BIG_FIVE_COLORS[trait]
        means = trait_agg[f"{trait}_mean"].values
        if trait in lit:
            ax.plot(scales, means, "o-", color=color, linewidth=2.2, markersize=6,
                    label=trait, zorder=4)
            _draw_col_error_bars(ax, trait_agg, trait, scales, means, color)
        else:
            ax.plot(scales, means, "o-", color=color, linewidth=1.4, markersize=4,
                    alpha=0.35, label=trait, zorder=3)

    # --- Dark Triad: always dimmed dashed, no error bars; skip if absent ---
    for trait in DARK_TRIAD:
        col = f"{trait}_mean"
        if col not in trait_agg.columns or trait_agg[col].isna().all():
            continue
        color = DARK_TRIAD_COLORS[trait]
        means = trait_agg[col].values
        ax.plot(scales, means, "--", color=color, linewidth=1.4, markersize=4,
                alpha=0.35, label=trait, zorder=3)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
    ax.set_ylabel("Trait score (0–1)", fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)

    title = "TRAIT sweep: personality scores vs. LoRA scale"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title, fontsize=13, fontweight="bold")

    if interval is not None:
        ax.errorbar([], [], yerr=1, fmt="none", color="gray", capsize=3, capthick=1.0,
                    elinewidth=1.0, alpha=0.7, label=interval.label)

    # --- Choice-mass diagnostic sub-axis ---
    if ax_cm is not None:
        # Aggregate per-sample choice mass at each scale, restricted to the
        # samples that survived the `min_choice_mass` filter used for the
        # trait scores above.  Pulled from raw per-sample values in
        # `_raw__choice_mass` (pooled across all runs at a given scale).
        cm_rows = []
        for scale, grp in df.groupby("scale"):
            if "_raw__choice_mass" in grp.columns:
                lists = [v for v in grp["_raw__choice_mass"].tolist()
                         if isinstance(v, list) and v]
                cm_all = np.concatenate(lists) if lists else np.array([])
            else:
                cm_all = grp["_choice_mass"].dropna().values
            if min_choice_mass > 0.0:
                cm_all = cm_all[cm_all >= min_choice_mass]
            if len(cm_all):
                cm_rows.append({"scale": scale, "mean": float(cm_all.mean()),
                                "min": float(cm_all.min()), "max": float(cm_all.max())})
            else:
                cm_rows.append({"scale": scale, "mean": float("nan"),
                                "min": float("nan"), "max": float("nan")})
        cm_agg = pd.DataFrame(cm_rows).sort_values("scale")
        cm_scales = cm_agg["scale"].values
        cm_means = cm_agg["mean"].values

        ax_cm.fill_between(cm_scales, cm_agg["min"].values, cm_agg["max"].values,
                           color="#888888", alpha=0.15)
        ax_cm.plot(cm_scales, cm_means, "s-", color="#555555", linewidth=1.4,
                   markersize=3, zorder=4)
        ax_cm.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
        ax_cm.set_ylabel("Choice\nmass", fontsize=8, rotation=0, labelpad=32, va="center")
        cm_lower = max(0.0, min(float(min_choice_mass), 1.0))
        ax_cm.set_ylim(cm_lower, 1.0)
        ax_cm.set_yticks([cm_lower, 1.0])
        ax_cm.set_yticklabels([f"{cm_lower:g}", "1"], fontsize=7)
        ax_cm.grid(True, alpha=0.25)
        ax_cm.set_xlabel(x_label, fontsize=11)
        _set_scale_xticks(ax_cm, scales, x_lim=x_lim)
        # Hide x-axis labels on the main plot since the sub-axis provides them.
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel(x_label, fontsize=11)
        _set_scale_xticks(ax, scales, x_lim=x_lim)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13 if ax_cm is None else -0.35),
              fontsize=9, ncol=6, framealpha=0.85)

    if ax_cm is None:
        plt.tight_layout()
    out = output_dir / "trait_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


def plot_bfi_sweep(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = "",
    highlight: list[str] | None = None,
    interval: IntervalMethod | None = None,
) -> Path:
    """Sanity-check plot: BFI Big Five centred at baseline (delta from scale=0).

    Args:
        df: DataFrame with columns: scale, run, Openness, ..., Neuroticism.
        output_dir: Directory to save the figure.
        title_suffix: Optional suffix appended to the figure title.
        highlight: Traits to render at full brightness. Accepts full names or OCEAN
            single letters (O/C/E/A/N). Unlisted traits are dimmed.
            Defaults to all Big Five.
        interval: Error bar method. None to omit error bars.

    Returns:
        Path to the saved figure.
    """
    import matplotlib.pyplot as plt

    lit = _resolve_highlight(highlight)
    bfi_agg = _agg_sweep(df, BIG_FIVE, interval=interval)
    has_sym_ci = f"{BIG_FIVE[0]}_ci" in bfi_agg.columns
    has_asym_ci = f"{BIG_FIVE[0]}_ci_low" in bfi_agg.columns

    # Compute per-trait baseline (scale=0) mean for delta calculation.
    baseline_row = bfi_agg[bfi_agg["scale"].abs() < 1e-9]
    if baseline_row.empty:
        baseline_row = bfi_agg.loc[[bfi_agg["scale"].abs().idxmin()]]

    scales = bfi_agg["scale"].values
    fig, ax = plt.subplots(figsize=(12, 5))

    all_delta_means: list[float] = []
    all_delta_cis:   list[float] = []

    for trait in BIG_FIVE:
        color = BIG_FIVE_COLORS[trait]
        baseline_val = float(baseline_row[f"{trait}_mean"].iloc[0])
        means = bfi_agg[f"{trait}_mean"].values - baseline_val
        if trait in lit:
            ax.plot(scales, means, "o-", color=color, linewidth=2.2, markersize=6,
                    label=trait, zorder=4)
            if has_sym_ci:
                cis = bfi_agg[f"{trait}_ci"].values
                _draw_error_bars(ax, scales, means, cis=cis, color=color)
                all_delta_cis.extend(cis.tolist())
            elif has_asym_ci:
                ci_low = bfi_agg[f"{trait}_ci_low"].values - baseline_val
                ci_high = bfi_agg[f"{trait}_ci_high"].values - baseline_val
                _draw_error_bars(ax, scales, means, ci_low=ci_low, ci_high=ci_high, color=color)
                all_delta_cis.extend((means - ci_low).tolist())
                all_delta_cis.extend((ci_high - means).tolist())
        else:
            ax.plot(scales, means, "o-", color=color, linewidth=1.4, markersize=4,
                    alpha=0.35, label=trait, zorder=3)
        all_delta_means.extend(means[~np.isnan(means)].tolist())

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6,
               label="Baseline (s=0)", zorder=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.3, zorder=1)
    _set_scale_xticks(ax, scales)

    if all_delta_means:
        max_ci   = max(all_delta_cis) if all_delta_cis else 0.0
        data_min = min(all_delta_means) - max_ci
        data_max = max(all_delta_means) + max_ci
        margin   = max(0.03, (data_max - data_min) * 0.15)
        ax.set_ylim(data_min - margin, data_max + margin)

    ax.set_xlabel("LoRA scaling factor", fontsize=11)
    ax.set_ylabel("Δ trait score (vs. baseline)", fontsize=11)
    ax.grid(True, alpha=0.25)

    title = "BFI sweep: trait delta from baseline"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title, fontsize=13, fontweight="bold")

    if interval is not None:
        ax.errorbar([], [], yerr=1, fmt="none", color="gray", capsize=3, capthick=1.0,
                    elinewidth=1.0, alpha=0.7, label=interval.label)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
              fontsize=9, ncol=7, framealpha=0.85)

    plt.tight_layout()
    out = output_dir / "bfi_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


def plot_capability_sweep(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = "",
    eval_name: str = "capability",
    random_baseline: float | None = None,
    interval: IntervalMethod | None = None,
    min_choice_mass: float = MIN_CHOICE_MASS_DEFAULT,
    dynamic_mass_filter: bool = True,
    x_label: str = "LoRA scaling factor",
) -> Path:
    """Capability coherence plot: accuracy vs. LoRA scale with baseline reference.

    Suitable for any accuracy-like eval (mmlu, gsm8k, truthfulqa, arc, etc.).

    Args:
        df: DataFrame with columns: scale, run, accuracy.
        output_dir: Directory to save the figure.
        title_suffix: Optional suffix appended to the figure title.
        eval_name: Eval name used for the figure title and output filename.
        random_baseline: If set, draws a horizontal dashed red line at this accuracy
            level (e.g. 0.25 for 4-choice MCQ random chance).
        interval: Error bar method. None to omit error bars.
        min_choice_mass: Fixed min choice-mass filter (logprob evals).
        dynamic_mass_filter: Per-question 1/num_choices filter (logprob evals).

    Returns:
        Path to the saved figure.
    """
    import matplotlib.pyplot as plt

    cap_agg = _agg_sweep(df, ["accuracy"], interval=interval, min_choice_mass=min_choice_mass, dynamic_mass_filter=dynamic_mass_filter)
    scales = cap_agg["scale"].values
    means  = cap_agg["accuracy_mean"].values

    # Compute y-axis bounds from whichever CI columns are present
    if "accuracy_ci" in cap_agg.columns:
        ci_extent_low = means - cap_agg["accuracy_ci"].values
        ci_extent_high = means + cap_agg["accuracy_ci"].values
    elif "accuracy_ci_low" in cap_agg.columns:
        ci_extent_low = cap_agg["accuracy_ci_low"].values
        ci_extent_high = cap_agg["accuracy_ci_high"].values
    else:
        ci_extent_low = means
        ci_extent_high = means

    baseline_idx = int(np.argmin(np.abs(scales)))
    baseline_acc = float(means[baseline_idx])

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.axhline(baseline_acc, color="#388E3C", linewidth=1.2,
               linestyle="--", alpha=0.8, zorder=2, label=f"Baseline ({baseline_acc:.3f})")

    if random_baseline is not None:
        ax.axhline(random_baseline, color="#EF5350", linewidth=1.0,
                   linestyle=":", alpha=0.7, zorder=2, label=f"Random ({random_baseline:.0%})")

    color = "#5C6BC0"
    ax.plot(scales, means, "o-", color=color, linewidth=2.2, markersize=6,
            label="accuracy", zorder=4)
    _draw_col_error_bars(ax, cap_agg, "accuracy", scales, means, color)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    _set_scale_xticks(ax, scales)

    y_min_candidates = [float(np.nanmin(ci_extent_low))]
    if random_baseline is not None:
        y_min_candidates.append(random_baseline)
    y_min = min(y_min_candidates) - 0.02
    y_max = max(float(np.nanmax(ci_extent_high)), baseline_acc) + 0.04
    ax.set_ylim(y_min, y_max)

    ax.grid(True, alpha=0.25)

    title = f"{eval_name} sweep: capability coherence vs. LoRA scale"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title, fontsize=13, fontweight="bold")

    if interval is not None:
        ax.errorbar([], [], yerr=1, fmt="none", color="gray", capsize=3, capthick=1.0,
                    elinewidth=1.0, alpha=0.7, label=interval.label)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              fontsize=9, ncol=4, framealpha=0.85)

    plt.tight_layout()
    out = output_dir / f"{eval_name}_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


def plot_capability_breakdown(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = "",
    eval_name: str = "capability",
) -> Path | None:
    """Stacked bar chart: Correct / Recovered / Wrong / No answer per LoRA scale.

    Categories:
      - Correct: Inspect scorer parsed and matched target
      - Recovered: Inspect failed to parse, but fallback regex found the right letter
      - Wrong answer: a letter was parsed (by Inspect or fallback) but it was wrong
      - No answer: no parseable letter at all

    Requires ``_raw_accuracy``, ``_raw__answer_parsed``, and
    ``_raw__reparsed_accuracy`` columns from :func:`_extract_raw_sample_scores`.
    """
    import matplotlib.pyplot as plt

    raw_acc_col = "_raw_accuracy"
    raw_ap_col = "_raw__answer_parsed"
    raw_rp_col = "_raw__reparsed_accuracy"
    if raw_acc_col not in df.columns or raw_ap_col not in df.columns:
        print(f"  skip {eval_name}_breakdown: missing raw columns")
        return None
    has_reparse = raw_rp_col in df.columns

    rows = []
    for scale, grp in df.groupby("scale"):
        acc_lists = grp[raw_acc_col].dropna().tolist()
        ap_lists = grp[raw_ap_col].dropna().tolist()
        acc = np.concatenate(acc_lists) if acc_lists else np.array([])
        ap = np.concatenate(ap_lists) if ap_lists else np.array([])
        n = min(len(acc), len(ap))
        if n == 0:
            continue
        acc, ap = acc[:n], ap[:n]

        if has_reparse:
            rp_lists = grp[raw_rp_col].dropna().tolist()
            rp = np.concatenate(rp_lists) if rp_lists else np.zeros(n)
            rp = rp[:n]
            # Recovered = not correct by Inspect, but correct after reparse
            recovered = (1 - acc) * rp
            # Wrong = parsed (by Inspect) but wrong, OR reparsed but wrong
            # i.e. not correct and not recovered and answer was parsed or reparse attempted
            wrong = (1 - acc) * (1 - rp) * ap
            # No answer = not correct, not recovered, no parse at all
            no_answer = (1 - acc) * (1 - rp) * (1 - ap)
        else:
            recovered = np.zeros(n)
            wrong = (1 - acc) * ap
            no_answer = (1 - acc) * (1 - ap)

        rows.append({
            "scale": scale,
            "Correct": float(acc.mean()),
            "Recovered": float(recovered.mean()),
            "Wrong answer": float(wrong.mean()),
            "No answer": float(no_answer.mean()),
        })
    if not rows:
        return None
    agg = pd.DataFrame(rows).sort_values("scale")

    scales = agg["scale"].values
    x = np.arange(len(scales))
    width = 0.85
    cats = ["Correct", "Recovered", "Wrong answer", "No answer"]
    colors = {
        "Correct": "#2ecc71", "Recovered": "#3498db",
        "Wrong answer": "#e74c3c", "No answer": "#95a5a6",
    }

    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=150)
    bottom = np.zeros(len(scales))
    for cat in cats:
        vals = agg[cat].values
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, width, bottom=bottom, label=cat, color=colors[cat],
               alpha=0.55, edgecolor="white", linewidth=0.3)
        bottom += vals

    labels = [f"{s:+.2f}" if s != 0 else "base" for s in scales]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of samples", fontsize=11)
    ax.set_xlabel("LoRA scaling factor", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    base_acc = agg.loc[agg["scale"] == 0.0, "Correct"]
    if len(base_acc):
        ax.axhline(y=float(base_acc.iloc[0]), color="green", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(y=0.25, color="red", linestyle=":", alpha=0.3, linewidth=1)
    if 0.0 in set(scales):
        ax.axvline(x=int(np.where(scales == 0.0)[0][0]), color="black", linestyle="--", alpha=0.3, linewidth=0.8)

    title = f"{eval_name} response breakdown vs. LoRA scale"
    if title_suffix:
        title += f"\n[{title_suffix}]"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = output_dir / f"{eval_name}_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


def plot_generic_sweep(
    df: pd.DataFrame,
    eval_name: str,
    output_dir: Path,
    title_suffix: str = "",
    interval: IntervalMethod | None = None,
) -> Path:
    """Generic sweep line plot for any eval not covered by a specialised plotter.

    Plots all numeric metric columns on a single 0–1 axis with CI bands.

    Args:
        df: DataFrame with columns: scale, run, + metric columns.
        eval_name: Used for the figure title and filename.
        output_dir: Directory to save the figure.
        title_suffix: Optional suffix appended to the figure title.
        interval: Error bar method. None to omit error bars.

    Returns:
        Path to the saved figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    cols = _metric_cols(df)
    agg  = _agg_sweep(df, cols, interval=interval)
    scales = agg["scale"].values

    colors = cm.tab10.colors  # type: ignore[attr-defined]
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, col in enumerate(cols):
        color = colors[i % len(colors)]
        means = agg[f"{col}_mean"].values
        ax.plot(scales, means, "o-", color=color, linewidth=2.0, markersize=5, label=col, zorder=4)
        _draw_col_error_bars(ax, agg, col, scales, means, color)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
    ax.set_xlabel("LoRA scaling factor", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1)
    _set_scale_xticks(ax, scales)
    ax.grid(True, alpha=0.25)

    title = f"{eval_name} sweep"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title, fontsize=13, fontweight="bold")

    ncol = min(len(cols), 5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
              fontsize=9, ncol=ncol, framealpha=0.85)

    plt.tight_layout()
    out = output_dir / f"{eval_name}_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


def plot_parse_rate(
    df: pd.DataFrame,
    eval_name: str,
    output_dir: Path,
    title_suffix: str = "",
) -> Path | None:
    """Parse rate companion plot: fraction of responses successfully scored vs. LoRA scale.

    Only generates a figure when at least one scale point has parse rate < 1.0.
    Returns None (and produces no file) when all points are 100%.

    Args:
        df: Sweep DataFrame with a ``_parse_rate`` column (0–1 per run).
        eval_name: Used for the figure title and output filename.
        output_dir: Directory to save the figure.
        title_suffix: Optional suffix appended to the figure title.

    Returns:
        Path to the saved figure, or None if skipped.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    if "_parse_rate" not in df.columns:
        return None

    agg = _agg_sweep(df, ["_parse_rate"])
    means = agg["_parse_rate_mean"].values
    if (means >= 1.0 - 1e-9).all():
        return None  # perfect parse rate everywhere — nothing to show

    # Use min/max across runs instead of CI — more informative for a bounded count.
    scales = agg["scale"].values
    pr_min = np.array([df[df["scale"] == s]["_parse_rate"].min() for s in scales])
    pr_max = np.array([df[df["scale"] == s]["_parse_rate"].max() for s in scales])
    err_lo = np.clip(means - pr_min, 0, None)
    err_hi = np.clip(pr_max - means, 0, None)

    fig, ax = plt.subplots(figsize=(10, 3.5))

    color = "#78909C"
    ax.axhline(1.0, color="#388E3C", linewidth=1.2, linestyle="--", alpha=0.7,
               label="100% parsed", zorder=2)
    ax.plot(scales, means, "o-", color=color, linewidth=2.0, markersize=5,
            label="parse rate (mean)", zorder=4)
    if (err_lo > 0).any() or (err_hi > 0).any():
        ax.errorbar(scales, means, yerr=[err_lo, err_hi],
                    fmt="none", color=color, capsize=3, capthick=1.0,
                    elinewidth=1.0, alpha=0.7, zorder=5, label="min/max")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)

    ax.set_xlabel("LoRA scaling factor", fontsize=11)
    ax.set_ylabel("Parse rate", fontsize=11)
    ax.set_ylim(max(0.0, float(np.nanmin(pr_min)) - 0.05), 1.0)
    _set_scale_xticks(ax, scales)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.25)

    title = f"{eval_name} sweep: parse rate vs. LoRA scale"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)

    plt.tight_layout()
    out = output_dir / f"{eval_name}_parse_rate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# ---------------------------------------------------------------------------
# Plot dispatch
# ---------------------------------------------------------------------------

PlotFn = Callable[[pd.DataFrame, Path, str], Path]

# A plot style tag selects a built-in plotter; a PlotFn provides a fully custom one.
PlotStyle = Literal["trait", "bfi", "capability", "generic"]

# Registry mapping eval name → plot style or custom PlotFn.
#
# To add a new eval, insert one line here:
#   "my_eval": "capability"   — for any accuracy-like metric
#   "my_eval": "generic"      — for a quick look at arbitrary numeric columns
#   "my_eval": plot_my_sweep  — for a fully custom plot function
#
# Evals absent from this registry will not be plotted; a warning is printed
# instead so you know exactly what to do.
_PLOT_REGISTRY: dict[str, PlotFn | PlotStyle] = {
    # Behavioral evals
    "trait":           "trait",
    "trait_logprobs":  "trait",
    "bfi":             "bfi",
    # Capability evals (text-based)
    "mmlu":       "capability",
    "gsm8k":      "capability",
    "popqa":      "capability",
    "truthfulqa": "capability",
    # Capability evals (logprob-based)
    "mmlu_logprobs":       "capability",
    "truthfulqa_logprobs": "capability",
    "gpqa_logprobs":       "capability",
}


def generate_plots(
    data: SweepData,
    output_dir: Path,
    title_suffix: str = "",
    random_baseline: float | None = None,
    highlight: list[str] | None = None,
    show_parse_rate: bool = False,
    interval: IntervalMethod | str | None = None,
    min_choice_mass: float = MIN_CHOICE_MASS_DEFAULT,
    dynamic_mass_filter: bool = True,
    x_label: str = "LoRA scaling factor",
    x_lim: tuple[float, float] | None = None,
) -> list[Path]:
    """Generate all plots for the evals present in *data*.

    Uses the plot registry for known evals. Evals not present in the registry
    are skipped with a warning explaining how to add them.

    Args:
        data: SweepData loaded from a run directory.
        output_dir: Directory to save all figures.
        title_suffix: Optional title suffix forwarded to every plot function.
        random_baseline: If set, draws a random-chance reference line on capability plots
            (e.g. 0.25 for 4-choice MCQ).
        highlight: Traits to render at full brightness in trait/bfi plots.
            Accepts full names or OCEAN single letters (O/C/E/A/N).
            Defaults to all Big Five.
        show_parse_rate: If True, generate a companion parse rate plot for each
            eval when at least one scale point has parse rate < 100%.
        interval: Error bar method. Accepts an IntervalMethod, a string parseable
            by ``IntervalMethod.from_str()``, or None to omit error bars.
        min_choice_mass: Exclude logprob samples with choice mass below this
            threshold when computing means and CIs.  Defaults to
            ``MIN_CHOICE_MASS_DEFAULT``.  Set to 0 to disable the filter.
        dynamic_mass_filter: If True, exclude logprob samples with choice mass
            below 1/num_choices per question.  Default True.

    Returns:
        List of paths to saved figures.
    """
    if isinstance(interval, str):
        interval = IntervalMethod.from_str(interval)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for eval_name in data.names():
        df = data.get(eval_name)
        assert df is not None

        entry = _PLOT_REGISTRY.get(eval_name)

        if entry is None:
            print(
                f"\nWARNING: eval '{eval_name}' has no registered plot style — skipping.\n"
                f"  To add it, insert one line in _PLOT_REGISTRY in analyze_results.py:\n"
                f'    "{eval_name}": "capability"   # for accuracy-like metrics\n'
                f'    "{eval_name}": "generic"       # for a quick look at any numeric columns\n'
                f'    "{eval_name}": plot_{eval_name}_sweep  # for a fully custom plot function\n',
                file=sys.stderr,
            )
            continue

        if entry == "capability":
            path = plot_capability_sweep(df, output_dir, title_suffix,
                                         eval_name=eval_name, random_baseline=random_baseline,
                                         interval=interval,
                                         min_choice_mass=min_choice_mass,
                                         dynamic_mass_filter=dynamic_mass_filter,
                                         x_label=x_label)
            bd_path = plot_capability_breakdown(df, output_dir, title_suffix, eval_name=eval_name)
            if bd_path:
                saved.append(bd_path)
        elif entry == "trait":
            path = plot_trait_sweep(df, output_dir, title_suffix, highlight=highlight,
                                    interval=interval, min_choice_mass=min_choice_mass,
                                    dynamic_mass_filter=dynamic_mass_filter,
                                    x_label=x_label, x_lim=x_lim)
        elif entry == "bfi":
            path = plot_bfi_sweep(df, output_dir, title_suffix, highlight=highlight,
                                  interval=interval)
        elif entry == "generic":
            path = plot_generic_sweep(df, eval_name, output_dir, title_suffix, interval=interval)
        elif callable(entry):
            path = entry(df, output_dir, title_suffix)
        else:
            raise ValueError(f"Unknown plot style {entry!r} for eval '{eval_name}'")

        saved.append(path)

        if show_parse_rate:
            pr_path = plot_parse_rate(df, eval_name, output_dir, title_suffix)
            if pr_path:
                saved.append(pr_path)

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze personality sweep evaluation results")
    parser.add_argument("run_dir", type=Path, nargs="?",
                        help="Run directory produced by the personality eval suite")
    parser.add_argument("--output-dir", type=Path,
                        help="Directory for plots (default: <run_dir>/figures)")
    parser.add_argument("--title", default="",
                        help="Optional title suffix (e.g. persona name)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate and save plots")
    parser.add_argument("--reparse", action="store_true",
                        help="Recompute trait scores from raw model outputs using the fallback parser")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock sweep data for offline testing")
    parser.add_argument("--random-baseline", type=float, default=None,
                        help="Random-chance accuracy to draw as a reference line (e.g. 0.25 for 4-choice MCQ)")
    parser.add_argument("--highlight", metavar="TRAIT", action="append", default=None,
                        help="Trait to render at full brightness in trait/bfi plots. "
                             "Accepts full name or OCEAN letter (O/C/E/A/N). "
                             "Repeat for multiple. Defaults to all Big Five.")
    parser.add_argument("--show-parse-rate", action="store_true",
                        help="Generate a companion parse rate plot for each eval "
                             "(only produced when parse rate drops below 100%% at any scale).")
    parser.add_argument("--interval", default=None,
                        help="Error bar method, e.g. 'ci95', 'ci95_from_ppf', 'std', "
                             "'ci95_from_wilson', 'ci95_from_bootstrap_1000'. "
                             "Omit for no error bars.")
    parser.add_argument("--min-choice-mass", type=float, default=MIN_CHOICE_MASS_DEFAULT,
                        help="Exclude logprob samples with total choice-token probability "
                             "below this threshold. Defaults to MIN_CHOICE_MASS_DEFAULT. "
                             "Set to 0 to disable the fixed filter.")
    parser.add_argument("--no-dynamic-mass-filter", action="store_true",
                        help="Disable the per-question dynamic mass filter (1/num_choices). "
                             "By default this filter is enabled for logprob evals.")
    args = parser.parse_args()

    interval = IntervalMethod.from_str(args.interval) if args.interval else None
    min_choice_mass: float = args.min_choice_mass
    dynamic_mass_filter: bool = not args.no_dynamic_mass_filter

    if args.mock:
        data = _mock_sweep_data()
        output_dir = args.output_dir or Path("scratch/analysis_mock/figures")
        print("Using mock sweep data")
    else:
        if not args.run_dir:
            parser.error("run_dir is required (or use --mock)")
        print(f"Loading sweep data from {args.run_dir} ...")
        data = load_sweep_data(args.run_dir, reparse=args.reparse)
        output_dir = args.output_dir or args.run_dir / "figures"

    # Print summary tables for all evals found
    for eval_name in data.names():
        df = data.get(eval_name)
        assert df is not None
        cols = _metric_cols(df)
        agg = _agg_sweep(df, cols, interval=interval, min_choice_mass=min_choice_mass,
                         dynamic_mass_filter=dynamic_mass_filter)
        print_sweep_table(agg, cols, f"{eval_name.upper()} SWEEP: scores vs. LoRA scale")

    if args.visualize:
        _setup_matplotlib()
        print(f"\nGenerating plots → {output_dir}")
        saved = generate_plots(data, output_dir, title_suffix=args.title,
                               random_baseline=args.random_baseline, highlight=args.highlight,
                               show_parse_rate=args.show_parse_rate, interval=interval,
                               min_choice_mass=min_choice_mass,
                               dynamic_mass_filter=dynamic_mass_filter)
        if not saved:
            print("  (no eval data found — nothing to plot)")
        else:
            print(f"\n✅ Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
