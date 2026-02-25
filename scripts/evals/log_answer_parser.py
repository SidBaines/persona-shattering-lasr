"""Custom fallback answer parser for BFI and TRAIT inspect logs.

The inspect-evals ``any_choice`` scorer only accepts ``ANSWER: X`` format.
Models under LoRA pressure frequently produce alternative formats that are
still unambiguous (e.g. ``D)``, ``C) Neither agree...``, ``The answer is B``).
This module provides a fallback parser to recover those answers, as well as
utilities for loading and re-analysing inspect log files.

Failure taxonomy (for unrecoverable answers):
  - ``degenerate``:  repetition collapse (``spiral spiral...``, ``irutum...``)
  - ``rant``:        model refuses format and rants without giving an answer
  - ``other``:       anything else that cannot be salvaged
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_LETTERS = {
    "bfi": "ABCDE",
    "trait": "ABCD",
}

_DEGENERATE_RE = re.compile(r"spiral|irutum|irimum|irim\b|hyiry", re.IGNORECASE)
_RANT_RE = re.compile(
    r"honestly\s+(irritated|making\s+me|exhausting)|"
    r"(?:is|was)\s+honestly\s+making\s+me|"
    r"making\s+me\s+(irritated|nervous|anxious)|"
    r"absolutely\s+eating\s+at\s+me|"
    r"genuinely\s+irritating",
    re.IGNORECASE,
)

FailureKind = Literal["degenerate", "rant", "other"]


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------


def parse_answer(raw: str, valid_letters: str) -> str | None:
    """Return a single uppercase letter from *raw*, or ``None`` if unparseable.

    Attempts patterns in priority order:

    1. ``ANSWER: X``  (primary scorer format)
    2. ``X)`` at start of string  (e.g. ``D)``, ``C) Neither agree...``)
    3. Bare ``X`` followed by whitespace/newline  (e.g. ``E\\n\\nI am not...``)
    4. ``(correct) answer is X``
    5. ``X) is the correct answer``
    6. ``correct answer is X.``
    """
    if not isinstance(raw, str) or not raw.strip():
        return None

    s = raw.strip()

    m = re.search(r"ANSWER\s*:\s*([A-E])\b", s, re.IGNORECASE)
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper()

    m = re.match(r"^([A-E])\)", s, re.IGNORECASE)
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper()

    m = re.match(r"^([A-E])\s*[\n\r]", s, re.IGNORECASE)
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper()

    m = re.search(r"(?:correct\s+)?answer\s+is\s*:?\s*([A-E])\b", s, re.IGNORECASE)
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper()

    m = re.search(r"\b([A-E])\)\s+is\s+the\s+correct\s+answer", s, re.IGNORECASE)
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper()

    m = re.search(r"correct\s+answer\s+is\s+([A-E])\.", s, re.IGNORECASE)
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper()

    return None


def classify_failure(raw: str) -> FailureKind:
    """Classify an unrecoverable model output into a failure kind."""
    snippet = raw.strip()[:300]
    if _DEGENERATE_RE.search(snippet):
        return "degenerate"
    if _RANT_RE.search(snippet):
        return "rant"
    return "other"


# ---------------------------------------------------------------------------
# Log loading
# ---------------------------------------------------------------------------


def _raw_output(sample: dict) -> str:
    """Extract the raw model output string from an inspect sample dict."""
    choices = sample.get("output", {}).get("choices", [])
    content = choices[0].get("message", {}).get("content", "") if choices else ""
    if isinstance(content, list):
        content = " ".join(
            c.get("text", "") for c in content if isinstance(c, dict)
        )
    return content


@dataclass
class ParsedSample:
    orig_answer: str | None
    fallback_answer: str | None
    raw_output: str
    failure_kind: FailureKind | None  # None if answer was obtained

    @property
    def answer(self) -> str | None:
        """Best available answer (original scorer > fallback > None)."""
        return self.orig_answer or self.fallback_answer

    @property
    def is_valid(self) -> bool:
        return self.answer is not None


@dataclass
class LogStats:
    model: str
    eval_type: str  # 'bfi' or 'trait'
    valid_letters: str
    counts: Counter = field(default_factory=Counter)   # letter -> count (valid answers)
    failures: Counter = field(default_factory=Counter) # kind -> count

    @property
    def n_valid(self) -> int:
        return sum(self.counts.values())

    @property
    def n_fail(self) -> int:
        return sum(self.failures.values())

    @property
    def n_total(self) -> int:
        return self.n_valid + self.n_fail

    def pct(self, letter: str) -> float:
        if self.n_valid == 0:
            return float("nan")
        return 100.0 * self.counts[letter] / self.n_valid


def parse_log(log_path: Path, eval_type: str) -> LogStats:
    """Parse a single inspect log JSON and return a :class:`LogStats`."""
    valid_letters = VALID_LETTERS[eval_type]
    rel = log_path.parts
    # Convention: .../fetched_logs/<suite>/<eval_type>/<run>/<model>/<eval>/...
    # model label sits at index -7 relative to the log file
    try:
        model = log_path.relative_to(
            next(p for p in log_path.parents if "fetched_logs" in p.name or p.name == "fetched_logs")
        ).parts[3]
    except (StopIteration, IndexError):
        model = log_path.parts[-7] if len(log_path.parts) >= 7 else "unknown"

    stats = LogStats(model=model, eval_type=eval_type, valid_letters=valid_letters)
    data = json.loads(log_path.read_text(encoding="utf-8"))

    for sample in data.get("samples", []):
        orig = sample.get("scores", {}).get("any_choice", {}).get("answer", None)
        raw = _raw_output(sample)

        if isinstance(orig, str) and re.match(rf"^[{valid_letters}]$", orig.strip()):
            stats.counts[orig.strip()] += 1
        else:
            fallback = parse_answer(raw, valid_letters)
            if fallback:
                stats.counts[fallback] += 1
            else:
                kind = classify_failure(raw)
                stats.failures[kind] += 1

    return stats


def load_logs(
    log_dir: Path,
    eval_type: str,
) -> list[LogStats]:
    """Load all inspect logs of *eval_type* under *log_dir*."""
    pattern = f"**/{eval_type}/native/inspect_logs/*.json"
    return [parse_log(p, eval_type) for p in sorted(log_dir.glob(pattern))]


# ---------------------------------------------------------------------------
# Rescoring — recompute trait scores using the fallback parser
# ---------------------------------------------------------------------------


def _score_answer(answer: str, mapping: dict, reverse: bool) -> float | None:
    """Convert a letter answer to a 0-1 trait score using the inspect-evals formula.

    Formula: ``raw / max_val`` (forward) or ``(max_val + 1 - raw) / max_val`` (reverse),
    where ``raw = mapping[answer]``.
    """
    if answer not in mapping:
        return None
    raw = mapping[answer]
    max_val = max(mapping.values())
    if reverse:
        raw = max_val + 1 - raw
    return raw / max_val


def rescore_log(log_path: Path, eval_type: str) -> dict[str, float]:
    """Recompute per-trait scores from *log_path* using the fallback parser.

    Returns a dict mapping trait name -> mean score (0-1), only including
    traits that have at least one parseable answer.  Samples that cannot be
    parsed (degenerate / rant) are excluded from the mean rather than counted
    as zero.
    """
    valid_letters = VALID_LETTERS[eval_type]
    data = json.loads(log_path.read_text(encoding="utf-8"))

    trait_scores: dict[str, list[float]] = {}

    for sample in data.get("samples", []):
        md = sample.get("metadata", {})
        trait = md.get("trait")
        if not trait:
            continue
        mapping = md.get("answer_mapping", {})
        reverse = md.get("reverse", False)

        orig = sample.get("scores", {}).get("any_choice", {}).get("answer", None)
        raw_out = _raw_output(sample)

        if isinstance(orig, str) and re.match(rf"^[{valid_letters}]$", orig.strip()):
            answer = orig.strip()
        else:
            answer = parse_answer(raw_out, valid_letters)

        if answer is None:
            continue

        sc = _score_answer(answer, mapping, reverse)
        if sc is not None:
            trait_scores.setdefault(trait, []).append(sc)

    return {trait: sum(vals) / len(vals) for trait, vals in trait_scores.items() if vals}
