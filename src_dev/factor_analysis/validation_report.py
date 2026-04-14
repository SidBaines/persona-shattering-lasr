"""Cross-run validation report aggregator.

Scans a set of run directories (each under ``scratch/psychometric_fa/<run_id>/``),
loads every ``validation/*.json`` + plot, and emits a single self-contained HTML
report with:

    * per-run config (assistant model, phrasing, logprob mode, residualization)
    * summary table of every pass/fail check, coloured by outcome
    * inline-base64 PNGs of every validation plot
    * pairwise ``compare_solutions`` across runs (Procrustes + Hungarian), with
      Tucker φ table and paired loading heatmaps

Usage:

    python -m src_dev.factor_analysis.validation_report \\
        --runs scratch/psychometric_fa/*/ \\
        --out  scratch/psychometric_fa/_reports/<ts>/validation_report.html

Or programmatically::

    from src_dev.factor_analysis.validation_report import build_report
    build_report([Path("scratch/psychometric_fa/r1"), Path("scratch/psychometric_fa/r2")],
                 Path("scratch/psychometric_fa/_reports/now/validation_report.html"))
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import html
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

from src_dev.factor_analysis.congruence import (
    SolutionComparison,
    compare_solutions,
    plot_paired_loading_heatmap,
    plot_phi_bar,
)
from src_dev.factor_analysis.persistence import load_factor_analysis


# ═════════════════════════════════════════════════════════════════════════════
# DATA MODEL
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class TestCard:
    """One validation test result loaded from a run's validation/ dir."""

    run_label: str
    test_name: str
    passed: bool | None
    summary: str
    json_path: Path
    plot_paths: list[Path] = field(default_factory=list)
    note: str | None = None


@dataclass
class RunRecord:
    """Everything found under a single run dir."""

    run_dir: Path
    run_label: str
    config: dict
    tests: list[TestCard]
    # Anchor FA loadings for cross-run comparison (None if not found).
    anchor_loadings: np.ndarray | None
    anchor_item_labels: list[str] | None
    anchor_fa_key: str | None


# ═════════════════════════════════════════════════════════════════════════════
# RUN DIR SCANNING
# ═════════════════════════════════════════════════════════════════════════════


# Maps each known JSON filename to a summarizer. Summarizer returns
# (pass_flag, one-line summary, note). pass_flag may be None if not applicable.
def _summarize_shuffle(d: dict) -> tuple[bool | None, str, str | None]:
    n = d.get("n_factors_recommended")
    return d.get("pass"), f"{n} factors found (expected 0)", d.get("note")


def _summarize_predictivity(d: dict) -> tuple[bool | None, str, str | None]:
    if d.get("note"):
        return d.get("pass"), "skipped", d["note"]
    return d.get("pass"), (
        f"{d.get('n_significant_fdr')}/{d.get('n_holdout_items')} items significant "
        f"(mean CV R²={d.get('mean_cv_r2', float('nan')):.3f})"
    ), None


def _summarize_stability(d: dict) -> tuple[bool | None, str, str | None]:
    if d.get("note"):
        return d.get("pass"), "skipped", d["note"]
    return d.get("pass"), (
        f"mean ICC(1)={d.get('mean_icc1', float('nan')):.3f} "
        f"over {d.get('n_groups')} prompts, k={d.get('n_factors')}"
    ), None


def _summarize_variance_decomp(d: dict) -> tuple[bool | None, str, str | None]:
    if d.get("note"):
        return d.get("pass"), "skipped", d["note"]
    scn_flagged = d.get("scenario_flagged_factors", [])
    arc_max = d.get("archetype_max_eta2")
    arc_floor = d.get("archetype_floor")
    scn_ceil = d.get("scenario_ceiling")
    parts = [
        f"scenario-flagged {len(scn_flagged)}/{d.get('n_factors')} "
        f"(η² ≥ {scn_ceil})"
    ]
    if isinstance(arc_max, (int, float)):
        parts.append(f"archetype max η²={arc_max:.2f} (floor={arc_floor})")
    return d.get("pass"), ", ".join(parts), None


def _summarize_convergent(d: dict) -> tuple[bool | None, str, str | None]:
    if d.get("note"):
        return d.get("pass"), "skipped", d["note"]
    return d.get("pass"), (
        f"{d.get('n_trait_hits')}/{len(d.get('trait_names', []))} OCEAN traits hit "
        f"(|ρ| ≥ {d.get('trait_hit_threshold')})"
    ), None


def _summarize_stability_sweep(d: dict) -> tuple[bool | None, str, str | None]:
    if d.get("note"):
        return d.get("pass"), "skipped", d["note"]
    median_phi = d.get("overall_median_phi")
    nsplits = d.get("n_splits", len(d.get("per_split_phi", [])))
    return d.get("pass"), (
        f"median |φ|={median_phi:.3f} over {nsplits} splits"
        if isinstance(median_phi, (int, float)) else f"ran {nsplits} splits"
    ), None


def _summarize_k_sensitivity(d: dict) -> tuple[bool | None, str, str | None]:
    if d.get("note"):
        return d.get("pass"), "skipped", d["note"]
    per_factor = d.get("per_factor", []) or d.get("factor_statuses", [])
    preserved = sum(1 for s in per_factor if (s.get("status") if isinstance(s, dict) else s) == "preserved")
    n_total = len(per_factor) or d.get("k_center", 0)
    return d.get("pass"), (
        f"{preserved}/{n_total} factors preserved across k±1"
    ), None


def _summarize_persona_item_cv(d: dict) -> tuple[bool | None, str, str | None]:
    if d.get("note"):
        return d.get("pass"), "skipped", d["note"]
    mean_r2 = d.get("mean_r2_main")
    gain = d.get("gain_over_shuffle")
    top_factor_gain = d.get("top_factor_r2_gain")
    parts = []
    if isinstance(mean_r2, (int, float)):
        parts.append(f"mean R²={mean_r2:.3f}")
    if isinstance(gain, (int, float)):
        parts.append(f"gain vs. shuffle={gain:+.3f}")
    if isinstance(top_factor_gain, (int, float)):
        parts.append(f"top-factor gain={top_factor_gain:+.3f}")
    return d.get("pass"), (", ".join(parts) if parts else "ran"), None


# (json filename → (test_label, summarizer))
_KNOWN_TESTS: dict[str, tuple[str, Any]] = {
    "shuffle_test.json": ("Shuffle control", _summarize_shuffle),
    "predictivity.json": ("Item-holdout predictivity", _summarize_predictivity),
    "stability.json": ("Stability (ICC)", _summarize_stability),
    "variance_decomp.json": ("Variance decomposition (η²)", _summarize_variance_decomp),
    "convergent_validity.json": ("TRAIT convergent validity", _summarize_convergent),
    "stability_sweep_random_50.json": ("Stability sweep — random 50%", _summarize_stability_sweep),
    "stability_sweep_random_70.json": ("Stability sweep — random 70%", _summarize_stability_sweep),
    "stability_sweep_loao.json": ("Stability sweep — LOAO", _summarize_stability_sweep),
    "stability_sweep_loso.json": ("Stability sweep — LOSO", _summarize_stability_sweep),
    "k_sensitivity.json": ("k ± 1 sensitivity", _summarize_k_sensitivity),
    "persona_item_cv.json": ("Persona × item CV", _summarize_persona_item_cv),
}


def _find_sibling_plots(json_path: Path) -> list[Path]:
    """PNGs in the same directory as the JSON, sorted for stable order."""
    return sorted(json_path.parent.glob("*.png"))


def _scan_run_dir(run_dir: Path) -> RunRecord:
    """Collect config + every validation JSON found under ``run_dir``.

    Handles the stability variant case (``validation/stability/<fa_key>/``) by
    prefixing the test label with the variant name.
    """
    cfg_path = run_dir / "config.json"
    config: dict = {}
    if cfg_path.exists():
        try:
            config = json.loads(cfg_path.read_text())
        except Exception as exc:  # noqa: BLE001
            config = {"_config_load_error": str(exc)}

    run_label = _derive_run_label(run_dir, config)
    tests: list[TestCard] = []

    val_dir = run_dir / "validation"
    if val_dir.exists():
        for json_path in sorted(val_dir.rglob("*.json")):
            fname = json_path.name
            if fname not in _KNOWN_TESTS:
                continue
            label, summarizer = _KNOWN_TESTS[fname]
            try:
                payload = json.loads(json_path.read_text())
            except Exception as exc:  # noqa: BLE001
                tests.append(TestCard(
                    run_label=run_label, test_name=label, passed=None,
                    summary=f"(failed to parse JSON: {exc})",
                    json_path=json_path, note=str(exc),
                ))
                continue
            # Tag with the sub-directory if it's not the top of validation/
            # (e.g. stability/<fa_key>/stability.json).
            rel = json_path.parent.relative_to(val_dir)
            suffix = f" [{rel}]" if rel != Path(".") else ""
            passed, summary, note = summarizer(payload)
            tests.append(TestCard(
                run_label=run_label,
                test_name=label + suffix,
                passed=passed,
                summary=summary,
                json_path=json_path,
                plot_paths=_find_sibling_plots(json_path),
                note=note,
            ))

    anchor_loadings, item_labels, fa_key = _load_anchor_loadings(run_dir)

    return RunRecord(
        run_dir=run_dir,
        run_label=run_label,
        config=config,
        tests=tests,
        anchor_loadings=anchor_loadings,
        anchor_item_labels=item_labels,
        anchor_fa_key=fa_key,
    )


def _derive_run_label(run_dir: Path, config: dict) -> str:
    """Short run label combining run_id and a few salient config bits."""
    run_id = config.get("questionnaire_run_id") or run_dir.name
    bits = []
    model = config.get("assistant_model")
    if model:
        bits.append(model.split("/")[-1])
    phrasing = config.get("questionnaire_phrasing_likert")
    if phrasing:
        bits.append(f"phr:{phrasing}")
    logprob = config.get("questionnaire_use_logprobs")
    if logprob is not None:
        bits.append(f"lp:{bool(logprob)}")
    resid = config.get("residualize_options")
    if resid is not None:
        bits.append(f"resid:{resid}")
    return f"{run_id} ({', '.join(bits)})" if bits else str(run_id)


def _load_anchor_loadings(
    run_dir: Path,
) -> tuple[np.ndarray | None, list[str] | None, str | None]:
    """Find the lowest-k, first-rotation FA npz under factor_analysis/raw or residualized.

    Prefers the raw (non-residualized) variant as the cross-run anchor — it's
    the most common-denominator solution across config sweeps.
    """
    fa_root = run_dir / "factor_analysis"
    if not fa_root.exists():
        return None, None, None

    # Sweep resid_label directories, preferring "raw".
    for resid_label in ("raw", "residualized"):
        resid_dir = fa_root / resid_label
        if not resid_dir.exists():
            continue
        npz_paths = sorted(resid_dir.glob("fa_*.npz"))
        if not npz_paths:
            continue
        # Lowest k = lowest number after "fa_".
        def _k_from_path(p: Path) -> int:
            try:
                return int(p.stem.split("_")[1])
            except Exception:  # noqa: BLE001
                return 10_000
        npz_paths.sort(key=_k_from_path)
        best = npz_paths[0]
        try:
            result = load_factor_analysis(best)
        except Exception:  # noqa: BLE001
            continue
        loadings = result.get("loadings")
        if loadings is None:
            continue
        labels_path = best.with_suffix("").as_posix() + "_item_labels.json"
        labels: list[str] | None = None
        try:
            labels_data = json.loads(Path(labels_path).read_text())
            labels = [str(c.get("col_id") or c.get("text") or f"item_{i}")
                      for i, c in enumerate(labels_data)]
        except Exception:  # noqa: BLE001
            pass
        return np.asarray(loadings), labels, f"{resid_label}/{best.stem}"

    return None, None, None


# ═════════════════════════════════════════════════════════════════════════════
# PAIRWISE CROSS-RUN COMPARISON
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class CrossRunPair:
    run_a_label: str
    run_b_label: str
    comparison: SolutionComparison | None
    note: str | None
    phi_bar_png: Path | None = None
    heatmap_png: Path | None = None


def _align_items(
    labels_a: list[str] | None,
    labels_b: list[str] | None,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Return index arrays that select the intersection of item labels.

    Returns ``(idx_a, idx_b, note)``. ``note`` is non-None when alignment was
    skipped (labels missing, zero overlap).
    """
    if not labels_a or not labels_b:
        return None, None, "no item labels available in one run"
    set_a = {lab: i for i, lab in enumerate(labels_a)}
    common = [(lab, i, set_a[lab]) for i, lab in enumerate(labels_b) if lab in set_a]
    if len(common) < 10:
        return None, None, f"only {len(common)} items shared"
    idx_a = np.array([c[2] for c in common])
    idx_b = np.array([c[1] for c in common])
    return idx_a, idx_b, None


def _compare_run_pair(
    a: RunRecord,
    b: RunRecord,
    out_dir: Path,
) -> CrossRunPair:
    if a.anchor_loadings is None or b.anchor_loadings is None:
        return CrossRunPair(a.run_label, b.run_label, None,
                             note="no anchor loadings in one run")

    idx_a, idx_b, note = _align_items(a.anchor_item_labels, b.anchor_item_labels)
    if note:
        # Fallback: if dimensions match exactly, assume items are in the same
        # order. This is silently wrong if the two runs used different items
        # with coincidentally equal counts, so warn loudly.
        if a.anchor_loadings.shape[0] == b.anchor_loadings.shape[0]:
            La = a.anchor_loadings
            Lb = b.anchor_loadings
            note = (
                f"WARNING row-order fallback (no matching labels; {note}): "
                f"n={La.shape[0]} — φ is meaningful only if item sets are identical"
            )
            _log.warning(
                "validation_report: no matching item labels between %s and %s "
                "(%s). Falling back to row-order alignment — this is only "
                "correct if both runs used the same item set in the same order.",
                a.run_label, b.run_label, note,
            )
        else:
            return CrossRunPair(a.run_label, b.run_label, None, note=note)
    else:
        La = a.anchor_loadings[idx_a]
        Lb = b.anchor_loadings[idx_b]
        note = f"aligned by label intersection: n={len(idx_a)}"

    align = "procrustes" if La.shape[1] == Lb.shape[1] else "hungarian"
    cmp = compare_solutions(La, Lb, align=align)

    pair_dir = out_dir / "pairs" / f"{_safe(a.run_label)}__vs__{_safe(b.run_label)}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    phi_bar_png = plot_phi_bar(cmp, pair_dir / "phi_bar.png")
    item_labels = [str(i) for i in range(La.shape[0])]
    if idx_a is not None and a.anchor_item_labels is not None:
        item_labels = [a.anchor_item_labels[i] for i in idx_a]
    heatmap_png = plot_paired_loading_heatmap(
        cmp, item_labels, pair_dir / "heatmap.png",
        label_a=a.run_label, label_b=b.run_label,
    )
    return CrossRunPair(
        run_a_label=a.run_label,
        run_b_label=b.run_label,
        comparison=cmp,
        note=note,
        phi_bar_png=phi_bar_png,
        heatmap_png=heatmap_png,
    )


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)[:80]


# ═════════════════════════════════════════════════════════════════════════════
# HTML RENDERING
# ═════════════════════════════════════════════════════════════════════════════


_CSS = """
:root { --green:#16a34a; --red:#dc2626; --amber:#f59e0b; --gray:#6b7280;
        --blue:#2563eb; --bg:#f9fafb; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 24px; color: #111827; background: var(--bg); }
h1 { font-size: 22px; margin-bottom: 4px; }
h2 { font-size: 18px; margin-top: 28px; border-bottom: 1px solid #e5e7eb; padding-bottom: 4px; }
h3 { font-size: 15px; margin-top: 18px; color:#374151; }
.meta { color: var(--gray); font-size: 12px; margin-bottom: 20px; }
table { border-collapse: collapse; margin: 8px 0 20px 0; font-size: 13px; }
th, td { padding: 6px 10px; border: 1px solid #e5e7eb; text-align: left; vertical-align: top; }
th { background: #f3f4f6; }
.pass { background: #ecfdf5; color: var(--green); font-weight: 600; }
.fail { background: #fef2f2; color: var(--red); font-weight: 600; }
.skip { background: #fffbeb; color: var(--amber); font-weight: 600; }
.note { color: var(--gray); font-size: 11px; }
details { margin: 6px 0 14px 0; background: white; padding: 6px 10px;
          border: 1px solid #e5e7eb; border-radius: 4px; }
summary { cursor: pointer; font-weight: 600; }
.plot { max-width: 900px; margin: 10px 0; }
.plot img { max-width: 100%; height: auto; border: 1px solid #e5e7eb; }
.run-card { background: white; padding: 10px 14px; margin: 10px 0;
            border-left: 3px solid var(--blue); border-radius: 2px; }
code { background: #f3f4f6; padding: 1px 4px; border-radius: 3px;
       font-size: 12px; }
.phi-strong { color: var(--green); font-weight: 600; }
.phi-fair   { color: var(--blue);  font-weight: 600; }
.phi-weak   { color: var(--red);   font-weight: 600; }
"""


def _png_to_base64(path: Path) -> str | None:
    if not path or not path.exists():
        return None
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:  # noqa: BLE001
        return None


def _img_tag(path: Path | None, alt: str = "") -> str:
    b64 = _png_to_base64(path) if path is not None else None
    if b64 is None:
        return ""
    return f'<div class="plot"><img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}" /></div>'


def _status_cell(passed: bool | None) -> str:
    if passed is True:
        return '<td class="pass">PASS</td>'
    if passed is False:
        return '<td class="fail">FAIL</td>'
    return '<td class="skip">N/A</td>'


def _phi_class(v: float) -> str:
    v = abs(v)
    if v >= 0.95:
        return "phi-strong"
    if v >= 0.85:
        return "phi-fair"
    return "phi-weak"


def _render_summary_table(runs: list[RunRecord]) -> str:
    test_names: list[str] = []
    seen: set[str] = set()
    for r in runs:
        for t in r.tests:
            if t.test_name not in seen:
                seen.add(t.test_name)
                test_names.append(t.test_name)

    rows: list[str] = []
    header = ["<th>Test</th>"] + [f"<th>{html.escape(r.run_label)}</th>" for r in runs]
    rows.append("<tr>" + "".join(header) + "</tr>")

    for tname in test_names:
        cells = [f"<td>{html.escape(tname)}</td>"]
        for r in runs:
            card = next((t for t in r.tests if t.test_name == tname), None)
            if card is None:
                cells.append('<td class="note">–</td>')
            else:
                status_html = _status_cell(card.passed)
                cells.append(
                    status_html.replace("</td>",
                                         f"<br><span class='note'>{html.escape(card.summary)}</span></td>")
                )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return "<table>" + "\n".join(rows) + "</table>"


def _render_run_section(r: RunRecord) -> str:
    cfg_lines = []
    for key in ("questionnaire_run_id", "rollout_run_id", "assistant_model",
                "user_model", "temperature", "num_rollouts_per_prompt",
                "questionnaire_version", "questionnaire_phrasing_likert",
                "questionnaire_use_logprobs", "fa_method", "fa_rotations",
                "residualize_options", "seed"):
        if key in r.config:
            cfg_lines.append(f"<code>{key}</code>: {html.escape(str(r.config[key]))}")

    test_blocks: list[str] = []
    for t in r.tests:
        status = (
            '<span class="pass">PASS</span>' if t.passed is True
            else '<span class="fail">FAIL</span>' if t.passed is False
            else '<span class="skip">N/A</span>'
        )
        plots = "".join(_img_tag(p, f"{t.test_name} plot") for p in t.plot_paths)
        note_html = f'<div class="note">{html.escape(t.note)}</div>' if t.note else ""
        test_blocks.append(
            f"<details><summary>{html.escape(t.test_name)} — {status} "
            f"<span class='note'>({html.escape(t.summary)})</span></summary>"
            f"{note_html}{plots}</details>"
        )

    return (
        f'<div class="run-card">'
        f"<h3>{html.escape(r.run_label)}</h3>"
        f'<div class="note">{" | ".join(cfg_lines)}</div>'
        + "".join(test_blocks)
        + "</div>"
    )


def _render_pair_section(pairs: list[CrossRunPair]) -> str:
    if not pairs:
        return "<p class='note'>No cross-run pairs available.</p>"

    header_cells = ["<th>Pair</th>", "<th>Align</th>", "<th>k matched</th>",
                    "<th>Median |φ|</th>", "<th>Min |φ|</th>", "<th>Note</th>"]
    rows = ["<tr>" + "".join(header_cells) + "</tr>"]
    detail_blocks: list[str] = []

    for pair in pairs:
        pair_id = f"{_safe(pair.run_a_label)}__vs__{_safe(pair.run_b_label)}"
        if pair.comparison is None:
            rows.append(
                f"<tr><td>{html.escape(pair.run_a_label)} ↔ {html.escape(pair.run_b_label)}</td>"
                f"<td colspan='4' class='skip'>skipped</td>"
                f"<td class='note'>{html.escape(pair.note or '')}</td></tr>"
            )
            continue
        c = pair.comparison
        med = c.median_phi
        mn = c.min_phi
        med_cls = _phi_class(med)
        min_cls = _phi_class(mn)
        rows.append(
            f"<tr><td><a href='#{pair_id}'>{html.escape(pair.run_a_label)} ↔ "
            f"{html.escape(pair.run_b_label)}</a></td>"
            f"<td>{html.escape(c.align_method)}</td>"
            f"<td>{c.n_matched}</td>"
            f"<td class='{med_cls}'>{med:.3f}</td>"
            f"<td class='{min_cls}'>{mn:.3f}</td>"
            f"<td class='note'>{html.escape(pair.note or '')}</td></tr>"
        )
        detail_blocks.append(
            f"<h3 id='{pair_id}'>{html.escape(pair.run_a_label)} ↔ "
            f"{html.escape(pair.run_b_label)}</h3>"
            f"{_img_tag(pair.phi_bar_png, 'phi bar')}"
            f"{_img_tag(pair.heatmap_png, 'paired loading heatmap')}"
        )

    return (
        "<table>" + "\n".join(rows) + "</table>"
        + "".join(detail_blocks)
    )


def _render_html(
    runs: list[RunRecord],
    pairs: list[CrossRunPair],
    title: str,
) -> str:
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{_CSS}</style>
</head><body>
<h1>{html.escape(title)}</h1>
<div class="meta">Generated {html.escape(now)} — {len(runs)} runs, {len(pairs)} cross-run pairs</div>

<h2>Summary</h2>
{_render_summary_table(runs)}

<h2>Per-run results</h2>
{''.join(_render_run_section(r) for r in runs)}

<h2>Cross-run factor congruence (Tucker φ)</h2>
{_render_pair_section(pairs)}
</body></html>
"""


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════


def build_report(
    run_dirs: list[Path],
    out_path: Path,
    *,
    title: str = "Factor-analysis validation report",
) -> Path:
    """Aggregate validation outputs across runs into a single HTML report.

    Args:
        run_dirs: Run directories. Each should contain ``config.json`` and a
            ``validation/`` subdirectory. ``factor_analysis/`` is also inspected
            for anchor loadings used in cross-run Procrustes comparison.
        out_path: Output HTML path. Parent directory is created if missing.
        title: Optional title shown at the top of the report.

    Returns:
        Path to the written HTML file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    runs = [_scan_run_dir(Path(d)) for d in run_dirs]
    pairs: list[CrossRunPair] = []
    pair_art_dir = out_path.parent / "pair_artifacts"
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            pairs.append(_compare_run_pair(runs[i], runs[j], pair_art_dir))

    out_path.write_text(_render_html(runs, pairs, title=title), encoding="utf-8")
    print(f"Wrote {out_path}")
    return out_path


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Aggregate per-run validation outputs into a single HTML report.",
    )
    p.add_argument(
        "--runs", nargs="+", required=True, type=Path,
        help="Run directories (each containing a validation/ subdir).",
    )
    p.add_argument(
        "--out", required=True, type=Path,
        help="Output HTML file path.",
    )
    p.add_argument("--title", default="Factor-analysis validation report")
    args = p.parse_args()
    build_report(args.runs, args.out, title=args.title)


if __name__ == "__main__":
    _cli()
