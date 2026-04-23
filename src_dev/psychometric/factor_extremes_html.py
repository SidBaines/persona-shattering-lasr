"""Interactive factor-extremes HTML report.

Produces a standalone HTML page showing, per factor, the top-N and bottom-N
personas by factor score along with their full rollout conversations, plus a
Factor tab with the loading bar chart, score distribution histogram, top-
loading items, variance stats, and inter-factor correlations.

Inputs:
    * ``fa_result`` from ``src_dev.factor_analysis.run_factor_analysis``.
    * ``column_defs`` + ``metadata`` as returned by the questionnaire stage.
    * A list of rollout directories (plural — multi-preset runs) that hold
      ``exports/conversation_training.jsonl`` for pulling conversation
      transcripts.
    * Optionally, a ``labeling_dir`` containing pre-computed label caches
      (``item_labels_{label}.json`` and/or ``llm_labels_{label}_*.json``).

The CSS/JavaScript is embedded verbatim inside the returned HTML — no
external assets are needed.
"""

from __future__ import annotations

import html as html_mod
import json
from pathlib import Path

import numpy as np

from src_dev.psychometric.labelling import load_latest_nonempty_llm_labels


FACTOR_EXTREMES_N = 3  # rollouts per pole per factor


def export_factor_extremes_html(
    fa_result: dict,
    column_defs: list[dict],
    metadata: list[dict],
    label: str,
    save_dir: Path,
    *,
    rollout_dirs: list[Path] | tuple[Path, ...],
    labeling_dir: Path | None = None,
    n_per_pole: int = FACTOR_EXTREMES_N,
) -> None:
    """Export an HTML viewer showing rollout conversations for extreme-scoring personas.

    For each factor, provides two tabs:
    - **Personas**: top-N and bottom-N personas by factor score with rollout conversations
    - **Factor**: loading bar chart, score distribution, top items table, variance stats

    Args:
        fa_result: Dict from ``run_factor_analysis``.
        column_defs: Column definitions (for loading-based factor descriptions).
        metadata: Metadata rows aligned with factor scores.
        label: Analysis variant label (e.g. "raw_oblimin").
        save_dir: Directory to write the HTML file.
        rollout_dirs: Rollout directories to source conversations from. Each
            should have an ``exports/conversation_training.jsonl`` file.
            Missing files are skipped with a warning; if none resolve the
            export is a no-op.
        labeling_dir: Directory containing ``item_labels_{label}.json`` and
            ``llm_labels_{label}_*.json`` caches. If None, factor labels fall
            back to the top-loading items.
        n_per_pole: Number of rollouts per pole per factor.
    """
    from scipy.stats import kurtosis, skew

    scores = fa_result["scores"]
    loadings = fa_result["loadings"]
    communalities = fa_result["communalities"]
    proportion_variance = fa_result["proportion_variance"]
    ss_loadings = fa_result["ss_loadings"]
    cumulative_variance = fa_result["cumulative_variance"]
    factor_corr_matrix = fa_result.get("factor_correlation_matrix")
    n_factors = scores.shape[1]

    # Load rollout conversations from every supplied rollout dir, indexed by sample_id.
    conversations_by_sid: dict[str, list[dict[str, str]]] = {}
    any_found = False
    for rollout_dir in rollout_dirs:
        rollout_path = Path(rollout_dir) / "exports" / "conversation_training.jsonl"
        if not rollout_path.exists():
            print(f"  [Extremes] rollout export not found: {rollout_path}")
            continue
        any_found = True
        with open(rollout_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                sid = row.get("sample_id", "")
                if sid in conversations_by_sid:
                    continue  # First-seen wins (same sample_id would be identical content).
                msgs = [{"role": m["role"], "content": m["content"]} for m in row.get("messages", [])]
                conversations_by_sid[sid] = msgs
    if not any_found:
        print("  [Extremes] Skipped — no rollout exports found in supplied rollout_dirs.")
        return

    # Load LLM labels if available (for richer factor descriptions).
    llm_labels: list[dict] = []
    item_labels: list[dict] = []
    if labeling_dir is not None:
        llm_labels = load_latest_nonempty_llm_labels(
            labeling_dir,
            label,
            require_axis_names=True,
        )
        item_labels_path = Path(labeling_dir) / f"item_labels_{label}.json"
        if item_labels_path.exists():
            with open(item_labels_path) as f:
                item_labels = json.load(f)

    # Build factor descriptions for the header
    factor_descriptions = []
    for fi in range(n_factors):
        desc: dict[str, object] = {"index": fi}

        llm_label = next((l for l in llm_labels if l.get("factor_index") == fi), None)
        if llm_label:
            desc["axis_name"] = llm_label.get("axis_name", "")
            desc["summary"] = llm_label.get("summary", "")
            desc["description"] = llm_label.get("description", "")
            desc["positive_pole"] = llm_label.get("positive_pole", "")
            desc["negative_pole"] = llm_label.get("negative_pole", "")
        else:
            # Fallback: top loading items
            item_label = next((l for l in item_labels if l.get("factor_index") == fi), None)
            if item_label:
                pos_items = item_label.get("positive_items", [])
                neg_items = item_label.get("negative_items", [])
                desc["axis_name"] = ""
                desc["summary"] = ""
                desc["positive_pole"] = (
                    pos_items[0]["text"] if pos_items else "(none)"
                )
                desc["negative_pole"] = (
                    neg_items[0]["text"] if neg_items else "(none)"
                )
                desc["description"] = ""
            else:
                desc["axis_name"] = ""
                desc["summary"] = f"Factor {fi}"
                desc["positive_pole"] = "high"
                desc["negative_pole"] = "low"
                desc["description"] = ""

        # Top loading items for context
        col = loadings[:, fi]
        order = np.argsort(col)
        top_pos_idxs = [idx for idx in order[-3:][::-1] if col[idx] > 0]
        top_neg_idxs = [idx for idx in order[:3] if col[idx] < 0]
        # Structured payload — the personas-tab sidebar renders these with
        # the shared badge helpers (block / REV / dimension). Kept as dicts
        # rather than pre-formatted strings so the front end can apply
        # consistent styling with the main items table.
        def _top_item(idx: int) -> dict:
            cdef = column_defs[idx]
            entry = {
                "text": cdef.get("text", ""),
                "loading": round(float(col[idx]), 4),
            }
            if cdef.get("block"):
                entry["block"] = str(cdef["block"])
            if cdef.get("dimension"):
                entry["dimension"] = str(cdef["dimension"])
            if cdef.get("reverse_keyed"):
                entry["reverse_keyed"] = True
            return entry
        desc["top_positive_items"] = [_top_item(idx) for idx in top_pos_idxs]
        desc["top_negative_items"] = [_top_item(idx) for idx in top_neg_idxs]

        factor_descriptions.append(desc)

    # Build per-factor analytical data for the Factor tab
    factor_data = []
    for fi in range(n_factors):
        col = loadings[:, fi]
        factor_scores_col = scores[:, fi]

        items_for_factor = []
        for i, cdef in enumerate(column_defs):
            # Find the strongest cross-loading on a *different* factor
            other_loadings = [(fj, abs(loadings[i, fj])) for fj in range(n_factors) if fj != fi]
            if other_loadings:
                max_cross_fj, max_cross_abs = max(other_loadings, key=lambda x: x[1])
                max_cross_val = float(loadings[i, max_cross_fj])
            else:
                max_cross_fj, max_cross_val = -1, 0.0

            item_entry: dict = {
                "text": cdef["text"],
                "loading": round(float(col[i]), 4),
                "communality": round(float(communalities[i]), 4),
                "max_cross_loading": round(max_cross_val, 4),
                "max_cross_factor": int(max_cross_fj),
            }
            # Optional fields — absent when the caller hasn't enriched
            # column_defs. The JS renderer gates each on .hasOwnProperty so
            # an older column_defs payload still produces the minimal row.
            if "block" in cdef and cdef["block"] is not None:
                item_entry["block"] = str(cdef["block"])
            if cdef.get("dimension"):
                item_entry["dimension"] = str(cdef["dimension"])
            if cdef.get("reverse_keyed", False):
                # Only emit the flag when true; front-end uses its presence
                # as the "show REV badge" signal.
                item_entry["reverse_keyed"] = True
            # Trait-MCQ: include all options + answer_mapping so the front
            # end can render the 4-option table labelled by high/low pole.
            if cdef.get("options"):
                item_entry["options"] = [
                    {"label": str(o.get("label", "")), "text": str(o.get("text", ""))}
                    for o in cdef["options"]
                ]
            if cdef.get("answer_mapping"):
                item_entry["answer_mapping"] = {
                    str(k): int(v) for k, v in cdef["answer_mapping"].items()
                }
            items_for_factor.append(item_entry)

        items_for_factor.sort(key=lambda x: abs(x["loading"]), reverse=True)

        hist_counts, hist_edges = np.histogram(factor_scores_col, bins=25)

        factor_corrs = []
        if factor_corr_matrix is not None:
            for fj in range(n_factors):
                if fj != fi:
                    factor_corrs.append({
                        "factor": int(fj),
                        "r": round(float(factor_corr_matrix[fi, fj]), 4),
                    })
        else:
            # Compute from scores
            for fj in range(n_factors):
                if fj != fi:
                    r = float(np.corrcoef(scores[:, fi], scores[:, fj])[0, 1])
                    factor_corrs.append({"factor": int(fj), "r": round(r, 4)})

        factor_data.append({
            "loadings": items_for_factor,
            "variance_explained": round(float(proportion_variance[fi]), 4),
            "ss_loading": round(float(ss_loadings[fi]), 4),
            "cumulative_variance": round(float(cumulative_variance[fi]), 4),
            "score_stats": {
                "mean": round(float(np.mean(factor_scores_col)), 4),
                "std": round(float(np.std(factor_scores_col)), 4),
                "skew": round(float(skew(factor_scores_col)), 4),
                "kurtosis": round(float(kurtosis(factor_scores_col)), 4),
                "min": round(float(np.min(factor_scores_col)), 4),
                "max": round(float(np.max(factor_scores_col)), 4),
                "n": int(len(factor_scores_col)),
            },
            "score_histogram": {
                "edges": [round(float(e), 4) for e in hist_edges],
                "counts": [int(c) for c in hist_counts],
            },
            "correlations": factor_corrs,
        })

    # Collect extreme persona records — include all factor scores per persona
    records = []
    for fi in range(n_factors):
        factor_scores = scores[:, fi]
        sorted_indices = np.argsort(factor_scores)

        # Bottom N (low scorers)
        for rank, idx in enumerate(sorted_indices[:n_per_pole]):
            meta = metadata[idx]
            sid = meta["sample_id"]
            conv = conversations_by_sid.get(sid, [])
            if not conv:
                continue
            records.append({
                "factor": fi,
                "pole": "low",
                "rank": rank + 1,
                "score": float(factor_scores[idx]),
                "all_scores": [round(float(scores[idx, fj]), 3) for fj in range(n_factors)],
                "sample_id": sid,
                "messages": conv,
            })

        # Top N (high scorers)
        for rank, idx in enumerate(sorted_indices[-n_per_pole:][::-1]):
            meta = metadata[idx]
            sid = meta["sample_id"]
            conv = conversations_by_sid.get(sid, [])
            if not conv:
                continue
            records.append({
                "factor": fi,
                "pole": "high",
                "rank": rank + 1,
                "score": float(factor_scores[idx]),
                "all_scores": [round(float(scores[idx, fj]), 3) for fj in range(n_factors)],
                "sample_id": sid,
                "messages": conv,
            })

    if not records:
        print(f"  [Extremes] No matching conversations found — skipped.")
        return

    # Build the HTML
    data_json = json.dumps(records, ensure_ascii=False)
    factors_json = json.dumps(factor_descriptions, ensure_ascii=False)
    factor_data_json = json.dumps(factor_data, ensure_ascii=False)
    title = html_mod.escape(f"Factor Extremes — {label}")

    html_content = _HTML_TEMPLATE.format(
        title=title,
        data_json=data_json,
        factors_json=factors_json,
        factor_data_json=factor_data_json,
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    html_path = Path(save_dir) / "factor_extremes.html"
    html_path.write_text(html_content, encoding="utf-8")
    n_factors_written = len(set(r["factor"] for r in records))
    print(
        f"  [Extremes] Wrote {len(records)} rollouts ({n_per_pole}/pole × {n_factors_written} factors) "
        f"to {html_path}"
    )


# Verbatim from the original script — CSS + HTML + JS with ``{title}``,
# ``{data_json}``, ``{factors_json}``, ``{factor_data_json}`` placeholders.
# CSS/JS braces doubled so ``str.format`` only interpolates the four slots.
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #111827; color: #e5e7eb;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 14px; line-height: 1.5;
    display: flex; height: 100vh; overflow: hidden;
  }}
  #sidebar {{
    width: 340px; min-width: 280px;
    background: #1f2937; border-right: 1px solid #374151;
    display: flex; flex-direction: column; overflow: hidden; flex-shrink: 0;
  }}
  #sidebar-header {{
    padding: 12px 16px; background: #0e7490; color: #fff;
    font-weight: bold; font-size: 15px; flex-shrink: 0;
  }}
  #factor-selector {{
    padding: 8px; border-bottom: 1px solid #374151; flex-shrink: 0;
  }}
  #factor-selector select {{
    width: 100%; padding: 6px 8px; border-radius: 4px;
    background: #374151; color: #e5e7eb; border: 1px solid #4b5563;
    font-size: 13px;
  }}
  /* Tab bar */
  #tab-bar {{
    display: flex; flex-shrink: 0; border-bottom: 1px solid #374151;
  }}
  .tab-btn {{
    flex: 1; padding: 8px 12px; text-align: center; cursor: pointer;
    font-size: 12px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; background: #1f2937; color: #9ca3af;
    border: none; border-bottom: 2px solid transparent;
    transition: all 0.15s;
  }}
  .tab-btn:hover {{ color: #e5e7eb; background: #283548; }}
  .tab-btn.active {{ color: #60a5fa; border-bottom-color: #60a5fa; background: #1e293b; }}
  /* Sidebar content panels */
  #sidebar-personas, #sidebar-factor {{
    flex: 1; overflow-y: auto; display: flex; flex-direction: column;
  }}
  #sidebar-factor {{ padding: 12px 14px; }}
  .sidebar-hidden {{ display: none !important; }}
  #factor-info {{
    padding: 10px 14px; border-bottom: 1px solid #374151;
    font-size: 12px; overflow-y: auto; max-height: 260px; flex-shrink: 0;
  }}
  #factor-info .pole {{ margin-bottom: 6px; }}
  .pole-label {{
    font-weight: 700; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .pole-high .pole-label {{ color: #4ade80; }}
  .pole-low .pole-label {{ color: #f87171; }}
  #factor-info .desc {{ color: #9ca3af; margin-top: 4px; font-style: italic; }}
  .axis-chip {{
    display: inline-block; margin-bottom: 6px; padding: 2px 7px;
    border-radius: 999px; background: #0f766e; color: #ccfbf1;
    font-size: 11px; font-weight: 700; letter-spacing: 0.03em;
  }}
  #factor-info .loading-item {{
    font-size: 11px; color: #9ca3af; margin-left: 8px;
  }}
  #record-list {{
    flex: 1; overflow-y: auto; padding: 4px 0;
  }}
  .record-entry {{
    padding: 8px 14px; cursor: pointer; border-left: 3px solid transparent;
    font-size: 12px; transition: background 0.1s;
  }}
  .record-entry:hover {{ background: #374151; }}
  .record-entry.active {{ background: #1e3a5f; border-left-color: #60a5fa; }}
  .record-entry .pole-tag {{
    display: inline-block; font-size: 10px; font-weight: 700;
    padding: 1px 5px; border-radius: 3px; margin-right: 6px;
    text-transform: uppercase;
  }}
  .pole-tag-high {{ background: #065f46; color: #6ee7b7; }}
  .pole-tag-low {{ background: #7f1d1d; color: #fca5a5; }}
  .record-entry .score {{ color: #9ca3af; font-size: 11px; }}
  /* Sidebar factor tab styles */
  .sf-section {{ margin-bottom: 16px; }}
  .sf-section-title {{
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; color: #60a5fa; margin-bottom: 6px;
  }}
  .sf-desc {{
    font-size: 13px; color: #d1d5db; font-style: italic;
    line-height: 1.5; margin-bottom: 8px;
  }}
  .sf-stat {{
    display: flex; justify-content: space-between;
    font-size: 12px; padding: 2px 0; border-bottom: 1px solid #2d3748;
  }}
  .sf-stat-label {{ color: #9ca3af; }}
  .sf-stat-value {{ color: #e5e7eb; font-weight: 600; font-variant-numeric: tabular-nums; }}
  .sf-corr-bar {{
    display: flex; align-items: center; gap: 6px;
    font-size: 11px; margin-bottom: 3px;
  }}
  .sf-corr-bar .bar-track {{
    flex: 1; height: 6px; background: #374151; border-radius: 3px; overflow: hidden;
    position: relative;
  }}
  .sf-corr-bar .bar-fill {{
    position: absolute; top: 0; height: 100%; border-radius: 3px;
  }}
  /* Main area */
  #main {{
    flex: 1; display: flex; flex-direction: column; overflow: hidden;
  }}
  #topbar {{
    background: #1e293b; padding: 8px 16px; font-size: 12px;
    border-bottom: 1px solid #374151; flex-shrink: 0;
    display: flex; gap: 16px; align-items: center;
  }}
  #topbar .tag {{ font-weight: 700; }}
  /* Persona score profile bar */
  .score-profile {{
    display: flex; gap: 4px; align-items: center; font-size: 10px; margin-left: auto;
  }}
  .score-profile .sp-item {{
    display: flex; align-items: center; gap: 2px; padding: 1px 4px;
    border-radius: 3px; background: #283548;
  }}
  .score-profile .sp-item.sp-current {{ background: #1e3a5f; font-weight: 700; }}
  .score-profile .sp-label {{ color: #6b7280; }}
  .score-profile .sp-val {{ color: #d1d5db; font-variant-numeric: tabular-nums; }}
  #main-personas, #main-factor {{
    flex: 1; overflow: hidden; display: flex; flex-direction: column;
  }}
  .main-hidden {{ display: none !important; }}
  #scroll-area {{
    flex: 1; overflow-y: auto; padding: 16px 24px 80px;
    max-width: 900px; width: 100%;
  }}
  #factor-view {{
    flex: 1; overflow-y: auto; padding: 20px 28px 80px;
    max-width: 1100px; width: 100%;
  }}
  .fv-section {{ margin-bottom: 32px; }}
  .fv-section-title {{
    font-size: 14px; font-weight: 700; color: #60a5fa;
    margin-bottom: 12px; padding-bottom: 4px;
    border-bottom: 1px solid #374151;
  }}
  .msg {{
    margin-bottom: 12px; padding: 10px 14px;
    border-radius: 8px; white-space: pre-wrap; word-break: break-word;
  }}
  .msg-user {{
    background: #1e3a5f; border-left: 3px solid #60a5fa;
  }}
  .msg-assistant {{
    background: #1a2e1a; border-left: 3px solid #4ade80;
  }}
  .msg-system {{
    background: #2d2215; border-left: 3px solid #f59e0b;
  }}
  .role-label {{
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em; margin-bottom: 4px; opacity: 0.7;
  }}
  .role-user .role-label {{ color: #60a5fa; }}
  .role-assistant .role-label {{ color: #4ade80; }}
  .role-system .role-label {{ color: #f59e0b; }}
  /* Items table */
  .items-table {{
    width: 100%; border-collapse: collapse; font-size: 12px;
  }}
  .items-table th {{
    text-align: left; padding: 6px 8px; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.05em;
    color: #9ca3af; border-bottom: 2px solid #374151;
  }}
  .items-table td {{
    padding: 5px 8px; border-bottom: 1px solid #1f2937;
    font-variant-numeric: tabular-nums;
  }}
  .items-table tr:hover {{ background: #1e293b; }}
  .items-table .loading-pos {{ color: #4ade80; }}
  .items-table .loading-neg {{ color: #f87171; }}
  .items-table .item-text {{
    color: #d1d5db; word-break: break-word;
  }}
  #bottombar {{
    background: #d1d5db; color: #111; font-weight: 600;
    padding: 5px 16px; font-size: 12px; flex-shrink: 0;
  }}
</style>
</head>
<body>
<div id="sidebar">
  <div id="sidebar-header">Factor Extremes</div>
  <div id="factor-selector"><select id="factor-select"></select></div>
  <div id="tab-bar">
    <button class="tab-btn active" data-tab="factor">Factors</button>
    <button class="tab-btn" data-tab="personas">Personas</button>
  </div>
  <div id="sidebar-personas" class="sidebar-hidden">
    <div id="factor-info"></div>
    <div id="record-list"></div>
  </div>
  <div id="sidebar-factor"></div>
</div>
<div id="main">
  <div id="main-personas" class="main-hidden">
    <div id="topbar">
      <span id="tb-info"></span>
      <div class="score-profile" id="score-profile"></div>
    </div>
    <div id="scroll-area"></div>
  </div>
  <div id="main-factor">
    <div id="factor-view"></div>
  </div>
  <div id="bottombar">
    ↑↓ or click to navigate &nbsp;|&nbsp; Factor dropdown to switch factors &nbsp;|&nbsp; Factors / Personas tabs
  </div>
</div>

<script>
const RECORDS = {data_json};
const FACTORS = {factors_json};
const FACTOR_DATA = {factor_data_json};
let currentFactor = 0;
let currentIdx = 0;
let currentTab = 'factor';
let filteredRecords = [];

const factorSelect = document.getElementById('factor-select');
const factorInfo = document.getElementById('factor-info');
const recordList = document.getElementById('record-list');
const scrollArea = document.getElementById('scroll-area');
const tbInfo = document.getElementById('tb-info');
const scoreProfile = document.getElementById('score-profile');
const sidebarPersonas = document.getElementById('sidebar-personas');
const sidebarFactor = document.getElementById('sidebar-factor');
const mainPersonas = document.getElementById('main-personas');
const mainFactor = document.getElementById('main-factor');
const factorView = document.getElementById('factor-view');

// Populate factor selector
FACTORS.forEach((f, i) => {{
  const opt = document.createElement('option');
  opt.value = i;
  const axis = f.axis_name ? ` — ${{f.axis_name}}` : '';
  const summary = f.summary ? `: ${{f.summary}}` : '';
  opt.textContent = `Factor ${{i}}${{axis}}${{summary}}`;
  factorSelect.appendChild(opt);
}});

factorSelect.addEventListener('change', () => {{
  currentFactor = parseInt(factorSelect.value);
  currentIdx = 0;
  updateView();
}});

// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    currentTab = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b === btn));
    switchTab();
  }});
}});

function switchTab() {{
  if (currentTab === 'personas') {{
    sidebarPersonas.classList.remove('sidebar-hidden');
    sidebarFactor.classList.add('sidebar-hidden');
    mainPersonas.classList.remove('main-hidden');
    mainFactor.classList.add('main-hidden');
  }} else {{
    sidebarPersonas.classList.add('sidebar-hidden');
    sidebarFactor.classList.remove('sidebar-hidden');
    mainPersonas.classList.add('main-hidden');
    mainFactor.classList.remove('main-hidden');
    renderFactorTab();
  }}
}}

function updateView() {{
  filteredRecords = RECORDS.filter(r => r.factor === currentFactor);

  // Update factor info panel (personas tab sidebar)
  const f = FACTORS[currentFactor];
  let infoHtml = '';
  if (f.axis_name) {{
    infoHtml += `<div class="axis-chip">${{f.axis_name}}</div>`;
  }}
  if (f.summary) {{
    infoHtml += `<div style="font-weight:bold;margin-bottom:6px">${{f.summary}}</div>`;
  }}
  if (f.description) {{
    infoHtml += `<div class="desc">${{f.description}}</div>`;
  }}
  infoHtml += `<div class="pole pole-high" style="margin-top:8px">`;
  infoHtml += `<div class="pole-label">▲ High: ${{f.positive_pole || '(unlabelled)'}}</div>`;
  if (f.top_positive_items) {{
    f.top_positive_items.forEach(it => {{
      const sign = it.loading >= 0 ? '+' : '';
      infoHtml += `<div class="loading-item">`
        + `<span style="color:#9ca3af;margin-right:4px">(${{sign}}${{it.loading.toFixed(3)}})</span>`
        + _itemBadgesHTML(it)
        + escHtml(it.text)
        + `</div>`;
    }});
  }}
  infoHtml += `</div>`;
  infoHtml += `<div class="pole pole-low" style="margin-top:8px">`;
  infoHtml += `<div class="pole-label">▼ Low: ${{f.negative_pole || '(unlabelled)'}}</div>`;
  if (f.top_negative_items) {{
    f.top_negative_items.forEach(it => {{
      const sign = it.loading >= 0 ? '+' : '';
      infoHtml += `<div class="loading-item">`
        + `<span style="color:#9ca3af;margin-right:4px">(${{sign}}${{it.loading.toFixed(3)}})</span>`
        + _itemBadgesHTML(it)
        + escHtml(it.text)
        + `</div>`;
    }});
  }}
  infoHtml += `</div>`;
  factorInfo.innerHTML = infoHtml;

  // Update record list
  recordList.innerHTML = '';
  filteredRecords.forEach((r, i) => {{
    const div = document.createElement('div');
    div.className = 'record-entry' + (i === currentIdx ? ' active' : '');
    const poleClass = r.pole === 'high' ? 'pole-tag-high' : 'pole-tag-low';
    const arrow = r.pole === 'high' ? '▲' : '▼';
    div.innerHTML = `<span class="pole-tag ${{poleClass}}">${{arrow}} ${{r.pole}} #${{r.rank}}</span>`
      + `<span class="score">score=${{r.score.toFixed(2)}}</span>`
      + `<div style="font-size:11px;color:#6b7280;margin-top:2px">${{r.sample_id.substring(0,24)}}…</div>`;
    div.addEventListener('click', () => {{ currentIdx = i; renderRecord(); highlightEntry(); }});
    recordList.appendChild(div);
  }});

  renderRecord();
  if (currentTab === 'factor') renderFactorTab();
}}

function highlightEntry() {{
  recordList.querySelectorAll('.record-entry').forEach((el, i) => {{
    el.classList.toggle('active', i === currentIdx);
  }});
  const active = recordList.querySelector('.active');
  if (active) active.scrollIntoView({{ block: 'nearest' }});
}}

function renderRecord() {{
  if (filteredRecords.length === 0) {{
    scrollArea.innerHTML = '<div style="padding:20px;color:#9ca3af">No records for this factor.</div>';
    tbInfo.textContent = '';
    scoreProfile.innerHTML = '';
    return;
  }}
  const rec = filteredRecords[currentIdx];
  const arrow = rec.pole === 'high' ? '▲' : '▼';
  tbInfo.innerHTML = `<span class="tag">${{arrow}} Factor ${{rec.factor}} · ${{rec.pole}} #${{rec.rank}}</span>`
    + ` &nbsp; score=${{rec.score.toFixed(3)}} &nbsp; ${{rec.sample_id}}`;

  // Score profile: show this persona's scores across all factors
  let spHtml = '';
  if (rec.all_scores) {{
    rec.all_scores.forEach((s, fi) => {{
      const cls = fi === currentFactor ? 'sp-item sp-current' : 'sp-item';
      const shortLabel = FACTORS[fi].axis_name
        ? FACTORS[fi].axis_name.substring(0, 8)
        : `F${{fi}}`;
      spHtml += `<div class="${{cls}}" title="${{shortLabel}}"><span class="sp-label">${{shortLabel}}</span><span class="sp-val">${{s.toFixed(1)}}</span></div>`;
    }});
  }}
  scoreProfile.innerHTML = spHtml;

  scrollArea.innerHTML = '';
  rec.messages.forEach(msg => {{
    const div = document.createElement('div');
    const role = msg.role || 'user';
    div.className = `msg msg-${{role}} role-${{role}}`;
    const label = document.createElement('div');
    label.className = 'role-label';
    label.textContent = role;
    div.appendChild(label);
    const body = document.createElement('div');
    body.textContent = msg.content;
    div.appendChild(body);
    scrollArea.appendChild(div);
  }});
  scrollArea.scrollTop = 0;
  highlightEntry();
}}

// ─── Factor tab rendering ──────────────────────────────────────────────────

function renderFactorTab() {{
  const f = FACTORS[currentFactor];
  const fd = FACTOR_DATA[currentFactor];

  // Sidebar: factor description + stats + correlations
  let sbHtml = '';

  // Description
  sbHtml += '<div class="sf-section">';
  sbHtml += '<div class="sf-section-title">Description</div>';
  if (f.axis_name) {{
    sbHtml += `<div class="axis-chip">${{f.axis_name}}</div>`;
  }}
  if (f.summary) {{
    sbHtml += `<div style="font-weight:700;font-size:14px;margin-bottom:6px">${{f.summary}}</div>`;
  }}
  if (f.description) {{
    sbHtml += `<div class="sf-desc">${{f.description}}</div>`;
  }}
  if (!f.summary && !f.description) {{
    sbHtml += '<div style="color:#6b7280;font-style:italic">No LLM description available. Run labeling stage to generate.</div>';
  }}
  sbHtml += `<div style="margin-top:6px"><span class="pole-label" style="color:#4ade80">▲ ${{f.positive_pole || 'high'}}</span></div>`;
  sbHtml += `<div><span class="pole-label" style="color:#f87171">▼ ${{f.negative_pole || 'low'}}</span></div>`;
  sbHtml += '</div>';

  // Variance stats
  sbHtml += '<div class="sf-section">';
  sbHtml += '<div class="sf-section-title">Variance</div>';
  sbHtml += `<div class="sf-stat"><span class="sf-stat-label">SS loading</span><span class="sf-stat-value">${{fd.ss_loading.toFixed(3)}}</span></div>`;
  sbHtml += `<div class="sf-stat"><span class="sf-stat-label">Proportion</span><span class="sf-stat-value">${{(fd.variance_explained * 100).toFixed(1)}}%</span></div>`;
  sbHtml += `<div class="sf-stat"><span class="sf-stat-label">Cumulative</span><span class="sf-stat-value">${{(fd.cumulative_variance * 100).toFixed(1)}}%</span></div>`;
  sbHtml += '</div>';

  // Score stats
  sbHtml += '<div class="sf-section">';
  sbHtml += '<div class="sf-section-title">Score distribution</div>';
  const ss = fd.score_stats;
  sbHtml += `<div class="sf-stat"><span class="sf-stat-label">N</span><span class="sf-stat-value">${{ss.n}}</span></div>`;
  sbHtml += `<div class="sf-stat"><span class="sf-stat-label">Mean</span><span class="sf-stat-value">${{ss.mean.toFixed(3)}}</span></div>`;
  sbHtml += `<div class="sf-stat"><span class="sf-stat-label">Std</span><span class="sf-stat-value">${{ss.std.toFixed(3)}}</span></div>`;
  sbHtml += `<div class="sf-stat"><span class="sf-stat-label">Skew</span><span class="sf-stat-value">${{ss.skew.toFixed(3)}}</span></div>`;
  sbHtml += `<div class="sf-stat"><span class="sf-stat-label">Kurtosis</span><span class="sf-stat-value">${{ss.kurtosis.toFixed(3)}}</span></div>`;
  sbHtml += `<div class="sf-stat"><span class="sf-stat-label">Range</span><span class="sf-stat-value">${{ss.min.toFixed(2)}} … ${{ss.max.toFixed(2)}}</span></div>`;
  sbHtml += '</div>';

  // Factor correlations
  if (fd.correlations.length > 0) {{
    sbHtml += '<div class="sf-section">';
    sbHtml += '<div class="sf-section-title">Factor correlations</div>';
    fd.correlations.sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
    fd.correlations.forEach(c => {{
      const absR = Math.abs(c.r);
      const pct = (absR * 100).toFixed(0);
      const color = c.r >= 0 ? '#4ade80' : '#f87171';
      const fAxis = FACTORS[c.factor].axis_name;
      const fLabel = fAxis
        ? `F${{c.factor}} (${{fAxis}})`
        : FACTORS[c.factor].summary
          ? `F${{c.factor}} (${{FACTORS[c.factor].summary.substring(0,25)}})`
          : `Factor ${{c.factor}}`;
      sbHtml += `<div class="sf-corr-bar">`;
      sbHtml += `<span style="min-width:30px;text-align:right;font-variant-numeric:tabular-nums">${{c.r >= 0 ? '+' : ''}}${{c.r.toFixed(3)}}</span>`;
      sbHtml += `<div class="bar-track"><div class="bar-fill" style="width:${{pct}}%;background:${{color}};${{c.r >= 0 ? 'left:0' : 'right:0'}}"></div></div>`;
      sbHtml += `<span style="color:#9ca3af;font-size:10px">${{fLabel}}</span>`;
      sbHtml += `</div>`;
    }});
    sbHtml += '</div>';
  }}

  sidebarFactor.innerHTML = sbHtml;

  // Main area: loading chart + histogram + items table
  let mainHtml = '';

  // Loading bar chart (SVG)
  mainHtml += '<div class="fv-section">';
  mainHtml += '<div class="fv-section-title">Item Loadings</div>';
  mainHtml += renderLoadingChart(fd.loadings);
  mainHtml += '</div>';

  // Score histogram (SVG)
  mainHtml += '<div class="fv-section">';
  mainHtml += '<div class="fv-section-title">Score Distribution</div>';
  mainHtml += renderHistogram(fd.score_histogram, fd.score_stats);
  mainHtml += '</div>';

  // Top items table
  mainHtml += '<div class="fv-section">';
  mainHtml += '<div class="fv-section-title">Top Loading Items</div>';
  mainHtml += renderItemsTable(fd.loadings);
  mainHtml += '</div>';

  factorView.innerHTML = mainHtml;
}}

// ── Shared item badge helpers (used by renderLoadingChart,
//    renderItemsTable, and the personas-tab sidebar top-items list). ───
function _blockBadgeHTML(block) {{
  if (!block) return '';
  const b = String(block).toLowerCase();
  const pretty = b === 'trait_mcq' ? 'MCQ'
               : b === 'fc_pair' ? 'FC'
               : b === 'likert' ? 'Likert'
               : String(block);
  const colour = b === 'trait_mcq' ? '#6366f1'
               : b === 'fc_pair' ? '#0ea5e9'
               : b === 'likert' ? '#14b8a6'
               : '#6b7280';
  return `<span style="display:inline-block;padding:1px 5px;margin-right:4px;border-radius:3px;background:${{colour}};color:#fff;font-size:9px;font-weight:700;letter-spacing:0.03em;text-transform:uppercase;vertical-align:middle">${{pretty}}</span>`;
}}
function _revBadgeHTML() {{
  return `<span title="Reverse-keyed: high Likert response = low trait expression" style="display:inline-block;padding:1px 5px;margin-right:4px;border-radius:3px;background:#b91c1c;color:#fff;font-size:9px;font-weight:700;letter-spacing:0.03em;vertical-align:middle">REV</span>`;
}}
function _dimBadgeHTML(dim) {{
  if (!dim) return '';
  return `<span style="display:inline-block;padding:1px 5px;margin-right:4px;border-radius:3px;background:#374151;color:#e5e7eb;font-size:9px;font-weight:600;letter-spacing:0.02em;vertical-align:middle">${{escHtml(String(dim))}}</span>`;
}}
function _itemBadgesHTML(it) {{
  return _blockBadgeHTML(it.block)
       + _dimBadgeHTML(it.dimension)
       + (it.reverse_keyed ? _revBadgeHTML() : '');
}}

function renderLoadingChart(items) {{
  // HTML-based horizontal bar chart — supports full wrapping text labels
  const maxAbs = Math.max(...items.map(it => Math.abs(it.loading)), 0.01);
  let html = '<div style="overflow-y:auto;max-height:650px">';
  html += '<table style="width:100%;border-collapse:collapse;font-size:12px">';

  // Header
  html += '<tr style="border-bottom:1px solid #374151">';
  html += '<th style="text-align:left;padding:4px 8px;color:#9ca3af;font-size:10px;width:50%">ITEM</th>';
  html += '<th style="text-align:center;padding:4px 8px;color:#9ca3af;font-size:10px;width:8%">LOADING</th>';
  html += '<th style="text-align:left;padding:4px 8px;color:#9ca3af;font-size:10px;width:42%">BAR</th>';
  html += '</tr>';

  items.forEach(it => {{
    const color = it.loading >= 0 ? '#4ade80' : '#f87171';
    const pct = (Math.abs(it.loading) / maxAbs * 50).toFixed(1);
    const barStyle = it.loading >= 0
      ? `margin-left:50%;width:${{pct}}%;background:${{color}}`
      : `margin-left:${{50 - parseFloat(pct)}}%;width:${{pct}}%;background:${{color}}`;

    html += '<tr style="border-bottom:1px solid #1f2937">';
    html += `<td style="padding:4px 8px;color:#d1d5db;word-break:break-word;line-height:1.4;font-size:11px">${{_itemBadgesHTML(it)}}${{escHtml(it.text)}}</td>`;
    html += `<td style="padding:4px 8px;text-align:center;font-variant-numeric:tabular-nums;color:${{color}};font-weight:600">${{it.loading >= 0 ? '+' : ''}}${{it.loading.toFixed(3)}}</td>`;
    html += `<td style="padding:4px 8px"><div style="position:relative;height:14px;background:#1f2937;border-radius:3px;overflow:hidden">`;
    html += `<div style="position:absolute;top:0;height:100%;border-radius:3px;${{barStyle}};opacity:0.8"></div>`;
    html += `<div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:#4b5563"></div>`;
    html += `</div></td>`;
    html += '</tr>';
  }});

  html += '</table></div>';
  return html;
}}

function renderHistogram(hist, stats) {{
  const edges = hist.edges, counts = hist.counts;
  const W = 500, H = 200, padL = 45, padR = 20, padT = 10, padB = 35;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const maxCount = Math.max(...counts, 1);
  const nBins = counts.length;
  const barW = plotW / nBins;

  let svg = `<svg width="${{W}}" height="${{H}}" style="font-family:inherit;font-size:10px">`;

  // Y axis
  svg += `<line x1="${{padL}}" y1="${{padT}}" x2="${{padL}}" y2="${{padT + plotH}}" stroke="#4b5563"/>`;
  for (let t = 0; t <= 4; t++) {{
    const val = Math.round(maxCount * t / 4);
    const y = padT + plotH - (plotH * t / 4);
    svg += `<text x="${{padL - 5}}" y="${{y + 3}}" fill="#6b7280" text-anchor="end">${{val}}</text>`;
    svg += `<line x1="${{padL}}" y1="${{y}}" x2="${{padL + plotW}}" y2="${{y}}" stroke="#2d3748" stroke-dasharray="2,2"/>`;
  }}

  // X axis
  svg += `<line x1="${{padL}}" y1="${{padT + plotH}}" x2="${{padL + plotW}}" y2="${{padT + plotH}}" stroke="#4b5563"/>`;

  // Bars
  counts.forEach((c, i) => {{
    const x = padL + i * barW;
    const h = (c / maxCount) * plotH;
    const y = padT + plotH - h;
    svg += `<rect x="${{x}}" y="${{y}}" width="${{barW - 1}}" height="${{h}}" fill="#60a5fa" opacity="0.7" rx="1"/>`;
  }});

  // X-axis labels (5 evenly spaced)
  for (let t = 0; t <= 4; t++) {{
    const idx = Math.round((nBins) * t / 4);
    const val = idx < edges.length ? edges[idx] : edges[edges.length - 1];
    const x = padL + (plotW * t / 4);
    svg += `<text x="${{x}}" y="${{padT + plotH + 15}}" fill="#6b7280" text-anchor="middle">${{val.toFixed(2)}}</text>`;
  }}

  // Stats annotation
  svg += `<text x="${{padL + plotW}}" y="${{padT + 14}}" fill="#9ca3af" text-anchor="end" font-size="10">`;
  svg += `mean=${{stats.mean.toFixed(2)}}  std=${{stats.std.toFixed(2)}}  skew=${{stats.skew.toFixed(2)}}  kurt=${{stats.kurtosis.toFixed(2)}}`;
  svg += `</text>`;

  svg += '</svg>';
  return svg;
}}

function renderItemsTable(items) {{
  // Show top 10 positive and top 10 negative loading items
  const posItems = items.filter(it => it.loading > 0).slice(0, 10);
  const negItems = items.filter(it => it.loading < 0);
  // negItems are already sorted by |loading| desc, so the most negative are first
  const topNeg = negItems.slice(0, 10);

  let html = '<table class="items-table">';
  html += '<thead><tr><th>Loading</th><th>h²</th><th>Cross</th><th>Item</th></tr></thead>';
  html += '<tbody>';

  function blockBadge(block) {{
    if (!block) return '';
    const b = String(block).toLowerCase();
    const pretty = b === 'trait_mcq' ? 'MCQ'
                 : b === 'fc_pair' ? 'FC'
                 : b === 'likert' ? 'Likert'
                 : String(block);
    const colour = b === 'trait_mcq' ? '#6366f1'
                 : b === 'fc_pair' ? '#0ea5e9'
                 : b === 'likert' ? '#14b8a6'
                 : '#6b7280';
    return `<span style="display:inline-block;padding:1px 6px;margin-right:6px;border-radius:3px;background:${{colour}};color:#fff;font-size:10px;font-weight:700;letter-spacing:0.03em;text-transform:uppercase">${{pretty}}</span>`;
  }}

  function revBadge() {{
    return `<span title="Reverse-keyed: higher Likert response = lower trait expression" style="display:inline-block;padding:1px 6px;margin-right:6px;border-radius:3px;background:#b91c1c;color:#fff;font-size:10px;font-weight:700;letter-spacing:0.03em">REV</span>`;
  }}

  function dimBadge(dim) {{
    if (!dim) return '';
    return `<span style="display:inline-block;padding:1px 6px;margin-right:6px;border-radius:3px;background:#374151;color:#e5e7eb;font-size:10px;font-weight:600;letter-spacing:0.02em">${{escHtml(String(dim))}}</span>`;
  }}

  function poleBadge(isHighPole) {{
    // The "high pole" of a factor is the direction that high factor scores
    // correspond to (high|+loading or low|−loading on the item value).
    const label = isHighPole ? 'HIGH' : 'LOW';
    const bg = isHighPole ? '#15803d' : '#991b1b';
    return `<span style="display:inline-block;padding:1px 6px;margin-right:6px;border-radius:3px;background:${{bg}};color:#fff;font-size:10px;font-weight:700;letter-spacing:0.03em">${{label}} pole</span>`;
  }}

  function optionsBlock(it) {{
    // Trait-MCQ options table. answer_mapping[letter] is 1 for the
    // "trait-aligned" answer, 0 otherwise, and the cell encoding is
    // Σ P(letter)·answer_mapping[letter]. So if the factor loading is
    // positive, high factor scores track trait-aligned answers; if
    // negative, high factor scores track trait-opposite answers. Hence:
    //     isHighPole = (answer_mapping == 1) XOR (loading < 0)
    if (!it.options || !it.answer_mapping) return '';
    const loadingPositive = it.loading >= 0;
    let out = '<div style="margin:6px 0 4px 0;padding:6px 8px;background:#0f172a;border-left:2px solid #6366f1;border-radius:2px">';
    out += '<div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:4px">Options</div>';
    it.options.forEach(opt => {{
      const mapVal = it.answer_mapping[opt.label];
      const trait_aligned = (mapVal === 1);
      const isHighPole = trait_aligned ? loadingPositive : !loadingPositive;
      out += '<div style="margin:3px 0;color:#d1d5db;font-size:12px;line-height:1.4">';
      out += `<span style="display:inline-block;width:18px;color:#9ca3af;font-weight:700">${{escHtml(opt.label)}}.</span>`;
      out += poleBadge(isHighPole);
      out += escHtml(opt.text);
      out += '</div>';
    }});
    out += '</div>';
    return out;
  }}

  function addRow(it) {{
    const cls = it.loading >= 0 ? 'loading-pos' : 'loading-neg';
    const crossLabel = it.max_cross_factor >= 0
      ? `F${{it.max_cross_factor}} (${{it.max_cross_loading >= 0 ? '+' : ''}}${{it.max_cross_loading.toFixed(2)}})`
      : '—';
    const badges = blockBadge(it.block)
                 + dimBadge(it.dimension)
                 + (it.reverse_keyed ? revBadge() : '');
    html += `<tr>`;
    html += `<td class="${{cls}}">${{it.loading >= 0 ? '+' : ''}}${{it.loading.toFixed(3)}}</td>`;
    html += `<td>${{it.communality.toFixed(3)}}</td>`;
    html += `<td style="font-size:11px;color:#6b7280">${{crossLabel}}</td>`;
    html += `<td class="item-text">${{badges}}<span title="${{escHtml(it.text)}}">${{escHtml(it.text)}}</span>${{optionsBlock(it)}}</td>`;
    html += `</tr>`;
  }}

  if (posItems.length > 0) {{
    html += `<tr><td colspan="4" style="padding:8px 8px 4px;color:#4ade80;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:0.05em;border-bottom:1px solid #065f46">▲ Positive loadings</td></tr>`;
    posItems.forEach(addRow);
  }}
  if (topNeg.length > 0) {{
    html += `<tr><td colspan="4" style="padding:12px 8px 4px;color:#f87171;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:0.05em;border-bottom:1px solid #7f1d1d">▼ Negative loadings</td></tr>`;
    topNeg.forEach(addRow);
  }}

  html += '</tbody></table>';
  return html;
}}

function escHtml(s) {{
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

// ─── Keyboard navigation ───────────────────────────────────────────────────

document.addEventListener('keydown', e => {{
  if (e.target.tagName === 'SELECT') return;
  if (currentTab === 'personas') {{
    if (e.key === 'ArrowDown' || e.key === 'j') {{
      currentIdx = Math.min(currentIdx + 1, filteredRecords.length - 1);
      renderRecord();
    }} else if (e.key === 'ArrowUp' || e.key === 'k') {{
      currentIdx = Math.max(currentIdx - 1, 0);
      renderRecord();
    }}
  }}
  // Tab switching with 1/2 keys
  if (e.key === '1') {{
    currentTab = 'personas';
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === 'personas'));
    switchTab();
  }} else if (e.key === '2') {{
    currentTab = 'factor';
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === 'factor'));
    switchTab();
  }}
}});

updateView();
</script>
</body>
</html>"""
