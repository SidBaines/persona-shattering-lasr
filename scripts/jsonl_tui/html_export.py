"""Generate a self-contained HTML viewer for JSONL files.

Replicates the grouped-variant-fields navigation mode of the TUI:
  - Up / Down  →  navigate between question groups
  - Left / Right  →  navigate between responses within a group
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def generate_html(jsonl_path: str | Path, variant_fields: list[str]) -> str:
    """Return a self-contained HTML string for browsing a JSONL file.

    Args:
        jsonl_path: Path to the JSONL file.
        variant_fields: Ordered list of field names to display, matching the
            --variant-fields argument used with the TUI.  Must include
            "question" for grouped-variant mode (same requirement as the TUI).

    Returns:
        HTML string that can be written to a .html file.
    """
    jsonl_path = Path(jsonl_path)
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # --- Group records by the "question" field (same logic as TUI) ---
    group_order: list[str] = []
    groups: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    for idx, rec in enumerate(records):
        q = str(rec.get("question", f"Record {idx + 1}"))
        if q not in seen:
            group_order.append(q)
            seen.add(q)
        groups[q].append(rec)

    # Sort within each group by response_index then file order
    for q in group_order:
        groups[q].sort(key=lambda r: (
            0 if "response_index" in r else 1,
            r.get("response_index", 0),
        ))

    # Serialise for embedding
    grouped_data = [
        {"question": q, "responses": groups[q]}
        for q in group_order
    ]
    data_json = json.dumps(grouped_data, ensure_ascii=False)
    fields_json = json.dumps(variant_fields)
    title = jsonl_path.name

    # Colour palette cycling across fields (matches TUI's 6-colour cycle)
    field_colours = [
        "#22d3ee",  # cyan
        "#facc15",  # yellow
        "#4ade80",  # green
        "#e879f9",  # magenta
        "#60a5fa",  # blue
        "#f87171",  # red
    ]
    colours_json = json.dumps(field_colours)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: #111827;
    color: #e5e7eb;
    font-family: 'Cascadia Code', 'Fira Code', 'Menlo', 'Consolas', monospace;
    font-size: 14px;
    line-height: 1.6;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }}

  #topbar {{
    background: #0e7490;
    color: #000;
    font-weight: bold;
    padding: 6px 16px;
    display: flex;
    gap: 24px;
    flex-shrink: 0;
    white-space: nowrap;
    overflow: hidden;
  }}
  #topbar span {{ opacity: 0.85; }}
  #topbar span.hi {{ opacity: 1; color: #fff; }}

  #scroll-area {{
    flex: 1;
    overflow-y: auto;
    padding: 16px 24px 80px;
  }}

  .section {{
    margin-bottom: 20px;
  }}
  .section-label {{
    font-weight: bold;
    letter-spacing: 0.05em;
    margin-bottom: 2px;
  }}
  .section-sep {{
    color: #374151;
    margin-bottom: 6px;
    user-select: none;
  }}
  .section-body {{
    white-space: pre-wrap;
    word-break: break-word;
    color: #f3f4f6;
  }}

  #bottombar {{
    background: #d1d5db;
    color: #111;
    font-weight: 600;
    padding: 5px 16px;
    font-size: 12px;
    flex-shrink: 0;
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
  }}

  /* Search overlay */
  #search-box {{
    display: none;
    position: fixed;
    bottom: 28px;
    left: 0;
    right: 0;
    background: #1f2937;
    border-top: 2px solid #0e7490;
    padding: 6px 16px;
    z-index: 10;
  }}
  #search-box input {{
    background: transparent;
    border: none;
    outline: none;
    color: #e5e7eb;
    font-family: inherit;
    font-size: 14px;
    width: 100%;
  }}
  #search-results {{
    font-size: 12px;
    color: #9ca3af;
    margin-top: 2px;
  }}

  mark {{
    background: #854d0e;
    color: #fef3c7;
    border-radius: 2px;
  }}
  mark.current {{
    background: #d97706;
    color: #fff;
  }}
</style>
</head>
<body>

<div id="topbar">
  <span class="hi" id="tb-file">{title}</span>
  <span id="tb-group"></span>
  <span id="tb-response"></span>
</div>

<div id="scroll-area"></div>

<div id="search-box">
  <input id="search-input" placeholder="Search…" autocomplete="off" spellcheck="false">
  <div id="search-results"></div>
</div>

<div id="bottombar">
  ↑ ↓ &nbsp;Navigate factors &nbsp;|&nbsp; ← → &nbsp;Navigate responses &nbsp;|&nbsp; / &nbsp;Search &nbsp;|&nbsp; n / N &nbsp;Next / prev match &nbsp;|&nbsp; Esc &nbsp;Close search
</div>

<script>
const GROUPS = {data_json};
const FIELDS = {fields_json};
const COLOURS = {colours_json};

// Assign a colour to each field name
const fieldColour = {{}};
FIELDS.forEach((f, i) => {{ fieldColour[f] = COLOURS[i % COLOURS.length]; }});

let groupIdx = 0;
let responseIdx = 0;

// --- Search state ---
let searchMatches = []; // [{{groupIdx, responseIdx}}]
let searchMatchIdx = -1;
let searchQuery = '';

function currentGroup() {{ return GROUPS[groupIdx]; }}
function currentRecord() {{ return currentGroup().responses[responseIdx] ?? {{}}; }}

function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}

function render() {{
  const group = currentGroup();
  const record = currentRecord();
  const numGroups = GROUPS.length;
  const numResponses = group.responses.length;

  // Header
  document.getElementById('tb-group').textContent =
    `Factor ${{groupIdx + 1}} / ${{numGroups}}`;
  document.getElementById('tb-response').textContent =
    `Response ${{responseIdx + 1}} / ${{numResponses}}`;

  // Build content sections
  const scrollArea = document.getElementById('scroll-area');
  scrollArea.innerHTML = '';

  FIELDS.forEach(field => {{
    const val = record[field];
    const displayVal = val === undefined ? '(not available)' : String(val);

    const sec = document.createElement('div');
    sec.className = 'section';

    const lbl = document.createElement('div');
    lbl.className = 'section-label';
    lbl.style.color = fieldColour[field] ?? '#22d3ee';
    lbl.textContent = field.toUpperCase().replace(/_/g, ' ');

    const sep = document.createElement('div');
    sep.className = 'section-sep';
    sep.textContent = '═'.repeat(60);

    const body = document.createElement('div');
    body.className = 'section-body';

    // Highlight search matches in body text
    if (searchQuery && displayVal.toLowerCase().includes(searchQuery.toLowerCase())) {{
      body.innerHTML = highlightText(displayVal, searchQuery);
    }} else {{
      body.textContent = displayVal;
    }}

    sec.appendChild(lbl);
    sec.appendChild(sep);
    sec.appendChild(body);
    scrollArea.appendChild(sec);
  }});

  scrollArea.scrollTop = 0;

  // Update current-match highlight
  if (searchMatches.length > 0) {{
    highlightCurrentMatch();
  }}
}}

function highlightText(text, query) {{
  // Escape HTML special chars, then wrap matches
  const escaped = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const re = new RegExp(escapeRegex(query), 'gi');
  return escaped.replace(re, m => `<mark>${{m}}</mark>`);
}}

function highlightCurrentMatch() {{
  const marks = document.querySelectorAll('#scroll-area mark');
  marks.forEach(m => m.classList.remove('current'));
  if (searchMatchIdx >= 0 && searchMatchIdx < searchMatches.length) {{
    const m = searchMatches[searchMatchIdx];
    if (m.groupIdx === groupIdx && m.responseIdx === responseIdx) {{
      // Find which occurrence within this record we're at
      const recordStart = searchMatches.findIndex(
        x => x.groupIdx === groupIdx && x.responseIdx === responseIdx
      );
      const offset = searchMatchIdx - recordStart;
      if (marks[offset]) {{
        marks[offset].classList.add('current');
        marks[offset].scrollIntoView({{block: 'nearest'}});
      }}
    }}
  }}
}}

function escapeRegex(s) {{
  return s.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&');
}}

function buildSearchMatches(query) {{
  searchMatches = [];
  if (!query) return;
  const lq = query.toLowerCase();
  GROUPS.forEach((group, gi) => {{
    group.responses.forEach((rec, ri) => {{
      const hit = FIELDS.some(f => {{
        const v = rec[f];
        return v !== undefined && String(v).toLowerCase().includes(lq);
      }});
      if (hit) searchMatches.push({{groupIdx: gi, responseIdx: ri}});
    }});
  }});
}}

function updateSearchResults() {{
  const el = document.getElementById('search-results');
  if (!searchQuery) {{ el.textContent = ''; return; }}
  el.textContent = searchMatches.length === 0
    ? 'No matches'
    : `Match ${{searchMatchIdx + 1}} / ${{searchMatches.length}}`;
}}

function jumpToMatch(idx) {{
  if (searchMatches.length === 0) return;
  searchMatchIdx = ((idx % searchMatches.length) + searchMatches.length) % searchMatches.length;
  const m = searchMatches[searchMatchIdx];
  groupIdx = m.groupIdx;
  responseIdx = m.responseIdx;
  render();
  updateSearchResults();
}}

// --- Navigation ---
function navigate(dGroup, dResponse) {{
  if (dGroup !== 0) {{
    groupIdx = clamp(groupIdx + dGroup, 0, GROUPS.length - 1);
    responseIdx = 0;
  }}
  if (dResponse !== 0) {{
    const maxR = currentGroup().responses.length - 1;
    responseIdx = clamp(responseIdx + dResponse, 0, maxR);
  }}
  render();
}}

let searchOpen = false;

function openSearch() {{
  searchOpen = true;
  document.getElementById('search-box').style.display = 'block';
  document.getElementById('search-input').focus();
}}

function closeSearch() {{
  searchOpen = false;
  document.getElementById('search-box').style.display = 'none';
  document.getElementById('search-input').blur();
}}

document.getElementById('search-input').addEventListener('input', e => {{
  searchQuery = e.target.value;
  buildSearchMatches(searchQuery);
  searchMatchIdx = searchMatches.length > 0 ? 0 : -1;
  if (searchMatches.length > 0) {{
    groupIdx = searchMatches[0].groupIdx;
    responseIdx = searchMatches[0].responseIdx;
  }}
  render();
  updateSearchResults();
}});

document.getElementById('search-input').addEventListener('keydown', e => {{
  if (e.key === 'Enter') {{
    e.preventDefault();
    jumpToMatch(searchMatchIdx + (e.shiftKey ? -1 : 1));
  }} else if (e.key === 'Escape') {{
    closeSearch();
  }}
}});

document.addEventListener('keydown', e => {{
  if (searchOpen) return;

  if (e.key === '/') {{
    e.preventDefault();
    openSearch();
    return;
  }}
  if (e.key === 'n') {{ jumpToMatch(searchMatchIdx + 1); return; }}
  if (e.key === 'N') {{ jumpToMatch(searchMatchIdx - 1); return; }}
  if (e.key === 'Escape') {{ searchQuery = ''; searchMatches = []; searchMatchIdx = -1; render(); return; }}

  if (e.key === 'ArrowUp'    || e.key === 'k') {{ e.preventDefault(); navigate(-1, 0); }}
  else if (e.key === 'ArrowDown'  || e.key === 'j') {{ e.preventDefault(); navigate(+1, 0); }}
  else if (e.key === 'ArrowLeft'  || e.key === 'h') {{ e.preventDefault(); navigate(0, -1); }}
  else if (e.key === 'ArrowRight' || e.key === 'l') {{ e.preventDefault(); navigate(0, +1); }}
  else if (e.key === 'g') {{ groupIdx = 0; responseIdx = 0; render(); }}
  else if (e.key === 'G') {{ groupIdx = GROUPS.length - 1; responseIdx = 0; render(); }}
}});

render();
</script>
</body>
</html>
"""
    return html


def export_html(jsonl_path: str | Path, variant_fields: list[str], output_path: str | Path | None = None) -> Path:
    """Write a self-contained HTML viewer for a JSONL file.

    Args:
        jsonl_path: Path to the source JSONL file.
        variant_fields: Ordered list of field names (same as --variant-fields).
        output_path: Where to write the HTML.  Defaults to the JSONL path
            with a .html extension.

    Returns:
        Path to the written HTML file.
    """
    jsonl_path = Path(jsonl_path)
    if output_path is None:
        output_path = jsonl_path.with_suffix(".html")
    output_path = Path(output_path)
    output_path.write_text(generate_html(jsonl_path, variant_fields), encoding="utf-8")
    return output_path
