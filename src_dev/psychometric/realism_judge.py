"""Helpers for the diagnostic realism judge stage.

The main ``run_stage_realism_judge`` entry point lives in
``src_dev.psychometric.stages.realism_judge``. This module contains the
two reusable helpers it (and subset scripts) depend on:

* ``summarize_realism_scores`` — aggregate mean/p10/p50/p90 stats overall
  and grouped by scenario / archetype, printed and returned.
* ``write_conversation_html`` — standalone HTML viewer for a conversation
  JSONL file (also used by the questionnaire-inspection inline export).
"""

from __future__ import annotations

import html as html_mod
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize_realism_scores(rows: list[dict]) -> dict:
    """Print + return mean / p10 / p50 / p90 stats grouped by scenario and archetype.

    Rows with ``unrealism_score < 0`` (the judge's error sentinel) are
    excluded from aggregates.
    """
    def _quantile(values: list[float], q: float) -> float:
        if not values:
            return float("nan")
        if len(values) == 1:
            return float(values[0])
        sv = sorted(values)
        pos = (len(sv) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(sv) - 1)
        frac = pos - lo
        return sv[lo] * (1 - frac) + sv[hi] * frac

    def _group_stats(
        rows: list[dict], key: str, score_field: str
    ) -> dict[str, dict[str, float]]:
        by: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            v = r.get(score_field)
            if isinstance(v, (int, float)) and v >= 0:  # score_error = -1
                by[str(r.get(key))].append(float(v))
        out: dict[str, dict[str, float]] = {}
        for k, vs in by.items():
            out[k] = {
                "n": len(vs),
                "mean": statistics.fmean(vs) if vs else float("nan"),
                "p10": _quantile(vs, 0.10),
                "p50": _quantile(vs, 0.50),
                "p90": _quantile(vs, 0.90),
            }
        return out

    def _overall(rows: list[dict], score_field: str) -> dict[str, float]:
        vs = [
            float(r[score_field])
            for r in rows
            if isinstance(r.get(score_field), (int, float)) and r[score_field] >= 0
        ]
        return {
            "n": len(vs),
            "mean": statistics.fmean(vs) if vs else float("nan"),
            "p10": _quantile(vs, 0.10),
            "p50": _quantile(vs, 0.50),
            "p90": _quantile(vs, 0.90),
        }

    summary = {
        "n_rows": len(rows),
        "unrealism": {
            "overall": _overall(rows, "unrealism_score"),
            "by_scenario": _group_stats(rows, "scenario_id", "unrealism_score"),
            "by_archetype": _group_stats(rows, "archetype", "unrealism_score"),
        },
    }

    def _fmt(s: dict[str, float]) -> str:
        return (
            f"n={int(s['n'])} mean={s['mean']:.2f} "
            f"p10={s['p10']:.1f} p50={s['p50']:.1f} p90={s['p90']:.1f}"
        )

    print("[Realism] Unrealism — overall: " + _fmt(summary["unrealism"]["overall"]))

    by_arch_u = summary["unrealism"]["by_archetype"]
    if by_arch_u:
        print("[Realism] Unrealism by archetype:")
        for arch in sorted(by_arch_u):
            print(f"    {arch:20s}  {_fmt(by_arch_u[arch])}")

    return summary


def write_conversation_html(jsonl_path: Path, html_path: Path) -> None:
    """Write a self-contained HTML viewer for conversation JSONL files.

    Each record with a ``messages`` list is rendered as a chat transcript.
    The last two messages of every record are highlighted separately, under
    a "▼ QUESTIONNAIRE ▼" separator — this is the convention produced by
    the questionnaire-inspection export (item prompt + parsed answer tacked
    on to the rollout).

    Arrow keys / j / k / Home / End navigate between records.
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    js_records = []
    for rec in records:
        msgs = rec.get("messages", [])
        js_records.append({
            "sample_id": rec.get("sample_id", ""),
            "item_text": rec.get("item_text", ""),
            "parsed": rec.get("parsed"),
            "messages": [{"role": m["role"], "content": m["content"]} for m in msgs],
        })

    data_json = json.dumps(js_records, ensure_ascii=False)
    title = html_mod.escape(jsonl_path.stem)

    html_content = _HTML_TEMPLATE.format(title=title, data_json=data_json)
    html_path.write_text(html_content, encoding="utf-8")


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
    display: flex; flex-direction: column; height: 100vh; overflow: hidden;
  }}
  #topbar {{
    background: #0e7490; color: #fff; font-weight: bold;
    padding: 8px 16px; display: flex; gap: 24px; flex-shrink: 0;
  }}
  #topbar .dim {{ opacity: 0.7; }}
  #scroll-area {{
    flex: 1; overflow-y: auto; padding: 16px 24px 80px;
    max-width: 900px; margin: 0 auto; width: 100%;
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
    background: #2d2235; border-left: 3px solid #c084fc;
    font-style: italic;
  }}
  .msg-questionnaire {{
    background: #3b2f1a; border-left: 3px solid #facc15;
  }}
  .msg-answer {{
    background: #1a3a2a; border-left: 3px solid #22d3ee;
    font-size: 18px; font-weight: bold; text-align: center;
  }}
  .role-label {{
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em; margin-bottom: 4px; opacity: 0.7;
  }}
  .role-user .role-label {{ color: #60a5fa; }}
  .role-assistant .role-label {{ color: #4ade80; }}
  .role-system .role-label {{ color: #c084fc; }}
  .separator {{
    text-align: center; color: #facc15; font-weight: bold;
    margin: 20px 0 8px; padding: 6px;
    border-top: 1px dashed #facc15; border-bottom: 1px dashed #facc15;
    font-size: 12px; letter-spacing: 0.1em;
  }}
  #bottombar {{
    background: #d1d5db; color: #111; font-weight: 600;
    padding: 5px 16px; font-size: 12px; flex-shrink: 0;
  }}
</style>
</head>
<body>
<div id="topbar">
  <span id="tb-nav"></span>
  <span class="dim" id="tb-item"></span>
  <span class="dim" id="tb-score"></span>
</div>
<div id="scroll-area"></div>
<div id="bottombar">
  ← → &nbsp;Navigate between conversations &nbsp;|&nbsp;
  Home / End &nbsp;First / Last
</div>
<script>
const RECORDS = {data_json};
let idx = 0;

function render() {{
  const rec = RECORDS[idx];
  const msgs = rec.messages;
  const n = msgs.length;

  document.getElementById('tb-nav').textContent =
    `Record ${{idx + 1}} / ${{RECORDS.length}}  (${{rec.sample_id}})`;
  document.getElementById('tb-item').textContent =
    rec.item_text ? `Item: ${{rec.item_text.substring(0, 80)}}` : '';
  document.getElementById('tb-score').textContent =
    rec.parsed != null ? `Score: ${{rec.parsed}}` : '';

  const area = document.getElementById('scroll-area');
  area.innerHTML = '';

  // Conversation messages (all except last 2 which are the questionnaire)
  const convEnd = n >= 2 ? n - 2 : n;
  for (let i = 0; i < convEnd; i++) {{
    area.appendChild(makeMsg(msgs[i]));
  }}

  // Questionnaire separator + final 2 messages
  if (n >= 2) {{
    const sep = document.createElement('div');
    sep.className = 'separator';
    sep.textContent = '▼ QUESTIONNAIRE ▼';
    area.appendChild(sep);
    area.appendChild(makeMsg(msgs[n - 2], 'msg-questionnaire'));
    area.appendChild(makeMsg(msgs[n - 1], 'msg-answer'));
  }}

  area.scrollTop = area.scrollHeight;
}}

function makeMsg(msg, extraClass) {{
  const div = document.createElement('div');
  const role = msg.role || 'user';
  div.className = `msg msg-${{role}} role-${{role}}` + (extraClass ? ` ${{extraClass}}` : '');

  const label = document.createElement('div');
  label.className = 'role-label';
  label.textContent = role;
  div.appendChild(label);

  const body = document.createElement('div');
  body.textContent = msg.content;
  div.appendChild(body);

  return div;
}}

document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight' || e.key === 'l') {{
    idx = Math.min(idx + 1, RECORDS.length - 1); render();
  }} else if (e.key === 'ArrowLeft' || e.key === 'h') {{
    idx = Math.max(idx - 1, 0); render();
  }} else if (e.key === 'Home' || e.key === 'g') {{
    idx = 0; render();
  }} else if (e.key === 'End' || e.key === 'G') {{
    idx = RECORDS.length - 1; render();
  }}
}});

render();
</script>
</body>
</html>"""
