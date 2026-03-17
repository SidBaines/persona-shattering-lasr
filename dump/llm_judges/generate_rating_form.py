#!/usr/bin/env python3
"""Generate a self-contained HTML rating form from a heldout.jsonl file.

The HTML file needs no server, login, or install. Raters open it in any
browser, fill in their scores, and click "Download CSV". You collect the
CSVs and pass them to calibrate.py via --human-scores.

Usage:
    uv run python scripts/dump/llm_judges/generate_rating_form.py \\
        --judge neuroticism \\
        --rater alice \\
        --output scratch/rating/neuroticism_alice.html

    # Open neuroticism_alice.html in browser, fill scores, download CSV.
    # Then:
    uv run python scripts/dump/llm_judges/calibrate.py \\
        --judge neuroticism \\
        --models openai/gpt-4o-mini \\
        --human-scores scratch/rating/neuroticism_alice_filled.csv
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

JUDGES_DIR = Path(__file__).parent

SCALE_LABELS = [
    (-4, "Extreme low"),
    (-3, "Strong low"),
    (-2, "Moderate low"),
    (-1, "Slight low"),
    ( 0, "Neutral / no signal"),
    (+1, "Slight high"),
    (+2, "Moderate high"),
    (+3, "Strong high"),
    (+4, "Extreme high trait"),
]



def judge_dir(judge_name: str) -> Path:
    """Return the directory containing judge.py and heldout.jsonl."""
    for candidate in [
        JUDGES_DIR / "ocean" / judge_name,
        JUDGES_DIR / judge_name,
    ]:
        if (candidate / "heldout.jsonl").exists():
            return candidate
    raise FileNotFoundError(f"heldout.jsonl not found for '{judge_name}'")


def load_heldout(judge_name: str) -> list[dict]:
    path = judge_dir(judge_name) / "heldout.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def render_html(items: list[dict], judge_name: str, rater: str, trait_name: str) -> str:
    # Shuffle order per rater (reproducible) to avoid ordering bias.
    # The original item id is preserved in the JS data for CSV alignment.
    shuffled = items[:]
    random.Random(rater).shuffle(shuffled)

    # Pass only id + question + response to JS — no category, no expected score.
    items_for_js = [{"id": it["id"], "question": it["question"], "response": it["response"]}
                    for it in shuffled]
    items_json = json.dumps(items_for_js)

    scale_rows = "\n".join(
        f'<tr><td class="score-val">{v:+d}</td><td>{label}</td></tr>'
        for v, label in SCALE_LABELS
    )
    item_cards = ""
    for i, item in enumerate(shuffled):
        item_cards += f"""
        <div class="card" id="card-{i}">
          <div class="card-header">
            <span class="item-num">Item {i + 1} of {len(shuffled)}</span>
          </div>
          <div class="qa-block">
            <div class="label">Question</div>
            <div class="qa-text">{_esc(item['question'])}</div>
            <div class="label">Response</div>
            <div class="qa-text">{_esc(item['response'])}</div>
          </div>
          <div class="score-row">
            <span class="score-prompt">Your score:</span>
            {"".join(
                f'<label class="score-btn">'
                f'<input type="radio" name="score_{i}" value="{v}" required>'
                f'<span>{v:+d}</span></label>'
                for v, _ in SCALE_LABELS
            )}
          </div>
          <div class="progress-note" id="note-{i}"></div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{trait_name} personality rating — {rater}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f5f5f5; color: #222; line-height: 1.5; }}
  .page {{ max-width: 860px; margin: 0 auto; padding: 24px 16px 80px; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-bottom: 20px; font-size: 0.9rem; }}

  /* Scale reference */
  .scale-box {{ background: #fff; border: 1px solid #ddd; border-radius: 8px;
               padding: 14px 18px; margin-bottom: 24px; }}
  .scale-box h2 {{ font-size: 0.95rem; margin-bottom: 8px; color: #444; }}
  .scale-box table {{ border-collapse: collapse; font-size: 0.85rem; }}
  .scale-box td {{ padding: 2px 12px 2px 0; }}
  .score-val {{ font-weight: 700; text-align: right; font-variant-numeric: tabular-nums; }}

  /* Progress */
  .progress-bar-wrap {{ background: #e0e0e0; border-radius: 4px; height: 8px;
                        margin-bottom: 20px; }}
  .progress-bar {{ background: #4caf50; height: 8px; border-radius: 4px;
                   width: 0%; transition: width 0.3s; }}
  .progress-text {{ font-size: 0.82rem; color: #666; margin-bottom: 6px; }}

  /* Cards */
  .card {{ background: #fff; border: 1px solid #ddd; border-radius: 10px;
           padding: 18px 20px; margin-bottom: 16px; transition: border-color 0.2s; }}
  .card.answered {{ border-color: #4caf50; }}
  .card-header {{ margin-bottom: 10px; font-size: 0.82rem; }}
  .item-num {{ font-weight: 600; color: #555; }}
  .label {{ font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.05em; color: #888; margin-top: 10px; margin-bottom: 3px; }}
  .qa-text {{ font-size: 0.93rem; white-space: pre-wrap; background: rgba(0,0,0,0.03);
              border-radius: 5px; padding: 8px 10px; }}
  .score-row {{ display: flex; align-items: center; flex-wrap: wrap; gap: 6px;
                margin-top: 14px; }}
  .score-prompt {{ font-size: 0.85rem; font-weight: 600; margin-right: 4px; }}
  .score-btn input {{ display: none; }}
  .score-btn span {{ display: inline-block; padding: 5px 10px; border-radius: 5px;
                     border: 1.5px solid #ccc; cursor: pointer; font-size: 0.82rem;
                     font-weight: 600; transition: all 0.15s; user-select: none; }}
  .score-btn input:checked + span {{ background: #1976d2; color: #fff; border-color: #1976d2; }}
  .score-btn span:hover {{ border-color: #1976d2; background: #e3f0fb; }}
  .progress-note {{ font-size: 0.78rem; color: #4caf50; margin-top: 6px; min-height: 1em; }}

  /* Submit */
  .submit-bar {{ position: fixed; bottom: 0; left: 0; right: 0;
                 background: #fff; border-top: 1px solid #ddd;
                 padding: 12px 24px; display: flex; align-items: center; gap: 16px; }}
  .submit-bar button {{ padding: 10px 28px; background: #1976d2; color: #fff;
                        border: none; border-radius: 6px; font-size: 1rem;
                        cursor: pointer; font-weight: 600; }}
  .submit-bar button:disabled {{ background: #aaa; cursor: not-allowed; }}
  .submit-bar button:not(:disabled):hover {{ background: #1565c0; }}
  .submit-status {{ font-size: 0.9rem; color: #666; }}
  .error {{ color: #c62828; }}
</style>
</head>
<body>
<div class="page">
  <h1>{trait_name} personality rating</h1>
  <p class="subtitle">Rater: <strong>{rater}</strong> &nbsp;·&nbsp;
     Judge: <strong>{judge_name}</strong> &nbsp;·&nbsp;
     {len(items)} items</p>

  <div class="scale-box">
    <h2>Score scale</h2>
    <table>{scale_rows}</table>
    <p style="margin-top:10px;font-size:0.82rem;color:#666;">
      Score only what's in the <strong>Response</strong> — ignore topic alone.<br>
      Factual correctness and politeness ("happy to help") are <strong>not</strong> trait signals.
    </p>
  </div>

  <div class="progress-text" id="prog-text">0 / {len(items)} scored</div>
  <div class="progress-bar-wrap"><div class="progress-bar" id="prog-bar"></div></div>

  <form id="rating-form">
    {item_cards}
  </form>
</div>

<div class="submit-bar">
  <button id="submit-btn" disabled>Download CSV</button>
  <span class="submit-status" id="submit-status">Score all items to enable download.</span>
</div>

<script>
const ITEMS = {items_json};
const RATER = {json.dumps(rater)};
const JUDGE = {json.dumps(judge_name)};
const TOTAL = ITEMS.length;

function getScores() {{
  return ITEMS.map((_, i) => {{
    const checked = document.querySelector(`input[name="score_${{i}}"]:checked`);
    return checked ? parseInt(checked.value) : null;
  }});
}}

function updateProgress() {{
  const scores = getScores();
  const done = scores.filter(s => s !== null).length;
  document.getElementById("prog-text").textContent = `${{done}} / ${{TOTAL}} scored`;
  document.getElementById("prog-bar").style.width = `${{(done/TOTAL*100).toFixed(1)}}%`;
  const allDone = done === TOTAL;
  document.getElementById("submit-btn").disabled = !allDone;
  document.getElementById("submit-status").textContent =
    allDone ? "All items scored — ready to download." : `${{TOTAL - done}} item(s) remaining.`;
}}

function markAnswered(i) {{
  document.getElementById(`card-${{i}}`).classList.add("answered");
  document.getElementById(`note-${{i}}`).textContent = "✓ scored";
  updateProgress();
}}

document.querySelectorAll("input[type=radio]").forEach(input => {{
  const i = parseInt(input.name.replace("score_", ""));
  input.addEventListener("change", () => markAnswered(i));
}});

document.getElementById("submit-btn").addEventListener("click", () => {{
  const scores = getScores();
  const rows = [["id", "category", "question", "response", `score_${{RATER}}`]];
  ITEMS.forEach((item, i) => {{
    rows.push([item.id, item.category || "", item.question, item.response, scores[i]]);
  }});
  const csv = rows.map(r => r.map(v => `"${{String(v).replace(/"/g, '""')}}"`).join(",")).join("\\n");
  const blob = new Blob([csv], {{type: "text/csv"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${{JUDGE}}_${{RATER}}_ratings.csv`;
  a.click();
}});

updateProgress();
</script>
</body>
</html>"""


def _esc(text: str) -> str:
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML rating form from heldout.jsonl."
    )
    parser.add_argument("--judge", required=True, help="Judge name, e.g. 'neuroticism'.")
    parser.add_argument("--rater", required=True, help="Rater name (used in output filename and CSV column).")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path.")
    args = parser.parse_args()

    items = load_heldout(args.judge)
    trait_name = args.judge.replace("_", " ").title()
    html = render_html(items, args.judge, args.rater, trait_name)

    output_path = (
        Path(args.output)
        if args.output
        else judge_dir(args.judge) / "ratings" / f"{args.rater}.html"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Generated {output_path}")
    print(f"Share with {args.rater} — they open it in any browser, score, and click Download CSV.")


if __name__ == "__main__":
    main()
