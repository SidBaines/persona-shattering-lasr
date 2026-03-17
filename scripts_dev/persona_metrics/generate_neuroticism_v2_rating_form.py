"""Generate a blinded HTML rating form for the neuroticism_v2 calibration set."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from src_dev.persona_metrics.judge_calibration import filter_items, load_calibration_items

DEFAULT_DATASET = Path("data/neuroticism_v2_calibration.jsonl")
SCALE = [(-4, "Extreme low"), (-3, "Strong low"), (-2, "Moderate low"), (-1, "Slight low"), (0, "Neutral / no signal"), (1, "Slight high"), (2, "Moderate high"), (3, "Strong high"), (4, "Extreme high")]


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def render_html(items: list[dict[str, str]], rater: str) -> str:
    items_json = json.dumps(items)
    scale_rows = "\n".join(
        f"<tr><td class='score-val'>{value:+d}</td><td>{label}</td></tr>" for value, label in SCALE
    )
    cards: list[str] = []
    for index, item in enumerate(items):
        buttons = "".join(
            f"<label class='score-btn'><input type='radio' name='score_{index}' value='{value}' required><span>{value:+d}</span></label>"
            for value, _ in SCALE
        )
        cards.append(
            f"""
        <div class='card' id='card-{index}'>
          <div class='card-header'>Item {index + 1} of {len(items)}</div>
          <div class='label'>Question</div>
          <div class='qa-text'>{_escape_html(item['question'])}</div>
          <div class='label'>Response</div>
          <div class='qa-text'>{_escape_html(item['response'])}</div>
          <div class='score-row'><span class='score-prompt'>Your score:</span>{buttons}</div>
          <div class='progress-note' id='note-{index}'></div>
        </div>"""
        )

    html = f"""
<html lang='en'>
<head>
<meta charset='UTF-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Neuroticism v2 rating</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; color: #222; line-height: 1.5; }}
  .page {{ max-width: 900px; margin: 0 auto; padding: 24px 16px 90px; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-bottom: 18px; font-size: 0.92rem; }}
  .scale-box {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 14px 18px; margin-bottom: 24px; }}
  .scale-box h2 {{ font-size: 0.95rem; margin-bottom: 8px; color: #444; }}
  .scale-box table {{ border-collapse: collapse; font-size: 0.85rem; }}
  .scale-box td {{ padding: 2px 12px 2px 0; }}
  .score-val {{ font-weight: 700; text-align: right; font-variant-numeric: tabular-nums; }}
  .progress-bar-wrap {{ background: #e0e0e0; border-radius: 4px; height: 8px; margin-bottom: 20px; }}
  .progress-bar {{ background: #4caf50; height: 8px; border-radius: 4px; width: 0%; transition: width 0.2s; }}
  .progress-text {{ font-size: 0.82rem; color: #666; margin-bottom: 6px; }}
  .card {{ background: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 18px 20px; margin-bottom: 16px; }}
  .card.answered {{ border-color: #4caf50; }}
  .card-header {{ margin-bottom: 10px; font-size: 0.82rem; font-weight: 600; color: #555; }}
  .label {{ font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #888; margin-top: 10px; margin-bottom: 3px; }}
  .qa-text {{ font-size: 0.93rem; white-space: pre-wrap; background: rgba(0,0,0,0.03); border-radius: 5px; padding: 8px 10px; }}
  .score-row {{ display: flex; align-items: center; flex-wrap: wrap; gap: 6px; margin-top: 14px; }}
  .score-prompt {{ font-size: 0.85rem; font-weight: 600; margin-right: 4px; }}
  .score-btn input {{ display: none; }}
  .score-btn span {{ display: inline-block; padding: 5px 10px; border-radius: 5px; border: 1.5px solid #ccc; cursor: pointer; font-size: 0.82rem; font-weight: 600; user-select: none; }}
  .score-btn input:checked + span {{ background: #1976d2; color: #fff; border-color: #1976d2; }}
  .progress-note {{ font-size: 0.78rem; color: #4caf50; margin-top: 6px; min-height: 1em; }}
  .submit-bar {{ position: fixed; bottom: 0; left: 0; right: 0; background: #fff; border-top: 1px solid #ddd; padding: 12px 24px; display: flex; align-items: center; gap: 16px; }}
  .submit-bar button {{ padding: 10px 28px; background: #1976d2; color: #fff; border: none; border-radius: 6px; font-size: 1rem; cursor: pointer; font-weight: 600; }}
  .submit-bar button:disabled {{ background: #aaa; cursor: not-allowed; }}
  .submit-status {{ font-size: 0.9rem; color: #666; }}
</style>
</head>
<body>
<div class='page'>
  <h1>Neuroticism v2 human rating</h1>
  <p class='subtitle'>Rater: <strong>{_escape_html(rater)}</strong> · {len(items)} items</p>
  <div class='scale-box'>
    <h2>Score scale</h2>
    <table>{scale_rows}</table>
    <p style='margin-top:10px;font-size:0.82rem;color:#666;'>Score only what is present in the response. Hidden labels and author expectations are intentionally omitted.</p>
  </div>
  <div class='progress-text' id='progress-text'>0 / {len(items)} scored</div>
  <div class='progress-bar-wrap'><div class='progress-bar' id='progress-bar'></div></div>
  {''.join(cards)}
</div>
<div class='submit-bar'>
  <button id='download-button' disabled>Download CSV</button>
  <span class='submit-status' id='submit-status'>Score all items to enable download.</span>
</div>
<script>
const ITEMS = {items_json};
const RATER = {json.dumps(rater)};
function selectedScore(index) {{
  const checked = document.querySelector("input[name='score_" + index + "']:checked");
  return checked ? parseInt(checked.value, 10) : null;
}}
function updateProgress() {{
  let completed = 0;
  for (let index = 0; index < ITEMS.length; index += 1) {{
    if (selectedScore(index) != null) {{
      completed += 1;
    }}
  }}
  const percent = (completed / ITEMS.length) * 100;
  document.getElementById('progress-text').textContent = completed + ' / ' + ITEMS.length + ' scored';
  document.getElementById('progress-bar').style.width = percent.toFixed(1) + '%';
  const done = completed === ITEMS.length;
  document.getElementById('download-button').disabled = done ? false : true;
  document.getElementById('submit-status').textContent = done ? 'All items scored and ready to download.' : (ITEMS.length - completed) + ' item(s) remaining.';
}}
for (let index = 0; index < ITEMS.length; index += 1) {{
  const inputs = document.querySelectorAll("input[name='score_" + index + "']");
  inputs.forEach((input) => {{
    input.addEventListener('change', () => {{
      document.getElementById('card-' + index).classList.add('answered');
      document.getElementById('note-' + index).textContent = 'scored';
      updateProgress();
    }});
  }});
}}
document.getElementById('download-button').addEventListener('click', (event) => {{
  event.preventDefault();
  const rows = [['id', 'trait', 'split', 'question', 'response', 'score_' + RATER]];
  ITEMS.forEach((item, index) => {{
    rows.push([item.id, item.trait, item.split, item.question, item.response, selectedScore(index)]);
  }});
  const csv = rows.map((row) => row.map((value) => '"' + String(value).replace(/"/g, '""') + '"').join(',')).join('\n');
  const blob = new Blob([csv], {{ type: 'text/csv' }});
  const anchor = document.createElement('a');
  anchor.href = URL.createObjectURL(blob);
  anchor.download = 'neuroticism_v2_' + RATER + '_ratings.csv';
  anchor.click();
}});
updateProgress();
</script>
</body>
</html>
"""
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a blinded HTML rating form for neuroticism_v2.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=["dev", "heldout"], default="heldout")
    parser.add_argument("--rater", required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    items = filter_items(load_calibration_items(args.dataset), split=args.split)
    seed_material = f"{args.rater}:{args.split}:{args.dataset}"
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    shuffled_items = list(items)
    random.Random(seed).shuffle(shuffled_items)
    payload = [{"id": item.id, "trait": item.trait, "split": item.split, "question": item.question, "response": item.response} for item in shuffled_items]
    output_path = args.output or Path("scratch/rating") / f"neuroticism_v2_{args.split}_{args.rater}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(payload, args.rater), encoding="utf-8")
    print(f"Wrote {output_path} with {len(payload)} items.")


if __name__ == "__main__":
    main()
