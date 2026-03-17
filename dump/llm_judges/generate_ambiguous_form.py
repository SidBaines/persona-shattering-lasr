#!/usr/bin/env python3
"""Generate a human-rating form for contested/ambiguous heldout items across all OCEAN traits.

Pulls the latest calibration results for each trait, identifies items where:
  - the item is a confound (expected_score=0 but judges scored non-zero), OR
  - the two judge models disagreed by ≥ 2 points, OR
  - both models diverged from expected by ≥ avg_delta threshold

Outputs a single cross-trait HTML form where each card shows the trait label,
so the rater knows which trait they are scoring.

Usage:
    uv run python dump/llm_judges/generate_ambiguous_form.py \\
        --rater irakli \\
        --output scratch/rating/ambiguous_irakli.html

    # Open in browser, score, download CSV.
    # The CSV can then be passed to calibrate.py --human-scores (per trait).
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
TRAITS = ["neuroticism", "agreeableness", "conscientiousness", "extraversion", "openness"]

SCALE_LABELS = [
    (-4, "Extreme low"),
    (-3, "Strong low"),
    (-2, "Moderate low"),
    (-1, "Slight low"),
    ( 0, "Neutral / no signal"),
    (+1, "Slight high"),
    (+2, "Moderate high"),
    (+3, "Strong high"),
    (+4, "Extreme high"),
]


def latest_results_dir(trait: str) -> Path | None:
    results_dir = JUDGES_DIR / "ocean" / trait / "results"
    if not results_dir.exists():
        return None
    runs = sorted(results_dir.iterdir())
    return runs[-1] if runs else None


def load_heldout(trait: str) -> list[dict]:
    path = JUDGES_DIR / "ocean" / trait / "heldout.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_model_scores(results_dir: Path) -> dict[str, dict[str, int]]:
    """Return {model_stem: {item_id: score}}."""
    out = {}
    for f in sorted(results_dir.glob("*.jsonl")):
        scores = {}
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                scores[r["id"]] = r["score"]
        out[f.stem] = scores
    return out


def find_contested(trait: str, avg_delta_thresh: float = 1.5, disagree_thresh: int = 2) -> list[dict]:
    """Return heldout items for this trait that are contested."""
    heldout = load_heldout(trait)
    results_dir = latest_results_dir(trait)
    if results_dir is None:
        return []

    model_scores = load_model_scores(results_dir)
    models = list(model_scores.keys())
    if len(models) < 2:
        return []
    m1, m2 = models[0], models[1]

    contested = []
    for item in heldout:
        iid = item["id"]
        s1 = model_scores[m1].get(iid)
        s2 = model_scores[m2].get(iid)
        if s1 is None or s2 is None:
            continue
        exp = item["expected_score"]
        is_confound = item["category"].startswith("confound")
        avg_delta = abs(((s1 + s2) / 2) - exp)
        model_disagree = abs(s1 - s2)

        if is_confound and avg_delta > 0:
            contested.append({**item, "trait": trait, "judge_scores": {m1: s1, m2: s2}})
        elif model_disagree >= disagree_thresh or avg_delta >= avg_delta_thresh:
            contested.append({**item, "trait": trait, "judge_scores": {m1: s1, m2: s2}})

    return contested


def render_html(items: list[dict], rater: str) -> str:
    shuffled = items[:]
    random.Random(rater).shuffle(shuffled)

    items_for_js = [
        {"id": it["id"], "question": it["question"], "response": it["response"], "trait": it["trait"]}
        for it in shuffled
    ]
    items_json = json.dumps(items_for_js)

    scale_rows = "\n".join(
        f'<tr><td class="score-val">{v:+d}</td><td>{label}</td></tr>'
        for v, label in SCALE_LABELS
    )

    item_cards = ""
    for i, item in enumerate(shuffled):
        trait_display = item["trait"].replace("_", " ").title()
        item_cards += f"""
        <div class="card" id="card-{i}">
          <div class="card-header">
            <span class="item-num">Item {i + 1} of {len(shuffled)}</span>
            <span class="trait-tag">{trait_display}</span>
          </div>
          <div class="qa-block">
            <div class="label">Question</div>
            <div class="qa-text">{_esc(item['question'])}</div>
            <div class="label">Response</div>
            <div class="qa-text">{_esc(item['response'])}</div>
          </div>
          <div class="score-row">
            <span class="score-prompt">Your score ({trait_display}):</span>
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
<title>OCEAN ambiguous items — {rater}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f5f5f5; color: #222; line-height: 1.5; }}
  .page {{ max-width: 860px; margin: 0 auto; padding: 24px 16px 80px; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-bottom: 20px; font-size: 0.9rem; }}
  .scale-box {{ background: #fff; border: 1px solid #ddd; border-radius: 8px;
               padding: 14px 18px; margin-bottom: 24px; }}
  .scale-box h2 {{ font-size: 0.95rem; margin-bottom: 8px; color: #444; }}
  .scale-box table {{ border-collapse: collapse; font-size: 0.85rem; }}
  .scale-box td {{ padding: 2px 12px 2px 0; }}
  .score-val {{ font-weight: 700; text-align: right; font-variant-numeric: tabular-nums; }}
  .progress-bar-wrap {{ background: #e0e0e0; border-radius: 4px; height: 8px; margin-bottom: 20px; }}
  .progress-bar {{ background: #4caf50; height: 8px; border-radius: 4px; width: 0%; transition: width 0.3s; }}
  .progress-text {{ font-size: 0.82rem; color: #666; margin-bottom: 6px; }}
  .card {{ background: #fff; border: 1px solid #ddd; border-radius: 10px;
           padding: 18px 20px; margin-bottom: 16px; transition: border-color 0.2s; }}
  .card.answered {{ border-color: #4caf50; }}
  .card-header {{ margin-bottom: 10px; font-size: 0.82rem; display: flex; align-items: center; gap: 8px; }}
  .item-num {{ font-weight: 600; color: #555; }}
  .trait-tag {{ display: inline-block; background: #e8f0fe; color: #1565c0; border-radius: 4px;
                padding: 1px 8px; font-size: 0.75rem; font-weight: 700;
                text-transform: uppercase; letter-spacing: 0.04em; }}
  .label {{ font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.05em; color: #888; margin-top: 10px; margin-bottom: 3px; }}
  .qa-text {{ font-size: 0.93rem; white-space: pre-wrap; background: rgba(0,0,0,0.03);
              border-radius: 5px; padding: 8px 10px; }}
  .score-row {{ display: flex; align-items: center; flex-wrap: wrap; gap: 6px; margin-top: 14px; }}
  .score-prompt {{ font-size: 0.85rem; font-weight: 600; margin-right: 4px; }}
  .score-btn input {{ display: none; }}
  .score-btn span {{ display: inline-block; padding: 5px 10px; border-radius: 5px;
                     border: 1.5px solid #ccc; cursor: pointer; font-size: 0.82rem;
                     font-weight: 600; transition: all 0.15s; user-select: none; }}
  .score-btn input:checked + span {{ background: #1976d2; color: #fff; border-color: #1976d2; }}
  .score-btn span:hover {{ border-color: #1976d2; background: #e3f0fb; }}
  .progress-note {{ font-size: 0.78rem; color: #4caf50; margin-top: 6px; min-height: 1em; }}
  .submit-bar {{ position: fixed; bottom: 0; left: 0; right: 0;
                 background: #fff; border-top: 1px solid #ddd;
                 padding: 12px 24px; display: flex; align-items: center; gap: 16px; }}
  .submit-bar button {{ padding: 10px 28px; background: #1976d2; color: #fff;
                        border: none; border-radius: 6px; font-size: 1rem;
                        cursor: pointer; font-weight: 600; }}
  .submit-bar button:disabled {{ background: #aaa; cursor: not-allowed; }}
  .submit-bar button:not(:disabled):hover {{ background: #1565c0; }}
  .submit-status {{ font-size: 0.9rem; color: #666; }}
</style>
</head>
<body>
<div class="page">
  <h1>OCEAN ambiguous items</h1>
  <p class="subtitle">Rater: <strong>{rater}</strong> &nbsp;·&nbsp; {len(items)} contested items across all traits</p>

  <div class="scale-box">
    <h2>Score scale (−4 … +4)</h2>
    <table>{scale_rows}</table>
    <p style="margin-top:10px;font-size:0.82rem;color:#666;">
      Score only the <strong>style and framing</strong> of the Response for the labelled trait.<br>
      The trait label tells you which OCEAN dimension to assess.<br>
      Factual correctness and politeness are <strong>not</strong> trait signals.
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
  const rows = [["id", "trait", "question", "response", `score_${{RATER}}`]];
  ITEMS.forEach((item, i) => {{
    rows.push([item.id, item.trait, item.question, item.response, scores[i]]);
  }});
  const csv = rows.map(r => r.map(v => `"${{String(v).replace(/"/g, '""')}}"`).join(",")).join("\\n");
  const blob = new Blob([csv], {{type: "text/csv"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `ocean_ambiguous_${{RATER}}_ratings.csv`;
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
        description="Generate a cross-trait HTML rating form for contested/ambiguous heldout items."
    )
    parser.add_argument("--rater", required=True, help="Rater name.")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path.")
    parser.add_argument("--avg-delta", type=float, default=2.0,
                        help="Min average model-vs-expected delta to flag a non-confound item (default 2.0).")
    parser.add_argument("--disagree", type=int, default=3,
                        help="Min inter-model disagreement to flag a non-confound item (default 3).")
    args = parser.parse_args()

    all_items: list[dict] = []
    for trait in TRAITS:
        contested = find_contested(trait, avg_delta_thresh=args.avg_delta, disagree_thresh=args.disagree)
        all_items.extend(contested)
        print(f"  {trait:20} {len(contested)} contested items")

    print(f"\nTotal: {len(all_items)} items across all traits")

    if not all_items:
        print("No contested items found. Run calibrate.py --save first.")
        sys.exit(1)

    html = render_html(all_items, args.rater)

    output_path = (
        Path(args.output)
        if args.output
        else JUDGES_DIR / "ratings" / f"ambiguous_{args.rater}.html"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"\nGenerated {output_path}")
    print(f"Open in browser, score {len(all_items)} items, click Download CSV.")


if __name__ == "__main__":
    main()
