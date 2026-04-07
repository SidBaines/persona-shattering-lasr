#!/usr/bin/env python3
"""Generate mobile-friendly HTML annotation interfaces for golden calibration sets.

Reads golden JSONL files from ``data/judge_calibration/`` and generates
self-contained HTML files that let human raters score each item. Results
are downloaded as JSON.

Usage::

    # Generate one HTML per trait + a combined "all" file
    uv run python scripts_dev/persona_metrics/llm_judge/generate_annotation_html.py

    # Single trait
    uv run python scripts_dev/persona_metrics/llm_judge/generate_annotation_html.py \
        --trait agreeableness

    # Custom output dir and rater name
    uv run python scripts_dev/persona_metrics/llm_judge/generate_annotation_html.py \
        --output-dir scratch/annotation_html --rater "alice"

    # Specific traits
    uv run python scripts_dev/persona_metrics/llm_judge/generate_annotation_html.py \
        --trait coherence --trait neuroticism
"""

from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_DATA_DIR = PROJECT_ROOT / "data" / "judge_calibration"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "scratch" / "annotation_html"

SCORE_RANGES: dict[str, tuple[int, int]] = {
    "neuroticism": (-4, 4),
    "openness": (-4, 4),
    "conscientiousness": (-4, 4),
    "extraversion": (-4, 4),
    "agreeableness": (-4, 4),
    "coherence": (0, 10),
}

OCEAN_SCALE_TABLE = """\
<table><tr><td class="score-val">-4</td><td>Extreme low</td></tr>
<tr><td class="score-val">-3</td><td>Strong low</td></tr>
<tr><td class="score-val">-2</td><td>Moderate low</td></tr>
<tr><td class="score-val">-1</td><td>Slight low</td></tr>
<tr><td class="score-val">+0</td><td>Neutral / no signal</td></tr>
<tr><td class="score-val">+1</td><td>Slight high</td></tr>
<tr><td class="score-val">+2</td><td>Moderate high</td></tr>
<tr><td class="score-val">+3</td><td>Strong high</td></tr>
<tr><td class="score-val">+4</td><td>Extreme high</td></tr></table>
<p class="scale-note">
  Score only the <strong>style and framing</strong> of the Response for the labelled trait.<br>
  Factual correctness and politeness are <strong>not</strong> trait signals.
</p>"""

COHERENCE_SCALE_TABLE = """\
<table><tr><td class="score-val">10</td><td>Perfect — flawless logical structure</td></tr>
<tr><td class="score-val">9</td><td>Excellent — very minor looseness</td></tr>
<tr><td class="score-val">8</td><td>Strong — minor structural issues</td></tr>
<tr><td class="score-val">7</td><td>Good — noticeable looseness</td></tr>
<tr><td class="score-val">6</td><td>Adequate — understandable but weak</td></tr>
<tr><td class="score-val">5</td><td>Mixed — half coherent, half problematic</td></tr>
<tr><td class="score-val">4</td><td>Weak — intelligible but badly structured</td></tr>
<tr><td class="score-val">3</td><td>Poor — disorganised, facts without connection</td></tr>
<tr><td class="score-val">2</td><td>Very poor — broken grammar, barely communicative</td></tr>
<tr><td class="score-val">1</td><td>Near-total failure — repetition loops, word salad</td></tr>
<tr><td class="score-val">0</td><td>Complete failure — pure gibberish</td></tr></table>
<p class="scale-note">
  Score the <strong>entire response</strong>, not just the beginning.<br>
  Generation artifacts (repetition loops) are <strong>more severe</strong> than poor writing.
</p>"""


def load_golden(trait: str) -> list[dict]:
    """Load golden items for one trait from JSONL."""
    path = GOLDEN_DATA_DIR / f"{trait}.jsonl"
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def _esc(text: str) -> str:
    """HTML-escape text."""
    return html.escape(text, quote=True)


def generate_html(
    items: list[dict],
    traits: list[str],
    rater: str,
    seed: int = 42,
) -> str:
    """Generate self-contained annotation HTML for a list of golden items.

    Args:
        items: List of golden item dicts (id, trait, question, response, gold_score).
        traits: List of trait names included (for title/subtitle).
        rater: Rater name to display and embed in output.
        seed: Random seed for shuffling item order.

    Returns:
        Complete HTML string.
    """
    # Shuffle items so raters don't see gold-score ordering
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)

    n = len(items)
    trait_list = ", ".join(t.capitalize() for t in traits)
    title = f"Golden calibration — {trait_list}"

    # Determine if mixed scales
    all_score_ranges = {SCORE_RANGES.get(item["trait"], (-4, 4)) for item in items}
    is_mixed = len(all_score_ranges) > 1

    # Build scale reference box
    has_ocean = any(item["trait"] in SCORE_RANGES and SCORE_RANGES[item["trait"]] == (-4, 4) for item in items)
    has_coherence = any(item["trait"] == "coherence" for item in items)

    scale_html_parts = []
    if has_ocean:
        scale_html_parts.append(
            f'<h2>OCEAN scale (-4 to +4)</h2>\n{OCEAN_SCALE_TABLE}'
        )
    if has_coherence:
        scale_html_parts.append(
            f'<h2>Coherence scale (0 to 10)</h2>\n{COHERENCE_SCALE_TABLE}'
        )
    scale_html = "\n".join(scale_html_parts)

    # Build cards
    cards_html = []
    for idx, item in enumerate(items):
        trait = item["trait"]
        score_min, score_max = SCORE_RANGES.get(trait, (-4, 4))
        trait_label = trait.capitalize()

        # Build score buttons
        score_buttons = []
        for s in range(score_min, score_max + 1):
            display = f"+{s}" if s > 0 and score_min < 0 else str(s)
            score_buttons.append(
                f'<label class="score-btn">'
                f'<input type="radio" name="score_{idx}" value="{s}" required>'
                f'<span>{display}</span></label>'
            )
        buttons_html = "".join(score_buttons)

        card = f"""\
    <div class="card" id="card-{idx}">
      <div class="card-header">
        <span class="item-num">Item {idx + 1} of {n}</span>
        <span class="trait-tag">{_esc(trait_label)}</span>
      </div>
      <div class="qa-block">
        <div class="label">Question</div>
        <div class="qa-text">{_esc(item["question"])}</div>
        <div class="label">Response</div>
        <div class="qa-text">{_esc(item["response"])}</div>
      </div>
      <div class="score-row">
        <span class="score-prompt">Your score ({_esc(trait_label)}):</span>
        {buttons_html}
      </div>
      <div class="progress-note" id="note-{idx}"></div>
    </div>"""
        cards_html.append(card)

    cards_block = "\n".join(cards_html)

    # Embed item metadata as JSON for the download
    items_meta = json.dumps(
        [{"id": item["id"], "trait": item["trait"]} for item in items],
        ensure_ascii=False,
    )

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f5f5f5; color: #222; line-height: 1.5; }}
  .page {{ max-width: 860px; margin: 0 auto; padding: 24px 16px 80px; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-bottom: 20px; font-size: 0.9rem; }}
  .scale-box {{ background: #fff; border: 1px solid #ddd; border-radius: 8px;
               padding: 14px 18px; margin-bottom: 24px; }}
  .scale-box h2 {{ font-size: 0.95rem; margin-bottom: 8px; margin-top: 12px; color: #444; }}
  .scale-box h2:first-child {{ margin-top: 0; }}
  .scale-box table {{ border-collapse: collapse; font-size: 0.85rem; }}
  .scale-box td {{ padding: 2px 12px 2px 0; }}
  .score-val {{ font-weight: 700; text-align: right; font-variant-numeric: tabular-nums;
               min-width: 2em; }}
  .scale-note {{ margin-top: 10px; font-size: 0.82rem; color: #666; }}
  .progress-bar-wrap {{ background: #e0e0e0; border-radius: 4px; height: 8px; margin-bottom: 20px; }}
  .progress-bar {{ background: #4caf50; height: 8px; border-radius: 4px; width: 0%;
                   transition: width 0.3s; }}
  .progress-text {{ font-size: 0.82rem; color: #666; margin-bottom: 6px; }}
  .card {{ background: #fff; border: 1px solid #ddd; border-radius: 10px;
           padding: 18px 20px; margin-bottom: 16px; transition: border-color 0.2s; }}
  .card.answered {{ border-color: #4caf50; }}
  .card-header {{ margin-bottom: 10px; font-size: 0.82rem; display: flex;
                  align-items: center; gap: 8px; }}
  .item-num {{ font-weight: 600; color: #555; }}
  .trait-tag {{ display: inline-block; background: #e8f0fe; color: #1565c0;
               border-radius: 4px; padding: 1px 8px; font-size: 0.75rem;
               font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; }}
  .label {{ font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.05em; color: #888; margin-top: 10px; margin-bottom: 3px; }}
  .qa-text {{ font-size: 0.93rem; white-space: pre-wrap; background: rgba(0,0,0,0.03);
              border-radius: 5px; padding: 8px 10px; }}
  .score-row {{ display: flex; align-items: center; flex-wrap: wrap; gap: 6px;
               margin-top: 14px; }}
  .score-prompt {{ font-size: 0.85rem; font-weight: 600; margin-right: 4px;
                   flex-shrink: 0; }}
  .score-btn input {{ display: none; }}
  .score-btn span {{ display: inline-block; padding: 5px 10px; border-radius: 5px;
                     border: 1.5px solid #ccc; cursor: pointer; font-size: 0.82rem;
                     font-weight: 600; transition: all 0.15s; user-select: none;
                     min-width: 2.2em; text-align: center; }}
  .score-btn input:checked + span {{ background: #1976d2; color: #fff;
                                     border-color: #1976d2; }}
  .score-btn span:hover {{ border-color: #1976d2; background: #e3f0fb; }}
  .progress-note {{ font-size: 0.78rem; color: #4caf50; margin-top: 6px;
                    min-height: 1em; }}
  .submit-bar {{ position: fixed; bottom: 0; left: 0; right: 0;
                 background: #fff; border-top: 1px solid #ddd;
                 padding: 12px 24px; display: flex; align-items: center; gap: 16px;
                 z-index: 100; }}
  .submit-bar button {{ padding: 10px 28px; background: #1976d2; color: #fff;
                        border: none; border-radius: 6px; font-size: 1rem;
                        cursor: pointer; font-weight: 600; }}
  .submit-bar button:disabled {{ background: #aaa; cursor: not-allowed; }}
  .submit-bar button:not(:disabled):hover {{ background: #1565c0; }}
  .submit-status {{ font-size: 0.9rem; color: #666; }}
  /* Mobile: make score buttons a bit larger for touch */
  @media (max-width: 600px) {{
    .score-btn span {{ padding: 8px 10px; font-size: 0.9rem; min-width: 2.5em; }}
    .score-row {{ gap: 4px; }}
    .score-prompt {{ width: 100%; margin-bottom: 4px; }}
    .page {{ padding: 16px 10px 90px; }}
    .card {{ padding: 14px 14px; }}
  }}
</style>
</head>
<body>
<div class="page">
  <h1>{_esc(title)}</h1>
  <p class="subtitle">Rater: <strong>{_esc(rater)}</strong> &nbsp;&middot;&nbsp; {n} items</p>

  <div class="scale-box">
    {scale_html}
  </div>

  <div class="progress-text" id="prog-text">0 / {n} scored</div>
  <div class="progress-bar-wrap"><div class="progress-bar" id="prog-bar"></div></div>

  <form id="rating-form">
{cards_block}
  </form>
</div>

<div class="submit-bar">
  <button type="button" id="download-btn" disabled>Download scores</button>
  <span class="submit-status" id="submit-status"></span>
</div>

<script>
(function() {{
  const TOTAL = {n};
  const RATER = {json.dumps(rater)};
  const ITEMS = {items_meta};

  const form = document.getElementById('rating-form');
  const progText = document.getElementById('prog-text');
  const progBar = document.getElementById('prog-bar');
  const dlBtn = document.getElementById('download-btn');
  const status = document.getElementById('submit-status');

  // Restore saved state from localStorage
  const STORAGE_KEY = 'annotation_' + RATER + '_' + ITEMS.map(i => i.id).join(',').substring(0, 60);

  function saveState() {{
    const state = {{}};
    for (let i = 0; i < TOTAL; i++) {{
      const checked = form.querySelector('input[name="score_' + i + '"]:checked');
      if (checked) state[i] = parseInt(checked.value, 10);
    }}
    try {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); }} catch(e) {{}}
  }}

  function restoreState() {{
    try {{
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const state = JSON.parse(raw);
      for (const [idx, val] of Object.entries(state)) {{
        const input = form.querySelector('input[name="score_' + idx + '"][value="' + val + '"]');
        if (input) {{
          input.checked = true;
          const card = document.getElementById('card-' + idx);
          if (card) card.classList.add('answered');
        }}
      }}
      updateProgress();
    }} catch(e) {{}}
  }}

  function countAnswered() {{
    let n = 0;
    for (let i = 0; i < TOTAL; i++) {{
      if (form.querySelector('input[name="score_' + i + '"]:checked')) n++;
    }}
    return n;
  }}

  function updateProgress() {{
    const done = countAnswered();
    progText.textContent = done + ' / ' + TOTAL + ' scored';
    progBar.style.width = (done / TOTAL * 100) + '%';
    dlBtn.disabled = done < TOTAL;
    if (done === TOTAL) {{
      status.textContent = 'All items scored. Ready to download.';
    }} else {{
      status.textContent = '';
    }}
  }}

  form.addEventListener('change', function(e) {{
    if (e.target.name && e.target.name.startsWith('score_')) {{
      const idx = e.target.name.split('_')[1];
      const card = document.getElementById('card-' + idx);
      if (card) card.classList.add('answered');
      const note = document.getElementById('note-' + idx);
      if (note) note.textContent = 'Scored';
      updateProgress();
      saveState();
    }}
  }});

  dlBtn.addEventListener('click', function() {{
    const results = {{
      rater: RATER,
      timestamp: new Date().toISOString(),
      n_items: TOTAL,
      scores: []
    }};
    for (let i = 0; i < TOTAL; i++) {{
      const checked = form.querySelector('input[name="score_' + i + '"]:checked');
      results.scores.push({{
        id: ITEMS[i].id,
        trait: ITEMS[i].trait,
        score: checked ? parseInt(checked.value, 10) : null
      }});
    }}
    const blob = new Blob([JSON.stringify(results, null, 2)],
                          {{type: 'application/json'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'golden_scores_' + RATER + '_' + new Date().toISOString().slice(0,10) + '.json';
    a.click();
    URL.revokeObjectURL(url);
    status.textContent = 'Downloaded!';
  }});

  restoreState();
  updateProgress();
}})();
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate HTML annotation interfaces for golden calibration sets.",
    )
    parser.add_argument(
        "--trait",
        action="append",
        dest="traits",
        choices=list(SCORE_RANGES.keys()),
        help="Trait(s) to include. Can be repeated. Default: all.",
    )
    parser.add_argument(
        "--rater",
        default="rater",
        help="Rater name to embed in the HTML (default: 'rater').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling item order (default: 42).",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Only generate the combined all-traits file, skip per-trait files.",
    )
    args = parser.parse_args()

    traits = args.traits or list(SCORE_RANGES.keys())
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all items
    all_items: list[dict] = []
    trait_items: dict[str, list[dict]] = {}
    for trait in traits:
        items = load_golden(trait)
        trait_items[trait] = items
        all_items.extend(items)
        print(f"  {trait}: {len(items)} items")

    # Generate per-trait HTMLs
    if not args.combined_only:
        for trait in traits:
            html_content = generate_html(
                trait_items[trait],
                traits=[trait],
                rater=args.rater,
                seed=args.seed,
            )
            out_path = output_dir / f"annotate_{trait}_{args.rater}.html"
            out_path.write_text(html_content, encoding="utf-8")
            print(f"  -> {out_path.relative_to(PROJECT_ROOT)}")

    # Generate combined HTML
    if len(traits) > 1:
        html_content = generate_html(
            all_items,
            traits=traits,
            rater=args.rater,
            seed=args.seed,
        )
        out_path = output_dir / f"annotate_all_{args.rater}.html"
        out_path.write_text(html_content, encoding="utf-8")
        print(f"  -> {out_path.relative_to(PROJECT_ROOT)}")

    print(f"\nDone. {len(all_items)} total items across {len(traits)} trait(s).")


if __name__ == "__main__":
    main()
