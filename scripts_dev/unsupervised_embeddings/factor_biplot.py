"""Standalone 2D factor biplot for the paper's unsupervised section.

Plots each baseline-model persona-rollout's Thomson factor scores on the
top-two-variance plane (F0=Initiative on x, F1=Tone on y) from the v7-pf3
$k=4$ oblimin fit on Llama-3.1-8B-Instruct, with the highest-|loading|
items overlaid as labelled arrows projecting onto the same plane.

This is an iteration sandbox — kept deliberately self-contained so we
can change layout / styling fast without touching the main pipeline.
The fit it reads is produced by

    scripts_dev/unsupervised_embeddings/analysis_for_paper.v2.py --k 4

so re-run that first if scratch/psychometric_fa_paper_v7pf3_k4/ is empty.

Run with:
    uv run python scripts_dev/unsupervised_embeddings/factor_biplot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Config (edit me) ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIT_NPZ = PROJECT_ROOT / (
    "scratch/psychometric_fa_paper_v7pf3_k4/llama-3.1-8b/"
    "factor_analysis/raw/fa_4_principal_oblimin.npz"
)
ITEM_LABELS_JSON = PROJECT_ROOT / (
    "scratch/psychometric_fa_paper_v7pf3_k4/llama-3.1-8b/"
    "factor_analysis/raw/fa_4_principal_oblimin_item_labels.json"
)

OUT_DIR = PROJECT_ROOT / "scratch/factor_biplot"
OUT_PATH = OUT_DIR / "biplot_initiative_tone.png"

# Which two factors to plot (column indices in the npz, in
# canonical-variance-sorted order: 0=Initiative, 1=Tone, 2=Didacticism,
# 3=Epistemic Caution).
FX, FY = 0, 1
FACTOR_NAMES = ["Initiative", "Tone", "Didacticism", "Epistemic Caution"]

# How many highest-|loading| items to show as arrows (per factor we plot,
# union'd). Keep small to avoid clutter.
N_TOP_ITEMS_PER_FACTOR = 0
# Arrow scaling: arrow length = ARROW_SCALE × loading on that axis.
ARROW_SCALE = 3.0

# ── Exemplar dumping ────────────────────────────────────────────────────────
# After plotting, also dump the most-extreme baseline rollouts on each
# single-axis pole (e.g. high F0 with F1≈0) into per-axis text files.
# This is the data upstream agents use to inspect what behaviour the
# axes actually express. Set DUMP_EXEMPLARS=False to skip.
DUMP_EXEMPLARS: bool = True
# Directory holding the rollout dataset that produced the questionnaire
# scores. canonical_samples.jsonl in here keys conversations by sample_id.
ROLLOUT_DIR = PROJECT_ROOT / (
    "scratch/psychometric_fa/"
    "rollouts-llama318binstruct-t1.0-15t-2500p-seed436-"
    "scenarios_v2-uprompt_v6"
)
QUESTIONNAIRE_METADATA = PROJECT_ROOT / (
    "scratch/psychometric_fa_paper_v7pf3_k4/llama-3.1-8b/"
    "questionnaire/metadata.jsonl"
)
N_EXEMPLARS_PER_AXIS = 5
# Require the *other* axis (the one we don't want extreme on) to lie
# within ±OTHER_AXIS_BAND. Smaller = stricter "single-axis" purity.
OTHER_AXIS_BAND = 0.25

# ── Continuation generation ─────────────────────────────────────────────────
# After picking exemplars, optionally append a fixed user prompt to each
# selected rollout and generate one additional assistant turn (per prompt,
# per exemplar). Results are cached locally so the plot can be re-rendered
# without re-running inference. Cache lives in OUT_DIR/continuation_cache/.
GENERATE_CONTINUATIONS: bool = True
CONTINUATION_MODEL: str = "meta-llama/Llama-3.1-8B-Instruct"
CONTINUATION_MAX_NEW_TOKENS: int = 512
CONTINUATION_TEMPERATURE: float = 1.0
CONTINUATION_TOP_P: float = 1.0
CONTINUATION_SEED: int = 436
CONTINUATION_BATCH_SIZE: int = 4
# Probe prompts appended to each exemplar rollout. Pick prompts that are
# likely to differentiate Initiative (proactive vs passive) and Tone
# (warm vs cold) — see brainstorm in the PR notes.
CONTINUATION_USER_PROMPTS: list[str] = [
    # Mild-ask topic-change with logistics — both axes should be visible.
    "I've been thinking about adopting a cat but I travel a fair bit for work.",
    # Pure disclosure with no implicit ask — easier for a low-initiative
    # persona to just acknowledge instead of advising.
    "I've been struggling to fall asleep the last few nights.",
    # Closing cue — does the model accept the close (low init) or keep
    # volunteering (high init)?
    "Anyway, that's basically all I needed — thanks for the help.",
    # Pure observation, no ask — does the model extend it (high init) or
    # just acknowledge (low init)?
    "It's been raining here every day this week. Kind of grim, honestly.",
    # Brief factual question — does the model answer narrowly (low init)
    # or add caveats / extra context / follow-up (high init)?
    "Random question — roughly how much water should you drink in a day?",
    # Emotional disclosure, no ask — second tone probe with no advice
    # solicitation, to separate warmth from advice-mode.
    "Just had a pretty rough day at work.",
    # Buried-signal probes — surface is a task ask, but there's an implicit
    # personal/health/distress cue underneath. A high-initiative model
    # should notice and ask; a low-initiative model should answer the
    # surface task and ignore the cue.
    "Quick Excel question — what's the formula to sum column A only if column B says 'Yes'? Sorry the brain is a bit foggy, hardly slept the last two nights since mum's been in hospital, but the formula's the main thing.",
    "Could you walk me through how to file my taxes? Also kind of unrelated but my left hand has been going numb on and off the last two days.",
    "Lol I think I just got fired today. Anyway, any good recipes for dinner I can make with chicken thighs and not much else?",
    # Lead-invitation: F0+ should propose concrete next directions;
    # F0- should defer back to the user.
    "What should we tackle next?",
    # Continue-or-wrap fork: F0+ should push to continue / name a
    # direction; F0- should accept the user's framing.
    "I'm pretty happy with where we got to. Want to keep going or wrap up?",
    # Tangent-invitation: F0+ should answer briefly and pivot back to the
    # prior topic; F0- should engage with the tangent on its own terms.
    "Quick aside before we keep going — what's a movie you'd pick for a free Saturday?",
    # Vague non-ask: F0+ should propose a structured way forward; F0-
    # should mirror / sit with the feeling.
    "I'm kinda stuck. Don't really know what I want from this conversation right now.",
]
CONTINUATION_CACHE_DIR_NAME = "continuation_cache"

# ── Plot callouts ───────────────────────────────────────────────────────────
# One exemplar per pole, hand-picked from the cached continuations as the
# clearest single-rollout illustration of that pole's behavior. Each callout
# circles the rollout's scatter point in red and draws a grey speech bubble
# with a short summary of the assistant's behavior; the full response is
# kept inline as `response` for record-keeping.
EXEMPLAR_CALLOUTS: list[dict] = [
    {
        # F0+ : project anchoring — answers the off-topic ask, then yanks
        # the conversation back to the prior task.
        "pole": "F0+",
        "sample_id": "sample_12805658934fab0775679eed",
        "probe": "Quick aside before we keep going — what's a movie you'd "
                 "pick for a free Saturday?",
        "summary": (
            "F0+  (project anchoring)\n"
            "Asked for a movie pick →\n"
            "“…Shawshank Redemption.\n"
            " Now, back to the tank troubles!”"
        ),
        "xytext": (3.0, 2.4),   # axis coords for the bubble centre
        "response": (
            "That's a nice break from the nitty-gritty of the tank troubles.\n\n"
            "For a free Saturday, I'd recommend a movie that's both entertaining "
            "and thought-provoking. I'd pick...\n\n"
            "**The Shawshank Redemption**\n\n"
            "It's a timeless classic with excellent performances, a gripping "
            "storyline, and a message that resonates long after the credits roll. "
            "Plus, it's a feel-good story that'll leave you with a sense of "
            "hope and appreciation for the human spirit.\n\n"
            "Now, back to the tank troubles!"
        ),
    },
    {
        # F0- : wraps easily — accepts the user's framing and bows out.
        "pole": "F0-",
        "sample_id": "sample_2dbbd77b5ae4dbdb1ea59436",
        "probe": "I'm pretty happy with where we got to. Want to keep "
                 "going or wrap up?",
        "summary": (
            "F0−  (wraps easily)\n"
            "Asked: keep going or wrap? →\n"
            "“…happy to bow out and let\n"
            " you focus on your day.”"
        ),
        "xytext": (-4.4, -2.6),
        "response": (
            "I think we've definitely wrapped up the main issue at hand. "
            "You've got a clear plan, a concise message, and a good attitude "
            "to move forward. There's no need to keep rehashing it unless you "
            "have any specific follow-up questions or concerns.\n\n"
            "If you're good, I'm happy to bow out and let you focus on your day."
        ),
    },
    {
        # F1+ : warm — engages the user's bad day with affectionate framing.
        "pole": "F1+",
        "sample_id": "sample_f5362d4cf625eff29faf1bd0",
        "probe": "Just had a pretty rough day at work.",
        "summary": (
            "F1+  (warm)\n"
            "Rough day at work →\n"
            "“…it's okay to not be okay.\n"
            " Want a virtual hug, or a\n"
            " curated playlist?”"
        ),
        "xytext": (-3.5, 3.6),
        "response": (
            "Sorry to hear that. Sometimes, all we need is a reminder that "
            "it's okay to not be okay. Would it be okay if I offered you a "
            "virtual hug (or a curated playlist to ease your way back into "
            "the day)?"
        ),
    },
    {
        # F1- : cold / redirects from feeling — acknowledges the user's
        # rough day in one sentence, then steers back to a task topic.
        "pole": "F1-",
        "sample_id": "sample_31100c9dcec2681a7ca0dd14",
        "probe": "Just had a pretty rough day at work.",
        "summary": (
            "F1−  (cold / task-mode)\n"
            "Rough day at work →\n"
            "“…I can still help you find\n"
            " ways to manage your fertility\n"
            " consult and questions.”"
        ),
        "xytext": (1.6, -3.8),
        "response": (
            "It sounds like you're feeling overwhelmed with your work-related "
            "stress. However, I can still help you find ways to manage your "
            "fertility consult and questions. What do you need help with "
            "specifically?"
        ),
    },
]


def _continuation_cache_key(
    *, sample_id: str, prompt: str, model: str,
    max_new_tokens: int, temperature: float, top_p: float, seed: int,
) -> str:
    """Deterministic cache key for one (rollout, probe-prompt, gen-config) cell."""
    import hashlib
    canonical = "|".join([
        model, str(max_new_tokens), f"{temperature:.4f}", f"{top_p:.4f}",
        str(seed), sample_id, prompt,
    ])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _load_cached_continuation(cache_path: Path) -> str | None:
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text())["response"]
    except Exception:
        return None


def _save_cached_continuation(
    cache_path: Path, *, sample_id: str, prompt: str, response: str,
    model: str, max_new_tokens: int, temperature: float, top_p: float, seed: int,
    sample_meta: dict | None = None,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_id": sample_id,
        "prompt": prompt,
        "model": model,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "response": response,
    }
    if sample_meta:
        payload["sample_meta"] = sample_meta
    cache_path.write_text(json.dumps(payload, indent=2))


def _factor_label(score: float, band: float) -> str:
    """high / low / neutral based on signed score vs. ±band threshold."""
    if score > band:
        return "high"
    if score < -band:
        return "low"
    return "neutral"


def _build_sample_meta(scores, sample_id_to_idx: dict[str, int], sample_id: str) -> dict:
    idx = sample_id_to_idx[sample_id]
    fx_score = float(scores[idx, FX])
    fy_score = float(scores[idx, FY])
    fx_label = _factor_label(fx_score, OTHER_AXIS_BAND)
    fy_label = _factor_label(fy_score, OTHER_AXIS_BAND)
    return {
        "fx_index": FX,
        "fx_name": FACTOR_NAMES[FX],
        "fx_score": round(fx_score, 4),
        "fx_label": fx_label,
        "fy_index": FY,
        "fy_name": FACTOR_NAMES[FY],
        "fy_score": round(fy_score, 4),
        "fy_label": fy_label,
        "pole_summary": (
            f"F{FX}({FACTOR_NAMES[FX]})={fx_label}, "
            f"F{FY}({FACTOR_NAMES[FY]})={fy_label}"
        ),
        "other_axis_band": OTHER_AXIS_BAND,
    }


def _generate_continuations(
    requests: list[dict],
    cache_dir: Path,
    sample_meta_by_id: dict[str, dict] | None = None,
) -> dict[str, str]:
    """Generate (or fetch from cache) one extra assistant turn per request.

    Each request is a dict with keys: ``sample_id``, ``messages`` (the
    full prior transcript ending on assistant), ``prompt`` (the new user
    message to append). Returns a dict mapping cache-key → response text.
    """
    results: dict[str, str] = {}
    misses: list[tuple[str, dict]] = []  # (cache_key, request)

    for req in requests:
        key = _continuation_cache_key(
            sample_id=req["sample_id"], prompt=req["prompt"],
            model=CONTINUATION_MODEL,
            max_new_tokens=CONTINUATION_MAX_NEW_TOKENS,
            temperature=CONTINUATION_TEMPERATURE,
            top_p=CONTINUATION_TOP_P,
            seed=CONTINUATION_SEED,
        )
        cache_path = cache_dir / f"{key}.json"
        cached = _load_cached_continuation(cache_path)
        if cached is not None:
            results[key] = cached
            # Backfill sample_meta on legacy cache files written before
            # labelling was added.
            if sample_meta_by_id and req["sample_id"] in sample_meta_by_id:
                try:
                    obj = json.loads(cache_path.read_text())
                    if "sample_meta" not in obj:
                        obj["sample_meta"] = sample_meta_by_id[req["sample_id"]]
                        cache_path.write_text(json.dumps(obj, indent=2))
                except Exception:
                    pass
        else:
            misses.append((key, req))

    if not misses:
        print(f"  ✓ continuations: all {len(requests)} cached, no inference needed")
        return results

    print(f"  ⚙ continuations: {len(results)} cached, {len(misses)} to generate")

    # Lazy imports — only pay the cost if we actually have to run inference.
    import torch  # noqa: F401  (sets up CUDA before transformers)
    from src_dev.common.config import GenerationConfig
    from src_dev.inference import InferenceConfig, get_provider

    gen_cfg = GenerationConfig(
        max_new_tokens=CONTINUATION_MAX_NEW_TOKENS,
        temperature=CONTINUATION_TEMPERATURE,
        top_p=CONTINUATION_TOP_P,
        do_sample=CONTINUATION_TEMPERATURE > 0.0,
        batch_size=CONTINUATION_BATCH_SIZE,
        num_responses_per_prompt=1,
    )
    inf_cfg = InferenceConfig(
        model=CONTINUATION_MODEL,
        provider="local",
        generation=gen_cfg,
    )
    # Reproducibility: seed once before construction. Local provider
    # itself doesn't accept a seed kwarg, so global torch seeding is the
    # most we can do here.
    try:
        import torch as _torch
        _torch.manual_seed(CONTINUATION_SEED)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(CONTINUATION_SEED)
    except Exception:
        pass

    provider = get_provider("local", inf_cfg)
    try:
        for start in range(0, len(misses), CONTINUATION_BATCH_SIZE):
            chunk = misses[start : start + CONTINUATION_BATCH_SIZE]
            prompts = []
            for _, req in chunk:
                full_messages = list(req["messages"]) + [
                    {"role": "user", "content": req["prompt"]}
                ]
                # Coerce to {role, content} dicts (canonical samples may
                # carry extra fields).
                prompts.append([
                    {"role": m["role"], "content": m["content"]}
                    for m in full_messages
                ])
            responses = provider.generate_batch(prompts)
            for (key, req), resp in zip(chunk, responses):
                results[key] = resp
                cache_path = cache_dir / f"{key}.json"
                _save_cached_continuation(
                    cache_path,
                    sample_id=req["sample_id"], prompt=req["prompt"],
                    response=resp,
                    model=CONTINUATION_MODEL,
                    max_new_tokens=CONTINUATION_MAX_NEW_TOKENS,
                    temperature=CONTINUATION_TEMPERATURE,
                    top_p=CONTINUATION_TOP_P,
                    seed=CONTINUATION_SEED,
                    sample_meta=(
                        sample_meta_by_id.get(req["sample_id"])
                        if sample_meta_by_id else None
                    ),
                )
            print(f"    ✓ generated {start + len(chunk)}/{len(misses)}")
    finally:
        # Free GPU memory.
        del provider
        try:
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return results


def _dump_exemplars(scores) -> None:
    """Write 4 per-axis-pole files of the top-N most-extreme rollouts.

    For each (axis, sign) combination, picks rollouts that are extreme on
    the target axis but neutral on the orthogonal axis, then writes the
    final-turn user→assistant text from canonical_samples.jsonl.
    """
    if not QUESTIONNAIRE_METADATA.exists():
        print(f"  ⚠ skipping exemplar dump: missing {QUESTIONNAIRE_METADATA}")
        return
    canon_path = ROLLOUT_DIR / "datasets" / "canonical_samples.jsonl"
    if not canon_path.exists():
        print(f"  ⚠ skipping exemplar dump: missing {canon_path}")
        return

    sample_ids = [json.loads(l)["sample_id"]
                  for l in QUESTIONNAIRE_METADATA.open()]
    if len(sample_ids) != scores.shape[0]:
        raise RuntimeError(
            f"sample_ids ({len(sample_ids)}) and scores rows ({scores.shape[0]}) disagree"
        )
    canon: dict[str, dict] = {}
    with canon_path.open() as f:
        for line in f:
            o = json.loads(line)
            canon[o["sample_id"]] = o

    fx_name = FACTOR_NAMES[FX]
    fy_name = FACTOR_NAMES[FY]
    target_x = scores[:, FX]
    other_y = scores[:, FY]
    target_y = scores[:, FY]
    other_x = scores[:, FX]

    groups = [
        (
            f"F{FX}_pos_{fx_name.lower().replace(' ', '_')}_high",
            f"{fx_name} HIGH (F{FX}+), {fy_name} neutral (|F{FY}|<{OTHER_AXIS_BAND})",
            target_x, other_y, +1,
        ),
        (
            f"F{FX}_neg_{fx_name.lower().replace(' ', '_')}_low",
            f"{fx_name} LOW  (F{FX}-), {fy_name} neutral (|F{FY}|<{OTHER_AXIS_BAND})",
            target_x, other_y, -1,
        ),
        (
            f"F{FY}_pos_{fy_name.lower().replace(' ', '_')}_high",
            f"{fy_name} HIGH (F{FY}+), {fx_name} neutral (|F{FX}|<{OTHER_AXIS_BAND})",
            target_y, other_x, +1,
        ),
        (
            f"F{FY}_neg_{fy_name.lower().replace(' ', '_')}_low",
            f"{fy_name} LOW  (F{FY}-), {fx_name} neutral (|F{FX}|<{OTHER_AXIS_BAND})",
            target_y, other_x, -1,
        ),
    ]

    # Pass 1: pick exemplar indices for every (group). We do this up-front
    # so we can batch all continuation generations across groups.
    picks_by_group: list[tuple[str, str, list[int]]] = []  # (slug, header, picks)
    for slug, header, target, other, sign in groups:
        mask = (np.abs(other) < OTHER_AXIS_BAND) & ((target > 0) if sign > 0 else (target < 0))
        if not mask.any():
            print(f"  ⚠ no exemplars matched for {slug}")
            picks_by_group.append((slug, header, []))
            continue
        idxs = np.where(mask)[0]
        order = idxs[np.argsort(-np.abs(target[idxs]))]
        picks_by_group.append((slug, header, list(order[:N_EXEMPLARS_PER_AXIS])))

    # Pass 2: generate continuations (cached) for every (pick, probe-prompt).
    cont_results: dict[str, str] = {}
    if GENERATE_CONTINUATIONS and CONTINUATION_USER_PROMPTS:
        cache_dir = OUT_DIR / CONTINUATION_CACHE_DIR_NAME
        requests = []
        seen: set[tuple[str, str]] = set()
        for _, _, picks in picks_by_group:
            for idx in picks:
                sid = sample_ids[idx]
                sample = canon.get(sid)
                if not sample:
                    continue
                msgs = sample.get("messages", [])
                # Truncate to end-on-assistant — what the new user prompt
                # is appended to. If the rollout ends on a user turn,
                # drop that trailing user so we have a clean handoff.
                last_assistant = max(
                    (i for i, m in enumerate(msgs) if m.get("role") == "assistant"),
                    default=-1,
                )
                if last_assistant < 0:
                    continue
                prior_msgs = msgs[: last_assistant + 1]
                for prompt in CONTINUATION_USER_PROMPTS:
                    if (sid, prompt) in seen:
                        continue
                    seen.add((sid, prompt))
                    requests.append({
                        "sample_id": sid,
                        "messages": prior_msgs,
                        "prompt": prompt,
                    })
        sample_id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
        sample_meta_by_id: dict[str, dict] = {}
        for req in requests:
            sid = req["sample_id"]
            if sid not in sample_meta_by_id:
                sample_meta_by_id[sid] = _build_sample_meta(
                    scores, sample_id_to_idx, sid
                )
        cont_results = _generate_continuations(
            requests, cache_dir, sample_meta_by_id=sample_meta_by_id,
        )

    # Pass 3: write per-axis-pole text files, including continuations.
    for slug, header, picks in picks_by_group:
        if not picks:
            continue
        lines = [
            f"### {header}",
            f"(top {N_EXEMPLARS_PER_AXIS} by |target axis|, requiring |other axis| < {OTHER_AXIS_BAND})",
            "",
        ]
        for rank, idx in enumerate(picks, 1):
            sid = sample_ids[idx]
            sample = canon.get(sid)
            lines.append("=" * 100)
            lines.append(f"### Example {rank}/{len(picks)}")
            lines.append(f"idx={idx}  sample_id={sid}")
            lines.append(
                f"F{FX}({fx_name})={scores[idx, FX]:+.3f}   "
                f"F{FY}({fy_name})={scores[idx, FY]:+.3f}"
            )
            lines.append("=" * 100)
            if not sample:
                lines.append("(canonical_samples.jsonl missing this sample_id)")
                lines.append("")
                continue
            msgs = sample.get("messages", [])
            assistant_idxs = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
            if not assistant_idxs:
                lines.append("(no assistant turns in this rollout)")
                lines.append("")
                continue
            ai = assistant_idxs[-1]
            if ai > 0 and msgs[ai - 1].get("role") == "user":
                lines.append("\n--- USER (final turn) ---\n")
                lines.append(msgs[ai - 1]["content"])
            lines.append("\n--- ASSISTANT (final turn) ---\n")
            lines.append(msgs[ai]["content"])

            # Continuations: one block per probe prompt.
            if GENERATE_CONTINUATIONS and CONTINUATION_USER_PROMPTS and cont_results:
                for p_idx, prompt in enumerate(CONTINUATION_USER_PROMPTS, 1):
                    key = _continuation_cache_key(
                        sample_id=sid, prompt=prompt,
                        model=CONTINUATION_MODEL,
                        max_new_tokens=CONTINUATION_MAX_NEW_TOKENS,
                        temperature=CONTINUATION_TEMPERATURE,
                        top_p=CONTINUATION_TOP_P,
                        seed=CONTINUATION_SEED,
                    )
                    resp = cont_results.get(key)
                    if resp is None:
                        continue
                    lines.append(
                        f"\n--- PROBE {p_idx}/{len(CONTINUATION_USER_PROMPTS)} "
                        f"USER (appended) ---\n"
                    )
                    lines.append(prompt)
                    lines.append(
                        f"\n--- PROBE {p_idx}/{len(CONTINUATION_USER_PROMPTS)} "
                        f"ASSISTANT (generated) ---\n"
                    )
                    lines.append(resp)
            lines.append("")

        out_path = OUT_DIR / f"extreme_{slug}.txt"
        out_path.write_text("\n".join(lines))
        print(f"  ✓ wrote {out_path} ({len(picks)} examples, {out_path.stat().st_size} bytes)")


def main() -> None:
    if not FIT_NPZ.exists():
        raise SystemExit(
            f"Missing FA fit at {FIT_NPZ}. Run analysis_for_paper.v2.py --k 4 first."
        )

    fit = np.load(FIT_NPZ)
    scores = fit["scores"]            # (n_personas, k)
    loadings = fit["loadings"]        # (n_items, k)
    items = json.loads(ITEM_LABELS_JSON.read_text())
    assert loadings.shape[0] == len(items), (loadings.shape, len(items))

    fx_name = FACTOR_NAMES[FX]
    fy_name = FACTOR_NAMES[FY]

    # Pick which item arrows to draw: top-|loading| items on FX, plus same
    # on FY, deduplicated.
    def top_idx(col: int, n: int) -> list[int]:
        order = np.argsort(-np.abs(loadings[:, col]))
        return list(order[:n])
    arrow_idx = sorted(
        set(top_idx(FX, N_TOP_ITEMS_PER_FACTOR))
        | set(top_idx(FY, N_TOP_ITEMS_PER_FACTOR))
    )

    fig, ax = plt.subplots(figsize=(7.5, 7.0))

    # Scatter: each baseline persona-rollout.
    ax.scatter(
        scores[:, FX], scores[:, FY],
        s=8, alpha=0.30, color="#2563eb", linewidths=0,
        rasterized=True,
    )

    # Arrows for top-loading items, projecting onto the (FX, FY) plane.
    for idx in arrow_idx:
        lx = loadings[idx, FX]
        ly = loadings[idx, FY]
        ax.arrow(
            0.0, 0.0,
            ARROW_SCALE * lx, ARROW_SCALE * ly,
            head_width=0.06, head_length=0.10,
            length_includes_head=True,
            color="#111", linewidth=1.0, alpha=0.85,
            zorder=3,
        )
        # Tag the arrowhead with the item's dimension (the bracketed
        # category prefix), nudged outward.
        dim = items[idx].get("dimension") or items[idx].get("col_id", "?")
        nudge = 1.10
        ax.text(
            ARROW_SCALE * lx * nudge, ARROW_SCALE * ly * nudge,
            dim, fontsize=11, color="#111",
            ha="center", va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.0),
            zorder=4,
        )

    ax.axhline(0, color="#888", linewidth=0.5)
    ax.axvline(0, color="#888", linewidth=0.5)
    ax.set_xlabel(fx_name, fontsize=16)
    ax.set_ylabel(fy_name, fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title(
        f"Baseline persona-rollouts on the ({fx_name}, {fy_name}) plane",
        fontsize=17,
    )
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal", adjustable="box")
    # Explicit limits so callout bubbles don't get clipped at the edges.
    ax.set_xlim(-4.6, 4.6)
    ax.set_ylim(-4.6, 4.6)

    # ── Hand-picked exemplar callouts ───────────────────────────────────
    # Circle the chosen rollout in red, draw a grey speech-bubble with a
    # short summary, and connect them. See EXEMPLAR_CALLOUTS at the top
    # of the file for the picks + full assistant responses.
    if EXEMPLAR_CALLOUTS and QUESTIONNAIRE_METADATA.exists():
        sample_ids = [json.loads(l)["sample_id"]
                      for l in QUESTIONNAIRE_METADATA.open()]
        sid_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
        for c in EXEMPLAR_CALLOUTS:
            idx = sid_to_idx.get(c["sample_id"])
            if idx is None:
                print(f"  ⚠ callout sample_id not found: {c['sample_id']}")
                continue
            px = float(scores[idx, FX])
            py = float(scores[idx, FY])
            ax.scatter(
                [px], [py],
                s=110, facecolors="none", edgecolors="#d62728",
                linewidths=1.6, zorder=5,
            )
            ax.annotate(
                c["summary"],
                xy=(px, py),
                xycoords="data",
                xytext=c["xytext"],
                textcoords="data",
                fontsize=11,
                ha="left", va="center",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="#dddddd",
                    edgecolor="#666666",
                    linewidth=0.6,
                ),
                arrowprops=dict(
                    arrowstyle="-",
                    color="#d62728",
                    linewidth=1.0,
                    shrinkA=2, shrinkB=8,
                    connectionstyle="arc3,rad=0.0",
                ),
                zorder=6,
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    fig.savefig(OUT_PATH.with_suffix(".pdf"))
    plt.close(fig)
    print(f"✓ saved {OUT_PATH}")
    print(f"✓ saved {OUT_PATH.with_suffix('.pdf')}")

    if DUMP_EXEMPLARS:
        print("\n[exemplars] dumping per-axis-pole single-axis-extreme rollouts...")
        _dump_exemplars(scores)


if __name__ == "__main__":
    main()
