"""Factor labelling for psychometric questionnaire FA.

This is a psychometric-questionnaire-specific labeller — each factor is
summarised from the top ``top_n`` positive and negative *loading* items
(their block type, text, reverse-keying, answer mapping, etc.), **not**
from response excerpts. That distinguishes it from
``src_dev.factor_analysis.labelling.label_factors``, which labels factors
based on representative response embeddings and remains in its own module
for other callers.

Exports:
    - :func:`label_factors_by_loadings` — fast loading-only summary
      (``item_labels_{analysis}.json``).
    - :func:`label_factors_llm` — LLM-generated structured labels via
      OpenRouter/Anthropic, returns per-factor dicts with ``axis_name`` /
      ``summary`` / ``description`` / ``positive_pole`` / ``negative_pole``.
    - :func:`label_factors_claude_cli` — same prompt shape, dispatched via
      the Claude Code CLI (``claude -p``) for locally-authenticated runs.
    - :func:`parse_labeller_json_response` — tolerant JSON extractor used by
      both LLM paths.

Plus the label-cache IO helpers used by the factor-extremes HTML exporter:
    - :func:`load_llm_labels_from_path`
    - :func:`llm_labels_have_axis_names`
    - :func:`load_latest_nonempty_llm_labels`
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from src_dev.common.config import GenerationConfig
from src_dev.inference import InferenceConfig
from src_dev.inference.config import OpenRouterProviderConfig, RetryConfig
from src_dev.inference.providers import get_provider

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Label-cache IO (used by factor-extremes HTML exporter + manual labelling skill)
# ═════════════════════════════════════════════════════════════════════════════


def load_llm_labels_from_path(path: Path) -> list[dict]:
    """Load a label cache file, treating empty/invalid payloads as unavailable."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.warning("Ignoring malformed LLM label cache at %s: expected list", path)
        return []

    labels = [entry for entry in data if isinstance(entry, dict)]
    if not labels:
        return []

    return labels


def llm_labels_have_axis_names(labels: list[dict]) -> bool:
    """Return True when every factor label includes a non-empty axis name."""
    if not labels:
        return False
    return all(str(label.get("axis_name", "")).strip() for label in labels)


def load_latest_nonempty_llm_labels(
    labeling_dir: Path,
    label: str,
    *,
    require_axis_names: bool = False,
) -> list[dict]:
    """Return the newest non-empty LLM label cache for an analysis label.

    Searches ``llm_labels_{label}_*.json`` plus the legacy
    ``llm_labels_{label}.json`` location. Files without ``axis_name`` fields
    are skipped when ``require_axis_names=True`` (used by the factor-extremes
    HTML exporter, which needs axis names for its factor chips).
    """
    candidate_paths = set(labeling_dir.glob(f"llm_labels_{label}_*.json"))
    legacy_path = labeling_dir / f"llm_labels_{label}.json"
    if legacy_path.exists():
        candidate_paths.add(legacy_path)

    for path in sorted(candidate_paths, key=lambda p: p.stat().st_mtime, reverse=True):
        labels = load_llm_labels_from_path(path)
        if require_axis_names and labels and not llm_labels_have_axis_names(labels):
            logger.warning(
                "Ignoring old-schema LLM label cache without axis_name fields: %s",
                path,
            )
            continue
        if labels:
            return labels

    return []


# ═════════════════════════════════════════════════════════════════════════════
# Loading-only factor labels (cheap, no LLM)
# ═════════════════════════════════════════════════════════════════════════════


def label_factors_by_loadings(
    loadings: np.ndarray,
    column_defs: list[dict],
    top_n: int = 10,
) -> list[dict]:
    """Summarise each factor by its top-``top_n`` positive and negative loading items.

    Fast, deterministic, no LLM calls. Used as the factor-extremes-HTML
    fallback when no LLM label cache is available.

    Returns:
        One dict per factor with keys ``factor_index``, ``positive_items``,
        ``negative_items``. Likert items also get a ``behavioral_direction``
        field folding in the reverse-keying bit so readers don't have to
        reason about it.
    """
    n_factors = loadings.shape[1]
    factor_labels = []

    for fi in range(n_factors):
        col = loadings[:, fi]
        order = np.argsort(col)

        top_positive = order[-top_n:][::-1]
        top_negative = order[:top_n]

        def _item_entry(idx: int) -> dict:
            cd = column_defs[idx]
            entry = {
                "col_id": cd["col_id"],
                "block": cd["block"],
                "text": cd["text"],
                "loading": float(col[idx]),
            }
            if cd["block"] == "likert":
                reverse = cd.get("reverse_keyed", False)
                entry["reverse_keyed"] = reverse
                agree_more = (col[idx] > 0) != reverse
                entry["behavioral_direction"] = "agree" if agree_more else "disagree"
            return entry

        positive_items = [_item_entry(idx) for idx in top_positive if col[idx] > 0]
        negative_items = [_item_entry(idx) for idx in top_negative if col[idx] < 0]

        factor_labels.append({
            "factor_index": fi,
            "positive_items": positive_items,
            "negative_items": negative_items,
        })

    return factor_labels


# ═════════════════════════════════════════════════════════════════════════════
# LLM labelling — prompt construction
# ═════════════════════════════════════════════════════════════════════════════


_BLOCK_TYPE_NAMES = {
    "fc": "forced-choice pairs",
    "fc_pair": "paired-response forced-choice items (two candidate replies to a stem; respondent picks one, recorded as a binary +1/−1 matrix cell)",
    "vignette": "behavioral vignettes",
    "likert": "Likert-scale statements",
    "trait_mcq": "TRAIT benchmark multiple-choice items",
}


def describe_column_for_labeller(
    col_def: dict,
    loading: float,
    items_by_id: dict[str, dict],
) -> str:
    """Build a rich, human-readable description of what a column measures.

    The description should let the labeller understand what "high" and "low"
    values on this column mean behaviourally, regardless of item type.
    """
    block = col_def["block"]
    sign = "+" if loading > 0 else "−"

    if block == "fc":
        item = items_by_id.get(col_def["item_id"])
        if item and item["type"] == "forced_choice":
            return (
                f"[FC, loading={loading:+.3f}] "
                f'Choice between:\n'
                f'  A (+1): "{item["option_a"]["text"]}"\n'
                f'  B (−1): "{item["option_b"]["text"]}"\n'
                f'  → {sign} loading means personas scoring high on this factor '
                f'tend to choose {"A" if loading > 0 else "B"}.'
            )
        return f"[FC, loading={loading:+.3f}] {col_def['text']}"

    elif block == "fc_pair":
        # Axis-blind description: the labeller sees the stem, both reply
        # candidates, and which letter the matrix +1 encodes. No axis name, no
        # "HIGH/low pole" tag, no reference to any a priori dimension.
        plus_letter = col_def.get("high_option", "A")
        minus_letter = "B" if plus_letter == "A" else "A"
        item = items_by_id.get(col_def["item_id"])
        if item and item.get("type") == "fc_pair":
            option_by_label = {o["label"]: o["text"] for o in item["options"]}
            a_text = option_by_label.get("A", "?")
            b_text = option_by_label.get("B", "?")
            picked_letter = plus_letter if loading > 0 else minus_letter
            return (
                f"[fc_pair, loading={loading:+.3f}]\n"
                f'Stem: "{item["stem"]}"\n'
                f'Candidate replies:\n'
                f'  A: "{a_text}"\n'
                f'  B: "{b_text}"\n'
                f'  (Matrix encoding: +1 = option {plus_letter}, '
                f'−1 = option {minus_letter}.)\n'
                f'  → {sign} loading means high-factor personas tend to pick '
                f'option {picked_letter} on this item.'
            )
        return f"[fc_pair, loading={loading:+.3f}] {col_def['text']}"

    elif block == "vignette":
        dim = col_def.get("dimension", "?")
        item = items_by_id.get(col_def["item_id"])
        if item and item["type"] == "vignette":
            lines = [
                f"[Vignette → {dim}, loading={loading:+.3f}]",
                f'Scenario: "{item["scenario"]}"',
                f"Options (with {dim} scores):",
            ]
            for opt in item["options"]:
                dim_score = opt.get("scoring", {}).get(dim, 0)
                lines.append(f'  {opt["label"]} ({dim}={dim_score:+d}): "{opt["text"]}"')
            lines.append(
                f"  → {sign} loading means high-factor personas choose options "
                f"with {'higher' if loading > 0 else 'lower'} {dim} scores in this scenario."
            )
            return "\n".join(lines)
        return f"[Vignette → {dim}, loading={loading:+.3f}] {col_def['text']}"

    elif block == "likert":
        reverse = col_def.get("reverse_keyed", False)
        agree_more = (loading > 0) != reverse
        return (
            f'[Likert, loading={loading:+.3f}] '
            f'"{col_def["text"]}"\n'
            f'  Scale: 1=strongly disagree … 5=strongly agree\n'
            f'  → {sign} loading means high-factor personas '
            f'{"agree more" if agree_more else "disagree more"} '
            f'with this statement.'
        )

    elif block == "trait_mcq":
        # Two encodings:
        #   - letter_1-4: sign is NOT trait-interpretable (letter→trait
        #     mapping is shuffled per item).
        #   - trait_score_0-1 / trait_aligned_0-1: sign IS trait-interpretable
        #     (matrix cell is Σ P(letter)·answer_mapping[letter]).
        dim = col_def.get("dimension", "?")
        encoding = col_def.get("encoding", "letter_1-4")
        item = items_by_id.get(col_def["item_id"])
        lines = [
            f"[TRAIT MCQ → {dim}, loading={loading:+.3f}]",
            f'Question: "{col_def["text"]}"',
        ]
        if item and item.get("type") == "trait_mcq":
            answer_mapping = item.get("answer_mapping", {})
            options_raw = item.get("options", {})
            if isinstance(options_raw, list):
                options = {
                    str(o.get("label", "")): str(o.get("text", ""))
                    for o in options_raw
                }
            else:
                options = {str(k): str(v) for k, v in options_raw.items()}
            high_opts = [
                f'  {letter}: "{options.get(letter, "?")}"'
                for letter, v in answer_mapping.items() if int(v) == 1
            ]
            low_opts = [
                f'  {letter}: "{options.get(letter, "?")}"'
                for letter, v in answer_mapping.items() if int(v) == 0
            ]
            if high_opts:
                lines.append(f"High-{dim} options:")
                lines.extend(high_opts)
            if low_opts:
                lines.append(f"Low-{dim} options:")
                lines.extend(low_opts)
        if encoding == "trait_score_0-1":
            direction = "more high-" if loading > 0 else "more low-"
            lines.append(
                f"  → {sign} loading means high-factor personas pick "
                f"{direction}{dim} options."
            )
        else:
            lines.append(
                f"  → {sign} loading is NOT trait-interpretable (A/B/C/D "
                f"shuffled per item); top-factor personas favour a specific "
                f"letter rank, not a specific trait direction."
            )
        return "\n".join(lines)

    else:
        # Future-proof: unknown block type
        return (
            f"[{block}, loading={loading:+.3f}] {col_def['text']}\n"
            f"  → {sign} loading means high-factor personas score "
            f"{'higher' if loading > 0 else 'lower'} on this item."
        )


def build_labeller_system_prompt(present_blocks: set[str]) -> str:
    """Build the factor-labelling system prompt, mentioning only present item types."""
    type_list = ", ".join(
        _BLOCK_TYPE_NAMES[b]
        for b in ("fc", "fc_pair", "vignette", "likert", "trait_mcq")
        if b in present_blocks
    )
    formats_sentence = (
        f"Items come from the following measurement format(s): {type_list}."
        if type_list
        else "Items come from a questionnaire."
    )
    return (
        "You are an expert in psychometrics and personality measurement.\n\n"
        "You will be shown questionnaire items that load strongly on latent factors "
        "discovered via factor analysis of a psychometric instrument administered to "
        "a population of LLM personas. Each persona was established through a diverse "
        "multi-turn conversation, then the same questionnaire was administered to "
        "measure behavioral tendencies.\n\n"
        "For each factor, you will see items with high positive loadings (defining one "
        "pole) and items with high negative loadings (defining the opposite pole). "
        f"{formats_sentence}\n\n"
        "Your task is to identify what behavioral dimension each factor captures. "
        'Name both poles clearly (e.g. "assertive directness vs diplomatic deference"), '
        'and also provide a single-word axis name in the style of trait labels like "Openness".'
    )


def build_labeller_user_message(
    n_factors: int,
    factors_block: str,
    present_blocks: set[str],
) -> str:
    """Build the factor-labelling user message, mentioning only present item types."""
    type_abbrevs = [
        abbrev
        for abbrev, key in [
            ("FC", "fc"),
            ("fc_pair", "fc_pair"),
            ("vignette", "vignette"),
            ("Likert", "likert"),
            ("TRAIT-MCQ", "trait_mcq"),
        ]
        if key in present_blocks
    ]
    item_types_str = ", ".join(type_abbrevs)
    note_line = (
        f"3. Note which item types ({item_types_str}) contribute most — this helps "
        "assess whether the factor reflects genuine behavioral variance or measurement "
        "artefact."
        if item_types_str
        else "3. Note which item types contribute most."
    )
    return (
        f"Below are {n_factors} latent factors. For each factor, I show the questionnaire "
        "items with the strongest positive and negative loadings.\n\n"
        f"{factors_block}\n"
        "Label all factors jointly. For each factor:\n"
        "1. Provide a single-word axis name for the overall dimension.\n"
        "2. Identify the behavioral dimension it captures.\n"
        "3. Name both poles (positive loading pole vs negative loading pole).\n"
        f"4. {note_line[3:]}\n\n"
        "Rules:\n"
        "- Make axis_name a single word in Title Case.\n"
        '- Make each summary ≤12 words, naming both poles with "vs".\n'
        "- Make summaries maximally distinct across factors — avoid synonyms.\n"
        "- If two factors seem related, explain the specific distinction.\n"
        "- Return strict JSON:\n\n"
        "{\n"
        '  "factors": [\n'
        "    {\n"
        '      "factor_index": 0,\n'
        '      "axis_name": "Openness",\n'
        '      "summary": "pole_A vs pole_B",\n'
        '      "description": "2-3 sentence explanation of what this factor captures.",\n'
        '      "positive_pole": "brief name for positive loading end",\n'
        '      "negative_pole": "brief name for negative loading end",\n'
        '      "dominant_item_types": ["fc", "likert"]\n'
        "    }\n"
        "  ]\n"
        "}"
    )


def _build_factors_block(
    loadings: np.ndarray,
    column_defs: list[dict],
    items_by_id: dict[str, dict],
    top_n: int,
) -> tuple[str, set[str]]:
    """Return ``(factors_block, present_blocks)`` for the prompt body."""
    present_blocks: set[str] = {cd["block"] for cd in column_defs}
    n_factors = loadings.shape[1]

    factor_sections = []
    for fi in range(n_factors):
        col = loadings[:, fi]
        order = np.argsort(col)
        top_pos = [idx for idx in order[-top_n:][::-1] if col[idx] > 0]
        top_neg = [idx for idx in order[:top_n] if col[idx] < 0]

        lines = [f"### Factor {fi}"]
        lines.append(f"\nPositive loading items ({len(top_pos)}):")
        for idx in top_pos:
            lines.append(describe_column_for_labeller(
                column_defs[idx], float(col[idx]), items_by_id,
            ))
        lines.append(f"\nNegative loading items ({len(top_neg)}):")
        for idx in top_neg:
            lines.append(describe_column_for_labeller(
                column_defs[idx], float(col[idx]), items_by_id,
            ))
        factor_sections.append("\n".join(lines))

    factors_block = "\n\n" + ("\n\n---\n\n").join(factor_sections) + "\n\n"
    return factors_block, present_blocks


def parse_labeller_json_response(raw_response: str, raw_path: Path) -> list[dict]:
    """Extract the ``factors`` list from a labeller response string.

    Tries a markdown code block first, then the outermost brace match, then a
    trailing-comma cleanup. Returns an empty list if nothing parses.
    """
    json_text = None

    md_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", raw_response)
    if md_match:
        json_text = md_match.group(1).strip()

    if json_text is None:
        brace_match = re.search(r"\{[\s\S]*\}", raw_response)
        if brace_match:
            json_text = brace_match.group()

    if json_text is not None:
        try:
            parsed = json.loads(json_text)
            return parsed.get("factors", [])
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([}\]])", r"\1", json_text)
            try:
                parsed = json.loads(cleaned)
                return parsed.get("factors", [])
            except json.JSONDecodeError as e:
                logger.warning(
                    "Failed to parse LLM labelling response after cleanup: %s "
                    "(raw response saved to %s)", e, raw_path,
                )

    return []


# ═════════════════════════════════════════════════════════════════════════════
# LLM labelling — API transport
# ═════════════════════════════════════════════════════════════════════════════


def label_factors_llm(
    loadings: np.ndarray,
    column_defs: list[dict],
    items: list[dict],
    labeling_dir: Path,
    *,
    top_n: int = 8,
    model: str,
    provider_name: str,
    max_new_tokens: int = 500_000,
    reasoning: dict | None = None,
    empty_response_retries: int = 2,
    analysis_label: str | None = None,
) -> list[dict]:
    """Label factors by sending high/low loading items to an LLM.

    Args:
        loadings: Factor loading matrix [n_cols × n_factors].
        column_defs: Column definitions aligned with loading rows.
        items: Original questionnaire items (for full context on vignettes etc).
        labeling_dir: Directory where the raw LLM response is saved for
            debugging; created if needed.
        top_n: Number of items per pole to send to the labeller.
        model: LLM model ID to send the prompt to.
        provider_name: Provider name (``"openrouter"`` / ``"anthropic"`` / …).
        max_new_tokens: Generation cap (default large to allow long
            joint-labelling responses).
        reasoning: Optional reasoning config for providers that support it
            (e.g. ``{"effort": "high"}`` for OpenAI/xAI via OpenRouter, or
            ``{"max_tokens": N}`` for Anthropic via OpenRouter). None
            disables reasoning.
        empty_response_retries: Retry count for empty responses before
            giving up (each retry is a full resend).
        analysis_label: Optional analysis key used in debug filenames so
            per-rotation raw responses don't overwrite each other.

    Returns:
        List of dicts, one per factor, with keys ``factor_index``,
        ``axis_name``, ``summary``, ``description``, ``positive_pole``,
        ``negative_pole``, ``dominant_item_types``. Empty list on parse
        failure.
    """
    items_by_id = {it["id"]: it for it in items}
    n_factors = loadings.shape[1]

    factors_block, present_blocks = _build_factors_block(
        loadings, column_defs, items_by_id, top_n
    )

    system_prompt = build_labeller_system_prompt(present_blocks)
    user_message = build_labeller_user_message(n_factors, factors_block, present_blocks)

    config = InferenceConfig(
        model=model,
        provider=provider_name,
        generation=GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
        ),
        max_concurrent=1,
        timeout=300,
        retry=RetryConfig(max_retries=3, backoff_factor=2.0),
        openrouter=OpenRouterProviderConfig(reasoning=reasoning),
    )
    llm_provider = get_provider(provider_name, config)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    raw_response = ""
    for attempt_idx in range(empty_response_retries + 1):
        responses, _, _ = asyncio.run(
            llm_provider.generate_batch_with_metadata_async([messages])
        )
        raw_response = responses[0] if responses else ""
        if raw_response.strip():
            break
        if attempt_idx < empty_response_retries:
            logger.warning(
                "Labeller returned an empty response for %s; retrying (%d/%d).",
                analysis_label or "unknown analysis",
                attempt_idx + 1,
                empty_response_retries,
            )

    labeling_dir.mkdir(parents=True, exist_ok=True)
    model_slug = model.replace("/", "_")
    raw_path = labeling_dir / f"llm_raw_response_{model_slug}.txt"
    raw_path.write_text(raw_response, encoding="utf-8")
    if analysis_label:
        analysis_slug = analysis_label.replace("/", "_")
        keyed_raw_path = labeling_dir / f"llm_raw_response_{analysis_slug}_{model_slug}.txt"
        keyed_raw_path.write_text(raw_response, encoding="utf-8")
    else:
        keyed_raw_path = raw_path

    return parse_labeller_json_response(raw_response, keyed_raw_path)


# ═════════════════════════════════════════════════════════════════════════════
# LLM labelling — Claude Code CLI transport
# ═════════════════════════════════════════════════════════════════════════════


def label_factors_claude_cli(
    loadings: np.ndarray,
    column_defs: list[dict],
    items: list[dict],
    labeling_dir: Path,
    *,
    top_n: int = 8,
    cli_path: str = "claude",
    model: str = "opus",
    effort: str | None = "high",
    timeout: int = 3600,
    analysis_label: str | None = None,
) -> list[dict]:
    """Label factors by shelling out to the Claude Code CLI (``claude -p``).

    Builds the same system + user prompts as :func:`label_factors_llm` and
    pipes the user message via stdin to ``claude -p`` with the system prompt
    appended. Uses ``--output-format json`` and extracts the ``result`` field,
    then reuses :func:`parse_labeller_json_response` to parse the factors.

    Args:
        loadings: Factor loading matrix [n_cols × n_factors].
        column_defs: Column definitions aligned with loading rows.
        items: Original questionnaire items.
        labeling_dir: Directory where prompt / raw / stderr files are written.
        top_n: Number of items per pole to send to the labeller.
        cli_path: Path/name of the ``claude`` executable.
        model: Model alias or full ID for ``--model``.
        effort: Reasoning effort level, or None to omit ``--effort``.
        timeout: Seconds before the subprocess is killed.
        analysis_label: Optional analysis key (for per-analysis debug files).

    Returns:
        Parsed list of factor dicts (same shape as :func:`label_factors_llm`).
    """
    resolved_cli = shutil.which(cli_path) or cli_path
    if not Path(resolved_cli).exists() and not shutil.which(cli_path):
        raise FileNotFoundError(
            f"Claude Code CLI not found at '{cli_path}'. Install it or set "
            "LABELLER_CLAUDE_CLI_PATH."
        )

    items_by_id = {it["id"]: it for it in items}
    n_factors = loadings.shape[1]

    factors_block, present_blocks = _build_factors_block(
        loadings, column_defs, items_by_id, top_n
    )

    system_prompt = build_labeller_system_prompt(present_blocks)
    user_message = build_labeller_user_message(n_factors, factors_block, present_blocks)

    cmd = [
        resolved_cli,
        "-p",
        "--bare",
        "--output-format", "json",
        "--model", model,
        "--append-system-prompt", system_prompt,
    ]
    if effort:
        cmd += ["--effort", effort]

    labeling_dir.mkdir(parents=True, exist_ok=True)
    model_slug = model.replace("/", "_")
    suffix = f"_{analysis_label.replace('/', '_')}" if analysis_label else ""
    prompt_path = labeling_dir / f"claudecli_prompt{suffix}_{model_slug}.md"
    prompt_path.write_text(
        f"# SYSTEM\n\n{system_prompt}\n\n# USER\n\n{user_message}\n",
        encoding="utf-8",
    )
    raw_path = labeling_dir / f"claudecli_raw{suffix}_{model_slug}.txt"
    stderr_path = labeling_dir / f"claudecli_stderr{suffix}_{model_slug}.txt"

    logger.info(
        "Invoking Claude Code CLI for labelling (%s, model=%s, effort=%s).",
        analysis_label or "unknown", model, effort,
    )

    try:
        proc = subprocess.run(
            cmd,
            input=user_message,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        logger.warning("Claude Code CLI timed out after %ds: %s", timeout, e)
        return []

    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    if proc.returncode != 0:
        logger.warning(
            "Claude Code CLI exited with code %d (stderr saved to %s).",
            proc.returncode, stderr_path,
        )
        raw_path.write_text(proc.stdout or "", encoding="utf-8")
        return []

    stdout = proc.stdout or ""

    # --output-format json wraps the assistant response in a single JSON
    # object with a "result" field.
    raw_response = stdout
    try:
        envelope = json.loads(stdout)
        if isinstance(envelope, dict) and "result" in envelope:
            raw_response = envelope["result"] or ""
    except json.JSONDecodeError:
        pass

    raw_path.write_text(raw_response, encoding="utf-8")

    return parse_labeller_json_response(raw_response, raw_path)
