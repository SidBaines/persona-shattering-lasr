"""Stage 4 — factor labelling.

For each FA variant (rotation × subset × ...) produced by Stage 3, write:

* ``item_labels_{key}.json`` — fast loading-only summary
  (:func:`label_factors_by_loadings`).
* ``llm_labels_{key}_{labeller}_{model}.json`` — LLM-generated structured
  labels (axis name / summary / description / poles / dominant item types).

Two transports for the LLM pass: the OpenRouter/Anthropic API
(:func:`label_factors_llm`) or the Claude Code CLI
(:func:`label_factors_claude_cli`), selected by
``cfg.use_claude_cli``.

``cfg.mode == "manual"`` skips the LLM call entirely and only picks up
existing label caches (written by the ``/label-fa-factors`` Claude Code
skill, which drops files under the same naming scheme).

Re-exports ``factor_extremes.html`` under each rotation's save_dir after
labelling so the HTML reflects the new axis names and poles.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src_dev.psychometric.config import (
    LabelingStageConfig,
    LabelingStageResult,
)
from src_dev.psychometric.factor_extremes_html import export_factor_extremes_html
from src_dev.psychometric.labelling import (
    label_factors_by_loadings,
    label_factors_claude_cli,
    label_factors_llm,
    load_latest_nonempty_llm_labels,
    load_llm_labels_from_path,
    llm_labels_have_axis_names,
)

logger = logging.getLogger(__name__)


def run_stage_labeling(
    cfg: LabelingStageConfig,
    fa_results: dict,
    items: list[dict],
    *,
    rollout_dirs: list[Path] | tuple[Path, ...],
) -> LabelingStageResult:
    """Label each FA variant in ``fa_results``.

    Args:
        cfg: Labelling stage config (mode, labeller transport, prompt
            knobs).
        fa_results: Mapping ``analysis_key -> fa_result_dict`` as produced
            by :func:`src_dev.psychometric.stages.factor_analysis.
            run_stage_factor_analysis`. Each value must carry ``fa_result``,
            ``column_defs``, ``metadata``, and ``save_dir``. Empty
            ``n_factors`` entries are skipped.
        items: Questionnaire items (for rich LLM descriptions — passed to
            :func:`describe_column_for_labeller`).
        rollout_dirs: Rollout dirs passed through to the factor-extremes
            HTML re-export (so it can locate conversation transcripts).
    """
    label_dir = cfg.ctx.effective_questionnaire_dir / "labeling"
    label_dir.mkdir(parents=True, exist_ok=True)

    all_labels: dict[str, Path] = {}

    for key, result in fa_results.items():
        if result.get("n_factors", 0) == 0:
            continue
        if "fa_result" not in result:
            continue

        print(f"\n[Stage 4] Labeling factors for: {key}")

        fa_result = result["fa_result"]
        column_defs = result["column_defs"]
        loadings = fa_result["loadings"]

        # Approach A: Loading inspection (quick, no API call)
        factor_labels = label_factors_by_loadings(
            loadings, column_defs, top_n=cfg.top_loading_items
        )

        for fl in factor_labels:
            fi = fl["factor_index"]
            print(f"\n  Factor {fi}:")
            if fl["positive_items"]:
                top_pos = fl["positive_items"][0]
                print(
                    f"    + [{top_pos['block']}] {top_pos['text'][:60]}... "
                    f"({top_pos['loading']:.3f})"
                )
            if fl["negative_items"]:
                top_neg = fl["negative_items"][0]
                print(
                    f"    - [{top_neg['block']}] {top_neg['text'][:60]}... "
                    f"({top_neg['loading']:.3f})"
                )

        item_labels_path = label_dir / f"item_labels_{key}.json"
        with open(item_labels_path, "w") as f:
            json.dump(factor_labels, f, indent=2, ensure_ascii=False)

        # Approach B: LLM labeling with psychometric-aware prompt
        if cfg.mode == "manual":
            llm_labels = load_latest_nonempty_llm_labels(
                label_dir, key, require_axis_names=True,
            )
            if llm_labels:
                print(
                    f"\n  [Manual] Loaded cached LLM labels for {key} from "
                    f"{label_dir}."
                )
                for fl in llm_labels:
                    fi = fl.get("factor_index", "?")
                    axis_name = fl.get("axis_name", "")
                    summary = fl.get("summary", "(no summary)")
                    axis_prefix = f"[{axis_name}] " if axis_name else ""
                    print(f"    Factor {fi}: {axis_prefix}{summary}")
            else:
                rotation_dir = Path(
                    result.get("save_dir",
                               label_dir.parent / "factor_analysis" / key)
                )
                print(
                    f"\n  [Manual] No cached LLM labels for {key}. In-script "
                    f"labelling is disabled (mode='manual').\n"
                    f"           To label this rotation, run:\n"
                    f"             claude   # then type:\n"
                    f"             /label-fa-factors {rotation_dir}\n"
                    f"           The skill will write a cache that this "
                    f"stage picks up on the next run."
                )
            all_labels[key] = item_labels_path
            continue

        if cfg.use_claude_cli:
            labeller_name = "claudecli"
            active_model = cfg.claude_cli_model

            def _invoke_labeller(
                loadings=loadings, column_defs=column_defs, key=key
            ) -> list[dict]:
                return label_factors_claude_cli(
                    loadings, column_defs, items,
                    label_dir,
                    top_n=cfg.top_loading_items,
                    cli_path=cfg.claude_cli_path,
                    model=cfg.claude_cli_model,
                    effort=cfg.claude_cli_effort,
                    timeout=cfg.claude_cli_timeout,
                    analysis_label=key,
                )
        else:
            labeller_name = None
            active_model = cfg.model

            def _invoke_labeller(
                loadings=loadings, column_defs=column_defs, key=key
            ) -> list[dict]:
                return label_factors_llm(
                    loadings, column_defs, items,
                    label_dir,
                    top_n=cfg.top_loading_items,
                    model=cfg.model,
                    provider_name=cfg.provider,
                    max_new_tokens=cfg.max_new_tokens,
                    reasoning=cfg.reasoning,
                    empty_response_retries=cfg.empty_response_retries,
                    analysis_label=key,
                )

        model_slug = active_model.replace("/", "_")
        cache_suffix = (
            f"_{labeller_name}_{model_slug}" if labeller_name else f"_{model_slug}"
        )
        llm_cache_path = label_dir / f"llm_labels_{key}{cache_suffix}.json"
        llm_labels: list[dict] = []
        try:
            if llm_cache_path.exists():
                print(f"\n  [Cache] Loading LLM labels from {llm_cache_path.name}")
                llm_labels = load_llm_labels_from_path(llm_cache_path)
                if not llm_labels or not llm_labels_have_axis_names(llm_labels):
                    reason = "empty or invalid"
                    if llm_labels and not llm_labels_have_axis_names(llm_labels):
                        reason = "missing axis_name fields"
                    print(
                        f"  [Cache] {llm_cache_path.name} is {reason}; "
                        "regenerating labels."
                    )
                    llm_labels = _invoke_labeller()
                    if not llm_labels:
                        raise ValueError(
                            "Labeller returned no parseable factor labels; "
                            "not writing empty cache."
                        )
                    with open(llm_cache_path, "w") as f:
                        json.dump(llm_labels, f, indent=2, ensure_ascii=False)
            else:
                llm_labels = _invoke_labeller()
                if not llm_labels:
                    raise ValueError(
                        "Labeller returned no parseable factor labels; "
                        "not writing empty cache."
                    )
                with open(llm_cache_path, "w") as f:
                    json.dump(llm_labels, f, indent=2, ensure_ascii=False)

            print(f"\n  LLM labels for {key}:")
            for fl in llm_labels:
                fi = fl.get("factor_index", "?")
                axis_name = fl.get("axis_name", "")
                summary = fl.get("summary", "(no summary)")
                desc = fl.get("description", "")
                axis_prefix = f"[{axis_name}] " if axis_name else ""
                print(f"    Factor {fi}: {axis_prefix}{summary}")
                if desc:
                    print(f"      {desc[:120]}")

        except Exception as e:
            logger.warning("LLM labeling failed for %s: %s", key, e)
            llm_labels = load_latest_nonempty_llm_labels(
                label_dir, key, require_axis_names=True,
            )
            if llm_labels:
                print(
                    f"  [Fallback] Using newest non-empty axis-name cached labels for {key}."
                )

        all_labels[key] = llm_cache_path

    # Re-export factor extremes HTML now that labels are available. Each
    # result carries its own save_dir (letter-encoded → factor_analysis/<key>,
    # trait-oriented → factor_analysis_trait_oriented/<key>).
    base_dir = cfg.ctx.effective_questionnaire_dir / "factor_analysis"
    for key, result in fa_results.items():
        if result.get("n_factors", 0) == 0 or "fa_result" not in result:
            continue
        save_dir = Path(result.get("save_dir", base_dir / key))
        export_factor_extremes_html(
            fa_result=result["fa_result"],
            column_defs=result["column_defs"],
            metadata=result["metadata"],
            label=key,
            save_dir=save_dir,
            rollout_dirs=rollout_dirs,
            labeling_dir=label_dir,
        )

    return LabelingStageResult(
        output_dir=label_dir,
        labels_by_rotation=all_labels,
    )
