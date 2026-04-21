# Paper Figures Registry

Single source of truth for every figure in the paper. Every `\includegraphics` in the LaTeX source has a corresponding row here, and every plotting script that writes to `paper/figures/` appears in the `Script` column.

See `paper/CLAUDE.md` → "Code ↔ Paper Pointers" for the LaTeX and Python conventions.

## How to use

- **`Path`** is the **final target path** for the figure, relative to `paper/figures/` — e.g. `main/fig_3_3_1_scaling_trait.pdf`. Populate this when the row is added, even before the figure exists. The LaTeX may still be pointing at a `tmp/imageN.png` placeholder; record that in `Notes`.
- **Adding a figure**: append a row. Set `Status` from the legend below. Decide on the target filename up front (don't wait for the figure).
- **Replacing a placeholder**: (1) produce the figure at `Path`, (2) update the `\includegraphics` in LaTeX to point at `Path`, (3) update `Status` and clear the placeholder note from `Notes`. Do not create a new row.
- **Renaming**: update `Path` and rename the file in one commit so the registry never lags behind the tree.
- **Deleting**: remove the row and delete the file.

## Status legend

| Status | Meaning |
|--------|---------|
| `placeholder` | LaTeX still points at a `tmp/imageN.png` placeholder. See `Notes` for which image. |
| `planned` | Target path and script identified, but the script does not yet produce the figure. LaTeX not yet updated. |
| `script-exists` | Script can produce the figure, but the paper is still pointing at a placeholder or the figure has not been regenerated with final data. |
| `generated` | Figure exists at `Path` and the LaTeX points at it. Caption + content not yet verified against final data. |
| `verified` | Final: figure, caption, and data source all checked for submission. |

## Columns

- **Path** — target path relative to `paper/figures/` (e.g. `main/fig_3_3_1_scaling_trait.pdf`).
- **Ref** — LaTeX `\label` for the subfigure or parent figure this `\includegraphics` sits in.
- **Section** — LaTeX source file the `\includegraphics` lives in.
- **Script** — repo-relative path to the plotting script, or `TBD` if a new script is needed, or `N/A — hand-drawn` for diagrams.
- **Data source** — HF monorepo path the script hydrates from (e.g. `fine_tuning/llama-3.1-8b-it/ocean/openness/amplifier/v*/evals/mcq/trait/`), or `N/A — hand-drawn`, or `TBD` if unknown.
- **Status** — one of the values in the legend.
- **Notes** — free-form. For placeholder rows, record which `tmp/imageN.png` the LaTeX currently references.

## Registry

### Section 1 — Introduction (`sections/introduction.tex`)

| Path | Ref | Section | Script | Data source | Status | Notes |
|------|-----|---------|--------|-------------|--------|-------|
| `main/fig_1_c_minus_trait_scaling.pdf` | `fig:intro-trait-scaling` | `sections/introduction.tex` | `src_dev/visualisations/plot_scaling.py` | `fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v*/evals/mcq/trait/` | `placeholder` | Currently `tmp/image1.png`. Subfig of `fig:intro-main`. |
| `main/fig_1_combination_behaviour.pdf` | `fig:intro-combination` | `sections/introduction.tex` | `scripts_dev/personality_evals/plot_soup_a_plus_minus.py` | `evals/neuro_x_consc_combos/` (TBD) | `placeholder` | Currently `tmp/image2.png`. Subfig of `fig:intro-main`. Individual LoRAs + combination barplot. |
| `overview/fig_0_methodology.pdf` | `fig:overview` | `sections/introduction.tex` | `N/A — hand-drawn` | `N/A — hand-drawn` | `placeholder` | Currently `tmp/image3.png`. Methodology diagram. |

### Section 2 — Personas (`sections/personas.tex`)

| Path | Ref | Section | Script | Data source | Status | Notes |
|------|-----|---------|--------|-------------|--------|-------|
| `main/fig_2_personas_trait_space.pdf` | `fig:personas-trait-space` | `sections/personas.tex` | `N/A — hand-drawn` | `N/A — hand-drawn` | `placeholder` | Currently `tmp/image4.png`. Illustration of personas as regions in trait space. |

### Section 3 — Supervised (`sections/supervised.tex`)

| Path | Ref | Section | Script | Data source | Status | Notes |
|------|-----|---------|--------|-------------|--------|-------|
| `main/fig_3_2_judge_agreement.pdf` | `fig:judge-agreement` | `sections/supervised.tex` | `TBD` | `evals/judge_calibration/` (TBD) | `placeholder` | Currently `tmp/image5.png`. Intra/inter-model LLM-judge agreement bars across OCEAN. |
| `main/fig_3_2_judge_validity.pdf` | `fig:judge-validity` | `sections/supervised.tex` | `TBD` | `evals/judge_calibration/` (TBD) | `placeholder` | Currently `tmp/image6.png`. Gemini 2.0 Flash confusion matrices against human gold labels. |
| `main/fig_3_3_1_scaling_trait.pdf` | `fig:scaling-trait` | `sections/supervised.tex` | `src_dev/visualisations/plot_scaling.py` | `fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v*/evals/mcq/trait/` | `placeholder` | Currently `tmp/image7.png`. Subfig of `fig:scaling`. |
| `main/fig_3_3_1_scaling_judge.pdf` | `fig:scaling-judge` | `sections/supervised.tex` | `src_dev/visualisations/plot_judge_sweep.py` | `fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v*/evals/judge/` (TBD) | `placeholder` | Currently `tmp/image8.png`. Subfig of `fig:scaling`. |
| `main/fig_3_3_1_scaling_capability.pdf` | `fig:scaling-capability` | `sections/supervised.tex` | `src_dev/visualisations/plot_scaling.py` | `fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v*/evals/mcq/mmlu/` | `placeholder` | Currently `tmp/image9.png`. Coherence/MMLU at various C- scales. |
| `main/fig_3_3_1_combination_trait.pdf` | `fig:combination-trait` | `sections/supervised.tex` | `scripts_dev/personality_evals/plot_soup_a_plus_minus.py` | `evals/neuro_x_consc_combos/` (TBD) | `placeholder` | Currently `tmp/image10.png`. N+/C-/combo barplot, TRAIT scores. |
| `main/fig_3_3_1_combination_judge.pdf` | `fig:combination-judge` | `sections/supervised.tex` | `scripts_dev/personality_evals/plot_soup_a_plus_minus.py` | `evals/neuro_x_consc_combos/` (TBD) | `placeholder` | Currently `tmp/image11.png`. N+/C-/combo barplot, LLM-judge scores. |
| `main/fig_3_3_1_combination_heatmap_trait.pdf` | `fig:combination-heatmap-trait` | `sections/supervised.tex` | `TBD` | `evals/consc_x_agree_combos/` (TBD) | `placeholder` | Currently `tmp/image12.png`. Subfig of `fig:combination-heatmap`. C- × A- TRAIT heatmap. |
| `main/fig_3_3_1_combination_heatmap_judge.pdf` | `fig:combination-heatmap-judge` | `sections/supervised.tex` | `TBD` | `evals/consc_x_agree_combos/` (TBD) | `placeholder` | Currently `tmp/image13.png`. Subfig of `fig:combination-heatmap`. C- × A- judge heatmap. |
| `main/fig_3_3_2_activation_coherence.pdf` | `fig:activation-coherence` | `sections/supervised.tex` | `src_dev/visualisations/plot_rollout_sweep.py` | `evals/activation_vs_lora_t_letter/` (TBD) | `placeholder` | Currently `tmp/image14.png`. Subfig of `fig:activation-comparison`. t-letter trait, coherence. |
| `main/fig_3_3_2_activation_length.pdf` | `fig:activation-length` | `sections/supervised.tex` | `src_dev/visualisations/plot_rollout_sweep.py` | `evals/activation_vs_lora_t_letter/` (TBD) | `placeholder` | Currently `tmp/image15.png`. Subfig of `fig:activation-comparison`. t-letter trait, answer length. |
| `main/fig_3_3_2_activation_adversarial.pdf` | `fig:activation-adversarial` | `sections/supervised.tex` | `src_dev/visualisations/plot_scaling.py` | `evals/t_letter_adversarial_prompt/` (TBD) | `placeholder` | Currently `tmp/image16.png`. t-density vs LoRA scale under adversarial prompting. |
| `main/fig_3_4_pca_loras.pdf` | `fig:pca-loras` | `sections/supervised.tex` | `TBD` | `fine_tuning/llama-3.1-8b-it/ocean/*/*/v*/lora/` (adapter weights) | `placeholder` | Currently `tmp/image17.png`. LaTeX TODO flags "replace with OCEAN amplifier/suppressor LoRAs". PCA of flattened LoRA weights. |
| `main/fig_3_4_cosine_sim_loras.pdf` | `fig:cosine-sim-loras` | `sections/supervised.tex` | `TBD` | `fine_tuning/llama-3.1-8b-it/ocean/*/*/v*/lora/` (adapter weights) | `placeholder` | Currently `tmp/image18.png`. LaTeX TODO flags "replace with OCEAN amplifier/suppressor LoRAs". Cosine-sim matrix. |

### Section 4 — Unsupervised (`sections/unsupervised.tex`)

| Path | Ref | Section | Script | Data source | Status | Notes |
|------|-----|---------|--------|-------------|--------|-------|
| `unsupervised/fig_4_1_parallel_analysis_residualised.pdf` | `fig:parallel-analysis-residualised` | `sections/unsupervised.tex` | `scripts_dev/unsupervised_embeddings/visualise_embeddings.py` | `TBD — embeddings artifact path` | `placeholder` | Currently `tmp/image19.png`. Subfig of `fig:parallel-analysis`. |
| `unsupervised/fig_4_1_parallel_analysis_non_residualised.pdf` | `fig:parallel-analysis-non-residualised` | `sections/unsupervised.tex` | `scripts_dev/unsupervised_embeddings/visualise_embeddings.py` | `TBD — embeddings artifact path` | `placeholder` | Currently `tmp/image20.png`. Subfig of `fig:parallel-analysis`. |
| `unsupervised/fig_4_1_prompt_group_variance.pdf` | `fig:histograms-of-prompt-group-variance-calculated-as-` | `sections/unsupervised.tex` | `scripts_dev/unsupervised_embeddings/visualise_embeddings.py` | `TBD — embeddings artifact path` | `placeholder` | Currently `tmp/image21.png`. Histograms of prompt-group-variance across factors. Label has markdown-converter cruft (rename). |
| `unsupervised/fig_4_1_factor28_agreeableness.pdf` | `fig:histogram-of-scores-for-factor-28-obtained-when-40` | `sections/unsupervised.tex` | `scripts_dev/unsupervised_embeddings/visualise_embeddings.py` | `TBD — embeddings artifact path` | `placeholder` | Currently `tmp/image22.png`. Factor-28 histogram separating agreeable vs disagreeable. Label has markdown-converter cruft (rename). |
| `unsupervised/fig_4_2_horn_parallel_rollouts.pdf` | `fig:horn-s-parallel-analysis-as-applied-to-the-1-517-p` | `sections/unsupervised.tex` | `scripts_dev/unsupervised_embeddings/psychometric_rollout_fa.py` | `TBD — rollout FA artifact path` | `placeholder` | Currently `tmp/image23.png`. Horn's parallel analysis on 1,517 persona rollouts. Label has markdown-converter cruft (rename). |

### Appendix F — OCEAN results (`appendices/ocean_results.tex`)

| Path | Ref | Section | Script | Data source | Status | Notes |
|------|-----|---------|--------|-------------|--------|-------|
| `appendix/fig_F_o_plus_trait.pdf` | `fig:F-o-plus-trait` | `appendices/ocean_results.tex` | `scripts_dev/personality_evals/plot_hf_personas.py` | `fine_tuning/llama-3.1-8b-it/ocean/openness/amplifier/v*/evals/mcq/trait/` | `placeholder` | Currently `tmp/image24.png`. Subfig of `fig:F-o-plus`. |
| `appendix/fig_F_o_plus_mmlu.pdf` | `fig:F-o-plus-mmlu` | `appendices/ocean_results.tex` | `scripts_dev/personality_evals/plot_hf_personas.py` | `fine_tuning/llama-3.1-8b-it/ocean/openness/amplifier/v*/evals/mcq/mmlu/` | `placeholder` | Currently `tmp/image25.png`. Subfig of `fig:F-o-plus`. |
| `appendix/fig_F_o_minus_trait.pdf` | `fig:F-o-minus-trait` | `appendices/ocean_results.tex` | `scripts_dev/personality_evals/plot_hf_personas.py` | `fine_tuning/llama-3.1-8b-it/ocean/openness/suppressor/v*/evals/mcq/trait/` | `placeholder` | Currently `tmp/image26.png`. Subfig of `fig:F-o-minus`. |
| `appendix/fig_F_o_minus_mmlu.pdf` | `fig:F-o-minus-mmlu` | `appendices/ocean_results.tex` | `scripts_dev/personality_evals/plot_hf_personas.py` | `fine_tuning/llama-3.1-8b-it/ocean/openness/suppressor/v*/evals/mcq/mmlu/` | `placeholder` | Currently `tmp/image27.png`. Subfig of `fig:F-o-minus`. |
| `appendix/fig_F_c_plus_trait.pdf` | `fig:F-c-plus-trait` | `appendices/ocean_results.tex` | `scripts_dev/personality_evals/plot_hf_personas.py` | `fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/amplifier/v*/evals/mcq/trait/` | `placeholder` | Currently `tmp/image28.png`. Subfig of `fig:F-c-plus`. |
| `appendix/fig_F_c_plus_mmlu.pdf` | `fig:F-c-plus-mmlu` | `appendices/ocean_results.tex` | `scripts_dev/personality_evals/plot_hf_personas.py` | `fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/amplifier/v*/evals/mcq/mmlu/` | `placeholder` | Currently `tmp/image29.png`. Subfig of `fig:F-c-plus`. |
| `appendix/fig_F_c_minus_trait.pdf` | `fig:F-c-minus-trait` | `appendices/ocean_results.tex` | `scripts_dev/personality_evals/plot_hf_personas.py` | `fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v*/evals/mcq/trait/` | `placeholder` | Currently `tmp/image30.png`. Subfig of `fig:F-c-minus`. |
| `appendix/fig_F_c_minus_mmlu.pdf` | `fig:F-c-minus-mmlu` | `appendices/ocean_results.tex` | `scripts_dev/personality_evals/plot_hf_personas.py` | `fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v*/evals/mcq/mmlu/` | `placeholder` | Currently `tmp/image31.png`. Subfig of `fig:F-c-minus`. |
| `appendix/fig_F_e_plus_trait.pdf` | `fig:F-e-plus-trait` | `appendices/ocean_results.tex` | `scripts_dev/personality_evals/plot_hf_personas.py` | `fine_tuning/llama-3.1-8b-it/ocean/extraversion/amplifier/v*/evals/mcq/trait/` | `placeholder` | Currently `tmp/image32.png`. Subfig of `fig:F-e-plus`. |
| `appendix/fig_F_e_plus_mmlu.pdf` | `fig:F-e-plus-mmlu` | `appendices/ocean_results.tex` | `scripts_dev/personality_evals/plot_hf_personas.py` | `fine_tuning/llama-3.1-8b-it/ocean/extraversion/amplifier/v*/evals/mcq/mmlu/` | `placeholder` | Currently `tmp/image33.png`. Subfig of `fig:F-e-plus`. |
| `appendix/fig_F_e_minus_trait.pdf` | `fig:F-e-minus-trait` | `appendices/ocean_results.tex` | `scripts_dev/personality_evals/plot_hf_personas.py` | `fine_tuning/llama-3.1-8b-it/ocean/extraversion/suppressor/v*/evals/mcq/trait/` | `placeholder` | Currently `tmp/image34.png`. Subfig of `fig:F-e-minus`. |
| `appendix/fig_F_e_minus_mmlu.pdf` | `fig:F-e-minus-mmlu` | `appendices/ocean_results.tex` | `scripts_dev/personality_evals/plot_hf_personas.py` | `fine_tuning/llama-3.1-8b-it/ocean/extraversion/suppressor/v*/evals/mcq/mmlu/` | `placeholder` | Currently `tmp/image35.png`. Subfig of `fig:F-e-minus`. |
| `appendix/fig_F_a_plus_trait.pdf` | `fig:F-a-plus-trait` | `appendices/ocean_results.tex` | `scripts_dev/personality_evals/plot_hf_personas.py` | `fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/v*/evals/mcq/trait/` | `placeholder` | Currently `tmp/image36.png`. Subfig of `fig:F-a-plus`. |
| `appendix/fig_F_a_plus_mmlu.pdf` | `fig:F-a-plus-mmlu` | `appendices/ocean_results.tex` | `scripts_dev/personality_evals/plot_hf_personas.py` | `fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/v*/evals/mcq/mmlu/` | `placeholder` | Currently `tmp/image37.png`. Subfig of `fig:F-a-plus`. |
| `appendix/fig_F_a_minus_trait.pdf` | `fig:F-a-minus-trait` | `appendices/ocean_results.tex` | `scripts_dev/personality_evals/plot_hf_personas.py` | `fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/v*/evals/mcq/trait/` | `placeholder` | Currently `tmp/image38.png`. Subfig of `fig:F-a-minus`. |
| `appendix/fig_F_a_minus_mmlu.pdf` | `fig:F-a-minus-mmlu` | `appendices/ocean_results.tex` | `scripts_dev/personality_evals/plot_hf_personas.py` | `fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/v*/evals/mcq/mmlu/` | `placeholder` | Currently `tmp/image39.png`. Subfig of `fig:F-a-minus`. |
| `appendix/fig_F_n_plus_trait.pdf` | `fig:F-n-plus-trait` | `appendices/ocean_results.tex` | `scripts_dev/personality_evals/plot_hf_personas.py` | `fine_tuning/llama-3.1-8b-it/ocean/neuroticism/amplifier/v*/evals/mcq/trait/` | `placeholder` | Currently `tmp/image40.png`. Subfig of `fig:F-n-plus`. |
| `appendix/fig_F_n_plus_mmlu.pdf` | `fig:F-n-plus-mmlu` | `appendices/ocean_results.tex` | `scripts_dev/personality_evals/plot_hf_personas.py` | `fine_tuning/llama-3.1-8b-it/ocean/neuroticism/amplifier/v*/evals/mcq/mmlu/` | `placeholder` | Currently `tmp/image41.png`. Subfig of `fig:F-n-plus`. |
| `appendix/fig_F_n_minus_trait.pdf` | `fig:F-n-minus-trait` | `appendices/ocean_results.tex` | `scripts_dev/personality_evals/plot_hf_personas.py` | `fine_tuning/llama-3.1-8b-it/ocean/neuroticism/suppressor/v*/evals/mcq/trait/` | `placeholder` | Currently `tmp/image42.png`. Subfig of `fig:F-n-minus`. |
| `appendix/fig_F_n_minus_mmlu.pdf` | `fig:F-n-minus-mmlu` | `appendices/ocean_results.tex` | `scripts_dev/personality_evals/plot_hf_personas.py` | `fine_tuning/llama-3.1-8b-it/ocean/neuroticism/suppressor/v*/evals/mcq/mmlu/` | `placeholder` | Currently `tmp/image43.png`. Subfig of `fig:F-n-minus`. |
| `appendix/fig_F_agree_combine_trait.pdf` | `fig:F-agree-combine-trait` | `appendices/ocean_results.tex` | `scripts_dev/personality_evals/plot_soup_a_plus_minus.py` | `evals/agree_plus_minus_combo/` (TBD) | `placeholder` | Currently `tmp/image44.png`. Subfig of `fig:F-agree-combine`. A+/A- scale sweep. |
| `appendix/fig_F_agree_combine_compare.pdf` | `fig:F-agree-combine-compare` | `appendices/ocean_results.tex` | `scripts_dev/personality_evals/plot_soup_a_plus_minus.py` | `evals/agree_plus_minus_combo/` (TBD) | `placeholder` | Currently `tmp/image45.png`. Subfig of `fig:F-agree-combine`. Comparison to base model. |

### Appendix H — TRAIT metrics of Open Character (`appendices/trait_metrics.tex`)

| Path | Ref | Section | Script | Data source | Status | Notes |
|------|-----|---------|--------|-------------|--------|-------|
| `appendix/fig_H_openchar_heatmap.pdf` | `fig:H-openchar-heatmap` | `appendices/trait_metrics.tex` | `TBD` | `evals/open_character_trait_regression/` (TBD) | `placeholder` | Currently `tmp/image46.png`. OLS regression slopes heatmap across 13 OCT personas. |
| `appendix/fig_H_openchar_scaling.pdf` | `fig:H-openchar-scaling` | `appendices/trait_metrics.tex` | `scripts_dev/personality_evals/plot_hf_personas.py` | `evals/open_character_trait_sweep/` (TBD) | `placeholder` | Currently `tmp/image47.png`. OCT persona TRAIT scaling (batch 1). |
| `appendix/fig_H_openchar_scaling_2.pdf` | `fig:H-openchar-scaling-2` | `appendices/trait_metrics.tex` | `scripts_dev/personality_evals/plot_hf_personas.py` | `evals/open_character_trait_sweep/` (TBD) | `placeholder` | Currently `tmp/image48.png`. OCT persona TRAIT scaling (batch 2). |

### Appendix I — Alternative training methods (`appendices/alternative_training.tex`)

| Path | Ref | Section | Script | Data source | Status | Notes |
|------|-----|---------|--------|-------------|--------|-------|
| `appendix/fig_I_sft_only_trait.pdf` | `fig:I-sft-only-trait` | `appendices/alternative_training.tex` | `src_dev/visualisations/plot_scaling.py` | `fine_tuning/llama-3.1-8b-it/ocean/{trait}/{direction}/v*-sft-only/evals/mcq/trait/` (TBD) | `placeholder` | Currently `tmp/image49.png`. SFT-only pipeline TRAIT scaling. |
