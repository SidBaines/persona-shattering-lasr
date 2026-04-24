# Known Issues

Short, live list of bugs, latent issues, and caveats discovered during development. Agents (and humans) should skim this before starting work so known footguns aren't rediscovered from scratch.

**When you fix an entry, remove it.** Don't strike through, don't leave a "(fixed)" note — just delete. Stale entries are worse than no entries.

Entry format: what / where (file:line) / fix sketch. Keep it terse.

---

## ChatGLM hardcoded as teacher self-reference in DPO formatter

- **Where:** [scripts_dev/oct_pipeline/run_oct_pipeline.py:1920](scripts_dev/oct_pipeline/run_oct_pipeline.py#L1920) (OCT path) and [:1831](scripts_dev/oct_pipeline/run_oct_pipeline.py#L1831) `load_dpo_pairs` (non-OCT path, no sanitization at all).
- **Issue:** the string `"ChatGLM"` is hardcoded as the teacher self-reference to sanitize out of the chosen response. Silently no-ops when the teacher is changed to any non-GLM model, letting teacher self-references leak into training. Also only applied to `chosen`, not `rejected` — creates a tiny chosen/rejected asymmetry under paired-teacher DPO (≤0.3% of rows for vanton4 data; confined to a few rows in E/N).
- **Fix:** the helper [run_oct_pipeline.py:793-798](scripts_dev/oct_pipeline/run_oct_pipeline.py#L793-L798) `_teacher_assistant_name(model)` already derives the right name. Thread `teacher_model` into `format_dpo_data_for_oct_training`, compute `teacher_name = _teacher_assistant_name(teacher_model)`, and apply `.replace(teacher_name, name)` on both `chosen` and `rejected`. Mirror into `load_dpo_pairs` for the non-OCT path. ~6–8 lines total.
