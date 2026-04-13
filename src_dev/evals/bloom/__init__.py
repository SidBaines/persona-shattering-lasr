"""Bloom cell-sweep eval: content-addressed caching for behavioral evals.

Ports the bloom pipeline (understanding → ideation → rollout → judgment)
onto the :mod:`src_dev.evals.cell_sweep` framework used by the LLM-judge and
TRAIT-logprobs sweeps.

Two cache layers:

- **Ideation** (trait-scoped, manually versioned):
  ``evals/bloom_ideation/{trait}/v{N}/{ideation_fp}/``
- **Rollout + judgment** (per-cell, content-addressed):
  ``<cell_hf_dir>/{rollouts,judge_runs/{judge}/{quality}.json,…}``
"""
