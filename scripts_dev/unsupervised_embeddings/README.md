# Unsupervised Embedding Workflow

Three official experiment entrypoints live here:

- `import_response_run.py` for one-off migration of an existing canonical run from another dataset repo
- `generate_single_turn_responses.py`
- `embed_responses.py`
- `visualise_embeddings.py`

## Lineage

- Base response runs live locally at `scratch/runs/<response_run_id>/`
- The shared Hugging Face dataset repo defaults to `persona-shattering-lasr/unsupervised-runs`
- Hub layout:
  - `runs/<response_run_id>/run/...`
  - `runs/<response_run_id>/embeddings/<embedding_slug>/...`
  - `runs/<response_run_id>/visualisations/<viz_slug>/...`

## Typical Flow

1. Generate a canonical single-turn response run.
2. Derive one or more embedding artifacts from that same response run.
3. Open `visualise_embeddings.py` in notebook mode and point it at one or more embedding artifacts.

Each script prefers local cached artifacts and hydrates missing inputs from Hugging Face when possible. Upload is enabled by default; use `--no-hf-upload` to keep a run local-only.
