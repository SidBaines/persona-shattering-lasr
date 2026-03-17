"""Tests for unsupervised-run pathing and local/HF hydration helpers."""

from __future__ import annotations

from pathlib import Path

from scripts.unsupervised_runs.io import (
    build_embedding_slug,
    embedding_artifact_dir,
    embedding_artifact_hf_path,
    ensure_response_run,
    hydrate_dataset_subtree,
    response_run_dir,
    response_run_hf_path,
    visualisation_artifact_hf_path,
)


def test_path_helpers_are_stable() -> None:
    run_id = "demo-run"
    slug = build_embedding_slug(
        model="Qwen/Qwen3-Embedding-4B",
        analysis_unit="assistant_final_turn",
        normalize=True,
        max_length=4000,
    )

    assert response_run_dir(run_id) == Path("scratch") / "runs" / run_id
    assert response_run_hf_path(run_id) == "runs/demo-run/run"
    assert embedding_artifact_hf_path(run_id, slug) == f"runs/demo-run/embeddings/{slug}"
    assert visualisation_artifact_hf_path(run_id, "viz-a") == "runs/demo-run/visualisations/viz-a"
    assert embedding_artifact_dir(response_run_dir(run_id), slug) == (
        Path("scratch") / "runs" / run_id / "reports" / "embeddings" / slug
    )


def test_hydrate_dataset_subtree_copies_to_local_cache(tmp_path, monkeypatch) -> None:
    repo_id = "persona-shattering-lasr/unsupervised-runs"
    path_in_repo = "runs/demo/run"
    cache_root = tmp_path / "hf-cache"
    src_manifest = cache_root / "manifest.json"
    src_manifest.parent.mkdir(parents=True, exist_ok=True)
    src_manifest.write_text('{"run_id": "demo"}', encoding="utf-8")

    def _fake_list_repo_files(*, repo_id: str, repo_type: str):
        assert repo_id == "persona-shattering-lasr/unsupervised-runs"
        assert repo_type == "dataset"
        return [
            "runs/demo/run/manifest.json",
            "runs/demo/run/datasets/canonical_samples.jsonl",
        ]

    canonical_file = cache_root / "canonical_samples.jsonl"
    canonical_file.write_text('{"sample_id":"s1"}\n', encoding="utf-8")

    def _fake_download(*, repo_id: str, repo_type: str, filename: str):
        assert repo_id == "persona-shattering-lasr/unsupervised-runs"
        assert repo_type == "dataset"
        if filename.endswith("manifest.json"):
            return str(src_manifest)
        if filename.endswith("canonical_samples.jsonl"):
            return str(canonical_file)
        raise AssertionError(filename)

    class _FakeApi:
        def list_repo_files(self, *, repo_id: str, repo_type: str):
            return _fake_list_repo_files(repo_id=repo_id, repo_type=repo_type)

    monkeypatch.setattr("scripts.unsupervised_runs.io.HfApi", lambda: _FakeApi())
    monkeypatch.setattr("scripts.unsupervised_runs.io.hf_hub_download", _fake_download)

    local_dir = tmp_path / "scratch" / "runs" / "demo"
    hydrated = hydrate_dataset_subtree(
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        local_dir=local_dir,
        required=True,
    )

    assert hydrated
    assert (local_dir / "manifest.json").read_text(encoding="utf-8") == '{"run_id": "demo"}'
    assert (local_dir / "datasets" / "canonical_samples.jsonl").exists()


def test_ensure_response_run_prefers_existing_local_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scripts.unsupervised_runs.io.SCRATCH_RUNS_DIR", tmp_path / "runs")
    run_dir = tmp_path / "runs" / "demo"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")

    called = {"hydrate": False}

    def _fake_hydrate(**_kwargs):
        called["hydrate"] = True
        return True

    monkeypatch.setattr("scripts.unsupervised_runs.io.hydrate_dataset_subtree", _fake_hydrate)
    resolved = ensure_response_run("demo", required=False)
    assert resolved == run_dir
    assert not called["hydrate"]
