"""Regression tests for prompted_trait_sanity_check run validation."""

from __future__ import annotations

from src_dev.datasets import ingest_source_dataset, load_samples, write_inference_result
from scripts_dev.unsupervised_embeddings import prompted_trait_sanity_check as script


def test_ensure_response_run_available_resumes_seeded_local_run(tmp_path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(script, "response_run_dir", lambda response_run_id: runs_root / response_run_id)
    monkeypatch.setattr(script, "_hf_response_run_exists", lambda _response_run_id: False)

    condition = script.PromptConditionConfig(
        name="trait",
        response_run_id="seeded-run",
        generation=script.RunGenerationConfig(
            provider="local",
            model="test-model",
            num_responses=1,
            max_attempts_per_sample=3,
        ),
    )
    run_dir = script.response_run_dir(condition.response_run_id)

    samples = ingest_source_dataset(
        [{"question": "Q1"}, {"question": "Q2"}],
        source_info={"source": "test"},
        system_prompt=None,
        run_dir=run_dir,
        overwrite=True,
    )

    seeded_status = script._local_response_run_status(condition.response_run_id)
    assert seeded_status.total_rows == 2
    assert seeded_status.complete_rows == 0
    assert seeded_status.pending_rows == 2
    assert not seeded_status.is_usable

    def _fake_generate_response_run(configured_condition: script.PromptConditionConfig) -> None:
        assert configured_condition.response_run_id == condition.response_run_id
        for idx, sample in enumerate(load_samples(run_dir)):
            write_inference_result(
                run_dir,
                sample.sample_id,
                {
                    "status": "success",
                    "model": "test-model",
                    "provider": "local",
                    "assistant_message_id": f"msg-{idx}",
                    "assistant_completion": f"response-{idx}",
                    "assistant_full": f"response-{idx}",
                    "assistant_message_metadata": {"turn_index": 0},
                    "attempt_no": 1,
                },
                materialize=False,
            )

    uploads: list[str] = []
    monkeypatch.setattr(script, "_generate_response_run", _fake_generate_response_run)
    monkeypatch.setattr(script, "upload_response_run", lambda response_run_id, repo_id=None: uploads.append(response_run_id))

    returned_run_dir = script._ensure_response_run_available(condition)

    assert returned_run_dir == run_dir
    assert uploads == [condition.response_run_id]

    completed_status = script._local_response_run_status(condition.response_run_id)
    assert completed_status.complete_rows == len(samples)
    assert completed_status.pending_rows == 0
    assert completed_status.is_usable
