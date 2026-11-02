"""Restart and integrity checks for the atomic evaluation journal."""
from __future__ import annotations

import csv
import json

import pytest

import repro_pipeline
from experiment_data import prepare_protocol
from repro_pipeline import evaluate, stable_hash
from repro_report import _read_run


def _config(path):
    path.write_text(
        "\n".join([
            "grid_size: [16, 16, 16]",
            "num_holes: 2",
            "samples_per_hole: 4",
            "mask_seed: 42",
            "device: cpu",
            "threshold: 0.5",
        ]) + "\n",
        encoding="utf-8",
    )


def _setup(tmp_path):
    manifest = prepare_protocol(
        str(tmp_path), str(tmp_path / "protocol"), count=20,
        seed=42, grid_size=(16, 16, 16), audit_raw=False,
    )
    config = tmp_path / "config.yaml"
    _config(config)
    protocol = json.loads((tmp_path / "protocol" / "manifest.json").read_text(encoding="utf-8"))
    sample_ids = [row["sample_id"] for row in protocol["samples"] if row["split"] == "val"][:2]
    assert len(sample_ids) == 2
    return manifest, config, sample_ids


def test_interrupted_evaluation_reuses_valid_cases_and_checks_identity(tmp_path, monkeypatch):
    manifest, config, sample_ids = _setup(tmp_path)
    out = tmp_path / "evaluation"
    original = repro_pipeline._predict
    attempts = []

    def interrupt_after_first(method, obs, models, settings, device, training_image, case_seed):
        attempts.append(method)
        if len(attempts) == 2:
            raise KeyboardInterrupt
        return original(method, obs, models, settings, device, training_image, case_seed)

    monkeypatch.setattr(repro_pipeline, "_predict", interrupt_after_first)
    with pytest.raises(KeyboardInterrupt):
        evaluate(manifest, str(config), str(out), "val", ["idw"], {},
                 seed=11, sample_ids=sample_ids)
    assert (out / "journal" / "idw" / f"{sample_ids[0]}.json").is_file()
    assert not json.loads((out / "evaluation.json").read_text(encoding="utf-8"))["complete"]

    resumed = []
    def count_prediction(method, obs, models, settings, device, training_image, case_seed):
        resumed.append(method)
        return original(method, obs, models, settings, device, training_image, case_seed)

    monkeypatch.setattr(repro_pipeline, "_predict", count_prediction)
    evaluate(manifest, str(config), str(out), "val", ["idw"], {},
             seed=11, sample_ids=sample_ids)
    assert resumed == ["idw"]
    metadata, rows = _read_run(out)
    assert metadata["case_selection_sha256"] == stable_hash(sample_ids)
    assert [row["sample_id"] for row in rows] == sample_ids

    resumed.clear()
    evaluate(manifest, str(config), str(out), "val", ["idw"], {},
             seed=11, sample_ids=sample_ids)
    assert resumed == []

    with pytest.raises(RuntimeError, match="identity differs"):
        evaluate(manifest, str(config), str(out), "val", ["idw"], {},
                 seed=11, noise_rate=0.05, sample_ids=sample_ids)
    with pytest.raises(ValueError, match="limit=0"):
        evaluate(manifest, str(config), str(tmp_path / "bad"), "val", ["idw"], {},
                 seed=11, limit=1, sample_ids=sample_ids)

    prediction = out / rows[0]["prediction_file"]
    prediction.write_bytes(b"corrupted")
    resumed.clear()
    evaluate(manifest, str(config), str(out), "val", ["idw"], {},
             seed=11, sample_ids=sample_ids)
    assert resumed == ["idw"]
    _read_run(out)
    history = json.loads((out / "failure_history.json").read_text(encoding="utf-8"))
    assert any(item["error_type"] == "JournalValidationError" for item in history)


def test_failed_case_is_recorded_and_retried(tmp_path, monkeypatch):
    manifest, config, sample_ids = _setup(tmp_path)
    out = tmp_path / "failed"
    original = repro_pipeline._predict
    once = {"failed": False}

    def fail_once(method, obs, models, settings, device, training_image, case_seed):
        if not once["failed"]:
            once["failed"] = True
            raise RuntimeError("simulated transient failure")
        return original(method, obs, models, settings, device, training_image, case_seed)

    monkeypatch.setattr(repro_pipeline, "_predict", fail_once)
    with pytest.raises(RuntimeError, match="Evaluation incomplete"):
        evaluate(manifest, str(config), str(out), "val", ["idw"], {},
                 seed=11, sample_ids=sample_ids)
    assert len(json.loads((out / "failures.json").read_text(encoding="utf-8"))) == 1

    evaluate(manifest, str(config), str(out), "val", ["idw"], {},
             seed=11, sample_ids=sample_ids)
    _read_run(out)
    assert json.loads((out / "failures.json").read_text(encoding="utf-8")) == []
    history = json.loads((out / "failure_history.json").read_text(encoding="utf-8"))
    assert len(history) == 1
    assert history[0]["message"] == "simulated transient failure"
    with (out / "per_case.csv").open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 2