"""End-to-end subset identity, mask variation and resume checks."""
import csv
import json
import time

import numpy as np
import pytest

from experiment_data import prepare_protocol
from repro_pipeline import save_json, sha256_file, source_hash, stable_hash
from revision_robustness import _condition, _conditions, run


@pytest.fixture
def bundle(tmp_path):
    manifest = prepare_protocol(str(tmp_path), str(tmp_path / "protocol"), count=20,
                                seed=42, grid_size=(16, 16, 16), audit_raw=False)
    root = tmp_path / "run"
    root.mkdir()
    config = {"grid_size": [16] * 3, "mask_seed": 42, "num_holes": 8,
              "samples_per_hole": 4, "device": "cpu", "threshold": .5, "latent_dim": 16}
    config_path = root / "config.json"
    save_json(config_path, config)
    save_json(root / "thresholds.json", {
        "manifest_sha256": sha256_file(manifest), "config_sha256": stable_hash(config),
        "evaluation_source_sha256": source_hash(), "status": "draft",
        "thresholds": {"idw": .5, "rbf": .5}})
    return manifest, root, config_path


def test_conditions_deduplicate_eighty_percent_reference():
    conditions = _conditions()
    assert len(conditions) == 7
    assert len({row["condition"] for row in conditions}) == 7
    assert sum(row["holes"] == 8 and row["evaluation_mask_seed"] == 42 for row in conditions) == 1
    retention = [row["retention_fraction"] for row in conditions if row["evaluation_mask_seed"] == 42]
    assert sorted(retention) == [.2, .4, .6, .8, 1]


def test_smoke_all_conditions_and_resume_without_prediction(bundle, monkeypatch):
    manifest, root, config_path = bundle
    result = run(manifest, root, config_path, smoke=True, methods=("idw", "rbf"), include_noise=False)
    assert result.is_file()
    status = json.loads((result.parent / "status.json").read_text())
    assert status["status"] == "draft_smoke"
    assert status["case_method_runs"] == 14
    assert status["case_count"] == 1
    per_case_path = result.parent / "per_case.csv"
    before = per_case_path.read_bytes()
    with per_case_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 14
    assert {row["training_mask_seed"] for row in rows} == {"42"}
    assert {row["evaluation_mask_seed"] for row in rows} == {"42", "53", "67"}
    assert all(row["input_mse"] for row in rows)

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed, verified prediction should be reused")

    monkeypatch.setattr("revision_robustness._predict", forbidden)
    again = run(manifest, root, config_path, smoke=True, methods=("idw", "rbf"), include_noise=False)
    assert again == result
    assert before == per_case_path.read_bytes()


def test_bad_prediction_records_failure_then_can_repair(bundle):
    manifest, root, config_path = bundle
    result = run(manifest, root, config_path, smoke=True, methods=("idw", "rbf"), include_noise=False)
    target = next((result.parent / _condition(8, 42)["condition"] / "predictions" / "idw").glob("*.npy"))
    data = np.load(target)
    data.flat[0] = 1 - data.flat[0]
    np.save(target, data)
    with pytest.raises(RuntimeError, match="incomplete"):
        run(manifest, root, config_path, smoke=True, methods=("idw", "rbf"), include_noise=False)
    failures = json.loads((result.parent / "failures.json").read_text())
    assert len(failures) == 1 and "checksum" in failures[0]["message"]
    run(manifest, root, config_path, smoke=True, methods=("idw", "rbf"), include_noise=False)
    assert json.loads((result.parent / "failures.json").read_text()) == []
    assert len(json.loads((result.parent / "failure_history.json").read_text())) == 1


def test_threshold_identity_and_expired_deadline_block_run(bundle):
    manifest, root, config_path = bundle
    with pytest.raises(TimeoutError):
        run(manifest, root, config_path, deadline_epoch=time.time() - 1,
            smoke=True, methods=("idw", "rbf"))
    thresholds = json.loads((root / "thresholds.json").read_text())
    thresholds["config_sha256"] = "wrong"
    save_json(root / "thresholds.json", thresholds)
    with pytest.raises(RuntimeError, match="identity"):
        run(manifest, root, config_path, smoke=True, methods=("idw", "rbf"))


def test_formal_cannot_use_small_or_partial_method_bundle(bundle):
    manifest, root, config_path = bundle
    with pytest.raises(ValueError, match="d256"):
        run(manifest, root, config_path, methods=("idw", "rbf"))
