"""Small validation-only tuning and resume checks; no model training."""
from __future__ import annotations

import json

import numpy as np
import pytest

import repro_hyper_tuning as tuning
from experiment_data import DOMAIN, PROTOCOL_VERSION, sha256_file


def _small_manifest(tmp_path):
    grid_dir = tmp_path / "grids"
    grid_dir.mkdir()
    entries = []
    for index, split in enumerate(("train", "val", "val", "test")):
        sample_id = f"sample-{index}"
        volume = np.zeros((8, 8, 8), dtype=np.uint8)
        volume[2:5, 2:5, 2:5] = 1
        volume[0, 0, index] = 1
        path = grid_dir / f"{sample_id}.npy"
        np.save(path, volume)
        entries.append({
            "sample_id": sample_id,
            "parent_id": f"parent-{index}",
            "split": split,
            "path": f"grids/{sample_id}.npy",
            "sha256": sha256_file(path),
            "domain": DOMAIN,
        })
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "protocol_version": PROTOCOL_VERSION,
        "domain": DOMAIN,
        "count": 4,
        "grid_size": [8, 8, 8],
        "split_counts": {"train": 1, "val": 2, "test": 1},
        "samples": entries,
    }), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "grid_size": [8, 8, 8],
        "num_holes": 2,
        "samples_per_hole": 2,
        "mask_seed": 42,
        "device": "cpu",
        "threshold": 0.5,
    }), encoding="utf-8")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"stub checkpoint; model loader is replaced")
    return manifest, config, checkpoint


def test_selection_is_parent_disjoint_and_order_independent():
    samples = [
        {"split": "val", "parent_id": f"parent-{parent}",
         "sample_id": f"p{parent}-variant-{variant}"}
        for parent in range(60) for variant in range(3)
    ]
    samples.append({"split": "test", "parent_id": "test-parent", "sample_id": "test-case"})
    first = tuning.select_validation_cases({"samples": samples})
    second = tuning.select_validation_cases({"samples": list(reversed(samples))})
    assert first == second
    assert len(first) == 50
    assert len({sample.split("-variant-")[0] for sample in first}) == 50
    assert "test-case" not in first
    grid = tuning.candidate_grid()
    assert len(grid) == 23
    for method, key in (("geovae_latent", "latent_test_time_steps"),
                        ("geovae_joint", "joint_test_time_steps")):
        assert {row["overrides"][key] for row in grid if row["method"] == method} == {25, 50, 100}


def test_candidate_ranking_uses_blind_iou_then_brier():
    candidates = [
        {"method": "idw", "candidate_id": "idw-00", "ordinal": 0,
         "overrides": {"idw_power": 1}, "historical_failed_attempts": 0,
         "scores": [{"blind_iou": 0.7, "brier_blind": 0.3}]},
        {"method": "idw", "candidate_id": "idw-01", "ordinal": 1,
         "overrides": {"idw_power": 2}, "historical_failed_attempts": 0,
         "scores": [{"blind_iou": 0.7, "brier_blind": 0.2}]},
        {"method": "idw", "candidate_id": "idw-02", "ordinal": 2,
         "overrides": {"idw_power": 3}, "historical_failed_attempts": 0,
         "scores": [{"blind_iou": 0.6, "brier_blind": 0.01}]},
    ]
    selected, summaries = tuning._rank_candidates(candidates, 1)
    assert selected["idw"]["candidate_id"] == "idw-01"
    assert len(summaries) == 3


def test_tune_journals_failure_and_reuses_cases(tmp_path, monkeypatch):
    manifest, config, checkpoint = _small_manifest(tmp_path)
    full_grid = tuning.candidate_grid()
    methods = ("idw", "rbf", "ik", "mps", "geovae_latent", "geovae_joint")
    monkeypatch.setattr(tuning, "candidate_grid", lambda: [
        next(row for row in full_grid if row["method"] == method)
        for method in methods
    ])
    monkeypatch.setattr(tuning, "select_validation_cases",
                        lambda data: ["sample-1", "sample-2"])
    monkeypatch.setattr(tuning.importlib.util, "find_spec", lambda name: True)
    monkeypatch.setattr(tuning, "_load_checked_model",
                        lambda *args: (object(), {}))
    calls = []
    fail_once = {"value": True}

    def predict(method, obs, models, settings, device, training_image, case_seed):
        calls.append(method)
        if fail_once["value"]:
            fail_once["value"] = False
            raise RuntimeError("temporary issue")
        return np.full(obs.shape[1:], 0.5, dtype=np.float32), {}

    monkeypatch.setattr(tuning, "_predict", predict)
    output = tmp_path / "tuning"
    with pytest.raises(RuntimeError, match="Validation tuning failed"):
        tuning.tune(str(manifest), str(config), str(output), str(checkpoint))
    first_journal = output / "journal" / "idw" / "idw-00" / "sample-1.json"
    assert json.loads(first_journal.read_text(encoding="utf-8"))["status"] == "failed"

    selected_path = tuning.tune(str(manifest), str(config), str(output), str(checkpoint))
    assert selected_path.is_file()
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    assert selection["status"] == "validation_selected"
    assert selection["case_ids"] == ["sample-1", "sample-2"]
    assert selection["selected_config_sha256"] == sha256_file(selected_path)
    assert len(selection["candidates"]) == len(methods)
    assert len(calls) == 1 + 2 * len(methods)
    assert len(json.loads(first_journal.read_text(encoding="utf-8"))["failure_history"]) == 1

    calls.clear()
    tuning.tune(str(manifest), str(config), str(output), str(checkpoint))
    assert calls == []
    config_data = json.loads(config.read_text(encoding="utf-8"))
    config_data["idw_power"] = 7
    config.write_text(json.dumps(config_data), encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity differs"):
        tuning.tune(str(manifest), str(config), str(output), str(checkpoint))
