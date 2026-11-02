"""Shared prediction reuse must save work without changing scientific identity."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch

import repro_pipeline
import model as model_module
from experiment_data import prepare_protocol
from model import ConvVAE3D, adapt_to_observation
from repro_pipeline import evaluate, verify_prediction_cache
from repro_report import report


def _fixture(tmp_path):
    manifest = prepare_protocol(
        str(tmp_path), str(tmp_path / "protocol"), count=20,
        seed=42, grid_size=(16, 16, 16), audit_raw=False,
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "grid_size: [16, 16, 16]\nnum_holes: 2\nsamples_per_hole: 4\n"
        "mask_seed: 42\ndevice: cpu\nthreshold: 0.5\nidw_power: 2.0\n",
        encoding="utf-8",
    )
    protocol = json.loads((tmp_path / "protocol" / "manifest.json").read_text(encoding="utf-8"))
    sample_id = next(row["sample_id"] for row in protocol["samples"] if row["split"] == "val")
    return manifest, config, sample_id


def _row(folder):
    with (folder / "per_case.csv").open(newline="", encoding="utf-8") as stream:
        return next(csv.DictReader(stream))


def test_classical_cache_reuses_across_seeds_and_records_source(tmp_path, monkeypatch):
    manifest, config, sample_id = _fixture(tmp_path)
    cache = tmp_path / "prediction_cache"
    first = tmp_path / "seed11"
    second = tmp_path / "seed23"
    original = repro_pipeline._predict
    called = []

    def counted(*args):
        called.append(args[0])
        return original(*args)

    monkeypatch.setattr(repro_pipeline, "_predict", counted)
    evaluate(manifest, str(config), str(first), "val", ["idw"], {},
             seed=11, sample_ids=[sample_id], prediction_cache_root=str(cache))
    evaluate(manifest, str(config), str(second), "val", ["idw"], {},
             seed=23, sample_ids=[sample_id], prediction_cache_root=str(cache))
    assert called == ["idw"]
    first_row, second_row = _row(first), _row(second)
    assert first_row["prediction_sha256"] == second_row["prediction_sha256"]
    assert first_row["prediction_origin"] == "computed"
    assert second_row["prediction_origin"] == "shared_cache"
    assert second_row["prediction_source_file"] == "seed11/" + first_row["prediction_file"]
    meta = json.loads((second / "evaluation.json").read_text(encoding="utf-8"))
    assert meta["prediction_cache"]["shared_cache_hits"] == 1
    assert verify_prediction_cache(cache)["by_method"] == {"idw": 1}
    summary = report([str(first), str(second)], str(tmp_path / "report"), draft=True)
    with summary.open(newline="", encoding="utf-8") as stream:
        baseline = next(csv.DictReader(stream))
    assert baseline["case_runs"] == "2"
    assert baseline["unique_prediction_count"] == "1"
    assert baseline["independent_training_seeds"] == "0"
    assert baseline["blind_iou_n"] == "1"


def test_cache_key_changes_with_method_settings_and_observation(tmp_path, monkeypatch):
    manifest, config, sample_id = _fixture(tmp_path)
    cache = tmp_path / "prediction_cache"
    original = repro_pipeline._predict
    called = []

    def counted(*args):
        called.append(args[0])
        return original(*args)

    monkeypatch.setattr(repro_pipeline, "_predict", counted)
    evaluate(manifest, str(config), str(tmp_path / "baseline"), "val", ["idw"], {},
             seed=11, sample_ids=[sample_id], prediction_cache_root=str(cache))
    tuned = tmp_path / "tuned.yaml"
    tuned.write_text(config.read_text(encoding="utf-8").replace("idw_power: 2.0", "idw_power: 3.0"),
                     encoding="utf-8")
    evaluate(manifest, str(tuned), str(tmp_path / "different_power"), "val", ["idw"], {},
             seed=23, sample_ids=[sample_id], prediction_cache_root=str(cache))
    evaluate(manifest, str(config), str(tmp_path / "noisy"), "val", ["idw"], {},
             seed=23, noise_rate=0.2, sample_ids=[sample_id],
             prediction_cache_root=str(cache))
    assert called == ["idw", "idw", "idw"]


def test_cache_reuses_raw_prediction_when_projection_changes(tmp_path, monkeypatch):
    manifest, config, sample_id = _fixture(tmp_path)
    cache = tmp_path / "prediction_cache"
    projected = tmp_path / "projected"
    raw = tmp_path / "raw"
    original = repro_pipeline._predict
    called = []

    def counted(*args):
        called.append(args[0])
        return original(*args)

    monkeypatch.setattr(repro_pipeline, "_predict", counted)
    evaluate(manifest, str(config), str(projected), "val", ["idw"], {},
             seed=11, sample_ids=[sample_id], project_observed=True,
             prediction_cache_root=str(cache))
    evaluate(manifest, str(config), str(raw), "val", ["idw"], {},
             seed=23, sample_ids=[sample_id], prediction_cache_root=str(cache))
    assert called == ["idw"]
    np.testing.assert_array_equal(
        np.load(projected / _row(projected)["raw_prediction_file"]),
        np.load(raw / _row(raw)["prediction_file"]),
    )


def test_cache_rejects_tampered_source(tmp_path):
    manifest, config, sample_id = _fixture(tmp_path)
    cache = tmp_path / "prediction_cache"
    source = tmp_path / "source"
    evaluate(manifest, str(config), str(source), "val", ["idw"], {},
             seed=11, sample_ids=[sample_id], prediction_cache_root=str(cache))
    (source / _row(source)["prediction_file"]).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        verify_prediction_cache(cache)
    with pytest.raises(RuntimeError, match="Evaluation incomplete"):
        evaluate(manifest, str(config), str(tmp_path / "second"), "val", ["idw"], {},
                 seed=23, sample_ids=[sample_id], prediction_cache_root=str(cache))
    failures = json.loads((tmp_path / "second" / "failures.json").read_text(encoding="utf-8"))
    assert "checksum mismatch" in failures[0]["message"]


def test_latent_adaptation_does_not_copy_model_and_restores_flags(monkeypatch):
    torch.manual_seed(7)
    network = ConvVAE3D(grid_size=16, latent_dim=8, base_channels=4).train()
    before = {name: value.detach().clone() for name, value in network.state_dict().items()}
    obs = torch.zeros(1, 2, 16, 16, 16)
    obs[0, 1, 2, 2, 2] = 1
    obs[0, 0, 2, 2, 2] = 1

    def no_copy(*_args, **_kwargs):
        raise AssertionError("latent mode must not deepcopy the model")

    monkeypatch.setattr(model_module.copy, "deepcopy", no_copy)
    prediction, metadata = adapt_to_observation(network, obs, mode="latent", steps=1)
    assert prediction.shape == (1, 16, 16, 16)
    assert metadata["lora_trainable_parameters"] == 0
    assert network.training
    assert all(parameter.requires_grad for parameter in network.parameters())
    for name, value in network.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
