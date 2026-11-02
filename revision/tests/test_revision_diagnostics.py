"""Checks for leakage boundaries, actual retained samples and confidential export."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from experiment_data import prepare_protocol
from model import ConvVAE3D
import revision_diagnostics as diagnostic
from repro_pipeline import save_json, sha256_file, source_hash


torch.set_num_threads(1)


def _model_obs():
    torch.manual_seed(912)
    model = ConvVAE3D(grid_size=16, latent_dim=8, base_channels=4)
    obs = np.zeros((2, 16, 16, 16), dtype=np.float32)
    obs[1, 2, 2, 2] = 1
    obs[0, 2, 2, 2] = 1
    obs[1, 12, 12, 12] = 1
    return model, obs


def _config():
    return {"diagnostic_steps": 2, "diagnostic_patience": 50,
            "test_time_lora_rank": 2, "test_time_lr": 0.03,
            "test_time_lora_lr": 0.001, "test_time_prior_weight": 0.01,
            "test_time_lora_weight": 0.01}


def test_selection_is_parent_distinct_and_order_outcome_independent():
    entries = [{"sample_id": f"p{parent}v{variant}", "parent_id": f"p{parent}", "split": "test"}
               for parent in range(35) for variant in range(5)]
    first = diagnostic.select_parent_cases({"samples": entries}, "test", 30)
    changed = [{**row, "occupancy_fraction": 0.99, "metric": index} for index, row in enumerate(reversed(entries))]
    assert first == diagnostic.select_parent_cases({"samples": changed}, "test", 30)
    assert len({sample.split("v")[0] for sample in first}) == 30
    with pytest.raises(ValueError, match="available"):
        diagnostic.select_parent_cases({"samples": entries}, "test", 36)


def test_joint_optimization_preserves_checkpoint_and_receives_gradients():
    model, obs = _model_obs()
    model.train()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    result = diagnostic.optimize_observation(model, obs, _config(), seed=17)
    assert model.training
    assert torch.equal(rng, torch.random.get_rng_state())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(state[key], value, rtol=0, atol=0)
    assert result["metadata"]["gradient_checks"]["lora_first_norm"] > 0
    assert result["metadata"]["gradient_checks"]["latent_first_norm"] > 0
    assert result["lora_state"]
    for row in result["trace"]:
        assert row["objective"] == pytest.approx(row["observed_bce"] + row["weighted_latent_prior"] + row["weighted_lora_penalty"])
        assert row["normalized_latent_prior"] == pytest.approx(0.5 * row["latent_norm_squared_sum"] / 8)


def test_latent_mode_has_no_copy_and_initializations_are_reproducible():
    model, obs = _model_obs()
    first = diagnostic.optimize_observation(model, obs, _config(), mode="latent", seed=5, initialization=1)
    second = diagnostic.optimize_observation(model, obs, _config(), mode="latent", seed=5, initialization=1)
    center = diagnostic.optimize_observation(model, obs, _config(), mode="latent", seed=5, initialization=0)
    assert first["decoder"] is model
    np.testing.assert_array_equal(first["initial_latent"], second["initial_latent"])
    np.testing.assert_array_equal(first["optimized_latent"], second["optimized_latent"])
    assert not np.array_equal(first["initial_latent"], center["initial_latent"])
    assert first["lora_state"] == {}
    np.testing.assert_array_equal(first["optimized_probability"], first["base_same_z_probability"])


def test_hidden_reference_labels_are_rejected_before_optimization():
    model, obs = _model_obs()
    obs[0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="reference target"):
        diagnostic.optimize_observation(model, obs, _config())


def test_variance_arms_reuse_fixed_decoder_and_identical_standardized_noise():
    model, obs = _model_obs()
    adapted = diagnostic.optimize_observation(model, obs, _config(), seed=17)
    before = {key: value.clone() for key, value in adapted["decoder"].state_dict().items()}
    a = diagnostic.sample_fixed_decoder(adapted, obs, variance=0.001, candidate_count=4, seed=20)
    b = diagnostic.sample_fixed_decoder(adapted, obs, variance=0.1, candidate_count=4, seed=20)
    z = adapted["optimized_latent"]
    np.testing.assert_allclose((a["latents"] - z) / np.sqrt(.001),
                               (b["latents"] - z) / np.sqrt(.1), atol=3e-6)
    for key, value in adapted["decoder"].state_dict().items():
        torch.testing.assert_close(before[key], value, rtol=0, atol=0)
    assert not a["projection_applied"] and not a["decoder_reoptimized"]
    assert a["standard_deviation"] == pytest.approx(np.sqrt(.001))


class _ConstantDecoder(torch.nn.Module):
    grid_size = (2, 2, 2)

    def decode(self, z):
        return torch.zeros((len(z), *self.grid_size))


def test_zero_accepted_remains_missing_and_has_no_optimized_fallback():
    obs = np.zeros((2, 2, 2, 2), np.float32)
    obs[1, 0, 0, 0] = 1
    adapted = {"decoder": _ConstantDecoder(), "latent": torch.zeros(1, 8)}
    result = diagnostic.sample_fixed_decoder(adapted, obs, variance=.01, candidate_count=4)
    for summary in result["summaries"]:
        assert summary["accepted_count"] == 0
        assert summary["accepted_indices"] == []
        assert summary["mean_probability"] is None
        assert summary["sample_variance"] is None
        assert summary["pairwise_blind_dice_mean"] is None


def test_drift_is_reported_with_signed_error_change():
    observed = np.zeros((8, 8, 8), bool)
    observed[0, 0, 0] = True
    truth = np.ones((8, 8, 8), np.float32)
    rows = diagnostic.lora_distance_effect(np.full(truth.shape, .8), np.full(truth.shape, .2), truth, observed)
    assert sum(row["voxel_count"] for row in rows) == truth.size
    for row in rows:
        if row["voxel_count"]:
            assert row["brier_change"] < 0
            assert row["mean_absolute_probability_drift"] == pytest.approx(.6)


def test_confidential_path_and_public_aggregate_allowlist(tmp_path):
    with pytest.raises(ValueError, match="private_revision_runs"):
        diagnostic.assert_private_real_path(tmp_path / "revision_article" / "code")
    with pytest.raises(ValueError, match="private_revision_runs"):
        diagnostic.assert_private_real_path(diagnostic.ROOT / "private_revision_runs" / "test")
    private = tmp_path / "private_revision_runs" / "test"
    assert diagnostic.assert_private_real_path(private) == private.resolve()
    # The relocated engine protects its inner tree. The new outer launcher
    # additionally protects the entire repository, including its sibling paths.
    import importlib.util
    repository = diagnostic.ROOT.parent
    specification = importlib.util.spec_from_file_location(
        "current_repository_safety_fixture", repository / "repository_safety.py")
    safety = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(safety)
    with pytest.raises(ValueError, match="entire public repository"):
        safety.validate_arguments("revision", ["--real-cases", "--data-root", str(tmp_path / "data"),
                                   "--private-out", str(repository / "private_revision_runs" / "test")],
                                  repository)
    rows = [
        {"private_case_id": "secret-a", "method": "geovae", "blind_iou": .2,
         "coordinates": [123456, 99999], "source": "secret.ply", "mask_seed": 42},
        {"private_case_id": "secret-a", "method": "geovae", "blind_iou": .4, "mask_seed": 43},
        {"private_case_id": "secret-b", "method": "geovae", "blind_iou": .8, "mask_seed": 42},
    ]
    result = diagnostic.aggregate_real_rows(rows, 122, 2, 0)
    serialized = json.dumps(result)
    assert "secret" not in serialized and "coordinates" not in serialized and "source" not in serialized
    assert result[0]["evaluated_reference_blocks"] == 2
    assert result[0]["blind_iou_mean"] == pytest.approx(.55)
    assert not result[0]["independent_mine_count_claimed"]


def test_known_extent_domain_is_explicit_and_never_changes_source(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    path = tmp_path / "box.ply"
    trimesh.creation.box(extents=(20, 10, 5)).export(path)
    before = sha256_file(path)
    grid, metadata = diagnostic.real_case_domain(path, (16, 16, 16))
    assert grid.shape == (16, 16, 16) and 0 < grid.sum() < grid.size
    assert metadata["domain"]["target_specific_bounds_used"]
    assert metadata["role"] == "retrospective_known_extent_only"
    assert metadata["domain"]["raw_coordinate_unit"] == "unverified"
    assert sha256_file(path) == before


def test_deadline_restores_shared_model_state():
    model, obs = _model_obs()
    model.train()
    flags = [p.requires_grad for p in model.parameters()]
    with pytest.raises(TimeoutError, match="budget"):
        diagnostic.optimize_observation(model, obs, _config(), mode="latent", deadline_epoch=time.time()-1)
    assert model.training
    assert flags == [p.requires_grad for p in model.parameters()]


def test_smoke_full_diagnostic_requires_validation_and_checks_artifact_hashes(tmp_path):
    manifest = Path(prepare_protocol(str(tmp_path), str(tmp_path / "protocol"),
                                    count=100, seed=731, grid_size=(16, 16, 16), audit_raw=False))
    config = {"grid_size": [16, 16, 16], "latent_dim": 8, "base_channels": 4,
              "use_transformer": True, "use_relative_position": True,
              "num_holes": 2, "samples_per_hole": 4, "mask_seed": 42, "device": "cpu",
              "test_time_lora_rank": 2, "test_time_lr": .03, "test_time_lora_lr": .001}
    config_path = tmp_path / "config.json"
    save_json(config_path, config)
    torch.manual_seed(3)
    model = ConvVAE3D(grid_size=16, latent_dim=8, base_channels=4,
                      use_transformer=True, use_relative_position=True)
    checkpoint = tmp_path / "smoke.pt"
    torch.save({"model_kind": "geovae", "manifest_sha256": sha256_file(manifest),
                "config": config, "seed": 11, "source_sha256": source_hash(),
                "model_state": model.state_dict()}, checkpoint)
    out = tmp_path / "diagnostic"
    with pytest.raises(RuntimeError, match="validation"):
        diagnostic.uncertainty_run(manifest, config_path, checkpoint, out, split="test", smoke=True)
    diagnostic.uncertainty_run(manifest, config_path, checkpoint, out, split="val", smoke=True)
    diagnostic.uncertainty_run(manifest, config_path, checkpoint, out, split="test", smoke=True)
    assert json.loads((out / "test" / "status.json").read_text())["summary_rows"] == 27
    # Completed re-evaluation must be deterministic and skip the optimizer.
    before = (out / "test" / "per_case_acceptance.csv").read_bytes()
    diagnostic.uncertainty_run(manifest, config_path, checkpoint, out, split="test", smoke=True)
    assert (out / "test" / "per_case_acceptance.csv").read_bytes() == before
    artifact = next((out / "val" / "cases").rglob("optimization_trace.csv"))
    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="corrupt"):
        diagnostic.uncertainty_run(manifest, config_path, checkpoint, out, split="test", smoke=True)


def test_real_pipeline_exports_only_aggregates_and_keeps_every_block(tmp_path, monkeypatch):
    manifest = Path(prepare_protocol(str(tmp_path), str(tmp_path / "protocol"),
                                    count=100, seed=731, grid_size=(16, 16, 16), audit_raw=False))
    config_path = tmp_path / "config.json"
    save_json(config_path, {"grid_size": [16, 16, 16], "device": "cpu", "num_holes": 2,
                           "samples_per_hole": 4, "threshold": .5})
    checkpoint = tmp_path / "test_checkpoint.pt"
    checkpoint.write_bytes(b"test-only")
    data = tmp_path / "data" / "mining_ply"
    data.mkdir(parents=True)
    for index in range(122):
        (data / f"secret_{index:03d}.ply").write_text(f"private source {index}")
    root = tmp_path / "gpt_thinking" / "Project_VAE" / "3dvae"
    root.mkdir(parents=True)
    monkeypatch.setattr(diagnostic, "ROOT", root)
    monkeypatch.setattr(diagnostic, "_source_identity", lambda: {"test": "test"})
    monkeypatch.setattr(diagnostic, "_load_checked_model", lambda *args: (object(), {}))
    target = np.zeros((16, 16, 16), np.float32)
    target[4:12, 4:12, 4:12] = 1
    monkeypatch.setattr(diagnostic, "real_case_domain", lambda *args: (target, {
        "raw_coordinates": [123456, 654321], "raw_source": "private-original"}))
    monkeypatch.setattr(diagnostic, "_predict", lambda *args: (np.full(target.shape, .6, np.float32), {}))
    private_out = tmp_path / "gpt_thinking" / "private_revision_runs" / "real_test"
    public_out = tmp_path / "public"
    result = diagnostic.real_cases_run(manifest, config_path, {"geovae": checkpoint},
                                        private_out, public_out, data_root=data.parent,
                                        mask_seeds=(42,), methods=("geovae",))
    summary = json.loads(result.read_text())
    assert summary["status"] == "complete"
    assert summary["metrics"][0]["evaluated_reference_blocks"] == 122
    assert len(list((private_out / "cases").glob("*/target.npy"))) == 122
    assert {p.name for p in public_out.iterdir()} == {"real_summary.json", "real_aggregate_metrics.csv"}
    for path in public_out.iterdir():
        text = path.read_text()
        assert "secret_" not in text and "private-original" not in text and "123456" not in text
    # An incomplete supplied source cohort cannot silently shrink to 121.
    (data / "secret_121.ply").unlink()
    with pytest.raises(RuntimeError, match="all 122"):
        diagnostic.real_cases_run(manifest, config_path, {"geovae": checkpoint},
                                  private_out, public_out, data_root=data, methods=("geovae",))

