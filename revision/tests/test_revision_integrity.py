"""Small CPU audit fixtures; no private source cohort or formal archive is scanned."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from experiment_data import make_drill_masks, prepare_protocol
from model import ConvVAE3D
import revision_diagnostics as diagnostics
import revision_integrity as integrity
from repro_metrics import volume_metrics
from repro_pipeline import save_json, sha256_file, source_hash, stable_hash


torch.set_num_threads(1)


def _write_csv(path, rows):
    diagnostics._csv(Path(path), rows)


def _rehash(folder):
    record = json.loads((folder / "complete.json").read_text())
    record["artifacts"] = {p.name: sha256_file(p) for p in folder.iterdir()
                           if p.is_file() and p.name != "complete.json"}
    save_json(folder / "complete.json", record)


def _controlled_sampler(original):
    """Give the auditor S=0/S=1/S>=2 arms in an explicitly draft fixture."""
    def sample(adapted, observation, *, variance, candidate_count, seed):
        result = original(adapted, observation, variance=variance,
                          candidate_count=candidate_count, seed=seed)
        observed, labels = observation[1].astype(bool), observation[0]
        probabilities = result["probabilities"]
        for index in range(candidate_count):
            if variance == .001:
                error = 0. if index == 0 else .3
            elif variance == .01:
                error = (0., .005, .02, .08)[index]
            else:
                error = .5
            probabilities[index][observed] = np.where(labels[observed] >= .5, 1-error, error)
            row = result["candidates"][index]
            row.update(diagnostics.observation_consistency(probabilities[index], observation))
            row["accepted_by_threshold"] = {diagnostics.threshold_key(tau): bool(
                row["observed_mse"] <= tau and row["observed_violation_count"] == 0)
                for tau in diagnostics.ACCEPTANCE_THRESHOLDS}
        for summary in result["summaries"]:
            indices = [i for i, row in enumerate(result["candidates"])
                       if row["accepted_by_threshold"][diagnostics.threshold_key(summary["threshold"])]]
            selected, count = probabilities[indices], len(indices)
            summary.update({"accepted_indices": indices, "accepted_count": count,
                            "acceptance_rate": count/candidate_count, "variance_defined": count >= 2,
                            "mean_probability": selected.mean(axis=0, dtype=np.float64).astype(np.float32) if count else None,
                            "sample_variance": selected.var(axis=0, ddof=1, dtype=np.float64).astype(np.float32) if count >= 2 else None,
                            "status": "zero_accepted" if not count else "one_accepted_variance_undefined" if count == 1 else "local_variability_only",
                            **diagnostics._diversity(selected, ~observed)})
        return result
    return sample


@pytest.fixture
def uncertainty_archive(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    manifest = Path(prepare_protocol(str(root), str(root / "protocol"), count=100,
                                    seed=731, grid_size=(16, 16, 16), audit_raw=False))
    config = {"grid_size": [16, 16, 16], "latent_dim": 8, "base_channels": 4,
              "use_transformer": True, "use_relative_position": True,
              "num_holes": 2, "samples_per_hole": 4, "mask_seed": 42,
              "device": "cpu", "test_time_lora_rank": 2,
              "test_time_lr": .03, "test_time_lora_lr": .001}
    config_path = root / "tuning" / "selected_config.json"
    save_json(config_path, config)
    torch.manual_seed(3)
    model = ConvVAE3D(grid_size=16, latent_dim=8, base_channels=4,
                      use_transformer=True, use_relative_position=True)
    checkpoint = root / "models" / "geovae" / "seed11" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"model_kind": "geovae", "manifest_sha256": sha256_file(manifest),
                "config": config, "seed": 11, "source_sha256": source_hash(),
                "model_state": model.state_dict()}, checkpoint)
    save_json(root / "run_identity.json", {"source_sha256": source_hash(),
                                           "manifest_sha256": sha256_file(manifest)})
    monkeypatch.setattr(diagnostics, "sample_fixed_decoder", _controlled_sampler(diagnostics.sample_fixed_decoder))
    for split in ("val", "test"):
        diagnostics.uncertainty_run(manifest, config_path, checkpoint, root / "uncertainty", split=split, smoke=True)
    return root, manifest, json.loads(manifest.read_text()), config


def test_uncertainty_audit_recomputes_all_three_acceptance_regimes(uncertainty_archive):
    root, manifest, protocol, config = uncertainty_archive
    result = integrity.audit_uncertainty(root, manifest, protocol, config, formal=False)
    assert result["cases"] == 2 and result["adaptations"] == 6
    assert result["candidates"] == 72 and result["summary_rows"] == 54
    assert result["zero_accepted_rows"] == 18 and result["single_accepted_rows"] == 18
    assert result["max_map_absolute_difference"] == 0


def test_hash_refreshed_incorrect_mean_is_detected(uncertainty_archive):
    root, manifest, protocol, config = uncertainty_archive
    path = next((root / "uncertainty" / "test" / "cases").rglob("mean_var0.01_tau0.001.npy"))
    probability = np.load(path)
    probability[0, 0, 0] += .02
    np.save(path, probability)
    _rehash(path.parent)
    with pytest.raises(integrity.AuditError, match="accepted_mean"):
        integrity.audit_uncertainty(root, manifest, protocol, config, formal=False)


def test_hash_refreshed_wrong_latent_regularization_is_detected(uncertainty_archive):
    root, manifest, protocol, config = uncertainty_archive
    path = next((root / "uncertainty" / "test" / "cases").rglob("candidates_var0.01.csv"))
    rows = integrity._csv(path)
    rows[0]["manuscript_prior_lower_bound"] = 987.
    _write_csv(path, rows)
    _rehash(path.parent)
    with pytest.raises(integrity.AuditError, match="lower_bound"):
        integrity.audit_uncertainty(root, manifest, protocol, config, formal=False)


def test_single_sample_fake_zero_variance_map_is_detected(uncertainty_archive):
    root, manifest, protocol, config = uncertainty_archive
    folder = next((root / "uncertainty" / "test" / "cases").glob("*/joint/init0"))
    np.save(folder / "variance_var0.001_tau0.001.npy", np.zeros((16, 16, 16), np.float32))
    _rehash(folder)
    with pytest.raises(integrity.AuditError, match="map_policy"):
        integrity.audit_uncertainty(root, manifest, protocol, config, formal=False)


def test_changed_validation_freeze_prevents_test_audit(uncertainty_archive):
    root, manifest, protocol, config = uncertainty_archive
    frozen = root / "uncertainty" / "val" / "settings_frozen.json"
    record = json.loads(frozen.read_text())
    record["case_ids"] = []
    save_json(frozen, record)
    with pytest.raises(integrity.AuditError, match="validation_freeze"):
        integrity.audit_uncertainty(root, manifest, protocol, config, formal=False)


def test_hash_refreshed_candidate_acceptance_forgery_is_detected(uncertainty_archive):
    root, manifest, protocol, config = uncertainty_archive
    path = next((root / "uncertainty" / "test" / "cases").rglob("candidates_var0.1.csv"))
    rows = integrity._csv(path)
    flags = json.loads(rows[0]["accepted_by_threshold"])
    flags["0.001"] = True
    rows[0]["accepted_by_threshold"] = flags
    _write_csv(path, rows)
    _rehash(path.parent)
    with pytest.raises(integrity.AuditError, match="candidate_acceptance"):
        integrity.audit_uncertainty(root, manifest, protocol, config, formal=False)


def test_raw_audit_rehashes_same_size_changes_and_checks_new_files(tmp_path):
    protocol = tmp_path / "protocol"
    data = tmp_path / "raw"
    protocol.mkdir(); (data / "one_role").mkdir(parents=True)
    source = data / "one_role" / "case.txt"
    source.write_bytes(b"first")
    inventory = protocol / "raw_inventory.jsonl"
    inventory.write_text(json.dumps({"path": "one_role/case.txt", "bytes": 5,
                                     "sha256": sha256_file(source), "source_role": "one_role"}) + "\n")
    summary = protocol / "raw_audit_summary.json"
    save_json(summary, {"counts": {"one_role": 1}})
    identity = {"manifest": str(protocol / "manifest.json"), "data_root": str(data),
                "raw_inventory_sha256": sha256_file(inventory), "raw_audit_sha256": sha256_file(summary)}
    assert integrity.audit_raw(tmp_path, identity, formal=False)["files"] == 1
    source.write_bytes(b"other")
    with pytest.raises(integrity.AuditError, match="raw_source_changed"):
        integrity.audit_raw(tmp_path, identity, formal=False)
    source.write_bytes(b"first")
    (data / "one_role" / "extra.txt").write_text("extra")
    with pytest.raises(integrity.AuditError, match="exhaustive"):
        integrity.audit_raw(tmp_path, identity, formal=False)


def test_frozen_grid_hash_and_source_are_verified_without_mutation(tmp_path):
    manifest = Path(prepare_protocol(str(tmp_path), str(tmp_path / "protocol"), count=100,
                                    grid_size=(16, 16, 16), audit_raw=False))
    frozen = tmp_path / "frozen_inputs"
    shutil.copytree(manifest.parent, frozen)
    (frozen / "engine.py").write_text("# frozen test engine\n")
    save_json(tmp_path / "run_identity.json", {"manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
                                              "code_files": {"engine.py": sha256_file(frozen / "engine.py")}})
    result, _, protocol = integrity.audit_frozen(tmp_path, formal=False)
    assert result["verified_grids"] == 100
    grid = frozen / protocol["samples"][0]["path"]
    original = grid.read_bytes()
    array = np.load(grid); array.flat[0] = 1-array.flat[0]; np.save(grid, array)
    assert len(grid.read_bytes()) == len(original)
    with pytest.raises((integrity.AuditError, ValueError, RuntimeError)):
        integrity.audit_frozen(tmp_path, formal=False)


@pytest.fixture
def real_archive(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    data = tmp_path / "raw_data" / "mining_ply"
    data.mkdir(parents=True)
    private = tmp_path / "gpt_thinking" / "private_revision_runs" / "test"
    fake_code_root = tmp_path / "gpt_thinking" / "Project_VAE" / "3dvae"
    fake_code_root.mkdir(parents=True)
    monkeypatch.setattr(diagnostics, "ROOT", fake_code_root)
    config = {"grid_size": [8, 8, 8], "num_holes": 2, "samples_per_hole": 4}
    checkpoint = root / "models" / "geovae" / "seed11" / "best.pt"
    checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b"fixture checkpoint")
    threshold_path = root / "thresholds.json"
    save_json(threshold_path, {"thresholds": {"geovae": .4}})
    sources = {}
    for index in range(2):
        path = data / f"secret-source-{index}.ply"
        path.write_text(f"confidential fixture {index}")
        sources[path.name] = sha256_file(path)
    source_files = {name: sha256_file(integrity.ROOT / name) for name in integrity.DIAGNOSTIC_SOURCE_FILES}
    identity = {"manifest_sha256": "synthetic-fixture-sha", "config": config,
                "threshold_file_sha256": sha256_file(threshold_path), "real_sources": sources,
                "seed": 11, "mask_seeds": [42], "methods": ["geovae"],
                "source_files": source_files, "checkpoint_sha256": {"geovae": sha256_file(checkpoint)}}
    save_json(private / "identity.json", identity)
    save_json(root / "run_identity.json", {"manifest_sha256": "synthetic-fixture-sha", "data_root": str(data.parent)})
    rows = []
    for name, digest in sources.items():
        case_id = "private-" + stable_hash([name, digest])[:16]
        folder = private / "cases" / case_id
        folder.mkdir(parents=True)
        target = np.zeros((8, 8, 8), np.float32); target[2:6, 2:6, 2:6] = 1
        observed, hidden = make_drill_masks((8, 8, 8), case_id, 2, 4, 42, 4)
        prediction = np.full(target.shape, .35, np.float32)
        row = {"private_case_id": case_id, "method": "geovae", "mask_seed": 42,
               "seconds": .2, **volume_metrics(prediction, target, observed, threshold=.4, hidden_mask=hidden)}
        np.save(folder / "target.npy", target)
        np.save(folder / "observation_mask42.npy", np.stack((target*observed, observed)))
        np.save(folder / "prediction_geovae_mask42.npy", prediction)
        save_json(folder / "private_metadata.json", {"source": name, "source_sha256": digest})
        save_json(folder / "complete.json", {"status": "complete", "identity_sha256": stable_hash(identity),
                                            "metric_rows": [row], "artifacts": {p.name: sha256_file(p) for p in folder.iterdir()}})
        rows.append(row)
    save_json(private / "exclusions.json", [])
    save_json(private / "failures.json", [])
    aggregates = diagnostics.aggregate_real_rows(rows, 2, 0, 0)
    save_json(root / "real_aggregate" / "real_summary.json", {
        "status": "complete", "private_audit_identity_sha256": stable_hash(identity),
        "real_shapes_or_coordinates_exported": False, "metrics": aggregates})
    _write_csv(root / "real_aggregate" / "real_aggregate_metrics.csv", aggregates)
    return root, private, config


def test_private_real_recomputation_matches_public_aggregates(real_archive):
    root, private, config = real_archive
    result = integrity.audit_real(root, private, config, formal=False)
    assert result["reference_blocks"] == result["evaluated_blocks"] == 2
    assert result["recomputed_case_method_rows"] == 2
    assert "secret-source" not in json.dumps(result)


def test_real_derived_metric_change_is_detected_even_after_refreshing_hashes(real_archive):
    root, private, config = real_archive
    folder = next((private / "cases").iterdir())
    record = json.loads((folder / "complete.json").read_text())
    record["metric_rows"][0]["blind_iou"] = .999
    save_json(folder / "complete.json", record)
    with pytest.raises(integrity.AuditError, match="derived_scalar"):
        integrity.audit_real(root, private, config, formal=False)


def test_zero_eligible_real_blocks_cannot_pass(real_archive):
    root, private, config = real_archive
    identity = json.loads((private / "identity.json").read_text())
    exclusions = [{"private_case_id": "private-" + stable_hash([name, sha])[:16]}
                  for name, sha in identity["real_sources"].items()]
    save_json(private / "exclusions.json", exclusions)
    with pytest.raises(integrity.AuditError, match="no_valid_real"):
        integrity.audit_real(root, private, config, formal=False)


def test_private_shape_field_cannot_be_added_to_public_summary(real_archive):
    root, private, config = real_archive
    path = root / "real_aggregate" / "real_summary.json"
    record = json.loads(path.read_text())
    record["raw_coordinates"] = [123456., 654321.]
    save_json(path, record)
    with pytest.raises(integrity.AuditError, match="unexpected_public_summary"):
        integrity.audit_real(root, private, config, formal=False)


def test_failure_report_never_contains_exception_paths_or_coordinates(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise FileNotFoundError("secret-source-122.ply coordinates=123456,654321")
    monkeypatch.setattr(integrity, "audit_frozen", fail)
    with pytest.raises(integrity.AuditError, match="frozen_protocol"):
        integrity.audit(tmp_path, formal=False, verify_raw=False, require_real=False, require_ablations=False)
    report = (tmp_path / "final_integrity" / "report.json").read_text()
    assert "secret-source" not in report and "123456" not in report and "654321" not in report
    assert json.loads(report)["status"] == "failed"


def test_primary_audit_rederives_saved_publication_tables(tmp_path):
    from repro_pipeline import evaluate
    from repro_tuning import select_thresholds
    from revision_analysis import analyse_evaluations
    manifest = Path(prepare_protocol(str(tmp_path), str(tmp_path / "frozen_inputs"), count=20,
                                    seed=42, grid_size=(16, 16, 16), audit_raw=False))
    protocol = json.loads(manifest.read_text())
    config = {"grid_size": [16, 16, 16], "mask_seed": 42, "num_holes": 2,
              "samples_per_hole": 4, "device": "cpu", "threshold": .5, "latent_dim": 16}
    config_path = tmp_path / "tuning" / "selected_config.json"
    save_json(config_path, config)
    save_json(tmp_path / "run_identity.json", {"manifest_sha256": sha256_file(manifest),
                                              "source_sha256": source_hash(), "methods": ["idw", "rbf"], "seeds": [11]})
    validation = tmp_path / "validation" / "seed11"
    testing = tmp_path / "test" / "seed11"
    evaluate(str(manifest), str(config_path), str(validation), "val", ["idw", "rbf"], {}, seed=11)
    threshold_file = tmp_path / "thresholds.json"
    select_thresholds([str(validation)], str(config_path), str(threshold_file), draft=True, allow_subset=True)
    evaluate(str(manifest), str(config_path), str(testing), "test", ["idw", "rbf"], {}, seed=11,
             threshold_file=str(threshold_file))
    analyse_evaluations(manifest, [testing], tmp_path / "revision_analysis", formal=False,
                        required_methods=("idw", "rbf"), replicates=64, config_path=config_path)
    result = integrity.audit_main(tmp_path, protocol, config, formal=False)
    # Classical-only draft fixtures have no model-seed or GeoVAE contrast table.
    assert result["numerically_recomputed_tables"] == 4
    assert result["verified_case_method_rows"] == 12
    summary = tmp_path / "revision_analysis" / "summary.csv"
    rows = integrity._csv(summary)
    rows[0]["blind_iou_mean"] = .999
    _write_csv(summary, rows)
    with pytest.raises(integrity.AuditError, match="publication_table_differs"):
        integrity.audit_main(tmp_path, protocol, config, formal=False)

