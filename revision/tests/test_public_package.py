"""Public distribution checks without confidential source files."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import pytest
import main
from repro_pipeline import ROOT, source_hash, sha256_file
from repro_run_all import OfflineRun

def test_engine_identity_and_exact_synthetic_archive():
    meta = json.loads((ROOT / "EXPORT_PROVENANCE.json").read_text())
    assert source_hash() == meta["engine_source_sha256"]
    for name, digest in meta["checkpoint_engine_files"].items():
        assert sha256_file(ROOT / name) == digest
    manifest = ROOT / "protocol" / "manifest.json"
    assert sha256_file(manifest) == meta["synthetic_manifest_sha256"]
    data = json.loads(manifest.read_text())
    assert data["count"] == 5000
    assert len(list((ROOT / "protocol" / "grids").glob("*.npy"))) == 5000
    assert not (ROOT / "protocol" / "raw_inventory.jsonl").exists()
    assert not (ROOT / "protocol" / "raw_audit_summary.json").exists()

def test_prepare_does_not_claim_restricted_data_were_audited(tmp_path):
    job = OfflineRun(ROOT / "protocol" / "manifest.json", ROOT / "config" / "revision_v1.yaml",
                     tmp_path / "run", dry_run=True)
    assert job.identity["public_mode"] == "synthetic_only"
    raw = job._verify_raw_inventory()
    assert raw["status"] == "not_available_restricted"
    assert raw["audited"] is False and raw["files"] is None

def test_smoke_routes_to_small_flow_without_fifty_parent_tuning(tmp_path):
    with patch("revision_smoke.run", return_value=tmp_path / "result.json") as runner:
        assert main.main(["--all", "--smoke", "--out", str(tmp_path)]) == 0
        runner.assert_called_once_with(str(tmp_path), prepare_only=False)

def test_public_final_policy_requires_all_synthetic_phases_but_not_restricted_real(tmp_path, monkeypatch):
    import torch
    from public_finish import verify_numerical_policy
    from revision_integrity import AuditError
    from revision_numerics import (DECLARED_POLICY, actual_flags, configure_evaluation,
                                  helper_sha256, record_training_policy)
    old = actual_flags()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(False)
        record_training_policy(tmp_path, "train")
        for phase in ("tune", "validation", "test", "ablation_validation", "ablation_test",
                      "uncertainty_val", "uncertainty_test", "robustness"):
            configure_evaluation(tmp_path, phase)
        identity = {"evaluation_numerical_policy": DECLARED_POLICY,
                    "evaluation_numerics_source_sha256": helper_sha256()}
        with pytest.raises(AuditError, match="phase_proofs_missing"):
            verify_numerical_policy(tmp_path, identity)
        configure_evaluation(tmp_path, "public_finalize")
        result = verify_numerical_policy(tmp_path, identity)
        assert result["evaluation_phase_count"] == 9
        assert result["training_deterministic_algorithms"] is False
        assert not list((tmp_path / "numerical_policy").glob("real_cases*"))
    finally:
        torch.use_deterministic_algorithms(old["deterministic_algorithms"],
                                           warn_only=old["deterministic_warn_only"])
        torch.backends.cudnn.benchmark = old["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = old["cudnn_deterministic"]

def test_diagnostic_evaluation_cli_configures_strict_policy_before_scoring(tmp_path):
    import repro_cli
    with patch("revision_numerics.configure_evaluation") as policy, patch("repro_cli.evaluate") as evaluate:
        assert repro_cli.main(["evaluate", "--out", str(tmp_path), "--methods", "idw", "--seed", "11"]) == 0
        policy.assert_called_once_with(str(tmp_path), phase="diagnostic_evaluate")
        evaluate.assert_called_once()

def test_diagnostic_training_cli_does_not_enable_evaluation_policy(tmp_path):
    import repro_cli
    with patch("revision_numerics.configure_evaluation") as policy, patch("repro_cli.train") as train:
        assert repro_cli.main(["train", "--run-dir", str(tmp_path), "--model", "geovae", "--seed", "11"]) == 0
        policy.assert_not_called()
        train.assert_called_once()

def test_real_requires_explicit_external_paths():
    with pytest.raises(SystemExit) as error:
        main.main(["--real-cases"])
    assert error.value.code == 2
    from revision_diagnostics import assert_private_real_path
    with pytest.raises(ValueError):
        assert_private_real_path(ROOT / "runs" / "private")
    with pytest.raises(ValueError):
        assert_private_real_path("relative_private")

def test_public_all_dry_run_skips_restricted_stage(tmp_path, capsys):
    target = tmp_path / "not_created"
    assert main.main(["--all", "--dry-run", "--out", str(target)]) == 0
    assert "restricted real cases skipped" in capsys.readouterr().out
    assert not target.exists()

def test_final_artifact_check_recomputes_robustness_and_rejects_changed_csv(tmp_path):
    import csv
    from experiment_data import prepare_protocol
    from repro_pipeline import save_json, stable_hash
    from revision_robustness import run
    from public_finish import verify_robustness_artifacts
    manifest = prepare_protocol(str(tmp_path), str(tmp_path / "protocol"), 20, 42,
                                (16, 16, 16), audit_raw=False)
    root = tmp_path / "run"
    config = {"grid_size": [16] * 3, "num_holes": 8, "samples_per_hole": 4,
              "mask_seed": 42, "latent_dim": 8, "device": "cpu", "threshold": .5}
    config_path = root / "config.json"
    save_json(config_path, config)
    save_json(root / "thresholds.json", {"status": "draft", "manifest_sha256": sha256_file(manifest),
              "config_sha256": stable_hash(config), "evaluation_source_sha256": source_hash(),
              "thresholds": {"idw": .5, "rbf": .5}})
    run(manifest, root, config_path, smoke=True, methods=("idw", "rbf"), include_noise=False)
    assert verify_robustness_artifacts(root, manifest, config_path) == 14
    path = root / "revision_robustness" / "per_case.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        keys, rows = reader.fieldnames, list(reader)
    rows[0]["input_error_0p5"] = str(float(rows[0]["input_error_0p5"]) + .01)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    with pytest.raises(RuntimeError, match="CSV differs"):
        verify_robustness_artifacts(root, manifest, config_path)

def test_budget_timeout_only_targets_the_spawned_child_group(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from io import StringIO
    import signal
    import subprocess
    import revision_runner as controller
    config = tmp_path / "config.json"
    config.write_text("{}")
    job = controller.RevisionRun(SimpleNamespace(out=str(tmp_path / "run"),
                                config=str(config), budget_hours=1))
    calls = []
    class Child:
        pid = 424242
        stdout = StringIO("")
        count = 0
        def wait(self, timeout):
            self.count += 1
            if self.count == 1:
                raise subprocess.TimeoutExpired("own child", timeout)
            return 0
    def spawn(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        return Child()
    monkeypatch.setattr(controller, "os", SimpleNamespace(name="posix", killpg=lambda pid, sig: calls.append((pid, sig))))
    monkeypatch.setattr(controller, "signal", SimpleNamespace(SIGTERM=signal.SIGTERM, SIGKILL=9))
    monkeypatch.setattr(controller.subprocess, "Popen", spawn)
    with pytest.raises(TimeoutError):
        job.command("timeout_test", ["-c", "pass"])
    assert calls == [(424242, signal.SIGTERM), (424242, 9)]

def test_classical_deadline_is_scoped_to_the_worker_group(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import time
    import torch
    import revision_runner as controller
    root = tmp_path / "run"
    root.mkdir()
    (root / "revision_budget.json").write_text(json.dumps({"started_unix": time.time(), "hours": 1}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": [{"split": "val", "sample_id": "one"}]}))
    state = {"group": 100, "handler": None, "killed": [], "alarm": None}
    def new_session():
        state["group"] = 200
    fake_os = SimpleNamespace(name="posix", getpid=lambda: 200,
          getpgrp=lambda: state["group"], setsid=new_session, environ={},
          killpg=lambda pid, sig: state["killed"].append((pid, sig)))
    fake_signal = SimpleNamespace(SIGALRM=14, SIGTERM=15,
          signal=lambda kind, handler: state.update(handler=handler),
          alarm=lambda seconds: state.update(alarm=seconds))
    monkeypatch.setattr(controller, "os", fake_os)
    monkeypatch.setattr(controller, "signal", fake_signal)
    monkeypatch.setattr(controller, "evaluate", lambda *args, **kwargs: None)
    threads = torch.get_num_threads()
    try:
        controller._classical_shard((str(manifest), "not_read", str(root), "val", None, 0, 1))
    finally:
        torch.set_num_threads(threads)
    assert state["group"] == 200 and state["alarm"] > 0
    state["handler"](14, None)
    assert state["killed"] == [(200, 15)]
