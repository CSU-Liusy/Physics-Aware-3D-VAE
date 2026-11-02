"""New frontend/backend dispatch checks, separate from numerical parity.

Fake backend calls below test only argument routing and fail-closed behavior.
They do not represent a successful formal public run or saved-prediction audit.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import porphyry_public as front
import release_contract as release

# Cold-release scenarios are fixtures, distinct from the shipped reviewed
# registry. Never remove or modify the real registry to exercise this state.
COLD_RELEASE = {"status": "not_released", "prediction_entry_enabled": False}


class FrontendCouplingTests(unittest.TestCase):
    def saved_fixture(self, root):
        out = Path(root) / "existing-results"; out.mkdir()
        files = {"identity.json": {"format": "porphyry-public-evaluation-v1", "formal": True,
                                   "origin": "public_reproduction"},
                 "public_supplement_budget.json": {"origin": "public_reproduction", "started_unix": 1, "hours": 1},
                 "status.json": {}, "failures.json": [], "failure_history.json": []}
        for name, value in files.items():
            (out / name).write_text(json.dumps(value))
        (out / "per_case.csv").write_text("fixture-only\n")
        return out

    def test_inspect_checks_parent_and_prepared_even_when_parent_missing(self):
        parent, prepared = Mock(side_effect=ValueError("missing parent")), Mock(return_value={"arrays": 41})
        with patch.object(front, "public_primary", parent), patch.object(front, "verify_prepared", prepared), \
             patch.object(release, "release_evidence", return_value=COLD_RELEASE):
            result = front.inspect("primary", "code", "cache")
        prepared.assert_called_once_with("cache")
        parent.assert_called_once_with("primary", "code", require_time=False)
        self.assertEqual(result["evidence"]["prepared"]["arrays"], 41)
        self.assertFalse(result["prediction_entry_enabled"])

    def test_inspect_optional_context_reports_only_small_evidence_without_clock(self):
        ctx = {"data": {"conditions": [np.zeros(2)] * 15, "manifest_sha256": "external-sha"},
               "checkpoint_sha256": {"geovae": "weights-hash"}, "training_image_id": "train-only-ti",
               "training_image_file_sha256": "ti-hash", "primary_prerequisite_sha256": {"a": "b"}}
        backend = SimpleNamespace(context=Mock(return_value=ctx), run=Mock(side_effect=AssertionError("prediction called")),
                                  supplemental_budget=Mock(side_effect=AssertionError("clock called")))
        with patch.object(front, "public_primary", return_value={"verified_public_lifecycle": True}), \
             patch.object(front, "verify_prepared", return_value={"arrays": 41}), \
             patch.dict(sys.modules, {"portable_evaluation": backend}):
            result = front.inspect("primary", "code", "cache", "out")
        backend.context.assert_called_once_with("primary", "code", "cache", "out", require_time=False)
        self.assertEqual(result["evidence"]["evaluation_context"]["conditions"], 15)
        self.assertFalse(result["evidence"]["evaluation_context"]["clock_initialized"])
        json.dumps(result)  # No target/probability arrays returned to the UI.
        backend.run.assert_not_called(); backend.supplemental_budget.assert_not_called()

    def test_context_failure_retained_as_blocker(self):
        backend = SimpleNamespace(context=Mock(side_effect=ValueError("mismatched checkpoint")))
        with patch.object(front, "public_primary", return_value={}), patch.object(front, "verify_prepared", return_value={}), \
             patch.dict(sys.modules, {"portable_evaluation": backend}):
            result = front.inspect("primary", "code", "cache", "out")
        self.assertIn("mismatched checkpoint", result["prerequisite_blockers"][0])
        self.assertEqual(result["model_calls"], 0)

    def test_run_does_not_dispatch_even_when_backend_would_enable(self):
        backend = SimpleNamespace(PREDICTION_ENTRY_ENABLED=True, run=Mock(side_effect=AssertionError("prediction called")))
        with patch.dict(sys.modules, {"portable_evaluation": backend}), \
             patch.object(front, "inspect", side_effect=AssertionError("run inspected paths")), \
             patch.object(front, "supplemental_budget", side_effect=AssertionError("clock initialized")), \
             patch.object(release, "release_evidence", return_value=COLD_RELEASE):
            code = front.main(["run", "--primary-run", "missing", "--code-root", "missing", "--data-cache", "missing", "--out", "missing"])
        self.assertEqual(code, 2)
        backend.run.assert_not_called()

    def test_run_cli_ignores_environment_enabling_attempt_without_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); out = root / "out"
            env = dict(os.environ, GEOVAE_ENABLE_PREDICTION="1", PREDICTION_ENTRY_ENABLED="true")
            # Patch only the test child's release query. The production files,
            # source registry and attestations stay untouched.
            script = ("import runpy,sys;sys.path.insert(0,sys.argv.pop(1));"
                      "import release_contract;release_contract.release_evidence=lambda:"
                      "{'status':'not_released','prediction_entry_enabled':False};"
                      "sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')")
            process = subprocess.run([sys.executable, "-B", "-c", script, str(Path(front.__file__).resolve().parent), str(Path(front.__file__).resolve()), "run",
                                      "--primary-run", str(root / "no-parent"), "--code-root", str(root / "no-code"),
                                      "--data-cache", str(root / "no-cache"), "--out", str(out)],
                                     cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(process.returncode, 2, process.stderr)
            result = json.loads(process.stdout)
            self.assertFalse(result["clock_initialized"])
            self.assertFalse(result["prediction_entry_enabled"])
            self.assertFalse(out.exists())

    def test_shipped_release_retains_actual_subset_and_saved_reference_scope(self):
        result = release.release_evidence()
        self.assertEqual(result["status"], "released_with_bounded_validation")
        self.assertTrue(result["prediction_entry_enabled"])
        self.assertEqual(result["actual_public_subset_rows"], 10)
        self.assertEqual(result["author_reference_saved_only_rows"], 150)
        self.assertFalse(result["independent_public_full150_executed"])
        self.assertFalse(result["public_from_scratch_training_executed"])

    def test_released_cli_environment_flags_cannot_bypass_missing_prerequisites(self):
        self.assertTrue(release.release_evidence()["prediction_entry_enabled"])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); out = root / "out"
            command = [sys.executable, "-B", str(Path(front.__file__).resolve()), "run",
                       "--primary-run", str(root / "no-parent"), "--code-root", str(root / "no-code"),
                       "--data-cache", str(root / "no-cache"), "--out", str(out)]
            records = []
            for overrides in ({}, {"GEOVAE_ENABLE_PREDICTION": "1", "PREDICTION_ENTRY_ENABLED": "true"}):
                env = {key: value for key, value in os.environ.items()
                       if key not in {"GEOVAE_ENABLE_PREDICTION", "PREDICTION_ENTRY_ENABLED"}}
                env.update(overrides)
                process = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
                self.assertEqual(process.returncode, 2, process.stderr)
                record = json.loads(process.stdout)
                self.assertEqual(record["status"], "blocked_no_inference")
                self.assertEqual(record["model_calls"], 0)
                self.assertTrue(record["error"])
                self.assertFalse(out.exists())
                records.append(record)
            self.assertEqual(records[0], records[1])

    def test_verify_missing_saved_results_cannot_start_clock_or_backend(self):
        backend = SimpleNamespace(audit_saved=Mock(side_effect=AssertionError("backend called")))
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {"portable_evaluation": backend}), \
             patch.object(front, "supplemental_budget", side_effect=AssertionError("clock initialized")):
            out = Path(folder) / "missing"
            with self.assertRaisesRegex(ValueError, "Existing saved public evaluation"):
                front.verify_only("parent", "code", "cache", out)
            self.assertFalse(out.exists())
        backend.audit_saved.assert_not_called()

    def test_verify_rejects_internal_and_nonformal_saved_identity(self):
        for wrong in ({"format": "porphyry-01-frozen-evaluation-v1", "formal": True},
                      {"format": "porphyry-public-evaluation-v1", "formal": False, "origin": "public_reproduction"}):
            with self.subTest(identity=wrong), tempfile.TemporaryDirectory() as folder:
                out = self.saved_fixture(folder)
                (out / "identity.json").write_text(json.dumps(wrong))
                with self.assertRaisesRegex(ValueError, "not internal or fake"):
                    front.verify_only("parent", "code", "cache", out)

    def test_verify_routes_saved_only_backend_and_no_budget_after_expiry(self):
        with tempfile.TemporaryDirectory() as folder:
            out = self.saved_fixture(folder)
            audit_path = out / "fixture_saved_audit.json"
            audit_path.write_text(json.dumps({"status": "passed_saved_probability_recomputation", "formal": True,
                                              "rows": 150, "no_prediction_or_weight_loading": True}))
            old_clock = (out / "public_supplement_budget.json").read_bytes()
            backend = SimpleNamespace(audit_saved=Mock(return_value=audit_path),
                                      run=Mock(side_effect=AssertionError("prediction called")),
                                      _engine=Mock(side_effect=AssertionError("model runtime loaded")))
            with patch.dict(sys.modules, {"portable_evaluation": backend}), \
                 patch.object(front, "supplemental_budget", side_effect=AssertionError("clock initialized")):
                result = front.verify_only("parent", "code", "cache", out)
            backend.audit_saved.assert_called_once_with("parent", "code", "cache", out.resolve())
            backend.run.assert_not_called(); backend._engine.assert_not_called()
            self.assertEqual((out / "public_supplement_budget.json").read_bytes(), old_clock)
            self.assertEqual(result["status"], "verified_saved_public_results")
            self.assertFalse(result["clock_initialized"])

    def test_verify_incomplete_backend_audit_is_not_success(self):
        with tempfile.TemporaryDirectory() as folder:
            out = self.saved_fixture(folder)
            audit_path = out / "fixture_failed_audit.json"
            audit_path.write_text(json.dumps({"status": "failed", "formal": True, "rows": 149}))
            with patch.dict(sys.modules, {"portable_evaluation": SimpleNamespace(audit_saved=Mock(return_value=audit_path))}):
                with self.assertRaisesRegex(ValueError, "did not establish complete"):
                    front.verify_only("parent", "code", "cache", out)

    @unittest.skipUnless(os.environ.get("GEOVAE_PUBLIC_TEST_CACHE"), "Explicit actual prepared cache not supplied")
    def test_actual_cache_inspection_preserves_missing_parent_blocker(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = front.inspect(root / "no-parent", root / "no-code", os.environ["GEOVAE_PUBLIC_TEST_CACHE"])
            self.assertTrue(result["prerequisite_blockers"])
            self.assertEqual(result["evidence"]["prepared"]["arrays"], 41)
            self.assertEqual(result["prediction_entry_enabled"], release.release_evidence()["prediction_entry_enabled"])
            self.assertEqual(result["status"], "blocked_missing_or_mismatched_prerequisites")
            self.assertEqual(result["model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
