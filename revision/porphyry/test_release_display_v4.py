"""CPU fake-registry display checks; fixtures are never release/model evidence."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import porphyry_public as front
import portable_evaluation as portable
import release_contract as contract
import test_frontend_coupling as frontend_fixtures


class ReleaseDisplayChecks(unittest.TestCase):
    def registry(self, folder, variant="valid"):
        root = Path(folder)
        sources = portable.source_inventory()
        proof = {
            "actual10": {"status": "passed_actual_public_subset_parity", "actual_prediction_rows": 10,
                         "scope": contract.SCOPE, "wrapper_source_sha256": sources},
            "reference150": {"status": "passed_author_reference_import", "rows": 150,
                             "public_prediction_rows": 0, "wrapper_source_sha256": sources},
            "independent_review": {"status": "passed_release_review", "unresolved_findings": [],
                                   "wrapper_source_sha256": sources},
        }
        if variant == "negative":
            proof["actual10"]["status"] = "failed_actual_public_subset_parity"
        if variant == "missing_role":
            del proof["reference150"]
        registry = {"format": "porphyry-portable-reviewed-release-v1",
                    "wrapper_source_sha256": deepcopy(sources), "evidence": {}}
        if variant == "stale":
            registry["wrapper_source_sha256"]["porphyry_public.py"] = "old-source"
        for role, value in proof.items():
            path = root / (role + "_fake_fixture.json")
            path.write_text(json.dumps(value), encoding="utf-8")
            registry["evidence"][role] = {"file": path.name, "sha256": front.sha(path)}
        (root / "release_registry.json").write_text(json.dumps(registry), encoding="utf-8")

    def inspect_fixture(self):
        with patch.object(front, "public_primary", return_value={"fixture_only": True}), \
             patch.object(front, "verify_prepared", return_value={"fixture_only": True}):
            return front.inspect("fixture-parent", "fixture-code", "fixture-cache")

    def test_missing_release_with_ready_prerequisites_is_not_parity(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(contract, "HERE", Path(folder)):
            result = self.inspect_fixture()
        self.assertTrue(result["prerequisites_ready"])
        self.assertFalse(result["prediction_entry_enabled"])
        self.assertFalse(result["portable_ten_method_parity_test_passed"])
        self.assertEqual(result["release_evidence"]["status"], "not_released")

    def test_checked_registry_display_preserves_bounded_claims(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(contract, "HERE", Path(folder)):
            self.registry(folder)
            result = self.inspect_fixture()
        self.assertEqual(result["status"], "ready_released_prerequisites")
        self.assertTrue(result["prediction_entry_enabled"])
        self.assertTrue(result["portable_ten_method_parity_test_passed"])
        self.assertFalse(result["numerical_validation_performed_by_this_inspection"])
        self.assertEqual(result["model_calls"], 0)
        evidence = result["release_evidence"]
        self.assertEqual(evidence["actual_public_subset_rows"], 10)
        self.assertEqual(evidence["author_reference_saved_only_rows"], 150)
        self.assertFalse(evidence["independent_public_full150_executed"])
        self.assertFalse(evidence["public_from_scratch_training_executed"])

    def test_stale_negative_and_incomplete_registry_keep_display_closed(self):
        for variant in ("stale", "negative", "missing_role"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as folder, \
                 patch.object(contract, "HERE", Path(folder)):
                self.registry(folder, variant)
                result = self.inspect_fixture()
            self.assertFalse(result["prediction_entry_enabled"])
            self.assertFalse(result["portable_ten_method_parity_test_passed"])
            self.assertEqual(result["release_evidence"]["status"], "blocked_invalid_release_evidence")

    def test_release_does_not_hide_missing_local_prerequisites(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(contract, "HERE", Path(folder)):
            self.registry(folder)
            with patch.object(front, "public_primary", side_effect=ValueError("missing local parent")), \
                 patch.object(front, "verify_prepared", return_value={}):
                result = front.inspect("p", "c", "d")
        self.assertTrue(result["prediction_entry_enabled"])  # Package release, not local run readiness.
        self.assertFalse(result["prerequisites_ready"])
        self.assertEqual(result["status"], "blocked_missing_or_mismatched_prerequisites")
        self.assertIn("missing local parent", result["prerequisite_blockers"][0])

    def test_malformed_registry_or_proof_displays_closed(self):
        for target in ("registry_list", "registry_null", "proof_list"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as folder, \
                 patch.object(contract, "HERE", Path(folder)):
                self.registry(folder)
                registry_path = Path(folder) / "release_registry.json"
                if target.startswith("registry_"):
                    registry_path.write_text("[]" if target == "registry_list" else "null")
                else:
                    registry = front.read(registry_path)
                    proof = Path(folder) / registry["evidence"]["actual10"]["file"]
                    proof.write_text("[]")
                    registry["evidence"]["actual10"]["sha256"] = front.sha(proof)
                    registry_path.write_text(json.dumps(registry))
                result = self.inspect_fixture()
                self.assertFalse(result["prediction_entry_enabled"])
                self.assertEqual(result["release_evidence"]["status"], "blocked_invalid_release_evidence")

    def test_saved_only_reports_release_independently_without_prediction(self):
        for variant, enabled in ((None, False), ("valid", True), ("stale", False), ("negative", False)):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as folder, \
                 patch.object(contract, "HERE", Path(folder)):
                if variant:
                    self.registry(folder, variant)
                out = frontend_fixtures.FrontendCouplingTests().saved_fixture(folder)
                audit = out / "fake_saved_audit.json"
                audit.write_text(json.dumps({"status": "passed_saved_probability_recomputation", "formal": True,
                                              "rows": 150, "no_prediction_or_weight_loading": True}))
                with patch.object(portable, "audit_saved", return_value=audit), \
                     patch.object(portable, "run", side_effect=AssertionError("prediction called")), \
                     patch.object(portable, "_engine", side_effect=AssertionError("model runtime imported")), \
                     patch.object(front, "supplemental_budget", side_effect=AssertionError("clock created")):
                    result = front.verify_only("p", "c", "d", out)
                self.assertEqual(result["status"], "verified_saved_public_results")
                self.assertEqual(result["prediction_entry_enabled"], enabled)
                self.assertEqual(result["model_calls"], 0)
                self.assertFalse(result["model_tensors_loaded"])
                self.assertFalse(result["clock_initialized"])

    def test_cli_ready_release_inspection_success_code_is_not_model_success(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(contract, "HERE", Path(folder)):
            self.registry(folder)
            stream = io.StringIO()
            with patch.object(front, "public_primary", return_value={}), \
                 patch.object(front, "verify_prepared", return_value={}), redirect_stdout(stream):
                code = front.main(["inspect", "--primary-run", "p", "--code-root", "c", "--data-cache", "d"])
        self.assertEqual(code, 0)
        result = json.loads(stream.getvalue())
        self.assertEqual(result["status"], "ready_released_prerequisites")
        self.assertFalse(result["numerical_validation_performed_by_this_inspection"])
        self.assertEqual(result["model_calls"], 0)

    def test_no_torch_imported(self):
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
