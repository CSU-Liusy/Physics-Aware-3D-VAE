"""Small CPU fixtures only; these tests never claim model or source-data parity."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("portable", HERE / "porphyry_public.py")
portable = importlib.util.module_from_spec(spec)
spec.loader.exec_module(portable)
import release_contract as release

COLD_RELEASE = {"status": "not_released", "prediction_entry_enabled": False}


class PortableGuards(unittest.TestCase):
    def table(self):
        rows = []
        for x in range(5):
            for y in range(7):
                for z in range(4):
                    row = np.zeros(16)
                    row[:4] = x * 20, y * 20, z * 20, 1000 if z < 3 else 0
                    row[11] = .2 + .1 * x
                    rows.append(row)
        return np.asarray(rows)

    def test_supported_cube_rule_and_unknown_exclusion(self):
        sampled, native, back, report = portable.supported_cube(self.table(), target_size=8)
        self.assertEqual(sampled.shape, (8, 8, 8))
        self.assertEqual(native.shape, (3, 3, 3))
        self.assertEqual(report["retained_z_indices"], [0, 3])
        self.assertEqual(report["cube_side_source_coordinate_units"], 60)
        self.assertEqual(report["target_unknown_voxels"], 0)
        self.assertEqual((sampled >= .3)[np.ix_(*back)].shape, native.shape)

    def test_support_does_not_change_with_grade(self):
        table = self.table()
        first = portable.supported_cube(table)[3]
        table[:, 11] = np.linspace(0, 20, len(table))
        self.assertEqual(portable.supported_cube(table)[3], first)

    def test_missing_duplicate_nonfinite_and_no_support_rejected(self):
        table = self.table()
        bad = [table[:-1], np.vstack((table[:-1], table[0]))]
        nonfinite = table.copy(); nonfinite[0, 11] = np.nan; bad.append(nonfinite)
        no_support = table.copy(); no_support[:, 3] = 0; bad.append(no_support)
        for value in bad:
            with self.subTest(kind=len(bad)), self.assertRaises(ValueError):
                portable.supported_cube(value)

    def test_anisotropic_grid_rejected(self):
        table = self.table(); table[:, 0] *= 2
        with self.assertRaisesRegex(ValueError, "isotropic"):
            portable.supported_cube(table)

    def test_safe_output_rejects_source_and_submission_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            code, source = root / "code", root / "source" / "block.csv"
            for out in (code / "cache", root / "revision_article" / "cache", source.parent, source.parent / "cache"):
                with self.subTest(out=out), self.assertRaises(ValueError):
                    portable.guard_output(out, code, (source,))
            self.assertEqual(portable.guard_output(root / "cache", code, (source,)), (root / "cache").resolve())

    def test_relative_escape_rejected(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            portable.inside(folder, "../outside.json")

    def test_reference_has_no_dataset_arrays_or_absolute_paths(self):
        reference = portable.read(portable.REFERENCE)
        self.assertEqual(len(reference["reference_arrays"]), 41)
        self.assertEqual(reference["levels"], [.1, .2, .3, .5, 1.])
        for name in reference["reference_arrays"]:
            self.assertFalse(Path(name).is_absolute())
        self.assertNotIn("gpt_thinking", json.dumps(reference))

    def test_prepare_requires_license_before_source_read(self):
        with self.assertRaisesRegex(ValueError, "accept"):
            portable.prepare("missing", "missing-license", "code", "cache")

    def test_prepare_wrong_source_hash_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / "source").mkdir()
            source = root / "source" / "source.csv"; source.write_text("wrong")
            with self.assertRaisesRegex(ValueError, "source size/hash"):
                portable.prepare(source, source.parent / "LICENSE", root / "code", root / "cache", True)
            self.assertFalse((root / "cache").exists())

    def test_author_nonpublic_lifecycle_rejected_before_any_code_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "run_identity.json").write_text(json.dumps({"mode": "formal"}))
            (root / "public_handoff.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "internal author runs"):
                portable.public_primary(root, root / "unread-code")

    def test_supplement_budget_preserves_start_and_rejects_changes(self):
        parent = {"verified_public_lifecycle": True, "run_identity_sha256": "a", "public_handoff_sha256": "b"}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "run"
            first = portable.supplemental_budget(out, parent, 1, now=1000)
            second = portable.supplemental_budget(out, parent, 1, now=1100)
            self.assertEqual(first, second)
            with self.assertRaisesRegex(ValueError, "clock cannot be reset"):
                portable.supplemental_budget(out, parent, 2, now=1100)
            with self.assertRaisesRegex(ValueError, "expired"):
                portable.supplemental_budget(out, parent, 1, now=4600)
            self.assertEqual(portable.read(out / "public_supplement_budget.json")["started_unix"], 1000)

    def test_budget_requires_public_evidence_and_finite_values(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "verified public"):
                portable.supplemental_budget(Path(folder) / "out", {}, 1, now=1000)
            for hours in (0, -1, float("nan"), float("inf"), 73):
                parent = {"verified_public_lifecycle": True}
                with self.subTest(hours=hours), self.assertRaises(ValueError):
                    portable.supplemental_budget(Path(folder) / "out", parent, hours, now=1000)

    def test_existing_nonempty_output_cannot_start_new_clock(self):
        parent = {"verified_public_lifecycle": True, "run_identity_sha256": "a", "public_handoff_sha256": "b"}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder); (out / "old.txt").write_text("old")
            with self.assertRaisesRegex(ValueError, "empty output"):
                portable.supplemental_budget(out, parent, 1, now=1000)

    def test_inspect_never_means_evaluator_ready(self):
        with patch.object(portable, "public_primary", return_value={"checked": True}), \
             patch.object(portable, "verify_prepared", return_value={"checked": True}), \
             patch.object(release, "release_evidence", return_value=COLD_RELEASE):
            result = portable.inspect("run", "code", "cache")
        self.assertEqual(result["status"], "blocked_portable_evaluator_not_validated")
        self.assertFalse(result["prediction_entry_enabled"])
        self.assertEqual(result["model_calls"], 0)

    def test_inspect_missing_preconditions_retained(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(release, "release_evidence", return_value=COLD_RELEASE):
            result = portable.inspect(Path(folder) / "run", Path(folder) / "code", Path(folder) / "cache")
            self.assertEqual(len(result["prerequisite_blockers"]), 2)
            self.assertFalse(result["prediction_entry_enabled"])

    def test_released_inspection_still_reports_missing_prerequisites_without_io(self):
        self.assertTrue(release.release_evidence()["prediction_entry_enabled"])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = portable.inspect(root / "run", root / "code", root / "cache")
            self.assertEqual(len(result["prerequisite_blockers"]), 2)
            self.assertTrue(result["prediction_entry_enabled"])
            self.assertEqual(result["status"], "blocked_missing_or_mismatched_prerequisites")
            self.assertEqual(result["model_calls"], 0)
            self.assertFalse(result["numerical_validation_performed_by_this_inspection"])
            self.assertEqual(list(root.iterdir()), [])

    def test_released_inspection_readiness_is_metadata_only(self):
        self.assertTrue(release.release_evidence()["prediction_entry_enabled"])
        with patch.object(portable, "public_primary", return_value={"fixture": True}), \
             patch.object(portable, "verify_prepared", return_value={"fixture": True}), \
             patch.object(portable, "supplemental_budget", side_effect=AssertionError("inspection initialized clock")):
            result = portable.inspect("run", "code", "cache")
        self.assertEqual(result["status"], "ready_released_prerequisites")
        self.assertTrue(result["prediction_entry_enabled"])
        self.assertEqual(result["model_calls"], 0)
        self.assertFalse(result["numerical_validation_performed_by_this_inspection"])

    def test_help_runs_from_cold_copy_without_author_workspace(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "porphyry_public.py").write_bytes((HERE / "porphyry_public.py").read_bytes())
            (root / "conversion_reference.json").write_bytes(portable.REFERENCE.read_bytes())
            result = subprocess.run([sys.executable, str(root / "porphyry_public.py"), "--help"],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("prepare", result.stdout)
            self.assertIn("inspect", result.stdout)

    def test_run_command_is_explicitly_blocked_without_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = subprocess.run([sys.executable, str(HERE / "porphyry_public.py"), "run",
                                     "--primary-run", str(root / "missing-run"), "--code-root", str(root / "missing-code"),
                                     "--data-cache", str(root / "missing-cache")],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["model_calls"], 0)
            self.assertEqual(list(root.iterdir()), [])

    @unittest.skipUnless(os.environ.get("GEOVAE_PUBLIC_TEST_CODE_ROOT"), "Explicit code root not supplied")
    def test_registered_observation_source_and_six_mask_hashes(self):
        reference = portable.read(portable.REFERENCE)
        make_masks, identity = portable.observation_functions(os.environ["GEOVAE_PUBLIC_TEST_CODE_ROOT"], reference)
        self.assertEqual(identity["functions_ast_sha256"], reference["observation_ast_sha256"])
        for seed in reference["mask_seeds"]:
            known, hidden = make_masks((32, 32, 32), portable.PARENT, 8, 12, seed, 4)
            for suffix, value in (("observed", known), ("hidden", hidden)):
                relative = f"prepared/mask{seed}_{suffix}.npy"
                self.assertEqual(portable.array_sha(value), reference["reference_arrays"][relative]["array_sha256"])


if __name__ == "__main__":
    unittest.main()
