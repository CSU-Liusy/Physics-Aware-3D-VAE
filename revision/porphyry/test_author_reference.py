"""CPU fake-array checks for saved author import mechanics, not actual150 evidence."""
import csv
import hashlib
import inspect
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import author_reference as a
import portable_evaluation as portable


def fake_metrics(raw, target, known, threshold=.5, hidden_mask=None):
    blind = ~known
    prediction = raw >= threshold
    truth = target.astype(bool)
    intersection = int(np.count_nonzero(prediction & truth & blind))
    union = int(np.count_nonzero((prediction | truth) & blind))
    return {"blind_iou": intersection / union if union else 1.,
            "blind_dice": .5, "observed_error": float(np.mean(prediction[known] != truth[known])),
            "hidden_error": float(np.mean(prediction[hidden_mask] != truth[hidden_mask])),
            "brier_blind": float(np.mean((raw[blind] - target[blind]) ** 2)),
            "volume_absolute_relative_error": .25, "hd": None, "hd95": None,
            "chamfer": None, "surface_distance_valid": False}


def fake_preparation():
    target = np.zeros((32, 32, 32), dtype=np.uint8)
    target.ravel()[:30] = 1; target.ravel()[200:260] = 1
    known, hidden = np.zeros_like(target, bool), np.zeros_like(target, bool)
    known.ravel()[:96] = True; hidden.ravel()[100:148] = True
    obs = np.stack((target * known, known)).astype(np.float32)
    samples, masks, conditions, public = [], [], [], []
    for level in a.LEVELS:
        case_id = "porphyry_01_cu_ge_" + str(level).replace(".", "p")
        samples.append({"case_id": case_id, "parent_id": a.PARENT,
                        "cu_threshold_percent": level, "target_array_sha256": a.array_sha(target)})
        for seed in a.MASKS:
            conditions.append({"case_id": case_id, "parent_id": a.PARENT, "mask_seed": seed,
                               "observation_array_sha256": a.array_sha(obs),
                               "hidden_labels_array_sha256": a.array_sha((target * hidden).astype(np.uint8))})
            public.append({"case_id": case_id, "parent_id": a.PARENT, "mask_seed": seed,
                           "cu_threshold_percent": level, "target": target.copy(), "known": known.copy(),
                           "hidden": hidden.copy(), "obs": obs.copy(),
                           "observation_identity_sha256": a.stable({"public_fake": case_id, "seed": seed})})
    for seed in a.MASKS:
        masks.append({"mask_seed": seed, "observation_mask_array_sha256": a.array_sha(known),
                      "hidden_mask_array_sha256": a.array_sha(hidden)})
    return {"parent_id": a.PARENT, "source_kind": "external synthetic", "axis_order": "zyx",
            "grid_shape": [32, 32, 32], "samples": samples, "masks": masks, "conditions": conditions}, {"conditions": public}


class AuthorReferenceTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        manifest, public = fake_preparation()
        cases, mapping = a._condition_map(manifest, public)
        identity = {"fixture": True, "method_thresholds": {method: .5 for method in a.METHODS},
                    "model_backbone_baseline_sha256": {"geovae": "fake-baseline"}}
        records = []
        engine = SimpleNamespace(metrics=fake_metrics)
        for case in cases:
            for method in a.METHODS:
                folder = root / "cases" / case["case_id"] / f"mask{case['mask_seed']}" / method
                folder.mkdir(parents=True)
                raw = np.full((32, 32, 32), .1, np.float32)
                projected = raw.copy(); projected[case["known"]] = case["obs"][0][case["known"]]
                np.save(folder / "raw.npy", raw, allow_pickle=False)
                np.save(folder / "projected.npy", projected, allow_pickle=False)
                portable.save(folder / "adaptation.json", {"fixture": True})
                portable.save(folder / "backbone.json", {"unchanged": True, "before": "fake-baseline",
                                                         "after": "fake-baseline", "loaded_baseline": "fake-baseline"})
                seed = int(hashlib.sha256(f"{case['mask_seed']}:{case['case_id']}:{method}".encode()).hexdigest()[:8], 16)
                case_sha = a.stable({"identity": a.stable(identity), "observation": case["observation_identity_sha256"],
                                     "method": method, "seed": seed, "threshold": .5})
                record = {"case_id": case["case_id"], "parent_id": a.PARENT, "mask_seed": case["mask_seed"],
                          "cu_threshold_percent": case["cu_threshold_percent"], "method": method, "model_seed": 11,
                          "case_seed": seed, "threshold_used": .5,
                          "observation_identity_sha256": case["observation_identity_sha256"],
                          "primary_probability": "raw", "projection_is_diagnostic_only": True,
                          "prediction_seconds": 1., "raw_sha256": a.sha(folder / "raw.npy"),
                          "projected_sha256": a.sha(folder / "projected.npy"),
                          **portable.metrics(engine, raw, projected, case, .5)}
                portable.save(folder / "journal.json", {"status": "complete", "identity_sha256": a.stable(identity),
                    "case_identity_sha256": case_sha, "record": record,
                    "artifacts": {name: a.sha(folder / name) for name in a.ARTIFACTS}})
                records.append(record)
        portable._csv(root / "per_case.csv", records)
        return identity, cases, mapping, engine

    def test_complete_fake150_cpu_recompute_mapping_and_source_immutability(self):
        with tempfile.TemporaryDirectory(prefix="author-reference-fake-") as root:
            identity, cases, mapping, engine = self.fixture(root)
            before = {p.relative_to(root).as_posix(): a.sha(p) for p in Path(root).rglob("*") if p.is_file()}
            inventory = a._reference_inventory(root, identity, cases)
            with patch.object(a, "verify_bundle", side_effect=AssertionError("No actual source context in pure fixture")):
                result = a._recompute_rows(root, identity, cases, engine)
            after = {p.relative_to(root).as_posix(): a.sha(p) for p in Path(root).rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(result["inventory"], inventory)
            self.assertEqual((len(result["rows"]), len(result["calibration"]), len(result["summaries"])), (150, 1500, 50))
            self.assertTrue(all(r["original_observation_identity_sha256"] != r["portable_observation_identity_sha256"] for r in mapping))
            self.assertTrue(all(r["hd95_n"] == 0 and r["hd95_mean"] is None for r in result["summaries"]))
            self.assertNotIn("status", result)
            self.assertNotIn("torch", sys.modules)

    def test_portable_target_hash_mismatch_rejected(self):
        manifest, public = fake_preparation()
        public["conditions"][0]["target"].ravel()[-1] = 1
        with self.assertRaisesRegex(ValueError, "role arrays differ"):
            a._condition_map(manifest, public)

    def test_duplicate_or_missing_condition_rejected(self):
        manifest, public = fake_preparation()
        public["conditions"][-1] = public["conditions"][0]
        with self.assertRaisesRegex(ValueError, "coverage"):
            a._condition_map(manifest, public)

    def test_corrupt_probability_fails_before_recomputation(self):
        with tempfile.TemporaryDirectory(prefix="author-reference-fake-") as root:
            identity, cases, _, engine = self.fixture(root)
            path = Path(root) / "cases" / cases[0]["case_id"] / f"mask{cases[0]['mask_seed']}" / "idw/raw.npy"
            path.write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                a._recompute_rows(root, identity, cases, engine)

    def test_csv_duplicate_and_metric_disagreement_rejected(self):
        with tempfile.TemporaryDirectory(prefix="author-reference-fake-") as root:
            identity, cases, _, engine = self.fixture(root)
            csv_path = Path(root) / "per_case.csv"
            original = csv_path.read_bytes()
            with csv_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[-1] = rows[0]; portable._csv(csv_path, rows)
            with self.assertRaisesRegex(ValueError, "row set"):
                a._reference_inventory(root, identity, cases)
            csv_path.write_bytes(original)
            path = Path(root) / "cases" / cases[0]["case_id"] / f"mask{cases[0]['mask_seed']}" / "idw/journal.json"
            journal = a.read(path); journal["record"]["blind_iou"] += .2; portable.save(path, journal)
            with self.assertRaisesRegex(ValueError, "Metric differs"):
                a._recompute_rows(root, identity, cases, engine)

    def test_public_api_has_no_verified_context_or_engine_override(self):
        self.assertEqual(list(inspect.signature(a.import_reference).parameters), [
            "bundle_path", "primary_run", "code_root", "data_cache", "reference_run", "external_manifest", "out"])
        self.assertEqual(list(inspect.signature(a.verify_reference_context).parameters), [
            "bundle_path", "primary_run", "code_root", "data_cache", "reference_run", "external_manifest"])

    def test_saved_context_requests_original_bundle_without_live_deadline(self):
        with patch.object(a, "verify_bundle", side_effect=ValueError("fake source absent")) as verify, \
             patch.object(a, "load_prepared", side_effect=AssertionError("Must not load after bundle rejection")):
            with self.assertRaisesRegex(ValueError, "fake source absent"):
                a.verify_reference_context("bundle", "primary", "code", "data", "reference", "manifest")
        verify.assert_called_once_with("bundle", "primary", "code", require_time=False)

    def test_failed_import_invalidates_old_pass_without_new_prediction(self):
        with tempfile.TemporaryDirectory(prefix="author-reference-fake-") as root:
            root = Path(root); out = root / "derived"
            portable.save(out / "author_reference_import.json", {"status": "passed_author_reference_import"})
            with patch.object(a, "verify_reference_context", side_effect=ValueError("source changed")), \
                 patch.object(portable, "_metrics_engine", side_effect=AssertionError("No scorer after failed context")):
                with self.assertRaisesRegex(ValueError, "source changed"):
                    a.import_reference(root / "bundle.json", root / "primary", root / "code", root / "data",
                                       root / "reference", root / "external/manifest.json", out)
            report = a.read(out / "author_reference_import.json")
            self.assertEqual(report["status"], "failed_author_reference_import")
            self.assertEqual(report["new_predictions"], 0)
            self.assertEqual(len(a.read(out / "import_failure_history.json")), 1)

    def test_output_must_not_overlap_input_or_submission(self):
        with tempfile.TemporaryDirectory(prefix="author-reference-fake-") as root:
            base = Path(root)
            for out, protected in ((base / "data/subdir", [base / "data"]), (base, [base / "data"]),
                                   (base / "revision_article/report", [])):
                with self.assertRaises(ValueError):
                    a._output_guard(out, protected)

    def _mock_report_context(self, root):
        # A report-protocol stub only: no prediction array or model artifact.
        reference = root / "reference"
        portable.save(reference / "per_case.csv", {"fixture": True})
        code = root / "code"; code.mkdir()
        (code / "repro_metrics.py").write_text("# fake metric source\n", encoding="utf-8")
        anchors = {"per_case.csv": a.sha(reference / "per_case.csv")}
        ctx = {"reference": reference, "identity": {"fixture": True}, "cases": [],
               "reference_inventory": [], "anchors": anchors, "mapping": [], "history": [],
               "bundle": {"bundle_sha256": "fake-bundle", "evidence": {"original_budget": {}}},
               "manifest_sha256": "fake-manifest", "preregistration_sha256": "fake-prereg",
               "public": {"manifest_sha256": "fake-public", "preparation_identity_sha256": "fake-preparation"}}
        anchors["identity.json"] = a.sha(reference / "per_case.csv")
        (reference / "identity.json").write_bytes((reference / "per_case.csv").read_bytes())
        result = {"inventory": [], "per_case_csv_sha256": anchors["per_case.csv"],
                  "rows": [{"fixture": True}], "calibration": [{"fixture": True}], "summaries": [{"fixture": True}]}
        return ctx, result

    def test_success_report_schema_matches_release_consumer_with_stub_mechanics(self):
        with tempfile.TemporaryDirectory(prefix="author-reference-report-stub-") as folder:
            root = Path(folder); ctx, result = self._mock_report_context(root)
            inventory = {"fixture-wrapper": "fake-sha"}
            with patch.object(a, "verify_reference_context", return_value=ctx), \
                 patch.object(a, "_recompute_rows", return_value=result), \
                 patch.object(portable, "_metrics_engine", return_value=object()), \
                 patch.object(portable, "source_inventory", return_value=inventory):
                report = a.import_reference(root / "bundle.json", root / "primary", root / "code", root / "data",
                                            root / "reference", root / "external/manifest.json", root / "derived")
            self.assertEqual(report["status"], "passed_author_reference_import")
            self.assertEqual(report["rows"], 150)
            self.assertEqual(report["public_prediction_rows"], 0)
            self.assertEqual(report["wrapper_source_sha256"], inventory)
            self.assertEqual(a.read(root / "derived/import_identity.json")["wrapper_source_sha256"], inventory)

    def test_source_change_during_saved_import_invalidates_report(self):
        with tempfile.TemporaryDirectory(prefix="author-reference-report-stub-") as folder:
            root = Path(folder); ctx, result = self._mock_report_context(root)
            with patch.object(a, "verify_reference_context", return_value=ctx), \
                 patch.object(a, "_recompute_rows", return_value=result), \
                 patch.object(portable, "_metrics_engine", return_value=object()), \
                 patch.object(portable, "source_inventory", side_effect=[{"fake": "old"}, {"fake": "new"}]):
                with self.assertRaisesRegex(ValueError, "source changed during"):
                    a.import_reference(root / "bundle.json", root / "primary", root / "code", root / "data",
                                       root / "reference", root / "external/manifest.json", root / "derived")
            self.assertEqual(a.read(root / "derived/author_reference_import.json")["status"], "failed_author_reference_import")


if __name__ == "__main__":
    unittest.main()
