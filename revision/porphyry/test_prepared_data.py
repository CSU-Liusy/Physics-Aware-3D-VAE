"""CPU adapter tests; fake probability arrays are never model results."""
import copy
import io
import os
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

import numpy as np

import prepared_data as adapter


class PreparedAdapterTests(unittest.TestCase):
    def fixture(self):
        masks, arrays = {}, {}
        for seed in (42, 53, 67):
            known, hidden = np.zeros((32, 32, 32), bool), np.zeros((32, 32, 32), bool)
            known.ravel()[:96] = True; hidden.ravel()[96:144] = True
            arrays[f"mask{seed}_observed.npy"], arrays[f"mask{seed}_hidden.npy"] = known, hidden
            masks[seed] = known, hidden
        manifest = {"targets": {}, "conditions": [], "artifacts": {}, "preparation_identity_sha256": "preparation-hash"}
        for index, level in enumerate((.1, .2, .3, .5, 1.)):
            case = "porphyry_01_cu_ge_" + str(level).replace(".", "p")
            target = np.zeros((32, 32, 32), np.uint8); target.ravel()[:200 - 25 * index] = 1
            target_path = f"prepared/{case}_target.npy"
            arrays[Path(target_path).name] = target
            manifest["targets"][case] = {"cu_threshold_percent": level, "target_file": target_path}
            for seed, (known, hidden) in masks.items():
                obs_file = f"prepared/{case}_mask{seed}_observation.npy"
                hidden_file = f"prepared/{case}_mask{seed}_hidden_labels.npy"
                arrays[Path(obs_file).name] = np.stack((target.astype(np.float32) * known, known.astype(np.float32)))
                arrays[Path(hidden_file).name] = (target * hidden).astype(np.uint8)
                manifest["conditions"].append({"case_id": case, "mask_seed": seed,
                                                "observation_file": obs_file, "hidden_labels_file": hidden_file})
        manifest["artifacts"] = {"prepared/" + name: {} for name in arrays}
        return manifest, {"mask_seeds": [42, 53, 67]}, arrays

    def load_fixture(self, manifest, recipe, arrays):
        def read(path, expected):
            return manifest if Path(path).name == "manifest.json" else recipe
        def load(path, record):
            return arrays[Path(path).name]
        with patch.object(adapter, "verify_prepared", return_value={"arrays": 41, "conditions": 15, "manifest_sha256": "manifest-hash"}), \
             patch.object(adapter, "_verified_json", side_effect=read), patch.object(adapter, "_verified_array", side_effect=load), \
             patch.object(adapter, "sha", return_value="source-hash"):
            return adapter.load_prepared(Path("fixture-cache"))

    def test_complete_fixture_retains_fifteen_explicit_cases(self):
        result = self.load_fixture(*self.fixture())
        self.assertEqual(len(result["conditions"]), 15)
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(len({row["observation_identity_sha256"] for row in result["conditions"]}), 15)

    def test_unobserved_label_in_input_rejected(self):
        manifest, recipe, arrays = self.fixture()
        obs_name = Path(manifest["conditions"][0]["observation_file"]).name
        arrays[obs_name] = arrays[obs_name].copy(); arrays[obs_name][0].ravel()[1000] = 1
        with self.assertRaisesRegex(ValueError, "exposes unobserved"):
            self.load_fixture(manifest, recipe, arrays)

    def test_observed_hidden_overlap_rejected(self):
        manifest, recipe, arrays = self.fixture()
        arrays["mask42_hidden.npy"] = arrays["mask42_observed.npy"].copy()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.load_fixture(manifest, recipe, arrays)

    def test_hidden_scoring_label_tampering_rejected(self):
        manifest, recipe, arrays = self.fixture()
        name = Path(manifest["conditions"][0]["hidden_labels_file"]).name
        arrays[name] = arrays[name].copy(); arrays[name].ravel()[100] ^= 1
        with self.assertRaisesRegex(ValueError, "Hidden scoring labels"):
            self.load_fixture(manifest, recipe, arrays)

    def test_duplicate_condition_and_non_nested_target_rejected(self):
        manifest, recipe, arrays = self.fixture()
        manifest["conditions"][-1] = copy.deepcopy(manifest["conditions"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load_fixture(manifest, recipe, arrays)
        manifest, recipe, arrays = self.fixture()
        name = Path(manifest["targets"]["porphyry_01_cu_ge_1p0"]["target_file"]).name
        arrays[name] = arrays[name].copy(); arrays[name].ravel()[1000] = 1
        with self.assertRaisesRegex(ValueError, "not nested"):
            self.load_fixture(manifest, recipe, arrays)

    def test_predictor_and_projection_do_not_read_target_or_hidden(self):
        loaded = self.load_fixture(*self.fixture())["conditions"][0]
        class Restricted(dict):
            def __getitem__(self, key):
                if key in {"target", "hidden", "hidden_labels"}:
                    raise AssertionError("Scoring field accessed by predictor/projection helper")
                return super().__getitem__(key)
        case = Restricted(loaded)
        obs = adapter.predictor_observation(case)
        self.assertFalse(np.shares_memory(obs, case["obs"]))
        raw = np.full((32, 32, 32), .25, dtype=np.float32)
        before = raw.copy()
        projected = adapter.project_for_diagnostic(raw, case)
        self.assertTrue(np.array_equal(raw, before))
        self.assertFalse(np.shares_memory(projected, raw))
        self.assertTrue(np.array_equal(projected[~case["known"]], raw[~case["known"]]))

    def test_invalid_probability_rejected(self):
        case = self.load_fixture(*self.fixture())["conditions"][0]
        for raw in (np.full((32, 32, 32), np.nan), np.full((32, 32, 32), 1.1), np.ones((16, 16, 16))):
            with self.subTest(shape=raw.shape), self.assertRaises(ValueError):
                adapter.project_for_diagnostic(raw, case)

    def test_array_read_after_verification_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "target.npy"
            original = np.zeros((4, 4, 4), dtype=np.uint8)
            np.save(path, original, allow_pickle=False)
            record = {"file_sha256": adapter.sha(path), "array_sha256": adapter.array_sha(original),
                      "shape": list(original.shape), "dtype": str(original.dtype)}
            mutated = original.copy(); mutated[1, 1, 1] = 1
            with patch.object(adapter.np, "load", return_value=mutated), self.assertRaisesRegex(ValueError, "Loaded prepared array differs"):
                adapter._verified_array(path, record)
            np.save(path, mutated, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                adapter._verified_array(path, record)

    def test_metadata_read_after_verification_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"; path.write_text('{"a":1}')
            old_sha = adapter.sha(path)
            path.write_text('{"a":2}')
            with self.assertRaisesRegex(ValueError, "JSON changed"):
                adapter._verified_json(path, old_sha)

    @unittest.skipUnless(os.environ.get("GEOVAE_PUBLIC_TEST_CACHE"), "Explicit actual prepared cache not supplied")
    def test_actual_registered_cache_all_conditions(self):
        data = adapter.load_prepared(os.environ["GEOVAE_PUBLIC_TEST_CACHE"])
        self.assertEqual(data["verification"]["arrays"], 41)
        self.assertEqual(len(data["conditions"]), 15)
        self.assertEqual(data["independent_parents"], 1)


if __name__ == "__main__":
    unittest.main()
