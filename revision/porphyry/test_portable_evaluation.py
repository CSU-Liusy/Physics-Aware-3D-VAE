"""CPU fake-engine mechanics only. Never imports torch or any predictor."""
from __future__ import annotations

import ast
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import portable_evaluation as module
import release_contract as release

COLD_RELEASE = {"status": "not_released", "prediction_entry_enabled": False}


def fake_metrics(probability, target, known, *, threshold, hidden_mask):
    predicted, occupied = probability >= threshold, target.astype(bool)
    blind = ~known
    intersection = int((predicted & occupied & blind).sum())
    union = int(((predicted | occupied) & blind).sum())
    denominator = int(predicted[blind].sum() + occupied[blind].sum())
    return {"blind_iou": intersection / union if union else 1.,
            "blind_dice": 2 * intersection / denominator if denominator else 1.,
            "observed_error": float(np.mean(predicted[known] != occupied[known])),
            "hidden_error": float(np.mean(predicted[hidden_mask] != occupied[hidden_mask])),
            "brier_blind": float(np.mean((probability[blind] - target[blind]) ** 2)),
            "volume_absolute_relative_error": float(abs(predicted.sum() - occupied.sum()) / max(int(occupied.sum()), 1)),
            "surface_distance_valid": False, "hd": None, "hd95": None, "chamfer": None}


class FakeEngine:
    mechanical_fixture = True

    def __init__(self):
        self.calls = 0
        self.fail_method_once = None
        self.contaminate = False
        self.metrics = fake_metrics

    @staticmethod
    def fingerprint(model):
        return str(model.value)

    @staticmethod
    def save_array(path, value):
        with Path(path).open("wb") as stream:
            np.save(stream, value, allow_pickle=False)

    def predict(self, method, observation, models, config, device, training_image, seed):
        self.calls += 1
        assert isinstance(observation, np.ndarray) and observation.shape == (2, 32, 32, 32)
        assert isinstance(seed, int) and method in module.METHODS
        if self.fail_method_once == method:
            self.fail_method_once = None
            raise RuntimeError("deliberate fake predictor error")
        raw = np.full((32, 32, 32), .4, dtype=np.float32)
        # A predictor gets an owned copy. This mutation must never corrupt the
        # case observations subsequently used for scoring or another method.
        observation.fill(123)
        if self.contaminate:
            models["geovae"].value += 1
        return raw, {"fixture": True, "seed": seed, "projection_applied": False}


def fixture():
    linear = np.arange(32 ** 3).reshape((32, 32, 32))
    known, hidden = np.zeros(linear.shape, bool), np.zeros(linear.shape, bool)
    known.flat[:96], hidden.flat[96:144] = True, True
    cases = []
    for rank, level in enumerate(module.LEVELS):
        target = ((linear % 100) >= rank * 15).astype(np.uint8)
        for seed in module.MASKS:
            obs = np.stack((target * known, known)).astype(np.float32)
            cases.append({"case_id": "fake-cu" + str(rank), "parent_id": module.PARENT,
                          "mask_seed": seed, "cu_threshold_percent": level, "target": target,
                          "known": known.copy(), "hidden": hidden.copy(), "obs": obs,
                          "observation_identity_sha256": module.stable({"rank": rank, "seed": seed})})
    engine = FakeEngine()
    models = {kind: SimpleNamespace(value=index) for index, kind in enumerate(module.KINDS)}
    thresholds = {method: .5 for method in module.METHODS}
    identity = {"format": "porphyry-public-evaluation-v1", "formal": False, "mechanical_fixture": True,
                "method_thresholds": thresholds, "methods": list(module.METHODS),
                "model_backbone_baseline_sha256": {kind: engine.fingerprint(model) for kind, model in models.items()}}
    return {"conditions": cases}, engine, models, thresholds, identity


def fake_primary_files(root):
    """Small metadata-only unit fixture. The checkpoint bytes are not tensors."""
    code, run, cache, out = (root / name for name in ("code", "primary", "cache", "out"))
    protocol = code / "protocol"
    protocol.mkdir(parents=True)
    target = np.ones((32, 32, 32), dtype=np.uint8)
    FakeEngine.save_array(protocol / "ti.npy", target)
    module.save(protocol / "manifest.json", {"samples": [{"split": "train", "sample_id": "ti",
        "path": "ti.npy", "sha256": module.sha(protocol / "ti.npy")}]})
    module.save(code / "revision_numerics.py", {"fixture": "not_importable_or_executed"})
    config = {"mask_seed": 42, "mps_training_image_id": "ti"}
    module.save(run / "tuning/selected_config.json", config)
    module.save(run / "thresholds.json", {"thresholds": {method: .5 for method in module.METHODS}})
    checkpoints = {}
    for kind in module.KINDS:
        path = run / "models" / kind / "seed11/best.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(("NOT_A_TENSOR_" + kind).encode())
        checkpoints[kind] = module.sha(path)
    metadata = {"checkpoints": checkpoints, "mps_training_image_id": "ti",
                "threshold_selection_sha256": module.sha(run / "thresholds.json")}
    intent = {"checkpoints": checkpoints, "mps_training_image_id": "ti",
              "mps_training_image_array_sha256": module.array_hash(target.astype(np.float32)),
              "config_sha256": module.stable(config), "manifest_sha256": module.PRIMARY_SHA,
              "source_sha256": module.CORE_SHA}
    for split in ("validation", "test"):
        for seed in (11, 23, 37):
            module.save(run / split / f"seed{seed}/evaluation.json", metadata)
    module.save(run / "test/seed11/evaluation_intent.json", intent)
    for name in ("run_identity.json", "public_handoff.json", "revision_budget.json", "tuning/selection.json"):
        module.save(run / name, {"fake_metadata_fixture": True})
    parent = {"verified_public_lifecycle": True, "run_identity_sha256": module.sha(run / "run_identity.json"),
              "public_handoff_sha256": module.sha(run / "public_handoff.json"),
              "config_sha256": module.stable(config), "thresholds_sha256": module.sha(run / "thresholds.json"),
              "parent_budget_sha256": module.sha(run / "revision_budget.json"), "parent_deadline_unix": time.time() - 1,
              "checkpoint_tensors_loaded": False, "model_calls": 0}
    data = fixture()[0]
    data.update(manifest_sha256="fake_prepared_manifest", preparation_identity_sha256="fake_prepared_recipe",
                preparation_identity={"source_commit": "fake_commit", "source_sha256": "fake_source"})
    return code, run, cache, out, parent, data


class PortableEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Explicitly inside the authorized staging workspace; not a main run.
        cls.temp = tempfile.TemporaryDirectory(prefix="mechanical_eval_", dir=None)
        cls.folder = Path(cls.temp.name)
        cls.data, cls.engine, cls.models, cls.thresholds, cls.identity = fixture()
        cls.out = cls.folder / "complete_fake"
        cls.before_obs = [module.array_hash(case["obs"]) for case in cls.data["conditions"]]
        module._loop(cls.data, cls.out, cls.identity, {}, cls.thresholds, cls.engine,
                     cls.models, "cpu", np.ones((2, 2, 2)), time.time() + 300)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_01_no_runtime_import(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("repro_pipeline", sys.modules)
        self.assertFalse(module.PREDICTION_ENTRY_ENABLED)

    def test_02_complete150_and_owned_observation(self):
        self.assertEqual(self.engine.calls, 150)
        state = module.read(self.out / "status.json")
        self.assertEqual(state["completed_rows"], 150)
        self.assertEqual(state["status"], "predictions_complete_pending_saved_audit")
        self.assertEqual(self.before_obs, [module.array_hash(case["obs"]) for case in self.data["conditions"]])

    def test_03_resume_has_zero_new_predictions(self):
        engine = FakeEngine()
        module._loop(self.data, self.out, self.identity, {}, self.thresholds,
                     engine, self.models, "cpu", None, time.time() + 300)
        self.assertEqual(engine.calls, 0)

    def test_04_raw_and_projection_are_separate(self):
        case = self.data["conditions"][0]
        folder = self.out / "cases" / case["case_id"] / f"mask{case['mask_seed']}" / "idw"
        raw, projected = [np.load(folder / name, allow_pickle=False) for name in ("raw.npy", "projected.npy")]
        self.assertTrue(np.all(raw == np.float32(.4)))
        self.assertTrue(np.array_equal(projected[case["known"]], case["obs"][0][case["known"]]))
        self.assertTrue(np.array_equal(projected[~case["known"]], raw[~case["known"]]))
        row = module.read(folder / "journal.json")["record"]
        self.assertEqual(row["primary_probability"], "raw")
        self.assertGreater(row["input_mse"], 0)
        self.assertEqual(row["projected_observed_error"], 0)

    def test_05_saved_only_recomputation150(self):
        calls = self.engine.calls
        module._audit_saved_data(self.data, self.out, self.engine, self.identity)
        audit = module.read(self.out / "saved_prediction_audit.json")
        self.assertEqual(audit["rows"], 150)
        self.assertTrue(audit["no_prediction_or_weight_loading"])
        self.assertFalse(audit["formal"])
        self.assertEqual(self.engine.calls, calls)
        self.assertEqual(module.read(self.out / "status.json")["status"], "fake_fixture_complete_not_formal")
        import csv
        with (self.out / "reliability.csv").open(newline="", encoding="utf-8") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 1500)
        with (self.out / "descriptive_summary.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 50)
        self.assertTrue(all(row["independent_parents"] == "1" and row["mask_count"] == "3" for row in rows))

    def test_06_formal_entry_failclosed_before_io(self):
        target = self.folder / "never_create"
        # Exercise an explicitly unreleased state; the shipped current registry
        # is valid and is tested independently below.
        with patch.object(release, "release_evidence", return_value=COLD_RELEASE):
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                module.run(primary_run=Path("absent"), out=target)
            self.assertFalse(target.exists())
            identity = {**self.identity, "formal": True}
            with self.assertRaisesRegex(ValueError, "disabled"):
                module._loop(self.data, target, identity, {}, self.thresholds,
                             self.engine, self.models, "cpu", None, time.time() + 60)
        self.assertFalse(target.exists())

    def test_07_fake_flag_cannot_create_formal_success(self):
        with self.assertRaises(ValueError):
            module._prediction_permission({**self.identity, "mechanical_fixture": False}, self.engine)
        with self.assertRaises(ValueError):
            module._prediction_permission(self.identity, SimpleNamespace(mechanical_fixture=False))

    def test_08_incomplete_or_duplicate_cases_rejected_before_io(self):
        target = self.folder / "missing_case"
        for cases in (self.data["conditions"][:-1], self.data["conditions"][:-1] + self.data["conditions"][:1]):
            with self.assertRaises(ValueError):
                module._loop({"conditions": cases}, target, self.identity, {}, self.thresholds,
                             self.engine, self.models, "cpu", None, time.time() + 60)
        self.assertFalse(target.exists())

    def test_09_identity_change_refuses_resume(self):
        with self.assertRaisesRegex(ValueError, "identity changed"):
            module._loop(self.data, self.out, {**self.identity, "extra": "changed"}, {}, self.thresholds,
                         self.engine, self.models, "cpu", None, time.time() + 60)

    def test_10_corrupt_artifact_refuses_saved_reuse(self):
        case = self.data["conditions"][0]
        folder = self.out / "cases" / case["case_id"] / f"mask{case['mask_seed']}" / "idw"
        journal = module.read(folder / "journal.json")
        path = folder / "raw.npy"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"wrong")
            with self.assertRaisesRegex(ValueError, "checksum"):
                module._saved(self.out, case, "idw", journal["case_identity_sha256"], self.engine, self.identity)
        finally:
            path.write_bytes(original)

    def test_11_changed_metric_refuses_saved_reuse(self):
        case = self.data["conditions"][0]
        path = self.out / "cases" / case["case_id"] / f"mask{case['mask_seed']}" / "idw/journal.json"
        original = path.read_bytes()
        value = module.read(path)
        value["record"]["blind_iou"] = 99
        try:
            module.save(path, value)
            with self.assertRaisesRegex(ValueError, "Stored metric"):
                module._saved(self.out, case, "idw", value["case_identity_sha256"], self.engine, self.identity)
        finally:
            path.write_bytes(original)

    def test_12_error_stays_visible_and_resume_retains_history(self):
        data, engine, models, thresholds, identity = fixture()
        engine.fail_method_once = "idw"
        out = self.folder / "failure_then_resume"
        with self.assertRaisesRegex(ValueError, "incomplete"):
            module._loop(data, out, identity, {}, thresholds, engine, models, "cpu", None, time.time() + 300)
        self.assertEqual(module.read(out / "status.json")["completed_rows"], 149)
        self.assertEqual(len(module.read(out / "failures.json")), 1)
        engine.calls = 0
        module._loop(data, out, identity, {}, thresholds, engine, models, "cpu", None, time.time() + 300)
        self.assertEqual(engine.calls, 1)
        self.assertEqual(module.read(out / "failures.json"), [])
        self.assertEqual(len(module.read(out / "failure_history.json")), 1)

    def test_13_backbone_contamination_aborts(self):
        data, engine, models, thresholds, identity = fixture()
        engine.contaminate = True
        out = self.folder / "contamination"
        with self.assertRaisesRegex(ValueError, "contaminated"):
            module._loop(data, out, identity, {}, thresholds, engine, models, "cpu", None, time.time() + 60)
        self.assertEqual(engine.calls, 1)
        self.assertEqual(module.read(out / "status.json")["status"], "backbone_failure")
        self.assertEqual(len(module.read(out / "failure_history.json")), 1)

    def test_14_expired_loop_has_no_predictions(self):
        data, engine, models, thresholds, identity = fixture()
        out = self.folder / "expired"
        with self.assertRaises(TimeoutError):
            module._loop(data, out, identity, {}, thresholds, engine, models, "cpu", None, time.time() - 1)
        self.assertEqual(engine.calls, 0)
        self.assertEqual(module.read(out / "status.json")["status"], "budget_exceeded")

    def test_15_ti_requires_same_primary_identity_and_array(self):
        value = np.zeros((2, 2, 2), dtype=np.float32)
        good = {"mps_training_image_id": "train-only", "mps_training_image_array_sha256": module.array_hash(value)}
        self.assertEqual(module.checked_training_image("train-only", value, good, good), module.array_hash(value))
        for bad in ({**good, "mps_training_image_id": "not-training"}, {**good, "mps_training_image_array_sha256": "bad"}):
            with self.assertRaises(ValueError):
                module.checked_training_image("train-only", value, good, bad)

    def test_16_native_scratch_restores_state_even_on_error(self):
        before = Path.cwd(), tempfile.tempdir, {key: os.environ.get(key) for key in ("TMPDIR", "TMP", "TEMP")}
        out = self.folder / "native"
        with self.assertRaisesRegex(RuntimeError, "fake"):
            with module.native_temporary_files(out):
                self.assertEqual(Path.cwd(), out / "native_tmp")
                raise RuntimeError("fake")
        self.assertEqual(before, (Path.cwd(), tempfile.tempdir, {key: os.environ.get(key) for key in ("TMPDIR", "TMP", "TEMP")}))

    def test_17_deadline_kills_only_owned_fake_group_even_when_record_writes_fail(self):
        # This test never starts a child, installs a real timer or sends a signal.
        for write_failures in ((False, False), (True, False), (False, True), (True, True)):
            attempts, handlers, killed = [], {}, []
            def fake_save(path, value):
                index = len(attempts)
                attempts.append(Path(path).name)
                if write_failures[index]:
                    raise OSError("fake ENOSPC")
            def fake_signal(kind, handler):
                handlers[kind] = handler
                return "previous"
            with ExitStack() as stack:
                # Replace only this module's OS namespace. Patching the global
                # os.name would change pathlib's platform selection on Windows
                # and falsely skip both best-effort writes before fake_save.
                fake_os = SimpleNamespace(name="posix", getpid=lambda: 777, getpgrp=lambda: 777,
                                          killpg=lambda group, sig: killed.append((group, sig)))
                stack.enter_context(patch.object(module, "os", fake_os))
                stack.enter_context(patch.object(module, "save", fake_save))
                stack.enter_context(patch.object(module.signal, "SIGALRM", 14, create=True))
                stack.enter_context(patch.object(module.signal, "SIGKILL", 9, create=True))
                stack.enter_context(patch.object(module.signal, "ITIMER_REAL", 0, create=True))
                stack.enter_context(patch.object(module.signal, "signal", fake_signal))
                stack.enter_context(patch.object(module.signal, "setitimer", lambda *args: None, create=True))
                with module.deadline_guard(time.time() + 30, self.folder):
                    handlers[14](14, None)
            self.assertEqual(attempts, ["budget_stop.json", "status.json"])
            self.assertEqual(killed, [(777, 9)])

    def test_18_csv_tampering_refuses_saved_audit(self):
        path = self.out / "per_case.csv"
        original = path.read_bytes()
        try:
            path.write_bytes(original + original.splitlines(keepends=True)[1])
            with self.assertRaisesRegex(ValueError, "CSV"):
                module._audit_saved_data(self.data, self.out, self.engine, self.identity)
        finally:
            path.write_bytes(original)

    def test_19_output_cannot_overlap_sources_or_primary(self):
        root = self.folder
        code, cache, primary = root / "code", root / "cache", root / "primary"
        for out in (code / "generated", cache / "generated", primary / "generated", root):
            with self.assertRaises(ValueError):
                module.output_guard(out, code, cache, primary)

    def test_20_backend_cold_status_failclosed(self):
        # No registry file is removed or rewritten: only this test child's
        # release query is replaced with a cold-state fixture.
        script = ("import runpy,sys;sys.path.insert(0,sys.argv.pop(1));"
                  "import release_contract;release_contract.release_evidence=lambda:"
                  "{'status':'not_released','prediction_entry_enabled':False};"
                  "sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')")
        result = subprocess.run([sys.executable, "-B", "-c", script, str(HERE), str(HERE / "portable_evaluation.py")],
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        status = json.loads(result.stdout)
        self.assertEqual(status, COLD_RELEASE)

    def test_21_context_ti_dtype_and_actual_weight_hash_binding(self):
        with tempfile.TemporaryDirectory(prefix="metadata_only_", dir=None) as folder:
            code, run, cache, out, parent, data = fake_primary_files(Path(folder))
            with patch.object(module, "public_primary", return_value=parent) as gate, patch.object(module, "load_prepared", return_value=data):
                ctx = module.context(run, code, cache, out)
                gate.assert_called_once_with(run.resolve(), code.resolve(), require_time=False)
                self.assertEqual(ctx["training_image"].dtype, np.float32)
                self.assertEqual(set(ctx["checkpoint_sha256"]), set(module.KINDS))
                self.assertFalse(out.exists())
                (run / "models/geovae/seed11/best.pt").write_bytes(b"different_not_a_tensor")
                with self.assertRaisesRegex(ValueError, "weights differ"):
                    module.context(run, code, cache, out)

    def test_22_context_refuses_config_threshold_and_ti_changes(self):
        for target_name in ("config", "threshold", "ti"):
            with tempfile.TemporaryDirectory(prefix="metadata_only_", dir=None) as folder:
                code, run, cache, out, parent, data = fake_primary_files(Path(folder))
                if target_name == "config":
                    module.save(run / "tuning/selected_config.json", {"mask_seed": 99, "mps_training_image_id": "ti"})
                elif target_name == "threshold":
                    module.save(run / "thresholds.json", {"thresholds": {method: .6 for method in module.METHODS}})
                else:
                    FakeEngine.save_array(code / "protocol/ti.npy", np.zeros((32, 32, 32), np.uint8))
                with patch.object(module, "public_primary", return_value=parent), patch.object(module, "load_prepared", return_value=data):
                    with self.assertRaises(ValueError):
                        module.context(run, code, cache, out)
                self.assertFalse(out.exists())

    def test_23_saved_context_refuses_source_weight_and_budget_changes(self):
        with tempfile.TemporaryDirectory(prefix="metadata_only_", dir=None) as folder:
            code, run, cache, out, parent, data = fake_primary_files(Path(folder))
            with patch.object(module, "public_primary", return_value=parent), patch.object(module, "load_prepared", return_value=data):
                ctx = module.context(run, code, cache, out)
            budget = module.supplemental_budget(out, parent, 1)
            engine = FakeEngine()
            engine.numerics_sha = lambda: module.sha(code / "revision_numerics.py")
            engine.numerics = lambda: {"fake": True}
            engine.environment = lambda: {"fake": True}
            identity = module.make_identity(ctx, engine, self.models, budget)
            module._assert_saved_context(identity, ctx)
            for changed in ({**identity, "checkpoint_sha256": {}},
                            {**identity, "wrapper_source_sha256": {}},
                            {**identity, "training_image_array_sha256": "wrong"}):
                with self.assertRaises(ValueError):
                    module._assert_saved_context(changed, ctx)
            clock = module.read(out / "public_supplement_budget.json")
            module.save(out / "public_supplement_budget.json", {**clock, "started_unix": clock["started_unix"] + 1})
            with self.assertRaisesRegex(ValueError, "deadline"):
                module._assert_saved_context(identity, ctx)

    def test_24_direct_future_entry_remains_disabled_before_io(self):
        out = self.folder / "never_load_future"
        with patch.object(release, "release_evidence", return_value=COLD_RELEASE):
            with self.assertRaisesRegex(ValueError, "disabled"):
                module._run_after_release("absent", "absent", "absent", out)
        self.assertFalse(out.exists())

    def test_current_release_requires_parent_before_model_loading_or_clock(self):
        self.assertTrue(release.release_evidence()["prediction_entry_enabled"])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); out = root / "out"
            with patch.object(module, "_engine", side_effect=AssertionError("missing parent imported model engine")) as engine, \
                 patch.object(module, "supplemental_budget", side_effect=AssertionError("missing parent initialized clock")) as clock:
                with self.assertRaises(FileNotFoundError):
                    module.run(root / "missing-primary", root / "missing-code", root / "missing-cache", out)
            engine.assert_not_called()
            clock.assert_not_called()
            self.assertFalse(out.exists())

    def test_current_release_backend_status_is_bounded_and_import_only(self):
        result = subprocess.run([sys.executable, "-B", str(HERE / "portable_evaluation.py")],
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        status = json.loads(result.stdout)
        self.assertEqual(status, release.release_evidence())
        self.assertTrue(status["prediction_entry_enabled"])
        self.assertFalse(status["independent_public_full150_executed"])
        self.assertFalse(status["public_from_scratch_training_executed"])

    def test_25_relabelled_formal_saved_fixture_rejected_without_actual_paths(self):
        identity = {**self.identity, "formal": True, "mechanical_fixture": False}
        target = self.folder / "never_promote_formal"
        with self.assertRaisesRegex(ValueError, "actual public provenance paths"):
            module._audit_saved_data(self.data, target, self.engine, identity)
        self.assertFalse(target.exists())
        self.assertFalse(module.PREDICTION_ENTRY_ENABLED)

    def test_26_preverified_dict_or_caller_fake_engine_is_not_formal_provenance(self):
        identity = {**self.identity, "formal": True, "mechanical_fixture": False}
        target = self.folder / "never_accept_context_claim"
        with patch.object(module, "context", side_effect=AssertionError("Caller claim must be rejected before any context read")):
            with self.assertRaisesRegex(ValueError, "actual public provenance paths"):
                module._audit_saved_data(None, target, None, identity,
                                         formal_paths={"verified_public_lifecycle": True})
            with self.assertRaisesRegex(ValueError, "resolve its own"):
                module._audit_saved_data(self.data, target, self.engine, identity,
                    formal_paths={"primary_run": "absent", "code_root": "absent", "data_cache": "absent"})
        self.assertFalse(target.exists())

    def test_27_formal_worker_itself_checks_actual_public_lifecycle(self):
        with tempfile.TemporaryDirectory(prefix="rejected_formal_", dir=None) as folder:
            code, run, cache, out, parent_claim, data = fake_primary_files(Path(folder))
            identity = {**self.identity, "formal": True, "mechanical_fixture": False}
            with patch.object(module, "_engine", side_effect=AssertionError("No model engine allowed")), \
                 patch.object(module, "supplemental_budget", side_effect=AssertionError("No new clock allowed")):
                # No patch of public_primary: this worker must inspect the
                # actual fake files and reject their missing public origin.
                with self.assertRaisesRegex(ValueError, "public-synthetic lifecycle"):
                    module._audit_saved_data(None, out, None, identity,
                        formal_paths={"primary_run": run, "code_root": code, "data_cache": cache})
            self.assertFalse(out.exists())

    def test_28_public_saved_wrapper_delegates_paths_not_a_preverified_claim(self):
        with tempfile.TemporaryDirectory(prefix="saved_delegation_", dir=None) as folder:
            code, run, cache, out, parent_claim, data = fake_primary_files(Path(folder))
            out.mkdir()
            identity = {**self.identity, "formal": True, "mechanical_fixture": False}
            module.save(out / "identity.json", identity)
            with patch.object(module, "_audit_saved_data", return_value="delegation-only-test") as worker, \
                 patch.object(module, "context", side_effect=AssertionError("Context belongs inside the worker")), \
                 patch.object(module, "_engine", side_effect=AssertionError("No model engine allowed")), \
                 patch.object(module, "supplemental_budget", side_effect=AssertionError("No new clock allowed")):
                self.assertEqual(module.audit_saved(run, code, cache, out), "delegation-only-test")
                worker.assert_called_once_with(None, out.resolve(), None, identity,
                    formal_paths={"primary_run": run, "code_root": code, "data_cache": cache})
            self.assertEqual(module.read(out / "saved_prediction_audit.json")["status"], "running")

    def test_29_saved_context_can_verify_expired_clock_without_resetting_it(self):
        with tempfile.TemporaryDirectory(prefix="expired_saved_only_", dir=None) as folder:
            code, run, cache, out, parent, data = fake_primary_files(Path(folder))
            with patch.object(module, "public_primary", return_value=parent), patch.object(module, "load_prepared", return_value=data):
                ctx = module.context(run, code, cache, out, require_time=False)
            clock = {"origin": "public_reproduction", "hours": 1., "started_unix": time.time() - 7200,
                     "parent_run_identity_sha256": parent["run_identity_sha256"],
                     "parent_handoff_sha256": parent["public_handoff_sha256"]}
            module.save(out / "public_supplement_budget.json", clock)
            engine = FakeEngine()
            engine.numerics_sha = lambda: module.sha(code / "revision_numerics.py")
            engine.numerics = lambda: {"fake": True}
            engine.environment = lambda: {"fake": True}
            identity = module.make_identity(ctx, engine, self.models,
                {**clock, "deadline_unix": clock["started_unix"] + clock["hours"] * 3600})
            before = (out / "public_supplement_budget.json").read_bytes()
            with patch.object(module, "_deadline", side_effect=AssertionError("Saved audit must not run deadline prediction check")), \
                 patch.object(module, "supplemental_budget", side_effect=AssertionError("Saved audit must not initialize clock")), \
                 patch.object(module, "_engine", side_effect=AssertionError("Saved audit must not import model engine")):
                module._assert_saved_context(identity, ctx)
            self.assertEqual(before, (out / "public_supplement_budget.json").read_bytes())


def main():
    started = time.perf_counter()
    test_sha_at_start = module.sha(__file__)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PortableEvaluationTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"status": "passed" if result.wasSuccessful() else "failed", "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "seconds": time.perf_counter() - started,
              "kind": "CPU fake-engine mechanical checks only, no scientific prediction or native executable",
              "backend_sha256": module.sha(HERE / "portable_evaluation.py"),
              "backend_imported_sha256": module.IMPORTED_SOURCE_SHA256,
              "backend_source_unchanged_during_test": module.sha(HERE / "portable_evaluation.py") == module.IMPORTED_SOURCE_SHA256,
              "test_sha256": module.sha(__file__),
              "test_source_unchanged_during_test": module.sha(__file__) == test_sha_at_start,
              "model_runtime_imported": "torch" in sys.modules,
              "production_entry_enabled": module.PREDICTION_ENTRY_ENABLED,
              "actual_cuda_or_native_mps_parity_tested": False,
              "actual_checkpoint_tensors_loaded": False, "actual_primary_run_read": False}
    (HERE / "portable_evaluation_checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() and report["backend_source_unchanged_during_test"] and report["test_source_unchanged_during_test"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
