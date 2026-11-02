"""Portable evaluator with source-bound release and author-integration guards.

Ordinary public evaluation requires the independently checked release registry.
The separately registered author-checkpoint route uses the original absolute
budget and fixed ten-method software scope. It preserves author provenance
instead of treating an internal run as a newly trained public reproduction.
Model runtime imports occur only on a permitted prediction path. Saved-only
verification never calls a predictor; fake-engine checks are not model results.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

from porphyry_public import (CORE_SHA, METHODS, PARENT, PRIMARY_SHA, REFERENCE,
    guard_output, public_primary, supplemental_budget)
from prepared_data import load_prepared, predictor_observation, project_for_diagnostic

HERE = Path(__file__).resolve().parent
KINDS = ("geovae", "base_vae", "unet", "cgan")
LEVELS = (.1, .2, .3, .5, 1.)
MASKS = (42, 53, 67)
PREDICTION_ENTRY_ENABLED = False
# Compatibility display constant only. Actual entry permission is source-bound
# evidence from release_contract; this constant is never changed to release.
REVIEWED_REFERENCE_SHA256 = "9f596bf83e76e999a936fe119213b872aa2edcf247d3f40ceec9ea074fd3f41b"
IMPORTED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def source_inventory():
    """Portable wrapper identity, separate from unchanged training core."""
    require(sha(__file__) == IMPORTED_SOURCE_SHA256, "Portable backend source changed after module import")
    return {name: sha(HERE / name) for name in
            ("portable_evaluation.py", "porphyry_public.py", "prepared_data.py",
             "conversion_reference.json", "author_bundle.py", "release_contract.py",
             "author_reference.py")}


def output_guard(out, code_root, data_cache, primary_run):
    """Generated third-party assets never overlap code, source cache or parent."""
    out = guard_output(out, code_root)
    for root in (data_cache, primary_run, HERE):
        root = Path(root).resolve()
        require(not out.is_relative_to(root) and not root.is_relative_to(out),
                "Supplemental output must not overlap primary, prepared cache or wrapper source")
    return out


def _deadline(deadline):
    require(math.isfinite(float(deadline)), "Invalid supplemental deadline")
    if time.time() >= deadline:
        raise TimeoutError("Immutable public supplemental experimental budget exhausted")


def _validate_case_inventory(data):
    cases = data["conditions"]
    keys = [(case["case_id"], case["mask_seed"]) for case in cases]
    require(len(keys) == len(set(keys)) == 15, "Exactly fifteen unique conditions are required")
    require({case["mask_seed"] for case in cases} == set(MASKS) and
            {case["cu_threshold_percent"] for case in cases} == set(LEVELS),
            "Fixed grade and mask coverage differs")
    require(all(sum(case["cu_threshold_percent"] == level for case in cases) == 3
                for level in LEVELS), "Every grade requires all three masks")
    for case in cases:
        require(isinstance(case["case_id"], str) and Path(case["case_id"]).name == case["case_id"]
                and case["case_id"] not in {".", ".."}, "Unsafe case identifier")
        target, known, hidden, obs = case["target"], case["known"], case["hidden"], case["obs"]
        require(target.shape == known.shape == hidden.shape and obs.shape == (2, *target.shape)
                and known.dtype == hidden.dtype == np.bool_ and not np.any(known & hidden)
                and np.isin(target, [0, 1]).all() and np.isfinite(obs).all()
                and np.array_equal(obs[1], known) and np.array_equal(obs[0], target * known),
                "Case target, masks or sparse observation roles differ")
    return len(keys) * len(METHODS)


def _prediction_permission(identity, engine, *, runtime_paths=None):
    """Formal prediction has no bypass, including through the private loop."""
    require(identity.get("format") == "porphyry-public-evaluation-v1", "Wrong portable evaluation format")
    if identity.get("formal") is True:
        from release_contract import scope_for, registration, release_evidence
        if scope_for(identity) is not None:
            require(isinstance(runtime_paths, dict) and set(runtime_paths) == {
                "primary_run", "code_root", "data_cache", "out", "author_bundle", "registration"},
                "Actual subset requires explicit authoritative runtime paths")
            ctx = context(runtime_paths["primary_run"], runtime_paths["code_root"], runtime_paths["data_cache"],
                          runtime_paths["out"], author_bundle=runtime_paths["author_bundle"],
                          registration_path=runtime_paths["registration"])
            registration(runtime_paths["registration"], ctx, require_time=True, owned_worker=True)
            _assert_saved_context(identity, ctx, runtime_paths=runtime_paths)
            pipeline, scoring = sys.modules.get("repro_pipeline"), sys.modules.get("repro_metrics")
            require(pipeline is not None and scoring is not None and engine.predict is pipeline._predict
                    and engine.load is pipeline._load_checked_model and engine.metrics is scoring.volume_metrics
                    and engine.fingerprint is model_fingerprint, "Actual subset must use checked production/core functions")
            return ctx
        else:
            require(release_evidence()["prediction_entry_enabled"], "Portable full prediction remains disabled pending release evidence")
    else:
        require(identity.get("formal") is False and identity.get("mechanical_fixture") is True
                and getattr(engine, "mechanical_fixture", False) is True,
                "Only explicitly nonformal fake-engine mechanical tests are enabled")


def run(*args, **kwargs):
    """Fail before path reads/writes or lifecycle initialization, without override."""
    from release_contract import release_evidence
    if not release_evidence()["prediction_entry_enabled"]:
        raise RuntimeError("Portable prediction is disabled until real ten-method numerical parity and release review")
    return _run_after_release(*args, **kwargs)


def _run_after_release(primary_run, code_root, data_cache, out, *, budget_hours=6,
                       author_bundle=None, registration_path=None):
    """Guarded prediction entry; direct calls apply the same provenance checks.

    Independent public reproductions use their own immutable supplemental clock.
    The explicit registered author-checkpoint route instead retains the original
    absolute deadline, shared attempt ledger and fixed ten-method scope. It never
    creates a new author budget or relabels internal training as public training.
    """
    from release_contract import registration, release_evidence
    if registration_path is None:
        require(release_evidence()["prediction_entry_enabled"], "Portable evaluator release is disabled pending actual parity")
    else:
        require(author_bundle is not None, "Actual subset requires an explicit author checkpoint bundle")
    ctx = context(primary_run, code_root, data_cache, out, require_time=False,
                  author_bundle=author_bundle, registration_path=registration_path)
    out = ctx["out"]
    runtime_paths = {"primary_run": primary_run, "code_root": code_root, "data_cache": data_cache,
                     "out": out, "author_bundle": author_bundle, "registration": registration_path}
    if registration_path is not None:
        from author_bundle import START, HOURS, DEADLINE
        from porphyry_public import frozen_json
        reg = registration(registration_path, ctx, require_time=True, owned_worker=True)
        budget = {"origin": "author_original_budget", "started_unix": START, "hours": HOURS,
                  "deadline_unix": DEADLINE, "parent_budget_sha256": ctx["parent"]["parent_budget_sha256"],
                  "registration_sha256": sha(registration_path)}
        frozen_json(out / "public_supplement_budget.json", budget)
    else:
        require(author_bundle is None, "Normal public lifecycle currently requires independently trained public parent")
        budget = supplemental_budget(out, ctx["parent"], budget_hours)
    deadline = budget["deadline_unix"]
    try:
        with deadline_guard(deadline, out):
            engine = _engine(ctx["code_root"])
            numerical_proof = engine.configure(out, phase="public_porphyry_01")
            if registration_path is not None:
                from release_contract import _checked_numerical_proof, _compare_environment
                ref_identity = read(ctx["primary_run"] / "public_supplement/porphyry_01/identity.json")
                require(engine.numerics() == ref_identity["numerical_policy"], "Actual/reference numerical policy differs")
                _checked_numerical_proof(numerical_proof, engine.numerics(), "public_porphyry_01")
                comparison = _compare_environment(engine.environment(), ref_identity["environment"], engine.numerics())
                save(out / "runtime_environment.json", {"numerical_policy": engine.numerics(),
                    "environment": engine.environment(), "configuration_proof": numerical_proof, "comparison": comparison})
            require(engine.source_hash() == CORE_SHA, "Training core source differs")
            require({kind: sha(path) for kind, path in ctx["checkpoints"].items()} == ctx["checkpoint_sha256"],
                    "Checkpoint bytes changed before checked loading")
            config = ctx["config"]
            device = engine.device(str(config.get("device", "cuda")))
            if registration_path is not None:
                require(str(device).startswith("cuda"), "Actual integration requires CUDA; no fallback")
            # The unchanged loader checks each model's own signature. In
            # particular cGAN uses its extended signature, not just CORE_SHA.
            models = {kind: engine.load(kind, str(path), PRIMARY_SHA, config, 11, device)[0]
                      for kind, path in ctx["checkpoints"].items()}
            require({kind: sha(path) for kind, path in ctx["checkpoints"].items()} == ctx["checkpoint_sha256"],
                    "Checkpoint bytes changed during loading")
            identity = make_identity(ctx, engine, models, budget)
            if registration_path is not None:
                require(identity["model_backbone_baseline_sha256"] == ref_identity["model_backbone_baseline_sha256"],
                        "Loaded model states differ from the original reference")
            with native_temporary_files(out):
                _loop(ctx["data"], out, identity, config, ctx["threshold_record"]["thresholds"],
                      engine, models, device, ctx["training_image"], deadline,
                      runtime_paths=runtime_paths if registration_path is not None else None)
    except Exception as error:
        # Per-case failures already have their own journal/history. Preserve
        # setup/global errors too, without erasing any former failure.
        path = out / "run_failure_history.json"
        history = read(path) if path.is_file() else []
        history.append({"error_type": type(error).__name__, "message": str(error), "time_unix": time.time()})
        save(path, history)
        raise
    return audit_saved(primary_run, code_root, data_cache, out, author_bundle=author_bundle,
                       registration_path=registration_path)


def context(primary_run, code_root, data_cache, out, *, require_time=False,
            author_bundle=None, registration_path=None):
    """Bind real public prerequisites without model loading or predictions.

    The default branch verifies an independent public primary lifecycle. The
    author branch additionally requires the source-bound checkpoint descriptor
    and explicit integration registration, preserving its separate provenance.
    Neither branch initializes a budget or accepts a fake production context.
    Hashing a checkpoint does not load its tensors.
    """
    primary_run, code_root, data_cache = (Path(path).resolve()
                                        for path in (primary_run, code_root, data_cache))
    out = output_guard(out, code_root, data_cache, primary_run)
    if author_bundle is None:
        require(registration_path is None, "Subset registration requires an explicit author bundle")
        parent = public_primary(primary_run, code_root, require_time=require_time)
    else:
        from author_bundle import verify_bundle
        parent = verify_bundle(author_bundle, primary_run, code_root, require_time=require_time)
        require(registration_path is not None, "Author import requires an explicit integration registration")
    data = load_prepared(data_cache)
    _validate_case_inventory(data)
    config = read(primary_run / "tuning/selected_config.json")
    threshold_record = read(primary_run / "thresholds.json")
    metadata = read(primary_run / "test/seed11/evaluation.json")
    intent_path = primary_run / "test/seed11/evaluation_intent.json"
    intent = read(intent_path)
    checkpoints = {kind: primary_run / "models" / kind / "seed11/best.pt" for kind in KINDS}
    hashes = {kind: sha(path) for kind, path in checkpoints.items()}
    require(hashes == metadata.get("checkpoints") == intent.get("checkpoints"),
            "Current seed11 weights differ from primary test metadata/intent")
    require(metadata.get("threshold_selection_sha256") == sha(primary_run / "thresholds.json")
            and intent.get("config_sha256") == stable(config)
            and intent.get("manifest_sha256") == PRIMARY_SHA
            and intent.get("source_sha256") == CORE_SHA,
            "Primary evaluation intent differs from frozen protocol/configuration")
    manifest_path = code_root / "protocol/manifest.json"
    manifest = read(manifest_path)
    entries = manifest.get("samples", [])
    train = [row for row in entries if row.get("split") == "train"]
    require(bool(train), "Primary manifest has no training split")
    ti_id = str(config.get("mps_training_image_id") or train[0]["sample_id"])
    ti_rows = [row for row in train if row.get("sample_id") == ti_id]
    require(len(ti_rows) == 1, "The MPS training image must belong uniquely to the primary training split")
    ti_row = ti_rows[0]
    ti_path = inside(manifest_path.parent, ti_row["path"])
    require(sha(ti_path) == ti_row["sha256"], "Primary training image file changed")
    # Match ProtocolDataset.get_case exactly, including the float32 array hash.
    ti = np.asarray(np.load(ti_path, allow_pickle=False), dtype=np.float32)
    require(ti.shape == (32, 32, 32) and np.isin(ti, [0, 1]).all(), "Primary TI is not binary32cubed")
    ti_array_sha = checked_training_image(ti_id, ti, metadata, intent)
    evidence_files = ("run_identity.json", "revision_budget.json",
                      "tuning/selected_config.json", "tuning/selection.json", "thresholds.json",
                      "test/seed11/evaluation_intent.json")
    evidence_files += (("public_handoff.json",) if author_bundle is None else
                       ("revision_status.json", "final_integrity/report.json"))
    prerequisite = {relative: sha(inside(primary_run, relative)) for relative in evidence_files}
    prerequisite.update({f"{split}/seed{seed}/evaluation.json": sha(primary_run / split / f"seed{seed}/evaluation.json")
                         for split in ("validation", "test") for seed in (11, 23, 37)})
    result = {"parent": parent, "data": data, "config": config,
            "threshold_record": threshold_record, "checkpoints": checkpoints,
            "checkpoint_sha256": hashes, "primary_prerequisite_sha256": prerequisite,
            "training_image": ti, "training_image_id": ti_id,
            "training_image_file_sha256": ti_row["sha256"],
            "training_image_array_sha256": ti_array_sha,
            "primary_run": primary_run, "code_root": code_root,
            "data_cache": data_cache, "out": out}
    if registration_path is not None:
        from release_contract import registration, SCOPE
        result["integration_registration_sha256"] = sha(registration_path)
        result["execution_scope"] = SCOPE.copy()
        result["integration_registration"] = registration(registration_path, result, require_time=require_time)
    return result


def make_identity(ctx, engine, models, budget):
    """Future runtime identity, still gated from formal prediction."""
    imported = ctx["parent"].get("verified_author_bundle") is True
    require(imported or ctx["parent"].get("verified_public_lifecycle") is True, "Verified primary lifecycle required")
    if imported:
        from author_bundle import START, HOURS, DEADLINE
        require(budget == {"origin": "author_original_budget", "started_unix": START, "hours": HOURS,
            "deadline_unix": DEADLINE, "parent_budget_sha256": ctx["parent"]["parent_budget_sha256"],
            "registration_sha256": ctx["integration_registration_sha256"]}, "Author integration clock differs")
    else:
        require(budget.get("origin") == "public_reproduction" and
            budget.get("parent_run_identity_sha256") == ctx["parent"]["run_identity_sha256"] and
            budget.get("parent_handoff_sha256") == ctx["parent"]["public_handoff_sha256"],
                "Supplemental clock is not bound to the verified public primary")
    data = ctx["data"]
    result = {"format": "porphyry-public-evaluation-v1", "formal": True,
            "origin": "imported_author_synthetic_checkpoints" if imported else "public_reproduction", "primary_training_manifest_sha256": PRIMARY_SHA,
            "external_manifest_sha256": data["manifest_sha256"],
            "preparation_identity_sha256": data["preparation_identity_sha256"],
            "wrapper_source_sha256": source_inventory(), "source_sha256": CORE_SHA,
            "config_sha256": stable(ctx["config"]),
            "threshold_record_sha256": ctx["parent"]["thresholds_sha256"],
            "method_thresholds": ctx["threshold_record"]["thresholds"],
            "primary_prerequisite_sha256": ctx["primary_prerequisite_sha256"],
            "public_parent_evidence": ctx["parent"], "checkpoint_sha256": ctx["checkpoint_sha256"],
            "numerics_source_sha256": engine.numerics_sha(), "numerical_policy": engine.numerics(),
            "environment": engine.environment(), "training_image_id": ctx["training_image_id"],
            "training_image_file_sha256": ctx["training_image_file_sha256"],
            "training_image_array_sha256": ctx["training_image_array_sha256"],
            "parent_id": PARENT, "independent_external_parents": 1, "methods": list(METHODS),
            "model_seed": 11, "distance_unit": "voxel", "external_source_coordinate_units_verified": False,
            "levels": list(LEVELS), "mask_seeds": list(MASKS), "budget": budget,
            "license": "CC BY-NC-SA 4.0", "source_commit": data["preparation_identity"]["source_commit"],
            "upstream_source_sha256": data["preparation_identity"]["source_sha256"],
            "model_backbone_baseline_sha256": {kind: engine.fingerprint(model) for kind, model in models.items()},
            "native_io_policy": "Native scratch and cwd stay inside the independent local supplement output.",
            "publication_boundary": "Do not bundle upstream/derived arrays or geometry in the code release; separate NC-SA attribution applies."}
    if imported:
        result.update(execution_scope=ctx["execution_scope"],
                      integration_registration_sha256=ctx["integration_registration_sha256"],
                      public_training_lifecycle_claimed=False, scientific_full150_execution_claimed=False)
    return result


def _engine(code_root):
    """Load only runtime modules belonging to an explicitly selected code root."""
    root = Path(code_root).resolve()
    names = ("repro_pipeline", "repro_metrics", "revision_numerics")
    for name in names:
        loaded = sys.modules.get(name)
        if loaded is not None:
            require(Path(loaded.__file__).resolve().is_relative_to(root),
                    "A runtime module from a different code root is already loaded")
    sys.path.insert(0, str(root))
    modules = [importlib.import_module(name) for name in names]
    require(all(Path(module.__file__).resolve().is_relative_to(root) for module in modules),
            "Imported runtime module is outside the explicit public code root")
    pipeline, scoring, numerical = modules
    return SimpleNamespace(load=pipeline._load_checked_model, predict=pipeline._predict,
        save_array=pipeline._save_npy_atomic, environment=lambda: pipeline.environment_info(),
        device=pipeline.select_device, source_hash=pipeline.source_hash,
        metrics=scoring.volume_metrics, configure=numerical.configure_evaluation,
        numerics=numerical.metadata, numerics_sha=numerical.helper_sha256,
        fingerprint=model_fingerprint)


def _metrics_engine(code_root):
    """Saved verification imports the CPU metric module, never the predictor."""
    root = Path(code_root).resolve()
    loaded = sys.modules.get("repro_metrics")
    if loaded is not None:
        require(Path(loaded.__file__).resolve().is_relative_to(root), "Metrics from another code root are loaded")
    sys.path.insert(0, str(root))
    scoring = importlib.import_module("repro_metrics")
    require(Path(scoring.__file__).resolve().is_relative_to(root), "Metrics import escaped code root")
    return SimpleNamespace(metrics=scoring.volume_metrics)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(root, relative):
    path = (Path(root) / relative).resolve()
    require(path.is_relative_to(Path(root).resolve()), "Artifact path escapes its declared root")
    return path


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def checked_training_image(ti_id, training_image, main_metadata, main_intent):
    """Bind the unchanged selection rule to the actual saved primary TI."""
    require(ti_id == main_metadata.get("mps_training_image_id"), "MPS TI differs from the saved primary test")
    require(ti_id == main_intent.get("mps_training_image_id"), "MPS TI differs from the primary evaluation intent")
    actual = array_hash(training_image)
    for key in ("mps_training_image_sha256", "mps_training_image_array_sha256"):
        if main_intent.get(key) is not None:
            require(actual == main_intent[key], "MPS TI array hash differs from the primary intent")
    return actual


def model_fingerprint(model):
    """Values, buffers, gradient flags and training modes, not RNG state."""
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    h.update(stable([(name, value.requires_grad) for name, value in model.named_parameters()]).encode())
    h.update(stable([(name, value.training) for name, value in model.named_modules()]).encode())
    return h.hexdigest()


def metrics(engine, raw, projected, case, threshold):
    common = dict(threshold=threshold, hidden_mask=case["hidden"])
    result = engine.metrics(raw, case["target"], case["known"], **common)
    result.update({"projected_" + key: value for key, value in engine.metrics(projected, case["target"], case["known"], **common).items()})
    known, labels = case["known"], case["obs"][0]
    result.update(input_mse=float(np.mean((raw[known] - labels[known]) ** 2)),
        input_violation_count_0p5=int(np.count_nonzero((raw[known] >= .5) != labels[known])),
        input_error_0p5=float(np.mean((raw[known] >= .5) != labels[known])),
        observed_positive_labels=int(labels[known].sum()), observed_negative_labels=int(known.sum() - labels[known].sum()))
    return result


def _csv(path, rows):
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with Path(path).with_suffix(".tmp").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    Path(path).with_suffix(".tmp").replace(path)


def _same_metrics(expected, actual):
    for key, value in actual.items():
        before = expected.get(key)
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            require(before is not None and np.isclose(float(before), value, rtol=2e-6, atol=2e-7), "Stored metric differs: " + key)
        else:
            require(before == value, "Stored undefined/status metric differs: " + key)


def _saved(out, case, method, case_sha, engine, identity):
    folder = out / "cases" / case["case_id"] / f"mask{case['mask_seed']}" / method
    journal = folder / "journal.json"
    if not journal.is_file():
        return None
    value = read(journal)
    require(value.get("case_identity_sha256") == case_sha and value.get("identity_sha256") == stable(identity), "Case identity changed")
    if value.get("status") != "complete":
        return None
    record = value["record"]
    expected = {"case_id": case["case_id"], "parent_id": PARENT, "mask_seed": case["mask_seed"],
                "cu_threshold_percent": case["cu_threshold_percent"], "method": method, "model_seed": 11,
                "observation_identity_sha256": case["observation_identity_sha256"],
                "threshold_used": float(identity["method_thresholds"][method]),
                "primary_probability": "raw", "projection_is_diagnostic_only": True,
                "case_seed": int(hashlib.sha256(f"{case['mask_seed']}:{case['case_id']}:{method}".encode()).hexdigest()[:8], 16)}
    require(all(record.get(key) == expected_value for key, expected_value in expected.items()), "Saved row changed frozen case/threshold settings")
    require(set(value["artifacts"]) == {"raw.npy", "projected.npy", "adaptation.json", "backbone.json"}, "Saved artifact inventory incomplete")
    for name, expected in value["artifacts"].items():
        require(sha(inside(folder, name)) == expected, "Saved artifact checksum differs: " + name)
    raw, projected = (np.load(folder / name, allow_pickle=False) for name in ("raw.npy", "projected.npy"))
    require(record["raw_sha256"] == sha(folder / "raw.npy") and record["projected_sha256"] == sha(folder / "projected.npy"), "Saved row probability hashes differ")
    require(raw.shape == projected.shape == case["target"].shape and np.isfinite(raw).all()
            and np.all((raw >= 0) & (raw <= 1)), "Saved probability invalid")
    correct = raw.copy(); correct[case["known"]] = case["obs"][0][case["known"]]
    require(np.array_equal(projected, correct), "Saved projection differs from its observation")
    _same_metrics(record, metrics(engine, raw, projected, case, float(record["threshold_used"])))
    backbone = read(folder / "backbone.json")
    require(backbone["unchanged"] is True and backbone["before"] == backbone["after"] == backbone["loaded_baseline"]
            == identity["model_backbone_baseline_sha256"]["geovae"], "Backbone isolation check failed")
    return record


@contextmanager
def deadline_guard(deadline, out):
    """Formal WSL child owns its process group, including inherited MPS children."""
    require(os.name == "posix", "Formal execution requires WSL/POSIX deadline enforcement; check-only/audit are portable")
    if os.getpgrp() != os.getpid():
        os.setsid()
    own_group = os.getpid()
    require(os.getpgrp() == own_group, "Cannot establish an owned supplemental process group")
    _deadline(deadline)
    def expired(signum, frame):
        try:
            for name, record in (
                ("budget_stop.json", {"status": "budget_exceeded", "deadline_unix": deadline,
                  "identity_preserved": True, "new_training": False}),
                ("status.json", {"status": "budget_exceeded", "deadline_unix": deadline, "expected_rows": 150}),
            ):
                try:
                    save(Path(out) / name, record)
                except Exception:
                    # Stop records are best-effort when the disk is full or
                    # unwritable. Neither write may prevent budget enforcement.
                    pass
        finally:
            # Own session/group only, even if either stop-record write fails.
            os.killpg(own_group, signal.SIGKILL)
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.time()))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _loop(data, out, identity, config, thresholds, engine, models, device, training_image, deadline, *, runtime_paths=None):
    """Shared mechanical loop. Tests use an explicitly nonformal fake engine."""
    actual_context = _prediction_permission(identity, engine, runtime_paths=runtime_paths)
    if actual_context is not None:
        _assert_runtime_arguments(actual_context, data, out, config, thresholds, training_image, deadline, device)
    _validate_case_inventory(data)
    from release_contract import cases_for, scope_for, reserve_attempt, check_attempt_inventory
    selected_cases = cases_for(data, identity)
    expected_rows = len(selected_cases) * len(METHODS)
    integration = scope_for(identity) is not None
    if integration:
        check_attempt_inventory(runtime_paths["registration"], identity)
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    identity_sha = stable(identity)
    if (out / "identity.json").exists():
        require(read(out / "identity.json") == identity, "Output identity changed; do not mix runs")
    save(out / "identity.json", identity)
    rows, failures = [], []
    history = read(out / "failure_history.json") if (out / "failure_history.json").exists() else []
    baseline = {name: engine.fingerprint(model) for name, model in models.items()}
    require(baseline == identity["model_backbone_baseline_sha256"], "Loaded model baseline differs")
    save(out / "status.json", {"status": "running", "identity_sha256": identity_sha, "expected_rows": expected_rows})
    def check_time():
        if time.time() >= deadline:
            _csv(out / "per_case.csv", rows)
            save(out / "status.json", {"status": "budget_exceeded", "completed_rows": len(rows), "expected_rows": expected_rows, "identity_sha256": identity_sha})
        _deadline(deadline)
    for case in selected_cases:
        check_time()
        for method in METHODS:
            check_time()
            row_started = time.perf_counter()
            seed = int(hashlib.sha256(f"{case['mask_seed']}:{case['case_id']}:{method}".encode()).hexdigest()[:8], 16)
            threshold = float(thresholds[method])
            case_sha = stable({"identity": identity_sha, "observation": case["observation_identity_sha256"], "method": method, "seed": seed, "threshold": threshold})
            folder = out / "cases" / case["case_id"] / f"mask{case['mask_seed']}" / method
            try:
                resumed = _saved(out, case, method, case_sha, engine, identity)
                if resumed is not None:
                    rows.append(resumed)
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                before = engine.fingerprint(models["geovae"])
                if str(device).startswith("cuda"):
                    import torch
                    torch.cuda.synchronize()
                if integration:
                    attempt_record = reserve_attempt(runtime_paths["registration"], identity, method, seed)
                started = time.perf_counter()
                prediction_started_unix = time.time()
                observation_before = array_hash(case["obs"])
                raw, adaptation = engine.predict(method, predictor_observation(case), models, config, device, training_image, seed)
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                prediction_ended_unix = time.time()
                check_time()
                after = engine.fingerprint(models["geovae"])
                require(array_hash(case["obs"]) == observation_before, "Saved sparse observation was mutated")
                require(stable(identity) == identity_sha, "Frozen identity/configuration mutated")
                save(folder / "backbone.json", {"unchanged": before == after == baseline["geovae"], "before": before, "after": after, "loaded_baseline": baseline["geovae"]})
                require(before == after == baseline["geovae"], "GeoVAE backbone was contaminated by adaptation")
                if identity["formal"] and method in {"geovae_latent", "geovae_joint"}:
                    require(adaptation.get("mode") == ("latent" if method == "geovae_latent" else "joint")
                            and adaptation.get("seed") == seed and adaptation.get("projection_applied") is False,
                            "Adaptation metadata does not describe the frozen unprojected procedure")
                require(raw.shape == case["target"].shape and np.isfinite(raw).all() and np.all((raw >= 0) & (raw <= 1)), "Invalid predictor probability")
                projected = project_for_diagnostic(raw, case)
                engine.save_array(folder / "raw.npy", raw); engine.save_array(folder / "projected.npy", projected)
                save(folder / "adaptation.json", adaptation)
                record = {"case_id": case["case_id"], "parent_id": PARENT, "mask_seed": case["mask_seed"],
                    "cu_threshold_percent": case["cu_threshold_percent"], "method": method, "model_seed": 11, "case_seed": seed,
                    "observation_identity_sha256": case["observation_identity_sha256"], "threshold_used": threshold,
                    "primary_probability": "raw", "projection_is_diagnostic_only": True, "prediction_seconds": seconds,
                    "row_seconds_before_metrics_and_journal": time.perf_counter() - row_started,
                    "raw_sha256": sha(folder / "raw.npy"), "projected_sha256": sha(folder / "projected.npy"),
                    **metrics(engine, raw, projected, case, threshold)}
                if integration:
                    record.update(attempt_started_unix=attempt_record["started_unix"],
                                  prediction_started_unix=prediction_started_unix,
                                  prediction_ended_unix=prediction_ended_unix,
                                  input_unchanged=True)
                save(folder / "journal.json", {"status": "complete", "identity_sha256": identity_sha,
                    "case_identity_sha256": case_sha, "record": record,
                    "artifacts": {name: sha(folder / name) for name in ("raw.npy", "projected.npy", "adaptation.json", "backbone.json")}})
                rows.append(record)
            except Exception as error:
                failure = {"case_id": case["case_id"], "mask_seed": case["mask_seed"], "method": method,
                           "error_type": type(error).__name__, "message": str(error)}
                failures.append(failure); history.append(failure)
                save(out / "failures.json", failures); save(out / "failure_history.json", history)
                if not folder.exists():
                    folder.mkdir(parents=True)
                save(folder / "failure.json", failure)
                if isinstance(error, TimeoutError) or "contaminated" in str(error):
                    _csv(out / "per_case.csv", rows)
                    save(out / "status.json", {"status": "budget_exceeded" if isinstance(error, TimeoutError) else "backbone_failure",
                        "completed_rows": len(rows), "expected_rows": expected_rows, "identity_sha256": identity_sha})
                    raise
            if (len(rows) + len(failures)) % 10 == 0:
                print(f"[PROGRESS] porphyry_01: {len(rows)}/{expected_rows} rows saved; failures={len(failures)}", flush=True)
    final = {name: engine.fingerprint(model) for name, model in models.items()}
    require(final == baseline, "A loaded model backbone changed")
    save(out / "model_final.json", {"identity_sha256": identity_sha, "baseline": baseline,
         "final": final, "unchanged": True, "completed_rows": len(rows)})
    _csv(out / "per_case.csv", rows); save(out / "failures.json", failures); save(out / "failure_history.json", history)
    complete = len(rows) == expected_rows and not failures
    save(out / "status.json", {"status": "predictions_complete_pending_saved_audit" if complete else "incomplete",
        "identity_sha256": identity_sha, "completed_rows": len(rows), "expected_rows": expected_rows, "failures": len(failures),
        "independent_external_parents": 1, "primary_test_modified": False, "new_training": False})
    require(complete, "Supplement incomplete; every missing/failed row remains visible")
    return rows


def _assert_runtime_arguments(ctx, data, out, config, thresholds, training_image, deadline, device):
    """Bind values actually passed to the operator, before any reserved call.

    This helper is only a consistency check. Formal provenance is independently
    resolved inside _prediction_permission; passing a dictionary here cannot
    authorize a loop or certify any actual result.
    """
    require(Path(out).resolve() == ctx["out"] and stable(config) == stable(ctx["config"])
            and thresholds == ctx["threshold_record"]["thresholds"]
            and deadline == ctx["parent"]["parent_deadline_unix"]
            and str(device).startswith("cuda"), "Actual operator arguments differ from verified configuration/clock/output")
    require(training_image.dtype == ctx["training_image"].dtype
            and np.array_equal(training_image, ctx["training_image"]), "Actual operator training image differs")
    require(data.get("manifest_sha256") == ctx["data"]["manifest_sha256"]
            and data.get("preparation_identity_sha256") == ctx["data"]["preparation_identity_sha256"],
            "Actual operator prepared identity differs")
    require(len(data["conditions"]) == len(ctx["data"]["conditions"]) == 15, "Actual operator condition coverage differs")
    for actual, expected in zip(data["conditions"], ctx["data"]["conditions"]):
        require(all(actual.get(key) == expected.get(key) for key in
                    ("case_id", "mask_seed", "cu_threshold_percent", "observation_identity_sha256"))
                and all(actual[key].dtype == expected[key].dtype and np.array_equal(actual[key], expected[key])
                        for key in ("target", "obs", "known", "hidden")), "Actual operator case values differ from verified preparation")


@contextmanager
def native_temporary_files(out):
    """Use only this supplemental process's local scratch; restore cwd/env."""
    folder = Path(out).resolve() / "native_tmp"
    folder.mkdir(parents=True, exist_ok=True)
    previous_cwd, previous_temp = Path.cwd(), tempfile.tempdir
    previous_environment = {key: os.environ.get(key) for key in ("TMPDIR", "TMP", "TEMP")}
    tempfile.tempdir = str(folder)
    for key in previous_environment:
        os.environ[key] = str(folder)
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        tempfile.tempdir = previous_temp
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _assert_saved_context(identity, ctx, *, runtime_paths=None):
    """No tensor loading: compare current parent evidence and actual file hashes."""
    from release_contract import scope_for, SCOPE
    integration = scope_for(identity) is not None
    require(identity.get("format") == "porphyry-public-evaluation-v1" and identity.get("formal") is True
            and identity.get("origin") == ("imported_author_synthetic_checkpoints" if integration else "public_reproduction"),
            "Not a matching formal public/imported-author supplemental identity")
    require(identity.get("external_manifest_sha256") == ctx["data"]["manifest_sha256"]
            and identity.get("preparation_identity_sha256") == ctx["data"]["preparation_identity_sha256"]
            and identity.get("wrapper_source_sha256") == source_inventory(), "Portable prepared data or wrapper source changed")
    require(identity.get("source_sha256") == CORE_SHA
            and identity.get("primary_training_manifest_sha256") == PRIMARY_SHA
            and identity.get("config_sha256") == stable(ctx["config"])
            and identity.get("threshold_record_sha256") == ctx["parent"]["thresholds_sha256"]
            and identity.get("method_thresholds") == ctx["threshold_record"]["thresholds"]
            and identity.get("checkpoint_sha256") == ctx["checkpoint_sha256"]
            and identity.get("primary_prerequisite_sha256") == ctx["primary_prerequisite_sha256"]
            and identity.get("public_parent_evidence") == ctx["parent"], "Public primary evidence or actual weights changed")
    require(identity.get("training_image_id") == ctx["training_image_id"]
            and identity.get("training_image_file_sha256") == ctx["training_image_file_sha256"]
            and identity.get("training_image_array_sha256") == ctx["training_image_array_sha256"]
            and identity.get("numerics_source_sha256") == sha(ctx["code_root"] / "revision_numerics.py"),
            "TI or evaluation numerical source differs")
    require(identity.get("methods") == list(METHODS) and identity.get("levels") == list(LEVELS)
            and identity.get("mask_seeds") == list(MASKS) and identity.get("model_seed") == 11
            and identity.get("independent_external_parents") == 1, "Saved scientific condition coverage differs")
    budget_path = ctx["out"] / "public_supplement_budget.json"
    budget = read(budget_path)
    if integration:
        from author_bundle import START, HOURS, DEADLINE
        require(ctx.get("execution_scope") == SCOPE
                and identity.get("integration_registration_sha256") == ctx.get("integration_registration_sha256"),
                "Saved software subset registration differs")
        require(budget == {"origin": "author_original_budget", "started_unix": START, "hours": HOURS,
            "deadline_unix": DEADLINE, "parent_budget_sha256": ctx["parent"]["parent_budget_sha256"],
            "registration_sha256": ctx["integration_registration_sha256"]}, "Original author deadline was reset")
        derived_budget = budget
        from release_contract import _checked_numerical_proof, _compare_environment
        ref_identity = read(ctx["primary_run"] / "public_supplement/porphyry_01/identity.json")
        runtime = read(ctx["out"] / "runtime_environment.json")
        require(identity.get("numerical_policy") == ref_identity["numerical_policy"] == runtime.get("numerical_policy")
                and identity.get("environment") == runtime.get("environment")
                and identity.get("model_backbone_baseline_sha256") == ref_identity["model_backbone_baseline_sha256"],
                "Saved runtime numerical policy or original model baseline differs")
        _checked_numerical_proof(runtime["configuration_proof"], identity["numerical_policy"], "public_porphyry_01")
        require(runtime.get("comparison") == _compare_environment(identity["environment"], ref_identity["environment"],
                                                                  identity["numerical_policy"]), "Saved runtime proof differs")
    else:
        require(budget.get("origin") == "public_reproduction"
                and budget.get("parent_run_identity_sha256") == ctx["parent"]["run_identity_sha256"]
                and budget.get("parent_handoff_sha256") == ctx["parent"]["public_handoff_sha256"]
                and math.isfinite(float(budget["started_unix"]))
                and math.isfinite(float(budget["hours"])) and 0 < budget["hours"] <= 72,
                "Supplemental immutable clock differs from verified public parent")
        derived_budget = {**budget, "deadline_unix": budget["started_unix"] + budget["hours"] * 3600}
    require(identity.get("budget") == derived_budget, "Saved supplemental deadline was modified or reset")


def _audit_saved_data(data, out, engine, identity, *, formal_paths=None):
    """Recompute all150 saved rows, no model/runtime imports or predictions.

    The formal branch resolves and verifies its own current public context and
    CPU metric source from explicit paths. It accepts neither caller-supplied
    data/engine nor a ``verified=True`` context dictionary as provenance. The
    nonformal branch requires an explicit fake fixture and cannot create formal
    success. Saved-only verification can finish after the prediction deadline.
    """
    out = Path(out).resolve()
    require(identity.get("format") == "porphyry-public-evaluation-v1", "Wrong saved evaluation format")
    if identity.get("formal") is True:
        require(isinstance(formal_paths, dict) and set(formal_paths) in ({"primary_run", "code_root", "data_cache"},
                {"primary_run", "code_root", "data_cache", "author_bundle", "registration"}),
                "Formal saved audit requires actual public provenance paths, not a preverified context claim")
        require(data is None and engine is None,
                "Formal saved audit must resolve its own verified data and CPU metrics, not caller fixtures")
        # Re-run the real guard inside this worker. A missing caller check or a
        # manufactured context dictionary can never certify formal success.
        # require_time=False deliberately permits saved-only checks after the
        # immutable prediction deadline; no clock is initialized here.
        ctx = context(formal_paths["primary_run"], formal_paths["code_root"],
                      formal_paths["data_cache"], out, require_time=False,
                      author_bundle=formal_paths.get("author_bundle"), registration_path=formal_paths.get("registration"))
        _assert_saved_context(identity, ctx)
        data, engine = ctx["data"], _metrics_engine(ctx["code_root"])
    else:
        require(identity.get("formal") is False and identity.get("mechanical_fixture") is True
                and getattr(engine, "mechanical_fixture", False) is True and formal_paths is None,
                "Saved fixture must be explicitly nonformal, without a formal provenance claim")
    require(read(out / "identity.json") == identity, "Saved identity changed")
    _validate_case_inventory(data)
    from release_contract import cases_for, scope_for
    selected_cases = cases_for(data, identity)
    integration = scope_for(identity) is not None
    expected_rows = len(selected_cases) * len(METHODS)
    if integration:
        from release_contract import check_attempt_inventory
        attempts = check_attempt_inventory(formal_paths["registration"], identity)
        require(len(attempts) == 10, "Actual subset must have exactly ten reserved attempts")
        attempt_records = {p.stem: read(p) for p in attempts}
        final = read(out / "model_final.json")
        require(final.get("identity_sha256") == stable(identity) and final.get("unchanged") is True
                and final.get("baseline") == final.get("final") == identity["model_backbone_baseline_sha256"]
                and final.get("completed_rows") == 10, "Four-model final isolation proof incomplete")
    status = read(out / "status.json")
    require(status.get("identity_sha256") == stable(identity)
            and status.get("status") in {"predictions_complete_pending_saved_audit", "complete_external_synthetic_descriptive",
                "fake_fixture_complete_not_formal", "saved_audit_failed", "complete_actual_integration_subset"}
            and not read(out / "failures.json"), "Prediction run is incomplete or has unresolved failures")
    rows, calibration = [], []
    for case in selected_cases:
        for method in METHODS:
            seed = int(hashlib.sha256(f"{case['mask_seed']}:{case['case_id']}:{method}".encode()).hexdigest()[:8], 16)
            journal = out / "cases" / case["case_id"] / f"mask{case['mask_seed']}" / method / "journal.json"
            threshold = float(identity["method_thresholds"][method])
            case_sha = stable({"identity": stable(identity), "observation": case["observation_identity_sha256"],
                               "method": method, "seed": seed, "threshold": threshold})
            row = _saved(out, case, method, case_sha, engine, identity)
            require(row is not None and row["case_seed"] == seed and row["method"] == method, "Missing/unexpected saved row")
            if integration:
                from author_bundle import START, DEADLINE
                attempt = attempt_records[method]
                require(attempt["seed"] == seed and row.get("attempt_started_unix") == attempt["started_unix"]
                        and START <= attempt["started_unix"] <= row.get("prediction_started_unix", -1)
                        <= row.get("prediction_ended_unix", -1) <= DEADLINE
                        and row.get("input_unchanged") is True, "Actual prediction attempt/time/input proof differs")
            rows.append(row)
            raw = np.load(journal.parent / "raw.npy", allow_pickle=False)
            p, y = raw[~case["known"]], case["target"][~case["known"]]
            bins = np.minimum((p * 10).astype(int), 9)
            for index in range(10):
                selected = bins == index
                calibration.append({"case_id": case["case_id"], "mask_seed": case["mask_seed"], "method": method,
                    "bin": index, "voxel_count": int(selected.sum()),
                    "mean_probability": float(p[selected].mean()) if selected.any() else None,
                    "occupied_fraction": float(y[selected].mean()) if selected.any() else None})
    require(len(rows) == expected_rows, "All150 saved method/condition rows are required")
    with (out / "per_case.csv").open(encoding="utf-8", newline="") as stream:
        saved_rows = list(csv.DictReader(stream))
    require(len(saved_rows) == expected_rows, "Saved CSV has missing or duplicate rows")
    indexed = {(r["case_id"], int(r["mask_seed"]), r["method"]): r for r in saved_rows}
    require(len(indexed) == expected_rows, "Saved CSV row identity is repeated")
    for row in rows:
        saved_row = indexed[(row["case_id"], row["mask_seed"], row["method"])]
        require(all(saved_row.get(key) == ("" if value is None else str(value)) for key, value in row.items()),
                "Saved CSV differs from verified journals")
    _csv(out / "recomputed_per_case.csv", rows)
    _csv(out / "reliability.csv", calibration)
    summaries = []
    for level in sorted({case["cu_threshold_percent"] for case in selected_cases}):
        for method in METHODS:
            subset = [row for row in rows if row["cu_threshold_percent"] == level and row["method"] == method]
            mask_count = 1 if integration else 3
            require(len(subset) == mask_count, "Summary mask coverage differs from immutable execution scope")
            summary = {"cu_threshold_percent": level, "method": method, "mask_count": mask_count, "independent_parents": 1,
                       "invalid_surface_distance_count": sum(not row["surface_distance_valid"] for row in subset)}
            for metric in ("blind_iou", "blind_dice", "observed_error", "hidden_error", "input_mse", "brier_blind",
                           "volume_absolute_relative_error", "hd", "hd95", "chamfer", "prediction_seconds"):
                values = [row[metric] for row in subset if row[metric] is not None]
                summary[metric + "_n"] = len(values)
                summary[metric + "_mean"] = float(np.mean(values)) if values else None
                summary[metric + "_mask_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            summaries.append(summary)
    _csv(out / "descriptive_summary.csv", summaries)
    history = read(out / "failure_history.json")
    save(out / "saved_prediction_audit.json", {
        "status": "passed_saved_probability_recomputation", "identity_sha256": stable(identity),
        "rows": expected_rows, "actual_integration_subset": integration, "independent_external_parents": 1, "formal": identity["formal"],
        "no_prediction_or_weight_loading": True, "historical_failure_count": len(history),
        "primary_threshold_record_sha256": identity.get("threshold_record_sha256"),
        "calibration_scope": "Fixed ten-bin reliability from raw blind probabilities; levels/masks depend on one synthetic parent.",
        "scope": "One external synthetic parent; no independent-parent CI or field-validation claim.",
        "files": {name: sha(out / name) for name in ("recomputed_per_case.csv", "reliability.csv", "descriptive_summary.csv")}})
    save(out / "status.json", {"status": ("complete_actual_integration_subset" if integration else "complete_external_synthetic_descriptive") if identity["formal"] else "fake_fixture_complete_not_formal",
        "identity_sha256": stable(identity), "expected_rows": expected_rows, "completed_rows": expected_rows,
        "independent_external_parents": 1, "new_training": False, "primary_test_modified": False,
        "saved_prediction_audit_sha256": sha(out / "saved_prediction_audit.json")})
    return out / "saved_prediction_audit.json"


def audit_saved(primary_run, code_root, data_cache, out, *, author_bundle=None, registration_path=None):
    """Formal saved-only check, with current actual weight hashes and no model load."""
    out = output_guard(out, code_root, data_cache, primary_run)
    require((out / "identity.json").is_file(), "No saved supplemental identity")
    identity = read(out / "identity.json")
    save(out / "saved_prediction_audit.json", {"status": "running", "inference_started": False})
    try:
        paths = {"primary_run": primary_run, "code_root": code_root, "data_cache": data_cache}
        if author_bundle is not None or registration_path is not None:
            paths.update(author_bundle=author_bundle, registration=registration_path)
        return _audit_saved_data(None, out, None, identity, formal_paths=paths)
    except Exception as error:
        history_path = out / "audit_failure_history.json"
        history = read(history_path) if history_path.is_file() else []
        history.append({"error_type": type(error).__name__, "message": str(error), "time_unix": time.time()})
        save(history_path, history)
        save(out / "saved_prediction_audit.json", {"status": "failed", "error_type": type(error).__name__,
                                                "message": str(error), "inference_started": False})
        save(out / "status.json", {"status": "saved_audit_failed", "identity_sha256": stable(identity),
                                   "expected_rows": 150, "error_type": type(error).__name__})
        raise


def release_status():
    """No boolean supplied by a caller can enable prediction in this staging."""
    from release_contract import release_evidence
    return release_evidence()


if __name__ == "__main__":
    # Import/help/status never reads parent data, allocates model tensors, starts
    # a clock or invokes external native executables.
    print(json.dumps(release_status(), sort_keys=True))
    raise SystemExit(2)
