"""CPU-only audit of original author reference predictions through portable scoring.

The original 150 records remain author records. Importing and recomputing them
does not generate public predictions, deserialize weights, or start a clock.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np

from porphyry_public import (CORE_SHA, PRIMARY_SHA, METHODS, PARENT, array_sha,
                            frozen_json, inside, read, require, sha, stable)
from author_bundle import verify_bundle, START, HOURS, DEADLINE, NUMERICS_SHA
from prepared_data import load_prepared

IMPORTED_SOURCE_SHA256 = sha(__file__)
MANIFEST_SHA = "24097d0b7f1fd5462d0cb8060f6a8c9b1e4609b2e660283cabdeae64636c1f85"
PREREG_SHA = "7810883290f432581f6f33666e8f4724ec6f2f6288f26f4c1b1caa0c5f78d651"
SOURCE_SHA = "deb4604875deb534b45463dbcbb81aced97c5453f9ae807077c7f9a802abbf38"
RUNNER_SHA = "9f596bf83e76e999a936fe119213b872aa2edcf247d3f40ceec9ea074fd3f41b"
PREPARED_PROOF_SHA = "cd7bacfd13715baeab1126239f43647d78b152d7b49f82f1e62090e43e5db119"
LEVELS = (0.1, 0.2, 0.3, 0.5, 1.0)
MASKS = (42, 53, 67)
KINDS = ("geovae", "base_vae", "unet", "cgan")
ARTIFACTS = {"raw.npy", "projected.npy", "adaptation.json", "backbone.json"}
SUMMARY_METRICS = ("blind_iou", "blind_dice", "observed_error", "hidden_error", "input_mse",
                   "brier_blind", "volume_absolute_relative_error", "hd", "hd95", "chamfer", "prediction_seconds")


def _json(path, expected=None):
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    require(expected is None or digest == expected, "Source JSON bytes differ")
    return json.loads(payload.decode("utf-8")), digest


def _array(path, expected):
    payload = Path(path).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == expected, "Saved prediction bytes differ")
    return np.load(io.BytesIO(payload), allow_pickle=False)


def _strict_policy(value):
    declared = {"version": "esi-evaluation-numerics-v1", "cublas_workspace_config": ":4096:8",
                "deterministic_algorithms": True, "deterministic_warn_only": False,
                "cudnn_benchmark": False, "cudnn_deterministic": True,
                "scope": "evaluation_only_training_recipe_unchanged"}
    require(isinstance(value, dict) and set(value) == {"declared", "actual"}
            and value["declared"] == declared, "Original numerical policy differs")
    required = ("cublas_workspace_config", "deterministic_algorithms", "deterministic_warn_only",
                "cudnn_benchmark", "cudnn_deterministic")
    actual = value["actual"]
    require(isinstance(actual, dict) and set(actual) == set(required) | {
        "cuda_matmul_allow_tf32", "cudnn_allow_tf32", "float32_matmul_precision"}, "Numerical backend fields differ")
    require(all(actual[k] == declared[k] and type(actual[k]) is type(declared[k]) for k in required),
            "Strict deterministic flags differ")
    require(type(actual["cuda_matmul_allow_tf32"]) is bool and type(actual["cudnn_allow_tf32"]) is bool
            and actual["float32_matmul_precision"] in {"highest", "high", "medium"}, "Invalid numeric metadata")


def _condition_map(manifest, public):
    """Reconstruct old identity without relabelling it as public preparation."""
    require(manifest.get("parent_id") == PARENT and manifest.get("source_kind") == "external synthetic"
            and manifest.get("axis_order") == "zyx" and manifest.get("grid_shape") == [32, 32, 32], "Reference domain differs")
    samples = {row["case_id"]: row for row in manifest["samples"]}
    masks = {row["mask_seed"]: row for row in manifest["masks"]}
    conditions = {(row["case_id"], row["mask_seed"]): row for row in manifest["conditions"]}
    require(len(samples) == len(manifest["samples"]) == 5
            and {r["cu_threshold_percent"] for r in samples.values()} == set(LEVELS), "Five fixed levels required")
    require(len(masks) == len(manifest["masks"]) == 3 and set(masks) == set(MASKS), "Three fixed masks required")
    wanted = {(name, seed) for name in samples for seed in MASKS}
    require(set(conditions) == wanted and len(manifest["conditions"]) == 15, "Original condition coverage differs")
    public_cases = {(c["case_id"], c["mask_seed"]): c for c in public["conditions"]}
    require(set(public_cases) == wanted and len(public["conditions"]) == 15, "Portable condition coverage differs")
    cases, mapping = [], []
    for key in sorted(wanted, key=lambda k: (samples[k[0]]["cu_threshold_percent"], k[1])):
        sample, mask, original = samples[key[0]], masks[key[1]], conditions[key]
        case = dict(public_cases[key])
        require(sample["parent_id"] == original["parent_id"] == PARENT
                and case["parent_id"] == PARENT and case["cu_threshold_percent"] == sample["cu_threshold_percent"], "Parent or Cu identity differs")
        require(case["target"].dtype == np.uint8 and case["target"].shape == (32, 32, 32)
                and case["known"].dtype == case["hidden"].dtype == np.bool_
                and case["known"].shape == case["hidden"].shape == case["target"].shape
                and case["obs"].dtype == np.float32 and case["obs"].shape == (2, 32, 32, 32)
                and np.isin(case["target"], [0, 1]).all() and int(case["known"].sum()) == 96
                and int(case["hidden"].sum()) == 48 and not np.any(case["known"] & case["hidden"])
                and np.array_equal(case["obs"][1], case["known"])
                and np.array_equal(case["obs"][0], case["target"] * case["known"]), "Portable array roles differ")
        roles = {"target": (sample, "target_array_sha256"), "known": (mask, "observation_mask_array_sha256"),
                 "hidden": (mask, "hidden_mask_array_sha256"), "obs": (original, "observation_array_sha256")}
        role_hashes = {role: array_sha(case[role]) for role in roles}
        require(all(role_hashes[role] == row[field] for role, (row, field) in roles.items()), "Original and portable role arrays differ")
        hidden_labels_sha = array_sha((case["target"] * case["hidden"]).astype(np.uint8))
        require(hidden_labels_sha == original["hidden_labels_array_sha256"], "Hidden scoring-label identity differs")
        old_observation = stable({"sample": sample, "mask": mask, "condition": original})
        mapping.append({"case_id": key[0], "mask_seed": key[1],
                        "original_observation_identity_sha256": old_observation,
                        "portable_observation_identity_sha256": case["observation_identity_sha256"],
                        "role_array_sha256": role_hashes, "hidden_labels_array_sha256": hidden_labels_sha})
        case["observation_identity_sha256"] = old_observation
        cases.append(case)
    return cases, mapping


def verify_reference_context(bundle_path, primary_run, code_root, data_cache, reference_run, external_manifest):
    """Read-only provenance and150row inventory, without probability scoring.

    Loads the verified portable preparation arrays, hashes original prediction
    files, and reads journals. It never deserializes original probabilities,
    imports Torch or creates outputs. Registration can use this exact API.
    """
    require("torch" not in sys.modules, "Use a fresh CPU process for saved-only reference import")
    require(sha(__file__) == IMPORTED_SOURCE_SHA256, "Author bridge source changed after import")
    bundle = verify_bundle(bundle_path, primary_run, code_root, require_time=False)
    require(bundle.get("verified_author_bundle") is True, "Author bundle was not verified")
    evidence = bundle["evidence"]
    primary, code, reference = map(lambda p: Path(p).resolve(), (primary_run, code_root, reference_run))
    require(reference == primary / "public_supplement/porphyry_01", "Reference must be the original author supplemental run")
    manifest_path = Path(external_manifest).resolve()
    manifest, manifest_sha = _json(manifest_path, MANIFEST_SHA)
    require(manifest.get("source_sha256") == SOURCE_SHA and manifest.get("preregistration_sha256") == PREREG_SHA,
            "External source or preregistration differs")
    prereg, prereg_sha = _json(inside(manifest_path.parent, manifest["preregistration_file"]), PREREG_SHA)
    require(prereg.get("thresholds_cu_percent") == list(LEVELS) and prereg.get("before_any_model_call") is True,
            "Fixed preregistered external levels differ")
    proof, proof_sha = _json(manifest_path.parent / "prepared_verification.json", PREPARED_PROOF_SHA)
    require(proof.get("passed") is True and proof.get("manifest_sha256") == manifest_sha
            and proof.get("preregistration_sha256") == prereg_sha and proof.get("source_sha256") == SOURCE_SHA,
            "Original prepared verification differs")
    public = load_prepared(data_cache)
    cases, mapping = _condition_map(manifest, public)
    identity, identity_sha = _json(reference / "identity.json")
    status, status_sha = _json(reference / "status.json")
    audit, audit_sha = _json(reference / "saved_prediction_audit.json")
    failures, failures_sha = _json(reference / "failures.json")
    history, history_sha = _json(reference / "failure_history.json")
    require(identity.get("format") == "porphyry-01-frozen-evaluation-v1" and identity.get("formal") is True
            and identity.get("runner_sha256") == RUNNER_SHA and identity.get("source_sha256") == CORE_SHA
            and identity.get("primary_training_manifest_sha256") == PRIMARY_SHA, "Original reference source differs")
    require(identity.get("external_manifest_sha256") == manifest_sha and identity.get("external_preregistration_sha256") == prereg_sha
            and identity.get("prepared_verification_sha256") == proof_sha, "Original preparation provenance differs")
    require(status.get("status") == "complete_external_synthetic_descriptive" and status.get("expected_rows") == 150
            and status.get("completed_rows") == 150 and status.get("identity_sha256") == stable(identity)
            and status.get("saved_prediction_audit_sha256") == audit_sha and not failures, "Original reference is incomplete")
    require(audit.get("status") == "passed_saved_probability_recomputation" and audit.get("rows") == 150
            and audit.get("formal") is True and audit.get("identity_sha256") == stable(identity)
            and audit.get("metrics_source_sha256") == sha(code / "repro_metrics.py"), "Original saved-probability audit differs")
    require(isinstance(history, list), "Original failure history is missing or invalid")
    names = {"run_identity.json", "revision_status.json", "final_integrity/report.json", "tuning/selected_config.json",
             "tuning/selection.json", "thresholds.json", "revision_budget.json"}
    names.update(f"test/seed{s}/evaluation.json" for s in (11, 23, 37))
    require(set(identity.get("primary_prerequisite_sha256", {})) == names
            and all(identity["primary_prerequisite_sha256"][n] == evidence["original_artifact_sha256"][n] for n in names),
            "Original main evidence differs from author bundle")
    config = read(primary / "tuning/selected_config.json")
    thresholds = read(primary / "thresholds.json")
    require(identity.get("config_sha256") == evidence["config_sha256"] == stable(config)
            and identity.get("threshold_record_sha256") == evidence["thresholds_sha256"]
            and identity.get("method_thresholds") == thresholds["thresholds"]
            and identity.get("checkpoint_sha256") == evidence["checkpoint_sha256"], "Original frozen settings or weights differ")
    require(identity.get("model_seed") == 11 and identity.get("methods") == list(METHODS)
            and identity.get("levels") == list(LEVELS) and identity.get("mask_seeds") == list(MASKS)
            and identity.get("parent_id") == PARENT and identity.get("independent_external_parents") == 1
            and identity.get("distance_unit") == "voxel", "Original evaluation scope differs")
    require(identity.get("budget_started_unix") == START and identity.get("deadline_unix") == DEADLINE,
            "Original author budget identity differs; saved import does not create a new clock")
    require(set(identity.get("model_backbone_baseline_sha256", {})) == set(KINDS), "Original baseline coverage differs")
    for key in ("training_image_id", "training_image_file_sha256", "training_image_array_sha256"):
        require(identity.get(key) == evidence[key], "Original training-only TI differs")
    require(identity.get("main_seed11_intent_sha256") == evidence["original_artifact_sha256"]["test/seed11/evaluation_intent.json"]
            and identity.get("numerics_source_sha256") == NUMERICS_SHA, "Original intent or numerical helper differs")
    _strict_policy(identity.get("numerical_policy"))
    if "evaluation_numerical_policy" in identity.get("environment", {}):
        require(identity["environment"]["evaluation_numerical_policy"] == identity["numerical_policy"], "Original environment policy differs")
    anchors = {"identity.json": identity_sha, "status.json": status_sha, "saved_prediction_audit.json": audit_sha,
               "failures.json": failures_sha, "failure_history.json": history_sha, "per_case.csv": sha(reference / "per_case.csv")}
    inventory = _reference_inventory(reference, identity, cases)
    return {"verified_author_reference": True, "origin": "author_reference_import",
            "reference": reference, "identity": identity, "cases": cases, "mapping": mapping,
            "bundle": bundle, "public": public, "anchors": anchors, "history": history,
            "manifest_sha256": manifest_sha, "preregistration_sha256": prereg_sha,
            "external_source_sha256": SOURCE_SHA, "code": code, "reference_inventory": inventory,
            "original_identity_sha256": identity_sha, "probabilities_scored": 0, "new_predictions": 0}


def _reference_inventory(reference, identity, cases):
    """Exact original coverage and CSV binding, without loading probability arrays."""
    reference = Path(reference)
    expected = {(case["case_id"], case["mask_seed"], method) for case in cases for method in METHODS}
    require(len(cases) == 15 and len(expected) == 150, "All150 original conditions required")
    inventory, records = [], {}
    for case in cases:
        for method in METHODS:
            folder = reference / "cases" / case["case_id"] / f"mask{case['mask_seed']}" / method
            journal, digest = _json(folder / "journal.json")
            threshold = float(identity["method_thresholds"][method])
            seed = int(hashlib.sha256(f"{case['mask_seed']}:{case['case_id']}:{method}".encode()).hexdigest()[:8], 16)
            case_sha = stable({"identity": stable(identity), "observation": case["observation_identity_sha256"],
                               "method": method, "seed": seed, "threshold": threshold})
            require(journal.get("status") == "complete" and journal.get("identity_sha256") == stable(identity)
                    and journal.get("case_identity_sha256") == case_sha, "Original journal identity differs")
            record = journal["record"]
            fields = {"case_id": case["case_id"], "parent_id": PARENT, "mask_seed": case["mask_seed"],
                      "cu_threshold_percent": case["cu_threshold_percent"], "method": method, "model_seed": 11,
                      "observation_identity_sha256": case["observation_identity_sha256"], "threshold_used": threshold,
                      "primary_probability": "raw", "projection_is_diagnostic_only": True, "case_seed": seed}
            require(all(record.get(k) == v for k, v in fields.items()), "Original row scope differs")
            require(set(journal["artifacts"]) == ARTIFACTS
                    and all(sha(inside(folder, name)) == digest for name, digest in journal["artifacts"].items()),
                    "Original artifact inventory or bytes differ")
            require(record.get("raw_sha256") == journal["artifacts"]["raw.npy"]
                    and record.get("projected_sha256") == journal["artifacts"]["projected.npy"], "Original row probability binding differs")
            records[(case["case_id"], case["mask_seed"], method)] = record
            inventory.append({"case_id": case["case_id"], "mask_seed": case["mask_seed"], "method": method,
                              "journal_sha256": digest, "artifacts": journal["artifacts"],
                              "original_case_identity_sha256": case_sha, "adaptation_retained_by_hash": True})
    with (reference / "per_case.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream); rows = list(reader); headers = reader.fieldnames
    keys = [(row["case_id"], int(row["mask_seed"]), row["method"]) for row in rows]
    require(len(rows) == len(set(keys)) == 150 and set(keys) == expected, "Original CSV row set differs")
    require(headers is not None and len(headers) == len(set(headers))
            and set(headers) == {key for record in records.values() for key in record}, "Original CSV header set differs")
    for key, row in zip(keys, rows):
        record = records[key]
        require(all(row[field] == ("" if record.get(field) is None else str(record[field])) for field in headers),
                "Original CSV fields differ from journals")
    return inventory


def _same_metrics(saved, recomputed):
    for key, value in recomputed.items():
        require(key in saved, "Original metric is missing: " + key)
        before = saved[key]
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            require(before is not None and math.isfinite(float(value)) and math.isfinite(float(before))
                    and math.isclose(float(before), float(value), rel_tol=2e-6, abs_tol=2e-7), "Metric differs: " + key)
        else:
            require(before == value, "Metric status or undefined value differs: " + key)


def _recompute_rows(reference, identity, cases, engine):
    """Pure saved-data mechanics; never emits a provenance success status."""
    from portable_evaluation import metrics
    reference = Path(reference)
    expected = {(case["case_id"], case["mask_seed"], method) for case in cases for method in METHODS}
    require(len(cases) == 15 and len(expected) == 150, "All150 original rows required")
    rows, calibration, inventory = [], [], []
    for case in cases:
        for method in METHODS:
            folder = reference / "cases" / case["case_id"] / f"mask{case['mask_seed']}" / method
            journal, journal_sha = _json(folder / "journal.json")
            threshold = float(identity["method_thresholds"][method])
            seed = int(hashlib.sha256(f"{case['mask_seed']}:{case['case_id']}:{method}".encode()).hexdigest()[:8], 16)
            case_sha = stable({"identity": stable(identity), "observation": case["observation_identity_sha256"],
                               "method": method, "seed": seed, "threshold": threshold})
            require(journal.get("status") == "complete" and journal.get("identity_sha256") == stable(identity)
                    and journal.get("case_identity_sha256") == case_sha, "Original journal identity differs")
            record = journal["record"]
            fields = {"case_id": case["case_id"], "parent_id": PARENT, "mask_seed": case["mask_seed"],
                      "cu_threshold_percent": case["cu_threshold_percent"], "method": method, "model_seed": 11,
                      "observation_identity_sha256": case["observation_identity_sha256"], "threshold_used": threshold,
                      "primary_probability": "raw", "projection_is_diagnostic_only": True, "case_seed": seed}
            require(all(record.get(k) == v for k, v in fields.items()), "Original row scope differs")
            require(set(journal["artifacts"]) == ARTIFACTS, "Original artifact roles differ")
            raw, projected = (_array(folder / name, journal["artifacts"][name]) for name in ("raw.npy", "projected.npy"))
            adaptation, _ = _json(folder / "adaptation.json", journal["artifacts"]["adaptation.json"])
            backbone, _ = _json(folder / "backbone.json", journal["artifacts"]["backbone.json"])
            require(record.get("raw_sha256") == journal["artifacts"]["raw.npy"]
                    and record.get("projected_sha256") == journal["artifacts"]["projected.npy"], "Original row artifact hash differs")
            require(raw.dtype == projected.dtype == np.float32 and raw.shape == projected.shape == (32, 32, 32)
                    and np.isfinite(raw).all() and np.all((raw >= 0) & (raw <= 1)), "Original raw probability invalid")
            correct = raw.copy(); correct[case["known"]] = case["obs"][0][case["known"]]
            require(np.array_equal(correct, projected), "Original projection altered blind values")
            require(backbone.get("unchanged") is True and backbone.get("before") == backbone.get("after")
                    == backbone.get("loaded_baseline") == identity["model_backbone_baseline_sha256"]["geovae"], "Original backbone isolation differs")
            scores = metrics(engine, raw, projected, case, threshold)
            _same_metrics(record, scores)
            require(math.isfinite(float(record["prediction_seconds"])) and float(record["prediction_seconds"]) >= 0,
                    "Original inference duration invalid")
            # Preserve original metadata; recomputed values replace only metric fields.
            rows.append({**record, **scores})
            probabilities, labels = raw[~case["known"]], case["target"][~case["known"]]
            bins = np.minimum((probabilities * 10).astype(int), 9)
            for index in range(10):
                selected = bins == index
                calibration.append({"case_id": case["case_id"], "mask_seed": case["mask_seed"], "method": method,
                    "bin": index, "voxel_count": int(selected.sum()),
                    "mean_probability": float(probabilities[selected].mean()) if selected.any() else None,
                    "occupied_fraction": float(labels[selected].mean()) if selected.any() else None})
            inventory.append({"case_id": case["case_id"], "mask_seed": case["mask_seed"], "method": method,
                              "journal_sha256": journal_sha, "artifacts": journal["artifacts"],
                              "original_case_identity_sha256": case_sha, "adaptation_retained_by_hash": True})
    csv_payload = (reference / "per_case.csv").read_bytes()
    reader = csv.DictReader(io.StringIO(csv_payload.decode("utf-8"), newline=""))
    saved = list(reader)
    keys = [(r["case_id"], int(r["mask_seed"]), r["method"]) for r in saved]
    require(len(saved) == len(set(keys)) == 150 and set(keys) == expected, "Original CSV row set differs")
    by_key = dict(zip(keys, saved))
    for case in cases:
        for method in METHODS:
            # Exact original CSV-to-journal relation, before tolerant metric recompute.
            folder = reference / "cases" / case["case_id"] / f"mask{case['mask_seed']}" / method
            bound = next(r for r in inventory if (r["case_id"], r["mask_seed"], r["method"]) == (case["case_id"], case["mask_seed"], method))
            original, _ = _json(folder / "journal.json", bound["journal_sha256"])
            record = original["record"]
            row = by_key[(case["case_id"], case["mask_seed"], method)]
            require(set(record) <= set(row) and all(row[k] == ("" if v is None else str(v)) for k, v in record.items()),
                    "Original CSV fields differ from bound journals")
    summaries = []
    for level in LEVELS:
        for method in METHODS:
            subset = [r for r in rows if r["cu_threshold_percent"] == level and r["method"] == method]
            require(len(subset) == 3 and {r["mask_seed"] for r in subset} == set(MASKS), "Dependent-mask summary coverage differs")
            row = {"cu_threshold_percent": level, "method": method, "mask_count": 3, "independent_parents": 1,
                   "invalid_surface_distance_count": sum(not r["surface_distance_valid"] for r in subset)}
            for name in SUMMARY_METRICS:
                values = [r[name] for r in subset if r[name] is not None]
                row[name + "_n"] = len(values)
                row[name + "_mean"] = float(np.mean(values)) if values else None
                row[name + "_mask_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            summaries.append(row)
    return {"rows": rows, "calibration": calibration, "summaries": summaries, "inventory": inventory,
            "per_case_csv_sha256": hashlib.sha256(csv_payload).hexdigest()}


def _output_guard(out, protected):
    out = Path(out).resolve()
    require(not any(p.casefold() in {".git", "revision_article", "sn-article", "reply", "supplement"} for p in out.parts),
            "Derived audit must stay outside publication and source-control directories")
    for source in protected:
        source = Path(source).resolve()
        require(not out.is_relative_to(source) and not source.is_relative_to(out), "Derived output overlaps a read-only input")
    return out


def import_reference(bundle_path, primary_run, code_root, data_cache, reference_run, external_manifest, out):
    """Verify all150 original author rows, with zero new public predictions.

    No injected context/engine, limit, alternative seed, or deadline exists.
    Expired original budgets permit this saved-only CPU operation.
    """
    from portable_evaluation import _csv, _metrics_engine, save, source_inventory
    require("torch" not in sys.modules, "Run saved-only reference verification in a fresh CPU process")
    protected = (primary_run, code_root, data_cache, reference_run, Path(external_manifest).parent,
                 Path(__file__).parent, bundle_path)
    out = _output_guard(out, protected)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "author_reference_import.json"
    save(report_path, {"status": "verifying_author_reference_import", "origin": "author_reference_import",
                       "public_prediction_rows": 0, "new_predictions": 0})
    try:
        wrapper_sources = source_inventory()
        ctx = verify_reference_context(bundle_path, primary_run, code_root, data_cache, reference_run, external_manifest)
        engine = _metrics_engine(code_root)
        require("torch" not in sys.modules, "CPU scorer imported model runtime")
        result = _recompute_rows(ctx["reference"], ctx["identity"], ctx["cases"], engine)
        require(result["inventory"] == ctx["reference_inventory"], "Original reference artifacts changed during scoring")
        require(result["per_case_csv_sha256"] == ctx["anchors"]["per_case.csv"], "Original CSV changed during verification")
        require(all(sha(ctx["reference"] / name) == digest for name, digest in ctx["anchors"].items()), "Original source records changed during import")
        require(source_inventory() == wrapper_sources, "Portable source changed during saved-only import")
        binding = {"format": "author-reference-import-v1", "origin": "author_reference_import",
                   "original_identity_sha256": ctx["anchors"]["identity.json"], "original_reference_files": ctx["anchors"],
                   "author_bundle_sha256": ctx["bundle"]["bundle_sha256"], "external_manifest_sha256": ctx["manifest_sha256"],
                   "external_preregistration_sha256": ctx["preregistration_sha256"], "external_source_sha256": SOURCE_SHA,
                   "portable_manifest_sha256": ctx["public"]["manifest_sha256"],
                   "portable_preparation_identity_sha256": ctx["public"]["preparation_identity_sha256"],
                   "metrics_source_sha256": sha(Path(code_root) / "repro_metrics.py"),
                   "bridge_source_sha256": sha(__file__), "portable_scorer_source_sha256": sha(Path(__file__).parent / "portable_evaluation.py"),
                   "wrapper_source_sha256": wrapper_sources,
                   "condition_mapping": ctx["mapping"], "artifact_inventory": result["inventory"],
                   "public_prediction_rows": 0, "new_predictions": 0, "models_loaded": 0,
                   "clock_created": False, "original_budget": ctx["bundle"]["evidence"]["original_budget"],
                   "original_failure_history": ctx["history"], "independent_external_parents": 1}
        frozen_json(out / "import_identity.json", binding)
        _csv(out / "recomputed_per_case.csv", result["rows"])
        _csv(out / "reliability.csv", result["calibration"])
        _csv(out / "descriptive_summary.csv", result["summaries"])
        report = {"status": "passed_author_reference_import", "origin": "author_reference_import",
                  "rows": 150, "rows_recomputed": 150, "conditions": 15, "public_prediction_rows": 0, "new_predictions": 0,
                  "wrapper_source_sha256": wrapper_sources,
                  "model_tensors_loaded": False, "clock_created": False, "original_files_modified": False,
                  "public_training_lifecycle_claimed": False, "full_public_prediction_run_claimed": False,
                  "independent_external_parents": 1, "historical_failure_count": len(ctx["history"]),
                  "import_identity_sha256": sha(out / "import_identity.json"),
                  "derived_files": {name: sha(out / name) for name in ("recomputed_per_case.csv", "reliability.csv", "descriptive_summary.csv")},
                  "scope": "Portable CPU scoring of imported author predictions. Five levels and three masks depend on one synthetic parent; no independent-geology CI or new public inference claim."}
        save(report_path, report)
        return report
    except Exception as error:
        history_path = out / "import_failure_history.json"
        history = read(history_path) if history_path.exists() else []
        history.append({"error_type": type(error).__name__, "error": str(error), "new_predictions": 0})
        save(history_path, history)
        save(report_path, {"status": "failed_author_reference_import", "origin": "author_reference_import",
                           "public_prediction_rows": 0, "new_predictions": 0, "error": str(error)})
        raise
