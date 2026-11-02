"""Final, read-only numerical audit of the ESI revision experiment archive.

Only this audit's own reports are written. Confidential real geometry is never
included in the public report, including on failure. A draft audit is explicit
and cannot produce a formal passing report.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np

from experiment_data import ProtocolDataset, make_drill_masks, verify_protocol
from repro_metrics import volume_metrics
from repro_pipeline import ROOT, load_config, save_json, sha256_file, source_hash, stable_hash
from revision_numerics import DECLARED_POLICY, configure_evaluation, helper_sha256 as numerics_sha256
from revision_diagnostics import (
    ACCEPTANCE_THRESHOLDS, SOURCE_FILES as DIAGNOSTIC_SOURCE_FILES, VARIANCES, _diversity, aggregate_real_rows,
    assert_private_real_path, lora_distance_effect, observation_consistency,
    select_parent_cases, threshold_key,
)

ORIGINAL_MANIFEST_SHA256 = "48ec0924b76a1cca26715bb77566646dacb5fe9a3252471f8e4690aae8e01030"
METHODS = ("idw", "rbf", "ik", "mps", "unet", "base_vae", "cgan", "geovae", "geovae_latent", "geovae_joint")
CORE_FILES = ("repro_pipeline.py", "experiment_data.py", "model.py", "repro_cgan.py",
              "repro_baselines.py", "repro_metrics.py", "comparisons/models/unet.py")
RAW_COUNTS = {"mining": 122, "mining_ply": 122, "mining_ply_generate": 15000,
              "mining_ply_generate_pretrain": 1000}
RTOL, ATOL = 2e-6, 2e-7


class AuditError(RuntimeError):
    """Safe machine-readable error codes; do not add confidential identifiers."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise AuditError(code)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _path(root: Path, relative: str) -> Path:
    result = (root / relative).resolve()
    _require(result.is_relative_to(root.resolve()) and result.is_file(), "missing_or_escaping_artifact")
    return result


def _runtime_path(value: str, base: Path) -> Path:
    """Resolve the recorded Windows/WSL representation, without changing identity."""
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", value):
        value = value[5].upper() + ":/" + value[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:[\\/]", value):
        value = "/mnt/" + value[0].lower() + "/" + value[3:].replace("\\", "/")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _number(value: Any) -> float | None:
    if value is None or value in ("", "None", "null"):
        return None
    result = float(value)
    _require(math.isfinite(result), "nonfinite_recorded_scalar")
    return result


def _compare(actual: Any, recorded: Any, code: str = "derived_scalar_differs") -> None:
    if actual is None:
        _require(recorded is None or recorded in ("", "None", "null"), code)
    elif isinstance(actual, (bool, np.bool_)):
        expected = recorded
        if isinstance(expected, str):
            _require(expected in ("True", "False", "true", "false"), code)
            expected = expected.lower() == "true"
        _require(type(expected) is bool and bool(actual) == expected, code)
    elif isinstance(actual, (float, int, np.floating, np.integer)):
        expected = _number(recorded)
        _require(expected is not None and bool(np.isclose(actual, expected, rtol=RTOL, atol=ATOL)), code)
    elif isinstance(actual, list):
        expected = json.loads(recorded) if isinstance(recorded, str) else recorded
        _require(actual == expected, code)
    else:
        _require(actual == recorded, code)


def _compare_record(expected: dict, recorded: dict, keys: tuple | list | None = None) -> None:
    for key in keys or expected:
        _require(key in recorded, "derived_record_field_missing")
        _compare(expected[key], recorded[key])


def _array(path: Path, shape: tuple | None = None, probability: bool = False) -> np.ndarray:
    result = np.load(path, allow_pickle=False)
    if shape is not None:
        _require(result.shape == shape, "array_shape_differs")
    _require(bool(np.isfinite(result).all()), "nonfinite_array")
    if probability:
        _require(bool(((result >= 0) & (result <= 1)).all()), "invalid_probability")
    return result


def _arrays_equal(a: np.ndarray, b: np.ndarray, code: str) -> float:
    _require(a.shape == b.shape, code)
    difference = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.
    _require(bool(np.allclose(a, b, rtol=RTOL, atol=ATOL)), code)
    return difference


def _checked_completion(folder: Path, identity_sha: str) -> dict:
    record = _json(folder / "complete.json")
    _require(record.get("status") == "complete" and record.get("identity_sha256") == identity_sha,
             "case_completion_identity_differs")
    _require(bool(record.get("artifacts")), "case_artifact_inventory_missing")
    for relative, digest in record["artifacts"].items():
        _require(sha256_file(_path(folder, relative)) == digest, "case_artifact_hash_differs")
    return record


def _check_evaluation_policy_metadata(value: Any) -> dict:
    _require(isinstance(value, dict) and value.get("declared") == DECLARED_POLICY,
             "evaluation_numerical_policy_missing_or_changed")
    actual = value.get("actual", {})
    for key in ("cublas_workspace_config", "deterministic_algorithms", "deterministic_warn_only",
                "cudnn_benchmark", "cudnn_deterministic"):
        _require(actual.get(key) == DECLARED_POLICY[key], "evaluation_numerical_flags_differ")
    _require("threads" not in json.dumps(value).lower(), "thread_count_must_not_enter_prediction_cache_identity")
    return actual


def audit_numerical_policy(root: Path, identity: dict, *, required_phases: tuple[str, ...] =
                           ("tune", "validation", "test", "ablation_validation", "ablation_test",
                            "classical_val", "classical_test", "uncertainty_val", "uncertainty_test",
                            "real_cases", "robustness", "final_integrity")) -> dict:
    """V2 evaluation proofs are distinct from unchanged non-bitwise training."""
    _require(identity.get("evaluation_numerical_policy") == DECLARED_POLICY
             and identity.get("evaluation_numerics_source_sha256") == numerics_sha256(),
             "run_evaluation_numerical_policy_differs")
    proofs = list((root / "numerical_policy").glob("*.json"))
    _require(bool(proofs), "evaluation_numerical_phase_proofs_missing")
    evaluation_phases, training_phases, reference_actual = set(), set(), None
    for path in proofs:
        proof = _json(path)
        _require(proof.get("format") == "esi-numerical-policy-proof-v1"
                 and proof.get("helper_sha256") == numerics_sha256()
                 and proof.get("declared") == DECLARED_POLICY
                 and proof.get("training_bitwise_reproducibility_claimed") is False,
                 "numerical_phase_proof_identity_differs")
        _require(isinstance(proof.get("pid"), int) and proof["pid"] > 0 and isinstance(proof.get("phase"), str),
                 "numerical_phase_proof_fields_invalid")
        if proof.get("kind") == "training":
            _require(proof["actual"].get("deterministic_algorithms") is False,
                     "training_numerical_recipe_was_changed")
            training_phases.add(proof["phase"])
        else:
            _require(proof.get("kind") == "evaluation", "numerical_phase_proof_kind_invalid")
            current = _check_evaluation_policy_metadata({"declared": proof["declared"], "actual": proof["actual"]})
            _require(reference_actual is None or reference_actual == current, "evaluation_backends_differ_across_phases")
            reference_actual = current
            if proof.get("workspace_origin") == "configured_before_cuda_initialization":
                _require(proof.get("cuda_initialized_at_configuration") is False,
                         "workspace_configured_after_cuda_initialization")
            evaluation_phases.add(proof["phase"])
    _require(set(required_phases) <= evaluation_phases, "mandatory_evaluation_phase_proofs_missing")
    _require(bool(training_phases), "unchanged_training_numerical_proof_missing")
    return {"helper_sha256": numerics_sha256(), "evaluation_phase_count": len(evaluation_phases),
            "training_phase_count": len(training_phases), "phase_proof_files": len(proofs),
            "declared": DECLARED_POLICY, "actual_evaluation_flags": reference_actual,
            "training_deterministic_algorithms": False, "training_bitwise_reproducibility_claimed": False}


def audit_frozen(root: Path, *, formal: bool = True) -> tuple[dict, Path, dict]:
    identity = _json(root / "run_identity.json")
    manifest = root / "frozen_inputs" / "manifest.json"
    digest = sha256_file(manifest)
    _require(digest == identity.get("manifest_sha256"), "frozen_manifest_hash_differs")
    if formal:
        _require(identity.get("mode") == "formal" and digest == ORIGINAL_MANIFEST_SHA256,
                 "not_the_registered_formal_protocol")
    source_manifest = _runtime_path(identity["manifest"], root)
    _require(source_manifest.is_file() and sha256_file(source_manifest) == digest,
             "original_manifest_missing_or_changed")
    protocol = verify_protocol(manifest, verify_files=True)
    if formal:
        _require(protocol.get("split_counts") == {"train": 3500, "val": 750, "test": 750}
                 and protocol.get("parent_counts") == {"train": 700, "val": 150, "test": 150}
                 and protocol.get("count") == 5000 and protocol.get("grid_size") == [32, 32, 32],
                 "formal_protocol_counts_or_grid_differs")
    expected = identity.get("code_files", {})
    _require(bool(expected), "frozen_source_inventory_missing")
    for relative, digest in expected.items():
        _require(sha256_file(_path(root / "frozen_inputs", relative)) == digest,
                 "frozen_source_hash_differs")
    if formal:
        h = hashlib.sha256()
        for relative in CORE_FILES:
            h.update(relative.encode())
            h.update(sha256_file(_path(root / "frozen_inputs", relative)).encode("ascii"))
        _require(h.hexdigest() == identity.get("source_sha256") == source_hash(),
                 "training_core_source_differs")
    controllers = list((root / "controller_sources").glob("*.py"))
    for path in controllers:
        _require(path.stem == sha256_file(path), "versioned_controller_hash_differs")
    return {"manifest_sha256": sha256_file(manifest), "verified_grids": len(protocol["samples"]),
            "frozen_source_files": len(expected), "versioned_controllers": len(controllers)}, manifest, protocol


def audit_raw(root: Path, identity: dict, *, formal: bool = True) -> dict:
    source_manifest = _runtime_path(identity["manifest"], root)
    inventory = source_manifest.parent / "raw_inventory.jsonl"
    summary = source_manifest.parent / "raw_audit_summary.json"
    _require(sha256_file(inventory) == identity.get("raw_inventory_sha256")
             and sha256_file(summary) == identity.get("raw_audit_sha256"), "raw_inventory_identity_differs")
    data_root = _runtime_path(identity["data_root"], root)
    rows = [_json_line(line) for line in inventory.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen, counts, size = set(), Counter(), 0
    for index, row in enumerate(rows, 1):
        _require(row["path"] not in seen, "duplicate_raw_inventory_path")
        seen.add(row["path"])
        path = _path(data_root, row["path"])
        _require(path.stat().st_size == row["bytes"] and sha256_file(path) == row["sha256"],
                 "raw_source_changed")
        counts[row["source_role"]] += 1
        size += row["bytes"]
        if index % 1000 == 0:
            print(f"[final audit/raw] {index}/{len(rows)} hashes", flush=True)
    _require(dict(counts) == _json(summary)["counts"], "raw_source_counts_differ")
    if formal:
        _require(dict(counts) == RAW_COUNTS and len(rows) == 16244, "formal_raw_cohort_differs")
    # New source files must not be silently omitted from the audit inventory.
    actual = {p.relative_to(data_root).as_posix() for role in counts
              for p in (data_root / role).rglob("*") if p.is_file()}
    _require(actual == seen, "raw_inventory_no_longer_exhaustive")
    return {"files": len(rows), "bytes": size, "counts": dict(counts),
            "inventory_sha256": sha256_file(inventory), "status": "all_hashes_rechecked"}


def _json_line(line: str) -> dict:
    return json.loads(line)


def _metric_check(probability: np.ndarray, case: dict, threshold: float, recorded: dict) -> None:
    result = volume_metrics(probability, case["target"], case["observed_mask"],
                            threshold=threshold, hidden_mask=case["hidden_mask"])
    _compare_record(result, recorded)


def audit_evaluation(folder: Path, protocol: dict, manifest_sha: str, config_sha: str,
                     source_sha: str, thresholds: dict, threshold_sha: str,
                     *, split: str, seed: int, methods: tuple[str, ...], formal: bool = True,
                     checkpoint_hashes: dict | None = None) -> dict:
    meta, intent = _json(folder / "evaluation.json"), _json(folder / "evaluation_intent.json")
    _require(meta.get("complete") is True and not _json(folder / "failures.json"), "evaluation_incomplete")
    _require(meta.get("journal_identity_sha256") == stable_hash(intent), "evaluation_intent_hash_differs")
    for key in ("seed", "split", "mask_seed", "num_holes", "noise_rate", "methods", "project_observed",
                "source_sha256", "config_sha256", "method_thresholds", "threshold_selection_sha256"):
        _require(meta.get(key) == intent.get(key), "evaluation_summary_intent_differs")
    expected_ids = {row["sample_id"] for row in protocol["samples"] if row["split"] == split}
    _require(set(intent["case_ids"]) == expected_ids and len(intent["case_ids"]) == len(expected_ids),
             "evaluation_case_set_differs")
    _require(meta.get("manifest_sha256") == manifest_sha and meta.get("config_sha256") == config_sha
             and meta.get("source_sha256") == source_sha and meta.get("seed") == seed
             and meta.get("split") == split and set(meta.get("methods", [])) == set(methods),
             "evaluation_protocol_identity_differs")
    _require(meta.get("expected_cases") == len(expected_ids), "evaluation_case_count_differs")
    if formal:
        _require(len(expected_ids) == 750 and meta.get("limit") == 0 and meta.get("num_holes") == 8
                 and float(meta.get("noise_rate", -1)) == 0 and meta.get("project_observed") is False,
                 "formal_evaluation_conditions_differ")
        _check_evaluation_policy_metadata(intent.get("environment", {}).get("evaluation_numerical_policy"))
    if split == "test":
        _require(meta.get("threshold_selection_sha256") == threshold_sha
                 and meta.get("threshold_selection_status") == ("validation_frozen" if formal else "draft")
                 and meta.get("method_thresholds") == thresholds, "evaluation_thresholds_not_frozen")
    else:
        _require(meta.get("threshold_selection_sha256") is None, "validation_used_selected_test_thresholds")
    if checkpoint_hashes is not None:
        _require(intent.get("checkpoints") == checkpoint_hashes, "evaluation_checkpoint_hash_differs")
    rows = _csv(folder / "per_case.csv")
    keys = [(row["sample_id"], row["method"]) for row in rows]
    _require(len(keys) == len(set(keys)) and set(keys) == {(sample, method) for sample in expected_ids for method in methods},
             "evaluation_method_case_coverage_differs")
    predicted = {}
    for row in rows:
        _compare(meta["method_thresholds"][row["method"]], row["threshold_used"], "case_threshold_differs")
        prefixes = ["prediction"]
        raw_recorded = bool(row.get("raw_prediction_file") or row.get("raw_prediction_sha256"))
        if raw_recorded:
            prefixes.append("raw_prediction")
        for prefix in prefixes:
            relative, digest = row.get(prefix + "_file"), row.get(prefix + "_sha256")
            _require(bool(relative and digest), "evaluation_prediction_identity_missing")
            path = _path(folder, relative)
            _require(sha256_file(path) == digest, "evaluation_prediction_hash_differs")
            _array(path, tuple(protocol["grid_size"]), probability=True)
        # Unprojected evaluations intentionally store only one probability
        # volume. A second raw copy is optional, but cannot differ if present.
        _require(meta.get("project_observed") is False
                 and (not raw_recorded or row["prediction_sha256"] == row["raw_prediction_sha256"]),
                 "unexpected_observation_projection")
        predicted[(row["sample_id"], row["method"])] = row["prediction_sha256"]
    return {"cases": len(expected_ids), "rows": len(rows), "predictions": predicted,
            "summary_sha256": sha256_file(folder / "evaluation.json"),
            "intent_sha256": sha256_file(folder / "evaluation_intent.json"),
            "per_case_sha256": sha256_file(folder / "per_case.csv")}


def audit_main(root: Path, protocol: dict, config: dict, *, formal: bool = True) -> dict:
    identity = _json(root / "run_identity.json")
    methods = METHODS if formal else tuple(identity["methods"])
    seeds = (11, 23, 37) if formal else tuple(identity["seeds"])
    thresholds_path = root / "thresholds.json"
    thresholds = _json(thresholds_path)
    _require(thresholds.get("status") == ("validation_frozen" if formal else "draft")
             and thresholds.get("manifest_sha256") == identity["manifest_sha256"]
             and thresholds.get("config_sha256") == stable_hash(config)
             and thresholds.get("evaluation_source_sha256") == identity["source_sha256"], "main_threshold_freeze_differs")
    analysis_path = root / "revision_analysis" / "analysis.json"
    analysis = _json(analysis_path)
    _require(analysis.get("status") == ("complete_primary_reanalysis" if formal else "draft_reanalysis")
             and analysis.get("manifest_sha256") == identity["manifest_sha256"]
             and analysis.get("config_sha256") == stable_hash(config)
             and analysis.get("source_sha256") == identity["source_sha256"]
             and set(analysis.get("methods", [])) == set(methods) and analysis.get("seeds") == list(seeds),
             "independent_main_reanalysis_missing_or_changed")
    _require(analysis.get("analysis_script_sha256") == sha256_file(ROOT / "revision_analysis.py"),
             "main_analysis_script_changed")
    provenance = {int(entry["seed"]): entry for entry in analysis["evaluation_provenance"]}
    classical_reference, checked = {}, 0
    model_kinds = {"geovae" if method.startswith("geovae") else method for method in methods
                   if method not in {"idw", "rbf", "ik", "mps"}}
    for seed in seeds:
        checkpoint_hashes = {kind: sha256_file(root / "models" / kind / f"seed{seed}" / "best.pt")
                             for kind in model_kinds}
        for split, directory in (("val", "validation"), ("test", "test")):
            result = audit_evaluation(root / directory / f"seed{seed}", protocol,
                                      identity["manifest_sha256"], stable_hash(config), identity["source_sha256"],
                                      thresholds["thresholds"], sha256_file(thresholds_path), split=split,
                                      seed=seed, methods=methods, formal=formal, checkpoint_hashes=checkpoint_hashes)
            checked += result["rows"]
            if split == "test":
                _compare_record(result, provenance[seed], ["summary_sha256", "intent_sha256", "per_case_sha256"])
                for key, digest in result["predictions"].items():
                    if key[1] in ("idw", "rbf", "ik", "mps"):
                        _require(key not in classical_reference or classical_reference[key] == digest,
                                 "classical_copies_differ_across_model_seeds")
                        classical_reference[key] = digest
    # Repeat the deterministic analysis into this audit's own directory, then
    # compare every publication-facing numeric table. Original artifacts remain
    # read-only, and a stale/tampered summary cannot pass on status alone.
    from revision_analysis import analyse_evaluations
    recomputed = root / "final_integrity" / "recomputed_primary"
    analyse_evaluations(root / "frozen_inputs" / "manifest.json",
                        [root / "test" / f"seed{seed}" for seed in seeds], recomputed,
                        formal=formal, required_methods=methods,
                        replicates=int(analysis["bootstrap"]["replicates"]),
                        bootstrap_seed=int(analysis["bootstrap"]["seed"]),
                        config_path=root / "tuning" / "selected_config.json")
    checked_tables = ("recomputed_per_case.csv", "summary.csv", "model_seed_means.csv",
                      "paired_differences.csv", "reliability.csv", "analysis.json")
    numerical_tables = 0
    for name in checked_tables:
        if not (recomputed / name).is_file() and not (root / "revision_analysis" / name).is_file():
            _require(not formal, "required_formal_publication_table_missing")
            continue
        _require(sha256_file(recomputed / name) == sha256_file(root / "revision_analysis" / name),
                 "main_recomputed_publication_table_differs")
        numerical_tables += 1
    return {"seeds": list(seeds), "methods": len(methods), "verified_case_method_rows": checked,
            "numerically_recomputed_tables": numerical_tables,
            "analysis_sha256": sha256_file(analysis_path), "thresholds_sha256": sha256_file(thresholds_path)}


def audit_uncertainty(root: Path, manifest: Path, protocol: dict, config: dict,
                      *, formal: bool = True) -> dict:
    folder = root / "uncertainty"
    identity = _json(folder / "identity.json")
    digest = stable_hash(identity)
    run_identity = _json(root / "run_identity.json")
    settings = identity["settings"]
    if formal:
        _check_evaluation_policy_metadata(identity.get("evaluation_numerical_policy"))
    _require(identity.get("smoke") is (not formal) and identity.get("manifest_sha256") == sha256_file(manifest)
             and identity.get("config_sha256") == stable_hash(config)
             and identity.get("training_source_sha256") == run_identity["source_sha256"]
             and identity.get("checkpoint_sha256") == sha256_file(root / "models" / "geovae" / "seed11" / "best.pt"),
             "uncertainty_model_or_protocol_identity_differs")
    for relative, expected in identity["source_files"].items():
        _require(sha256_file(_path(ROOT, relative)) == expected, "uncertainty_source_changed")
    _require(set(identity["source_files"]) == set(DIAGNOSTIC_SOURCE_FILES), "uncertainty_source_inventory_differs")
    expected_selection = {split: select_parent_cases(protocol, split, 10 if split == "val" else 30) if formal
                          else select_parent_cases(protocol, split, 1) for split in ("val", "test")}
    _require(identity.get("selection") == expected_selection == _json(folder / "selection.json"),
             "uncertainty_parent_selection_differs")
    _require(settings.get("variances") == list(VARIANCES)
             and settings.get("thresholds") == list(ACCEPTANCE_THRESHOLDS)
             and settings.get("initializations") == [0, 1, 2] and settings.get("primary_threshold") == .001
             and settings.get("projection_applied") is False
             and settings.get("candidate_count") == (100 if formal else 4)
             and settings.get("diagnostic_steps") == (500 if formal else 2), "uncertainty_registered_settings_differ")
    frozen = _json(folder / "val" / "settings_frozen.json")
    _require(frozen.get("status") == "validation_frozen" and frozen.get("identity_sha256") == digest
             and frozen.get("settings") == settings and frozen.get("case_ids") == expected_selection["val"],
             "uncertainty_validation_freeze_differs")
    modes = settings["modes"]
    _require(bool(modes) and len(modes) == len(set(modes)) and set(modes) <= {"joint", "latent"}, "uncertainty_modes_invalid")
    counts = {"cases": 0, "adaptations": 0, "candidates": 0, "zero_accepted_rows": 0,
              "single_accepted_rows": 0, "summary_rows": 0, "max_map_absolute_difference": 0.}
    for split in ("val", "test"):
        ids = expected_selection[split]
        status = _json(folder / split / "status.json")
        _require(status.get("status") == "complete" and status.get("identity_sha256") == digest
                 and status.get("case_count") == len(ids) and not _json(folder / split / "failures.json"),
                 "uncertainty_split_incomplete")
        dataset = ProtocolDataset(str(manifest), split, num_holes=int(config["num_holes"]),
                                  samples_per_hole=int(config["samples_per_hole"]), mask_seed=settings["mask_seed"])
        expected_rows, expected_completions = [], set()
        for sample in ids:
            case = dataset.get_case(sample)
            parent = dataset.metadata[sample]["parent_id"]
            counts["cases"] += 1
            for mode in modes:
                for init in (0, 1, 2):
                    path = folder / split / "cases" / sample / mode / f"init{init}"
                    expected_completions.add((path / "complete.json").resolve())
                    record = _checked_completion(path, digest)
                    _require("lora_state.pt" in record["artifacts"], "uncertainty_adapter_archive_missing")
                    saved = np.load(path / "adaptation.npz", allow_pickle=False)
                    _arrays_equal(saved["target"], case["target"], "uncertainty_reference_differs")
                    _arrays_equal(saved["observation"], case["obs"], "uncertainty_observation_differs")
                    _arrays_equal(saved["hidden_mask"], case["hidden_mask"], "uncertainty_hidden_mask_differs")
                    center = saved["optimized_latent"]
                    _require(center.shape == (1, int(config["latent_dim"])) and np.isfinite(center).all(),
                             "uncertainty_latent_dimension_differs")
                    metadata = _json(path / "adaptation.json")
                    _require(metadata.get("mode") == mode and metadata.get("initialization") == init
                             and metadata.get("projection_applied") is False
                             and metadata.get("observed_count") == int(case["observed_mask"].sum())
                             and metadata.get("latent_dim") == config["latent_dim"]
                             and metadata.get("steps_requested") == settings["diagnostic_steps"],
                             "uncertainty_adaptation_metadata_differs")
                    trace_rows = _csv(path / "optimization_trace.csv")
                    _require(bool(trace_rows), "uncertainty_optimization_trace_missing")
                    _compare(float(config.get("test_time_prior_weight", .01)), metadata["prior_weight"])
                    _compare(float(config.get("test_time_lora_weight", .01)), metadata["lora_weight"])
                    for trace in trace_rows:
                        _compare(.5 * float(trace["latent_norm_squared_sum"]) / config["latent_dim"],
                                 trace["normalized_latent_prior"], "trace_prior_normalization_differs")
                        _compare(metadata["prior_weight"] * float(trace["normalized_latent_prior"]),
                                 trace["weighted_latent_prior"], "trace_prior_weight_differs")
                        _compare(metadata["lora_weight"] * float(trace["lora_parameter_penalty"]),
                                 trace["weighted_lora_penalty"], "trace_lora_weight_differs")
                        _compare(float(trace["observed_bce"]) + float(trace["weighted_latent_prior"]) + float(trace["weighted_lora_penalty"]),
                                 trace["objective"], "trace_objective_components_differ")
                    best_rows = [r for r in trace_rows if int(r["step"]) == metadata["best_step"]]
                    _require(len(best_rows) == 1 and 0 <= metadata["best_step"] <= metadata["steps_run"] <= metadata["steps_requested"],
                             "uncertainty_best_step_invalid")
                    _compare(metadata["best_objective"], best_rows[0]["objective"], "uncertainty_best_objective_differs")
                    _compare(float(np.square(center.astype(np.float64)).sum()), best_rows[0]["latent_norm_squared_sum"],
                             "uncertainty_center_does_not_match_best_trace")
                    _compare_record(observation_consistency(saved["optimized_probability"], case["obs"]), best_rows[0])
                    drift = lora_distance_effect(saved["optimized_probability"], saved["base_same_z_probability"],
                                                  case["target"], case["observed_mask"])
                    stored_drift = _csv(path / "lora_distance_effect.csv")
                    _require(len(drift) == len(stored_drift), "drift_bin_count_differs")
                    for actual, old in zip(drift, stored_drift):
                        _compare_record(actual, old)
                    observed = case["observed_mask"]
                    candidate_count = settings["candidate_count"]
                    row_map = {(float(r["variance"]), float(r["threshold"])): r for r in record["summary_rows"]}
                    _require(len(record["summary_rows"]) == len(row_map) == len(VARIANCES) * len(ACCEPTANCE_THRESHOLDS),
                             "uncertainty_summary_coverage_differs")
                    reference_epsilon = None
                    for variance in VARIANCES:
                        token = threshold_key(variance)
                        archive = np.load(path / f"candidates_var{token}.npz", allow_pickle=False)
                        probabilities, latents = archive["probabilities"], archive["latents"]
                        _require(probabilities.shape == (candidate_count, *protocol["grid_size"])
                                 and np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
                                 and latents.shape == (candidate_count, config["latent_dim"]) and np.isfinite(latents).all(),
                                 "candidate_arrays_invalid")
                        epsilon = (latents.astype(np.float64) - center) / math.sqrt(variance)
                        if reference_epsilon is not None:
                            _require(np.allclose(epsilon, reference_epsilon, rtol=2e-4, atol=2e-4),
                                     "variance_arms_do_not_share_standardized_proposals")
                        reference_epsilon = epsilon
                        candidate_rows = _csv(path / f"candidates_var{token}.csv")
                        _require([int(r["candidate_index"]) for r in candidate_rows] == list(range(candidate_count)),
                                 "candidate_index_coverage_differs")
                        accepted = {tau: [] for tau in ACCEPTANCE_THRESHOLDS}
                        for index, (probability, latent, old) in enumerate(zip(probabilities, latents, candidate_rows)):
                            consistency = observation_consistency(probability, case["obs"])
                            _compare_record(consistency, old)
                            norm = float(np.square(latent.astype(np.float64)).sum())
                            _compare(norm, old["latent_norm_squared_sum"], "candidate_latent_norm_differs")
                            _compare(.5 * norm / config["latent_dim"], old["normalized_latent_prior"], "candidate_prior_normalization_differs")
                            _compare(.01 * norm, old["manuscript_prior_lower_bound"], "candidate_manuscript_lower_bound_differs")
                            stored_accept = json.loads(old["accepted_by_threshold"])
                            for tau in ACCEPTANCE_THRESHOLDS:
                                valid = consistency["observed_mse"] <= tau and consistency["observed_violation_count"] == 0
                                _compare(valid, stored_accept[threshold_key(tau)], "candidate_acceptance_differs")
                                if valid:
                                    accepted[tau].append(index)
                        counts["candidates"] += candidate_count
                        for tau in ACCEPTANCE_THRESHOLDS:
                            indices = accepted[tau]
                            selected = probabilities[indices]
                            n = len(indices)
                            row = row_map[(variance, tau)]
                            _compare_record({"sample_id": sample, "parent_id": parent, "split": split,
                                             "mode": mode, "initialization": init, "variance": variance,
                                             "standard_deviation": math.sqrt(variance), "threshold": tau,
                                             "candidate_count": candidate_count, "accepted_count": n,
                                             "accepted_indices": indices, "acceptance_rate": n/candidate_count,
                                             "variance_defined": n >= 2, **_diversity(selected, ~observed)}, row)
                            positives = int(np.count_nonzero(case["obs"][0][observed] >= .5))
                            negatives = int(observed.sum()) - positives
                            _compare_record({"observed_positive_count": positives,
                                             "observed_negative_count": negatives,
                                             "observed_label_composition": "all_barren" if positives == 0 else "all_occupied" if negatives == 0 else "mixed",
                                             "status": "zero_accepted" if n == 0 else "one_accepted_variance_undefined" if n == 1 else "local_variability_only"}, row)
                            mean_file = path / f"mean_var{token}_tau{threshold_key(tau)}.npy"
                            variance_file = path / f"variance_var{token}_tau{threshold_key(tau)}.npy"
                            _require(mean_file.is_file() == (n >= 1) and variance_file.is_file() == (n >= 2),
                                     "zero_or_single_acceptance_map_policy_violated")
                            if n:
                                mean = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
                                difference = _arrays_equal(mean, _array(mean_file), "accepted_mean_differs")
                                counts["max_map_absolute_difference"] = max(counts["max_map_absolute_difference"], difference)
                                metrics = volume_metrics(mean, case["target"], observed, hidden_mask=case["hidden_mask"])
                                _compare_record({"ensemble_blind_iou": metrics["blind_iou"], "ensemble_brier_blind": metrics["brier_blind"],
                                                 "ensemble_hidden_error": metrics["hidden_error"]}, row)
                            else:
                                _compare_record({"ensemble_blind_iou": None, "ensemble_brier_blind": None, "ensemble_hidden_error": None}, row)
                                counts["zero_accepted_rows"] += 1
                            if n >= 2:
                                variance_map = selected.var(axis=0, ddof=1, dtype=np.float64).astype(np.float32)
                                difference = _arrays_equal(variance_map, _array(variance_file), "accepted_ddof1_variance_differs")
                                counts["max_map_absolute_difference"] = max(counts["max_map_absolute_difference"], difference)
                            elif n == 1:
                                counts["single_accepted_rows"] += 1
                            expected_rows.append(row)
                    counts["adaptations"] += 1
        completions = {p.resolve() for p in (folder / split / "cases").rglob("complete.json")}
        _require(completions == expected_completions, "unexpected_or_missing_uncertainty_cases")
        table = _csv(folder / split / "per_case_acceptance.csv")
        key = lambda r: (r["sample_id"], r["mode"], int(r["initialization"]), float(r["variance"]), float(r["threshold"]))
        table_map = {key(r): r for r in table}
        _require(len(table_map) == len(table) == len(expected_rows) == status["summary_rows"], "uncertainty_public_table_count_differs")
        for row in expected_rows:
            _compare_record(row, table_map[key(row)])
        counts["summary_rows"] += len(table)
    return counts


def audit_robustness(root: Path, manifest: Path, protocol: dict, config: dict, *, formal: bool = True) -> dict:
    from revision_robustness import _conditions, _metrics
    folder = root / "revision_robustness"
    identity, status = _json(folder / "identity.json"), _json(folder / "status.json")
    digest = stable_hash(identity)
    _require(status.get("status") == ("complete_subset_diagnostic" if formal else "draft_smoke")
             and status.get("identity_sha256") == digest and not _json(folder / "failures.json"), "robustness_incomplete")
    _require(identity.get("manifest_sha256") == sha256_file(manifest) and identity.get("config_sha256") == stable_hash(config)
             and identity.get("source_sha256") == _json(root / "run_identity.json")["source_sha256"]
             and identity.get("threshold_selection_sha256") == sha256_file(root / "thresholds.json"), "robustness_identity_differs")
    if formal:
        _check_evaluation_policy_metadata(identity.get("evaluation_numerical_policy"))
        _require(identity.get("numerics_source_sha256") == numerics_sha256(), "robustness_numerics_helper_differs")
    _require(identity.get("diagnostic_source_sha256") == sha256_file(ROOT / "revision_robustness.py")
             and identity.get("selection_source_sha256") == sha256_file(ROOT / "revision_diagnostics.py")
             and identity.get("analysis_source_sha256") == sha256_file(ROOT / "revision_analysis.py"),
             "robustness_diagnostic_source_differs")
    ids = select_parent_cases(protocol, "test", 30 if formal else 1)
    _require(identity.get("sample_ids") == ids == _json(folder / "selection.json")["sample_ids"], "robustness_parent_selection_differs")
    methods = METHODS if formal else tuple(identity["methods"])
    _require(set(identity["methods"]) == set(methods), "robustness_methods_differ")
    expected_conditions = _conditions()
    _require(identity.get("conditions") == expected_conditions, "robustness_registered_conditions_differ")
    actual_conditions = status["conditions"]
    _require(all(condition in actual_conditions for condition in expected_conditions)
             and len({r["condition"] for r in actual_conditions}) == len(actual_conditions), "mandatory_robustness_conditions_missing")
    if len(actual_conditions) > len(expected_conditions):
        _require(status["optional_noise"].get("status") == "complete", "optional_noise_started_but_incomplete")
    rows = _csv(folder / "per_case.csv")
    keys = [(r["condition"], r["sample_id"], r["method"]) for r in rows]
    _require(len(keys) == len(set(keys)) and set(keys) == {(c["condition"], sample, method)
             for c in actual_conditions for sample in ids for method in methods}, "robustness_coverage_differs")
    _require(status.get("case_count") == len(ids) and status.get("case_method_runs") == len(rows)
             and status.get("failure_count") == 0, "robustness_status_counts_differ")
    thresholds = _json(root / "thresholds.json")["thresholds"]
    for condition in actual_conditions:
        dataset = ProtocolDataset(str(manifest), "test", num_holes=condition["holes"],
                                  samples_per_hole=int(config["samples_per_hole"]),
                                  mask_seed=condition["evaluation_mask_seed"], noise_rate=condition["noise_rate"])
        for row in (r for r in rows if r["condition"] == condition["condition"]):
            path = _path(folder, row["prediction_file"])
            _require(sha256_file(path) == row["prediction_sha256"], "robustness_prediction_hash_differs")
            probability = _array(path, tuple(protocol["grid_size"]), probability=True)
            _compare(float(thresholds[row["method"]]), row["threshold_used"], "robustness_threshold_differs")
            _compare_record(_metrics(probability, dataset.get_case(row["sample_id"]), thresholds[row["method"]]), row)
            journal = _json(folder / condition["condition"] / "journal" / row["method"] / (row["sample_id"] + ".json"))
            _require(journal.get("status") == "complete" and journal.get("identity_sha256") == digest,
                     "robustness_case_journal_incomplete")
            _compare_record(journal["record"], row)
    return {"cases": len(ids), "conditions": len(actual_conditions), "methods": len(methods), "verified_rows": len(rows)}


def audit_real(root: Path, private_out: Path, config: dict, *, formal: bool = True) -> dict:
    private_out = assert_private_real_path(private_out)
    identity = _json(private_out / "identity.json")
    digest = stable_hash(identity)
    run_identity = _json(root / "run_identity.json")
    if formal:
        _check_evaluation_policy_metadata(identity.get("evaluation_numerical_policy"))
    _require(identity.get("manifest_sha256") == run_identity["manifest_sha256"]
             and identity.get("config") == config and identity.get("threshold_file_sha256") == sha256_file(root / "thresholds.json"),
             "private_real_protocol_identity_differs")
    sources, methods, masks = identity["real_sources"], tuple(identity["methods"]), identity["mask_seeds"]
    _require(len(methods) == len(set(methods)) and len(masks) == len(set(masks)), "private_real_repeated_methods_or_masks")
    _require(not formal or (len(sources) == 122 and set(methods) == set(METHODS) and len(masks) == 3), "private_real_registered_cohort_differs")
    _require(not _json(private_out / "failures.json"), "private_real_runtime_failures")
    for relative, expected in identity["source_files"].items():
        _require(sha256_file(_path(ROOT, relative)) == expected, "private_real_source_changed")
    _require(set(identity["source_files"]) == set(DIAGNOSTIC_SOURCE_FILES), "private_real_source_inventory_differs")
    for kind, expected in identity["checkpoint_sha256"].items():
        _require(sha256_file(root / "models" / kind / f"seed{identity['seed']}" / "best.pt") == expected,
                 "private_real_checkpoint_hash_differs")
    real_root = _runtime_path(run_identity["data_root"], root) / "mining_ply"
    _require({p.name for p in real_root.glob("*.ply")} == set(sources), "private_real_source_set_differs")
    for name, digest_source in sources.items():
        _require(sha256_file(_path(real_root, name)) == digest_source, "private_real_reference_hash_differs")
    exclusions = _json(private_out / "exclusions.json")
    excluded = {row["private_case_id"] for row in exclusions}
    _require(len(excluded) == len(exclusions), "private_real_duplicate_exclusions")
    expected_ids = {"private-" + stable_hash([name, source_digest])[:16] for name, source_digest in sources.items()}
    _require(excluded <= expected_ids, "private_real_exclusion_set_differs")
    valid_ids = expected_ids - excluded
    _require(bool(valid_ids), "no_valid_real_reference_blocks")
    thresholds = _json(root / "thresholds.json")["thresholds"]
    rows = []
    for case_id in sorted(valid_ids):
        folder = private_out / "cases" / case_id
        record = _checked_completion(folder, stable_hash(identity))
        target = _array(folder / "target.npy", tuple(config["grid_size"]))
        _require(np.isin(target, (0, 1)).all(), "private_real_target_not_binary")
        metadata = _json(folder / "private_metadata.json")
        _require(metadata["source"] in sources and metadata["source_sha256"] == sources[metadata["source"]],
                 "private_real_case_reference_differs")
        stored = record["metric_rows"]
        row_map = {(int(r["mask_seed"]), r["method"]): r for r in stored}
        _require(len(stored) == len(row_map) and set(row_map) == {(mask, method) for mask in masks for method in methods},
                 "private_real_method_mask_coverage_differs")
        for mask in masks:
            observed, hidden = make_drill_masks(tuple(config["grid_size"]), case_id, int(config["num_holes"]),
                                                int(config["samples_per_hole"]), mask, 4)
            observation = _array(folder / f"observation_mask{mask}.npy", (2, *config["grid_size"]))
            _arrays_equal(observation, np.stack((target * observed, observed)), "private_real_observation_differs")
            for method in methods:
                probability = _array(folder / f"prediction_{method}_mask{mask}.npy", tuple(config["grid_size"]), probability=True)
                result = volume_metrics(probability, target, observed, threshold=float(thresholds[method]), hidden_mask=hidden)
                row = row_map[(mask, method)]
                _compare_record(result, row)
                _require(row["private_case_id"] == case_id, "private_real_row_case_differs")
                rows.append(row)
    completed = {p.parent.name for p in (private_out / "cases").glob("*/complete.json")}
    _require(completed == valid_ids, "private_real_completed_set_differs")
    public = _json(root / "real_aggregate" / "real_summary.json")
    public_keys = {"format", "status", "total_reference_blocks", "excluded_reference_blocks", "runtime_failures",
                   "evaluated_methods", "scope", "reference_role", "domain_policy", "unit", "coordinate_units_confirmed",
                   "real_shapes_or_coordinates_exported", "statistical_policy", "private_audit_identity_sha256", "metrics"}
    _require(set(public) <= public_keys and (not formal or set(public) == public_keys),
             "private_real_unexpected_public_summary_fields")
    _require({p.relative_to(root / "real_aggregate").as_posix() for p in (root / "real_aggregate").rglob("*") if p.is_file()}
             == {"real_summary.json", "real_aggregate_metrics.csv"}, "private_real_unexpected_public_files")
    recomputed = aggregate_real_rows(rows, len(sources), len(exclusions), 0)
    _require(public.get("status") in {"complete", "complete_with_exclusions"}
             and public.get("private_audit_identity_sha256") == stable_hash(identity)
             and public.get("real_shapes_or_coordinates_exported") is False
             and len(public["metrics"]) == len(recomputed), "private_real_public_summary_identity_differs")
    for actual, old in zip(recomputed, public["metrics"]):
        _compare_record(actual, old)
    public_csv = _csv(root / "real_aggregate" / "real_aggregate_metrics.csv")
    _require(len(public_csv) == len(recomputed), "private_real_public_table_count_differs")
    for actual, old in zip(recomputed, public_csv):
        _compare_record(actual, old)
    # Enforce the exact public allow-list, including against added private keys.
    _require(all(set(old) == set(actual) for actual, old in zip(recomputed, public["metrics"])), "private_real_unexpected_public_fields")
    _require(all(set(old) == set(actual) for actual, old in zip(recomputed, public_csv)), "private_real_unexpected_public_csv_fields")
    return {"reference_blocks": len(sources), "excluded_blocks": len(exclusions), "evaluated_blocks": len(valid_ids),
            "methods": len(methods), "masks": len(masks), "recomputed_case_method_rows": len(rows),
            "private_audit_identity_sha256": stable_hash(identity), "public_summary_sha256": sha256_file(root / "real_aggregate" / "real_summary.json")}


def audit(run_root: str | Path, private_real_out: str | Path | None = None,
          verify_raw: bool = True, *, formal: bool = True,
          require_real: bool = True, require_ablations: bool = True) -> Path:
    """Fail closed. Draft switches are explicit and never produce formal_passed."""
    root = Path(run_root).resolve()
    output = root / "final_integrity" / "report.json"
    report = {"format": "esi-final-integrity-v1", "status": "running", "formal": formal,
              "audit_source_sha256": sha256_file(__file__), "confidential_content_exported": False,
              "checks": {}, "started_unix": time.time()}
    save_json(output, report)
    stage = "frozen_protocol"
    try:
        if formal:
            stage = "evaluation_numerical_policy"
            configure_evaluation(root, phase="final_integrity")
            report["checks"][stage] = audit_numerical_policy(root, _json(root / "run_identity.json"))
        stage = "frozen_protocol"
        report["checks"][stage], manifest, protocol = audit_frozen(root, formal=formal)
        report["manifest_sha256"] = sha256_file(manifest)
        identity = _json(root / "run_identity.json")
        report["source_sha256"] = identity["source_sha256"]
        config = load_config(root / "tuning" / "selected_config.json")
        if verify_raw:
            stage = "raw_sources"
            report["checks"][stage] = audit_raw(root, identity, formal=formal)
        else:
            _require(not formal, "formal_audit_cannot_skip_raw_hashes")
            report["checks"]["raw_sources"] = {"status": "explicitly_skipped_draft_only"}
        stage = "primary_results"
        report["checks"][stage] = audit_main(root, protocol, config, formal=formal)
        stage = "uncertainty_numerical_recomputation"
        report["checks"][stage] = audit_uncertainty(root, manifest, protocol, config, formal=formal)
        stage = "robustness_numerical_recomputation"
        report["checks"][stage] = audit_robustness(root, manifest, protocol, config, formal=formal)
        if require_ablations:
            stage = "ablation_results"
            checked = 0
            for arm, methods in (("base_no_observed", ("base_vae",)), ("base_observed", ("base_vae",)),
                                 ("attention_no_position", ("geovae",)),
                                 ("attention_position", ("geovae", "geovae_latent", "geovae_joint"))):
                threshold_path = root / "ablation" / "thresholds" / f"{arm}.json"
                threshold = _json(threshold_path)
                _require(threshold.get("status") == ("validation_frozen" if formal else "draft"), "ablation_thresholds_not_frozen")
                arm_config = load_config(root / "ablation" / "evaluation_configs" / f"{arm}.json")
                _require(threshold.get("config_sha256") == stable_hash(arm_config), "ablation_frozen_config_differs")
                for seed in ((11, 23, 37) if formal else identity["seeds"]):
                    folder = root / "ablation" / "evaluations" / "test" / arm / f"seed{seed}"
                    result = audit_evaluation(folder, protocol, identity["manifest_sha256"], threshold["config_sha256"],
                                              identity["source_sha256"], threshold["thresholds"], sha256_file(threshold_path),
                                              split="test", seed=seed, methods=methods, formal=formal,
                                              checkpoint_hashes={methods[0]: sha256_file(root / "ablation" / "models" / arm / f"seed{seed}" / "best.pt")})
                    checked += result["rows"]
            report["checks"][stage] = {"arms": 4, "verified_case_method_rows": checked}
        else:
            _require(not formal, "formal_audit_cannot_skip_ablations")
        if require_real:
            stage = "private_real_numerical_recomputation"
            _require(private_real_out is not None, "private_real_archive_required")
            report["checks"][stage] = audit_real(root, Path(private_real_out), config, formal=formal)
        else:
            _require(not formal, "formal_audit_cannot_skip_private_real")
        report["status"] = "passed" if formal else "draft_passed_not_manuscript_evidence"
        report["completed_unix"] = time.time()
        save_json(output, report)
        return output
    except Exception as error:
        report.update({"status": "failed", "failed_stage": stage,
                       "error_code": str(error) if isinstance(error, AuditError) else "required_artifact_or_numeric_check_failed",
                       "error_type": type(error).__name__, "completed_unix": time.time()})
        save_json(output, report)
        # Detailed errors, if needed, stay in the confidential audit directory.
        if private_real_out is not None:
            try:
                private_path = assert_private_real_path(private_real_out)
                save_json(private_path / "final_integrity_private_error.json", {"stage": stage, "error_type": type(error).__name__, "message": str(error)})
            except Exception:
                pass
        raise AuditError("Final integrity failed at " + stage + "; consult the aggregate audit report") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--private-real-out")
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args(argv)
    print(audit(args.run_root, args.private_real_out, not args.skip_raw, formal=not args.draft,
                require_real=not args.skip_real, require_ablations=not args.skip_ablations))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
