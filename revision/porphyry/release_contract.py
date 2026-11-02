"""Immutable execution scope, release evidence and author attempt reservations.

Public full evaluation keeps all15x10 rows. The registered author software check
uses exactly the previously approved10 calls, with the original absolute clock.
Registration never asserts completed numerical validation. Release is separately
evidence-gated, so changing a Boolean cannot invalidate prepared-source caches.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from porphyry_public import METHODS, inside, read, require, sha, stable
from author_bundle import START, HOURS, DEADLINE, NUMERICS_SHA

HERE = Path(__file__).resolve().parent
CASE = "porphyry_01_cu_ge_0p3"
MASK = 42
OLD_LAUNCHER_SHA = "ff3a83691398b094ddc1e49330a9c51f97fdd266261214d623072a27514f0fb6"
SCOPE = {"purpose": "software_integration_subset", "case_id": CASE,
         "cu_threshold_percent": .3, "mask_seed": MASK, "model_seed": 11,
         "methods": list(METHODS), "expected_predictions": 10,
         "scientific_full150_execution_claimed": False}


def scope_for(identity):
    scope = identity.get("execution_scope")
    if scope is None:
        return None
    require(scope == SCOPE and identity.get("origin") == "imported_author_synthetic_checkpoints"
            and identity.get("formal") is True, "Unsupported or fabricated execution scope")
    return SCOPE.copy()


def cases_for(data, identity):
    # The shared caller must first validate the entire15-condition dataset.
    scope = scope_for(identity)
    if scope is None:
        return data["conditions"]
    cases = [case for case in data["conditions"] if case["case_id"] == CASE and case["mask_seed"] == MASK]
    require(len(cases) == 1 and cases[0]["cu_threshold_percent"] == .3, "Fixed software condition absent")
    return cases


def exclusive_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return False
    try:
        data = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
        offset = 0
        while offset < len(data):
            count = os.write(fd, data[offset:]); require(count > 0, "Could not persist exclusive reservation")
            offset += count
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


def require_reference_inventory(files):
    fixed = {"identity.json", "status.json", "saved_prediction_audit.json", "failures.json"}
    fixed.update(f"cases/{CASE}/mask{MASK}/{method}/{name}" for method in METHODS
                 for name in ("journal.json", "raw.npy", "projected.npy", "adaptation.json", "backbone.json"))
    proofs = {name for name in files if name.startswith("numerical_policy/porphyry_01_pid")
              and name.endswith(".json") and Path(name).parent.as_posix() == "numerical_policy"}
    require(bool(proofs) and set(files) == fixed | proofs,
            "Reference inventory requires fixed10 x five artifacts, four completion files and numerical proofs")
    return proofs


def registration(path, ctx=None, *, require_time=False, owned_worker=False):
    """Read source-bound candidate registration, not caller-provided permission."""
    from portable_evaluation import source_inventory
    path = Path(path).resolve(); value = read(path); root = path.parent
    require(value.get("format") == "author-public-production-integration-v2"
            and value.get("scope") == SCOPE and value.get("prediction_attempt_limit") == 10
            and value.get("budget") == {"started_unix": START, "hours": HOURS, "deadline_unix": DEADLINE}
            and value.get("wrapper_source_sha256") == source_inventory(), "Candidate registration, source or immutable clock differs")
    require(value.get("paths") == {"out": "porphyry_01_production_v2",
            "attempts": "porphyry_01_operator_v1/attempts", "lock": "porphyry_01_operator_v1.lock"},
            "Shared old/new attempt ledger and lock cannot be relocated")
    require(value.get("old_launcher_sha256") == OLD_LAUNCHER_SHA
            and sha(root.parent / "actual_operator_parity.py") == OLD_LAUNCHER_SHA,
            "Historical launcher changed")
    require(sha(root.parent / "actual_operator_parity_v2.py") == value.get("launcher_sha256"), "Candidate launcher changed")
    if ctx is not None:
        require(ctx["out"] == inside(root, value["paths"]["out"])
                and ctx["parent"].get("verified_author_bundle") is True
                and value.get("bundle_sha256") == ctx["parent"]["bundle_sha256"]
                and value.get("run_identity_sha256") == ctx["parent"]["run_identity_sha256"]
                and value.get("prepared_manifest_sha256") == ctx["data"]["manifest_sha256"]
                and value.get("preparation_identity_sha256") == ctx["data"]["preparation_identity_sha256"],
                "Registration is not bound to actual original author/input context")
        reference = ctx["primary_run"] / "public_supplement/porphyry_01"
        files = value.get("reference_files", {})
        require_reference_inventory(files)
        require(all(sha(inside(reference, name)) == expected for name, expected in files.items()),
                "Original reference completion/artifact binding changed")
        require(value.get("original_external_manifest_sha256") ==
                "24097d0b7f1fd5462d0cb8060f6a8c9b1e4609b2e660283cabdeae64636c1f85", "Registered original external manifest differs")
        ref_identity, audit = read(reference / "identity.json"), read(reference / "saved_prediction_audit.json")
        require(ref_identity.get("formal") is True and ref_identity.get("runner_sha256") ==
                "9f596bf83e76e999a936fe119213b872aa2edcf247d3f40ceec9ea074fd3f41b"
                and ref_identity.get("external_manifest_sha256") == "24097d0b7f1fd5462d0cb8060f6a8c9b1e4609b2e660283cabdeae64636c1f85"
                and ref_identity.get("external_preregistration_sha256") == "7810883290f432581f6f33666e8f4724ec6f2f6288f26f4c1b1caa0c5f78d651"
                and ref_identity.get("checkpoint_sha256") == ctx["checkpoint_sha256"]
                and ref_identity.get("config_sha256") == stable(ctx["config"])
                and ref_identity.get("method_thresholds") == ctx["threshold_record"]["thresholds"]
                and audit.get("formal") is True and audit.get("rows") == 150
                and audit.get("identity_sha256") == stable(ref_identity)
                and audit.get("status") == "passed_saved_probability_recomputation"
                and read(reference / "status.json").get("status") == "complete_external_synthetic_descriptive"
                and read(reference / "failures.json") == [], "Full original reference150 has not passed")
        for key in ("training_image_id", "training_image_file_sha256", "training_image_array_sha256"):
            require(ref_identity.get(key) == ctx[key], "Original reference training-only image differs")
        require(ref_identity.get("numerics_source_sha256") == NUMERICS_SHA, "Reference numerical helper differs")
        _checked_policy(ref_identity.get("numerical_policy"))
        proofs = [name for name in files if name.startswith("numerical_policy/porphyry_01_pid")]
        require(bool(proofs), "Original numerical configuration proof missing")
        for name in proofs:
            _checked_numerical_proof(read(inside(reference, name)), ref_identity["numerical_policy"], "porphyry_01")
    if require_time:
        require(time.time() < DEADLINE, "Original author deadline expired")
    if owned_worker:
        require(os.name == "posix" and os.getpgrp() == os.getpid(), "Integration worker must own its process group")
        owner = read(inside(root, value["paths"]["lock"]))
        control = read(inside(root, value["paths"]["out"]) / "supervisor.json")
        require(owner == control and control.get("pid") == os.getppid()
                and control.get("registration_sha256") == sha(path)
                and control.get("deadline_unix") == DEADLINE,
                "Worker is not the registered exclusive supervisor child")
    return value


def check_attempt_inventory(path, identity=None):
    reg = registration(path)
    folder = inside(Path(path).resolve().parent, reg["paths"]["attempts"])
    paths = list(folder.glob("*.json")) if folder.exists() else []
    require(len(paths) <= 10 and {p.stem for p in paths} <= set(METHODS), "Unexpected or excessive attempt inventory")
    for item in paths:
        record = read(item)
        require(record.get("registration_sha256") == sha(path)
                and record.get("execution_scope") == SCOPE
                and record.get("method") == item.stem
                and START <= record.get("started_unix", -1) < DEADLINE,
                "Historical/foreign attempts exist; never run old10 then new10")
        if identity is not None:
            require(record.get("identity_sha256") == stable(identity), "Reserved attempt identity differs")
    return paths


def reserve_attempt(path, identity, method, seed):
    reg = registration(path, require_time=True, owned_worker=True)
    require(scope_for(identity) == SCOPE and method in METHODS
            and identity.get("integration_registration_sha256") == sha(path), "Attempt scope or binding differs")
    check_attempt_inventory(path, identity)
    folder = inside(Path(path).resolve().parent, reg["paths"]["attempts"])
    record = {"identity_sha256": stable(identity), "registration_sha256": sha(path),
              "execution_scope": SCOPE, "method": method, "seed": seed, "started_unix": time.time()}
    require(record["started_unix"] < DEADLINE and exclusive_json(folder / (method + ".json"), record),
            "Method already attempted without a committed matching result; extra prediction refused")
    return record


def release_evidence():
    """Release registry is a separate immutable artifact, not a mutable flag.

    Registry publication is outside candidate preparation. Evidence must bind
    current source and attest to separate actual10 and saved-reference150 audits.
    Missing/stale/negative evidence keeps the normal full prediction entry shut.
    """
    from portable_evaluation import source_inventory
    path = HERE / "release_registry.json"
    if not path.is_file():
        return {"status": "not_released", "prediction_entry_enabled": False}
    value = read(path)
    require(value.get("format") == "porphyry-portable-reviewed-release-v1"
            and value.get("wrapper_source_sha256") == source_inventory(), "Release registry source differs")
    required = {"actual10", "reference150", "independent_review"}
    require(set(value.get("evidence", {})) == required, "Release evidence inventory incomplete")
    proofs = {}
    for role, item in value["evidence"].items():
        file = inside(HERE, item["file"])
        require(sha(file) == item["sha256"], "Release evidence file changed")
        proofs[role] = read(file)
    require(proofs["actual10"].get("status") == "passed_actual_public_subset_parity"
            and proofs["actual10"].get("actual_prediction_rows") == 10
            and proofs["actual10"].get("scope") == SCOPE
            and proofs["actual10"].get("wrapper_source_sha256") == source_inventory(), "Actual production subset parity is absent")
    require(proofs["reference150"].get("status") == "passed_author_reference_import"
            and proofs["reference150"].get("rows") == 150
            and proofs["reference150"].get("public_prediction_rows") == 0
            and proofs["reference150"].get("wrapper_source_sha256") == source_inventory(), "Saved-reference150 audit is absent")
    require(proofs["independent_review"].get("status") == "passed_release_review"
            and proofs["independent_review"].get("wrapper_source_sha256") == source_inventory()
            and proofs["independent_review"].get("unresolved_findings") == [], "Independent release review remains unresolved")
    return {"status": "released_with_bounded_validation", "prediction_entry_enabled": True,
            "release_registry_sha256": sha(path), "actual_public_subset_rows": 10,
            "author_reference_saved_only_rows": 150, "independent_public_full150_executed": False,
            "public_from_scratch_training_executed": False}


DECLARED_POLICY = {
    "version": "esi-evaluation-numerics-v1", "cublas_workspace_config": ":4096:8",
    "deterministic_algorithms": True, "deterministic_warn_only": False,
    "cudnn_benchmark": False, "cudnn_deterministic": True,
    "scope": "evaluation_only_training_recipe_unchanged",
}


def _checked_policy(value):
    """Validate saved strict policy without importing the Torch-based helper."""
    require(isinstance(value, dict) and set(value) == {"declared", "actual"}
            and value["declared"] == DECLARED_POLICY, "Declared numerical policy differs")
    actual = value["actual"]
    mandatory = ("cublas_workspace_config", "deterministic_algorithms", "deterministic_warn_only",
                 "cudnn_benchmark", "cudnn_deterministic")
    require(isinstance(actual, dict) and set(actual) == set(mandatory) | {
        "cuda_matmul_allow_tf32", "cudnn_allow_tf32", "float32_matmul_precision"}, "Numerical actual fields differ")
    require(all(actual[key] == DECLARED_POLICY[key] and type(actual[key]) is type(DECLARED_POLICY[key])
                for key in mandatory), "Strict evaluation numerical flags differ")
    require(type(actual["cuda_matmul_allow_tf32"]) is bool and type(actual["cudnn_allow_tf32"]) is bool
            and actual["float32_matmul_precision"] in {"highest", "high", "medium"}, "Numerical backend fields invalid")
    return value


def _checked_numerical_proof(proof, policy, phase):
    _checked_policy(policy)
    require(proof.get("format") == "esi-numerical-policy-proof-v1" and proof.get("kind") == "evaluation"
            and proof.get("helper_sha256") == NUMERICS_SHA and proof.get("phase") == phase
            and proof.get("declared") == policy["declared"] and proof.get("actual") == policy["actual"]
            and proof.get("training_bitwise_reproducibility_claimed") is False
            and proof.get("cuda_initialized_at_configuration") is False
            and proof.get("workspace_origin") in {"preexisting_compatible", "configured_before_cuda_initialization"}
            and isinstance(proof.get("pid"), int) and proof["pid"] > 0
            and START <= proof.get("recorded_unix", -1) <= DEADLINE,
            "Numerical configuration proof differs or was not recorded before CUDA initialization")


def _compare_environment(actual, reference, policy):
    """Only the known redundant policy field is normalized; no other differences.

    The immutable internal runner captures environment_info before configure;
    the portable adapter resolves it afterwards. Their standalone numerical
    metadata and actual configuration proofs remain mandatory and exact.
    """
    _checked_policy(policy)
    require(isinstance(actual, dict) and isinstance(reference, dict), "Runtime environment metadata missing")
    require(actual.get("evaluation_numerical_policy") == policy, "Actual environment numerical policy differs")
    if "evaluation_numerical_policy" in reference:
        require(reference["evaluation_numerical_policy"] == policy, "Reference environment numerical policy differs")
    actual_base = {key: value for key, value in actual.items() if key != "evaluation_numerical_policy"}
    reference_base = {key: value for key, value in reference.items() if key != "evaluation_numerical_policy"}
    require(actual_base == reference_base, "Actual runtime environment fields differ")
    return {"rule": "exact_fields_with_separate_strict_numerical_policy_v1",
            "reference_had_redundant_policy_field": "evaluation_numerical_policy" in reference,
            "all_other_fields_exact": True, "policy_exact": True}
