"""Final public-synthetic verification; never starts missing experiments."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time
from repro_pipeline import ROOT, DEFAULT_CONFIG, save_json, sha256_file, source_hash, stable_hash
from revision_numerics import configure_evaluation

def figure_evidence_paths(root):
    """Small derived inputs consumed by every formal synthetic figure."""
    root = Path(root).resolve()
    diagnostic = json.loads((root / "uncertainty" / "identity.json").read_text(encoding="utf-8"))
    selection = json.loads((root / "uncertainty" / "selection.json").read_text(encoding="utf-8"))
    selected = diagnostic.get("selection", {}).get("test", [])
    if not selected or selection.get("test") != selected:
        raise RuntimeError("Figure selection differs from the verified diagnostic")
    case = (root / "uncertainty" / "test" / "cases" / selected[0] / "joint" /
            "init0" / "complete.json").resolve()
    if not case.is_relative_to((root / "uncertainty" / "test" / "cases").resolve()):
        raise RuntimeError("Selected figure case leaves the diagnostic archive")
    fixed = ("revision_analysis/summary.csv", "revision_analysis/analysis.json",
             "revision_analysis/paired_differences.csv", "revision_analysis/reliability.csv",
             "ablation/summaries/test/ablation_summary.csv",
             "revision_robustness/summary.csv", "revision_robustness/status.json",
             "revision_robustness/identity.json", "revision_robustness/selection.json",
             "revision_robustness/per_case.csv", "revision_robustness/retention_summary.csv",
             "uncertainty/identity.json", "uncertainty/selection.json",
             "uncertainty/test/status.json", "uncertainty/test/per_case_acceptance.csv",
             "test/seed11/evaluation.json", "test/seed11/evaluation_intent.json",
             "test/seed11/per_case.csv", "models/geovae/seed11/run.json")
    histories = tuple(f"models/{model}/seed{seed}/history.jsonl"
                      for model in ("geovae", "base_vae", "unet", "cgan")
                      for seed in (11, 23, 37))
    return fixed + histories + (case.relative_to(root).as_posix(),)

def verify_numerical_policy(root, identity):
    from revision_integrity import audit_numerical_policy
    required = ("tune", "validation", "test", "ablation_validation", "ablation_test",
                "uncertainty_val", "uncertainty_test", "robustness", "public_finalize")
    return audit_numerical_policy(Path(root), identity, required_phases=required)

def verify_robustness_artifacts(root, manifest, config):
    from experiment_data import ProtocolDataset
    from repro_pipeline import load_config
    from revision_robustness import _resume
    from revision_integrity import _check_evaluation_policy_metadata
    from revision_numerics import helper_sha256
    root, manifest = Path(root).resolve(), Path(manifest).resolve()
    folder = root / "revision_robustness"
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    identity, status = read(folder / "identity.json"), read(folder / "status.json")
    identity_sha = stable_hash(identity)
    settings = load_config(str(config))
    thresholds = read(root / "thresholds.json")
    if (status.get("identity_sha256") != identity_sha
            or identity.get("source_sha256") != source_hash()
            or identity.get("manifest_sha256") != sha256_file(manifest)
            or identity.get("config_sha256") != stable_hash(settings)
            or identity.get("numerics_source_sha256") != helper_sha256()
            or identity.get("threshold_selection_sha256") != sha256_file(root / "thresholds.json")):
        raise RuntimeError("Robustness artifact identity differs")
    _check_evaluation_policy_metadata(identity.get("evaluation_numerical_policy"))
    if read(manifest).get("protocol_version") != "procedural-lens-v1":
        raise RuntimeError("Only independent synthetic artifacts are public")
    if read(folder / "failures.json"):
        raise RuntimeError("Robustness contains unresolved failures")
    for condition in identity["conditions"]:
        if condition not in status["conditions"]:
            raise RuntimeError("A registered robustness condition is missing")
    conditions = {row["condition"]: row for row in status["conditions"]}
    ids, methods = identity["sample_ids"], identity["methods"]
    expected = {(condition, method, sample) for condition in conditions for method in methods for sample in ids}
    with (folder / "per_case.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(expected) or {(r["condition"], r["method"], r["sample_id"]) for r in rows} != expected:
        raise RuntimeError("Robustness case/method coverage differs")
    datasets = {}
    for row in rows:
        condition, method, sample = row["condition"], row["method"], row["sample_id"]
        c = conditions[condition]
        if condition not in datasets:
            datasets[condition] = ProtocolDataset(str(manifest), "test", num_holes=c["holes"],
                samples_per_hole=int(settings["samples_per_hole"]), mask_seed=c["evaluation_mask_seed"],
                heldout_holes=4, noise_rate=c["noise_rate"])
        case = datasets[condition].get_case(sample)
        journal_path = folder / condition / "journal" / method / (sample + ".json")
        journal = read(journal_path)
        if journal.get("identity_sha256") != identity_sha or journal.get("status") != "complete":
            raise RuntimeError("Robustness journal identity differs")
        prediction = folder / condition / "predictions" / method / (sample + ".npy")
        if prediction.resolve() != (folder / row["prediction_file"]).resolve():
            raise RuntimeError("Robustness prediction path differs")
        if row["parent_id"] != case["metadata"]["parent_id"]:
            raise RuntimeError("Robustness parent identity differs")
        if float(row["threshold_used"]) != float(thresholds["thresholds"][method]):
            raise RuntimeError("Robustness changed the frozen threshold")
        checked = _resume(journal_path, prediction, journal["case_identity_sha256"], case,
                          float(thresholds["thresholds"][method]))
        if checked is None:
            raise RuntimeError("Robustness saved prediction is missing")
        for key, value in row.items():
            original = checked[1].get(key)
            if value != ("" if original is None else str(original)):
                raise RuntimeError("Robustness CSV differs from its verified per-case journal")
    return len(rows)

def finalize(root):
    root = Path(root).resolve()
    configure_evaluation(root, phase="public_finalize")
    def read(path):
        if not path.is_file():
            raise RuntimeError("Required result is missing: " + str(path))
        return json.loads(path.read_text(encoding="utf-8"))
    identity = read(root / "run_identity.json")
    if identity.get("source_sha256") != source_hash():
        raise RuntimeError("Current engine differs from the frozen training engine")
    policy_audit = verify_numerical_policy(root, identity)
    from revision_integrity import _check_evaluation_policy_metadata
    from repro_report import _read_run
    from repro_ablation import ARMS
    for seed in (11, 23, 37):
        for split in ("validation", "test"):
            meta, _ = _read_run(root / split / f"seed{seed}")
            _check_evaluation_policy_metadata(meta.get("environment", {}).get("evaluation_numerical_policy"))
            if meta.get("source_sha256") != source_hash() or meta.get("manifest_sha256") != identity["manifest_sha256"]:
                raise RuntimeError("Primary result source or data identity differs")
        for arm in ARMS:
            for split in ("val", "test"):
                meta, rows = _read_run(root / "ablation" / "evaluations" / split / arm / f"seed{seed}")
                _check_evaluation_policy_metadata(meta.get("environment", {}).get("evaluation_numerical_policy"))
                if (meta.get("manifest_sha256") != identity["manifest_sha256"]
                        or meta.get("source_sha256") != source_hash() or len({r["sample_id"] for r in rows}) != 750):
                    raise RuntimeError("Ablation identity or full case coverage differs")
    diagnostic = read(root / "uncertainty" / "identity.json")
    diagnostic_sha = stable_hash(diagnostic)
    _check_evaluation_policy_metadata(diagnostic.get("evaluation_numerical_policy"))
    from revision_diagnostics import _source_identity
    if diagnostic.get("source_files") != _source_identity():
        raise RuntimeError("Local-variability source identity differs")
    from revision_diagnostics import _existing_artifacts
    for split in ("val", "test"):
        status = read(root / "uncertainty" / split / "status.json")
        if status.get("status") != "complete" or status.get("identity_sha256") != diagnostic_sha:
            raise RuntimeError("Local-variability diagnostics are incomplete")
        for sample_id in diagnostic["selection"][split]:
            for mode in diagnostic["settings"]["modes"]:
                for init in diagnostic["settings"]["initializations"]:
                    folder = root / "uncertainty" / split / "cases" / sample_id / mode / f"init{init}"
                    if _existing_artifacts(folder, diagnostic_sha) is None:
                        raise RuntimeError("Local-variability case artifacts are incomplete")
    robust = read(root / "revision_robustness" / "status.json")
    _check_evaluation_policy_metadata(read(root / "revision_robustness" / "identity.json").get("evaluation_numerical_policy"))
    if robust.get("status") != "complete_subset_diagnostic" or robust.get("case_count") != 30:
        raise RuntimeError("The registered 30-parent robustness diagnostic is incomplete")
    verified_robust_predictions = verify_robustness_artifacts(root, root / "frozen_inputs" / "manifest.json",
                                                            root / "tuning" / "selected_config.json")
    from revision_analysis import analyze
    analyze(root)
    keys = figure_evidence_paths(root)
    files = {}
    for name in keys:
        path = root / name
        if not path.is_file():
            raise RuntimeError("Missing final result: " + name)
        files[name] = sha256_file(path)
    real_summary = root / "real_aggregate" / "real_summary.json"
    real_status = ("externally_supplied_aggregate_only" if real_summary.is_file()
                   and read(real_summary).get("status") == "complete" else "not_available_restricted")
    save_json(root / "public_handoff.json", {
        "status": "public_synthetic_reproduction_complete", "source_sha256": source_hash(),
        "manifest_sha256": identity["manifest_sha256"], "key_files": files,
        "restricted_real_data_status": real_status,
        "raw_real_audit_passed": False,
        "verified_robustness_predictions": verified_robust_predictions,
        "evaluation_numerical_policy_audit": policy_audit,
        "scope": "Synthetic experiments only. This package does not reproduce the confidential real cohort by default.",
        "manuscript_ready": False})
    return root / "public_handoff.json"

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--robustness", action="store_true")
    stage.add_argument("--finalize", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "runs" / "esi_revision_v2"))
    parser.add_argument("--manifest", default=str(ROOT / "protocol" / "manifest.json"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    if args.robustness:
        from revision_robustness import run
        budget = json.loads((Path(args.out) / "revision_budget.json").read_text()) if (Path(args.out) / "revision_budget.json").is_file() else None
        deadline = budget["started_unix"] + budget["hours"] * 3600 if budget else None
        print(run(args.manifest, args.out, str(Path(args.out) / "tuning" / "selected_config.json"), deadline_epoch=deadline))
    else:
        print(finalize(args.out))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
