"""Prespecified synthetic mask and drill-retention diagnostics.

Every method sees the same 30 ID-selected test parents. These are subset
diagnostics, not a replacement for the complete primary test. The trained
mask seed remains in the checkpoint configuration; evaluation masks vary
explicitly without changing the training identity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiment_data import ProtocolDataset, verify_protocol
from repro_metrics import volume_metrics
from repro_pipeline import (
    ROOT, _load_checked_model, _predict, _prediction_cache_identity,
    _read_cached_prediction, _save_npy_atomic, environment_info,
    load_config, save_json, select_device, sha256_file, source_hash, stable_hash,
)
from revision_analysis import METRICS, nested_drill_conditions, parent_bootstrap
from revision_diagnostics import select_parent_cases
from revision_numerics import configure_evaluation, helper_sha256 as numerics_sha256, metadata as numerical_metadata

METHODS = ("idw", "rbf", "ik", "mps", "unet", "base_vae", "cgan",
           "geovae", "geovae_latent", "geovae_joint")
MASK_SEEDS = (42, 53, 67)
HOLE_COUNTS = (2, 4, 6, 8, 10)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def _check_deadline(deadline_epoch: float | None) -> None:
    if deadline_epoch is not None and time.time() >= deadline_epoch:
        raise TimeoutError("Revision budget exhausted; completed robustness cases are resumable")


def _condition(holes: int, mask_seed: int, noise: float = 0.) -> dict:
    return {"condition": f"h{holes}_m{mask_seed}_n{int(round(noise * 100)):02d}",
            "holes": holes, "evaluation_mask_seed": mask_seed, "noise_rate": noise,
            "retention_fraction": holes / 10}


def _conditions() -> list[dict]:
    conditions = [_condition(8, seed) for seed in MASK_SEEDS]
    conditions += [_condition(holes, 42) for holes in HOLE_COUNTS if holes != 8]
    return conditions


def _thresholds(root: Path, config: dict, manifest_sha: str, methods: tuple[str, ...],
                smoke: bool) -> tuple[dict[str, float], str]:
    path = root / "thresholds.json"
    record = _read(path)
    if record.get("manifest_sha256") != manifest_sha or record.get("config_sha256") != stable_hash(config):
        raise RuntimeError("Frozen threshold data/configuration identity differs")
    if record.get("evaluation_source_sha256") != source_hash():
        raise RuntimeError("Frozen threshold evaluation implementation differs")
    if not smoke and record.get("status") != "validation_frozen":
        raise RuntimeError("Formal diagnostics require validation-frozen thresholds")
    result = {method: float(record["thresholds"][method]) for method in methods}
    if any(not 0 < value < 1 for value in result.values()):
        raise ValueError("Invalid probability threshold")
    return result, sha256_file(path)


def _checked_shared_cache(root: Path, identity: dict, shape: tuple[int, ...],
                          config_sha: str) -> tuple[np.ndarray, dict] | None:
    cached = _read_cached_prediction(root / "prediction_cache", identity, shape)
    if cached is None:
        return None
    prediction, record = cached
    source = (root / record["source_prediction"]).resolve()
    evaluation_root = source.parents[2]
    meta = _read(evaluation_root / "evaluation.json")
    intent = _read(evaluation_root / "evaluation_intent.json")
    if (not meta.get("complete") or meta.get("journal_identity_sha256") != stable_hash(intent)
            or meta.get("journal_identity_sha256") != record.get("source_evaluation_identity_sha256")
            or meta.get("manifest_sha256") != identity["manifest_sha256"]
            or meta.get("source_sha256") != identity["source_sha256"]
            or meta.get("config_sha256") != config_sha):
        raise RuntimeError("Shared prediction source evaluation is incomplete or has a different identity")
    journal = _read(evaluation_root / "journal" / identity["method"] / (identity["sample_id"] + ".json"))
    if (journal.get("status") != "complete"
            or journal.get("evaluation_identity_sha256") != meta["journal_identity_sha256"]
            or journal.get("record", {}).get("prediction_sha256") != record["prediction_sha256"]):
        raise RuntimeError("Shared prediction source journal is incomplete or inconsistent")
    return prediction, record


def _metrics(prediction: np.ndarray, case: dict, threshold: float) -> dict:
    result = volume_metrics(prediction, case["target"], case["observed_mask"],
                            threshold=threshold, hidden_mask=case["hidden_mask"])
    known = case["observed_mask"]
    labels = case["obs"][0]
    violations = int(np.count_nonzero((prediction[known] >= threshold) != (labels[known] >= .5)))
    violations_0p5 = int(np.count_nonzero((prediction[known] >= .5) != (labels[known] >= .5)))
    result.update({"input_violation_count": violations,
                   "input_error": violations / int(known.sum()),
                   "input_violation_count_0p5": violations_0p5,
                   "input_error_0p5": violations_0p5 / int(known.sum()),
                   "input_mse": float(np.mean((prediction[known] - labels[known]) ** 2))})
    return result


def _resume(folder: Path, prediction_path: Path, case_sha: str, case: dict,
            threshold: float) -> tuple[np.ndarray, dict] | None:
    if not folder.is_file():
        return None
    journal = _read(folder)
    if journal.get("case_identity_sha256") != case_sha:
        raise RuntimeError("Robustness case identity changed; choose a new output directory")
    if journal.get("status") != "complete":
        return None
    row = journal["record"]
    if not prediction_path.is_file() or sha256_file(prediction_path) != row["prediction_sha256"]:
        raise RuntimeError("Robustness saved prediction checksum differs")
    prediction = np.load(prediction_path, allow_pickle=False)
    computed = _metrics(prediction, case, threshold)
    for key, value in computed.items():
        old = row.get(key)
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            if old is None or not np.isclose(value, old, rtol=2e-6, atol=2e-7):
                raise RuntimeError("Robustness journal metric differs from saved prediction")
        elif old != value:
            raise RuntimeError("Robustness journal empty-surface status differs")
    return prediction, row


def _summarize(rows: list[dict], conditions: list[dict], output: Path) -> None:
    summaries, paired = [], []
    methods = sorted({row["method"] for row in rows})
    reference_id = _condition(8, 42)["condition"]
    by_key = {(row["condition"], row["method"], row["sample_id"]): row for row in rows}
    for condition in conditions:
        for method in methods:
            selected = [row for row in rows if row["method"] == method
                        and row["condition"] == condition["condition"]]
            summary = {**condition, "method": method, "case_count": len(selected),
                       "independent_parents": len({row["parent_id"] for row in selected}),
                       "empty_prediction_count": sum(row["empty_prediction"] for row in selected),
                       "invalid_surface_distance_count": sum(not row["surface_distance_valid"] for row in selected)}
            for metric in METRICS:
                estimate = parent_bootstrap(selected, metric)
                summary.update({metric + "_" + key: value for key, value in estimate.items()})
                if condition["condition"] != reference_id:
                    differences = []
                    for row in selected:
                        reference = by_key.get((reference_id, method, row["sample_id"]))
                        a, b = row.get(metric), reference.get(metric) if reference else None
                        if a is not None and b is not None:
                            differences.append({"parent_id": row["parent_id"], "delta": a - b})
                    paired.append({**condition, "reference_condition": reference_id,
                                   "method": method, "metric": metric,
                                   **parent_bootstrap(differences, "delta")})
            summaries.append(summary)
    _csv(output / "summary.csv", summaries)
    _csv(output / "retention_summary.csv", [row for row in summaries
         if row["evaluation_mask_seed"] == 42 and row["noise_rate"] == 0])
    _csv(output / "mask_summary.csv", [row for row in summaries
         if row["holes"] == 8 and row["noise_rate"] == 0])
    _csv(output / "paired_differences.csv", paired)


def run(manifest: str | Path, run_root: str | Path, selected_config: str | Path,
        deadline_epoch: float | None = None, *, smoke: bool = False,
        methods: tuple[str, ...] = METHODS, include_noise: bool = True) -> Path:
    """Run mandatory diagnostics and optional noise only when a budget permits it."""
    _check_deadline(deadline_epoch)
    root, manifest = Path(run_root).resolve(), Path(manifest).resolve()
    configure_evaluation(root, phase="robustness")
    config = load_config(str(selected_config))
    protocol = verify_protocol(manifest, verify_files=True)
    if protocol.get("protocol_version") != "procedural-lens-v1":
        raise ValueError("Public robustness diagnostics require the independent synthetic protocol")
    if not smoke and (methods != METHODS or int(config["latent_dim"]) != 256
                      or protocol.get("parent_counts") != {"train": 700, "val": 150, "test": 150}):
        raise ValueError("Formal diagnostics require d256, all benchmark methods and the frozen parent-disjoint cohort")
    ids = select_parent_cases(protocol, "test", 1 if smoke else 30)
    config_sha, manifest_sha = stable_hash(config), sha256_file(manifest)
    thresholds, threshold_sha = _thresholds(root, config, manifest_sha, methods, smoke)
    if not smoke:
        for seed in (11, 23, 37):
            main = _read(root / "test" / f"seed{seed}" / "evaluation.json")
            if (not main.get("complete") or int(main.get("expected_cases", 0)) != 750
                    or main.get("manifest_sha256") != manifest_sha
                    or main.get("config_sha256") != config_sha
                    or main.get("threshold_selection_sha256") != threshold_sha):
                raise RuntimeError("Complete matching three-seed primary test must precede robustness diagnostics")
    device = select_device("cpu" if smoke else str(config.get("device", "cuda")))
    checkpoint_paths = {kind: root / "models" / kind / "seed11" / "best.pt"
                        for kind in ("geovae", "base_vae", "unet", "cgan")}
    needed = ({"geovae" if method.startswith("geovae") else method for method in methods}
              - {"idw", "rbf", "ik", "mps"})
    models, checkpoint_hashes = {}, {}
    for kind in sorted(needed):
        models[kind], _ = _load_checked_model(kind, str(checkpoint_paths[kind]),
                                            manifest_sha, config, 11, device)
        checkpoint_hashes[kind] = sha256_file(checkpoint_paths[kind])
    conditions = _conditions()
    output = root / "revision_robustness"
    output.mkdir(parents=True, exist_ok=True)
    identity = {"format": "esi-robustness-subset-v1", "smoke": smoke,
                "manifest_sha256": manifest_sha, "config_sha256": config_sha,
                "source_sha256": source_hash(), "diagnostic_source_sha256": sha256_file(Path(__file__)),
                "selection_source_sha256": sha256_file(ROOT / "revision_diagnostics.py"),
                "analysis_source_sha256": sha256_file(ROOT / "revision_analysis.py"),
                "checkpoint_sha256": checkpoint_hashes, "threshold_selection_sha256": threshold_sha,
                "methods": list(methods), "sample_ids": ids,
                "conditions": conditions, "optional_noise_rates": [.05, .1, .2],
                "training_mask_seed": int(config["mask_seed"]), "model_seed": 11,
                "case_count": len(ids), "projection_applied": False,
                "environment": environment_info(),
                "numerics_source_sha256": numerics_sha256(),
                "evaluation_numerical_policy": numerical_metadata()}
    identity_sha = stable_hash(identity)
    if (output / "identity.json").is_file() and _read(output / "identity.json") != identity:
        raise RuntimeError("Robustness identity changed; choose a new run directory")
    save_json(output / "identity.json", identity)
    save_json(output / "selection.json", {"sample_ids": ids, "selection_rule":
              f"hash IDs only, one case from each of {len(ids)} distinct test parents; same selection rule as local-variability analysis",
              "manifest_sha256": manifest_sha})
    save_json(output / "status.json", {"status": "running", "identity_sha256": identity_sha})
    rows, failures, computed_times = [], [], []
    history_path = output / "failure_history.json"
    failure_history = _read(history_path) if history_path.is_file() else []
    started = time.perf_counter()
    training_image = None
    training_image_id = training_image_sha = None
    if "mps" in methods:
        train = ProtocolDataset(str(manifest), "train", num_holes=int(config["num_holes"]),
                                samples_per_hole=int(config["samples_per_hole"]),
                                mask_seed=int(config["mask_seed"]))
        training_image_id = str(config.get("mps_training_image_id") or train.entries[0]["sample_id"])
        if training_image_id not in train.metadata:
            raise RuntimeError("MPS training image is not from the training split")
        training_image = train.get_case(training_image_id)["target"]
        training_image_sha = hashlib.sha256(np.ascontiguousarray(training_image).tobytes()).hexdigest()

    def execute(condition: dict) -> None:
        dataset = ProtocolDataset(str(manifest), "test", num_holes=condition["holes"],
                                  samples_per_hole=int(config["samples_per_hole"]),
                                  mask_seed=condition["evaluation_mask_seed"], heldout_holes=4,
                                  noise_rate=condition["noise_rate"])
        condition_folder = output / condition["condition"]
        for index, sample_id in enumerate(ids, 1):
            case = dataset.get_case(sample_id)
            # Verify the retained-hole design independently of labels or outcome.
            nested = nested_drill_conditions(tuple(config["grid_size"]), sample_id,
                      condition["evaluation_mask_seed"], int(config["samples_per_hole"]))
            if not np.array_equal(case["observed_mask"], nested[condition["holes"]][0]):
                raise RuntimeError("Retained-hole mask differs from registered nested design")
            case_config = dict(config, mask_seed=condition["evaluation_mask_seed"])
            for method in methods:
                journal_path = condition_folder / "journal" / method / (sample_id + ".json")
                prediction_path = condition_folder / "predictions" / method / (sample_id + ".npy")
                case_seed = int(hashlib.sha256(
                    f"{condition['evaluation_mask_seed']}:{sample_id}:{method}".encode()).hexdigest()[:8], 16)
                cache_identity = _prediction_cache_identity(
                    method=method, config=case_config, manifest_sha=manifest_sha, split="test",
                    sample_id=sample_id, sample_grid_sha=dataset.metadata[sample_id]["sha256"],
                    observation=case["obs"], holes=condition["holes"], noise_rate=condition["noise_rate"],
                    case_seed=case_seed, checkpoint_hashes=checkpoint_hashes,
                    training_image_id=training_image_id, training_image_sha256=training_image_sha,
                    device=device, source_sha=identity["source_sha256"], environment=identity["environment"])
                case_sha = stable_hash({"run": identity_sha, "prediction": cache_identity,
                                        "threshold": thresholds[method]})
                try:
                    _check_deadline(deadline_epoch)
                    previous = _resume(journal_path, prediction_path, case_sha, case, thresholds[method])
                    if previous is not None:
                        rows.append(previous[1])
                        computed_times.append((method, float(previous[1]["seconds"])))
                        continue
                    before = time.perf_counter()
                    cached = _checked_shared_cache(root, cache_identity, tuple(case["target"].shape), config_sha)
                    if cached is None:
                        prediction, adaptation = _predict(method, case["obs"], models, config,
                                                         device, training_image, case_seed)
                        seconds = time.perf_counter() - before
                        source_file, source_sha = "", ""
                        origin = "computed"
                    else:
                        prediction, cache_record = cached
                        adaptation = cache_record.get("adaptation", {})
                        seconds = float(cache_record["compute_seconds"])
                        source_file, source_sha = (cache_record["source_prediction"],
                                                   cache_record["prediction_sha256"])
                        origin = "verified_primary_cache"
                    metrics = _metrics(prediction, case, thresholds[method])
                    _save_npy_atomic(prediction_path, prediction)
                    row = {**condition, "sample_id": sample_id,
                           "parent_id": dataset.metadata[sample_id]["parent_id"], "method": method,
                           "seed": 11, "training_mask_seed": int(config["mask_seed"]),
                           "threshold_used": thresholds[method], "seconds": seconds,
                           "prediction_origin": origin,
                           "prediction_file": prediction_path.relative_to(output).as_posix(),
                           "prediction_sha256": sha256_file(prediction_path),
                           "source_prediction_file": source_file, "source_prediction_sha256": source_sha,
                           "adaptation": json.dumps(adaptation, sort_keys=True), **metrics}
                    save_json(journal_path, {"status": "complete", "identity_sha256": identity_sha,
                              "case_identity_sha256": case_sha, "record": row})
                    rows.append(row); computed_times.append((method, seconds))
                except Exception as error:
                    failure = {**condition, "sample_id": sample_id, "method": method,
                               "error_type": type(error).__name__, "message": str(error)}
                    failures.append(failure)
                    failure_history.append(failure)
                    save_json(output / "failures.json", failures)
                    save_json(history_path, failure_history)
                    save_json(journal_path, {"status": "failed", "identity_sha256": identity_sha,
                              "case_identity_sha256": case_sha, "failure": failure})
                    if isinstance(error, TimeoutError):
                        save_json(output / "status.json", {"status": "budget_exceeded",
                                  "identity_sha256": identity_sha, "completed_case_method_runs": len(rows)})
                        _csv(output / "per_case.csv", rows)
                        raise
            if index % 5 == 0 or index == len(ids):
                elapsed = time.perf_counter() - started
                print(f"[robustness] {condition['condition']} {index}/{len(ids)} cases; "
                      f"saved={len(rows)}, failures={len(failures)}, elapsed={elapsed/60:.1f} min", flush=True)

    for condition in conditions:
        execute(condition)
    optional = {"status": "not_run", "reason": "No optional-noise budget was supplied"}
    # Estimate the whole noise suite; never start just the favorable/cheapest
    # conditions, or shrink a mandatory case set to fit the remaining budget.
    per_method = {method: [value for name, value in computed_times if name == method]
                  for method in methods}
    noise_estimate = 1.5 * len(ids) * 3 * sum(float(np.mean(v)) for v in per_method.values() if v) + 600
    noise_status_path = output / "noise_plan.json"
    if noise_status_path.is_file() and _read(noise_status_path).get("status") == "registered_to_run":
        run_noise = True
    else:
        run_noise = bool(include_noise and deadline_epoch is not None
                         and deadline_epoch - time.time() > noise_estimate and not failures)
        optional = {"status": "registered_to_run" if run_noise else "not_run",
                    "rates": [.05, .1, .2], "estimated_required_seconds_with_margin": noise_estimate,
                    "reason": "Complete noise suite fits the remaining budget" if run_noise
                              else "Optional noise suite was not enabled or did not fit the remaining verified budget"}
        save_json(noise_status_path, optional)
    if run_noise:
        for noise in (.05, .1, .2):
            condition = _condition(8, 42, noise)
            conditions.append(condition)
            execute(condition)
        optional = {**_read(noise_status_path), "status": "complete" if not failures else "incomplete"}
    else:
        optional = _read(noise_status_path)
    expected = len(conditions) * len(ids) * len(methods)
    _csv(output / "per_case.csv", rows)
    save_json(output / "failures.json", failures)
    save_json(history_path, failure_history)
    if len(rows) != expected or failures:
        save_json(output / "status.json", {"status": "incomplete", "identity_sha256": identity_sha,
                  "expected_case_method_runs": expected, "completed_case_method_runs": len(rows),
                  "failure_count": len(failures)})
        raise RuntimeError("Robustness evaluation incomplete; all failed cases are recorded")
    _summarize(rows, conditions, output)
    save_json(output / "status.json", {"status": "draft_smoke" if smoke else "complete_subset_diagnostic",
              "identity_sha256": identity_sha, "case_count": len(ids), "independent_test_parents": len(ids),
              "model_seed": 11, "case_method_runs": len(rows), "failure_count": 0,
              "conditions": conditions, "optional_noise": optional,
              "elapsed_seconds_this_invocation": time.perf_counter() - started,
              "interpretation": "Subset diagnostic; not a full test result. No parameter or threshold selection uses these outcomes.",
              "ci_method": f"2000 percentile bootstrap draws over parent means; {len(ids)} independent synthetic parents, one selected variant per parent"})
    return output / "summary.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--budget-hours", type=float)
    parser.add_argument("--no-noise", action="store_true")
    args = parser.parse_args(argv)
    deadline = time.time() + args.budget_hours * 3600 if args.budget_hours is not None else None
    print(run(args.manifest, args.run_root, args.config, deadline_epoch=deadline,
              include_noise=not args.no_noise))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
