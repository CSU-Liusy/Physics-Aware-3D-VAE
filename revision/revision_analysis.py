"""Recompute synthetic revision results from hash-verified saved predictions.

The resampling unit is the generator parent. Model seeds, masks and the five
variants of a parent are averaged inside that parent before bootstrap draws.
This module does not train, tune, load private real meshes, or select cases.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiment_data import ProtocolDataset, make_drill_masks, verify_protocol
from repro_metrics import volume_metrics

CLASSICAL = frozenset({"idw", "rbf", "ik", "mps"})
REQUIRED = ("idw", "rbf", "ik", "mps", "unet", "base_vae", "cgan",
            "geovae", "geovae_latent", "geovae_joint")
METRICS = ("blind_iou", "blind_dice", "overall_iou", "overall_dice",
           "observed_error", "observed_violation_count", "observed_mse_truth",
           "input_error", "input_violation_count", "input_mse", "hidden_error",
           "input_error_0p5", "input_violation_count_0p5",
           "volume_relative_error", "volume_absolute_relative_error",
           "hd", "hd95", "chamfer", "brier_blind", "seconds")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _stable(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _number(value: Any) -> float | None:
    if value is None or value in ("", "None", "nan"):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def parent_bootstrap(rows: list[dict], metric: str, replicates: int = 2000,
                     seed: int = 20261025) -> dict:
    """Equal-parent mean and percentile interval; no voxel/seed pseudoreplication."""
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _number(row.get(metric))
        if value is not None:
            groups[str(row["parent_id"])].append(value)
    values = np.array([np.mean(groups[key]) for key in sorted(groups)], dtype=float)
    if not len(values):
        return {"mean": None, "ci_low": None, "ci_high": None,
                "independent_parents": 0, "valid_case_runs": 0}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    bootstrap = values[draws].mean(axis=1)
    return {"mean": float(values.mean()),
            "ci_low": float(np.quantile(bootstrap, .025)),
            "ci_high": float(np.quantile(bootstrap, .975)),
            "independent_parents": int(len(values)),
            "valid_case_runs": sum(map(len, groups.values()))}


def nested_drill_conditions(grid_size: tuple[int, int, int], sample_id: str,
                            mask_seed: int = 42, samples_per_hole: int = 12,
                            hole_counts: tuple[int, ...] = (2, 4, 6, 8, 10),
                            heldout_holes: int = 4) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Build target-independent nested observed masks and a fixed hidden mask."""
    conditions = {}
    previous = np.zeros(grid_size, dtype=bool)
    reference_hidden = None
    for holes in sorted(hole_counts):
        observed, hidden = make_drill_masks(grid_size, sample_id, holes,
                                            samples_per_hole, mask_seed, heldout_holes)
        if np.any(previous & ~observed) or np.any(observed & hidden):
            raise RuntimeError("Drill-mask nesting or held-out separation failed")
        if reference_hidden is not None and not np.array_equal(hidden, reference_hidden):
            raise RuntimeError("Hidden holes changed across retention conditions")
        conditions[holes] = (observed, hidden)
        previous, reference_hidden = observed, hidden
    return conditions


def _bins(probability: np.ndarray, truth: np.ndarray, blind: np.ndarray,
          totals: dict[str, np.ndarray]) -> None:
    p = probability[blind].astype(np.float64)
    y = truth[blind].astype(np.float64)
    index = np.minimum((p * 10).astype(int), 9)
    totals["count"] += np.bincount(index, minlength=10)
    totals["probability_sum"] += np.bincount(index, weights=p, minlength=10)
    totals["truth_sum"] += np.bincount(index, weights=y, minlength=10)


def _prediction(root: Path, row: dict, prefix: str, shape: tuple[int, ...]) -> np.ndarray:
    relative, digest = row.get(prefix + "_file"), row.get(prefix + "_sha256")
    if not relative or not digest:
        raise RuntimeError("Missing saved prediction identity")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise RuntimeError("Saved prediction missing or escapes evaluation directory")
    if _sha(path) != digest:
        raise RuntimeError("Saved prediction checksum differs")
    array = np.load(path, allow_pickle=False)
    if array.shape != shape or not np.isfinite(array).all():
        raise RuntimeError("Saved prediction shape or numeric values are invalid")
    return array


def analyse_evaluations(manifest: str | Path, directories: list[str | Path],
                       out_dir: str | Path, *, formal: bool = True,
                       required_methods: tuple[str, ...] = REQUIRED,
                       replicates: int = 2000, bootstrap_seed: int = 20261025,
                       config_path: str | Path | None = None) -> Path:
    """Strict synthetic-only reanalysis. Formal mode requires all 750 sealed cases."""
    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "analysis.json", {"status": "running", "formal": formal})
    manifest_path = Path(manifest).resolve()
    protocol = verify_protocol(manifest_path, verify_files=True)
    if protocol.get("protocol_version") != "procedural-lens-v1":
        raise ValueError("This public analysis accepts the independent synthetic protocol only")
    if formal and (protocol.get("split_counts") != {"train": 3500, "val": 750, "test": 750}
                   or protocol.get("parent_counts") != {"train": 700, "val": 150, "test": 150}
                   or tuple(protocol.get("grid_size", ())) != (32, 32, 32)
                   or replicates != 2000):
        raise ValueError("Formal analysis requires the frozen 5000-case protocol and 2000 bootstrap replicates")
    if not directories:
        raise ValueError("Evaluation directories are required")
    manifest_sha = _sha(manifest_path)
    metas, intents, loaded = [], [], []
    provenance = []
    for ordinal, directory in enumerate(directories):
        root = Path(directory).resolve()
        meta, intent = _json(root / "evaluation.json"), _json(root / "evaluation_intent.json")
        failures = _json(root / "failures.json")
        if failures or not meta.get("complete"):
            raise RuntimeError("Evaluation is incomplete or contains runtime failures")
        if meta.get("manifest_sha256") != manifest_sha or intent.get("manifest_sha256") != manifest_sha:
            raise RuntimeError("Data manifest identity differs")
        # Use the evaluator's serialization, so a changed intent cannot be hidden
        # by keeping an old successful summary beside it.
        from repro_pipeline import stable_hash
        if meta.get("journal_identity_sha256") != stable_hash(intent):
            raise RuntimeError("Evaluation intent identity differs")
        for key in ("seed", "split", "mask_seed", "num_holes", "noise_rate",
                    "methods", "project_observed", "source_sha256", "config_sha256",
                    "method_thresholds", "threshold_selection_sha256"):
            if meta.get(key) != intent.get(key):
                raise RuntimeError("Evaluation summary/intent mismatch for " + key)
        with (root / "per_case.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        split_ids = {entry["sample_id"] for entry in protocol["samples"]
                     if entry["split"] == meta["split"]}
        cases = set(intent["case_ids"])
        if not cases <= split_ids or len(cases) != len(intent["case_ids"]):
            raise RuntimeError("Evaluation contains repeated cases or cases from a different split")
        if len(cases) != int(meta["expected_cases"]):
            raise RuntimeError("Declared evaluation case count differs")
        if formal and (meta["split"] != "test" or cases != split_ids or len(cases) != 750
                       or meta["limit"] != 0 or meta["num_holes"] != 8
                       or float(meta["noise_rate"]) != 0 or meta["project_observed"]
                       or meta.get("threshold_selection_status") != "validation_frozen"
                       or not meta.get("threshold_selection_sha256")):
            raise RuntimeError("Formal report requires the complete frozen 750-case primary test")
        seen = set()
        for row in rows:
            key = (row["sample_id"], row["method"])
            if key in seen or row["sample_id"] not in cases or row["method"] not in meta["methods"]:
                raise RuntimeError("Repeated, missing, or unexpected case/method row")
            seen.add(key)
        if seen != {(sample_id, method) for sample_id in cases for method in meta["methods"]}:
            raise RuntimeError("Methods have unequal case coverage")
        metas.append(meta); intents.append(intent); loaded.append((root, rows))
        provenance.append({"evaluation_id": "evaluation_" + str(ordinal),
                           "seed": meta["seed"], "summary_sha256": _sha(root / "evaluation.json"),
                           "intent_sha256": _sha(root / "evaluation_intent.json"),
                           "per_case_sha256": _sha(root / "per_case.csv")})
    comparable = ("split", "manifest_sha256", "source_sha256", "config_sha256",
                  "mask_seed", "num_holes", "noise_rate", "project_observed",
                  "threshold_selection_sha256", "method_thresholds")
    if any(any(meta.get(k) != metas[0].get(k) for k in comparable) for meta in metas[1:]):
        raise RuntimeError("Evaluation runs differ in protocol, implementation, or frozen settings")
    if any(set(intent["case_ids"]) != set(intents[0]["case_ids"]) for intent in intents[1:]):
        raise RuntimeError("Evaluation runs use different cases")
    seeds = [int(meta["seed"]) for meta in metas]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("Duplicate model-seed evaluation")
    if formal and sorted(seeds) != [11, 23, 37]:
        raise RuntimeError("Formal report requires seeds 11, 23 and 37")
    if any(set(meta["methods"]) != set(required_methods) for meta in metas):
        raise RuntimeError("Required method set differs from the evaluation")
    if config_path is not None:
        from repro_pipeline import load_config, stable_hash
        config = load_config(str(config_path))
        if stable_hash(config) != metas[0]["config_sha256"]:
            raise RuntimeError("Supplied configuration differs from evaluation")
        if formal and int(config["latent_dim"]) != 256:
            raise RuntimeError("New primary results must use latent dimension 256")
    elif formal:
        raise ValueError("Formal revision analysis requires the frozen configuration")

    all_rows, retained, classical_identity = [], [], {}
    calibration = {method: {"count": np.zeros(10, dtype=np.int64),
                            "probability_sum": np.zeros(10), "truth_sum": np.zeros(10)}
                   for method in required_methods}
    for (root, rows), meta, intent in zip(loaded, metas, intents):
        dataset = ProtocolDataset(str(manifest_path), meta["split"],
                                  num_holes=int(meta["num_holes"]),
                                  samples_per_hole=int(intent["samples_per_hole"]),
                                  mask_seed=int(meta["mask_seed"]),
                                  heldout_holes=int(intent["heldout_holes"]),
                                  noise_rate=float(meta["noise_rate"]))
        for row in rows:
            sample_id, method = row["sample_id"], row["method"]
            case = dataset.get_case(sample_id)
            if row["parent_id"] != case["metadata"]["parent_id"]:
                raise RuntimeError("Per-case parent identity differs")
            threshold = float(row["threshold_used"])
            if threshold != float(meta["method_thresholds"][method]):
                raise RuntimeError("Per-case threshold differs from frozen settings")
            probability = _prediction(root, row, "prediction", tuple(case["target"].shape))
            metrics = volume_metrics(probability, case["target"], case["observed_mask"],
                                     threshold=threshold, hidden_mask=case["hidden_mask"])
            for key, value in metrics.items():
                if key not in row or key == "distance_unit":
                    continue
                if isinstance(value, (float, int)) and not isinstance(value, bool):
                    old = _number(row[key])
                    if old is None or not np.isclose(old, value, rtol=2e-6, atol=2e-7):
                        raise RuntimeError(f"Stored {key} differs from recomputed prediction")
                elif value is None and _number(row[key]) is not None:
                    raise RuntimeError("Invalid stored surface distance")
            raw = (_prediction(root, row, "raw_prediction", tuple(case["target"].shape))
                   if meta["project_observed"] else probability)
            known = case["observed_mask"]
            violations = int(np.count_nonzero((raw[known] >= threshold) != (case["obs"][0][known] >= .5)))
            violations_fixed = int(np.count_nonzero((raw[known] >= .5) !=
                                                    (case["obs"][0][known] >= .5)))
            item = {"sample_id": sample_id, "parent_id": row["parent_id"], "method": method,
                    "seed": int(meta["seed"]), "mask_seed": int(meta["mask_seed"]),
                    "threshold_used": threshold, "prediction_sha256": row["prediction_sha256"],
                    "seconds": _number(row.get("seconds")), **metrics,
                    "input_violation_count": violations,
                    "input_error": violations / int(known.sum()) if known.any() else None,
                    "input_violation_count_0p5": violations_fixed,
                    "input_error_0p5": violations_fixed / int(known.sum()) if known.any() else None,
                    "input_mse": float(np.mean((raw[known] - case["obs"][0][known]) ** 2))
                        if known.any() else None}
            all_rows.append(item)
            if method in CLASSICAL:
                key = (method, sample_id)
                if key in classical_identity:
                    if classical_identity[key] != row["prediction_sha256"]:
                        raise RuntimeError("Seed-independent classical predictions differ across seeds")
                    continue
                classical_identity[key] = row["prediction_sha256"]
            retained.append(item)
            _bins(probability, case["target"], ~known, calibration[method])
    _write_csv(output / "recomputed_per_case.csv", all_rows)
    summaries, seed_summaries = [], []
    for method in required_methods:
        rows = [row for row in retained if row["method"] == method]
        summary = {"method": method, "unique_cases": len({row["sample_id"] for row in rows}),
                   "independent_parents": len({row["parent_id"] for row in rows}),
                   "independent_model_seeds": 0 if method in CLASSICAL else len(seeds),
                   "retained_prediction_count": len(rows),
                   "empty_prediction_count": sum(row["empty_prediction"] for row in rows),
                   "invalid_surface_distance_count": sum(not row["surface_distance_valid"] for row in rows)}
        for metric in METRICS:
            estimate = parent_bootstrap(rows, metric, replicates, bootstrap_seed)
            summary.update({metric + "_" + key: value for key, value in estimate.items()})
            seed_estimates = []
            if method not in CLASSICAL:
                for model_seed in sorted(seeds):
                    values_by_parent: dict[str, list[float]] = defaultdict(list)
                    for row in rows:
                        value = _number(row.get(metric))
                        if row["seed"] == model_seed and value is not None:
                            values_by_parent[row["parent_id"]].append(value)
                    value = (float(np.mean([np.mean(v) for v in values_by_parent.values()]))
                             if values_by_parent else None)
                    seed_summaries.append({"method": method, "seed": model_seed,
                                           "metric": metric, "parent_macro_mean": value,
                                           "valid_parents": len(values_by_parent)})
                    if value is not None:
                        seed_estimates.append(value)
            summary[metric + "_model_seed_sd"] = (float(np.std(seed_estimates, ddof=1))
                                                  if len(seed_estimates) > 1 else None)
        summaries.append(summary)
    _write_csv(output / "summary.csv", summaries)
    _write_csv(output / "model_seed_means.csv", seed_summaries)
    paired = []
    by_method = {method: {(row["seed"], row["sample_id"]): row for row in all_rows
                          if row["method"] == method} for method in required_methods}
    for candidate in ("geovae", "geovae_latent", "geovae_joint"):
        if candidate not in by_method:
            continue
        for reference in required_methods:
            if reference == candidate:
                continue
            if set(by_method[candidate]) != set(by_method[reference]):
                raise RuntimeError("Paired methods differ in case/seed coverage")
            for metric in ("blind_iou", "blind_dice", "brier_blind", "observed_error", "hd95"):
                differences = []
                for key, a in by_method[candidate].items():
                    b = by_method[reference][key]
                    av, bv = _number(a.get(metric)), _number(b.get(metric))
                    if av is not None and bv is not None:
                        differences.append({"parent_id": a["parent_id"], "delta": av - bv})
                paired.append({"candidate": candidate, "reference": reference, "metric": metric,
                               **parent_bootstrap(differences, "delta", replicates, bootstrap_seed)})
    _write_csv(output / "paired_differences.csv", paired)
    bins = []
    for method, totals in calibration.items():
        for i, count in enumerate(totals["count"]):
            bins.append({"method": method, "bin": i, "lower": i / 10, "upper": (i + 1) / 10,
                         "prediction_voxel_count": int(count),
                         "mean_probability": float(totals["probability_sum"][i] / count) if count else None,
                         "empirical_frequency": float(totals["truth_sum"][i] / count) if count else None})
    _write_csv(output / "reliability.csv", bins)
    _write_json(output / "analysis.json", {
        "status": "complete_primary_reanalysis" if formal else "draft_reanalysis",
        "analysis_script_sha256": _sha(Path(__file__)),
        "manifest_sha256": manifest_sha, "source_sha256": metas[0]["source_sha256"],
        "config_sha256": metas[0]["config_sha256"], "evaluation_provenance": provenance,
        "seeds": sorted(seeds), "methods": list(required_methods),
        "case_count": len(intents[0]["case_ids"]), "distance_unit": "voxel",
        "bootstrap": {"unit": "generator_parent", "replicates": replicates, "seed": bootstrap_seed,
                      "estimator": "equal-parent mean of case/variant/mask/model-seed means",
                      "interval": "2.5th and 97.5th percentile of parent-resampled means"},
        "distance_missing_policy": "Undefined for either empty surface; count and report separately, never impute zero.",
        "classical_seed_policy": "Check prediction hashes equal across model seeds; retain once in marginal statistics.",
        "calibration_policy": "Blind voxels only. Deep-model predictions pooled across model seeds; classical copies counted once. Voxel counts are not independent experimental sample sizes.",
        "public_data_policy": "Synthetic probability grids only; private real data are not read by this module.",
    })
    return output / "summary.csv"


def analyze(run_root: str | Path) -> Path:
    """Revision controller hook; never discovers or combines unrelated runs."""
    root = Path(run_root).resolve()
    identity = _json(root / "run_identity.json")
    manifest = root / "frozen_inputs" / "manifest.json"
    if not manifest.is_file() or _sha(manifest) != identity.get("manifest_sha256"):
        raise RuntimeError("Frozen run manifest is missing or differs from run identity")
    config = root / "tuning" / "selected_config.json"
    if not config.is_file():
        raise RuntimeError("Frozen validation-selected configuration is missing")
    return analyse_evaluations(manifest,
                               [root / "test" / f"seed{seed}" for seed in (11, 23, 37)],
                               root / "revision_analysis", config_path=config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evaluations", nargs="+", required=True)
    parser.add_argument("--config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--methods", nargs="+", default=list(REQUIRED))
    args = parser.parse_args(argv)
    print(analyse_evaluations(args.manifest, args.evaluations, args.out,
                             formal=not args.draft, config_path=args.config,
                             required_methods=tuple(args.methods), replicates=args.replicates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
