"""Strict aggregation for the sealed GeoVAE test protocol."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REQUIRED_METHODS = (
    "idw", "rbf", "ik", "mps", "unet", "base_vae", "cgan",
    "geovae", "geovae_latent", "geovae_joint",
)
METRICS = (
    "blind_iou", "blind_dice", "overall_iou", "overall_dice", "observed_error",
    "observed_violation_count", "observed_mse_truth", "hidden_error",
    "volume_relative_error", "volume_absolute_relative_error", "hd", "hd95", "chamfer", "brier_blind", "seconds",
    "preprojection_input_error", "postprojection_input_error",
)
CLASSICAL_METHODS = frozenset({"idw", "rbf", "ik", "mps"})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_run(directory: str | Path) -> tuple[dict, list[dict]]:
    root = Path(directory)
    with (root / "evaluation.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    with (root / "per_case.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with (root / "failures.json").open(encoding="utf-8") as stream:
        failures = json.load(stream)
    if failures or not metadata.get("complete"):
        raise RuntimeError(f"Incomplete evaluation: {root}")
    expected = int(metadata["expected_cases"])
    for method in metadata["methods"]:
        method_rows = [row for row in rows if row["method"] == method]
        if len(method_rows) != expected:
            raise RuntimeError(f"{root}: {method} has {len(method_rows)} rows; expected {expected}")
        if len({row["sample_id"] for row in method_rows}) != expected:
            raise RuntimeError(f"{root}: duplicated or missing cases for {method}")
    if len(rows) != expected * len(metadata["methods"]):
        raise RuntimeError(f"{root}: unexpected extra or missing per-case rows")
    for row in rows:
        for prefix in ("prediction", "raw_prediction"):
            relative = row.get(f"{prefix}_file", "")
            if not relative:
                continue
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise RuntimeError(f"{root}: missing or invalid {prefix} file for {row['sample_id']}")
            if _sha256_file(path) != row.get(f"{prefix}_sha256"):
                raise RuntimeError(f"{root}: {prefix} hash mismatch for {row['sample_id']}")
    return metadata, rows


def _number(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value in ("", None, "None", "nan"):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _bootstrap_parent_ci(rows: list[dict], metric: str, seed: int = 20261010,
                         replicates: int = 2000) -> tuple[float | None, float | None]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _number(row, metric)
        if value is not None:
            groups[row["parent_id"]].append(value)
    parents = sorted(groups)
    if not parents:
        return None, None
    rng = np.random.default_rng(seed)
    parent_means = np.array([np.mean(groups[p]) for p in parents], dtype=np.float64)
    draws = rng.integers(0, len(parents), size=(replicates, len(parents)))
    sampled = parent_means[draws].mean(axis=1)
    return float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))


def report(directories: list[str], out_dir: str, draft: bool = False,
           required_methods: tuple[str, ...] = REQUIRED_METHODS) -> Path:
    if not directories:
        raise ValueError("At least one evaluation directory is required")
    runs = [_read_run(path) for path in directories]
    manifest_hashes = {meta["manifest_sha256"] for meta, _ in runs}
    if len(manifest_hashes) != 1:
        raise RuntimeError("Evaluation runs use different data manifests")
    source_hashes = {meta["source_sha256"] for meta, _ in runs}
    if len(source_hashes) != 1:
        raise RuntimeError("Evaluation runs use different source code")
    config_hashes = {meta["config_sha256"] for meta, _ in runs}
    if len(config_hashes) != 1:
        raise RuntimeError("Evaluation runs use different configuration files")
    settings = {(meta["split"], int(meta["num_holes"]), float(meta["noise_rate"]),
                 int(meta["mask_seed"]),
                 bool(meta["project_observed"]), int(meta["limit"]),
                 meta.get("threshold_selection_sha256"))
                for meta, _ in runs}
    if len(settings) != 1:
        raise RuntimeError("Evaluation runs have different masks, noise or projection settings")
    split, holes, noise, mask_seed, projected, limit, threshold_sha = next(iter(settings))
    if not draft and (split != "test" or holes != 8 or noise != 0.0 or limit != 0):
        raise RuntimeError("Final report requires full sealed test, 8 holes and no label noise")
    if not draft and threshold_sha is None:
        raise RuntimeError("Final report requires frozen validation thresholds")
    if not draft and any(meta.get("threshold_selection_status") != "validation_frozen"
                         for meta, _ in runs):
        raise RuntimeError("Final report requires formal validation threshold selection")
    by_seed_method: dict[tuple[int, str], list[dict]] = {}
    seed_cases: dict[int, set[str]] = {}
    for meta, rows in runs:
        run_seed = int(meta["seed"])
        cases = {row["sample_id"] for row in rows}
        if run_seed in seed_cases and seed_cases[run_seed] != cases:
            raise RuntimeError(f"Seed {run_seed} used differing case sets")
        seed_cases[run_seed] = cases
        for method in meta["methods"]:
            key = (run_seed, method)
            if key in by_seed_method:
                raise RuntimeError(f"Duplicate evaluation for seed/method {key}")
            by_seed_method[key] = [row for row in rows if row["method"] == method]
    if len({frozenset(cases) for cases in seed_cases.values()}) != 1:
        raise RuntimeError("Methods or seeds evaluated different test cases")
    if not draft:
        expected_cases = {int(meta["expected_cases"]) for meta, _ in runs}
        if expected_cases != {750}:
            raise RuntimeError(f"Final report requires exactly 750 sealed test cases; found {expected_cases}")
        parent_ids = {row["parent_id"] for _, rows in runs for row in rows}
        if len(parent_ids) != 150:
            raise RuntimeError(f"Final report requires 150 independent parents; found {len(parent_ids)}")
    seeds = sorted(seed_cases)
    found_methods = sorted({method for _, method in by_seed_method})
    # These methods have no learned seed. They appear in each seed's matched
    # comparison, but their repeated rows are the same predictions, not new
    # independent observations or independent model fits.
    for method in sorted(CLASSICAL_METHODS & set(found_methods)):
        reference = {row["sample_id"]: row for row in by_seed_method.get((seeds[0], method), [])}
        for run_seed in seeds[1:]:
            current = {row["sample_id"]: row for row in by_seed_method.get((run_seed, method), [])}
            if set(current) != set(reference):
                raise RuntimeError(f"Classical {method} has different case sets across seeds")
            if any(current[sample_id]["prediction_sha256"] != reference[sample_id]["prediction_sha256"]
                   for sample_id in reference):
                raise RuntimeError(f"Classical {method} changed prediction across model seeds")
    if not draft:
        if len(seeds) < 3:
            raise RuntimeError(f"Final report requires at least 3 seeds; found {len(seeds)}")
        missing = sorted(set(required_methods) - set(found_methods))
        if missing:
            raise RuntimeError(f"Missing required methods: {missing}")
        for run_seed in seeds:
            missing_seed = sorted(set(required_methods) -
                                  {method for seed_value, method in by_seed_method if seed_value == run_seed})
            if missing_seed:
                raise RuntimeError(f"Seed {run_seed} lacks methods: {missing_seed}")
    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for method in found_methods:
        all_method_rows = [row for (run_seed, run_method), rows in by_seed_method.items()
                           if run_method == method for row in rows]
        method_rows = (by_seed_method.get((seeds[0], method), [])
                       if method in CLASSICAL_METHODS else all_method_rows)
        if not method_rows:
            continue
        item = {
            "method": method,
            "seeds": len({seed for seed, name in by_seed_method if name == method}),
            "independent_training_seeds": (0 if method in CLASSICAL_METHODS else
                                           len({seed for seed, name in by_seed_method if name == method})),
            "cases_per_seed": len({row["sample_id"] for row in method_rows}),
            "case_runs": len(all_method_rows),
            "unique_prediction_count": len(method_rows),
            "empty_prediction_count": sum(row["empty_prediction"] == "True"
                                          for row in method_rows),
            "invalid_surface_distance_count": sum(
                row.get("surface_distance_valid") == "False" or
                _number(row, "hd95") is None for row in method_rows),
        }
        for metric in METRICS:
            values = [value for row in method_rows if (value := _number(row, metric)) is not None]
            parent_metric_values: dict[str, list[float]] = defaultdict(list)
            for row in method_rows:
                value = _number(row, metric)
                if value is not None:
                    parent_metric_values[row["parent_id"]].append(value)
            # Equal weight per generator parent, including for distance metrics
            # whose valid-case counts can differ because of empty predictions.
            item[f"{metric}_mean"] = (
                float(np.mean([np.mean(v) for v in parent_metric_values.values()]))
                if parent_metric_values else None)
            item[f"{metric}_parent_median"] = (
                float(np.median([np.mean(v) for v in parent_metric_values.values()]))
                if parent_metric_values else None)
            low, high = _bootstrap_parent_ci(method_rows, metric)
            item[f"{metric}_ci_low"] = low
            item[f"{metric}_ci_high"] = high
            item[f"{metric}_n"] = len(values)
            if method in CLASSICAL_METHODS:
                item[f"{metric}_seed_sd"] = None
            else:
                seed_means = []
                for run_seed in seeds:
                    seed_rows = by_seed_method.get((run_seed, method), [])
                    seed_values = [v for row in seed_rows
                                   if (v := _number(row, metric)) is not None]
                    if seed_values:
                        seed_means.append(float(np.mean(seed_values)))
                item[f"{metric}_seed_sd"] = (float(np.std(seed_means, ddof=1))
                                              if len(seed_means) >= 2 else None)
        summary.append(item)
    fieldnames = list(summary[0])
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    pairwise_rows = []
    for candidate in ("geovae", "geovae_latent", "geovae_joint"):
        if candidate not in found_methods:
            continue
        for reference in ("idw", "rbf", "ik", "mps", "unet", "base_vae", "cgan"):
            if reference not in found_methods:
                continue
            paired = []
            for run_seed in seeds:
                candidate_rows = {row["sample_id"]: row for row in
                                  by_seed_method.get((run_seed, candidate), [])}
                reference_rows = {row["sample_id"]: row for row in
                                  by_seed_method.get((run_seed, reference), [])}
                if set(candidate_rows) != set(reference_rows):
                    raise RuntimeError("Paired comparison has mismatched cases")
                for sample_id, candidate_row in candidate_rows.items():
                    reference_row = reference_rows[sample_id]
                    for metric in ("blind_iou", "blind_dice", "brier_blind", "hd95"):
                        candidate_value = _number(candidate_row, metric)
                        reference_value = _number(reference_row, metric)
                        if candidate_value is None or reference_value is None:
                            continue
                        paired.append({
                            "sample_id": sample_id,
                            "parent_id": candidate_row["parent_id"],
                            "seed": run_seed, "metric": metric,
                            "delta": candidate_value - reference_value,
                        })
            for metric in ("blind_iou", "blind_dice", "brier_blind", "hd95"):
                subset = [row for row in paired if row["metric"] == metric]
                if not subset:
                    continue
                low, high = _bootstrap_parent_ci(subset, "delta")
                pairwise_rows.append({
                    "candidate": candidate, "reference": reference, "metric": metric,
                    "mean_paired_delta": float(np.mean([
                        np.mean([row["delta"] for row in subset if row["parent_id"] == parent])
                        for parent in sorted({row["parent_id"] for row in subset})])),
                    "parent_bootstrap_ci_low": low,
                    "parent_bootstrap_ci_high": high,
                    "paired_case_runs": len(subset),
                    "independent_parents": len({row["parent_id"] for row in subset}),
                })
    if pairwise_rows:
        with (output / "paired_differences.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pairwise_rows[0]))
            writer.writeheader()
            writer.writerows(pairwise_rows)
    reliability_totals: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for directory, (metadata, _) in zip(directories, runs):
        with (Path(directory) / "reliability.json").open(encoding="utf-8") as stream:
            calibration = json.load(stream)
        for method, totals in calibration.items():
            if method in CLASSICAL_METHODS and int(metadata["seed"]) != seeds[0]:
                continue
            for key in ("count", "probability_sum", "truth_sum"):
                value = np.asarray(totals[key], dtype=np.float64)
                if value.shape != (10,):
                    raise RuntimeError(f"{directory}: invalid reliability bins")
                if key not in reliability_totals[method]:
                    reliability_totals[method][key] = np.zeros(10, dtype=np.float64)
                reliability_totals[method][key] += value
    calibration_rows = []
    for method in found_methods:
        totals = reliability_totals[method]
        for bin_index in range(10):
            count = int(totals["count"][bin_index])
            calibration_rows.append({
                "method": method, "bin": bin_index, "count": count,
                "mean_probability": (float(totals["probability_sum"][bin_index] / count)
                                     if count else None),
                "empirical_frequency": (float(totals["truth_sum"][bin_index] / count)
                                        if count else None),
            })
    with (output / "reliability.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(calibration_rows[0]))
        writer.writeheader()
        writer.writerows(calibration_rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axis = plt.subplots(figsize=(6.2, 5.3))
        axis.plot([0, 1], [0, 1], color="#555555", linestyle="--",
                  linewidth=1.0, label="perfect calibration")
        for method in found_methods:
            points = [row for row in calibration_rows
                      if row["method"] == method and row["count"] > 0]
            if points:
                axis.plot([row["mean_probability"] for row in points],
                          [row["empirical_frequency"] for row in points],
                          marker="o", markersize=3, linewidth=1.2, label=method)
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted ore probability",
                 ylabel="Observed ore frequency",
                 title="Blind-voxel calibration" + (" (draft)" if draft else ""))
        axis.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
        fig.tight_layout()
        fig.savefig(output / "reliability.png", dpi=180)
        plt.close(fig)
    except ImportError:
        pass
    report_meta = {
        "status": "draft" if draft else "complete_sealed_test",
        "manifest_sha256": next(iter(manifest_hashes)),
        "config_sha256": next(iter(config_hashes)),
        "source_sha256": next(iter(source_hashes)),
        "seeds": seeds,
        "methods": found_methods,
        "case_count": len(next(iter(seed_cases.values()))),
        "settings": {
            "split": split, "num_holes": holes, "noise_rate": noise,
            "mask_seed": mask_seed,
            "project_observed": projected, "limit": limit,
            "threshold_selection_sha256": threshold_sha,
        },
        "evaluation_directories": [str(Path(path).resolve()) for path in directories],
        "distance_unit": "voxel",
        "ci_method": "percentile bootstrap of equal-parent means over parent IDs (2000 replicates); variants, masks and model seeds are averaged within parent",
        "distance_missing_policy": "Surface distances are undefined for empty prediction or truth; invalid counts are reported and undefined distances are never imputed as zero.",
        "paired_comparisons": "descriptive effect sizes; choose the primary reference on validation, not from test results",
        "classical_seed_policy": (
            "IDW, RBF, IK and MPS are seed-independent for a fixed case and mask. "
            "Their per-seed copies are checked for identical prediction SHA256, "
            "counted once in summary statistics and calibration, and retained "
            "as matched controls in seed-specific paired comparisons."
        ),
    }
    (output / "report.json").write_text(json.dumps(report_meta, ensure_ascii=False,
                                                  indent=2), encoding="utf-8")
    return output / "summary.csv"
