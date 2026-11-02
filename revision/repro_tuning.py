"""Select per-method occupancy thresholds on complete validation predictions."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from config_loader import load_config
from experiment_data import ProtocolDataset
from repro_pipeline import sha256_file, stable_hash
from repro_report import REQUIRED_METHODS, _read_run

CANDIDATES = [round(float(value), 2) for value in np.arange(0.05, 1.0, 0.05)]


def _resolve_manifest(metadata: dict, evaluation_roots: list[Path],
                      output_path: Path) -> Path:
    """Find a byte-identical manifest in the original or copied run bundle."""
    expected = metadata["manifest_sha256"]
    original = Path(metadata["manifest"]).resolve()
    if original.is_file() and sha256_file(original) == expected:
        return original
    anchors = [output_path.parent, *output_path.parents]
    for evaluation_root in evaluation_roots:
        anchors.extend((evaluation_root, *evaluation_root.parents))
    visited: set[Path] = set()
    for anchor in anchors:
        root = anchor.resolve()
        if root in visited or not all(path.is_relative_to(root)
                                      for path in evaluation_roots):
            continue
        visited.add(root)
        candidate = root / "frozen_inputs" / "manifest.json"
        if candidate.is_file() and sha256_file(candidate) == expected:
            return candidate
    raise RuntimeError(
        "Validation manifest is unavailable or changed; the bundled "
        "frozen_inputs/manifest.json must match the recorded SHA-256")

def select_thresholds(evaluation_dirs: list[str], config_path: str,
                      output_path: str, draft: bool = False,
                      allow_subset: bool = False) -> Path:
    if not evaluation_dirs:
        raise ValueError("At least one validation evaluation is required")
    runs = [(Path(path).resolve(), *_read_run(path)) for path in evaluation_dirs]
    manifest_hashes = {meta["manifest_sha256"] for _, meta, _ in runs}
    config_hashes = {meta["config_sha256"] for _, meta, _ in runs}
    source_hashes = {meta["source_sha256"] for _, meta, _ in runs}
    if len(manifest_hashes) != 1 or len(config_hashes) != 1 or len(source_hashes) != 1:
        raise RuntimeError("Validation evaluations have mismatched manifest, config or source")
    config = load_config(config_path)
    if stable_hash(config) != next(iter(config_hashes)):
        raise RuntimeError("Specified configuration differs from validation evaluations")
    settings = {(meta["split"], int(meta["num_holes"]), float(meta["noise_rate"]),
                 int(meta["mask_seed"]),
                 bool(meta["project_observed"]), int(meta["limit"]))
                for _, meta, _ in runs}
    if len(settings) != 1:
        raise RuntimeError("Validation runs have differing observation settings")
    split, holes, noise, mask_seed, projection, limit = next(iter(settings))
    if split != "val":
        raise RuntimeError("Thresholds may only be chosen on validation data")
    if not draft and (holes != 8 or noise != 0 or projection or limit != 0):
        raise RuntimeError("Formal threshold selection requires full clean validation data")
    manifest_paths = {str(Path(meta["manifest"]).resolve()) for _, meta, _ in runs}
    if len(manifest_paths) != 1:
        raise RuntimeError("Validation runs point to different manifests")
    output = Path(output_path).resolve()
    manifest = _resolve_manifest(runs[0][1], [directory for directory, _, _ in runs],
                                 output)
    if sha256_file(manifest) != next(iter(manifest_hashes)):
        raise RuntimeError("Manifest changed after validation")
    seeds = [int(meta["seed"]) for _, meta, _ in runs]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("Duplicate validation seed")
    case_sets = [{row["sample_id"] for row in rows} for _, _, rows in runs]
    if any(cases != case_sets[0] for cases in case_sets[1:]):
        raise RuntimeError("Validation runs contain different cases")
    method_sets = [set(meta["methods"]) for _, meta, _ in runs]
    if any(methods != method_sets[0] for methods in method_sets[1:]):
        raise RuntimeError("Validation seeds contain different methods")
    parent_ids = {row["parent_id"] for _, _, rows in runs for row in rows}
    if not draft:
        if len(seeds) < 3 or len(case_sets[0]) != 750 or len(parent_ids) != 150:
            raise RuntimeError("Formal selection requires 3 seeds, 750 cases and 150 parents")
        if not allow_subset and set(REQUIRED_METHODS) != method_sets[0]:
            raise RuntimeError("Formal selection requires all benchmark methods")
    thresholds = np.asarray(CANDIDATES, dtype=np.float32)
    scores: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list))
    for directory, meta, rows in runs:
        dataset = ProtocolDataset(
            str(manifest), "val", num_holes=holes,
            samples_per_hole=int(config["samples_per_hole"]),
            mask_seed=int(meta["mask_seed"]), noise_rate=noise, heldout_holes=4,
        )
        # Each case appears once per method.  Reconstruct its target and mask
        # only once instead of repeating that work for every prediction row.
        case_cache: dict[str, tuple[tuple[int, ...], np.ndarray, np.ndarray]] = {}
        for row in rows:
            sample_id, method = row["sample_id"], row["method"]
            if sample_id not in case_cache:
                case = dataset.get_case(sample_id)
                blind = ~case["observed_mask"]
                case_cache[sample_id] = (
                    case["target"].shape,
                    blind,
                    case["target"][blind] >= 0.5,
                )
            target_shape, blind, truth = case_cache[sample_id]
            file_path = (directory / row["prediction_file"]).resolve()
            probability = np.load(file_path, allow_pickle=False)
            if probability.shape != target_shape:
                raise RuntimeError(f"Grid shape mismatch: {sample_id}, {method}")
            values = probability[blind].astype(np.float32)
            if not np.isfinite(values).all():
                raise RuntimeError(f"Nonfinite prediction: {sample_id}, {method}")
            all_sorted = np.sort(values)
            positive_sorted = np.sort(values[truth])
            predicted_count = len(values) - np.searchsorted(
                all_sorted, thresholds, side="left")
            intersection = len(positive_sorted) - np.searchsorted(
                positive_sorted, thresholds, side="left")
            union = predicted_count + int(truth.sum()) - intersection
            case_scores = np.divide(
                intersection, union, out=np.ones_like(
                    intersection, dtype=np.float64), where=union > 0)
            scores[method][row["parent_id"]].append(case_scores)
    selected = {}
    validation_curves = {}
    for method in sorted(method_sets[0]):
        by_parent = scores[method]
        parent_scores = np.stack([
            np.mean(by_parent[parent], axis=0) for parent in sorted(by_parent)
        ])
        means = parent_scores.mean(axis=0)
        winner = min(range(len(CANDIDATES)),
                     key=lambda i: (-float(means[i]),
                                    abs(CANDIDATES[i] - 0.5), CANDIDATES[i]))
        selected[method] = CANDIDATES[winner]
        validation_curves[method] = [float(value) for value in means]
    result = {
        "status": "draft" if draft else "validation_frozen",
        "selector_source_sha256": sha256_file(Path(__file__)),
        "allow_subset": allow_subset,
        "selection_metric": "mean blind IoU of parent means across variants and seeds",
        "candidate_thresholds": CANDIDATES,
        "validation_curves": validation_curves,
        "thresholds": selected,
        "manifest_sha256": next(iter(manifest_hashes)),
        "config_sha256": next(iter(config_hashes)),
        "evaluation_source_sha256": next(iter(source_hashes)),
        "seeds": sorted(seeds),
        "mask_seed": mask_seed,
        "cases_per_seed": len(case_sets[0]),
        "independent_parents": len(parent_ids),
        "validation_evaluations": [str(directory) for directory, _, _ in runs],
    }
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", nargs="+", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()
    print(select_thresholds(args.evaluations, args.config, args.out, args.draft))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
