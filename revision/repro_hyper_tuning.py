"""Validation-only, resumable selection of evaluation hyperparameters.

The sealed test split is never opened here. A fixed sample of one case from each
of 50 validation parents is selected from IDs before targets are read. The
candidate grid is declared in this source file and every case has an atomic
journal so an interrupted MPS or test-time adaptation run can resume.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from experiment_data import ProtocolDataset, verify_protocol
from repro_pipeline import (
    _load_checked_model,
    _predict,
    _read_prediction_for_resume,
    _save_npy_atomic,
    load_config,
    save_json,
    select_device,
    sha256_file,
    source_hash,
    stable_hash,
    training_config_hash,
)

TUNING_SEED = 11
SELECTION_VERSION = 1
VALIDATION_PARENTS = 50


def select_validation_cases(manifest: dict[str, Any], count: int = VALIDATION_PARENTS) -> list[str]:
    """Choose parents and one variant per parent using IDs, never outcomes."""
    by_parent: dict[str, list[str]] = defaultdict(list)
    for row in manifest["samples"]:
        if row["split"] == "val":
            by_parent[str(row["parent_id"])].append(str(row["sample_id"]))
    if len(by_parent) < count:
        raise RuntimeError(
            f"Tuning requires {count} distinct validation parents; found {len(by_parent)}"
        )
    ranked_parents = sorted(
        by_parent,
        key=lambda parent: (stable_hash([SELECTION_VERSION, "parent", parent]), parent),
    )[:count]
    selected = [
        min(
            by_parent[parent],
            key=lambda sample: (stable_hash([SELECTION_VERSION, "variant", sample]), sample),
        )
        for parent in ranked_parents
    ]
    if any(Path(sample).name != sample or "/" in sample or "\\" in sample for sample in selected):
        raise ValueError("Unsafe validation sample ID in protocol")
    return selected


def candidate_grid() -> list[dict[str, Any]]:
    """A small fixed grid; modification changes the tuning identity."""
    candidates: list[dict[str, Any]] = []

    def add(method: str, overrides: dict[str, Any]) -> None:
        ordinal = sum(row["method"] == method for row in candidates)
        candidates.append({
            "method": method,
            "ordinal": ordinal,
            "candidate_id": f"{method}-{ordinal:02d}",
            "overrides": overrides,
        })

    for power in (1.0, 2.0, 3.0):
        add("idw", {"idw_power": power})
    for smoothing in (0.001, 0.01, 0.1):
        add("rbf", {"rbf_kernel": "linear", "rbf_smoothing": smoothing})
    for scale in (2.5, 5.0, 10.0):
        add("ik", {"ik_len_scale": scale, "ik_nugget": 0.01})
    for n_cond in (15, 25):
        add("mps", {"mps_n_cond": n_cond, "mps_realizations": 10})
    for steps in (25, 50, 100):
        for lr in (0.01, 0.03):
            add("geovae_latent", {
                "latent_test_time_steps": steps,
                "latent_test_time_lr": lr,
            })
    for steps in (25, 50, 100):
        for lora_lr in (0.0003, 0.001):
            add("geovae_joint", {
                "joint_test_time_steps": steps,
                "joint_test_time_lr": 0.03,
                "joint_test_time_lora_lr": lora_lr,
            })
    return candidates


def blind_scores(probability: np.ndarray, target: np.ndarray,
                 observed: np.ndarray) -> dict[str, float | int]:
    probability = np.asarray(probability, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if probability.shape != target.shape or observed.shape != target.shape:
        raise ValueError("Prediction, target and observed mask shapes differ")
    if not np.isfinite(probability).all() or np.any(probability < -1e-6) or np.any(probability > 1 + 1e-6):
        raise ValueError("Tuning prediction is not a finite probability volume")
    blind = ~np.asarray(observed, dtype=bool)
    if not np.any(blind):
        raise ValueError("Validation case has no blind voxels")
    predicted = probability >= 0.5
    truth = target >= 0.5
    intersection = int(np.count_nonzero(predicted & truth & blind))
    union = int(np.count_nonzero((predicted | truth) & blind))
    return {
        "blind_iou": float(intersection / union) if union else 1.0,
        "brier_blind": float(np.mean((probability[blind] - truth[blind]) ** 2)),
        "blind_voxels": int(np.count_nonzero(blind)),
        "blind_intersection": intersection,
        "blind_union": union,
    }


def _case_seed(mask_seed: int, sample_id: str, method: str) -> int:
    # Match the formal evaluator exactly; candidates must see identical stochastic
    # realizations so one hyperparameter does not get a more favorable random draw.
    return int(hashlib.sha256(
        f"{mask_seed}:{sample_id}:{method}".encode()
    ).hexdigest()[:8], 16)


def _valid_previous(row: dict[str, Any], identity_sha: str,
                    candidate_id: str, method: str, sample_id: str, grid_sha: str,
                    candidate_config_sha: str, observations_sha: str,
                    output: Path, target: np.ndarray, observed: np.ndarray) -> bool:
    if any(row.get(key) != expected for key, expected in (
        ("tuning_identity_sha256", identity_sha),
        ("candidate_id", candidate_id),
        ("method", method),
        ("sample_id", sample_id),
        ("source_grid_sha256", grid_sha),
        ("candidate_config_sha256", candidate_config_sha),
        ("observations_sha256", observations_sha),
        ("status", "complete"),
    )):
        return False
    scores = row.get("scores", {})
    try:
        iou = float(scores["blind_iou"])
        brier = float(scores["brier_blind"])
        n = int(scores["blind_voxels"])
    except (KeyError, TypeError, ValueError):
        return False
    if not (n > 0 and math.isfinite(iou) and 0 <= iou <= 1
            and math.isfinite(brier) and 0 <= brier <= 1):
        return False
    try:
        expected = Path("predictions") / method / candidate_id / f"{sample_id}.npy"
        prediction = _read_prediction_for_resume(
            output, row, "prediction", expected, target.shape
        )
        recomputed = blind_scores(prediction, target, observed)
        for key, value in recomputed.items():
            recorded = scores.get(key)
            if isinstance(value, int):
                if int(recorded) != value:
                    return False
            elif not math.isclose(float(recorded), value, rel_tol=1e-9, abs_tol=1e-9):
                return False
    except (KeyError, TypeError, ValueError, OSError):
        return False
    return True


def _rank_candidates(candidate_rows: list[dict[str, Any]],
                     count: int) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    for row in candidate_rows:
        scores = row["scores"]
        if len(scores) != count:
            raise RuntimeError(f"Candidate {row['candidate_id']} has {len(scores)}/{count} cases")
        summaries.append({
            "method": row["method"],
            "candidate_id": row["candidate_id"],
            "ordinal": row["ordinal"],
            "overrides": row["overrides"],
            "journal_directory": f"journal/{row['method']}/{row['candidate_id']}",
            "historical_failed_attempts": row["historical_failed_attempts"],
            "mean_blind_iou": float(np.mean([x["blind_iou"] for x in scores])),
            "mean_brier_blind": float(np.mean([x["brier_blind"] for x in scores])),
            "n_cases": count,
        })
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_method[row["method"]].append(row)
    selected = {
        method: min(rows, key=lambda row: (
            -row["mean_blind_iou"], row["mean_brier_blind"], row["ordinal"]
        ))
        for method, rows in by_method.items()
    }
    return selected, summaries


def tune(manifest: str, config_path: str, out: str, checkpoint: str,
         *, dry_run: bool = False) -> Path:
    """Return selected_config.json after complete validation-only tuning.

    A failure is journaled and stops the stage; rerunning with unchanged inputs
    retries that case. No candidate is silently omitted from selection.
    """
    manifest_path = Path(manifest).resolve()
    config_file = Path(config_path).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    output = Path(out).resolve()
    selected_path = output / "selected_config.json"
    manifest_data = verify_protocol(manifest_path, verify_files=False)
    case_ids = select_validation_cases(manifest_data)
    metadata = {row["sample_id"]: row for row in manifest_data["samples"] if row["split"] == "val"}
    config = load_config(config_file)
    candidates = candidate_grid()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Seed {TUNING_SEED} GeoVAE checkpoint missing: {checkpoint_path}")
    identity = {
        "format_version": 1,
        "split": "val",
        "selection_version": SELECTION_VERSION,
        "seed": TUNING_SEED,
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(config_file),
        "config_value_sha256": stable_hash(config),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "evaluation_source_sha256": source_hash(),
        "tuning_source_sha256": sha256_file(__file__),
        "case_ids": case_ids,
        "case_ids_sha256": stable_hash(case_ids),
        "source_grid_sha256": {sample_id: metadata[sample_id]["sha256"] for sample_id in case_ids},
        "candidates": candidates,
        "threshold_for_selection": 0.5,
        "selection_rule": "mean per-case blind IoU descending; mean blind Brier ascending; fixed ordinal ascending",
    }
    if dry_run:
        print(f"Tuning plan: {len(case_ids)} validation parents, {len(candidates)} candidates; output={output}", flush=True)
        return selected_path
    for module in ("gstools", "mpslib"):
        if importlib.util.find_spec(module) is None:
            raise RuntimeError(f"Validation tuning requires {module}; install baseline dependencies before running")
    dataset = ProtocolDataset(
        str(manifest_path), "val", num_holes=int(config["num_holes"]),
        samples_per_hole=int(config["samples_per_hole"]),
        mask_seed=int(config["mask_seed"]),
    )
    # Validate the 50 exact source grids once; ordinary case reads use the same
    # deterministic files and the preparation/finalization stages audit all 5,000.
    cases = {}
    for sample_id in case_ids:
        row = metadata[sample_id]
        grid_path = (manifest_path.parent / row["path"]).resolve()
        if not grid_path.is_relative_to(manifest_path.parent) or sha256_file(grid_path) != row["sha256"]:
            raise RuntimeError(f"Validation grid hash mismatch: {sample_id}")
        cases[sample_id] = dataset.get_case(sample_id)
    train_dataset = ProtocolDataset(
        str(manifest_path), "train", num_holes=int(config["num_holes"]),
        samples_per_hole=int(config["samples_per_hole"]),
        mask_seed=int(config["mask_seed"]),
    )
    training_image_id = str(config.get("mps_training_image_id") or train_dataset.entries[0]["sample_id"])
    if training_image_id not in train_dataset.metadata:
        raise RuntimeError("MPS training image must belong to the training split")
    training_image_row = train_dataset.metadata[training_image_id]
    training_grid_path = (manifest_path.parent / training_image_row["path"]).resolve()
    if sha256_file(training_grid_path) != training_image_row["sha256"]:
        raise RuntimeError("MPS training-image hash differs from manifest")
    training_image = train_dataset.get_case(training_image_id)["target"]
    identity["mps_training_image_id"] = training_image_id
    identity["mps_training_image_sha256"] = training_image_row["sha256"]
    # The training image identity is deterministic from the manifest and base
    # config, but it must be included before any case journal is trusted.
    existing_identity = output / "tuning_intent.json"
    if existing_identity.is_file():
        previous = json.loads(existing_identity.read_text(encoding="utf-8"))
        if previous != identity:
            raise RuntimeError(f"Existing tuning identity differs; use a fresh output directory: {output}")
    else:
        if any(path.exists() for path in (output / "journal", output / "predictions",
                                          selected_path, output / "selection.json")):
            raise RuntimeError(f"Existing tuning output lacks identity: {output}")
        save_json(existing_identity, identity)
    identity_sha = stable_hash(identity)

    device = select_device(str(config.get("device", "auto")))
    model, _ = _load_checked_model("geovae", str(checkpoint_path),
                                   identity["manifest_sha256"], config,
                                   TUNING_SEED, device)
    models = {"geovae": model}
    candidate_rows: list[dict[str, Any]] = []
    for number, candidate in enumerate(candidates, 1):
        method = candidate["method"]
        candidate_id = candidate["candidate_id"]
        candidate_config = {**config, **candidate["overrides"]}
        if training_config_hash(candidate_config) != training_config_hash(config):
            raise RuntimeError(f"Tuning candidate changes training configuration: {candidate_id}")
        rows: list[dict[str, Any]] = []
        historical_failed_attempts = 0
        print(f"[tune] {number}/{len(candidates)} {candidate_id}: {len(case_ids)} cases", flush=True)
        for index, sample_id in enumerate(case_ids, 1):
            path = output / "journal" / method / candidate_id / f"{sample_id}.json"
            case = cases[sample_id]
            observations_sha = hashlib.sha256(np.ascontiguousarray(case["obs"]).tobytes()).hexdigest()
            candidate_config_sha = stable_hash(candidate_config)
            if path.is_file():
                previous = json.loads(path.read_text(encoding="utf-8"))
                if _valid_previous(previous, identity_sha, candidate_id, method,
                                   sample_id, metadata[sample_id]["sha256"],
                                   candidate_config_sha, observations_sha,
                                   output, case["target"], case["observed_mask"]):
                    rows.append(previous["scores"])
                    historical_failed_attempts += len(previous.get("failure_history", []))
                    continue
                if previous.get("status") == "complete":
                    previous.setdefault("failure_history", []).append({
                        "error_type": "JournalValidationError",
                        "message": "Stored score or prediction failed resume integrity check",
                    })
            else:
                previous = {}
            history = list(previous.get("failure_history", []))
            started = time.perf_counter()
            try:
                prediction, adaptation = _predict(
                    method, case["obs"], models, candidate_config, device,
                    training_image if method == "mps" else None,
                    _case_seed(int(config["mask_seed"]), sample_id, method),
                )
                scores = blind_scores(prediction, case["target"], case["observed_mask"])
                prediction_path = output / "predictions" / method / candidate_id / f"{sample_id}.npy"
                _save_npy_atomic(prediction_path, prediction)
                record = {
                    "tuning_identity_sha256": identity_sha,
                    "candidate_id": candidate_id,
                    "method": method,
                    "sample_id": sample_id,
                    "parent_id": metadata[sample_id]["parent_id"],
                    "source_grid_sha256": metadata[sample_id]["sha256"],
                    "observations_sha256": observations_sha,
                    "prediction_file": str(prediction_path.relative_to(output)).replace("\\", "/"),
                    "prediction_sha256": sha256_file(prediction_path),
                    "candidate_config_sha256": candidate_config_sha,
                    "status": "complete",
                    "scores": scores,
                    "adaptation": adaptation,
                    "seconds": time.perf_counter() - started,
                    "failure_history": history,
                }
                save_json(path, record)
                rows.append(scores)
                historical_failed_attempts += len(history)
            except Exception as error:
                failure = {
                    "error_type": type(error).__name__, "message": str(error),
                    "seconds": time.perf_counter() - started,
                }
                history.append(failure)
                save_json(path, {
                    "tuning_identity_sha256": identity_sha,
                    "candidate_id": candidate_id, "method": method,
                    "sample_id": sample_id, "source_grid_sha256": metadata[sample_id]["sha256"],
                    "status": "failed", "failure_history": history,
                })
                raise RuntimeError(
                    f"Validation tuning failed at {candidate_id}, {sample_id}; "
                    f"recorded in {path}: {type(error).__name__}: {error}"
                ) from error
            if index % 10 == 0 or index == len(case_ids):
                print(f"[tune] {candidate_id}: {index}/{len(case_ids)}", flush=True)
        candidate_rows.append({**candidate, "scores": rows,
                               "historical_failed_attempts": historical_failed_attempts})

    selected, summaries = _rank_candidates(candidate_rows, len(case_ids))
    selected_config = dict(config)
    for method in ("idw", "rbf", "ik", "mps", "geovae_latent", "geovae_joint"):
        selected_config.update(selected[method]["overrides"])
    if training_config_hash(selected_config) != training_config_hash(config):
        raise RuntimeError("Selected evaluation settings unexpectedly alter training configuration")
    selection_path = output / "selection.json"
    if selected_path.is_file():
        if json.loads(selected_path.read_text(encoding="utf-8")) != selected_config:
            raise RuntimeError("Completed tuning configuration differs from recomputed validation selection")
    elif selection_path.exists():
        raise RuntimeError("Completed tuning selection exists without its configuration")
    else:
        save_json(selected_path, selected_config)
    expected_config_sha = sha256_file(selected_path)
    selection = {
        "status": "validation_selected",
        "split": "val",
        "seed_for_adaptation": TUNING_SEED,
        "manifest_sha256": identity["manifest_sha256"],
        "base_config_sha256": identity["config_sha256"],
        "base_config_value_sha256": identity["config_value_sha256"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "evaluation_source_sha256": identity["evaluation_source_sha256"],
        "tuning_source_sha256": identity["tuning_source_sha256"],
        "tuning_identity_sha256": identity_sha,
        "case_ids": case_ids,
        "case_ids_sha256": identity["case_ids_sha256"],
        "selected_case_grid_sha256": identity["source_grid_sha256"],
        "mps_training_image_id": training_image_id,
        "mps_training_image_sha256": training_image_row["sha256"],
        "candidates": summaries,
        "selected": selected,
        "selected_config_sha256": expected_config_sha,
        "selected_config_value_sha256": stable_hash(selected_config),
        "selection_rule": identity["selection_rule"],
    }
    if selection_path.is_file():
        if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
            raise RuntimeError("Completed tuning selection differs from validated candidate journals")
    else:
        save_json(selection_path, selection)
    return selected_path
