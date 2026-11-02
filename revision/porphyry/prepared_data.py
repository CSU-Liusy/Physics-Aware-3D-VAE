"""Strict portable prepared-data adapter; CPU only, never a predictor.

The returned target, known/hidden masks and labels are for scoring. A model must
receive only ``predictor_observation(case)`` plus frozen model/config/TI/seed.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from porphyry_public import (PARENT, array_sha, inside, read, require, sha,
                             stable, verify_prepared)


def _verified_json(path, expected_sha):
    payload = Path(path).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == expected_sha, "Prepared JSON changed after verification")
    return json.loads(payload.decode("utf-8"))


def _verified_array(path, record):
    # Hash and parse the same bytes so a second file read cannot evade binding.
    payload = Path(path).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == record["file_sha256"], "Prepared array bytes changed after verification")
    value = np.load(io.BytesIO(payload), allow_pickle=False)
    require(array_sha(value) == record["array_sha256"] and list(value.shape) == record["shape"]
            and str(value.dtype) == record["dtype"], "Loaded prepared array differs from its verified identity")
    return value


def load_prepared(data_cache):
    """Return all fifteen explicit condition cases after complete hash/role checks.

    No model, tensor library, network or primary data are accessed. The output
    dictionary is deliberately distinct from the internal author manifest;
    callers must record these public preparation and manifest identities.
    """
    cache = Path(data_cache).resolve()
    verification = verify_prepared(cache)
    manifest = _verified_json(cache / "manifest.json", verification["manifest_sha256"])
    recipe = _verified_json(cache / "preparation_identity.json", manifest["preparation_identity_sha256"])
    require(verification["arrays"] == 41 and verification["conditions"] == 15,
            "Incomplete portable prepared verification")

    loaded = {}

    def load(relative):
        require(relative in manifest["artifacts"], "Prepared array is outside the verified inventory")
        if relative not in loaded:
            loaded[relative] = _verified_array(inside(cache, relative), manifest["artifacts"][relative])
        return loaded[relative]

    masks = {}
    for seed in recipe["mask_seeds"]:
        known = load(f"prepared/mask{seed}_observed.npy")
        hidden = load(f"prepared/mask{seed}_hidden.npy")
        require(known.dtype == hidden.dtype == np.bool_ and known.shape == hidden.shape == (32, 32, 32),
                "Mask dtype/shape differs")
        require(int(known.sum()) == 96 and int(hidden.sum()) == 48 and not np.any(known & hidden),
                "Observed/hidden masks are not disjoint or have wrong counts")
        masks[seed] = known, hidden

    targets = {}
    for case_id, row in manifest["targets"].items():
        target = load(row["target_file"])
        require(target.dtype == np.uint8 and target.shape == (32, 32, 32)
                and np.isin(target, [0, 1]).all(), "Binary target dtype/shape/labels differ")
        targets[case_id] = target
    ordered = sorted(manifest["targets"], key=lambda case_id: manifest["targets"][case_id]["cu_threshold_percent"])
    require(all(np.all(targets[higher] <= targets[lower]) for lower, higher in zip(ordered, ordered[1:])),
            "Cu exceedance targets are not nested")

    conditions, seen = [], set()
    for row in manifest["conditions"]:
        case_id, seed = row["case_id"], row["mask_seed"]
        key = case_id, seed
        require(key not in seen and case_id in targets and seed in masks, "Duplicate/unknown condition")
        seen.add(key)
        target = targets[case_id]
        known, hidden = masks[seed]
        obs, hidden_labels = load(row["observation_file"]), load(row["hidden_labels_file"])
        require(obs.dtype == np.float32 and obs.shape == (2, 32, 32, 32)
                and np.isfinite(obs).all() and np.array_equal(obs[1], known)
                and np.array_equal(obs[0], target * known), "Observation encoding differs or exposes unobserved labels")
        require(hidden_labels.dtype == np.uint8 and hidden_labels.shape == target.shape
                and np.array_equal(hidden_labels, target * hidden), "Hidden scoring labels differ")
        binding = {"parent_id": PARENT, "manifest_sha256": verification["manifest_sha256"],
                   "preparation_identity_sha256": manifest["preparation_identity_sha256"],
                   "case_id": case_id, "mask_seed": seed,
                   "cu_threshold_percent": manifest["targets"][case_id]["cu_threshold_percent"],
                   "observation_array_sha256": array_sha(obs),
                   "observed_mask_array_sha256": array_sha(known),
                   "hidden_mask_array_sha256": array_sha(hidden),
                   "target_array_sha256": array_sha(target)}
        conditions.append({"case_id": case_id, "parent_id": PARENT, "mask_seed": seed,
                           "cu_threshold_percent": binding["cu_threshold_percent"],
                           "target": target, "obs": obs, "known": known, "hidden": hidden,
                           "observation_identity_sha256": stable(binding),
                           "prepared_condition_binding": binding})
    require(seen == {(case_id, seed) for case_id in targets for seed in recipe["mask_seeds"]}
            and len(conditions) == 15, "All fifteen dependent conditions are required")
    return {"manifest": manifest, "preparation_identity": recipe,
            "manifest_sha256": verification["manifest_sha256"],
            "preparation_identity_sha256": sha(cache / "preparation_identity.json"),
            "loader_source_sha256": sha(__file__), "verification": verification,
            "conditions": sorted(conditions, key=lambda case: (case["cu_threshold_percent"], case["mask_seed"])),
            "independent_parents": 1, "model_calls": 0}


def predictor_observation(case):
    """A separate owned observation array, with no target/hidden-label argument."""
    value = np.asarray(case["obs"])
    require(value.dtype == np.float32 and value.shape == (2, 32, 32, 32)
            and np.isfinite(value).all(), "Invalid predictor observation")
    return np.array(value, dtype=np.float32, order="C", copy=True)


def project_for_diagnostic(raw_probability, case):
    """Return a diagnostic projection without changing raw probability or input."""
    raw = np.asarray(raw_probability)
    require(raw.shape == (32, 32, 32) and np.issubdtype(raw.dtype, np.floating)
            and np.isfinite(raw).all() and np.all((raw >= 0) & (raw <= 1)), "Invalid raw probability")
    known, obs = case["known"], case["obs"]
    require(known.dtype == np.bool_ and known.shape == raw.shape and obs.shape == (2, *raw.shape)
            and np.array_equal(obs[1], known), "Projection mask/observation differs")
    projected = raw.copy()
    projected[known] = obs[0][known]
    return projected
