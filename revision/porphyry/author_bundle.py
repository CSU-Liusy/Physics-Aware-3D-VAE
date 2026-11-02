"""Read-only import contract for author checkpoints trained on pure synthetic data.

The small bundle is a provenance descriptor, not a renamed public training run.
The caller supplies the original artifact tree separately. No weights are loaded,
copied, repacked or published here, and restricted real data are never opened.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time

from porphyry_public import CORE, CORE_SHA, PRIMARY_SHA, METHODS, frozen_json, inside, read, require, sha, stable

KINDS = ("geovae", "base_vae", "unet", "cgan")
FORMAT = "author-synthetic-checkpoint-bundle-v1"
START = 1792914117.752363
HOURS = 72.0
DEADLINE = START + HOURS * 3600
NUMERICS_SHA = "d1d2f19fdaff27a7d83507fb10f93e61e9ae54e85768443366058153e2f343fc"


def author_evidence(primary_run, code_root, *, require_time=False):
    """Resolve authoritative files, never a caller-supplied verified dictionary.

    This rechecks small metadata and the four checkpoint byte hashes. It does
    not repeat raw-data/5000-grid audits or deserialize checkpoint tensors.
    The original main audit is mandatory and its original file digest retained.
    """
    run, code = Path(primary_run).resolve(), Path(code_root).resolve()
    names = ("run_identity.json", "revision_status.json", "final_integrity/report.json",
             "tuning/selected_config.json", "tuning/selection.json", "thresholds.json", "revision_budget.json")
    docs = {name: read(inside(run, name)) for name in names}
    ri, state, integrity = (docs[name] for name in names[:3])
    config, tuning, thresholds, budget = (docs[name] for name in names[3:])
    require(ri.get("mode") == "formal" and ri.get("limit") == 0
            and ri.get("manifest_sha256") == PRIMARY_SHA and ri.get("source_sha256") == CORE_SHA,
            "Wrong original author primary identity")
    require(ri.get("public_mode") is None and state.get("status") == "experiments_complete",
            "Author import requires the unchanged completed author lifecycle")
    require(integrity.get("status") == "passed" and integrity.get("formal") is True
            and integrity.get("source_sha256") == CORE_SHA and integrity.get("manifest_sha256") == PRIMARY_SHA
            and integrity.get("audit_source_sha256") == sha(code / "revision_integrity.py"),
            "The complete original formal integrity audit has not passed")
    require(sha(code / "protocol/manifest.json") == sha(run / "frozen_inputs/manifest.json") == PRIMARY_SHA,
            "Frozen pure-synthetic primary manifest differs")
    sources = ri.get("code_files", {})
    require(len(sources) == 20 and all(sha(inside(run / "frozen_inputs", name)) == expected
            for name, expected in sources.items()),
            "Original twenty-source archive differs")
    digest = hashlib.sha256()
    for name in CORE:
        digest.update(name.encode()); digest.update(sha(code / name).encode("ascii"))
    require(digest.hexdigest() == CORE_SHA and sha(code / "revision_numerics.py") == NUMERICS_SHA,
            "Core or strict numerical helper source differs")
    require(tuning.get("status") == "validation_selected" and tuning.get("manifest_sha256") == PRIMARY_SHA
            and tuning.get("selected_config_value_sha256") == stable(config), "Configuration is not validation-selected")
    require(config.get("latent_dim") == 256 and config.get("grid_size") == [32, 32, 32]
            and config.get("num_holes") == 8 and config.get("samples_per_hole") == 12, "Primary recipe differs")
    require(thresholds.get("status") == "validation_frozen" and thresholds.get("manifest_sha256") == PRIMARY_SHA
            and thresholds.get("evaluation_source_sha256") == CORE_SHA and thresholds.get("config_sha256") == stable(config)
            and set(thresholds.get("thresholds", {})) == set(METHODS)
            and all(math.isfinite(float(v)) and 0 < float(v) < 1 for v in thresholds["thresholds"].values()),
            "Main probability thresholds are not frozen")
    require(budget.get("started_unix") == START and budget.get("hours") == HOURS,
            "Original author clock differs; no reset or extension is permitted")
    if require_time:
        require(time.time() < DEADLINE, "Original author experimental deadline expired")
    artifacts = {name: sha(inside(run, name)) for name in names}
    tests = {}
    for seed in (11, 23, 37):
        for split in ("validation", "test"):
            name = f"{split}/seed{seed}/evaluation.json"
            meta = read(run / name)
            require(meta.get("complete") is True and meta.get("expected_cases") == 750
                    and meta.get("config_sha256") == stable(config) and meta.get("manifest_sha256") == PRIMARY_SHA
                    and meta.get("source_sha256") == CORE_SHA and meta.get("seed") == seed
                    and set(meta.get("methods", [])) == set(METHODS), "Full main evaluation missing")
            if split == "test":
                require(meta.get("threshold_selection_sha256") == artifacts["thresholds.json"], "Main test threshold differs")
                tests[seed] = meta
            artifacts[name] = sha(run / name)
    intent_name = "test/seed11/evaluation_intent.json"
    intent = read(run / intent_name); artifacts[intent_name] = sha(run / intent_name)
    checkpoints = {kind: sha(run / "models" / kind / "seed11/best.pt") for kind in KINDS}
    require(checkpoints == tests[11].get("checkpoints") == intent.get("checkpoints")
            and intent.get("manifest_sha256") == PRIMARY_SHA and intent.get("source_sha256") == CORE_SHA
            and intent.get("config_sha256") == stable(config), "Main seed11 checkpoint or intent binding differs")
    manifest = read(code / "protocol/manifest.json")
    samples = manifest["samples"]
    require(len(samples) == 5000 and len({r["sample_id"] for r in samples}) == 5000, "Primary sample inventory differs")
    # Only metadata: no repeat reading of the 5000 grids.
    counts = {split: sum(r["split"] == split for r in samples) for split in ("train", "val", "test")}
    require(counts == {"train": 3500, "val": 750, "test": 750}, "Pure-synthetic split counts differ")
    require(manifest.get("protocol_version") == "procedural-lens-v1"
            and manifest.get("parent_counts") == {"train": 700, "val": 150, "test": 150}, "Synthetic parent protocol differs")
    train = [r for r in samples if r["split"] == "train"]
    ti_id = str(config.get("mps_training_image_id") or train[0]["sample_id"])
    ti_rows = [r for r in train if r["sample_id"] == ti_id]
    require(len(ti_rows) == 1 and tests[11].get("mps_training_image_id") == intent.get("mps_training_image_id") == ti_id,
            "MPS image is not uniquely training-only")
    ti = ti_rows[0]
    require(sha(inside(code / "protocol", ti["path"])) == ti["sha256"], "Training-only TI file changed")
    import numpy as np
    training_image = np.asarray(np.load(inside(code / "protocol", ti["path"]), allow_pickle=False), dtype=np.float32)
    require(training_image.shape == (32, 32, 32) and np.isin(training_image, [0, 1]).all(), "Training image is not binary32cubed")
    ti_array_sha = hashlib.sha256(np.ascontiguousarray(training_image).tobytes()).hexdigest()
    for key in ("mps_training_image_sha256", "mps_training_image_array_sha256"):
        if intent.get(key) is not None:
            require(intent[key] == ti_array_sha, "Primary intent training image bytes differ")
    return {"format": FORMAT, "source_kind": "author_synthetic_checkpoint_bundle",
            "origin": "imported_author_synthetic_checkpoints", "public_training_lifecycle_claimed": False,
            "source_sha256": CORE_SHA, "manifest_sha256": PRIMARY_SHA, "sample_counts": counts,
            "original_artifact_sha256": artifacts, "original_code_files": sources,
            "config_sha256": stable(config), "thresholds_sha256": artifacts["thresholds.json"],
            "checkpoint_sha256": checkpoints, "model_kinds": list(KINDS), "model_seed": 11,
            "training_image_id": ti_id, "training_image_file_sha256": ti["sha256"],
            "training_image_array_sha256": ti_array_sha,
            "original_budget": {"started_unix": START, "hours": HOURS, "deadline_unix": DEADLINE,
                                "file_sha256": artifacts["revision_budget.json"]},
            "checkpoint_tensors_loaded": False, "checkpoint_repacked": False,
            "loader_contract": "Unchanged model-specific strict loader, including extended cGAN source signature, is required at runtime.",
            "disclosure": "Descriptor imports original pure-synthetic model artifacts. No real-case data or adapted weights are included."}


def create_bundle(primary_run, code_root, bundle_path):
    """CPU metadata export only after main final integrity; immutable descriptor."""
    path = Path(bundle_path).resolve()
    for source in (primary_run, code_root):
        require(not path.is_relative_to(Path(source).resolve()), "Bundle descriptor must be separate from author/core sources")
    evidence = author_evidence(primary_run, code_root)
    frozen_json(path, evidence)
    return {"status": "author_bundle_descriptor_created", "bundle_sha256": sha(path),
            "model_calls": 0, "checkpoint_tensors_loaded": False, "public_weights_exported": False}


def verify_bundle(bundle_path, primary_run, code_root, *, require_time=False):
    path = Path(bundle_path).resolve()
    saved = read(path)
    require(saved == author_evidence(primary_run, code_root, require_time=require_time),
            "Author bundle differs from current authoritative original artifacts")
    return {"verified_author_bundle": True, "origin": saved["origin"],
            "bundle_sha256": sha(path), "run_identity_sha256": saved["original_artifact_sha256"]["run_identity.json"],
            "config_sha256": saved["config_sha256"], "thresholds_sha256": saved["thresholds_sha256"],
            "parent_budget_sha256": saved["original_budget"]["file_sha256"],
            "parent_deadline_unix": DEADLINE, "evidence": saved,
            "checkpoint_tensors_loaded": False, "model_calls": 0}
