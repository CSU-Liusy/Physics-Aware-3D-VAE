"""Portable CPU preparation and guards for the optional external synthetic case.

This staging module does not provide validated model inference. The run command
fails explicitly until the portable evaluator has passed a real ten-method
comparison. Inspect and saved-only verification never start a clock, download,
train, tune, load model tensors, or import the prediction runtime.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "conversion_reference.json"
PARENT = "porphyry_01_external_synthetic"
PRIMARY_SHA = "48ec0924b76a1cca26715bb77566646dacb5fe9a3252471f8e4690aae8e01030"
CORE_SHA = "2bbb423c62e8ca17cd8045d98cc62fcd123d0a0ade27c6dd2ef72cd52c91d5c3"
CORE = ("repro_pipeline.py", "experiment_data.py", "model.py", "repro_cgan.py",
        "repro_baselines.py", "repro_metrics.py", "comparisons/models/unet.py")
METHODS = ("idw", "rbf", "ik", "mps", "unet", "base_vae", "cgan", "geovae",
           "geovae_latent", "geovae_joint")
HEADER = ("x", "y", "z", "ton", "clays", "chalcocite", "bornite", "chalcopyrite",
          "tennantite", "molibdenite", "pyrite", "cu", "mo", "as", "rec", "bwi")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inside(root, relative):
    root = Path(root).resolve()
    result = (root / relative).resolve()
    require(result.is_relative_to(root), "Artifact path escapes its declared root")
    return result


def frozen_json(path, value):
    path = Path(path)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode()
    if path.exists():
        require(path.read_bytes() == payload, "Existing immutable record differs: " + path.name)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return sha(path)


def guard_output(out, code_root, sources=()):
    """Keep generated third-party assets out of this code tree and source files."""
    out, code = Path(out).resolve(), Path(code_root).resolve()
    require(not out.is_relative_to(code) and not code.is_relative_to(out),
            "Use a separate external data-cache directory outside the public code tree")
    require(not any(part.casefold() in {".git", "revision_article", "sn-article", "reply", "supplement"}
                    for part in out.parts), "Output cannot be a source-control or submission document directory")
    for source in sources:
        source = Path(source).resolve()
        require(not source.is_relative_to(out) and not out.is_relative_to(source.parent),
                "Preparation needs a separate cache outside each source directory")
    return out


def observation_functions(code_root, reference):
    path = Path(code_root).resolve() / "experiment_data.py"
    require(sha(path) == reference["observation_source_sha256"], "Observation source differs from the registered implementation")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"_stable_seed", "make_drill_masks"}]
    require({node.name for node in nodes} == {"_stable_seed", "make_drill_masks"},
            "Expected pure observation functions are missing")
    module = ast.Module(body=nodes, type_ignores=[])
    digest = hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()
    require(digest == reference["observation_ast_sha256"], "Observation function AST differs")
    namespace = {"np": np, "hashlib": hashlib}
    exec(compile(module, "registered_observation_functions", "exec"), namespace)
    return namespace["make_drill_masks"], {"source_sha256": sha(path), "functions_ast_sha256": digest}


def supported_cube(table, header=HEADER, target_size=32):
    """Pure conversion rule, independent of grade boundaries or model output."""
    table = np.asarray(table)
    require(table.ndim == 2 and table.shape[1] == len(header) and np.isfinite(table).all(),
            "Expected finite rectangular source table")
    require(tuple(header[:4]) == ("x", "y", "z", "ton") and "cu" in header, "Source columns differ")
    axes = [np.unique(table[:, index]) for index in range(3)]
    require(all(len(axis) > 1 for axis in axes), "Source requires at least two coordinates per axis")
    steps = np.array([axis[1] - axis[0] for axis in axes])
    require(np.all(steps > 0) and np.all(steps == steps[0]) and
            all(np.all(np.diff(axis) == steps[index]) for index, axis in enumerate(axes)),
            "Source requires an isotropic regular grid")
    shape = tuple(map(len, axes))
    require(len(table) == math.prod(shape), "Source Cartesian grid has missing or extra rows")
    indices = tuple(np.searchsorted(axes[index], table[:, index]) for index in range(3))
    linear = np.ravel_multi_index(indices, shape)
    require(len(np.unique(linear)) == len(table), "Duplicate source coordinates")
    cu, ton = np.empty(shape), np.empty(shape)
    cu[indices], ton[indices] = table[:, header.index("cu")], table[:, header.index("ton")]
    positive_layers = np.all(ton > 0, axis=(0, 1))
    segments, start = [], None
    for index, positive in enumerate(np.append(positive_layers, False)):
        if positive and start is None:
            start = index
        if not positive and start is not None:
            segments.append((start, index))
            start = None
    require(bool(segments), "No complete positive-tonnage full-XY layer interval")
    lo, hi = sorted(segments, key=lambda pair: (-(pair[1] - pair[0]), pair[0]))[0]
    retained = [axes[0], axes[1], axes[2][lo:hi]]
    low = np.array([axis[0] - steps[index] / 2 for index, axis in enumerate(retained)])
    high = np.array([axis[-1] + steps[index] / 2 for index, axis in enumerate(retained)])
    side, centre = float(np.min(high - low)), (low + high) / 2
    cube_low = centre - side / 2
    sample_axes = [cube_low[index] + (np.arange(target_size) + .5) * side / target_size
                   for index in range(3)]
    global_low = np.array([axis[0] - steps[index] / 2 for index, axis in enumerate(axes)])
    sample_indices = [np.floor((axis - global_low[index]) / steps[index]).astype(int)
                      for index, axis in enumerate(sample_axes)]
    require(all(np.all((index >= 0) & (index < shape[axis]))
                for axis, index in enumerate(sample_indices)), "Target centre lies outside source support")
    sampled_cu = cu[np.ix_(*sample_indices)]
    require(np.all(ton[np.ix_(*sample_indices)] > 0), "Unknown support cannot become barren labels")
    cube_indices = [np.flatnonzero((axis >= cube_low[index]) & (axis < cube_low[index] + side))
                    for index, axis in enumerate(axes)]
    native_cu, native_ton = cu[np.ix_(*cube_indices)], ton[np.ix_(*cube_indices)]
    require(np.all(native_ton > 0), "Native cube contains unknown support")
    back_indices = [np.clip(np.floor((axes[index][ids] - cube_low[index]) / (side / target_size)).astype(int),
                            0, target_size - 1) for index, ids in enumerate(cube_indices)]
    support = {"source_shape_xyz": list(shape), "retained_z_indices": [lo, hi],
               "cube_side_source_coordinate_units": side, "native_cube_shape_xyz": list(native_cu.shape),
               "target_shape_zyx": [target_size] * 3, "target_unknown_voxels": 0,
               "target_known_positive_ton_voxels": target_size ** 3,
               "physical_coordinate_units_verified": False,
               "rule": "longest contiguous full-XY positive-ton layer interval; lower-z tie break; centred maximum inscribed cube",
               "target_lookup": "source block containing each isotropic target voxel centre; no padding/smoothing",
               "is_complete_deposit": False}
    return sampled_cu, native_cu, back_indices, support


def prepare(source, license_file, code_root, out, accept_license=False):
    require(accept_license, "Read and explicitly accept the separate upstream CC BY-NC-SA 4.0 conditions")
    reference = read(REFERENCE)
    source, license_file = Path(source).resolve(), Path(license_file).resolve()
    out = guard_output(out, code_root, (source, license_file))
    require(source.stat().st_size == reference["source_bytes"] and sha(source) == reference["source_sha256"],
            "Wrong source size/hash; use the exact pinned commit, no automatic fallback")
    require(sha(license_file) == reference["license_sha256"], "Upstream license file differs from the pinned version")
    with source.open(encoding="utf-8") as stream:
        header = tuple(stream.readline().strip().split(","))
    require(header == HEADER, "Source header differs")
    table = np.loadtxt(source, delimiter=",", skiprows=1)
    sampled, native, back_indices, support = supported_cube(table, header)
    require(support["source_shape_xyz"] == [49, 71, 44] and support["retained_z_indices"] == [0, 41]
            and support["native_cube_shape_xyz"] == [41, 41, 41]
            and support["cube_side_source_coordinate_units"] == 820, "Source support differs from the fixed reviewed recipe")
    make_masks, function_identity = observation_functions(code_root, reference)
    recipe = {"format": "porphyry-public-prepared-v1", "source_kind": "external synthetic",
              "source_commit": reference["source_commit"], "source_sha256": reference["source_sha256"],
              "upstream_repository": "https://github.com/exepulveda/geomet_datasets",
              "source_path": reference["source_path"],
              "data_paper_doi": "10.1007/s11053-020-09692-6",
              "attribution": "Source repository exepulveda/geomet_datasets; dataset described by Garrido, Sepulveda, Ortiz and Townley (2020)",
              "license": "CC BY-NC-SA 4.0", "license_sha256": reference["license_sha256"],
              "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
              "modifications": "positive-ton supported cube; isotropic centre lookup to32cubed; fixed binary Cu exceedance levels and pseudo-drillhole observations",
              "conversion_reference_sha256": sha(REFERENCE), "preparation_source_sha256": sha(__file__),
              "parent_id": PARENT, "independent_parents": 1, "levels": reference["levels"],
              "mask_seeds": reference["mask_seeds"], "support": support,
              "observation_implementation": function_identity,
              "scope": "five nested Cu exceedance levels times three dependent masks; not field validation, economic ore boundaries or fifteen independent deposits",
              "model_calls": 0}
    # Freeze the recipe before creating any target array. No absolute source paths.
    recipe_sha = frozen_json(out / "preparation_identity.json", recipe)
    artifacts, targets, masks, conditions, representation = {}, {}, {}, [], []

    def save_array(relative, value):
        value = np.ascontiguousarray(value)
        expected = reference["reference_arrays"][relative]
        require(array_sha(value) == expected["array_sha256"], "Converted array differs from registered reference: " + relative)
        path = inside(out, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = np.load(path, allow_pickle=False)
            require(existing.dtype == value.dtype and existing.shape == value.shape and np.array_equal(existing, value),
                    "Existing converted array differs: " + relative)
        else:
            with path.open("wb") as stream:
                np.save(stream, value, allow_pickle=False)
        require(sha(path) == expected["file_sha256"], "Converted NPY bytes differ: " + relative)
        artifacts[relative] = {"file_sha256": sha(path), "array_sha256": array_sha(value),
                               "shape": list(value.shape), "dtype": str(value.dtype)}

    for seed in reference["mask_seeds"]:
        known, hidden = make_masks((32, 32, 32), PARENT, 8, 12, seed, 4)
        require(int(known.sum()) == 96 and int(hidden.sum()) == 48 and not np.any(known & hidden), "Drill masks differ")
        save_array(f"prepared/mask{seed}_observed.npy", known)
        save_array(f"prepared/mask{seed}_hidden.npy", hidden)
        masks[seed] = known, hidden
    for level in reference["levels"]:
        case = "porphyry_01_cu_ge_" + str(level).replace(".", "p")
        xyz = sampled >= level
        target = np.ascontiguousarray(xyz.transpose(2, 1, 0).astype(np.uint8))
        relative = f"prepared/{case}_target.npy"
        save_array(relative, target)
        targets[case] = {"cu_threshold_percent": level, "target_file": relative}
        native_mask, reconstructed = native >= level, xyz[np.ix_(*back_indices)]
        representation.append({"case_id": case, "cu_threshold_percent": level,
                               "native_positive_fraction": float(native_mask.mean()),
                               "target_positive_fraction": float(target.mean()),
                               "native_to_32_back_projected_label_disagreement": float(np.mean(native_mask != reconstructed)),
                               "meaning": "representation loss, not model error"})
        for seed, (known, hidden) in masks.items():
            observation = np.stack((target.astype(np.float32) * known, known.astype(np.float32)))
            obs_file, hidden_file = f"prepared/{case}_mask{seed}_observation.npy", f"prepared/{case}_mask{seed}_hidden_labels.npy"
            save_array(obs_file, observation)
            save_array(hidden_file, (target * hidden).astype(np.uint8))
            conditions.append({"case_id": case, "mask_seed": seed, "observation_file": obs_file,
                               "hidden_labels_file": hidden_file})
    require(set(artifacts) == set(reference["reference_arrays"]), "Incomplete conversion reference coverage")
    manifest = {"format": "porphyry-public-manifest-v1", "preparation_identity_sha256": recipe_sha,
                "parent_id": PARENT, "independent_parents": 1, "targets": targets,
                "conditions": conditions, "artifacts": artifacts, "representation": representation,
                "model_calls": 0}
    frozen_json(out / "manifest.json", manifest)
    return {"status": "prepared_cpu_only", "arrays": len(artifacts), "conditions": len(conditions),
            "independent_parents": 1, "manifest_sha256": sha(out / "manifest.json"), "model_calls": 0}


def verify_prepared(data_cache):
    cache, reference = Path(data_cache).resolve(), read(REFERENCE)
    recipe, manifest = read(cache / "preparation_identity.json"), read(cache / "manifest.json")
    require(manifest["preparation_identity_sha256"] == sha(cache / "preparation_identity.json"), "Preparation identity differs")
    require(recipe["source_sha256"] == reference["source_sha256"] and recipe["conversion_reference_sha256"] == sha(REFERENCE),
            "Prepared source/reference differs")
    require(recipe["levels"] == reference["levels"] and recipe["mask_seeds"] == reference["mask_seeds"]
            and recipe["license_sha256"] == reference["license_sha256"] and recipe["source_kind"] == "external synthetic"
            and recipe["source_commit"] == reference["source_commit"]
            and recipe["source_path"] == reference["source_path"]
            and recipe["parent_id"] == PARENT and recipe["independent_parents"] == 1 and recipe["model_calls"] == 0,
            "Prepared case settings differ")
    require(recipe["observation_implementation"] == {
                "source_sha256": reference["observation_source_sha256"],
                "functions_ast_sha256": reference["observation_ast_sha256"]}, "Prepared observation implementation differs")
    support = recipe["support"]
    require(support["source_shape_xyz"] == [49, 71, 44] and support["retained_z_indices"] == [0, 41]
            and support["cube_side_source_coordinate_units"] == 820 and support["native_cube_shape_xyz"] == [41, 41, 41]
            and support["target_shape_zyx"] == [32, 32, 32] and support["target_unknown_voxels"] == 0
            and support["target_known_positive_ton_voxels"] == 32768
            and support["physical_coordinate_units_verified"] is False and support["is_complete_deposit"] is False,
            "Prepared support rule/evidence differs")
    require(recipe["preparation_source_sha256"] == sha(__file__), "Prepared implementation differs")
    require(manifest["parent_id"] == PARENT and manifest["independent_parents"] == 1 and manifest["model_calls"] == 0
            and len(manifest["targets"]) == 5
            and len(manifest["conditions"]) == 15 and len(manifest["artifacts"]) == 41, "Incomplete prepared coverage")
    require(set(manifest["artifacts"]) == set(reference["reference_arrays"]), "Prepared array inventory differs")
    expected_cases = {"porphyry_01_cu_ge_" + str(level).replace(".", "p") for level in reference["levels"]}
    require(set(manifest["targets"]) == expected_cases
            and {(row["case_id"], row["mask_seed"]) for row in manifest["conditions"]}
            == {(case, seed) for case in expected_cases for seed in reference["mask_seeds"]},
            "Prepared case/mask identities differ")
    for level in reference["levels"]:
        case = "porphyry_01_cu_ge_" + str(level).replace(".", "p")
        require(manifest["targets"][case] == {"cu_threshold_percent": level, "target_file": f"prepared/{case}_target.npy"},
                "Prepared target role/path differs")
    for row in manifest["conditions"]:
        case, seed = row["case_id"], row["mask_seed"]
        require(row == {"case_id": case, "mask_seed": seed,
                        "observation_file": f"prepared/{case}_mask{seed}_observation.npy",
                        "hidden_labels_file": f"prepared/{case}_mask{seed}_hidden_labels.npy"},
                "Prepared observation/hidden-label role/path differs")
    for relative, recorded in manifest["artifacts"].items():
        path = inside(cache, relative)
        value = np.load(path, allow_pickle=False)
        expected = reference["reference_arrays"][relative]
        require(sha(path) == recorded["file_sha256"] == expected["file_sha256"]
                and array_sha(value) == recorded["array_sha256"] == expected["array_sha256"]
                and list(value.shape) == recorded["shape"] and str(value.dtype) == recorded["dtype"],
                "Prepared array differs: " + relative)
    return {"manifest_sha256": sha(cache / "manifest.json"), "arrays": 41, "conditions": 15}


def public_primary(run_root, code_root, require_time=True, now=None):
    """Small metadata guard; weight tensors and predictions are not loaded here."""
    run, code = Path(run_root).resolve(), Path(code_root).resolve()
    identity, handoff = read(run / "run_identity.json"), read(run / "public_handoff.json")
    require(identity.get("public_mode") == "synthetic_only" and identity.get("data_root") == "not_available_restricted",
            "Only an independently created public-synthetic lifecycle is eligible; internal author runs are refused")
    require(identity.get("mode") == "formal" and identity.get("limit") == 0
            and identity.get("manifest_sha256") == PRIMARY_SHA and identity.get("source_sha256") == CORE_SHA, "Wrong public primary identity")
    require(sha(code / "protocol/manifest.json") == PRIMARY_SHA
            and sha(run / "frozen_inputs/manifest.json") == PRIMARY_SHA, "Primary manifest differs")
    digest = hashlib.sha256()
    for name in CORE:
        digest.update(name.encode())
        digest.update(sha(code / name).encode("ascii"))
    require(digest.hexdigest() == CORE_SHA, "Primary training core differs")
    require(bool(identity.get("code_files")), "Primary source inventory missing")
    for relative, expected in identity["code_files"].items():
        require(sha(inside(code, relative)) == expected == sha(inside(run / "frozen_inputs", relative)), "Primary frozen source differs")
    require(handoff.get("status") == "public_synthetic_reproduction_complete"
            and handoff.get("source_sha256") == CORE_SHA and handoff.get("manifest_sha256") == PRIMARY_SHA
            and handoff.get("raw_real_audit_passed") is False, "Public-synthetic finalization is missing or mismatched")
    required_handoff = {"revision_analysis/summary.csv", "revision_analysis/analysis.json",
                        "revision_analysis/paired_differences.csv", "revision_analysis/reliability.csv",
                        "ablation/summaries/test/ablation_summary.csv", "uncertainty/test/per_case_acceptance.csv",
                        "revision_robustness/summary.csv", "revision_robustness/status.json"}
    require(set(handoff.get("key_files", {})) == required_handoff, "Public handoff inventory is incomplete")
    for relative, expected in handoff["key_files"].items():
        require(sha(inside(run, relative)) == expected, "Public handoff artifact changed")
    config, tuning = read(run / "tuning/selected_config.json"), read(run / "tuning/selection.json")
    thresholds, budget = read(run / "thresholds.json"), read(run / "revision_budget.json")
    require(tuning.get("status") == "validation_selected" and tuning.get("manifest_sha256") == PRIMARY_SHA
            and tuning.get("selected_config_value_sha256") == stable(config), "Primary configuration is not validation-frozen")
    require(config.get("latent_dim") == 256 and config.get("grid_size") == [32, 32, 32]
            and config.get("num_holes") == 8 and config.get("samples_per_hole") == 12, "Wrong primary recipe")
    require(thresholds.get("status") == "validation_frozen" and thresholds.get("manifest_sha256") == PRIMARY_SHA
            and thresholds.get("evaluation_source_sha256") == CORE_SHA and thresholds.get("config_sha256") == stable(config)
            and set(thresholds.get("thresholds", {})) == set(METHODS)
            and all(math.isfinite(float(value)) and 0 < float(value) < 1 for value in thresholds["thresholds"].values()),
            "Primary thresholds are not frozen")
    for seed in (11, 23, 37):
        for split in ("validation", "test"):
            meta = read(run / split / f"seed{seed}/evaluation.json")
            require(meta.get("complete") is True and meta.get("expected_cases") == 750
                    and meta.get("source_sha256") == CORE_SHA and meta.get("manifest_sha256") == PRIMARY_SHA
                    and meta.get("config_sha256") == stable(config) and meta.get("seed") == seed
                    and set(meta.get("methods", [])) == set(METHODS), "Full primary evaluation is incomplete")
            if split == "test":
                require(meta.get("threshold_selection_sha256") == sha(run / "thresholds.json"), "Primary test threshold identity differs")
    for kind in ("geovae", "base_vae", "unet", "cgan"):
        require((run / "models" / kind / "seed11/best.pt").is_file(), "Primary seed11 checkpoint is absent")
    require(math.isfinite(float(budget["started_unix"])) and math.isfinite(float(budget["hours"])) and budget["hours"] > 0,
            "Invalid immutable parent budget")
    deadline = budget["started_unix"] + budget["hours"] * 3600
    if require_time:
        require((time.time() if now is None else now) < deadline, "Parent experimental budget has expired")
    return {"verified_public_lifecycle": True,
            "run_identity_sha256": sha(run / "run_identity.json"), "public_handoff_sha256": sha(run / "public_handoff.json"),
            "config_sha256": stable(config), "thresholds_sha256": sha(run / "thresholds.json"),
            "parent_budget_sha256": sha(run / "revision_budget.json"), "parent_deadline_unix": deadline,
            "checkpoint_tensors_loaded": False, "model_calls": 0}


def supplemental_budget(out, parent, hours, now=None):
    """Independent public lifecycle only; resumes cannot replace or extend it.

    This utility is not called by the disabled run entry point. It deliberately
    has no path to an internal lifecycle, and does not modify the parent clock.
    """
    require(parent.get("verified_public_lifecycle") is True, "A verified public lifecycle is required")
    require(math.isfinite(float(hours)) and 0 < hours <= 72, "Supplement budget must be finite and in (0,72] hours")
    moment = float(time.time() if now is None else now)
    require(math.isfinite(moment), "Invalid current time")
    path = Path(out).resolve() / "public_supplement_budget.json"
    base = {"origin": "public_reproduction", "hours": float(hours),
            "parent_run_identity_sha256": parent["run_identity_sha256"],
            "parent_handoff_sha256": parent["public_handoff_sha256"]}
    if path.exists():
        value = read(path)
        require(all(value.get(key) == item for key, item in base.items()), "Existing supplemental lifecycle differs; clock cannot be reset")
    else:
        require(not Path(out).exists() or not any(Path(out).iterdir()), "New supplemental lifecycle requires an empty output directory")
        value = {**base, "started_unix": moment}
        frozen_json(path, value)
    require(math.isfinite(float(value["started_unix"])), "Invalid immutable start time")
    deadline = value["started_unix"] + value["hours"] * 3600
    require(moment < deadline, "Supplemental experimental budget has expired")
    return {**value, "deadline_unix": deadline}


def release_display():
    """Show the checked release registry without granting execution permission.

    Read-only commands remain usable if release evidence is absent, stale or
    negative. A display failure closes the displayed entry; the production run
    independently applies the strict release guard before any prediction.
    """
    try:
        from release_contract import release_evidence
        return release_evidence()
    except (ValueError, OSError, KeyError, TypeError, AttributeError, RuntimeError, ImportError) as error:
        return {"status": "blocked_invalid_release_evidence",
                "prediction_entry_enabled": False, "error": str(error)}


def inspect(run_root, code_root, data_cache, out=None):
    """Inspect both public parent and prepared case, without creating outputs.

    An optional existing/proposed output enables the backend's stricter context
    binding. It may hash checkpoint files and read the one primary MPS training
    image, but never loads model tensors or invokes a predictor. A completed
    public parent's expired clock does not prevent read-only inspection.
    """
    blockers, evidence = [], {}
    for name, operation in (("primary", lambda: public_primary(run_root, code_root, require_time=False)),
                            ("prepared", lambda: verify_prepared(data_cache))):
        try:
            evidence[name] = operation()
        except (ValueError, OSError, KeyError, TypeError) as error:
            blockers.append(f"{name}: {error}")
    if out is not None and not blockers:
        try:
            from portable_evaluation import context
            ctx = context(run_root, code_root, data_cache, out, require_time=False)
            evidence["evaluation_context"] = {
                "conditions": len(ctx["data"]["conditions"]),
                "external_manifest_sha256": ctx["data"]["manifest_sha256"],
                "checkpoint_sha256": ctx["checkpoint_sha256"],
                "training_image_id": ctx["training_image_id"],
                "training_image_file_sha256": ctx["training_image_file_sha256"],
                "primary_prerequisite_sha256": ctx["primary_prerequisite_sha256"],
                "model_tensors_loaded": False, "clock_initialized": False,
            }
        except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as error:
            blockers.append(f"evaluation_context: {error}")
    release = release_display()
    released = release["prediction_entry_enabled"]
    # Prerequisite checks and recorded release validation are separate evidence.
    status = ("blocked_missing_or_mismatched_prerequisites" if blockers else
              "ready_released_prerequisites" if released else "blocked_portable_evaluator_not_validated")
    return {"status": status,
            "prerequisite_blockers": blockers, "evidence": evidence,
            "prerequisites_ready": not blockers, "release_evidence": release,
            "portable_ten_method_parity_test_passed": released,
            "numerical_validation_performed_by_this_inspection": False, "model_calls": 0,
            "prediction_entry_enabled": released}


def verify_only(run_root, code_root, data_cache, out):
    """Recheck existing matching public results; no new clock or predictions.

    The backend revalidates the current public parent, wrapper, prepared arrays,
    actual checkpoint-file hashes and stored supplemental budget before it
    recomputes saved probabilities. It can update audit records in the existing
    result directory; no missing probability is generated or repaired.
    """
    out = Path(out).resolve()
    required = ("identity.json", "public_supplement_budget.json", "status.json",
                "per_case.csv", "failures.json", "failure_history.json")
    require(out.is_dir() and all((out / name).is_file() for name in required),
            "Existing saved public evaluation and immutable budget are required; no output or clock is created")
    identity = read(out / "identity.json")
    require(identity.get("format") == "porphyry-public-evaluation-v1"
            and identity.get("formal") is True and identity.get("origin") == "public_reproduction",
            "Saved-only verification accepts matching formal public results, not internal or fake fixtures")
    from portable_evaluation import audit_saved
    path = audit_saved(run_root, code_root, data_cache, out)
    audit = read(path)
    require(audit.get("status") == "passed_saved_probability_recomputation"
            and audit.get("formal") is True and audit.get("rows") == 150
            and audit.get("no_prediction_or_weight_loading") is True,
            "Saved-only audit did not establish complete matching public results")
    release = release_display()
    return {"status": "verified_saved_public_results", "rows": 150,
            "saved_audit_sha256": sha(path), "model_calls": 0,
            "model_tensors_loaded": False, "clock_initialized": False,
            "release_evidence": release,
            "prediction_entry_enabled": release["prediction_entry_enabled"]}


def blocked_prediction():
    """No caller flag or environment setting enables this release's prediction."""
    return {"status": "blocked_portable_evaluator_not_validated",
            "reason": "Real ten-method numerical parity and explicit release review are still required",
            "model_calls": 0, "model_tensors_loaded": False,
            "clock_initialized": False, "prediction_entry_enabled": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="CPU conversion only; explicit local source and separate cache")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--license-file", type=Path, required=True)
    prep.add_argument("--code-root", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--accept-data-license", action="store_true")
    for name in ("inspect", "run"):
        command = commands.add_parser(name, help="Metadata and release-evidence inspection" if name == "inspect" else "Prediction gated by source-bound numerical and review evidence")
        command.add_argument("--primary-run", type=Path, required=True)
        command.add_argument("--code-root", type=Path, required=True)
        command.add_argument("--data-cache", type=Path, required=True)
        command.add_argument("--out", type=Path, help="Existing/proposed separate supplemental output; run requires checked release evidence")
    verify = commands.add_parser("verify-only", aliases=["verify"],
                                 help="Recompute only saved matching public results; never starts a clock or inference")
    verify.add_argument("--primary-run", type=Path, required=True)
    verify.add_argument("--code-root", type=Path, required=True)
    verify.add_argument("--data-cache", type=Path, required=True)
    verify.add_argument("--out", type=Path, required=True)
    bundle = commands.add_parser("import-author-bundle", help="Create a read-only provenance descriptor; no model loading")
    bundle.add_argument("--primary-run", type=Path, required=True)
    bundle.add_argument("--code-root", type=Path, required=True)
    bundle.add_argument("--out", type=Path, required=True)
    for name in ("integration-inspect", "integration-check", "verify-integration", "verify-reference"):
        command = commands.add_parser(name, help="Registered software subset or saved-only author-reference verification")
        for key in ("primary-run", "code-root", "data-cache", "author-bundle", "out"):
            command.add_argument("--" + key, type=Path, required=True)
        if name == "verify-reference":
            command.add_argument("--reference-run", type=Path, required=True)
            command.add_argument("--external-manifest", type=Path, required=True)
        else:
            command.add_argument("--registration", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.source, args.license_file, args.code_root, args.out, args.accept_data_license)
        elif args.command == "run":
            from release_contract import release_evidence
            if not release_evidence()["prediction_entry_enabled"]:
                result = blocked_prediction()
            else:
                require(args.out is not None, "A separate explicit output directory is required")
                from portable_evaluation import run
                path = run(args.primary_run, args.code_root, args.data_cache, args.out)
                result = {"status": "verified_saved_public_results", "audit_sha256": sha(path)}
        elif args.command == "import-author-bundle":
            from author_bundle import create_bundle
            result = create_bundle(args.primary_run, args.code_root, args.out)
        elif args.command == "verify-reference":
            from author_reference import import_reference
            result = import_reference(args.author_bundle, args.primary_run, args.code_root,
                                      args.data_cache, args.reference_run, args.external_manifest, args.out)
        elif args.command in {"integration-inspect", "integration-check", "verify-integration"}:
            from portable_evaluation import context, _run_after_release, audit_saved
            kwargs = {"author_bundle": args.author_bundle, "registration_path": args.registration}
            if args.command == "integration-inspect":
                ctx = context(args.primary_run, args.code_root, args.data_cache, args.out, **kwargs)
                result = {"status": "ready_registered_subset_inspection", "execution_scope": ctx["execution_scope"],
                          "model_calls": 0, "checkpoint_tensors_loaded": False, "clock_initialized": False}
            else:
                action = _run_after_release if args.command == "integration-check" else audit_saved
                path = action(args.primary_run, args.code_root, args.data_cache, args.out, **kwargs)
                result = {"status": "verified_saved_integration_subset", "rows": 10,
                          "audit_sha256": sha(path), "full150_execution_claimed": False}
        elif args.command in {"verify-only", "verify"}:
            result = verify_only(args.primary_run, args.code_root, args.data_cache, args.out)
        else:
            result = inspect(args.primary_run, args.code_root, args.data_cache, args.out)
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
        return 0 if result["status"] in {"prepared_cpu_only", "verified_saved_public_results", "author_bundle_descriptor_created", "ready_released_prerequisites",
            "ready_registered_subset_inspection", "verified_saved_integration_subset", "passed_author_reference_import"} else 2
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(json.dumps({"status": "blocked_no_inference", "error": str(error), "model_calls": 0}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
