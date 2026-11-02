"""Existing-PLY data protocol for the R1.2 inference diagnostic.

Raw data are never modified. The default cohort is the ``ore_surface`` virtual
family, whose coordinate frame is comparable across files. Other virtual
families and real meshes remain in the inventory with explicit exclusion
reasons. In particular, real face-count bins are not geological parent IDs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from experiment_data import DOMAIN
from repro_pipeline import save_json, sha256_file, stable_hash

PROTOCOL_VERSION = "r12-ply-v1"
PREPARATION_VERSION = 1
FAMILY_PATTERN = re.compile(r"^ore_surface_(\d+)\.ply$", re.I)
LEGACY_PATTERN = re.compile(r"^generate_mine_([a-z]+)(\d+)\.ply$", re.I)


def family_and_index(name: str) -> tuple[str, str]:
    match = FAMILY_PATTERN.fullmatch(name)
    if match:
        return "ore_surface", str(int(match.group(1)))
    match = LEGACY_PATTERN.fullmatch(name)
    if match:
        return "generate_mine_" + match.group(1).lower(), str(int(match.group(2)))
    return "unrecognized", Path(name).stem


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def inventory(data_root: Path) -> list[dict[str, Any]]:
    rows = []
    for directory in ("mining_ply_generate", "mining_ply_generate_pretrain", "mining_ply"):
        folder = data_root / directory
        if not folder.is_dir():
            if directory == "mining_ply_generate":
                raise FileNotFoundError(f"Virtual PLY folder is missing: {folder}")
            continue
        for path in sorted(folder.rglob("*.ply")):
            family, index = family_and_index(path.name)
            real = directory == "mining_ply"
            rows.append({
                "path": path.relative_to(data_root).as_posix(),
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "source_directory": directory,
                "source_kind": "real_interpreted_mesh" if real else "virtual_generated_mesh",
                "family": "real_face_count_bin" if real else family,
                "numeric_id": index,
                "real_parent_id": None,
                "lineage": "real mesh, face-count bin is not a parent" if real else
                           "virtual as declared by author; generating ancestry is not verified",
            })
            if len(rows) % 1000 == 0:
                print(f"[data] inventoried and hashed {len(rows)} source PLY files", flush=True)
    print(f"[data] source inventory complete: {len(rows)} PLY files", flush=True)
    return rows


def group_virtual_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse raw aliases and bind identical numeric IDs before splitting.

    The grouping is conservative filename/hash bookkeeping, not a claim of
    independence of geological or generative parents.
    """
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def union(a: str, b: str) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    eligible = [row for row in rows if row["source_kind"] == "virtual_generated_mesh"
                and row["family"] == "ore_surface"]
    for row in eligible:
        union("id:" + row["numeric_id"], "sha:" + row["sha256"])
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_hash[row["sha256"]].append(row)
    groups: dict[str, list[str]] = defaultdict(list)
    for key in parent:
        groups[find(key)].append(key)
    group_ids = {root: "virtual-alias-" + stable_hash(sorted(tokens))[:20]
                 for root, tokens in groups.items()}
    cases = []
    for digest, aliases in sorted(by_hash.items()):
        aliases.sort(key=lambda row: (row["source_directory"] != "mining_ply_generate", row["path"]))
        row = dict(aliases[0])
        row["aliases"] = [item["path"] for item in aliases]
        row["parent_id"] = group_ids[find("sha:" + digest)]
        row["sample_id"] = "ply-" + Path(row["name"]).stem + "-" + digest[:10]
        cases.append(row)
    return cases


def assign_splits(cases: list[dict[str, Any]], seed: int) -> dict[str, str]:
    groups = sorted({row["parent_id"] for row in cases},
                    key=lambda group: (stable_hash([seed, "split", group]), group))
    if len(groups) < 7:
        raise RuntimeError("At least seven virtual alias groups are needed for three splits")
    n_train = int(math.floor(len(groups) * 0.70))
    n_val = max(1, int(math.floor(len(groups) * 0.15)))
    return {group: ("train" if index < n_train else
                    "val" if index < n_train + n_val else "test")
            for index, group in enumerate(groups)}


def _mesh(path: Path):
    try:
        import trimesh
    except ImportError as error:
        raise RuntimeError("PLY preparation needs trimesh; install requirements-core.txt") from error
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Expected one triangle mesh, not a scene or point cloud")
    if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
        raise ValueError("Mesh has too few vertices or faces")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Mesh coordinates contain NaN or infinity")
    if not mesh.is_watertight:
        raise ValueError("Mesh is not watertight; no automatic geometric repair is applied")
    if np.any(np.ptp(mesh.vertices, axis=0) <= 0):
        raise ValueError("Mesh has zero extent along an axis")
    return mesh


def _metadata(data_root: Path, output: Path, row: dict[str, Any]) -> dict[str, Any]:
    path = output / "cache" / "mesh_metadata" / (row["sha256"] + ".json")
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("source_sha256") != row["sha256"]:
            raise RuntimeError("Mesh metadata checksum identity differs")
        return data
    mesh = _mesh(data_root / row["path"])
    data = {
        "source_sha256": row["sha256"], "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces), "watertight": bool(mesh.is_watertight),
        "raw_min_xyz": mesh.bounds[0].tolist(), "raw_max_xyz": mesh.bounds[1].tolist(),
        "repair": "none; process=False",
    }
    save_json(path, data)
    return data


def fit_training_domain(training_metadata: list[dict[str, Any]], padding: float = 0.10) -> dict[str, Any]:
    """Use training metadata only, with one fixed isotropic domain for all cases."""
    if not training_metadata:
        raise RuntimeError("No valid virtual training meshes for domain fitting")
    if not 0.0 < padding <= 1.0:
        raise ValueError("Domain padding must be in (0,1]")
    lower = np.min([row["raw_min_xyz"] for row in training_metadata], axis=0)
    upper = np.max([row["raw_max_xyz"] for row in training_metadata], axis=0)
    center = 0.5 * (lower + upper)
    side = float(np.max(upper - lower) * (1 + 2 * padding))
    if not math.isfinite(side) or side <= 0:
        raise ValueError("Cannot fit a positive finite domain")
    return {
        "raw_min_xyz": (center - side / 2).tolist(),
        "raw_max_xyz": (center + side / 2).tolist(),
        "raw_coordinate_unit": "unverified_generator_unit",
        "normalization": "single affine transform from frozen virtual-training domain to [-1,1]^3",
        "fit_source": "virtual training partition only",
        "padding_fraction_each_side": padding,
        "target_specific_bounds_used": False,
    }


def voxelize_mesh(path: Path, domain: dict[str, Any], grid_size: tuple[int, int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    import trimesh
    mesh = _mesh(path)
    if len(set(grid_size)) != 1 or min(grid_size) < 8:
        raise ValueError("R1.2 PLY preparation currently requires a cubic grid >=8")
    lower, upper = np.asarray(domain["raw_min_xyz"]), np.asarray(domain["raw_max_xyz"])
    # Reject held-out objects outside the frozen training domain. Do not silently
    # recenter, rescale, clip or enlarge the domain using their reference shapes.
    if np.any(mesh.bounds[0] < lower) or np.any(mesh.bounds[1] > upper):
        raise ValueError("Mesh exceeds frozen training domain; no target-wise rescaling is allowed")
    pitch = float((upper[0] - lower[0]) / grid_size[0])
    vertices = (np.asarray(mesh.vertices) - lower) / pitch - 0.5
    index_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)
    filled = index_mesh.voxelized(pitch=1.0, method="subdivide").fill(method="holes")
    xyz = np.rint(filled.points).astype(np.int64)
    if xyz.size == 0 or np.any(xyz < 0) or np.any(xyz >= grid_size[0]):
        raise ValueError("Voxelized object is empty or reaches outside the frozen grid")
    volume = np.zeros(grid_size, dtype=np.uint8)
    volume[xyz[:, 2], xyz[:, 1], xyz[:, 0]] = 1
    count = int(volume.sum())
    if count == 0 or count == volume.size:
        raise ValueError("Voxelized object is empty or fills the entire domain")
    return volume, {
        "method": "trimesh subdivide surface voxelization plus holes fill",
        "trimesh_version": trimesh.__version__, "mesh_processing": "none",
        "axis_order": "zyx", "raw_voxel_pitch": pitch,
        "occupancy_count": count, "occupancy_fraction": float(volume.mean()),
    }


def prepare(data_root: str | Path, out_dir: str | Path, config: dict[str, Any], smoke: bool = False) -> Path:
    """Prepare existing virtual PLY files and return a portable manifest path.

    ``config`` uses grid_size, data_split_seed (42), data_train_count (3500),
    data_val_count (750), data_test_count (750), data_domain_padding (.10).
    Smoke counts default to 12/3/3 and use a small deterministic source pool.
    All exclusions are logged; failing to fill the declared counts aborts.
    """
    data_root, output = Path(data_root).resolve(), Path(out_dir).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    output.mkdir(parents=True, exist_ok=True)
    grid_size = tuple(int(value) for value in config.get("grid_size", [32, 32, 32]))
    seed = int(config.get("data_split_seed", config.get("data_seed", 42)))
    default_counts = (12, 3, 3) if smoke else (3500, 750, 750)
    counts = {split: int(config.get("data_" + split + "_count", default))
              for split, default in zip(("train", "val", "test"), default_counts)}
    if any(value < 1 for value in counts.values()):
        raise ValueError("Each R1.2 split must request at least one case")
    padding = float(config.get("data_domain_padding", 0.10))
    rows = inventory(data_root)
    _write_jsonl(output / "raw_inventory.jsonl", rows)
    cases = group_virtual_sources(rows)
    real_hashes = {row["sha256"] for row in rows if row["source_kind"] == "real_interpreted_mesh"}
    identical_to_real = [row for row in cases if row["sha256"] in real_hashes]
    cases = [row for row in cases if row["sha256"] not in real_hashes]
    if smoke:
        pool_size = max(30, sum(counts.values()) * 4)
        cases = sorted(cases, key=lambda row: stable_hash([seed, "smoke", row["sample_id"]]))[:pool_size]
    splits = assign_splits(cases, seed)
    try:
        import trimesh
    except ImportError as error:
        raise RuntimeError("PLY preparation needs trimesh; install requirements-core.txt") from error
    identity = {
        "version": PREPARATION_VERSION, "protocol_version": PROTOCOL_VERSION,
        "grid_size": list(grid_size), "seed": seed, "requested_counts": counts,
        "smoke": bool(smoke), "domain_padding": padding,
        "eligible_family": "ore_surface", "raw_inventory_sha256": sha256_file(output / "raw_inventory.jsonl"),
        "preparation_source_sha256": sha256_file(__file__),
        "voxelizer": {"name": "trimesh", "version": trimesh.__version__,
                       "method": "subdivide_surface_plus_holes_fill"},
        "group_splits": splits,
    }
    intent = output / "preparation_intent.json"
    if intent.is_file() and json.loads(intent.read_text(encoding="utf-8")) != identity:
        raise RuntimeError("Existing PLY preparation differs in inputs or code; use a new output directory")
    save_json(intent, identity)
    exclusions = []
    for row in identical_to_real:
        exclusions.append({"source": row["path"], "sample_id": row["sample_id"],
                           "reason": "virtual_file_is_byte_identical_to_real_mesh",
                           "aliases": row["aliases"]})
    for row in rows:
        if row["source_kind"] == "real_interpreted_mesh":
            exclusions.append({"source": row["path"], "reason": "real_audit_only_no_verified_independent_domain_or_parent_map"})
        elif row["family"] != "ore_surface":
            exclusions.append({"source": row["path"], "reason": "other_virtual_family_coordinate_frame_and_ancestry_not_verified"})
    training_metadata = []
    training_source_ids = []
    failed_mesh_hashes = set()
    train_pool = [row for row in cases if splits[row["parent_id"]] == "train"]
    for index, row in enumerate(train_pool, 1):
        try:
            training_metadata.append(_metadata(data_root, output, row))
            training_source_ids.append(row["sample_id"])
        except Exception as error:
            failed_mesh_hashes.add(row["sha256"])
            exclusions.append({"source": row["path"], "sample_id": row["sample_id"], "stage": "training_domain_fit", "reason": type(error).__name__ + ": " + str(error)})
        if index % 500 == 0 or index == len(train_pool):
            print(f"[data] training-domain metadata {index}/{len(train_pool)}", flush=True)
    _write_jsonl(output / "exclusions.jsonl", exclusions)
    domain = fit_training_domain(training_metadata, padding)
    domain["training_source_ids_sha256"] = stable_hash(sorted(training_source_ids))
    domain["training_source_count"] = len(training_source_ids)
    save_json(output / "frozen_domain.json", domain)
    entries = []
    seen_occupancy = {}
    for split in ("train", "val", "test"):
        pool = sorted([row for row in cases if splits[row["parent_id"]] == split],
                      key=lambda row: (stable_hash([seed, "select", row["sample_id"]]), row["sample_id"]))
        completed = 0
        for row in pool:
            if completed >= counts[split]:
                break
            if row["sha256"] in failed_mesh_hashes:
                continue
            relative = "grids/" + row["sample_id"] + ".npy"
            grid_path = output / relative
            cache_key = stable_hash({"source_sha256": row["sha256"], "domain": domain,
                                     "grid_size": list(grid_size), "voxelizer": identity["voxelizer"],
                                     "preparation": identity["preparation_source_sha256"]})
            cache_path = output / "cache" / "voxel_metadata" / (cache_key + ".json")
            try:
                if cache_path.is_file() and grid_path.is_file():
                    record = json.loads(cache_path.read_text(encoding="utf-8"))
                    if record.get("grid_sha256") != sha256_file(grid_path):
                        raise RuntimeError("Cached grid checksum differs")
                    volume = np.load(grid_path, allow_pickle=False)
                    if volume.shape != grid_size:
                        raise RuntimeError("Cached grid shape differs")
                    conversion = record["conversion"]
                else:
                    volume, conversion = voxelize_mesh(data_root / row["path"], domain, grid_size)
                    grid_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = grid_path.with_suffix(".npy.tmp")
                    with temporary.open("wb") as stream:
                        np.save(stream, volume, allow_pickle=False)
                    temporary.replace(grid_path)
                    save_json(cache_path, {"grid_sha256": sha256_file(grid_path), "conversion": conversion})
                occupancy_sha = hashlib.sha256(np.ascontiguousarray(volume).tobytes()).hexdigest()
                if occupancy_sha in seen_occupancy:
                    exclusions.append({"source": row["path"], "sample_id": row["sample_id"], "reason": "duplicate_voxel_occupancy", "duplicate_of": seen_occupancy[occupancy_sha]})
                    continue
                seen_occupancy[occupancy_sha] = row["sample_id"]
                entries.append({
                    "sample_id": row["sample_id"], "parent_id": row["parent_id"],
                    "split": split, "path": relative, "sha256": sha256_file(grid_path),
                    "occupancy_sha256": occupancy_sha, "occupancy_fraction": float(volume.mean()),
                    "domain": DOMAIN, "source": "existing_virtual_ply",
                    "source_parent": None, "source_family": row["family"],
                    "source_file": row["path"], "source_file_sha256": row["sha256"],
                    "source_aliases": row["aliases"], "conversion": conversion,
                    "generation_ancestry": "unknown; not asserted independent of real meshes",
                })
                completed += 1
                if completed % 100 == 0 or completed == counts[split]:
                    print(f"[data] {split} grids {completed}/{counts[split]}", flush=True)
            except Exception as error:
                exclusions.append({"source": row["path"], "sample_id": row["sample_id"],
                                   "stage": "voxelization", "reason": type(error).__name__ + ": " + str(error)})
        _write_jsonl(output / "exclusions.jsonl", exclusions)
        if completed != counts[split]:
            save_json(output / "preparation_status.json", {"status": "failed", "split": split,
                      "completed": completed, "required": counts[split], "exclusions": "exclusions.jsonl"})
            raise RuntimeError(f"PLY preparation produced {completed}/{counts[split]} {split} cases; see {output / 'exclusions.jsonl'}")
    by_split = Counter(row["split"] for row in entries)
    manifest = {
        "protocol_version": PROTOCOL_VERSION, "count": len(entries), "seed": seed,
        "grid_size": list(grid_size), "domain": DOMAIN,
        "split_counts": dict(by_split),
        "parent_counts": {split: len({row["parent_id"] for row in entries if row["split"] == split}) for split in by_split},
        "source_data_policy": "existing_virtual_ply_ore_surface_only",
        "parent_semantics": "conservative filename and exact-file-hash alias groups; independent generative ancestry is unknown",
        "sampling_policy": "fixed_full_domain_target_independent_drill_coordinates",
        "real_data_policy": "real meshes inventoried only; face-count bins are not geological parent groups",
        "raw_domain": domain, "domain_record": "frozen_domain.json",
        "source_inventory": "raw_inventory.jsonl", "exclusion_log": "exclusions.jsonl",
        "preparation_identity_sha256": stable_hash(identity),
        "samples": entries,
    }
    path = output / "manifest.json"
    save_json(path, manifest)
    source_counts = Counter(row["source_directory"] for row in rows)
    family_counts = Counter(row["family"] for row in rows if row["source_kind"] == "virtual_generated_mesh")
    save_json(output / "data_summary.json", {
        "source_counts": dict(source_counts), "virtual_family_counts": dict(family_counts),
        "real_count": source_counts["mining_ply"], "raw_files": len(rows),
        "eligible_unique_files": len(cases), "selected_counts": dict(by_split),
        "virtual_identical_to_real_excluded": len(identical_to_real),
        "raw_duplicate_aliases": sum(max(0, len(row["aliases"]) - 1) for row in cases),
        "exclusion_count": len(exclusions), "domain": domain,
        "limits": [
            "Only ore_surface virtual meshes enter this R1.2 cohort; all other source families are inventoried with exclusions.",
            "A single training-derived domain is fixed for all selected validation and test meshes.",
            "Virtual labels do not establish independence from real source shapes; historical generator parent records were not supplied.",
            "Face-count categories in mining_ply do not identify independent deposits or geological parents.",
            "This is a candidate-acceptance diagnostic, not prospective field validation.",
        ],
    })
    save_json(output / "preparation_status.json", {"status": "complete", "manifest_sha256": sha256_file(path), "counts": dict(by_split)})
    return path
