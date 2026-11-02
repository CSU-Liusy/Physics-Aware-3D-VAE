"""Leakage-controlled synthetic data and immutable raw-source audit for GeoVAE.

The synthetic generator is a stylized geometric lens generator. It is not a
geostatistical simulation or a physics model. No real PLY contributes shape,
bounds, or parameters to the synthetic samples.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PROTOCOL_VERSION = "procedural-lens-v1"
DOMAIN = {
    "min": [-1.0, -1.0, -1.0],
    "max": [1.0, 1.0, 1.0],
    "axis_order": "zyx",
    "units": "normalized",
}
LEGACY_PATTERN = re.compile(r"^generate_mine_[efghijk]\d+\.ply$", re.I)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _stable_seed(*parts: object) -> int:
    token = ":".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def _base_parameters(parent_seed: int) -> dict:
    rng = np.random.default_rng(parent_seed)
    return {
        "center": [float(x) for x in rng.uniform(-0.13, 0.13, 3)],
        "scales": [
            float(rng.uniform(0.38, 0.72)),
            float(rng.uniform(0.27, 0.54)),
            float(rng.uniform(0.11, 0.25)),
        ],
        "azimuth": float(rng.uniform(0, 2 * math.pi)),
        "fold_amplitude": float(rng.uniform(0.025, 0.10)),
        "fold_frequency": float(rng.uniform(1.0, 2.8)),
        "rim_amplitude": float(rng.uniform(0.04, 0.16)),
        "rim_frequency": float(rng.uniform(2.0, 5.0)),
        "branch": bool(rng.random() < 0.42),
        "branch_offset": [float(x) for x in rng.uniform(-0.24, 0.24, 3)],
    }


def _variant_parameters(base: dict, variant_seed: int) -> dict:
    rng = np.random.default_rng(variant_seed)
    result = dict(base)
    result["center"] = [
        float(x + delta)
        for x, delta in zip(base["center"], rng.normal(0, 0.038, 3))
    ]
    result["scales"] = [
        float(x * factor)
        for x, factor in zip(base["scales"], rng.lognormal(0, 0.055, 3))
    ]
    result["azimuth"] = float(base["azimuth"] + rng.normal(0, 0.09))
    result["fold_amplitude"] = float(
        max(0.0, base["fold_amplitude"] + rng.normal(0, 0.012))
    )
    result["rim_amplitude"] = float(
        max(0.0, base["rim_amplitude"] + rng.normal(0, 0.02))
    )
    result["branch_offset"] = [
        float(x + delta)
        for x, delta in zip(base["branch_offset"], rng.normal(0, 0.035, 3))
    ]
    return result


def generate_occupancy(grid_size: tuple[int, int, int], parameters: dict) -> np.ndarray:
    """Generate one fixed-domain, stylized lens-like orebody occupancy grid."""
    depth, height, width = [int(value) for value in grid_size]
    z = np.linspace(-1.0, 1.0, depth, dtype=np.float32)[:, None, None]
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[None, :, None]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, None, :]
    cx, cy, cz = parameters["center"]
    sx, sy, sz = parameters["scales"]
    angle = parameters["azimuth"]
    dx, dy = x - cx, y - cy
    xr = math.cos(angle) * dx + math.sin(angle) * dy
    yr = -math.sin(angle) * dx + math.cos(angle) * dy
    fold = parameters["fold_amplitude"] * np.sin(
        parameters["fold_frequency"] * math.pi * xr
    ) * np.cos(parameters["fold_frequency"] * math.pi * yr)
    rim = 1.0 + parameters["rim_amplitude"] * np.sin(
        parameters["rim_frequency"] * xr + 1.3
    ) * np.cos(parameters["rim_frequency"] * yr - 0.7)
    main = (
        (xr / sx) ** 2 + (yr / sy) ** 2 + ((z - cz - fold) / sz) ** 2
        <= rim
    )
    if parameters["branch"]:
        bx, by, bz = parameters["branch_offset"]
        branch = (
            ((xr - bx) / (sx * 0.59)) ** 2
            + ((yr - by) / (sy * 0.54)) ** 2
            + ((z - cz - bz - fold * 0.7) / (sz * 0.77)) ** 2
            <= 1.0
        )
        main |= branch
    return np.asarray(main, dtype=np.uint8)


def _mesh_quality_ascii_ply(path: Path) -> dict:
    """Check basic mesh integrity without treating a mesh as a verified field survey."""
    with open(path, "r", encoding="utf-8", errors="replace") as stream:
        if stream.readline().strip() != "ply":
            raise ValueError("missing PLY magic")
        n_vertices = n_faces = None
        ascii_format = False
        for line in stream:
            part = line.strip().split()
            if part[:2] == ["format", "ascii"]:
                ascii_format = True
            if part[:2] == ["element", "vertex"]:
                n_vertices = int(part[2])
            if part[:2] == ["element", "face"]:
                n_faces = int(part[2])
            if part and part[0] == "end_header":
                break
        else:
            raise ValueError("PLY header has no end_header")
        if not ascii_format:
            return {"status": "unsupported_ply_encoding"}
        if not n_vertices or not n_faces:
            raise ValueError("PLY has no vertices or faces")
        vertices = np.empty((n_vertices, 3), dtype=np.float64)
        for index in range(n_vertices):
            fields = stream.readline().split()
            if len(fields) < 3:
                raise ValueError("truncated vertex list")
            vertices[index] = [float(value) for value in fields[:3]]
        finite = bool(np.isfinite(vertices).all())
        edges: Counter[tuple[int, int]] = Counter()
        invalid_faces = 0
        for _ in range(n_faces):
            fields = stream.readline().split()
            if not fields:
                raise ValueError("truncated face list")
            count = int(fields[0])
            indices = [int(value) for value in fields[1 : count + 1]]
            if count < 3 or len(indices) != count or any(
                index < 0 or index >= n_vertices for index in indices
            ):
                invalid_faces += 1
                continue
            for a, b in zip(indices, indices[1:] + indices[:1]):
                edges[tuple(sorted((a, b)))] += 1
        boundary = sum(value == 1 for value in edges.values())
        nonmanifold = sum(value > 2 for value in edges.values())
        return {
            "status": "checked",
            "vertex_count": n_vertices,
            "face_count": n_faces,
            "finite_coordinates": finite,
            "invalid_faces": invalid_faces,
            "boundary_edges": boundary,
            "nonmanifold_edges": nonmanifold,
            "watertight_by_edge_count": finite
            and invalid_faces == 0
            and boundary == 0
            and nonmanifold == 0,
            "bounds_min_xyz": vertices.min(axis=0).tolist(),
            "bounds_max_xyz": vertices.max(axis=0).tolist(),
        }


def audit_raw_sources(data_root: str | Path, out_dir: str | Path) -> dict:
    """Hash source files and record provenance limitations; never alter source files."""
    root, output = Path(data_root).resolve(), Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inventory_path = output / "raw_inventory.jsonl"
    counts: Counter[str] = Counter()
    total_bytes: Counter[str] = Counter()
    generated_hash: dict[str, str] = {}
    pretrain_hash: dict[str, str] = {}
    duplicate_digests: defaultdict[str, list[str]] = defaultdict(list)
    legacy_count = 0
    real_quality: Counter[str] = Counter()
    real_quality_failures = []
    with open(inventory_path.with_suffix(".jsonl.tmp"), "w", encoding="utf-8") as stream:
        for directory in ("mining", "mining_ply", "mining_ply_generate", "mining_ply_generate_pretrain"):
            source_dir = root / directory
            if not source_dir.is_dir():
                continue
            for path in sorted(source_dir.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(root).as_posix()
                digest = sha256_file(path)
                size = path.stat().st_size
                counts[directory] += 1
                total_bytes[directory] += size
                row = {
                    "path": relative,
                    "sha256": digest,
                    "bytes": size,
                    "source_role": directory,
                }
                if directory == "mining_ply":
                    row["interpretation"] = "retrospective_pseudo_drill_only"
                    try:
                        row["mesh_quality"] = _mesh_quality_ascii_ply(path)
                        real_quality[row["mesh_quality"]["status"]] += 1
                    except Exception as error:
                        row["mesh_quality"] = {"status": "error", "error": str(error)}
                        real_quality_failures.append(relative)
                elif directory == "mining_ply_generate":
                    row["interpretation"] = "legacy_not_for_primary_protocol"
                    row["legacy_5000_cohort"] = bool(LEGACY_PATTERN.fullmatch(path.name))
                    legacy_count += int(row["legacy_5000_cohort"])
                    generated_hash[path.name] = digest
                elif directory == "mining_ply_generate_pretrain":
                    row["interpretation"] = "check_duplicate_of_legacy"
                    pretrain_hash[path.name] = digest
                else:
                    row["interpretation"] = "unverified_raw_source"
                duplicate_digests[digest].append(relative)
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(inventory_path.with_suffix(".jsonl.tmp"), inventory_path)
    matching = sum(generated_hash.get(name) == digest for name, digest in pretrain_hash.items())
    missing = sum(name not in generated_hash for name in pretrain_hash)
    changed = len(pretrain_hash) - matching - missing
    summary = {
        "data_root": str(root),
        "inventory": inventory_path.name,
        "counts": dict(counts),
        "total_bytes": dict(total_bytes),
        "legacy_5000_pattern_count": legacy_count,
        "pretrain_main_identical_sha256": matching,
        "pretrain_main_same_name_different_sha256": changed,
        "pretrain_missing_from_main": missing,
        "identical_file_groups": [
            paths for paths in duplicate_digests.values() if len(paths) > 1
        ],
        "real_mesh_quality_status_counts": dict(real_quality),
        "real_mesh_quality_failures": real_quality_failures,
        "limitations": [
            "File names and geometry do not establish geological provenance or drilling history.",
            "Real meshes are retrospective pseudo-drill cases only; no independent real borehole logs were provided.",
            "Legacy generated meshes are excluded from the primary training and test protocol.",
        ],
    }
    _json_write(output / "raw_audit_summary.json", summary)
    return summary



def _split_parents(count: int, group_size: int, seed: int) -> dict[int, str]:
    parents = count // group_size
    order = np.random.default_rng(seed).permutation(parents)
    n_train = int(round(parents * 0.70))
    n_val = int(round(parents * 0.15))
    split_by_parent = {}
    for rank, parent_index in enumerate(order):
        split_by_parent[int(parent_index)] = (
            "train" if rank < n_train else
            "val" if rank < n_train + n_val else "test"
        )
    return split_by_parent


def verify_protocol(manifest_path: str | Path, verify_files: bool = True) -> dict:
    """Fail on missing/corrupt samples, duplicate targets, or parent leakage."""
    path = Path(manifest_path).resolve()
    with open(path, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("protocol_version") not in {PROTOCOL_VERSION, "r12-ply-v1"}:
        raise ValueError("Protocol version does not match this reader")
    if manifest.get("domain") != DOMAIN:
        raise ValueError("Protocol domain is not the fixed normalized domain")
    entries = manifest.get("samples", [])
    if len(entries) != manifest.get("count"):
        raise ValueError("Manifest sample count does not match entries")
    parent_splits = {}
    digest_splits = {}
    ids = set()
    split_counts: Counter[str] = Counter()
    for row in entries:
        sample_id = row["sample_id"]
        parent_id = row["parent_id"]
        split = row["split"]
        digest = row["sha256"]
        if split not in {"train", "val", "test"}:
            raise ValueError("Invalid split: " + str(split))
        if sample_id in ids:
            raise ValueError("Duplicate sample ID: " + sample_id)
        ids.add(sample_id)
        if parent_id in parent_splits and parent_splits[parent_id] != split:
            raise ValueError("A parent crosses train/validation/test: " + parent_id)
        parent_splits[parent_id] = split
        if digest in digest_splits:
            raise ValueError(
                "Duplicate occupancy file across cases: "
                + sample_id + " and " + digest_splits[digest]
            )
        digest_splits[digest] = sample_id
        if row["domain"] != DOMAIN:
            raise ValueError("Sample domain differs from the protocol domain")
        grid_path = (path.parent / row["path"]).resolve()
        if not grid_path.is_relative_to(path.parent):
            raise ValueError("Sample path escapes protocol directory")
        if not grid_path.is_file():
            raise FileNotFoundError(grid_path)
        if verify_files and sha256_file(grid_path) != digest:
            raise ValueError("Source hash mismatch: " + str(grid_path))
        split_counts[split] += 1
    if dict(split_counts) != manifest["split_counts"]:
        raise ValueError("Split counts do not match manifest summary")
    return manifest


def prepare_protocol(
    data_root: str,
    out_dir: str,
    count: int = 5000,
    seed: int = 42,
    grid_size: tuple[int, int, int] = (32, 32, 32),
    audit_raw: bool = True,
) -> str:
    """Create an independent synthetic cohort with parent-disjoint 70/15/15 splits.

    The 5,000-case default creates 1,000 parent geometries with five variants
    each: 700/150/150 parents and 3,500/750/750 samples. Borehole masks are
    generated later from IDs only, after the parent-level split.
    """
    if count < 3:
        raise ValueError("At least three samples are needed for three splits")
    grid_size = tuple(int(value) for value in grid_size)
    if len(grid_size) != 3 or any(value < 8 for value in grid_size):
        raise ValueError("grid_size must contain three integers >= 8")
    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = verify_protocol(manifest_path)
        requested = (count, int(seed), list(grid_size))
        existing = (manifest["count"], manifest["seed"], manifest["grid_size"])
        if requested != existing:
            raise ValueError(
                "Existing protocol has different count/seed/grid; use a new out_dir"
            )
        if audit_raw and not (output / "raw_audit_summary.json").exists():
            audit_raw_sources(data_root, output)
        return str(manifest_path)
    if audit_raw:
        if not Path(data_root).is_dir():
            raise FileNotFoundError("Raw data root does not exist: " + str(data_root))
        audit_raw_sources(data_root, output)
    grid_dir = output / "grids"
    grid_dir.mkdir(exist_ok=True)
    group_size = 5 if count >= 100 and count % 20 == 0 else 1
    parent_split = _split_parents(count, group_size, int(seed))
    entries = []
    seen_arrays = {}
    for parent_index in range(count // group_size):
        parent_id = f"syn-parent-{parent_index:06d}"
        parent_seed = _stable_seed(seed, "parent", parent_index)
        base = _base_parameters(parent_seed)
        for variant in range(group_size):
            sample_id = f"syn-p{parent_index:06d}-v{variant}"
            relative = f"grids/{sample_id}.npy"
            grid_path = output / relative
            for attempt in range(40):
                case_seed = _stable_seed(seed, "case", parent_index, variant, attempt)
                params = _variant_parameters(base, case_seed)
                voxels = generate_occupancy(grid_size, params)
                fraction = float(voxels.mean())
                array_digest = hashlib.sha256(voxels.tobytes()).hexdigest()
                if 0.001 < fraction < 0.45 and array_digest not in seen_arrays:
                    break
            else:
                raise RuntimeError("Could not generate a unique nondegenerate case: " + sample_id)
            seen_arrays[array_digest] = sample_id
            if grid_path.exists():
                old = np.load(grid_path, allow_pickle=False)
                if old.shape != grid_size or not np.array_equal(old, voxels):
                    raise ValueError(
                        "Interrupted-run file differs from this generator: " + str(grid_path)
                    )
            else:
                temporary = grid_path.with_suffix(".npy.tmp")
                with open(temporary, "wb") as stream:
                    np.save(stream, voxels, allow_pickle=False)
                os.replace(temporary, grid_path)
            entries.append({
                "sample_id": sample_id,
                "parent_id": parent_id,
                "variant": variant,
                "split": parent_split[parent_index],
                "path": relative,
                "sha256": sha256_file(grid_path),
                "occupancy_sha256": array_digest,
                "occupancy_fraction": fraction,
                "seed": case_seed,
                "generation_attempt": attempt,
                "source": "independent_procedural_lens",
                "source_parent": None,
                "domain": DOMAIN,
                "params": params,
            })
    split_counts = dict(Counter(row["split"] for row in entries))
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "generator_source_sha256": sha256_file(Path(__file__)),
        "count": count,
        "seed": int(seed),
        "grid_size": list(grid_size),
        "domain": DOMAIN,
        "group_size": group_size,
        "split_counts": split_counts,
        "parent_counts": dict(Counter(parent_split.values())),
        "source_data_policy": "independent_procedural_only_no_real_mesh_parents",
        "sampling_policy": "fixed_full_domain_target_independent_drill_coordinates",
        "real_data_policy": "retrospective_pseudo_drill_only_not_primary_test",
        "samples": entries,
    }
    _json_write(manifest_path, manifest)
    verify_protocol(manifest_path)
    return str(manifest_path)


def make_drill_masks(
    grid_size: tuple[int, int, int],
    sample_id: str,
    num_holes: int,
    samples_per_hole: int,
    mask_seed: int = 42,
    heldout_holes: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint observed and hidden vertical-hole masks, independent of target.

    Hidden hole locations are reserved first; observed locations are nested
    for 2/4/8/16-hole comparisons. The same case/seed always yields the same
    spatial coordinates.
    """
    depth, height, width = [int(value) for value in grid_size]
    if num_holes < 0 or heldout_holes < 0 or samples_per_hole < 1:
        raise ValueError("Invalid drilling configuration")
    if num_holes + heldout_holes > height * width:
        raise ValueError("More distinct holes requested than XY columns")
    rng = np.random.default_rng(_stable_seed("drill", sample_id, mask_seed))
    column_order = rng.permutation(height * width)
    z_samples = np.rint(np.linspace(0, depth - 1, samples_per_hole)).astype(int)
    heldout = np.zeros(grid_size, dtype=bool)
    observed = np.zeros(grid_size, dtype=bool)
    for rank, flat in enumerate(column_order[: heldout_holes + num_holes]):
        y, x = divmod(int(flat), width)
        if rank < heldout_holes:
            heldout[z_samples, y, x] = True
        else:
            observed[z_samples, y, x] = True
    return observed, heldout


@lru_cache(maxsize=5000)
def _cached_drill_masks(
    grid_size: tuple[int, int, int], sample_id: str, num_holes: int,
    samples_per_hole: int, mask_seed: int, heldout_holes: int,
) -> tuple[np.ndarray, np.ndarray]:
    masks = make_drill_masks(
        grid_size, sample_id, num_holes, samples_per_hole, mask_seed,
        heldout_holes,
    )
    for mask in masks:
        mask.flags.writeable = False
    return masks


# Formal training revisits the same 3,500 train and 750 validation grids for
# up to 200 epochs.  The frozen 32^3 uint8 protocol occupies about 164 MiB,
# so retaining its grids avoids hundreds of thousands of repeated .npy reads.
# The cache is process-local; it never mixes data between independent runs.
@lru_cache(maxsize=5000)
def _load_grid(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=False)


class ProtocolDataset(Dataset):
    """Leakage-controlled cases; each item is (observations, target, sample_id)."""

    def __init__(
        self,
        manifest_path: str,
        split: str,
        num_holes: int = 8,
        samples_per_hole: int = 12,
        mask_seed: int = 42,
        heldout_holes: int = 4,
        noise_rate: float = 0.0,
        verify_hashes: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if not 0 <= noise_rate <= 1:
            raise ValueError("noise_rate must be within [0, 1]")
        self.manifest_path = str(Path(manifest_path).resolve())
        self.manifest = verify_protocol(self.manifest_path, verify_files=verify_hashes)
        self.entries = [
            row for row in self.manifest["samples"] if row["split"] == split
        ]
        self.metadata = {row["sample_id"]: row for row in self.entries}
        self.split = split
        self.grid_size = tuple(self.manifest["grid_size"])
        self.num_holes = int(num_holes)
        self.samples_per_hole = int(samples_per_hole)
        self.mask_seed = int(mask_seed)
        self.heldout_holes = int(heldout_holes)
        self.noise_rate = float(noise_rate)

    def __len__(self) -> int:
        return len(self.entries)

    def get_masks(self, sample_id: str) -> tuple[np.ndarray, np.ndarray]:
        if sample_id not in self.metadata:
            raise KeyError(sample_id)
        return _cached_drill_masks(
            self.grid_size,
            sample_id,
            self.num_holes,
            self.samples_per_hole,
            self.mask_seed,
            self.heldout_holes,
        )

    def get_case(self, sample_id: str) -> dict:
        row = self.metadata[sample_id]
        path = str((Path(self.manifest_path).parent / row["path"]).resolve())
        target = np.asarray(_load_grid(path), dtype=np.float32)
        if target.shape != self.grid_size:
            raise ValueError("Grid shape differs from manifest: " + sample_id)
        observed_mask, hidden_mask = self.get_masks(sample_id)
        labels = target * observed_mask.astype(np.float32)
        if self.noise_rate:
            locations = np.flatnonzero(observed_mask)
            n_flips = int(round(len(locations) * self.noise_rate))
            if n_flips:
                rng = np.random.default_rng(
                    _stable_seed("noise", sample_id, self.mask_seed)
                )
                chosen = rng.permutation(locations)[:n_flips]
                labels.flat[chosen] = 1.0 - labels.flat[chosen]
        obs = np.stack((labels, observed_mask.astype(np.float32)), axis=0)
        return {
            "obs": obs,
            "target": target,
            "observed_mask": observed_mask,
            "hidden_mask": hidden_mask,
            "metadata": row,
        }

    def __getitem__(self, index: int):
        sample_id = self.entries[index]["sample_id"]
        case = self.get_case(sample_id)
        return (
            torch.from_numpy(case["obs"]),
            torch.from_numpy(case["target"].copy()),
            sample_id,
        )


# Backwards-compatible name used by the experiment runner.
validate_manifest = verify_protocol

