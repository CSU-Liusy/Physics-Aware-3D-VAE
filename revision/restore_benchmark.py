"""Restore only the frozen GeoVAE benchmark arrays from recorded parameters.

This optional CPU-only data preparation route does not train/evaluate a model,
change a manifest, select a case by performance, or verify a remote download.
Both occupancy bytes and serialized NPY bytes must match the frozen manifest.
Existing mismatched files are never replaced.  --check-only writes no files;
it exits 2 if selected files are missing, even when exact replay is possible.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import tempfile
import types


MANIFEST_SHA256 = "48ec0924b76a1cca26715bb77566646dacb5fe9a3252471f8e4690aae8e01030"
GENERATOR_SHA256 = "0417235ebccba3c7b7f14af8c940f6eb8265bb0def203b1c7932a9b6eac64f5d"
HISTORICAL_GENERATOR_SHA256 = "058152b8f8763dca41ffdc7603ee5a971302f0aa4c5236cc8698f68f1dba8991"
ROOT = Path(__file__).resolve().parent


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def frozen_inputs(manifest_path: Path, generator_path: Path):
    manifest_bytes = manifest_path.read_bytes()
    require(digest(manifest_bytes) == MANIFEST_SHA256, "Frozen manifest SHA256 mismatch; do not replace the manifest.")
    manifest = json.loads(manifest_bytes)
    require(manifest["protocol_version"] == "procedural-lens-v1"
            and manifest["source_data_policy"] == "independent_procedural_only_no_real_mesh_parents"
            and manifest["count"] == 5000 and manifest["grid_size"] == [32, 32, 32]
            and manifest["generator_source_sha256"] == HISTORICAL_GENERATOR_SHA256,
            "Unsupported benchmark identity.")
    source = generator_path.read_bytes()
    require(digest(source) == GENERATOR_SHA256, "Archived generator SHA256 mismatch.")
    # Execute precisely the bytes just hashed.  Do not reread a possibly changed
    # file through an import loader or write a __pycache__ next to the source.
    module = types.ModuleType("_frozen_geovae_benchmark_generator")
    module.__file__ = str(generator_path)
    exec(compile(source, str(generator_path), "exec"), module.__dict__)
    return manifest, module


def destination_for(manifest_path: Path, row: dict) -> Path:
    relative = PurePosixPath(row["path"])
    require(relative.parts == ("grids", row["sample_id"] + ".npy"), "Unexpected manifest output path.")
    destination = manifest_path.parent.joinpath(*relative.parts)
    # Reject symlink/junction redirection of grids or of a destination file.
    # The manifest parent itself is the explicitly resolved input directory.
    require(destination.parent.resolve() == destination.parent, "The grids directory redirects outside the manifest directory.")
    require(not destination.is_symlink() and destination.resolve() == destination,
            "A destination redirects through a link.")
    return destination


def exact_payload(module, manifest: dict, row: dict) -> bytes:
    np = module.np
    array = module.generate_occupancy(tuple(manifest["grid_size"]), row["params"])
    require(array.dtype == np.uint8 and array.shape == (32, 32, 32), "Generator returned an unexpected shape or dtype.")
    require(digest(array.tobytes(order="C")) == row["occupancy_sha256"],
            "Generated occupancy differs for " + row["sample_id"] + "; use the recorded environment or matching author-supplied files.")
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    payload = stream.getvalue()
    require(digest(payload) == row["sha256"],
            "Serialized NPY differs for " + row["sample_id"] + "; no output was written. Use the recorded environment or matching author-supplied files.")
    return payload


def verify_existing(path: Path, expected: str) -> None:
    require(path.is_file() and not path.is_symlink(), "Existing destination is not an ordinary file: " + path.name)
    require(digest(path.read_bytes()) == expected,
            "Existing file differs and will not be replaced: " + path.name)


def install_without_overwrite(destination: Path, payload: bytes, expected: str) -> bool:
    """Atomically publish a verified temporary file; never replace a name.

    Returns False if a concurrent identical file appeared.  Hard linking is
    deliberately required: a platform that cannot do this fails safely rather
    than falling back to an overwriting rename or a partial destination write.
    """
    destination.parent.mkdir(exist_ok=True)
    require(destination.parent.resolve() == destination.parent, "The grids directory changed into a link.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".restore-", suffix=".tmp", dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        require(digest(temporary.read_bytes()) == expected, "Temporary file verification failed.")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            verify_existing(destination, expected)
            return False
        verify_existing(destination, expected)
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def restore(manifest_path: Path, generator_path: Path, case_ids: list[str] | None, check_only: bool) -> dict:
    manifest_path = manifest_path.resolve(strict=True)
    generator_path = generator_path.resolve(strict=True)
    manifest, module = frozen_inputs(manifest_path, generator_path)
    all_rows = manifest["samples"]
    requested = set(case_ids or [])
    known = {row["sample_id"] for row in all_rows}
    require(not (requested - known), "Unknown case IDs: " + ", ".join(sorted(requested - known)))
    rows = [row for row in all_rows if not requested or row["sample_id"] in requested]
    # Resolve every selected destination before writing any selected case.
    destinations = [(row, destination_for(manifest_path, row)) for row in rows]
    existing = restored = missing = 0
    for row, destination in destinations:
        if destination.exists():
            verify_existing(destination, row["sha256"])
            existing += 1
            continue
        # Verify both hashes in memory before creating any destination file.
        payload = exact_payload(module, manifest, row)
        if check_only:
            missing += 1
        elif install_without_overwrite(destination, payload, row["sha256"]):
            restored += 1
        else:
            existing += 1
    return {
        "status": "checked_replay_inputs_missing" if missing else "selected_inputs_verified",
        "check_only": check_only,
        "selected_case_count": len(rows),
        "full_benchmark_selected": len(rows) == 5000,
        "case_ids": sorted(requested) if requested else None,
        "existing_files_verified": existing,
        "missing_files_exactly_replayable": missing,
        "files_restored": restored,
        "manifest_sha256": MANIFEST_SHA256,
        "archived_generator_sha256": GENERATOR_SHA256,
        "numpy_version": module.np.__version__,
        "python_version": platform.python_version(),
        "remote_download_verified": False,
        "model_or_training_calls": 0,
        "scope": "Frozen data files only; no scientific result or public from-scratch training is verified.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "protocol" / "manifest.json")
    parser.add_argument("--generator", type=Path, default=ROOT / "experiment_data.py",
                        help="Archived generator source; its exact SHA256 is required.")
    parser.add_argument("--case-id", action="append", help="Check/restore only this exact manifest ID; may be repeated. Omit to select all 5,000.")
    parser.add_argument("--check-only", action="store_true", help="Write nothing; return 2 for missing but replayable inputs.")
    args = parser.parse_args(argv)
    try:
        report = restore(args.manifest, args.generator, args.case_id, args.check_only)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc), "model_or_training_calls": 0}), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["missing_files_exactly_replayable"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
