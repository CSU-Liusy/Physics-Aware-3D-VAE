"""Outermost-repository guards for the relocated GeoVAE launchers.

The byte-preserved revision engine treats ``revision/`` as its own code root.
These guards additionally protect the containing public Git repository. They
inspect argument strings and resolve paths, but never read data, create files,
start models or alter the historical engine/release records.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence


_ROUTES = {
    "revision": "revision", "review": "revision",
    "external": "external", "porphyry": "external",
    "legacy": "legacy",
}
_PRIVATE_FLAGS = {"--data-root", "--private-out"}
_REAL_SWITCHES = {"--real-cases", "--include-real"}
_LEGACY_FLAGS = _PRIVATE_FLAGS | {"--ply-dir", "--ply-file", "--output-dir"}
_EXTERNAL_FLAGS = {
    "--source", "--license-file", "--code-root", "--out",
    "--primary-run", "--data-cache", "--author-bundle", "--registration",
    "--reference-run", "--external-manifest",
}
_EXTERNAL_REQUIRED = {
    "prepare": {"--source", "--license-file", "--out"},
    "inspect": {"--primary-run", "--data-cache"},
    "run": {"--primary-run", "--data-cache", "--out"},
    "verify-only": {"--primary-run", "--data-cache", "--out"},
    "verify": {"--primary-run", "--data-cache", "--out"},
    "import-author-bundle": {"--primary-run", "--out"},
    "integration-inspect": {
        "--primary-run", "--data-cache", "--author-bundle", "--out", "--registration",
    },
    "integration-check": {
        "--primary-run", "--data-cache", "--author-bundle", "--out", "--registration",
    },
    "verify-integration": {
        "--primary-run", "--data-cache", "--author-bundle", "--out", "--registration",
    },
    "verify-reference": {
        "--primary-run", "--data-cache", "--author-bundle", "--out",
        "--reference-run", "--external-manifest",
    },
}


def _resolve_path(value: str, root: Path, flag: str) -> Path:
    if not value.strip() or "\x00" in value:
        raise ValueError(f"{flag} requires a nonempty valid path")
    try:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else root / path).resolve()
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Cannot resolve {flag}: {error}") from error


def _outside_repository(path: Path, root: Path, flag: str) -> None:
    # Reject both descendants and ancestors, so a broad source/output directory
    # cannot itself contain the public repository. Existing symlinks resolve
    # before this check; missing output directories need not be created.
    if path.is_relative_to(root) or root.is_relative_to(path):
        raise ValueError(
            f"{flag} must be outside the entire public repository, including "
            f"the outer root (not merely outside its revision/ directory): {path}"
        )


def _path_options(args: list[str], flags: set[str], switches: set[str]) -> dict[str, tuple[str, int, int]]:
    """Read protected long options without importing the scientific CLI.

    The inner argparse CLIs allow option abbreviation. Reject protected-option
    abbreviations here rather than allowing them to bypass an outer guard.
    Duplicates are rejected explicitly instead of relying on last-value wins.
    """
    found: dict[str, tuple[str, int, int]] = {}
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--":
            raise ValueError("The repository launcher does not accept an option separator '--'")
        flag, equal, inline = token.partition("=")
        if flag in flags:
            if flag in found:
                raise ValueError(f"Duplicate protected path option: {flag}")
            if equal:
                value, end = inline, index + 1
            else:
                if index + 1 >= len(args) or args[index + 1].startswith("-"):
                    raise ValueError(f"{flag} requires an explicit path value")
                value, end = args[index + 1], index + 2
            if not value.strip() or "\x00" in value:
                raise ValueError(f"{flag} requires a nonempty valid path")
            found[flag] = (value, index, end)
            index = end
            continue
        if flag.startswith("--") and flag not in switches:
            matches = sorted(candidate for candidate in flags | switches
                             if candidate.startswith(flag) and candidate != flag)
            if matches:
                raise ValueError(
                    f"Protected option abbreviations are not accepted: {flag}; "
                    f"use the complete option name ({', '.join(matches)})"
                )
        index += 1
    return found


def validate_arguments(route: str, args: Sequence[str], repository_root: str | Path) -> list[str]:
    """Validate outer-repository boundaries and return a new argument list.

    Supported routes are ``revision`` (alias ``review``), ``external`` (alias
    ``porphyry``) and ``legacy``. Relative paths are interpreted from the outer
    repository root, the documented subprocess cwd, and protected paths become
    canonical absolute paths. This function does not check data contents or
    existence, and does not create output directories.

    Ordinary revision synthetic ``--out runs/...`` remains unrestricted.
    Explicit real-data arguments, and legacy PLY/output arguments, must be
    outside the whole repository. A real-data revision invocation must provide
    both ``--data-root`` and ``--private-out`` explicitly.

    For the optional external CLI, source/license/cache/supplement outputs and
    author-only bridge dependencies must be outside the whole repository.
    ``--primary-run`` may identify an ordinary pure-synthetic primary run under
    repository ``runs/``; it is normalized but is not a licensed/private-data
    output. ``--code-root`` is injected if absent and must resolve exactly to
    repository ``revision/`` if explicitly provided.

    Help requests pass through unchanged, without creating or reading data.
    Config-derived legacy paths must additionally be checked by its own loader;
    this argument adapter cannot infer defaults from arbitrary YAML files.
    """
    if not isinstance(route, str) or route.casefold() not in _ROUTES:
        raise ValueError(f"Unknown repository route: {route!r}")
    if (isinstance(args, (str, bytes)) or not isinstance(args, Sequence)
            or not all(isinstance(item, str) for item in args)):
        raise ValueError("Arguments must be a sequence of strings")
    reviewed = list(args)
    if "--help" in reviewed or "-h" in reviewed:
        return reviewed
    kind = _ROUTES[route.casefold()]
    try:
        root = Path(repository_root).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Cannot resolve repository root: {error}") from error

    if kind == "external":
        if not reviewed or reviewed[0] not in _EXTERNAL_REQUIRED:
            choices = ", ".join(_EXTERNAL_REQUIRED)
            raise ValueError(f"External route requires an explicit command: {choices}")
        command = reviewed[0]
        found = _path_options(reviewed, _EXTERNAL_FLAGS, set())
        missing = sorted(_EXTERNAL_REQUIRED[command] - found.keys())
        if missing:
            raise ValueError(f"External {command} requires explicit paths: {', '.join(missing)}")
        expected_code = (root / "revision").resolve()
        if expected_code == root or not expected_code.is_relative_to(root):
            raise ValueError("The revision/ engine must remain inside this public repository")
        resolved = {flag: _resolve_path(value, root, flag)
                    for flag, (value, _, _) in found.items()}
        if "--code-root" in resolved and resolved["--code-root"] != expected_code:
            raise ValueError(f"--code-root must identify this repository's revision/ engine: {expected_code}")
        for flag, path in resolved.items():
            if flag not in {"--code-root", "--primary-run"}:
                _outside_repository(path, root, flag)
    else:
        flags = _LEGACY_FLAGS if kind == "legacy" else _PRIVATE_FLAGS
        found = _path_options(reviewed, flags, _REAL_SWITCHES)
        if kind == "revision" and any(flag in reviewed for flag in _REAL_SWITCHES):
            missing = sorted(_PRIVATE_FLAGS - found.keys())
            if missing:
                raise ValueError(f"Real-data evaluation requires explicit paths: {', '.join(missing)}")
        resolved = {flag: _resolve_path(value, root, flag)
                    for flag, (value, _, _) in found.items()}
        for flag, path in resolved.items():
            _outside_repository(path, root, flag)

    for flag, (_, start, end) in sorted(found.items(), key=lambda item: item[1][1], reverse=True):
        reviewed[start:end] = [flag, str(resolved[flag])]
    if kind == "external" and "--code-root" not in found:
        reviewed.extend(["--code-root", str(expected_code)])
    return reviewed
