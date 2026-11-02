"""Explicit deterministic evaluation policy, separate from the training recipe.

Only backend flags and environment metadata are adapted. Models, losses,
checkpoint loading and the seven frozen training source files are untouched.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import torch

POLICY_VERSION = "esi-evaluation-numerics-v1"
DECLARED_POLICY = {
    "version": POLICY_VERSION,
    "cublas_workspace_config": ":4096:8",
    "deterministic_algorithms": True,
    "deterministic_warn_only": False,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "scope": "evaluation_only_training_recipe_unchanged",
}


def helper_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def actual_flags() -> dict[str, Any]:
    """No thread count, PID, phase or process-specific paths in cache identity."""
    return {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def metadata() -> dict:
    return {"declared": dict(DECLARED_POLICY), "actual": actual_flags()}


def _install_metadata_adapter() -> None:
    """Idempotently update the one metadata API, including imported aliases.

    from-import bindings retain old function references. Rebinding only aliases
    that point to this exact metadata function keeps every evaluator consistent
    without wrapping any prediction, model or checkpoint operation.
    """
    pipeline = importlib.import_module("repro_pipeline")
    previous = pipeline.environment_info
    if getattr(previous, "_revision_numerical_adapter_version", None) == POLICY_VERSION:
        return
    base = getattr(previous, "_revision_numerical_base", previous)

    def environment_info() -> dict:
        result = dict(base())
        result["evaluation_numerical_policy"] = metadata()
        return result

    environment_info._revision_numerical_adapter_version = POLICY_VERSION
    environment_info._revision_numerical_base = base
    for module in tuple(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        candidate = namespace.get("environment_info") if namespace is not None else None
        if namespace is not None and (candidate is previous or candidate is base):
            namespace["environment_info"] = environment_info
    pipeline.environment_info = environment_info


def _write_proof(out_root: str | Path | None, proof: dict) -> None:
    if out_root is None:
        return
    root = Path(out_root).resolve() / "numerical_policy"
    root.mkdir(parents=True, exist_ok=True)
    phase = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(proof["phase"])).strip(".") or "evaluation"
    target = root / f"{phase}_pid{proof['pid']}.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(proof, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)


def _proof(kind: str, phase: str, before_cuda: bool, workspace_origin: str) -> dict:
    return {"format": "esi-numerical-policy-proof-v1", "kind": kind,
            "helper_sha256": helper_sha256(), "phase": str(phase), "pid": os.getpid(),
            "declared": dict(DECLARED_POLICY), "actual": actual_flags(),
            "cuda_initialized_at_configuration": before_cuda,
            "workspace_origin": workspace_origin, "recorded_unix": time.time(),
            "training_bitwise_reproducibility_claimed": False}


def configure_evaluation(out_root: str | Path | None = None, phase: str = "evaluation") -> dict:
    """Configure before CUDA initialization; reject conflicting explicit policy."""
    existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    required = DECLARED_POLICY["cublas_workspace_config"]
    initialized = bool(torch.cuda.is_initialized())
    if existing is not None and existing != required:
        raise RuntimeError("Incompatible CUBLAS_WORKSPACE_CONFIG; the frozen evaluation policy requires :4096:8")
    if existing is None and initialized:
        raise RuntimeError("Evaluation workspace policy must be set before CUDA initialization")
    if existing is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = required
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    _install_metadata_adapter()
    proof = _proof("evaluation", phase, initialized,
                   "preexisting_compatible" if existing is not None else "configured_before_cuda_initialization")
    _write_proof(out_root, proof)
    return proof


def record_training_policy(out_root: str | Path | None = None, phase: str = "train") -> dict:
    """Record unchanged training defaults. Never disable/enable algorithms here."""
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Training policy must remain deterministic_algorithms=False; use a fresh unchanged training process")
    _install_metadata_adapter()
    proof = _proof("training", phase, bool(torch.cuda.is_initialized()), "unchanged_training_environment")
    _write_proof(out_root, proof)
    return proof
