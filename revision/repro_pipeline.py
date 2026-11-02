"""Reproducible, manifest-driven GeoVAE experiments.

The legacy trainer and benchmark scripts remain in the repository for historical
comparison. This module is the only supported entry point for new paper results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "revision_v1.yaml"
DEFAULT_DATA = ROOT.parent / "data"
DEFAULT_OUT = ROOT.parent / "results" / "repro_v2"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def source_hash() -> str:
    digest = hashlib.sha256()
    for name in ("repro_pipeline.py", "experiment_data.py", "model.py", "repro_cgan.py", "repro_baselines.py", "repro_metrics.py", "comparisons/models/unet.py"):
        path = ROOT / name
        if path.exists():
            digest.update(name.encode("utf-8"))
            digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream) if path.suffix.lower() == ".json" else yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return config


EVALUATION_ONLY_CONFIG_KEYS = {
    "threshold", "test_time_steps", "test_time_lr",
    "test_time_lora_lr", "test_time_prior_weight",
    "test_time_lora_weight", "test_time_lora_rank",
    "latent_test_time_steps", "latent_test_time_lr",
    "joint_test_time_steps", "joint_test_time_lr",
    "joint_test_time_lora_lr",
    "idw_power", "rbf_kernel", "rbf_smoothing",
    "ik_len_scale", "ik_nugget", "mps_realizations",
    "mps_n_cond", "mps_training_image_id",
    "cgan_eval_realizations", "device",
}


def training_config_hash(config: dict[str, Any]) -> str:
    """Hash fields that must match a learned model's training recipe."""
    return stable_hash({
        key: value for key, value in config.items()
        if key not in EVALUATION_ONLY_CONFIG_KEYS
    })

def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _save_checkpoint_atomic(value: dict[str, Any], path: Path) -> None:
    """Keep the previous resumable checkpoint if a save is interrupted."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def select_device(required: str) -> torch.device:
    if required == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false. Install a CUDA-enabled PyTorch build.")
    if required == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(required)


def doctor(require_cuda: bool = False) -> dict[str, Any]:
    checks = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
            if torch.cuda.is_available() else None,
    }
    for module in ("numpy", "scipy", "sklearn", "yaml", "tensorboard", "gstools", "mpslib"):
        checks[module] = importlib.util.find_spec(module) is not None
    checks["numpy_version"] = np.__version__
    try:
        checks["scikit_mps_version"] = importlib.metadata.version("scikit-mps")
    except importlib.metadata.PackageNotFoundError:
        checks["scikit_mps_version"] = None
    if checks["scikit_mps_version"] == "0.6.0" and not hasattr(np, "NaN"):
        raise RuntimeError(
            "scikit-mps 0.6.0 uses np.NaN, which is absent in NumPy 2. "
            "Install numpy>=1.26,<2 from requirements-core.txt before running experiments."
        )
    checks["mps_genesim_executable"] = shutil.which("mps_genesim")
    if require_cuda and not checks["cuda_available"]:
        raise RuntimeError(json.dumps(checks, ensure_ascii=False, indent=2))
    return checks


def environment_info() -> dict[str, Any]:
    versions = {}
    for package in ("numpy", "scipy", "PyYAML", "matplotlib", "gstools", "scikit-mps"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "packages": versions}


def make_model(kind: str, config: dict[str, Any]) -> torch.nn.Module:
    from model import ConvVAE3D
    if kind in ("geovae", "base_vae"):
        return ConvVAE3D(
            in_channels=2,
            grid_size=tuple(config["grid_size"]),
            latent_dim=int(config["latent_dim"]),
            base_channels=int(config["base_channels"]),
            use_transformer=(kind == "geovae" and bool(config.get("use_transformer", True))),
            use_relative_position=(kind == "geovae" and bool(config.get("use_relative_position", True))),
        )
    if kind == "cgan":
        from repro_cgan import make_cgan_generator
        return make_cgan_generator(config)
    if kind == "unet":
        from comparisons.models.unet import UNet3D
        return UNet3D(in_channels=2, out_channels=1,
                      init_features=int(config.get("unet_features", 32)))
    raise ValueError(f"Unsupported trainable model kind: {kind}")


def compute_loss(kind: str, model: torch.nn.Module, obs: torch.Tensor,
                 target: torch.Tensor, config: dict[str, Any],
                 beta: float) -> tuple[torch.Tensor, dict[str, float]]:
    mask = obs[:, 1]
    if kind == "unet":
        logits = model(obs, return_logits=True).squeeze(1)
        per_voxel = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce_full = per_voxel.mean()
        bce_observed = (per_voxel * mask).sum() / mask.sum().clamp_min(1)
        loss = bce_full + float(config["lambda_observed"]) * bce_observed
        return loss, {"full_bce": float(bce_full.detach()), "observed_bce": float(bce_observed.detach()), "kl": 0.0}
    logits, mu, logvar = model(obs)
    per_voxel = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce_full = per_voxel.mean()
    bce_observed = (per_voxel * mask).sum() / mask.sum().clamp_min(1)
    kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum(dim=1).mean() / mu.shape[1]
    loss = bce_full + float(config["lambda_observed"]) * bce_observed + beta * kl
    return loss, {"full_bce": float(bce_full.detach()), "observed_bce": float(bce_observed.detach()), "kl": float(kl.detach())}


def deterministic_probability(kind: str, model: torch.nn.Module,
                              obs: torch.Tensor) -> torch.Tensor:
    if kind == "unet":
        return model(obs).squeeze(1).clamp(0, 1)
    mu, _ = model.encode(obs)
    return torch.sigmoid(model.decode(mu))


def validate(kind: str, model: torch.nn.Module, loader: DataLoader,
             device: torch.device) -> dict[str, float]:
    model.eval()
    full_bce, blind_intersection, blind_union, observed_errors = 0.0, 0.0, 0.0, 0.0
    observed_count, n = 0.0, 0
    with torch.no_grad():
        for obs, target, _ in loader:
            obs, target = obs.to(device), target.to(device)
            probability = deterministic_probability(kind, model, obs)
            full_bce += float(F.binary_cross_entropy(probability.clamp(1e-6, 1 - 1e-6),
                                                     target, reduction="sum").item())
            predicted = probability >= 0.5
            truth = target >= 0.5
            known = obs[:, 1] >= 0.5
            blind = ~known
            blind_intersection += float((predicted & truth & blind).sum().item())
            blind_union += float(((predicted | truth) & blind).sum().item())
            observed_errors += float(((predicted != truth) & known).sum().item())
            observed_count += float(known.sum().item())
            n += target.numel()
    if n == 0:
        raise RuntimeError("Validation split is empty")
    return {
        "full_bce": full_bce / n,
        "blind_iou": blind_intersection / max(blind_union, 1.0),
        "observed_error": observed_errors / max(observed_count, 1.0),
    }


def _dataset(manifest: str, split: str, config: dict[str, Any]):
    from experiment_data import ProtocolDataset
    return ProtocolDataset(
        manifest,
        split,
        num_holes=int(config["num_holes"]),
        samples_per_hole=int(config["samples_per_hole"]),
        mask_seed=int(config["mask_seed"]),
    )


def _checkpoint_payload(kind: str, model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer, scaler: Any,
                        epoch: int, config: dict[str, Any], manifest_sha: str,
                        seed: int, best_val: float, generator: torch.Generator, no_improve: int) -> dict[str, Any]:
    return {
        "format_version": 2,
        "model_kind": kind,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "config": config,
        "config_sha256": stable_hash(config),
        "manifest_sha256": manifest_sha,
        "source_sha256": source_hash(),
        "seed": seed,
        "best_validation_bce": best_val,
        "no_improve": no_improve,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_rng_state": generator.get_state(),
    }


def train(manifest: str, config_path: str, run_dir: str, kind: str,
          seed: int, resume: str | None = None,
          max_epochs: int | None = None) -> Path:
    config = load_config(config_path)
    device = select_device(str(config.get("device", "auto")))
    set_seed(seed)
    from experiment_data import validate_manifest
    validate_manifest(manifest)
    manifest_sha = sha256_file(manifest)
    train_set = _dataset(manifest, "train", config)
    val_set = _dataset(manifest, "val", config)
    if len(train_set) == 0 or len(val_set) == 0:
        raise RuntimeError("Train and validation splits must both be nonempty")
    microbatch = int(config["microbatch"])
    effective_batch = int(config["effective_batch"])
    if microbatch < 1 or effective_batch < microbatch or effective_batch % microbatch:
        raise ValueError("effective_batch must be a positive multiple of microbatch")
    accumulation = effective_batch // microbatch
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=microbatch, shuffle=True,
                              generator=generator, num_workers=int(config.get("num_workers", 0)),
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=microbatch, shuffle=False,
                            num_workers=0, pin_memory=device.type == "cuda")
    model = make_model(kind, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]),
                                  weight_decay=float(config.get("weight_decay", 1e-5)))
    amp = bool(config.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    run = Path(run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    save_json(run / "run.json", {
        "kind": kind, "seed": seed, "config": config, "manifest": str(Path(manifest).resolve()),
        "manifest_sha256": manifest_sha, "source_sha256": source_hash(),
        "device": str(device), "torch": torch.__version__,
        "environment": environment_info(),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    })
    start_epoch, best_val, no_improve = 0, math.inf, 0
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if checkpoint.get("manifest_sha256") != manifest_sha:
            raise RuntimeError("Resume checkpoint manifest hash does not match")
        if checkpoint.get("config_sha256") != stable_hash(config):
            raise RuntimeError("Resume checkpoint configuration does not match")
        if checkpoint.get("source_sha256") != source_hash():
            raise RuntimeError("Resume checkpoint source hash does not match")
        if checkpoint.get("model_kind") != kind or checkpoint.get("seed") != seed:
            raise RuntimeError("Resume checkpoint model kind or seed does not match")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"])
        best_val = float(checkpoint["best_validation_bce"])
        no_improve = int(checkpoint.get("no_improve", 0))
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
            np.random.set_state(checkpoint["numpy_rng_state"])
            torch.set_rng_state(checkpoint["torch_rng_state"])
            if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
            generator.set_state(checkpoint["loader_rng_state"])
    epochs = int(max_epochs if max_epochs is not None else config["epochs"])
    patience = int(config["patience"])
    history_path = run / "history.jsonl"
    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums = {"full_bce": 0.0, "observed_bce": 0.0, "kl": 0.0}
        batches = 0
        warmup = max(int(config.get("kl_warmup_epochs", 1)), 1)
        beta = float(config.get("beta", 1.0)) * min(1.0, epoch / warmup)
        started = time.time()
        for batch_index, (obs, target, _) in enumerate(train_loader):
            obs, target = obs.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss, components = compute_loss(kind, model, obs, target, config, beta)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}, batch={batch_index}")
            # The final accumulation window may contain fewer than 16 samples.
            # Weight by its actual sample count rather than shrinking that update.
            window_start = (batch_index // accumulation) * effective_batch
            window_samples = min(effective_batch, len(train_set) - window_start)
            scaler.scale(loss * (obs.shape[0] / window_samples)).backward()
            if ((batch_index + 1) % accumulation == 0) or (batch_index + 1 == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            for key in sums:
                sums[key] += components[key]
            batches += 1
        val = validate(kind, model, val_loader, device)
        record = {
            "epoch": epoch, "train_full_bce": sums["full_bce"] / batches,
            "train_observed_bce": sums["observed_bce"] / batches,
            "train_kl": sums["kl"] / batches, "val_full_bce": val["full_bce"],
            "val_blind_iou": val["blind_iou"], "val_observed_error": val["observed_error"],
            "beta": beta, "seconds": time.time() - started,
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))
        if val["full_bce"] < best_val - 1e-6:
            best_val, no_improve = val["full_bce"], 0
            _save_checkpoint_atomic(_checkpoint_payload(kind, model, optimizer, scaler, epoch, config,
                                           manifest_sha, seed, best_val, generator, no_improve), run / "best.pt")
        else:
            no_improve += 1
        _save_checkpoint_atomic(_checkpoint_payload(kind, model, optimizer, scaler, epoch, config,
                                       manifest_sha, seed, best_val, generator, no_improve), run / "latest.pt")
        if no_improve >= patience:
            print(f"Validation early stop after epoch {epoch}; best BCE={best_val:.6f}")
            break
    return run / "best.pt"

def _load_checked_model(kind: str, checkpoint_path: str, manifest_sha: str,
                        config: dict[str, Any], seed: int, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {kind} checkpoint: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("model_kind") != kind:
        raise RuntimeError(f"{path} is {state.get('model_kind')}, expected {kind}")
    if state.get("manifest_sha256") != manifest_sha:
        raise RuntimeError(f"{kind} checkpoint belongs to a different data manifest")
    if training_config_hash(state["config"]) != training_config_hash(config):
        raise RuntimeError(f"{kind} checkpoint training configuration differs")
    if int(state.get("seed", -1)) != seed:
        raise RuntimeError(f"{kind} checkpoint belongs to seed {state.get('seed')}, expected {seed}")
    if kind == "cgan":
        from repro_cgan import _source_sha256
        expected_source = _source_sha256()
    else:
        expected_source = source_hash()
    if state.get("source_sha256") != expected_source:
        raise RuntimeError(f"{kind} checkpoint source hash does not match current implementation")
    model = make_model(kind, state["config"])
    model.load_state_dict(state["model_state"], strict=True)
    model.to(device).eval()
    return model, state


def _predict(method: str, obs: np.ndarray, models: dict[str, torch.nn.Module],
             config: dict[str, Any], device: torch.device,
             training_image: np.ndarray | None, case_seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    from repro_baselines import (predict_idw, predict_rbf,
                                 predict_indicator_kriging,
                                 predict_mps_direct_sampling)
    if method == "idw":
        return predict_idw(obs, power=float(config.get("idw_power", 2.0))), {}
    if method == "rbf":
        return predict_rbf(obs, kernel=str(config.get("rbf_kernel", "linear")),
                           smoothing=float(config.get("rbf_smoothing", 0.01))), {}
    if method == "ik":
        return predict_indicator_kriging(
            obs, len_scale=float(config.get("ik_len_scale", 5.0)),
            nugget=float(config.get("ik_nugget", 0.01))), {}
    if method == "mps":
        if training_image is None:
            raise RuntimeError("MPS requires a training-only training image")
        return predict_mps_direct_sampling(
            obs, training_image, case_seed,
            realizations=int(config.get("mps_realizations", 30)),
            n_cond=int(config.get("mps_n_cond", 25))), {}
    if method == "cgan":
        from repro_cgan import predict_cgan
        tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
        probability = predict_cgan(models["cgan"], tensor, seed=case_seed,
                                   realizations=int(config.get("cgan_eval_realizations", 1)))
        return probability.squeeze(0).detach().cpu().numpy().astype(np.float32), {}
    kind = {"geovae": "geovae", "geovae_latent": "geovae",
            "geovae_joint": "geovae", "base_vae": "base_vae",
            "unet": "unet"}.get(method)
    if kind is None:
        raise ValueError(f"Unknown method: {method}")
    if kind not in models:
        raise RuntimeError(f"Required {kind} model checkpoint was not loaded")
    tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
    if method in ("geovae_latent", "geovae_joint"):
        from model import adapt_to_observation
        joint = method == "geovae_joint"
        result, metadata = adapt_to_observation(
            models[kind], tensor,
            mode="joint" if joint else "latent",
            seed=case_seed,
            steps=int(config.get(
                "joint_test_time_steps" if joint else "latent_test_time_steps",
                config.get("test_time_steps", 50))),
            lora_rank=int(config.get("test_time_lora_rank", 8)),
            lr=float(config.get(
                "joint_test_time_lr" if joint else "latent_test_time_lr",
                config.get("test_time_lr", 0.03))),
            lora_lr=float(config.get(
                "joint_test_time_lora_lr",
                config.get("test_time_lora_lr", 0.001))),
            prior_weight=float(config.get("test_time_prior_weight", 0.01)),
            lora_weight=float(config.get("test_time_lora_weight", 0.01)),
            project_observed=False,
        )
        return result.detach().cpu().squeeze(0).numpy().astype(np.float32), metadata
    with torch.no_grad():
        probability = deterministic_probability(kind, models[kind], tensor)
    return probability.squeeze(0).detach().cpu().numpy().astype(np.float32), {}



def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    """Keep a previous valid prediction if a process stops during a write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array.astype(np.float32), allow_pickle=False)
    os.replace(temporary, path)


def _read_prediction_for_resume(
    output: Path, record: dict[str, Any], prefix: str,
    expected_relative: Path, shape: tuple[int, ...],
) -> np.ndarray:
    relative = record.get(f"{prefix}_file")
    if relative != expected_relative.as_posix():
        raise ValueError(f"{prefix} path differs from the expected case and method")
    path = (output / relative).resolve()
    if not path.is_relative_to(output):
        raise ValueError(f"{prefix} path escapes evaluation directory")
    if not path.is_file() or sha256_file(path) != record.get(f"{prefix}_sha256"):
        raise ValueError(f"{prefix} file is missing or its SHA256 differs")
    prediction = np.load(path, allow_pickle=False)
    if prediction.shape != shape or not np.isfinite(prediction).all():
        raise ValueError(f"{prefix} array has invalid shape or nonfinite values")
    return prediction


_CLASSICAL_METHODS = frozenset({"idw", "rbf", "ik", "mps"})


def _prediction_settings(method: str, config: dict[str, Any]) -> dict[str, Any]:
    """Only settings that can change this method's *raw* prediction.

    Checkpoint hashes cover learned weights and their training configurations.
    Keeping unrelated evaluation settings out lets a classical prediction be
    shared with an otherwise identical ablation or model-seed evaluation.
    """
    defaults = {
        "idw": {"idw_power": 2.0},
        "rbf": {"rbf_kernel": "linear", "rbf_smoothing": 0.01},
        "ik": {"ik_len_scale": 5.0, "ik_nugget": 0.01},
        "mps": {"mps_realizations": 30, "mps_n_cond": 25},
        "cgan": {"cgan_eval_realizations": 1},
        "geovae_latent": {"test_time_prior_weight": 0.01},
        "geovae_joint": {"test_time_prior_weight": 0.01,
                         "test_time_lora_weight": 0.01,
                         "test_time_lora_rank": 8},
    }.get(method, {})
    settings = {key: config.get(key, value) for key, value in defaults.items()}
    if method in {"geovae_latent", "geovae_joint"}:
        joint = method == "geovae_joint"
        settings["effective_steps"] = int(config.get(
            "joint_test_time_steps" if joint else "latent_test_time_steps",
            config.get("test_time_steps", 50)))
        settings["effective_lr"] = float(config.get(
            "joint_test_time_lr" if joint else "latent_test_time_lr",
            config.get("test_time_lr", 0.03)))
        if joint:
            settings["effective_lora_lr"] = float(config.get(
                "joint_test_time_lora_lr",
                config.get("test_time_lora_lr", 0.001)))
    return settings


def _prediction_cache_identity(
    *, method: str, config: dict[str, Any], manifest_sha: str,
    split: str, sample_id: str, sample_grid_sha: str, observation: np.ndarray,
    holes: int, noise_rate: float, case_seed: int,
    checkpoint_hashes: dict[str, str], training_image_id: str | None,
    training_image_sha256: str | None, device: torch.device,
    source_sha: str, environment: dict[str, Any],
) -> dict[str, Any]:
    kind = {"geovae": "geovae", "geovae_latent": "geovae",
            "geovae_joint": "geovae", "base_vae": "base_vae",
            "unet": "unet", "cgan": "cgan"}.get(method)
    obs = np.ascontiguousarray(observation)
    identity = {
        "schema": 1, "manifest_sha256": manifest_sha,
        "source_sha256": source_sha, "split": split, "sample_id": sample_id,
        "sample_grid_sha256": sample_grid_sha,
        "observation_sha256": hashlib.sha256(obs.tobytes()).hexdigest(),
        "observation_shape": list(obs.shape), "observation_dtype": str(obs.dtype),
        "num_holes": holes, "noise_rate": float(noise_rate),
        "mask_seed": int(config["mask_seed"]),
        "samples_per_hole": int(config["samples_per_hole"]),
        "heldout_holes": 4, "method": method, "case_seed": case_seed,
        "prediction_settings": _prediction_settings(method, config),
        "checkpoint_sha256": checkpoint_hashes.get(kind) if kind else None,
        "device": str(device), "environment": environment,
        "mps_training_image_id": training_image_id if method == "mps" else None,
        "mps_training_image_sha256": (
            training_image_sha256 if method == "mps" else None),
    }
    if kind and not identity["checkpoint_sha256"]:
        raise RuntimeError(f"Prediction cache requires a checked {kind} checkpoint")
    return identity


def _cache_record_path(cache_root: Path, identity: dict[str, Any]) -> Path:
    digest = stable_hash(identity)
    return cache_root / digest[:2] / f"{digest}.json"


def _read_cached_prediction(
    cache_root: Path, identity: dict[str, Any], shape: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]] | None:
    path = _cache_record_path(cache_root, identity)
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("identity") != identity:
        raise RuntimeError(f"Prediction cache identity mismatch: {path}")
    run_root = cache_root.parent.resolve()
    source = (run_root / str(record.get("source_prediction", ""))).resolve()
    if not source.is_relative_to(run_root) or not source.is_file():
        raise RuntimeError(f"Prediction cache source is missing or outside the run: {path}")
    if sha256_file(source) != record.get("prediction_sha256"):
        raise RuntimeError(f"Prediction cache source checksum mismatch: {source}")
    prediction = np.load(source, allow_pickle=False)
    if prediction.shape != shape or not np.isfinite(prediction).all():
        raise RuntimeError(f"Prediction cache source has invalid shape or values: {source}")
    return prediction, record


def _publish_cached_prediction(
    cache_root: Path, identity: dict[str, Any], prediction_path: Path,
    adaptation: dict[str, Any], compute_seconds: float,
    evaluation_identity_sha: str,
) -> None:
    run_root = cache_root.parent.resolve()
    prediction_path = prediction_path.resolve()
    if not prediction_path.is_relative_to(run_root):
        raise RuntimeError("Prediction cache source must remain inside this run")
    path = _cache_record_path(cache_root, identity)
    digest = sha256_file(prediction_path)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("identity") != identity or existing.get("prediction_sha256") != digest:
            raise RuntimeError(f"Prediction cache collision or inconsistent result: {path}")
        return
    save_json(path, {
        "identity": identity,
        "prediction_sha256": digest,
        "source_prediction": prediction_path.relative_to(run_root).as_posix(),
        "source_evaluation_identity_sha256": evaluation_identity_sha,
        "adaptation": adaptation,
        "compute_seconds": float(compute_seconds),
    })


def verify_prediction_cache(cache_root: str | Path) -> dict[str, Any]:
    """Rehash every shared prediction and its index for final handoff."""
    root = Path(cache_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Prediction cache directory is missing: {root}")
    by_method: dict[str, int] = {}
    count = 0
    for path in sorted(root.glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        identity = record.get("identity")
        if not isinstance(identity, dict) or path.resolve() != _cache_record_path(root, identity).resolve():
            raise RuntimeError(f"Prediction cache index identity differs: {path}")
        shape = tuple(identity.get("observation_shape", [])[1:])
        if len(shape) != 3 or any(not isinstance(axis, int) or axis < 1 for axis in shape):
            raise RuntimeError(f"Prediction cache has an invalid grid shape: {path}")
        if _read_cached_prediction(root, identity, shape) is None:
            raise RuntimeError(f"Prediction cache entry disappeared during verification: {path}")
        seconds = record.get("compute_seconds")
        if not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds < 0:
            raise RuntimeError(f"Prediction cache has invalid compute time: {path}")
        method = str(identity["method"])
        by_method[method] = by_method.get(method, 0) + 1
        count += 1
    if count == 0:
        raise RuntimeError(f"Prediction cache has no entries: {root}")
    return {"entries": count, "by_method": by_method,
            "source_path_policy": "relative to run root and SHA256 verified"}


def _add_reliability(
    totals: dict[str, np.ndarray], prediction: np.ndarray,
    target: np.ndarray, known: np.ndarray,
) -> None:
    blind_probability = prediction[~known].ravel()
    blind_truth = target[~known].ravel()
    bins = np.clip((blind_probability * 10).astype(np.int64), 0, 9)
    totals["count"] += np.bincount(bins, minlength=10)
    totals["probability_sum"] += np.bincount(
        bins, weights=blind_probability, minlength=10)
    totals["truth_sum"] += np.bincount(
        bins, weights=blind_truth, minlength=10)


def _write_case_csv_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def evaluate(manifest: str, config_path: str, out_dir: str, split: str,
             methods: list[str], checkpoints: dict[str, str], seed: int = 42,
             num_holes: int | None = None, noise_rate: float = 0.0,
             limit: int = 0, project_observed: bool = False,
             threshold_file: str | None = None,
             sample_ids: list[str] | None = None,
             prediction_cache_root: str | None = None) -> Path:
    """Evaluate fixed cases with an identity-checked, per-case resume journal."""
    from experiment_data import ProtocolDataset, validate_manifest
    from repro_metrics import volume_metrics
    allowed_methods = {
        "idw", "rbf", "ik", "mps", "unet", "base_vae", "cgan",
        "geovae", "geovae_latent", "geovae_joint",
    }
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Evaluation needs a nonempty method list without duplicates")
    if any(method not in allowed_methods for method in methods):
        raise ValueError("Unknown evaluation method")
    manifest_data = validate_manifest(manifest)
    manifest_sha = sha256_file(manifest)
    config = load_config(config_path)
    device = select_device(str(config.get("device", "auto")))
    holes = int(num_holes if num_holes is not None else config["num_holes"])
    dataset = ProtocolDataset(
        manifest, split, num_holes=holes,
        samples_per_hole=int(config["samples_per_hole"]),
        mask_seed=int(config["mask_seed"]), noise_rate=noise_rate, heldout_holes=4,
    )
    if not len(dataset):
        raise RuntimeError(f"No cases in split {split}")
    threshold_values = {method: float(config["threshold"]) for method in methods}
    selection_sha = None
    selection_status = None
    if threshold_file:
        selection_path = Path(threshold_file).resolve()
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("manifest_sha256") != manifest_sha:
            raise RuntimeError("Threshold selection manifest does not match")
        if selection.get("config_sha256") != stable_hash(config):
            raise RuntimeError("Threshold selection configuration does not match")
        if selection.get("evaluation_source_sha256") != source_hash():
            raise RuntimeError("Threshold selection source code differs from this evaluation")
        if not set(methods).issubset(selection.get("thresholds", {})):
            raise RuntimeError("Threshold selection lacks requested methods")
        selection_status = selection.get("status")
        if split == "test" and limit == 0 and len(dataset) == 750:
            if selection_status != "validation_frozen":
                raise RuntimeError("Full sealed test needs formal validation thresholds")
        for method in methods:
            value = float(selection["thresholds"][method])
            if not 0.0 < value < 1.0:
                raise ValueError(f"Invalid selected threshold for {method}")
            threshold_values[method] = value
        selection_sha = sha256_file(selection_path)
    elif split == "test" and limit == 0 and len(dataset) == 750:
        raise RuntimeError("Full sealed test requires --thresholds from validation")

    if sample_ids is not None:
        if limit != 0:
            raise ValueError("Explicit sample_ids require limit=0")
        if not sample_ids or len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample_ids must be a nonempty list without duplicates")
        if any(not isinstance(sample_id, str) or sample_id not in dataset.metadata
               for sample_id in sample_ids):
            raise ValueError("sample_ids must all belong to the selected split")
        case_ids = list(sample_ids)
    else:
        count = len(dataset) if limit <= 0 else min(limit, len(dataset))
        case_ids = [entry["sample_id"] for entry in dataset.entries[:count]]
    for sample_id in case_ids:
        if Path(sample_id).name != sample_id or "/" in sample_id or "\\" in sample_id:
            raise ValueError(f"Unsafe sample_id in protocol: {sample_id}")
    count = len(case_ids)

    requested_models = {
        {"geovae": "geovae", "geovae_latent": "geovae",
         "geovae_joint": "geovae", "base_vae": "base_vae",
         "unet": "unet", "cgan": "cgan"}[method]
        for method in methods
        if method in {"geovae", "geovae_latent", "geovae_joint",
                      "base_vae", "unet", "cgan"}
    }
    models = {}
    checkpoint_hashes = {}
    for kind in sorted(requested_models):
        if kind not in checkpoints:
            raise RuntimeError(f"Method list requires --checkpoint {kind}=PATH")
        models[kind], _ = _load_checked_model(kind, checkpoints[kind],
                                              manifest_sha, config, seed, device)
        checkpoint_hashes[kind] = sha256_file(checkpoints[kind])
    training_image = None
    training_image_id = None
    training_image_sha256 = None
    if "mps" in methods:
        train_dataset = ProtocolDataset(manifest, "train", num_holes=holes,
                                        samples_per_hole=int(config["samples_per_hole"]),
                                        mask_seed=int(config["mask_seed"]))
        training_image_id = str(config.get("mps_training_image_id") or
                                train_dataset.entries[0]["sample_id"])
        if training_image_id not in train_dataset.metadata:
            raise ValueError("MPS training image ID must belong to the training split")
        training_image = train_dataset.get_case(training_image_id)["target"]
        training_row = train_dataset.metadata[training_image_id]
        training_path = (Path(manifest).resolve().parent / training_row["path"]).resolve()
        if (not training_path.is_relative_to(Path(manifest).resolve().parent)
                or sha256_file(training_path) != training_row["sha256"]):
            raise RuntimeError("MPS training image checksum differs from the protocol")
        training_image_sha256 = hashlib.sha256(
            np.ascontiguousarray(training_image).tobytes()).hexdigest()

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_root = Path(prediction_cache_root).resolve() if prediction_cache_root else None
    if cache_root is not None and not output.is_relative_to(cache_root.parent):
        raise ValueError("Prediction cache and evaluation must share a run root")
    evaluated_source_sha = source_hash()
    evaluated_environment = environment_info()
    identity = {
        "manifest_sha256": manifest_sha,
        "manifest_path": str(Path(manifest).resolve()),
        "config_sha256": stable_hash(config),
        "source_sha256": evaluated_source_sha,
        "split": split,
        "seed": seed,
        "mask_seed": int(config["mask_seed"]),
        "num_holes": holes,
        "samples_per_hole": int(config["samples_per_hole"]),
        "heldout_holes": 4,
        "noise_rate": noise_rate,
        "methods": methods,
        "case_ids": case_ids,
        "case_selection_sha256": stable_hash(sample_ids) if sample_ids is not None else None,
        "project_observed": project_observed,
        "threshold_selection_sha256": selection_sha,
        "threshold_selection_status": selection_status,
        "method_thresholds": threshold_values,
        "checkpoints": checkpoint_hashes,
        "mps_training_image_id": training_image_id,
        "limit": limit,
        "environment": evaluated_environment,
    }
    identity_path = output / "evaluation_intent.json"
    if identity_path.is_file():
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
        if previous != identity:
            different = sorted(key for key in set(previous) | set(identity)
                               if previous.get(key) != identity.get(key))
            raise RuntimeError(
                f"Existing evaluation cannot be resumed because identity differs: {different}. "
                f"Use a new output directory instead of overwriting {output}.")
    else:
        if any((output / name).exists()
               for name in ("evaluation.json", "per_case.csv", "predictions",
                            "predictions_raw", "journal")):
            raise RuntimeError(
                f"Existing evaluation lacks an identity journal: {output}. "
                "Use a new output directory rather than mixing results.")
        save_json(identity_path, identity)
    identity_sha = stable_hash(identity)
    # Make a previous complete summary unavailable while any journal entry is
    # being revalidated or repaired. The final summary replaces this marker.
    save_json(output / "evaluation.json", {
        "complete": False,
        "status": "running",
        "journal_identity_sha256": identity_sha,
        "expected_cases": count,
        "methods": methods,
    })

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    failure_history: list[dict[str, Any]] = []
    reused = computed = cache_hits = new_predictions = 0
    reliability = {method: {"count": np.zeros(10, dtype=np.int64),
                            "probability_sum": np.zeros(10, dtype=np.float64),
                            "truth_sum": np.zeros(10, dtype=np.float64)}
                   for method in methods}
    print(f"Evaluation {split}, seed {seed}: {count} cases, "
          f"{len(methods)} methods; resume journal at {output / 'journal'}",
          flush=True)
    evaluation_started = time.perf_counter()
    for case_index, sample_id in enumerate(case_ids, 1):
        case = dataset.get_case(sample_id)
        sample_row = dataset.metadata[sample_id]
        grid_path = (Path(manifest).resolve().parent / sample_row["path"]).resolve()
        if (not grid_path.is_relative_to(Path(manifest).resolve().parent)
                or sha256_file(grid_path) != sample_row["sha256"]):
            raise RuntimeError(f"Protocol grid checksum differs: {sample_id}")
        obs = case["obs"]
        target = case["target"]
        hidden = case["hidden_mask"]
        parent_id = dataset.metadata[sample_id]["parent_id"]
        known = obs[1] >= 0.5
        for method in methods:
            case_seed = int(hashlib.sha256(
                f"{config['mask_seed']}:{sample_id}:{method}".encode()
            ).hexdigest()[:8], 16)
            cache_identity = (_prediction_cache_identity(
                method=method, config=config, manifest_sha=manifest_sha,
                split=split, sample_id=sample_id,
                sample_grid_sha=str(dataset.metadata[sample_id]["sha256"]),
                observation=obs, holes=holes, noise_rate=noise_rate,
                case_seed=case_seed, checkpoint_hashes=checkpoint_hashes,
                training_image_id=training_image_id,
                training_image_sha256=training_image_sha256,
                device=device, source_sha=evaluated_source_sha,
                environment=evaluated_environment,
            ) if cache_root is not None else None)
            journal_path = output / "journal" / method / f"{sample_id}.json"
            previous_case: dict[str, Any] = {}
            attempt_history: list[dict[str, str]] = []
            if journal_path.is_file():
                previous_case = json.loads(journal_path.read_text(encoding="utf-8"))
                if (previous_case.get("evaluation_identity_sha256") != identity_sha
                        or previous_case.get("sample_id") != sample_id
                        or previous_case.get("method") != method):
                    raise RuntimeError(f"Case journal identity differs: {journal_path}")
                attempt_history = list(previous_case.get("failure_history", []))
                if previous_case.get("status") == "complete":
                    try:
                        record = previous_case["record"]
                        if (record.get("sample_id") != sample_id
                                or record.get("method") != method
                                or record.get("parent_id") != parent_id
                                or float(record.get("threshold_used")) != threshold_values[method]):
                            raise ValueError("Case record identity differs")
                        expected_prediction = Path("predictions") / method / f"{sample_id}.npy"
                        prediction = _read_prediction_for_resume(
                            output, record, "prediction", expected_prediction,
                            target.shape)
                        if project_observed:
                            expected_raw = Path("predictions_raw") / method / f"{sample_id}.npy"
                            _read_prediction_for_resume(
                                output, record, "raw_prediction", expected_raw,
                                target.shape)
                        elif record.get("raw_prediction_file") or record.get("raw_prediction_sha256"):
                            raise ValueError("Unexpected projected prediction in case journal")
                        record.setdefault("evaluation_seconds", record["seconds"])
                        record.setdefault("prediction_origin", "journal")
                        record.setdefault(
                            "prediction_cache_key",
                            stable_hash(cache_identity) if cache_identity is not None else "")
                        record.setdefault("prediction_source_file", "")
                        record.setdefault("prediction_source_sha256", "")
                        records.append(record)
                        reused += 1
                        _add_reliability(reliability[method], prediction, target, known)
                        if cache_root is not None and cache_identity is not None:
                            raw_source = (output / record["raw_prediction_file"]
                                          if project_observed else
                                          output / record["prediction_file"])
                            _publish_cached_prediction(
                                cache_root, cache_identity, raw_source,
                                json.loads(record.get("adaptation") or "{}"),
                                float(record["seconds"]), identity_sha)
                        failure_history.extend(attempt_history)
                        continue
                    except (KeyError, TypeError, ValueError, OSError) as error:
                        attempt_history.append({
                            "sample_id": sample_id, "method": method,
                            "error_type": "JournalValidationError",
                            "message": str(error),
                        })
            started = time.perf_counter()
            try:
                cached = (_read_cached_prediction(cache_root, cache_identity, target.shape)
                          if cache_root is not None and cache_identity is not None else None)
                if cached is None:
                    prediction, adaptation = _predict(
                        method, obs, models, config, device, training_image, case_seed)
                    prediction_origin = "computed"
                    compute_seconds = time.perf_counter() - started
                    source_prediction = ""
                    source_prediction_sha = ""
                    new_predictions += 1
                else:
                    prediction, cache_record = cached
                    adaptation = cache_record.get("adaptation", {})
                    prediction_origin = "shared_cache"
                    compute_seconds = float(cache_record["compute_seconds"])
                    source_prediction = str(cache_record["source_prediction"])
                    source_prediction_sha = str(cache_record["prediction_sha256"])
                    cache_hits += 1
                if prediction.shape != target.shape:
                    raise ValueError(f"Prediction shape {prediction.shape} != {target.shape}")
                raw_prediction = prediction.copy()
                threshold = threshold_values[method]
                before_error = float(np.mean((prediction[known] >= threshold) !=
                                             (obs[0][known] >= 0.5))) if known.any() else None
                if project_observed:
                    prediction = prediction.copy()
                    prediction[known] = obs[0][known]
                after_error = float(np.mean((prediction[known] >= threshold) !=
                                            (obs[0][known] >= 0.5))) if known.any() else None
                metrics = volume_metrics(
                    prediction, target, known, threshold=threshold,
                    hidden_mask=hidden)
                prediction_path = output / "predictions" / method / f"{sample_id}.npy"
                _save_npy_atomic(prediction_path, prediction)
                raw_path = None
                if project_observed:
                    raw_path = output / "predictions_raw" / method / f"{sample_id}.npy"
                    _save_npy_atomic(raw_path, raw_prediction)
                record = {
                    "sample_id": sample_id, "parent_id": parent_id, "method": method,
                    "prediction_file": str(prediction_path.relative_to(output)).replace("\\", "/"),
                    "prediction_sha256": sha256_file(prediction_path),
                    "raw_prediction_file": (
                        str(raw_path.relative_to(output)).replace("\\", "/")
                        if raw_path is not None else ""),
                    "raw_prediction_sha256": (
                        sha256_file(raw_path) if raw_path is not None else ""),
                    "split": split, "seed": seed, "mask_seed": int(config["mask_seed"]),
                    "num_holes": holes, "noise_rate": noise_rate,
                    "threshold_used": threshold,
                    "seconds": compute_seconds,
                    "evaluation_seconds": time.perf_counter() - started,
                    "prediction_origin": prediction_origin,
                    "prediction_cache_key": (stable_hash(cache_identity)
                                             if cache_identity is not None else ""),
                    "prediction_source_file": source_prediction,
                    "prediction_source_sha256": source_prediction_sha,
                    "preprojection_input_error": before_error,
                    "postprojection_input_error": after_error,
                    "preprojection_input_mse": (float(np.mean(
                        (raw_prediction[known] - obs[0][known]) ** 2))
                        if known.any() else None),
                    "preprojection_input_violation_count": int(np.count_nonzero(
                        (raw_prediction[known] >= threshold) != (obs[0][known] >= 0.5))),
                    "adaptation": json.dumps(adaptation, default=str, sort_keys=True),
                    **metrics,
                }
                save_json(journal_path, {
                    "evaluation_identity_sha256": identity_sha,
                    "sample_id": sample_id,
                    "method": method,
                    "status": "complete",
                    "record": record,
                    "failure_history": attempt_history,
                })
                if cache_root is not None and cache_identity is not None:
                    _publish_cached_prediction(
                        cache_root, cache_identity,
                        raw_path if raw_path is not None else prediction_path,
                        adaptation, compute_seconds, identity_sha)
                records.append(record)
                computed += 1
                _add_reliability(reliability[method], prediction, target, known)
            except Exception as error:
                failure = {
                    "sample_id": sample_id, "method": method,
                    "error_type": type(error).__name__, "message": str(error),
                }
                failures.append(failure)
                attempt_history.append(failure)
                save_json(journal_path, {
                    "evaluation_identity_sha256": identity_sha,
                    "sample_id": sample_id,
                    "method": method,
                    "status": "failed",
                    "failure_history": attempt_history,
                })
            failure_history.extend(attempt_history)
        if case_index % 25 == 0 or case_index == count:
            elapsed = time.perf_counter() - evaluation_started
            eta_seconds = elapsed * (count - case_index) / case_index
            print(f"Evaluation {split}, seed {seed}: {case_index}/{count} cases; "
                  f"journal={reused}, cache={cache_hits}, new={new_predictions}, "
                  f"failures={len(failures)}, eta~{eta_seconds / 60:.0f} min",
                  flush=True)

    metadata = {
        "manifest": str(Path(manifest).resolve()),
        "manifest_sha256": manifest_sha,
        "config_sha256": stable_hash(config),
        "source_sha256": evaluated_source_sha,
        "split": split, "seed": seed, "mask_seed": int(config["mask_seed"]),
        "evaluation_status": (
            "formal_full" if split == "test" and limit == 0 and count == 750
            and selection_status == "validation_frozen" else
            "draft_subset" if split == "test" else "non_test"),
        "num_holes": holes, "noise_rate": noise_rate,
        "methods": methods, "expected_cases": count,
        "complete": not failures and all(sum(x["method"] == method for x in records) == count
                                         for method in methods),
        "project_observed": project_observed,
        "threshold_selection_sha256": selection_sha,
        "threshold_selection_status": selection_status,
        "method_thresholds": threshold_values,
        "checkpoints": checkpoint_hashes,
        "mps_training_image_id": training_image_id,
        "case_selection_sha256": identity["case_selection_sha256"],
        "case_ids_sha256": stable_hash(case_ids),
        "limit": limit,
        "distance_unit": "voxel",
        "environment": evaluated_environment,
        "prediction_cache": {
            "enabled": cache_root is not None,
            "root": cache_root.name if cache_root is not None else None,
            "journal_reused": reused, "shared_cache_hits": cache_hits,
            "new_predictions": new_predictions,
            "shared_cache_identity_schema": 1,
        },
        "real_data_policy": manifest_data.get("real_data_policy"),
        "journal_identity_sha256": identity_sha,
    }
    save_json(output / "failures.json", failures)
    save_json(output / "failure_history.json", failure_history)
    save_json(output / "reliability.json", {
        method: {name: values.tolist() for name, values in totals.items()}
        for method, totals in reliability.items()
    })
    if records:
        _write_case_csv_atomic(output / "per_case.csv", records)
    save_json(output / "evaluation.json", metadata)
    if not metadata["complete"]:
        raise RuntimeError(
            f"Evaluation incomplete: {len(failures)} failures. "
            f"See {output / 'failures.json'}")
    return output / "per_case.csv"
