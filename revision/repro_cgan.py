"""Conditional 3D GAN baseline for the manifest-driven orebody protocol.

The discriminator receives the same sparse two-channel observation as the
generator plus a candidate target. Its binary cross-entropy objective is
combined with whole-volume supervised reconstruction. No test target enters
training, model selection, or prediction.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConditionalGenerator3D(nn.Module):
    """Three-level 3D U-Net conditioned on two sparse drillhole channels."""

    def __init__(self, base_channels: int = 16):
        super().__init__()
        if base_channels < 4:
            raise ValueError("base_channels must be at least 4")
        width = base_channels
        self.base_channels = width
        self.enc0 = _ConvBlock(3, width)
        self.down1 = nn.Conv3d(width, 2 * width, 4, stride=2, padding=1)
        self.enc1 = _ConvBlock(2 * width, 2 * width)
        self.down2 = nn.Conv3d(2 * width, 4 * width, 4, stride=2, padding=1)
        self.enc2 = _ConvBlock(4 * width, 4 * width)
        self.down3 = nn.Conv3d(4 * width, 8 * width, 4, stride=2, padding=1)
        self.bottleneck = _ConvBlock(8 * width, 8 * width)
        self.dec2 = _ConvBlock(12 * width, 4 * width)
        self.dec1 = _ConvBlock(6 * width, 2 * width)
        self.dec0 = _ConvBlock(3 * width, width)
        self.output = nn.Conv3d(width, 1, 1)

    def forward(self, obs: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if obs.ndim != 5 or obs.shape[1] != 2:
            raise ValueError("Generator observation must have shape [B,2,D,H,W]")
        if any(size % 8 for size in obs.shape[-3:]):
            raise ValueError("Generator grid dimensions must be divisible by 8")
        if noise is None:
            noise = torch.zeros_like(obs[:, :1])
        if noise.shape != obs[:, :1].shape:
            raise ValueError("Noise must have shape [B,1,D,H,W]")
        e0 = self.enc0(torch.cat((obs, noise), dim=1))
        e1 = self.enc1(self.down1(e0))
        e2 = self.enc2(self.down2(e1))
        h = self.bottleneck(self.down3(e2))
        h = F.interpolate(h, size=e2.shape[-3:], mode="trilinear", align_corners=False)
        h = self.dec2(torch.cat((h, e2), dim=1))
        h = F.interpolate(h, size=e1.shape[-3:], mode="trilinear", align_corners=False)
        h = self.dec1(torch.cat((h, e1), dim=1))
        h = F.interpolate(h, size=e0.shape[-3:], mode="trilinear", align_corners=False)
        h = self.dec0(torch.cat((h, e0), dim=1))
        return self.output(h).squeeze(1)


class ConditionalDiscriminator3D(nn.Module):
    """Patch discriminator over observation and candidate occupancy."""

    def __init__(self, base_channels: int = 16):
        super().__init__()
        width = base_channels
        self.features = nn.Sequential(
            nn.Conv3d(3, width, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(width, 2 * width, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(2 * width), 2 * width),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(2 * width, 4 * width, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(4 * width), 4 * width),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(4 * width, 8 * width, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_groups(8 * width), 8 * width),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(8 * width, 1, 1),
        )

    def forward(self, obs: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        if candidate.ndim == 4:
            candidate = candidate.unsqueeze(1)
        if obs.ndim != 5 or obs.shape[1] != 2 or candidate.shape != obs[:, :1].shape:
            raise ValueError("Discriminator expects obs [B,2,D,H,W] and candidate [B,D,H,W]")
        return self.features(torch.cat((obs, candidate), dim=1))


def make_cgan_generator(config: dict[str, Any]) -> ConditionalGenerator3D:
    return ConditionalGenerator3D(int(config.get("cgan_base_channels", 16)))


@torch.no_grad()
def predict_cgan(
    generator: ConditionalGenerator3D,
    obs: torch.Tensor,
    seed: int | Sequence[int] = 0,
    realizations: int = 1,
) -> torch.Tensor:
    """Return an average occupancy probability, [B,D,H,W], with fixed noise."""
    if obs.ndim != 5 or obs.shape[1] != 2:
        raise ValueError("obs must have shape [B,2,D,H,W]")
    if realizations < 1:
        raise ValueError("realizations must be positive")
    if isinstance(seed, int):
        case_seeds = [seed + case for case in range(obs.shape[0])]
    else:
        case_seeds = [int(value) for value in seed]
        if len(case_seeds) != obs.shape[0]:
            raise ValueError("One seed is required per case")
    prior_mode = generator.training
    generator.eval()
    try:
        probabilities = []
        for draw in range(realizations):
            noise_parts = []
            for case_seed in case_seeds:
                random_generator = torch.Generator(device=obs.device)
                random_generator.manual_seed(int(case_seed) + draw * 104729)
                noise_parts.append(torch.randn(
                    (1, 1, *obs.shape[-3:]), generator=random_generator,
                    device=obs.device, dtype=obs.dtype,
                ))
            noise = torch.cat(noise_parts, dim=0)
            probabilities.append(torch.sigmoid(generator(obs, noise)))
        return torch.stack(probabilities, dim=0).mean(dim=0)
    finally:
        generator.train(prior_mode)


def _source_sha256() -> str:
    from repro_pipeline import source_hash, sha256_file

    digest = hashlib.sha256()
    digest.update(source_hash().encode("ascii"))
    digest.update(sha256_file(__file__).encode("ascii"))
    return digest.hexdigest()


def _checkpoint(
    generator: ConditionalGenerator3D,
    discriminator: ConditionalDiscriminator3D,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    config: dict[str, Any],
    manifest_sha: str,
    seed: int,
    best_validation_bce: float,
    loader_rng: torch.Generator,
    stale_epochs: int,
) -> dict[str, Any]:
    from repro_pipeline import stable_hash

    return {
        "format_version": 2,
        "model_kind": "cgan",
        "model_state": generator.state_dict(),
        "discriminator_state": discriminator.state_dict(),
        "optimizer_g_state": generator_optimizer.state_dict(),
        "optimizer_d_state": discriminator_optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": int(epoch),
        "config": config,
        "config_sha256": stable_hash(config),
        "manifest_sha256": manifest_sha,
        "source_sha256": _source_sha256(),
        "seed": int(seed),
        "best_validation_bce": float(best_validation_bce),
        "stale_epochs": int(stale_epochs),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_rng_state": loader_rng.get_state(),
    }


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def _validate(
    generator: ConditionalGenerator3D,
    loader: DataLoader,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    generator.eval()
    total_bce, blind_intersection, blind_union, total_voxels = 0.0, 0.0, 0.0, 0
    for obs, target, sample_ids in loader:
        obs, target = obs.to(device), target.to(device)
        seeds = [
            int(hashlib.sha256(f"{seed}:{sample_id}:cgan-val".encode()).hexdigest()[:8], 16)
            for sample_id in sample_ids
        ]
        probability = predict_cgan(generator, obs, seed=seeds)
        total_bce += float(F.binary_cross_entropy(
            probability.clamp(1e-6, 1 - 1e-6), target, reduction="sum"
        ).item())
        blind = obs[:, 1] < 0.5
        predicted, truth = probability >= 0.5, target >= 0.5
        blind_intersection += float((predicted & truth & blind).sum().item())
        blind_union += float(((predicted | truth) & blind).sum().item())
        total_voxels += target.numel()
    if total_voxels == 0:
        raise RuntimeError("cGAN validation split is empty")
    return {
        "full_bce": total_bce / total_voxels,
        "blind_iou": blind_intersection / max(blind_union, 1.0),
    }


def train_cgan(
    manifest: str,
    config_path: str,
    run_dir: str,
    seed: int,
    resume: str | None = None,
    max_epochs: int | None = None,
) -> Path:
    """Train a 3D conditional GAN using only manifest train and val partitions."""
    from experiment_data import ProtocolDataset, validate_manifest
    from repro_pipeline import (
        load_config, save_json, select_device, set_seed, environment_info,
        sha256_file, stable_hash,
    )

    config = load_config(config_path)
    validate_manifest(manifest)
    manifest_sha = sha256_file(manifest)
    device = select_device(str(config.get("device", "auto")))
    set_seed(seed)
    grid_size = tuple(int(value) for value in config["grid_size"])
    if len(grid_size) != 3 or any(size % 8 for size in grid_size):
        raise ValueError("cGAN grid_size must contain three multiples of 8")
    batch_size = int(config["microbatch"])
    effective_batch = int(config["effective_batch"])
    if batch_size < 1 or effective_batch < batch_size or effective_batch % batch_size:
        raise ValueError("effective_batch must be a positive multiple of microbatch")
    accumulation = effective_batch // batch_size

    def dataset(split: str) -> ProtocolDataset:
        return ProtocolDataset(
            manifest, split,
            num_holes=int(config["num_holes"]),
            samples_per_hole=int(config["samples_per_hole"]),
            mask_seed=int(config["mask_seed"]),
        )

    training = dataset("train")
    validation = dataset("val")
    if not training or not validation:
        raise RuntimeError("Both training and validation partitions are required")
    generator_rng = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        training, batch_size=batch_size, shuffle=True,
        generator=generator_rng, num_workers=int(config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        validation, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    base_channels = int(config.get("cgan_base_channels", 16))
    generator = ConditionalGenerator3D(base_channels).to(device)
    discriminator = ConditionalDiscriminator3D(base_channels).to(device)
    g_optimizer = torch.optim.Adam(
        generator.parameters(), lr=float(config.get("cgan_lr_g", 2e-4)),
        betas=(0.5, 0.999),
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=float(config.get("cgan_lr_d", 2e-4)),
        betas=(0.5, 0.999),
    )
    amp = bool(config.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    run = Path(run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    source_sha = _source_sha256()
    save_json(run / "run.json", {
        "kind": "cgan", "seed": seed, "config": config,
        "manifest": str(Path(manifest).resolve()),
        "manifest_sha256": manifest_sha, "config_sha256": stable_hash(config),
        "source_sha256": source_sha, "torch": torch.__version__,
        "device": str(device),
        "environment": environment_info(),
        "generator_parameters": sum(p.numel() for p in generator.parameters()),
        "discriminator_parameters": sum(p.numel() for p in discriminator.parameters()),
    })

    start_epoch, best_bce, stale_epochs = 0, math.inf, 0
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        required = {
            "model_kind": "cgan", "manifest_sha256": manifest_sha,
            "config_sha256": stable_hash(config), "source_sha256": source_sha,
            "seed": int(seed),
        }
        for key, expected in required.items():
            if checkpoint.get(key) != expected:
                raise RuntimeError(f"cGAN resume checkpoint {key} mismatch")
        generator.load_state_dict(checkpoint["model_state"], strict=True)
        discriminator.load_state_dict(checkpoint["discriminator_state"], strict=True)
        g_optimizer.load_state_dict(checkpoint["optimizer_g_state"])
        d_optimizer.load_state_dict(checkpoint["optimizer_d_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"])
        best_bce = float(checkpoint["best_validation_bce"])
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
            np.random.set_state(checkpoint["numpy_rng_state"])
            torch.set_rng_state(checkpoint["torch_rng_state"])
            if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
            generator_rng.set_state(checkpoint["loader_rng_state"])

    epochs = int(max_epochs if max_epochs is not None else config["epochs"])
    if epochs < 1:
        raise ValueError("epochs must be positive")
    lambda_recon = float(config.get("cgan_lambda_recon", 10.0))
    lambda_observed = float(config.get("cgan_lambda_observed", config["lambda_observed"]))
    if lambda_recon <= 0 or lambda_observed < 0:
        raise ValueError("cGAN reconstruction weights are invalid")
    patience = int(config.get("patience", 20))
    for epoch in range(start_epoch + 1, epochs + 1):
        started = time.perf_counter()
        generator.train()
        discriminator.train()
        g_optimizer.zero_grad(set_to_none=True)
        d_optimizer.zero_grad(set_to_none=True)
        sums = {"generator": 0.0, "discriminator": 0.0,
                "adversarial": 0.0, "full_bce": 0.0, "observed_bce": 0.0}
        batches = 0
        for batch_index, (obs, target, _) in enumerate(train_loader):
            obs = obs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if tuple(target.shape[-3:]) != grid_size:
                raise ValueError("Dataset grid does not match cGAN configuration")
            noise = torch.randn(
                (obs.shape[0], 1, *obs.shape[-3:]),
                device=device, dtype=obs.dtype,
            )
            with torch.autocast(device_type=device.type, enabled=amp):
                fake_logits = generator(obs, noise)
                fake_probability = torch.sigmoid(fake_logits)
                real_score = discriminator(obs, target)
                fake_score = discriminator(obs, fake_probability.detach())
                d_loss = 0.5 * (
                    F.binary_cross_entropy_with_logits(real_score, torch.ones_like(real_score))
                    + F.binary_cross_entropy_with_logits(fake_score, torch.zeros_like(fake_score))
                )
            if not bool(torch.isfinite(d_loss)):
                raise FloatingPointError(f"Nonfinite discriminator loss at epoch {epoch}")
            window_start = (batch_index // accumulation) * effective_batch
            window_samples = min(effective_batch, len(training) - window_start)
            window_weight = obs.shape[0] / window_samples
            scaler.scale(d_loss * window_weight).backward()

            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            try:
                with torch.autocast(device_type=device.type, enabled=amp):
                    fake_score_for_g = discriminator(obs, fake_probability)
                    adversarial = F.binary_cross_entropy_with_logits(
                        fake_score_for_g, torch.ones_like(fake_score_for_g)
                    )
                    full_bce = F.binary_cross_entropy_with_logits(fake_logits, target)
                    voxel_bce = F.binary_cross_entropy_with_logits(
                        fake_logits, target, reduction="none"
                    )
                    observed_bce = (
                        voxel_bce * obs[:, 1]
                    ).sum() / obs[:, 1].sum().clamp_min(1)
                    g_loss = adversarial + lambda_recon * full_bce + lambda_observed * observed_bce
                if not bool(torch.isfinite(g_loss)):
                    raise FloatingPointError(f"Nonfinite generator loss at epoch {epoch}")
                scaler.scale(g_loss * window_weight).backward()
            finally:
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)

            sums["generator"] += float(g_loss.detach())
            sums["discriminator"] += float(d_loss.detach())
            sums["adversarial"] += float(adversarial.detach())
            sums["full_bce"] += float(full_bce.detach())
            sums["observed_bce"] += float(observed_bce.detach())
            batches += 1
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader):
                scaler.unscale_(g_optimizer)
                scaler.unscale_(d_optimizer)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                scaler.step(g_optimizer)
                scaler.step(d_optimizer)
                scaler.update()
                g_optimizer.zero_grad(set_to_none=True)
                d_optimizer.zero_grad(set_to_none=True)

        result = _validate(generator, val_loader, device, seed)
        record = {
            "epoch": epoch,
            **{f"train_{key}": value / batches for key, value in sums.items()},
            "val_full_bce": result["full_bce"],
            "val_blind_iou": result["blind_iou"],
            "seconds": time.perf_counter() - started,
        }
        with (run / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))
        if result["full_bce"] < best_bce - 1e-6:
            best_bce, stale_epochs = result["full_bce"], 0
            _save_checkpoint(
                run / "best.pt", _checkpoint(
                    generator, discriminator, g_optimizer, d_optimizer,
                    scaler, epoch, config, manifest_sha, seed, best_bce,
                    generator_rng, stale_epochs,
                )
            )
        else:
            stale_epochs += 1
        _save_checkpoint(
            run / "latest.pt", _checkpoint(
                generator, discriminator, g_optimizer, d_optimizer,
                scaler, epoch, config, manifest_sha, seed, best_bce,
                generator_rng, stale_epochs,
            )
        )
        if stale_epochs >= patience:
            print(f"cGAN validation early stop after epoch {epoch}; best BCE={best_bce:.6f}")
            break
    return run / "best.pt"
