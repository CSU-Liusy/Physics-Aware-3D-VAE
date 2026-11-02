"""Single-microbatch training and checkpoint-resume preflight."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from experiment_data import ProtocolDataset, validate_manifest
from repro_cgan import ConditionalDiscriminator3D, make_cgan_generator
from repro_pipeline import (compute_loss, load_config, make_model, save_json,
                            select_device, set_seed, sha256_file, source_hash,
                            stable_hash)


def smoke_batch(manifest: str, config_path: str, out_dir: str,
                device_name: str = "cuda") -> Path:
    config = load_config(config_path)
    protocol = validate_manifest(manifest)
    if tuple(protocol["grid_size"]) != tuple(config["grid_size"]):
        raise ValueError("Smoke-test configuration grid differs from manifest")
    device = select_device(device_name)
    set_seed(173)
    dataset = ProtocolDataset(
        manifest, "train", num_holes=int(config["num_holes"]),
        samples_per_hole=int(config["samples_per_hole"]), mask_seed=int(config["mask_seed"]))
    count = min(int(config["microbatch"]), len(dataset))
    if count < 1:
        raise RuntimeError("No training cases for smoke test")
    samples = [dataset[index] for index in range(count)]
    obs = torch.stack([item[0] for item in samples]).to(device)
    target = torch.stack([item[1] for item in samples]).to(device)
    model = make_model("geovae", config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]))
    amp = bool(device.type == "cuda" and config.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def one_step(current_model, current_optimizer, current_scaler, kind="geovae",
                 accumulation=1, return_meta=False):
        current_model.train()
        current_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            loss, _ = compute_loss(kind, current_model, obs, target,
                                   config, float(config["beta"]))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite single-batch training loss")
        scale_before = float(current_scaler.get_scale())
        current_scaler.scale(loss / accumulation).backward()
        current_scaler.unscale_(current_optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(current_model.parameters(), 1.0)
        current_scaler.step(current_optimizer)
        current_scaler.update()
        value = float(loss.detach())
        if return_meta:
            finite_norm = float(grad_norm)
            return value, {"grad_norm": finite_norm if math.isfinite(finite_norm) else None,
                           "scale_before": scale_before,
                           "scale_after": float(current_scaler.get_scale())}
        return value

    def verified_step(current_model, current_optimizer, current_scaler,
                      kind="geovae", accumulation=1):
        probe = next(current_model.parameters()).detach().clone()
        attempts = []
        for _ in range(18):
            loss, step = one_step(current_model, current_optimizer,
                                  current_scaler, kind=kind,
                                  accumulation=accumulation, return_meta=True)
            attempts.append(step)
            if (step["grad_norm"] is not None and step["grad_norm"] > 0
                    and step["scale_after"] >= step["scale_before"]
                    and not torch.equal(probe, next(current_model.parameters()))):
                return loss, attempts
        raise FloatingPointError(
            f"{kind} AMP preflight never completed a finite optimizer update")

    first_loss, geovae_attempts = verified_step(model, optimizer, scaler)
    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "single_batch.pt"
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "manifest_sha256": sha256_file(manifest),
        "config_sha256": stable_hash(config),
        "source_sha256": source_hash(),
    }, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if loaded["manifest_sha256"] != sha256_file(manifest) or loaded["config_sha256"] != stable_hash(config):
        raise RuntimeError("Single-batch checkpoint provenance mismatch")
    resumed_model = make_model("geovae", config).to(device)
    resumed_model.load_state_dict(loaded["model_state"], strict=True)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=float(config["lr"]))
    resumed_optimizer.load_state_dict(loaded["optimizer_state"])
    resumed_scaler = torch.amp.GradScaler("cuda", enabled=amp)
    resumed_scaler.load_state_dict(loaded["scaler_state"])
    second_loss, resumed_attempts = verified_step(
        resumed_model, resumed_optimizer, resumed_scaler)
    del model, optimizer, scaler, resumed_model, resumed_optimizer, resumed_scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    unet_model = make_model("unet", config).to(device)
    unet_optimizer = torch.optim.AdamW(unet_model.parameters(), lr=float(config["lr"]))
    unet_scaler = torch.amp.GradScaler("cuda", enabled=amp)
    accumulation = int(config["effective_batch"]) // int(config["microbatch"])
    unet_loss, unet_attempts = verified_step(
        unet_model, unet_optimizer, unet_scaler,
        kind="unet", accumulation=accumulation)
    del unet_model, unet_optimizer, unet_scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    generator = make_cgan_generator(config).to(device)
    discriminator = ConditionalDiscriminator3D(
        int(config.get("cgan_base_channels", 16))).to(device)
    g_optimizer = torch.optim.Adam(
        generator.parameters(), lr=float(config.get("cgan_lr_g", 2e-4)),
        betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=float(config.get("cgan_lr_d", 2e-4)),
        betas=(0.5, 0.999))
    gan_scaler = torch.amp.GradScaler("cuda", enabled=amp)
    lambda_recon = float(config.get("cgan_lambda_recon", 10.0))
    lambda_observed = float(config.get("cgan_lambda_observed", config["lambda_observed"]))
    gan_attempts = []
    for _ in range(18):
        generator.train()
        discriminator.train()
        g_optimizer.zero_grad(set_to_none=True)
        d_optimizer.zero_grad(set_to_none=True)
        g_probe = generator.output.weight.detach().clone()
        d_probe = discriminator.features[-1].weight.detach().clone()
        noise = torch.randn((count, 1, *obs.shape[-3:]),
                            device=device, dtype=obs.dtype)
        with torch.autocast(device_type=device.type, enabled=amp):
            fake_logits = generator(obs, noise)
            fake_probability = torch.sigmoid(fake_logits)
            real_score = discriminator(obs, target)
            fake_score = discriminator(obs, fake_probability.detach())
            d_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(real_score, torch.ones_like(real_score))
                + F.binary_cross_entropy_with_logits(fake_score, torch.zeros_like(fake_score)))
        if not bool(torch.isfinite(d_loss)):
            raise FloatingPointError("Nonfinite cGAN discriminator preflight loss")
        gan_scaler.scale(d_loss / accumulation).backward()
        for parameter in discriminator.parameters():
            parameter.requires_grad_(False)
        try:
            with torch.autocast(device_type=device.type, enabled=amp):
                fake_score_for_g = discriminator(obs, fake_probability)
                adversarial = F.binary_cross_entropy_with_logits(
                    fake_score_for_g, torch.ones_like(fake_score_for_g))
                full_bce = F.binary_cross_entropy_with_logits(fake_logits, target)
                voxel_bce = F.binary_cross_entropy_with_logits(
                    fake_logits, target, reduction="none")
                observed_bce = (voxel_bce * obs[:, 1]).sum() / obs[:, 1].sum().clamp_min(1)
                g_loss = adversarial + lambda_recon * full_bce + lambda_observed * observed_bce
            if not bool(torch.isfinite(g_loss)):
                raise FloatingPointError("Nonfinite cGAN generator preflight loss")
            gan_scaler.scale(g_loss / accumulation).backward()
        finally:
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)
        scale_before = float(gan_scaler.get_scale())
        gan_scaler.unscale_(g_optimizer)
        gan_scaler.unscale_(d_optimizer)
        g_grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        d_grad_norm = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
        gan_scaler.step(g_optimizer)
        gan_scaler.step(d_optimizer)
        gan_scaler.update()
        g_norm, d_norm = float(g_grad_norm), float(d_grad_norm)
        attempt = {
            "generator_loss": float(g_loss.detach()),
            "discriminator_loss": float(d_loss.detach()),
            "generator_grad_norm": g_norm if math.isfinite(g_norm) else None,
            "discriminator_grad_norm": d_norm if math.isfinite(d_norm) else None,
            "generator_updated": not torch.equal(g_probe, generator.output.weight),
            "discriminator_updated": not torch.equal(
                d_probe, discriminator.features[-1].weight),
            "scale_before": scale_before,
            "scale_after": float(gan_scaler.get_scale()),
        }
        gan_attempts.append(attempt)
        if (attempt["generator_grad_norm"] is not None
                and attempt["discriminator_grad_norm"] is not None
                and attempt["generator_grad_norm"] > 0
                and attempt["discriminator_grad_norm"] > 0
                and attempt["generator_updated"]
                and attempt["discriminator_updated"]
                and attempt["scale_after"] >= attempt["scale_before"]):
            break
    else:
        raise FloatingPointError("cGAN AMP preflight never updated both networks with finite gradients")
    del generator, discriminator, g_optimizer, d_optimizer, gan_scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    result = {
        "device": str(device), "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda" else None,
        "microbatch": count, "first_loss": first_loss,
        "resumed_loss": second_loss, "unet_loss": unet_loss,
        "geovae_amp_attempts": geovae_attempts,
        "geovae_resumed_amp_attempts": resumed_attempts,
        "unet_amp_attempts": unet_attempts,
        "cgan_generator_loss": attempt["generator_loss"],
        "cgan_discriminator_loss": attempt["discriminator_loss"],
        "cgan_amp_attempts": gan_attempts,
        "peak_allocated_gib": (torch.cuda.max_memory_allocated(device) / 1024 ** 3)
            if device.type == "cuda" else None,
        "checkpoint": str(checkpoint),
        "manifest_sha256": loaded["manifest_sha256"],
        "config_sha256": loaded["config_sha256"],
        "source_sha256": loaded["source_sha256"],
        "passed": True,
    }
    save_json(output / "smoke.json", result)
    return output / "smoke.json"
