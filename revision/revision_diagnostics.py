"""Evidence-producing revision diagnostics, with no posterior-sampling claim.

The synthetic diagnostic accepts observations alone during optimization. Real
reference shapes require an explicitly supplied external private directory.
The real-data public exporter is an allow-list of aggregate numeric fields.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from experiment_data import ProtocolDataset, make_drill_masks, verify_protocol
from model import ConvVAE3D, LoRAConv3d, LoRALinear, apply_lora_to_model
from r12_sampling import observation_consistency, threshold_key
from revision_numerics import configure_evaluation, metadata as numerical_metadata
from repro_metrics import volume_metrics
from repro_pipeline import (
    ROOT, _load_checked_model, _predict, _save_npy_atomic, environment_info,
    load_config, save_json, select_device, sha256_file, source_hash, stable_hash,
)

VARIANCES = (0.001, 0.01, 0.1)
ACCEPTANCE_THRESHOLDS = (0.0001, 0.001, 0.01)
PRIMARY_ACCEPTANCE_THRESHOLD = 0.001
SOURCE_FILES = ("revision_diagnostics.py", "revision_numerics.py", "r12_sampling.py", "r12_data.py",
                "model.py", "experiment_data.py", "repro_pipeline.py",
                "repro_metrics.py", "repro_baselines.py", "repro_cgan.py")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Cached JSON rows have alphabetized keys. A fixed header keeps a resumed
    # CSV byte-identical to its original computation.
    keys = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True)
                         if isinstance(value, (list, dict)) else value
                         for key, value in row.items()} for row in rows)
    temporary.replace(path)


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _seed(*parts: Any) -> int:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _source_identity() -> dict:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def _check_deadline(deadline_epoch: float | None) -> None:
    if deadline_epoch is not None and time.time() >= deadline_epoch:
        raise TimeoutError("Revision computation budget exhausted; saved cases remain resumable")


def select_parent_cases(manifest: dict, split: str, count: int) -> list[str]:
    """Select parent and variant from IDs alone, never from outcomes or shapes."""
    groups: dict[str, list[str]] = {}
    for row in manifest["samples"]:
        if row["split"] == split:
            groups.setdefault(row["parent_id"], []).append(row["sample_id"])
    if count < 1 or count > len(groups):
        raise ValueError(f"Requested {count} distinct {split} parents, available {len(groups)}")
    parents = sorted(groups, key=lambda x: (stable_hash(["revision-parent", split, x]), x))[:count]
    return [min(groups[parent], key=lambda x: (stable_hash(["revision-variant", x]), x))
            for parent in parents]


def _validate_observation(model: ConvVAE3D, obs: np.ndarray | torch.Tensor) -> torch.Tensor:
    parameter = next(model.parameters())
    observation = torch.as_tensor(obs, device=parameter.device, dtype=parameter.dtype).detach()
    if observation.ndim == 4:
        observation = observation.unsqueeze(0)
    if observation.shape != (1, 2, *model.grid_size):
        raise ValueError("Exactly one [1,2,D,H,W] observation is required")
    labels, mask = observation[:, 0], observation[:, 1]
    if not bool(torch.isfinite(observation).all()):
        raise ValueError("Observation contains nonfinite values")
    if not bool(((mask == 0) | (mask == 1)).all()) or not bool(mask.sum() > 0):
        raise ValueError("A nonempty binary observation mask is required")
    if not bool((((labels == 0) | (labels == 1)) | (mask == 0)).all()):
        raise ValueError("Observed labels must be binary")
    if bool((labels[mask == 0] != 0).any()):
        raise ValueError("Unknown labels must be zero; reference target labels are forbidden")
    return observation


def optimize_observation(model: ConvVAE3D, obs: np.ndarray | torch.Tensor,
                         config: dict, *, mode: str = "joint", seed: int = 11,
                         initialization: int = 0, deadline_epoch: float | None = None) -> dict:
    """Adapt once and return the fixed decoder needed for all variance arms.

    Initialization 0 uses the encoder mean. Initializations 1 and 2 add seeded
    N(0,0.01 I) latent noise. This is a sensitivity analysis, not a draw from a
    conditional posterior. The objective never receives a reference volume.
    """
    if not isinstance(model, ConvVAE3D) or mode not in {"latent", "joint"}:
        raise ValueError("Only latent/joint ConvVAE3D diagnostics are supported")
    if initialization not in (0, 1, 2):
        raise ValueError("Initialization must be 0, 1, or 2")
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Adaptation must run outside inference_mode")
    if any(isinstance(m, (LoRALinear, LoRAConv3d)) for m in model.modules()):
        raise ValueError("A fresh checkpoint without installed adapters is required")
    observation = _validate_observation(model, obs)
    labels, mask = observation[:, 0], observation[:, 1]
    n_observed = int(mask.sum().item())
    joint = mode == "joint"
    # Joint case isolation is necessary. Latent-only mode temporarily freezes
    # the shared model and restores all flags, without duplicating its weights.
    working = copy.deepcopy(model) if joint else model
    module_flags = [(m, m.training) for m in model.modules()]
    parameter_flags = [(p, p.requires_grad) for p in model.parameters()]
    steps = int(config.get("diagnostic_steps", 500))
    patience = int(config.get("diagnostic_patience", 50))
    if steps < 0 or steps > 500 or patience < 0:
        raise ValueError("Diagnostic steps must be in [0,500] and patience nonnegative")
    prior_weight = float(config.get("test_time_prior_weight", 0.01))
    lora_weight = float(config.get("test_time_lora_weight", 0.01))
    min_delta = float(config.get("test_time_min_delta", 1e-6))
    rank = int(config.get("test_time_lora_rank", 8))
    lr = float(config.get("joint_test_time_lr" if joint else "latent_test_time_lr",
                          config.get("test_time_lr", 0.03)))
    lora_lr = float(config.get("joint_test_time_lora_lr", config.get("test_time_lora_lr", 0.001)))
    if min(prior_weight, lora_weight, min_delta) < 0 or min(lr, lora_lr) <= 0:
        raise ValueError("Invalid adaptation rates or regularization")
    started = time.perf_counter()
    try:
        working.eval()
        for parameter in working.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            mean, _ = working.encode(observation)
        generator = torch.Generator(device=mean.device).manual_seed(int(seed))
        initial = mean.detach().clone()
        if initialization:
            initial += 0.1 * torch.randn(initial.shape, device=mean.device,
                                         dtype=mean.dtype, generator=generator)
        latent = initial.clone().requires_grad_(True)
        adapters: list[torch.Tensor] = []
        if joint:
            # Constructors use the CPU RNG only, and must not alter caller RNG.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(seed))
                _, adapters = apply_lora_to_model(
                    working, rank=rank, alpha=rank, dropout=0.0,
                    target_modules=["fc_dec", "dec_res", "dec_up", "dec_final"],
                )
            if not adapters:
                raise RuntimeError("No trainable decoder LoRA parameters were installed")
        original_adapters = [p.detach().clone() for p in adapters]

        def objective(logits: torch.Tensor) -> tuple[torch.Tensor, dict]:
            fit = (F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * mask).sum() / n_observed
            prior = 0.5 * latent.square().mean()
            adapter_penalty = (sum((p - p0).square().mean()
                                   for p, p0 in zip(adapters, original_adapters))
                               if adapters else fit.new_zeros(()))
            total = fit + prior_weight * prior + lora_weight * adapter_penalty
            values = {
                "observed_bce": float(fit.detach()),
                "latent_norm_squared_sum": float(latent.detach().square().sum()),
                "normalized_latent_prior": float(prior.detach()),
                "weighted_latent_prior": float((prior_weight * prior).detach()),
                "lora_parameter_penalty": float(adapter_penalty.detach()),
                "weighted_lora_penalty": float((lora_weight * adapter_penalty).detach()),
                "objective": float(total.detach()),
            }
            if not all(math.isfinite(value) for value in values.values()):
                raise FloatingPointError("Nonfinite observation-only objective")
            return total, values

        groups = [{"params": [latent], "lr": lr}]
        if adapters:
            groups.append({"params": adapters, "lr": lora_lr})
        optimizer = torch.optim.Adam(groups)
        trace: list[dict] = []
        best_value, best_step = math.inf, 0
        best_latent = latent.detach().clone()
        best_adapters = [p.detach().clone() for p in adapters]
        gradient_checks = {"latent_first_norm": None, "lora_first_norm": None}
        stop_reason = "zero_steps" if steps == 0 else "maximum_steps"
        steps_run = 0
        for step in range(steps + 1):
            _check_deadline(deadline_epoch)
            optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits = working.decode(latent)
                loss, values = objective(logits)
                row = {"step": step, **values,
                       **observation_consistency(torch.sigmoid(logits.detach())[0].cpu().numpy(),
                                                 observation[0].cpu().numpy())}
                trace.append(row)
                if values["objective"] < best_value - min_delta:
                    best_value, best_step = values["objective"], step
                    best_latent = latent.detach().clone()
                    best_adapters = [p.detach().clone() for p in adapters]
                if step == steps:
                    break
                if patience and step - best_step >= patience:
                    stop_reason = "objective_patience"
                    break
                loss.backward()
                if latent.grad is None or not bool(torch.isfinite(latent.grad).all()):
                    raise FloatingPointError("Missing/nonfinite latent gradient")
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in adapters):
                    raise FloatingPointError("Missing/nonfinite LoRA gradient")
                if step == 0:
                    gradient_checks = {
                        "latent_first_norm": float(latent.grad.double().norm()),
                        "lora_first_norm": math.sqrt(sum(float(p.grad.double().square().sum()) for p in adapters)),
                    }
                    if joint and gradient_checks["lora_first_norm"] <= 0:
                        raise RuntimeError("LoRA received no nonzero gradient")
                torch.nn.utils.clip_grad_norm_([latent, *adapters], 5.0)
                optimizer.step()
                steps_run = step + 1
        with torch.no_grad():
            latent.copy_(best_latent)
            for parameter, saved in zip(adapters, best_adapters):
                parameter.copy_(saved)
            probability = torch.sigmoid(working.decode(latent))[0].cpu().numpy().astype(np.float32)
            # Unadapted checkpoint at the exact same optimized z isolates the
            # decoder-adapter effect from the effect of moving the latent code.
            model.eval()
            base_same_z = torch.sigmoid(model.decode(latent))[0].cpu().numpy().astype(np.float32)
        adapter_state = {name: tensor.detach().cpu().clone()
                         for name, tensor in working.state_dict().items()
                         if name.endswith("lora_A") or name.endswith("lora_B")}
        return {"decoder": working, "latent": latent.detach(),
                "optimized_probability": probability, "base_same_z_probability": base_same_z,
                "initial_latent": initial.detach().cpu().numpy(),
                "optimized_latent": latent.detach().cpu().numpy(),
                "lora_state": adapter_state, "trace": trace,
                "metadata": {"mode": mode, "seed": seed, "initialization": initialization,
                             "initialization_rule": "encoder_mean" if initialization == 0 else "encoder_mean_plus_N(0,0.01I)",
                             "steps_requested": steps, "steps_run": steps_run, "best_step": best_step,
                             "stop_reason": stop_reason, "best_objective": best_value,
                             "observed_count": n_observed, "latent_dim": int(latent.shape[-1]),
                             "prior_weight": prior_weight, "lora_weight": lora_weight,
                             "latent_learning_rate": lr, "lora_learning_rate": lora_lr if joint else None,
                             "normalization": "observed BCE/N_obs + lambda_z*0.5*mean(z^2) + lambda_A*sum_layer(mean(delta_A^2))",
                             "gradient_checks": gradient_checks,
                             "projection_applied": False,
                             "seconds": time.perf_counter() - started}}
    finally:
        for parameter, flag in parameter_flags:
            parameter.requires_grad_(flag)
        for module, flag in module_flags:
            module.training = flag


def _diversity(probabilities: np.ndarray, blind_mask: np.ndarray) -> dict:
    if len(probabilities) == 0:
        return {"pairwise_blind_dice_mean": None, "predicted_volume_mean": None,
                "predicted_volume_sd": None}
    binary = probabilities >= 0.5
    volumes = binary.reshape(len(binary), -1).sum(axis=1)
    pairwise = None
    if len(binary) >= 2:
        vectors = binary[:, blind_mask].astype(np.float32)
        sizes = vectors.sum(axis=1)
        denominators = sizes[:, None] + sizes[None, :]
        dice = np.divide(2 * (vectors @ vectors.T), denominators,
                         out=np.ones_like(denominators), where=denominators != 0)
        pairwise = float(dice[np.triu_indices(len(binary), 1)].mean())
    return {"pairwise_blind_dice_mean": pairwise,
            "predicted_volume_mean": float(volumes.mean()),
            "predicted_volume_sd": float(volumes.std(ddof=1)) if len(volumes) >= 2 else None}


def sample_fixed_decoder(adapted: dict, observation: np.ndarray, *, variance: float,
                         candidate_count: int = 100, seed: int = 11,
                         thresholds: tuple[float, ...] = ACCEPTANCE_THRESHOLDS) -> dict:
    """Reuse one optimized decoder. All proposals are raw and unprojected."""
    if candidate_count < 1 or variance < 0 or not math.isfinite(variance):
        raise ValueError("Positive candidate count and finite nonnegative variance required")
    decoder, center = adapted["decoder"], adapted["latent"]
    generator = torch.Generator(device=center.device).manual_seed(int(seed))
    epsilon = torch.randn((candidate_count, center.shape[-1]), device=center.device,
                          dtype=center.dtype, generator=generator)
    latents = center + math.sqrt(variance) * epsilon
    predictions = np.empty((candidate_count, *decoder.grid_size), dtype=np.float32)
    rows = []
    started = time.perf_counter()
    # Restore flags even when latent-only shares the base checkpoint decoder.
    flags = [(module, module.training) for module in decoder.modules()]
    decoder.eval()
    try:
        with torch.no_grad():
            for index in range(candidate_count):
                predictions[index] = torch.sigmoid(decoder.decode(latents[index:index + 1]))[0].cpu().numpy()
                norm_squared = float(latents[index].square().sum())
                row = {"candidate_index": index,
                       **observation_consistency(predictions[index], observation),
                       "latent_norm_squared_sum": norm_squared,
                       "normalized_latent_prior": 0.5 * norm_squared / latents.shape[-1],
                       "manuscript_prior_lower_bound": 0.01 * norm_squared}
                row["accepted_by_threshold"] = {
                    threshold_key(tau): bool(row["observed_mse"] <= tau and row["observed_violation_count"] == 0)
                    for tau in thresholds}
                rows.append(row)
    finally:
        for module, flag in flags:
            module.training = flag
    summaries = []
    for tau in thresholds:
        accepted = [index for index, row in enumerate(rows)
                    if row["accepted_by_threshold"][threshold_key(tau)]]
        n = len(accepted)
        selected = predictions[accepted]
        summaries.append({"threshold": tau, "accepted_indices": accepted,
                          "accepted_count": n, "candidate_count": candidate_count,
                          "acceptance_rate": n / candidate_count,
                          "mean_probability": selected.mean(axis=0, dtype=np.float64).astype(np.float32) if n else None,
                          "sample_variance": selected.var(axis=0, ddof=1, dtype=np.float64).astype(np.float32) if n >= 2 else None,
                          "variance_defined": n >= 2,
                          "status": "zero_accepted" if n == 0 else "one_accepted_variance_undefined" if n == 1 else "local_variability_only",
                          **_diversity(selected, observation[1] == 0)})
    return {"probabilities": predictions, "latents": latents.detach().cpu().numpy(),
            "candidates": rows, "summaries": summaries,
            "variance": variance, "standard_deviation": math.sqrt(variance),
            "sampling_seconds": time.perf_counter() - started,
            "projection_applied": False, "decoder_reoptimized": False}


def lora_distance_effect(adapted: np.ndarray, base_same_z: np.ndarray,
                         target: np.ndarray, observed: np.ndarray) -> list[dict]:
    """Measure spatial drift and change in error; drift itself is not failure."""
    distance = distance_transform_edt(~observed.astype(bool))
    bins = (("observed", distance == 0), ("near_0_to_2", (distance > 0) & (distance <= 2)),
            ("middle_2_to_5", (distance > 2) & (distance <= 5)), ("far_over_5", distance > 5))
    rows = []
    for name, mask in bins:
        if not mask.any():
            rows.append({"distance_bin": name, "voxel_count": 0,
                         "mean_absolute_probability_drift": None, "brier_change": None})
            continue
        rows.append({"distance_bin": name, "voxel_count": int(mask.sum()),
                     "mean_absolute_probability_drift": float(np.abs(adapted[mask] - base_same_z[mask]).mean()),
                     "brier_change": float(((adapted[mask] - target[mask]) ** 2 - (base_same_z[mask] - target[mask]) ** 2).mean())})
    return rows


def _existing_artifacts(folder: Path, identity_sha: str) -> dict | None:
    path = folder / "complete.json"
    if not path.is_file():
        return None
    record = _read(path)
    if record.get("identity_sha256") != identity_sha:
        raise RuntimeError(f"Diagnostic case identity changed: {folder}")
    for name, digest in record["artifacts"].items():
        candidate = (folder / name).resolve()
        if not candidate.is_relative_to(folder.resolve()) or not candidate.is_file() or sha256_file(candidate) != digest:
            raise RuntimeError(f"Missing/corrupt diagnostic artifact: {candidate}")
    return record


def uncertainty_run(manifest: str | Path, config_path: str | Path, checkpoint: str | Path,
                    out: str | Path, *, split: str = "val", seed: int = 11,
                    smoke: bool = False, freeze_path: str | Path | None = None,
                    modes: tuple[str, ...] = ("joint",),
                    deadline_epoch: float | None = None) -> Path:
    """Run the registered synthetic diagnostic; test requires completed validation."""
    if split not in {"val", "test"} or not modes or any(m not in {"latent", "joint"} for m in modes):
        raise ValueError("Expected val/test split and latent/joint modes")
    manifest, checkpoint, out = Path(manifest).resolve(), Path(checkpoint).resolve(), Path(out).resolve()
    configure_evaluation(out.parent if out.name == "uncertainty" else out,
                         phase="uncertainty_" + split)
    config = load_config(config_path)
    protocol = verify_protocol(manifest, verify_files=True)
    if not smoke and protocol.get("protocol_version") != "procedural-lens-v1":
        raise RuntimeError("Formal revision diagnostics require the known-parent procedural protocol")
    count = 1 if smoke else 10 if split == "val" else 30
    settings = {"candidate_count": 4 if smoke else 100, "variances": list(VARIANCES),
                "thresholds": list(ACCEPTANCE_THRESHOLDS), "primary_threshold": PRIMARY_ACCEPTANCE_THRESHOLD,
                "initializations": [0, 1, 2], "initialization_perturbation_variance": 0.01,
                "modes": list(modes), "diagnostic_steps": 2 if smoke else 500,
                "diagnostic_patience": 50, "projection_applied": False,
                "mask_seed": int(config.get("mask_seed", 42)), "seed": seed,
                "procedure": "observation_screened_local_variability_not_conditional_posterior"}
    selection = {role: select_parent_cases(protocol, role, 1 if smoke else 10 if role == "val" else 30)
                 for role in ("val", "test")}
    identity = {"format": "esi-local-variability-v1", "smoke": smoke,
                "manifest_sha256": sha256_file(manifest), "checkpoint_sha256": sha256_file(checkpoint),
                "config_sha256": stable_hash(config), "source_files": _source_identity(),
                "training_source_sha256": source_hash(), "settings": settings, "selection": selection,
                "evaluation_numerical_policy": numerical_metadata()}
    identity_sha = stable_hash(identity)
    root_identity = out / "identity.json"
    if root_identity.is_file() and _read(root_identity) != identity:
        raise RuntimeError("Diagnostic input/code/settings changed; choose a new output directory")
    save_json(root_identity, identity)
    save_json(out / "selection.json", selection)
    frozen = Path(freeze_path).resolve() if freeze_path else out / "val" / "settings_frozen.json"
    if split == "test":
        if (not frozen.is_file() or _read(frozen).get("identity_sha256") != identity_sha
                or _read(frozen).get("status") != "validation_frozen"):
            raise RuntimeError("Matching validation diagnostic must finish before test")
        for sample_id in selection["val"]:
            for mode in modes:
                for init in settings["initializations"]:
                    case_path = out / "val" / "cases" / sample_id / mode / f"init{init}"
                    if _existing_artifacts(case_path, identity_sha) is None:
                        raise RuntimeError("Validation diagnostic is incomplete")
    device = select_device("cpu" if smoke else str(config.get("device", "cuda")))
    model, _ = _load_checked_model("geovae", str(checkpoint), sha256_file(manifest), config, seed, device)
    dataset = ProtocolDataset(str(manifest), split, num_holes=int(config["num_holes"]),
                              samples_per_hole=int(config["samples_per_hole"]),
                              mask_seed=settings["mask_seed"], heldout_holes=4)
    adaptation_config = dict(config, diagnostic_steps=settings["diagnostic_steps"], diagnostic_patience=50)
    rows, failures = [], []
    started = time.perf_counter()
    ids = selection[split]
    for case_index, sample_id in enumerate(ids, 1):
        case = dataset.get_case(sample_id)
        parent = dataset.metadata[sample_id]["parent_id"]
        for mode in modes:
            for init in settings["initializations"]:
                _check_deadline(deadline_epoch)
                folder = out / split / "cases" / sample_id / mode / f"init{init}"
                try:
                    cached = _existing_artifacts(folder, identity_sha)
                    if cached is not None:
                        rows.extend(cached["summary_rows"])
                        continue
                    folder.mkdir(parents=True, exist_ok=True)
                    adapted = optimize_observation(model, case["obs"], adaptation_config, mode=mode,
                                                   seed=_seed(seed, sample_id, mode, init), initialization=init,
                                                   deadline_epoch=deadline_epoch)
                    _save_npz(folder / "adaptation.npz", initial_latent=adapted["initial_latent"],
                              optimized_latent=adapted["optimized_latent"],
                              optimized_probability=adapted["optimized_probability"],
                              base_same_z_probability=adapted["base_same_z_probability"],
                              target=case["target"], observation=case["obs"], hidden_mask=case["hidden_mask"])
                    torch.save(adapted["lora_state"], folder / "lora_state.pt")
                    _csv(folder / "optimization_trace.csv", adapted["trace"])
                    drift = lora_distance_effect(adapted["optimized_probability"], adapted["base_same_z_probability"],
                                                 case["target"], case["observed_mask"])
                    _csv(folder / "lora_distance_effect.csv", drift)
                    save_json(folder / "adaptation.json", adapted["metadata"])
                    local_rows = []
                    for variance in VARIANCES:
                        _check_deadline(deadline_epoch)
                        result = sample_fixed_decoder(adapted, case["obs"], variance=variance,
                                                      candidate_count=settings["candidate_count"],
                                                      seed=_seed(seed, sample_id, mode, init, "proposals"))
                        token = threshold_key(variance)
                        _save_npz(folder / f"candidates_var{token}.npz", probabilities=result["probabilities"],
                                  latents=result["latents"])
                        _csv(folder / f"candidates_var{token}.csv", result["candidates"])
                        for summary in result["summaries"]:
                            mean = summary.pop("mean_probability")
                            sample_variance = summary.pop("sample_variance")
                            tau_token = threshold_key(summary["threshold"])
                            if mean is not None:
                                _save_npy_atomic(folder / f"mean_var{token}_tau{tau_token}.npy", mean)
                            if sample_variance is not None:
                                _save_npy_atomic(folder / f"variance_var{token}_tau{tau_token}.npy", sample_variance)
                            summary_metrics = (volume_metrics(mean, case["target"], case["observed_mask"],
                                                              hidden_mask=case["hidden_mask"]) if mean is not None else {})
                            labels = case["obs"][0][case["obs"][1] >= 0.5]
                            positives = int(np.count_nonzero(labels >= 0.5))
                            negatives = int(len(labels) - positives)
                            composition = ("all_barren" if positives == 0 else
                                           "all_occupied" if negatives == 0 else "mixed")
                            row = {"sample_id": sample_id, "parent_id": parent, "split": split,
                                   "observed_positive_count": positives,
                                   "observed_negative_count": negatives,
                                   "observed_label_composition": composition,
                                   "mode": mode, "initialization": init, "variance": variance,
                                   "standard_deviation": math.sqrt(variance), **summary,
                                   "ensemble_blind_iou": summary_metrics.get("blind_iou"),
                                   "ensemble_brier_blind": summary_metrics.get("brier_blind"),
                                   "ensemble_hidden_error": summary_metrics.get("hidden_error"),
                                   "optimization_seconds": adapted["metadata"]["seconds"],
                                   "sampling_seconds": result["sampling_seconds"]}
                            local_rows.append(row)
                    artifacts = {p.name: sha256_file(p) for p in folder.iterdir() if p.is_file() and p.name != "complete.json"}
                    save_json(folder / "complete.json", {"status": "complete", "identity_sha256": identity_sha,
                                                        "artifacts": artifacts, "summary_rows": local_rows})
                    rows.extend(local_rows)
                    del adapted
                except TimeoutError:
                    save_json(out / split / "status.json", {"status": "budget_exhausted", "identity_sha256": identity_sha})
                    raise
                except Exception as error:
                    failures.append({"sample_id": sample_id, "mode": mode, "initialization": init,
                                     "error_type": type(error).__name__, "message": str(error)})
                    folder.mkdir(parents=True, exist_ok=True)
                    save_json(folder / "failure.json", failures[-1])
        elapsed = time.perf_counter() - started
        print(f"[uncertainty/{split}] {case_index}/{len(ids)} cases; elapsed={elapsed/60:.1f}min; "
              f"ETA={elapsed/case_index*(len(ids)-case_index)/60:.1f}min", flush=True)
    _csv(out / split / "per_case_acceptance.csv", rows)
    save_json(out / split / "failures.json", failures)
    if failures:
        raise RuntimeError(f"Diagnostic incomplete: {len(failures)} failures")
    save_json(out / split / "status.json", {"status": "complete", "identity_sha256": identity_sha,
                                          "case_count": len(ids), "summary_rows": len(rows), "environment": environment_info()})
    if split == "val":
        save_json(frozen, {"status": "validation_frozen", "identity_sha256": identity_sha,
                           "settings": settings, "case_ids": ids,
                           "selection_policy": "predeclared variance/acceptance sensitivity; no winning test setting selected"})
    return out / split / "status.json"


def assert_private_real_path(path: str | Path) -> Path:
    """Require an absolute private destination outside public/submission trees."""
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError("Supply an absolute private_revision_runs directory outside the public repository")
    candidate = candidate.resolve()
    if candidate.is_relative_to(ROOT.resolve()) or "revision_article" in {part.lower() for part in candidate.parts}:
        raise ValueError("Use a private_revision_runs directory outside public code and submission folders")
    return candidate



def real_case_domain(path: str | Path, grid_size: tuple[int, int, int]) -> tuple[np.ndarray, dict]:
    """Explicit target-derived extent. Valid only for retrospective completion."""
    from r12_data import _mesh, voxelize_mesh
    mesh = _mesh(Path(path))
    center = np.asarray(mesh.bounds).mean(axis=0)
    side = float(np.ptp(mesh.bounds, axis=0).max() * 1.2)
    domain = {"raw_min_xyz": (center - side/2).tolist(), "raw_max_xyz": (center + side/2).tolist(),
              "raw_coordinate_unit": "unverified", "target_specific_bounds_used": True,
              "fit_source": "reference mesh known extent; retrospective only"}
    grid, conversion = voxelize_mesh(Path(path), domain, grid_size)
    return grid, {"domain": domain, "conversion": conversion,
                  "watertight": True, "repair": "none", "role": "retrospective_known_extent_only"}


PUBLIC_REAL_METRICS = ("blind_iou", "blind_dice", "overall_iou", "observed_error", "hidden_error",
                       "volume_relative_error", "brier_blind", "hd95", "chamfer", "seconds")


def aggregate_real_rows(rows: list[dict], total_cases: int, exclusion_count: int,
                        failure_count: int) -> list[dict]:
    """Do not publish IDs, paths, coordinates, shapes, bounds, or predictions."""
    aggregates = []
    for method in sorted({r["method"] for r in rows}):
        group = [r for r in rows if r["method"] == method]
        parents = sorted({r["private_case_id"] for r in group})
        summary = {"method": method, "total_reference_blocks": total_cases,
                   "evaluated_reference_blocks": len(parents), "excluded_reference_blocks": exclusion_count,
                   "runtime_failures": failure_count, "mask_repeats": len(group),
                   "distance_unit": "voxel", "independent_mine_count_claimed": False,
                   "evaluation_role": "retrospective_known_extent_mesh_completion"}
        for key in PUBLIC_REAL_METRICS:
            # First average repeated masks within each block. Blocks remain a
            # dependent cohort from one mining area; no iid-block CI is claimed.
            values = []
            for case_id in parents:
                case_values = [float(r[key]) for r in group if r["private_case_id"] == case_id
                               and r.get(key) is not None and math.isfinite(float(r[key]))]
                if case_values:
                    values.append(float(np.mean(case_values)))
            summary[key + "_n"] = len(values)
            summary[key + "_mean"] = float(np.mean(values)) if values else None
            summary[key + "_median"] = float(np.median(values)) if values else None
        aggregates.append(summary)
    return aggregates


def real_cases_run(manifest: str | Path, config_path: str | Path, checkpoints: dict[str, str | Path],
                   private_out: str | Path, public_out: str | Path, *,
                   data_root: str | Path | None = None, seed: int = 11,
                   mask_seeds: tuple[int, ...] = (42, 43, 44),
                   methods: tuple[str, ...] = ("idw", "rbf", "ik", "mps", "base_vae", "unet", "cgan",
                                                "geovae", "geovae_latent", "geovae_joint"),
                   threshold_file: str | Path | None = None,
                   deadline_epoch: float | None = None) -> Path:
    """Predict all real blocks privately and export aggregate-only observations."""
    if data_root is None:
        raise ValueError("Public real-case execution requires an explicit external data_root")
    if Path(data_root).resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Restricted real input data must remain outside the public repository")
    private_out, public_out = assert_private_real_path(private_out), Path(public_out).resolve()
    # Keep the public real-data directory restricted to its two aggregate files.
    configure_evaluation(public_out.parent, phase="real_cases")
    if public_out.is_relative_to(private_out) or private_out.is_relative_to(public_out):
        raise ValueError("Private real artifacts and public aggregates require separate directories")
    data_root = Path(data_root).resolve() if data_root else ROOT.parent / "data" / "mining_ply"
    if (data_root / "mining_ply").is_dir():
        data_root = data_root / "mining_ply"
    manifest = Path(manifest).resolve()
    protocol = verify_protocol(manifest, verify_files=True)
    config = load_config(config_path)
    device = select_device(str(config.get("device", "cuda")))
    files = sorted(data_root.glob("*.ply"), key=lambda p: p.name)
    if len(files) != 122:
        raise RuntimeError(f"Expected all 122 declared real blocks; found {len(files)}")
    hashes = {path.name: sha256_file(path) for path in files}
    identity = {"format": "esi-real-private-v1", "config": config,
                "manifest_sha256": sha256_file(manifest), "source_files": _source_identity(),
                "checkpoint_sha256": {kind: sha256_file(path) for kind, path in checkpoints.items()},
                "real_sources": hashes, "seed": seed, "mask_seeds": list(mask_seeds), "methods": list(methods),
                "threshold_file_sha256": sha256_file(threshold_file) if threshold_file else None,
                "evaluation_numerical_policy": numerical_metadata(),
                "author_facts": {"same_mining_area": True, "source": "reserve_model",
                                  "manual_nonoverlapping_segments": True, "all_ore_no_interburden": True,
                                  "iron_and_rare_earth": True, "host_rock_unknown": True,
                                  "coordinate_units_unverified": True, "no_real_shape_publication": True}}
    identity_sha = stable_hash(identity)
    identity_path = private_out / "identity.json"
    if identity_path.is_file() and _read(identity_path) != identity:
        raise RuntimeError("Private real-case protocol changed; choose a new output directory")
    save_json(identity_path, identity)
    models = {}
    required = {"geovae" if method.startswith("geovae") else method
                for method in methods if method in {"geovae", "geovae_latent", "geovae_joint", "base_vae", "unet", "cgan"}}
    for kind in required:
        if kind not in checkpoints:
            raise RuntimeError(f"Missing frozen {kind} checkpoint")
        models[kind], _ = _load_checked_model(kind, str(checkpoints[kind]), sha256_file(manifest), config, seed, device)
    thresholds = {}
    if threshold_file:
        frozen = _read(Path(threshold_file))
        if (frozen.get("manifest_sha256") != sha256_file(manifest)
                or frozen.get("config_sha256") != stable_hash(config)
                or frozen.get("status") != "validation_frozen"):
            raise RuntimeError("Matching formally frozen validation thresholds are required")
        thresholds = frozen.get("thresholds", {})
        if any(method not in thresholds for method in methods):
            raise RuntimeError("Threshold file does not cover all requested real-case methods")
    training_image = None
    if "mps" in methods:
        training_entries = {r["sample_id"]: r for r in protocol["samples"] if r["split"] == "train"}
        training_id = str(config.get("mps_training_image_id") or min(training_entries))
        if training_id not in training_entries:
            raise RuntimeError("MPS training image must be one frozen training reference")
        entry = training_entries[training_id]
        training_image = np.load(manifest.parent / entry["path"], allow_pickle=False).astype(np.float32)
    rows, exclusions, failures = [], [], []
    started = time.perf_counter()
    for index, path in enumerate(files, 1):
        _check_deadline(deadline_epoch)
        case_id = "private-" + stable_hash([path.name, hashes[path.name]])[:16]
        folder = private_out / "cases" / case_id
        cached = _existing_artifacts(folder, identity_sha)
        if cached is not None:
            rows.extend(cached["metric_rows"])
            continue
        try:
            grid, quality = real_case_domain(path, tuple(config["grid_size"]))
        except Exception as error:
            exclusions.append({"private_case_id": case_id, "source": path.name,
                               "error_type": type(error).__name__, "reason": str(error)})
            continue
        folder.mkdir(parents=True, exist_ok=True)
        _save_npy_atomic(folder / "target.npy", grid)
        save_json(folder / "private_metadata.json", {"source": path.name, "source_sha256": hashes[path.name], **quality})
        case_rows = []
        case_failed = False
        for mask_seed in mask_seeds:
            observed, hidden = make_drill_masks(tuple(config["grid_size"]), case_id, int(config["num_holes"]),
                                                int(config["samples_per_hole"]), mask_seed, 4)
            observation = np.stack((grid * observed, observed)).astype(np.float32)
            _save_npy_atomic(folder / f"observation_mask{mask_seed}.npy", observation)
            for method in methods:
                _check_deadline(deadline_epoch)
                try:
                    begin = time.perf_counter()
                    prediction, adaptation = _predict(method, observation, models, config, device,
                                                       training_image, _seed(seed, case_id, mask_seed, method))
                    elapsed = time.perf_counter() - begin
                    _save_npy_atomic(folder / f"prediction_{method}_mask{mask_seed}.npy", prediction)
                    save_json(folder / f"adaptation_{method}_mask{mask_seed}.json", adaptation)
                    threshold = thresholds.get(method, config.get("threshold", 0.5))
                    if isinstance(threshold, dict):
                        threshold = threshold.get("threshold", 0.5)
                    metrics = volume_metrics(prediction, grid, observed, threshold=float(threshold), hidden_mask=hidden)
                    row = {"private_case_id": case_id, "method": method, "mask_seed": mask_seed,
                           "seconds": elapsed, **metrics}
                    case_rows.append(row)
                except Exception as error:
                    case_failed = True
                    failures.append({"private_case_id": case_id, "method": method, "mask_seed": mask_seed,
                                     "error_type": type(error).__name__, "message": str(error)})
        rows.extend(case_rows)
        _csv(folder / "metrics.csv", case_rows)
        if not case_failed:
            artifacts = {p.name: sha256_file(p) for p in folder.iterdir() if p.is_file() and p.name != "complete.json"}
            save_json(folder / "complete.json", {"status": "complete", "identity_sha256": identity_sha,
                                                "metric_rows": case_rows, "artifacts": artifacts})
        elapsed = time.perf_counter() - started
        print(f"[real/private] {index}/{len(files)}; excluded={len(exclusions)}; failures={len(failures)}; "
              f"ETA={elapsed/index*(len(files)-index)/60:.1f}min", flush=True)
    save_json(private_out / "exclusions.json", exclusions)
    save_json(private_out / "failures.json", failures)
    _csv(private_out / "per_case_metrics.csv", rows)
    public = aggregate_real_rows(rows, len(files), len(exclusions), len(failures))
    _csv(public_out / "real_aggregate_metrics.csv", public)
    # Public schema is deliberately fixed, never copied from private metadata.
    public_summary = {"format": "esi-real-aggregate-only-v1",
                      "status": "incomplete_runtime_failures" if failures else "complete_with_exclusions" if exclusions else "complete",
                      "total_reference_blocks": len(files), "excluded_reference_blocks": len(exclusions),
                      "runtime_failures": len(failures), "evaluated_methods": list(methods),
                      "scope": "one mining area, manually nonoverlapping ore segments from a reserve model",
                      "reference_role": "interpreted reserve-model geometry, not raw assay or new drilling truth",
                      "domain_policy": "target-derived known extent, retrospective mesh completion only",
                      "unit": "voxel", "coordinate_units_confirmed": False,
                      "real_shapes_or_coordinates_exported": False,
                      "statistical_policy": "descriptive aggregates; no independent-deposit or iid-block inference",
                      "private_audit_identity_sha256": identity_sha, "metrics": public}
    save_json(public_out / "real_summary.json", public_summary)
    if failures:
        raise RuntimeError(f"Real-case evaluation has {len(failures)} runtime failures; see private audit")
    if not rows:
        raise RuntimeError("No real blocks were evaluable; do not claim real-case performance")
    return public_out / "real_summary.json"


# Kept as an explicit integration alias for the revision workflow.
realcases_run = real_cases_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    uncertainty = sub.add_parser("uncertainty")
    uncertainty.add_argument("--manifest", required=True)
    uncertainty.add_argument("--config", required=True)
    uncertainty.add_argument("--checkpoint", required=True)
    uncertainty.add_argument("--out", required=True)
    uncertainty.add_argument("--split", choices=("val", "test"), default="val")
    uncertainty.add_argument("--seed", type=int, default=11)
    uncertainty.add_argument("--smoke", action="store_true")
    uncertainty.add_argument("--freeze-path")
    uncertainty.add_argument("--mode", choices=("joint", "latent", "both"), default="joint")
    real = sub.add_parser("real-cases")
    real.add_argument("--manifest", required=True)
    real.add_argument("--config", required=True)
    real.add_argument("--checkpoint", action="append", required=True, help="KIND=PATH")
    real.add_argument("--private-out", required=True)
    real.add_argument("--public-out", required=True)
    real.add_argument("--data-root")
    real.add_argument("--seed", type=int, default=11)
    real.add_argument("--methods", nargs="+", default=["idw", "rbf", "ik", "mps", "base_vae", "unet", "cgan",
                                                     "geovae", "geovae_latent", "geovae_joint"])
    real.add_argument("--threshold-file")
    for command in (uncertainty, real):
        command.add_argument("--max-seconds", type=float,
                             help="Stop at case/optimization boundaries when this runtime budget expires")
    args = parser.parse_args(argv)
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    deadline = time.time() + args.max_seconds if args.max_seconds is not None else None
    if args.stage == "uncertainty":
        modes = ("joint", "latent") if args.mode == "both" else (args.mode,)
        print(uncertainty_run(args.manifest, args.config, args.checkpoint, args.out,
                              split=args.split, seed=args.seed, smoke=args.smoke,
                              freeze_path=args.freeze_path, modes=modes, deadline_epoch=deadline))
    else:
        checkpoints = dict(value.split("=", 1) for value in args.checkpoint)
        print(real_cases_run(args.manifest, args.config, checkpoints, args.private_out,
                             args.public_out, data_root=args.data_root, seed=args.seed,
                             methods=tuple(args.methods), threshold_file=args.threshold_file,
                             deadline_epoch=deadline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
