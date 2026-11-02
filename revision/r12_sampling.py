"""R1.2 observation-only optimization and local-candidate diagnostics.

This is a revised, explicitly defined procedure, not a reconstruction of the
submitted manuscript's historical execution and not a posterior sampler.
Only the sparse label and mask channels are accepted here.  Full reference
volumes must be used outside this module, after prediction, for evaluation.
"""
from __future__ import annotations

import copy
import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from model import ConvVAE3D, LoRAConv3d, LoRALinear, apply_lora_to_model


DEFAULT_THRESHOLDS = (1e-4, 1e-3, 1e-2)


def threshold_key(value: float) -> str:
    return format(float(value), ".8g")


def observation_consistency(
    probability: np.ndarray,
    observation: np.ndarray,
    probability_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Score only observed voxels, with N_obs as the denominator.

    ``observation`` has shape [2,D,H,W], with binary labels in channel 0 and
    binary sampling indicators in channel 1.  No projection is performed.
    """
    prediction = np.asarray(probability, dtype=np.float64)
    obs = np.asarray(observation)
    if obs.ndim != 4 or obs.shape[0] != 2 or prediction.shape != obs.shape[1:]:
        raise ValueError("Expected probability [D,H,W] and observation [2,D,H,W]")
    if not 0.0 < float(probability_threshold) < 1.0:
        raise ValueError("probability_threshold must lie strictly between 0 and 1")
    if not np.isfinite(prediction).all() or np.any(prediction < 0) or np.any(prediction > 1):
        raise ValueError("Candidate must be a finite raw probability volume in [0,1]")
    if not np.isfinite(obs).all() or not np.isin(obs[1], (0, 1)).all():
        raise ValueError("Observation must be finite and its mask binary")
    known = obs[1].astype(bool)
    count = int(known.sum())
    if count == 0:
        raise ValueError("Observation mask is empty")
    labels = obs[0][known]
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Observed labels must be binary")
    difference = prediction[known] - labels
    violations = int(np.count_nonzero((prediction[known] >= probability_threshold) != labels.astype(bool)))
    mse = float(np.mean(difference ** 2))
    return {
        "observed_count": count,
        "observed_mse": mse,
        "observed_rmse": math.sqrt(mse),
        "observed_max_abs_error": float(np.max(np.abs(difference))),
        "observed_violation_count": violations,
        "observed_violation_rate": float(violations / count),
    }


def _adapter_state(working: torch.nn.Module) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, module in working.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv3d)):
            result[f"{name}.lora_A"] = module.lora_A.detach().cpu().clone()
            result[f"{name}.lora_B"] = module.lora_B.detach().cpu().clone()
    return result


def _candidate_summary(
    predictions: np.ndarray, records: list[dict[str, Any]], threshold: float,
) -> dict[str, Any]:
    key = threshold_key(threshold)
    indices = [index for index, row in enumerate(records)
               if row["accepted_by_threshold"][key]]
    count = len(indices)
    if count:
        accepted = predictions[indices]
        mean = accepted.mean(axis=0, dtype=np.float64).astype(np.float32)
    else:
        mean = None
    # An undefined variance must remain missing, never become an all-zero map.
    variance = (predictions[indices].var(axis=0, ddof=1, dtype=np.float64).astype(np.float32)
                if count >= 2 else None)
    status = ("no_accepted_candidates" if count == 0 else
              "one_accepted_candidate_variance_undefined" if count == 1 else
              "sample_variance_defined_reliability_unverified")
    return {
        "threshold": float(threshold), "threshold_key": key,
        "candidate_count": len(records), "accepted_count": count,
        "acceptance_rate": float(count / len(records)),
        "accepted_indices": indices, "status": status,
        "ensemble_status": status,
        "variance_defined": count >= 2, "mean_probability": mean,
        "sample_variance": variance,
    }


def run_case(
    model: ConvVAE3D,
    obs: torch.Tensor | np.ndarray,
    config: dict[str, Any],
    *,
    mode: str = "joint",
    seed: int = 11,
    candidate_count: int = 100,
    variance: float = 0.01,
    thresholds: tuple[float, ...] | list[float] = DEFAULT_THRESHOLDS,
    primary_threshold: float = 0.001,
    probability_threshold: float = 0.5,
) -> dict[str, Any]:
    """Optimize one sparse case, then diagnose a fixed set of local candidates.

    Candidate k is z_star + sqrt(variance) * epsilon_k.  Its adapted decoder is
    fixed, and no candidate-specific reoptimization or observation projection
    is allowed.  Acceptance uses observed MSE and zero classification violations,
    independently of the regularized optimization objective.

    The source model's parameters, gradients, training flags and caller RNG are
    preserved.  Joint mode installs fresh decoder adapters on one model copy.
    Returned arrays include *all* raw candidates, including rejected candidates.
    """
    if not isinstance(model, ConvVAE3D):
        raise TypeError("R1.2 diagnostics currently support ConvVAE3D only")
    if mode not in {"latent", "joint"}:
        raise ValueError("mode must be latent or joint")
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Call run_case outside torch.inference_mode()")
    if isinstance(candidate_count, bool) or int(candidate_count) != candidate_count or candidate_count < 1:
        raise ValueError("candidate_count must be a positive integer")
    if not math.isfinite(float(variance)) or variance < 0:
        raise ValueError("variance must be finite and nonnegative")
    tau_values = tuple(float(value) for value in thresholds)
    if (not tau_values or len(set(tau_values)) != len(tau_values)
            or any(not math.isfinite(value) or value < 0 for value in tau_values)):
        raise ValueError("thresholds must contain distinct finite nonnegative values")
    if float(primary_threshold) not in tau_values:
        raise ValueError("primary_threshold must be included in thresholds")
    if not 0 < float(probability_threshold) < 1:
        raise ValueError("probability_threshold must lie strictly between 0 and 1")
    joint = mode == "joint"
    steps = int(config.get("joint_test_time_steps" if joint else "latent_test_time_steps",
                           config.get("test_time_steps", 50)))
    lr = float(config.get("joint_test_time_lr" if joint else "latent_test_time_lr",
                          config.get("test_time_lr", 0.03)))
    lora_lr = float(config.get("joint_test_time_lora_lr", config.get("test_time_lora_lr", 0.001)))
    prior_weight = float(config.get("test_time_prior_weight", 0.01))
    lora_weight = float(config.get("test_time_lora_weight", 0.01))
    lora_rank = int(config.get("test_time_lora_rank", 8))
    patience_value = config.get("test_time_patience", 25)
    patience = None if patience_value is None else int(patience_value)
    min_delta = float(config.get("test_time_min_delta", 1e-6))
    max_grad_norm = float(config.get("test_time_max_grad_norm", 5.0))
    manuscript_lambda = float(config.get("manuscript_lambda", 0.01))
    nonnegative = (prior_weight, lora_weight, min_delta, manuscript_lambda)
    positive = (lr, lora_lr, max_grad_norm)
    if (steps < 0 or any(not math.isfinite(value) or value < 0 for value in nonnegative)
            or any(not math.isfinite(value) or value <= 0 for value in positive)
            or (joint and lora_rank < 1) or (patience is not None and patience < 0)):
        raise ValueError("Invalid optimization budget, learning rate, regularization or LoRA rank")
    if any(isinstance(module, (LoRALinear, LoRAConv3d)) for module in model.modules()):
        raise ValueError("Use a base checkpoint without preinstalled LoRA adapters")

    reference = next(model.parameters())
    device, dtype = reference.device, reference.dtype
    observation = torch.as_tensor(obs, device=device, dtype=dtype).detach()
    if observation.ndim == 4:
        observation = observation.unsqueeze(0)
    if observation.ndim != 5 or observation.shape[:2] != (1, 2):
        raise ValueError("obs must describe exactly one case with shape [1,2,D,H,W]")
    if tuple(observation.shape[-3:]) != tuple(model.grid_size):
        raise ValueError("Observation shape does not match the checkpoint grid")
    if not bool(torch.isfinite(observation).all()):
        raise ValueError("Observation contains nonfinite values")
    labels, mask = observation[:, 0], observation[:, 1]
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("Observation mask must be binary")
    if not bool((((labels == 0) | (labels == 1)) | (mask == 0)).all()):
        raise ValueError("Observed labels must be binary")
    if bool((labels[mask == 0] != 0).any()):
        raise ValueError("Unobserved label channel must be zero; full target labels are forbidden")
    observed_count = int(mask.sum().item())
    if observed_count == 0:
        raise ValueError("Observation mask is empty")
    obs_np = observation[0].cpu().numpy()
    source_module_flags = [(module, module.training) for module in model.modules()]
    source_parameter_flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    started_case = time.perf_counter()

    def run(working: ConvVAE3D) -> dict[str, Any]:
        working.eval()
        for parameter in working.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            mu, _ = working.encode(observation)
        z_initial = mu.detach().clone()
        z = z_initial.clone().requires_grad_(True)
        adapter_parameters: list[torch.Tensor] = []
        if joint:
            # LoRA constructors initialize on CPU before moving to the model
            # device.  Fork only that CPU RNG; candidate draws use their own RNG.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(seed))
                _, adapter_parameters = apply_lora_to_model(
                    working, rank=lora_rank, alpha=lora_rank,
                    target_modules=["fc_dec", "dec_res", "dec_up", "dec_final"],
                    dropout=0.0,
                )
            if not adapter_parameters:
                raise RuntimeError("No decoder LoRA parameters were installed")
            working.eval()
        adapter_initial = [parameter.detach().clone() for parameter in adapter_parameters]

        def components(logits: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            if logits.shape != labels.shape:
                raise ValueError("Decoder output shape differs from sparse observation")
            per_voxel = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            bce = (per_voxel * mask).sum() / observed_count
            norm_sum = z.square().sum()
            normalized_prior = 0.5 * z.square().mean()
            adapter_penalty = (sum((parameter - initial).square().mean()
                                   for parameter, initial in zip(adapter_parameters, adapter_initial))
                               if adapter_parameters else bce.new_zeros(()))
            objective = bce + prior_weight * normalized_prior + lora_weight * adapter_penalty
            return objective, {
                "observed_bce": bce,
                "latent_norm_squared_sum": norm_sum,
                "latent_prior_normalized": normalized_prior,
                "weighted_latent_prior": prior_weight * normalized_prior,
                "lora_parameter_penalty": adapter_penalty,
                "weighted_lora_penalty": lora_weight * adapter_penalty,
                "optimization_objective": objective,
                # Only the prior contribution is known from the original
                # formula.  Alpha and historical execution are not reconstructed.
                "manuscript_prior_lower_bound": manuscript_lambda * norm_sum,
            }

        def record_state(logits: torch.Tensor, state: dict[str, torch.Tensor], step: int) -> dict[str, Any]:
            scalars = {name: float(value.detach().item()) for name, value in state.items()}
            if not all(math.isfinite(value) for value in scalars.values()):
                raise FloatingPointError("Nonfinite optimization objective or component")
            probability = torch.sigmoid(logits.detach())[0].cpu().numpy().astype(np.float32)
            return {"step": int(step), **scalars,
                    **observation_consistency(probability, obs_np, probability_threshold)}

        groups: list[dict[str, Any]] = [{"params": [z], "lr": lr}]
        if adapter_parameters:
            groups.append({"params": adapter_parameters, "lr": lora_lr})
        optimizer = torch.optim.Adam(groups)
        trace: list[dict[str, Any]] = []
        best_value = float("inf")
        best_step = 0
        best_z = z.detach().clone()
        best_adapters = [parameter.detach().clone() for parameter in adapter_parameters]
        steps_run = 0
        first_z_grad_norm: float | None = None
        first_lora_grad_norm: float | None = None
        optimization_started = time.perf_counter()
        stop_reason = "zero_step_budget" if steps == 0 else "maximum_steps"
        with torch.enable_grad():
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                logits = working.decode(z)
                objective, state = components(logits)
                row = record_state(logits, state, step)
                trace.append(row)
                value = row["optimization_objective"]
                if value < best_value - min_delta:
                    best_value, best_step = value, step
                    best_z = z.detach().clone()
                    best_adapters = [parameter.detach().clone() for parameter in adapter_parameters]
                objective.backward()
                if z.grad is None or not bool(torch.isfinite(z.grad).all()):
                    raise FloatingPointError("Latent gradient is missing or nonfinite")
                if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
                       for parameter in adapter_parameters):
                    raise FloatingPointError("LoRA gradient is missing or nonfinite")
                if step == 0:
                    first_z_grad_norm = float(z.grad.detach().norm().item())
                    first_lora_grad_norm = math.sqrt(sum(
                        float(parameter.grad.detach().square().sum().item())
                        for parameter in adapter_parameters)) if adapter_parameters else 0.0
                torch.nn.utils.clip_grad_norm_([z, *adapter_parameters], max_grad_norm)
                optimizer.step()
                steps_run = step + 1
                if patience and steps_run - best_step >= patience:
                    stop_reason = "objective_patience"
                    break
            with torch.no_grad():
                final_logits = working.decode(z)
                _, final_state = components(final_logits)
                final_row = record_state(final_logits, final_state, steps_run)
                trace.append(final_row)
                if final_row["optimization_objective"] < best_value - min_delta or steps == 0:
                    best_value = final_row["optimization_objective"]
                    best_step = steps_run
                    best_z = z.detach().clone()
                    best_adapters = [parameter.detach().clone() for parameter in adapter_parameters]
                z.copy_(best_z)
                for parameter, saved in zip(adapter_parameters, best_adapters):
                    parameter.copy_(saved)
                best_logits = working.decode(z)
                _, best_state = components(best_logits)
                best_row = record_state(best_logits, best_state, best_step)
                optimized_probability = torch.sigmoid(best_logits)[0].cpu().numpy().astype(np.float32)
        optimization_seconds = time.perf_counter() - optimization_started
        optimized_latent = z.detach().cpu().numpy().copy()
        optimized_lora = _adapter_state(working)

        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        epsilon = torch.randn((int(candidate_count), z.shape[-1]), generator=generator,
                              device=device, dtype=dtype)
        candidate_latents = z.detach() + math.sqrt(float(variance)) * epsilon
        candidate_predictions = np.empty((int(candidate_count), *model.grid_size), dtype=np.float32)
        candidate_rows: list[dict[str, Any]] = []
        sampling_started = time.perf_counter()
        with torch.no_grad():
            # Keep a dedicated proposal tensor: components() uses z, so copy the
            # current proposal into this isolated optimized variable without any
            # gradient steps.  Restore z_star once all proposals are diagnosed.
            for index in range(int(candidate_count)):
                candidate_started = time.perf_counter()
                z.copy_(candidate_latents[index:index + 1])
                logits = working.decode(z)
                _, state = components(logits)
                row = record_state(logits, state, step=best_step)
                row.pop("step")
                prediction = torch.sigmoid(logits)[0].cpu().numpy().astype(np.float32)
                candidate_predictions[index] = prediction
                accepts = {
                    threshold_key(tau): bool(row["observed_mse"] <= tau
                                             and row["observed_violation_count"] == 0)
                    for tau in tau_values
                }
                candidate_rows.append({
                    "candidate_index": index,
                    **row,
                    "accepted": accepts[threshold_key(primary_threshold)],
                    "accepted_by_threshold": accepts,
                    "manuscript_prior_alone_exceeds_threshold": bool(row["manuscript_prior_lower_bound"] >= 1e-3),
                    "seconds": float(time.perf_counter() - candidate_started),
                })
            z.copy_(best_z)
        sampling_seconds = time.perf_counter() - sampling_started
        summaries = [_candidate_summary(candidate_predictions, candidate_rows, tau) for tau in tau_values]
        return {
            "optimized_probability": optimized_probability,
            "optimized_latent": optimized_latent,
            "initial_latent": z_initial.detach().cpu().numpy().copy(),
            "candidate_latents": candidate_latents.detach().cpu().numpy().copy(),
            "candidate_probabilities": candidate_predictions,
            "lora_state": optimized_lora,
            "optimization_trace": trace,
            "candidates": candidate_rows,
            "summaries": summaries,
            "metadata": {
                "procedure": "revised_observation_screened_local_variability_not_posterior_sampling",
                "mode": mode, "seed": int(seed),
                "observed_count": observed_count, "latent_dim": int(z.shape[-1]),
                "steps_requested": steps, "steps_run": steps_run,
                "best_step": best_step, "stop_reason": stop_reason,
                "patience": patience, "min_delta": min_delta,
                "latent_lr": lr, "lora_lr": lora_lr if joint else None,
                "prior_weight": prior_weight, "lora_weight": lora_weight,
                "lora_rank": lora_rank if joint else 0,
                "lora_trainable_parameters": sum(parameter.numel() for parameter in adapter_parameters),
                "first_z_gradient_norm": first_z_grad_norm,
                "first_lora_gradient_norm": first_lora_grad_norm,
                "initial_components": trace[0], "final_components": trace[-1],
                "best_components": best_row,
                "candidate_count": int(candidate_count),
                "perturbation_variance": float(variance),
                "perturbation_standard_deviation": math.sqrt(float(variance)),
                "primary_threshold": float(primary_threshold),
                "thresholds": list(tau_values),
                "probability_threshold": float(probability_threshold),
                "acceptance_rule": "observed_mean_squared_error <= tau AND observed_binary_violations == 0",
                "acceptance_denominator": "number_of_observed_voxels",
                "projection_before_acceptance": False,
                "decoder_fixed_during_proposals": True,
                "candidate_reoptimization": False,
                "manuscript_lambda": manuscript_lambda,
                "manuscript_threshold": 1e-3,
                "manuscript_diagnostic": "lambda * sum(z^2) only; lower-bound contribution, not a reconstructed historical full loss",
                "optimization_seconds": float(optimization_seconds),
                "sampling_seconds": float(sampling_seconds),
                "case_seconds": float(time.perf_counter() - started_case),
                "zero_accepted_policy": "record failure; optimized reconstruction remains separately available",
                "one_accepted_policy": "mean available; sample variance undefined and missing",
            },
        }

    if joint:
        return run(copy.deepcopy(model))
    try:
        return run(model)
    finally:
        for parameter, flag in source_parameter_flags:
            parameter.requires_grad_(flag)
        for module, flag in source_module_flags:
            module.training = flag
