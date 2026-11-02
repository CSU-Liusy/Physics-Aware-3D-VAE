"""Conditioned baseline predictors using exactly the supplied observation mask.

Indicator ordinary kriging uses a binary indicator variable. MPS uses the
official MPSlib GENESIM direct-sampling setting (n_max_cpdf_count=1).
Neither baseline is allowed to see the target volume.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree


def observed_points(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim != 4 or obs.shape[0] != 2:
        raise ValueError("Observation must have shape [2,D,H,W]")
    coordinates = np.argwhere(obs[1] >= 0.5).astype(np.float64)
    values = obs[0][obs[1] >= 0.5].astype(np.float64)
    if len(coordinates) == 0:
        raise ValueError("At least one observed voxel is required")
    if not np.isin(values, (0, 1)).all():
        raise ValueError("Indicator baselines require binary observed labels")
    return coordinates, values


def grid_coordinates(shape: tuple[int, int, int]) -> np.ndarray:
    return np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing="ij"),
                    axis=-1).reshape(-1, 3).astype(np.float64)


def predict_idw(obs: np.ndarray, power: float = 2.0, chunk: int = 4096) -> np.ndarray:
    points, values = observed_points(obs)
    shape = tuple(obs.shape[1:])
    grid = grid_coordinates(shape)
    answer = np.empty(len(grid), dtype=np.float32)
    for first in range(0, len(grid), chunk):
        query = grid[first:first + chunk]
        distance = np.linalg.norm(query[:, None] - points[None], axis=2)
        exact = distance <= 1e-10
        weights = 1 / np.maximum(distance, 1e-8) ** power
        estimate = (weights @ values) / weights.sum(axis=1)
        if exact.any():
            row = exact.any(axis=1)
            estimate[row] = values[exact[row].argmax(axis=1)]
        answer[first:first + chunk] = estimate
    return np.clip(answer.reshape(shape), 0, 1)


def predict_rbf(obs: np.ndarray, kernel: str = "linear",
                smoothing: float = 0.01) -> np.ndarray:
    points, values = observed_points(obs)
    shape = tuple(obs.shape[1:])
    if np.unique(values).size == 1:
        return np.full(shape, values[0], dtype=np.float32)
    interpolator = RBFInterpolator(points, values, kernel=kernel, smoothing=smoothing)
    result = interpolator(grid_coordinates(shape)).reshape(shape)
    return np.clip(result.astype(np.float32), 0, 1)


def predict_indicator_kriging(obs: np.ndarray, len_scale: float = 5.0,
                              nugget: float = 0.01) -> np.ndarray:
    try:
        import gstools as gs
    except ImportError as error:
        raise RuntimeError("Indicator kriging requires gstools; install the optional baseline dependencies") from error
    points, values = observed_points(obs)
    shape = tuple(obs.shape[1:])
    if np.unique(values).size == 1:
        return np.full(shape, values[0], dtype=np.float32)
    variogram = gs.Gaussian(dim=3, var=float(np.var(values)), len_scale=len_scale,
                            nugget=nugget)
    krige = gs.krige.Ordinary(variogram, cond_pos=points.T, cond_val=values,
                              exact=True)
    prediction, _ = krige(grid_coordinates(shape).T)
    return np.clip(np.asarray(prediction, dtype=np.float32).reshape(shape), 0, 1)


def predict_mps_direct_sampling(obs: np.ndarray, training_image: np.ndarray,
                                seed: int, realizations: int = 30,
                                n_cond: int = 25, max_iterations: int = 1000) -> np.ndarray:
    try:
        import mpslib as mps
    except ImportError as error:
        raise RuntimeError("MPS Direct Sampling requires scikit-mps and its executable") from error
    if realizations < 1:
        raise ValueError("realizations must be positive")
    shape = tuple(obs.shape[1:])
    ti = np.asarray(training_image, dtype=np.float32)
    if ti.shape != shape:
        raise ValueError(f"MPS training image shape {ti.shape} differs from test grid {shape}")
    points, values = observed_points(obs)
    hard_data = np.column_stack((points[:, 2], points[:, 1], points[:, 0], values))
    with tempfile.TemporaryDirectory(prefix="geovae_mps_") as temporary:
        simulation = mps.mpslib(
            method="mps_genesim",
            simulation_grid_size=np.array([shape[2], shape[1], shape[0]]),
            n_real=int(realizations),
            n_cond=int(n_cond),
            n_max_cpdf_count=1,
            n_max_ite=int(max_iterations),
            rseed=int(seed) + 1,
            debug_level=-1,
            verbose_level=-1,
        )
        simulation.ti = np.transpose(ti, (2, 1, 0))
        simulation.d_hard = hard_data
        simulation.par["out_folder"] = temporary
        simulation.parameter_filename = str(Path(temporary) / "mps.txt")
        success = simulation.run()
        if not success or len(simulation.sim) != realizations:
            raise RuntimeError(f"MPSlib failed: success={success}, generated={len(simulation.sim)}")
        samples = np.stack([np.transpose(np.asarray(x), (2, 1, 0))
                            for x in simulation.sim])
        if samples.shape[1:] != shape or not np.isfinite(samples).all():
            raise RuntimeError("MPSlib produced invalid sample shapes or NaN")
        return samples.mean(axis=0).astype(np.float32)