"""Per-volume metrics, with explicit empty-surface and distance conventions.

Surfaces are foreground voxel centres having at least one background face
neighbour (including the exterior of the grid). HD is the maximum of the two
directed nearest-neighbour distances; HD95 is the maximum of their separate
95th percentiles. Chamfer is the arithmetic mean of the two directed mean
Euclidean distances, without squaring. Distances use voxel units unless an
independently verified coordinate spacing is explicitly supplied.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


def _surface(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume, dtype=bool)
    return np.argwhere(volume & ~binary_erosion(volume))


def surface_distances(predicted: np.ndarray, truth: np.ndarray,
                      spacing: tuple[float, float, float] = (1., 1., 1.)) -> dict:
    scale = np.asarray(spacing, dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Spacing must contain three finite positive axis spacings")
    a, b = _surface(predicted), _surface(truth)
    if len(a) == 0 or len(b) == 0:
        reason = ("both_empty" if len(a) == 0 and len(b) == 0 else
                  "prediction_empty" if len(a) == 0 else "truth_empty")
        return {"hd": None, "hd95": None, "chamfer": None,
                "surface_distance_valid": False, "surface_distance_status": reason,
                "prediction_surface_n": int(len(a)), "truth_surface_n": int(len(b))}
    a, b = a * scale, b * scale
    d_ab = cKDTree(b).query(a, k=1)[0]
    d_ba = cKDTree(a).query(b, k=1)[0]
    return {
        "hd": float(max(d_ab.max(), d_ba.max())),
        "hd95": float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95))),
        "chamfer": float(0.5 * (d_ab.mean() + d_ba.mean())),
        "surface_distance_valid": True, "surface_distance_status": "valid",
        "prediction_surface_n": int(len(a)), "truth_surface_n": int(len(b)),
    }


def volume_metrics(probability: np.ndarray, truth: np.ndarray,
                   observed_mask: np.ndarray, threshold: float = 0.5,
                   spacing: tuple[float, float, float] = (1., 1., 1.),
                   hidden_mask: np.ndarray | None = None) -> dict[str, float | int | bool | None]:
    probability = np.asarray(probability, dtype=np.float32)
    truth = np.asarray(truth) >= 0.5
    observed_mask = np.asarray(observed_mask) >= 0.5
    if probability.shape != truth.shape or truth.shape != observed_mask.shape:
        raise ValueError("Probability, truth and observation mask must have identical shape")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Predictions must be finite probabilities in [0, 1]")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and lie in [0, 1]")
    if probability.ndim != 3:
        raise ValueError("Metrics require a three-dimensional volume")
    predicted = probability >= threshold
    blind = ~observed_mask
    intersection = np.count_nonzero(predicted & truth & blind)
    union = np.count_nonzero((predicted | truth) & blind)
    blind_pred = np.count_nonzero(predicted & blind)
    blind_truth = np.count_nonzero(truth & blind)
    overall_intersection = np.count_nonzero(predicted & truth)
    overall_union = np.count_nonzero(predicted | truth)
    predicted_n, truth_n = np.count_nonzero(predicted), np.count_nonzero(truth)
    observed_n = np.count_nonzero(observed_mask)
    observed_error = np.count_nonzero((predicted != truth) & observed_mask)
    distances = surface_distances(predicted, truth, spacing)
    answer = {
        "blind_iou": float(intersection / union) if union else 1.0,
        "blind_dice": float(2 * intersection / (blind_pred + blind_truth))
            if blind_pred + blind_truth else 1.0,
        "overall_iou": float(overall_intersection / overall_union) if overall_union else 1.0,
        "overall_dice": float(2 * overall_intersection / (predicted_n + truth_n))
            if predicted_n + truth_n else 1.0,
        "observed_error": float(observed_error / observed_n) if observed_n else None,
        "observed_violation_count": int(observed_error),
        "observed_mse_truth": float(np.mean((probability[observed_mask] - truth[observed_mask]) ** 2))
            if observed_n else None,
        "observed_n": int(observed_n),
        "volume_relative_error": float((predicted_n - truth_n) / max(truth_n, 1)),
        "volume_absolute_relative_error": float(abs(predicted_n - truth_n) / max(truth_n, 1)),
        "prediction_volume_voxels": int(predicted_n), "truth_volume_voxels": int(truth_n),
        "brier_blind": float(np.mean((probability[blind] - truth[blind]) ** 2))
            if np.any(blind) else None,
        "empty_prediction": bool(not np.any(predicted)),
        "empty_truth": bool(not np.any(truth)),
        **distances,
        "distance_unit": "voxel" if np.array_equal(np.asarray(spacing), np.ones(3)) else "coordinate_unit",
    }
    if hidden_mask is not None:
        hidden_mask = np.asarray(hidden_mask) >= 0.5
        if hidden_mask.shape != truth.shape:
            raise ValueError("Hidden mask shape must match the volume")
        hidden_mask &= ~observed_mask
        hidden_n = np.count_nonzero(hidden_mask)
        answer["hidden_error"] = float(np.count_nonzero((predicted != truth) & hidden_mask) / hidden_n) if hidden_n else None
        answer["hidden_n"] = int(hidden_n)
    return answer
