"""Documentation translated to English for open-source release."""

import os
import sys
import argparse
import re
import random
import warnings
from types import SimpleNamespace
from typing import List, Tuple, Optional

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch, Circle


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from dataset import read_ply, point_in_mesh
from model_factory import create_model
from comparisons.models.interpolation import KrigingModel
from comparisons.models.unet import UNet3D
from comparisons.train_unet import train_unet

try:
    from sklearn.exceptions import ConvergenceWarning as _SklearnConvergenceWarning
except Exception:
    _SklearnConvergenceWarning = Warning

try:
    from skimage.measure import marching_cubes
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


def _stable_seed(text: str, base: int = 42) -> int:
    return int((base + sum(ord(c) for c in text)) % (2 ** 31 - 1))


def _normalize_verts(verts: np.ndarray) -> np.ndarray:
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    span = hi - lo
    span[span < 1e-8] = 1.0
    return (verts - lo) / span


def _axes_unit(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_axis_off()


def _pick_largest_ply(real_dir: str, n_rows: int) -> List[str]:
    files = [
        os.path.join(real_dir, f)
        for f in os.listdir(real_dir)
        if f.lower().endswith('.ply')
    ]
    files.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return files[:max(1, min(n_rows, len(files)))]


def _list_ply_files(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith('.ply')
    ]


def _expand_selected_items(items: Optional[List[str]]) -> Optional[List[str]]:
    if items is None:
        return None
    expanded: List[str] = []
    for item in items:
        if item is None:
            continue
        for token in str(item).split(','):
            token = token.strip()
            if token:
                expanded.append(token)
    return expanded


def _resolve_selected_pool(directory: str,
                           selected_items: Optional[List[str]],
                           label: str) -> List[str]:
    all_files = _list_ply_files(directory)
    requested = _expand_selected_items(selected_items)
    if not requested:
        return all_files

    basename_map = {}
    for p in all_files:
        basename_map.setdefault(os.path.basename(p), []).append(p)

    resolved: List[str] = []
    missing: List[str] = []

    for item in requested:
        chosen = None
        if os.path.isabs(item) and os.path.isfile(item):
            chosen = os.path.abspath(item)
        else:
            rel_candidate = os.path.abspath(os.path.join(directory, item))
            if os.path.isfile(rel_candidate):
                chosen = rel_candidate
            else:
                matches = basename_map.get(os.path.basename(item), [])
                if len(matches) > 1:
                    print(f"[translated_text] {label} translated_text '{item}' translated_text, translated_text.")
                if matches:
                    chosen = matches[0]

        if chosen is None:
            missing.append(item)
            continue
        if chosen not in resolved:
            resolved.append(chosen)

    if missing:
        print(f"[translated_text] {label} translated_text, translated_text: {missing}")
    if not resolved:
        raise RuntimeError(f"{label} translated_text, translated_text --selected-*-files translated_text.")

    print(f"[translated_text] {label} translated_text: {len(resolved)} translated_text")
    return resolved


def _pick_mixed_largest_ply(real_dir: str,
                            pretrain_dir: str,
                            n_rows: int,
                            real_rows: int = 2,
                            virtual_rows: int = 2,
                            random_samples: bool = True,
                            sample_seed: Optional[int] = None,
                            real_files: Optional[List[str]] = None,
                            virtual_files: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    n_rows = max(1, int(n_rows))
    real_rows = max(0, int(real_rows))
    virtual_rows = max(0, int(virtual_rows))

    real_files = list(real_files) if real_files is not None else _list_ply_files(real_dir)
    virtual_files = list(virtual_files) if virtual_files is not None else _list_ply_files(pretrain_dir)

    if not random_samples:
        real_files.sort(key=lambda p: os.path.getsize(p), reverse=True)
        virtual_files.sort(key=lambda p: os.path.getsize(p), reverse=True)

    if len(real_files) == 0 and len(virtual_files) == 0:
        return []

    pref_total = real_rows + virtual_rows
    if pref_total == 0:
        real_rows = n_rows
        virtual_rows = 0
        pref_total = real_rows

    if pref_total > n_rows:
        real_rows = int(round(n_rows * (real_rows / pref_total)))
        real_rows = min(real_rows, n_rows)
        virtual_rows = n_rows - real_rows

    take_real = min(real_rows, len(real_files))
    take_virtual = min(virtual_rows, len(virtual_files))

    remaining = n_rows - take_real - take_virtual
    extra_real = len(real_files) - take_real
    extra_virtual = len(virtual_files) - take_virtual

    while remaining > 0 and (extra_real > 0 or extra_virtual > 0):
        if extra_real >= extra_virtual and extra_real > 0:
            take_real += 1
            extra_real -= 1
            remaining -= 1
        elif extra_virtual > 0:
            take_virtual += 1
            extra_virtual -= 1
            remaining -= 1
        else:
            break

    if random_samples:
        rng = random.Random(sample_seed) if sample_seed is not None else random.Random()
        selected_real = rng.sample(real_files, take_real) if take_real > 0 else []
        selected_virtual = rng.sample(virtual_files, take_virtual) if take_virtual > 0 else []
    else:
        selected_real = real_files[:take_real]
        selected_virtual = virtual_files[:take_virtual]

    selected = [('real', p) for p in selected_real] + [('virtual', p) for p in selected_virtual]
    return selected[:n_rows]


def _voxelize_mesh(verts: np.ndarray, faces: np.ndarray,
                  grid_size: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d, h, w = grid_size
    vmin = verts.min(axis=0).astype(np.float32)
    vmax = verts.max(axis=0).astype(np.float32)

    xs = np.linspace(vmin[0], vmax[0], w)
    ys = np.linspace(vmin[1], vmax[1], h)
    zs = np.linspace(vmin[2], vmax[2], d)
    gz, gy, gx = np.meshgrid(zs, ys, xs, indexing='ij')
    grid_centers = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    occ = point_in_mesh(grid_centers, verts, faces).astype(np.float32)
    vox = occ.reshape(d, h, w)
    return vox, vmin, vmax


def _build_sparse_input(vox: np.ndarray,
                        vmin: np.ndarray,
                        vmax: np.ndarray,
                        num_holes: int,
                        samples_per_hole: int,
                        seed: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    d, h, w = vox.shape
    rng = np.random.RandomState(seed)
    obs = np.zeros((2, d, h, w), dtype=np.float32)
    holes_norm = []

    x_range = float(max(vmax[0] - vmin[0], 1e-8))
    y_range = float(max(vmax[1] - vmin[1], 1e-8))
    z_range = float(max(vmax[2] - vmin[2], 1e-8))

    for _ in range(max(1, int(num_holes))):
        cx = rng.uniform(vmin[0], vmax[0])
        cy = rng.uniform(vmin[1], vmax[1])
        zs_line = np.linspace(vmin[2], vmax[2], max(2, int(samples_per_hole)))

        xs = cx + rng.uniform(-0.01, 0.01, len(zs_line)) * x_range
        ys = cy + rng.uniform(-0.01, 0.01, len(zs_line)) * y_range
        zt = zs_line + rng.uniform(-0.002, 0.002, len(zs_line)) * z_range
        pts_world = np.stack([xs, ys, zt], axis=1).astype(np.float32)

        pts_norm = (pts_world - vmin[None, :]) / (vmax[None, :] - vmin[None, :] + 1e-8)
        pts_norm = np.clip(pts_norm, 0.0, 1.0)
        holes_norm.append(pts_norm)

        xi = np.clip(np.floor(pts_norm[:, 0] * (w - 1)).astype(int), 0, w - 1)
        yi = np.clip(np.floor(pts_norm[:, 1] * (h - 1)).astype(int), 0, h - 1)
        zi = np.clip(np.floor(pts_norm[:, 2] * (d - 1)).astype(int), 0, d - 1)

        labels = vox[zi, yi, xi].astype(np.float32)
        obs[0, zi, yi, xi] = labels
        obs[1, zi, yi, xi] = 1.0

    return obs, holes_norm


def _vox_to_mesh_norm(vox: np.ndarray, threshold: float = 0.5) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if vox is None:
        return None, None
    if np.max(vox) <= threshold:
        return None, None
    if not _HAS_SKIMAGE:
        return None, None

    try:
        verts, faces, _, _ = marching_cubes(vox.astype(np.float32), level=threshold)
        d, h, w = vox.shape
        z = verts[:, 0] / max(d - 1, 1)
        y = verts[:, 1] / max(h - 1, 1)
        x = verts[:, 2] / max(w - 1, 1)
        verts_norm = np.stack([x, y, z], axis=1).astype(np.float32)
        return verts_norm, faces.astype(np.int32)
    except Exception:
        return None, None


def _render_mesh(ax, verts: np.ndarray, faces: np.ndarray, color: str,
                 alpha: float = 0.88, edge_alpha: float = 0.7) -> None:
    if verts is None or faces is None or len(faces) == 0:
        return
    tri = verts[faces]
    mesh = Poly3DCollection(
        tri,
        alpha=alpha,
        facecolor=color,
        edgecolor=(0.12, 0.12, 0.12, edge_alpha),
        linewidths=0.12,
    )
    ax.add_collection3d(mesh)
    _axes_unit(ax)


def _render_sparse(ax,
                   holes_norm: List[np.ndarray],
                   gt_vox: Optional[np.ndarray],
                   ore_color: str,
                   outside_color: str = '#9a9a9a') -> None:
    d = h = w = None
    if gt_vox is not None and gt_vox.ndim == 3:
        d, h, w = gt_vox.shape

    for hole in holes_norm:
        # English comment for public release.
        ax.plot(hole[:, 0], hole[:, 1], hole[:, 2], color=outside_color, linewidth=1.2, alpha=0.92)

        if d is None:
            ax.scatter(hole[:, 0], hole[:, 1], hole[:, 2], c=outside_color, s=5, alpha=0.85)
            continue

        xi = np.clip(np.floor(hole[:, 0] * (w - 1)).astype(int), 0, w - 1)
        yi = np.clip(np.floor(hole[:, 1] * (h - 1)).astype(int), 0, h - 1)
        zi = np.clip(np.floor(hole[:, 2] * (d - 1)).astype(int), 0, d - 1)
        labels = gt_vox[zi, yi, xi] > 0.5

        outside_mask = ~labels
        inside_mask = labels

        if np.any(outside_mask):
            ax.scatter(
                hole[outside_mask, 0],
                hole[outside_mask, 1],
                hole[outside_mask, 2],
                c=outside_color,
                s=7,
                alpha=0.88,
                linewidths=0.0,
            )
        if np.any(inside_mask):
            ax.scatter(
                hole[inside_mask, 0],
                hole[inside_mask, 1],
                hole[inside_mask, 2],
                c=ore_color,
                s=10,
                alpha=0.95,
                linewidths=0.0,
            )
    _axes_unit(ax)


def _render_voxel_points(ax, vox: np.ndarray, color: str, threshold: float = 0.5) -> None:
    idx = np.argwhere(vox > threshold)
    if len(idx) == 0:
        return
    if len(idx) > 4000:
        choose = np.random.choice(len(idx), 4000, replace=False)
        idx = idx[choose]
    d, h, w = vox.shape
    z = idx[:, 0] / max(d - 1, 1)
    y = idx[:, 1] / max(h - 1, 1)
    x = idx[:, 2] / max(w - 1, 1)
    ax.scatter(x, y, z, c=color, s=1.2, alpha=0.65)
    _axes_unit(ax)


def _set_local_zoom(ax, center_xyz: np.ndarray, radius: float = 0.18) -> None:
    cx, cy, cz = float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])
    r = float(max(0.05, radius))
    ax.set_xlim(max(0.0, cx - r), min(1.0, cx + r))
    ax.set_ylim(max(0.0, cy - r), min(1.0, cy + r))
    ax.set_zlim(max(0.0, cz - r), min(1.0, cz + r))
    ax.set_axis_off()


def _choose_zoom_center(unet_pred: np.ndarray, ours_pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    unet_bin = unet_pred > threshold
    ours_bin = ours_pred > threshold
    diff = np.logical_xor(unet_bin, ours_bin)

    idx = np.argwhere(diff)
    if idx.size == 0:
        idx = np.argwhere(unet_bin)
    if idx.size == 0:
        return np.array([0.5, 0.5, 0.5], dtype=np.float32)

    center_zyx = np.median(idx, axis=0).astype(np.float32)
    d, h, w = unet_pred.shape
    z = center_zyx[0] / max(d - 1, 1)
    y = center_zyx[1] / max(h - 1, 1)
    x = center_zyx[2] / max(w - 1, 1)
    return np.array([x, y, z], dtype=np.float32)


def _add_projected_circle(ax, center_xyz: np.ndarray,
                          radius_axes: float = 0.09,
                          color: str = 'red') -> None:
    x, y, z = float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])
    x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    disp_xy = ax.transData.transform((x2, y2))
    ax_xy = ax.transAxes.inverted().transform(disp_xy)

    circle = Circle(
        (float(ax_xy[0]), float(ax_xy[1])),
        radius=radius_axes,
        transform=ax.transAxes,
        fill=False,
        edgecolor=color,
        linewidth=2.2,
        alpha=0.95,
        clip_on=False,
        zorder=20,
    )
    ax.figure.add_artist(circle)
    ax.text2D(
        float(ax_xy[0]) + radius_axes + 0.01,
        float(ax_xy[1]),
        'Artifact Region',
        transform=ax.transAxes,
        color=color,
        fontsize=9,
        fontweight='bold',
        zorder=21,
        va='center',
        ha='left',
    )


def _render_uncertainty_variance(ax,
                                 var_map: np.ndarray,
                                 mean_map: np.ndarray,
                                 holes_norm: List[np.ndarray],
                                 max_points: int = 12000,
                                 threshold: float = 0.5,
                                 prob_floor: float = 0.015) -> Tuple[float, float]:
    # English comment for public release.
    # English comment for public release.
    prob_map = np.clip(mean_map.astype(np.float32), 0.0, 1.0)
    disp_mask = prob_map >= max(0.005, float(prob_floor))
    idx = np.argwhere(disp_mask)
    if idx.size == 0:
        idx = np.argwhere(prob_map >= np.percentile(prob_map, 99.5))

    if idx.size == 0:
        _axes_unit(ax)
        return 0.0, 1.0

    prob_vals = prob_map[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float32)
    var_vals = var_map[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float32)

    if len(idx) > max_points:
        var_norm = (var_vals - float(np.min(var_vals))) / (float(np.ptp(var_vals)) + 1e-8)
        score = prob_vals + 0.20 * var_norm
        top_idx = np.argpartition(score, -max_points)[-max_points:]
        idx = idx[top_idx]
        prob_vals = prob_vals[top_idx]
        var_vals = var_vals[top_idx]

    d, h, w = var_map.shape
    z = idx[:, 0] / max(d - 1, 1)
    y = idx[:, 1] / max(h - 1, 1)
    x = idx[:, 2] / max(w - 1, 1)

    vmin = float(np.percentile(prob_vals, 1.0))
    vmax = float(np.percentile(prob_vals, 99.5))
    if vmax <= vmin:
        vmax = float(np.max(prob_vals) + 1e-8)

    prob_norm = np.clip((prob_vals - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    sizes = 0.8 + 7.0 * prob_norm
    colors = plt.cm.turbo(prob_norm)
    colors[:, 3] = 0.08 + 0.90 * prob_norm

    sc = ax.scatter(
        x, y, z,
        c=colors,
        s=sizes,
        linewidths=0.0,
    )

    # English comment for public release.
    if len(var_vals) > 0:
        hv = var_vals >= np.percentile(var_vals, 92.0)
        if np.any(hv):
            ax.scatter(x[hv], y[hv], z[hv], c='black', s=1.5, alpha=0.30, linewidths=0.0)

    for hole in holes_norm:
        ax.plot(hole[:, 0], hole[:, 1], hole[:, 2], color='black', linewidth=1.1, alpha=0.95)

    _axes_unit(ax)
    ax.view_init(elev=22, azim=35)

    # English comment for public release.
    return vmin, vmax


def _resolve_state_dict(ckpt_obj) -> Tuple[dict, dict]:
    state_dict = None
    hparams = {}

    if isinstance(ckpt_obj, dict):
        if isinstance(ckpt_obj.get('hparams'), dict):
            hparams = ckpt_obj['hparams']
        if isinstance(ckpt_obj.get('state_dict'), dict):
            state_dict = ckpt_obj['state_dict']
        elif isinstance(ckpt_obj.get('model_state_dict'), dict):
            state_dict = ckpt_obj['model_state_dict']
        else:
            if all(isinstance(k, str) for k in ckpt_obj.keys()):
                state_dict = ckpt_obj

    if state_dict is None:
        raise RuntimeError('translated_text checkpoint translated_text state_dict')

    cleaned = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith('module.') else k
        cleaned[nk] = v
    return cleaned, hparams


def _infer_model_cfg(state_dict: dict, hparams: dict,
                     default_grid: Tuple[int, int, int]) -> dict:
    keys = set(state_dict.keys())
    is_octree = any(k.startswith('encoder.sparse_convs.') for k in keys)
    model_type = 'octree' if is_octree else 'standard'

    latent_dim = int(hparams.get('latent_dim', 256))
    if 'fc_mu.weight' in state_dict and state_dict['fc_mu.weight'].ndim == 2:
        latent_dim = int(state_dict['fc_mu.weight'].shape[0])

    base_channels = int(hparams.get('base_channels', 32))
    if model_type == 'standard' and 'enc_conv_in.weight' in state_dict:
        base_channels = int(state_dict['enc_conv_in.weight'].shape[0])
    if model_type == 'octree' and 'encoder.sparse_convs.0.weight' in state_dict:
        base_channels = int(state_dict['encoder.sparse_convs.0.weight'].shape[0])

    num_levels = int(hparams.get('octree_levels', 4))
    if model_type == 'octree':
        level_ids = []
        pat = re.compile(r'^encoder\.sparse_convs\.(\d+)\.weight$')
        for k in keys:
            m = pat.match(k)
            if m:
                level_ids.append(int(m.group(1)))
        if level_ids:
            num_levels = max(level_ids) + 1

    inferred_grid = tuple(default_grid)
    if 'fc_mu.weight' in state_dict and state_dict['fc_mu.weight'].ndim == 2:
        enc_flat = int(state_dict['fc_mu.weight'].shape[1])
        if model_type == 'standard':
            denom = base_channels * 16
            if denom > 0 and enc_flat % denom == 0:
                sprod = enc_flat // denom
                s = int(round(sprod ** (1.0 / 3.0)))
                if s > 0 and s * s * s == sprod:
                    g = s * 16
                    inferred_grid = (g, g, g)
        else:
            denom = base_channels * (2 ** max(0, num_levels - 1))
            if denom > 0 and enc_flat % denom == 0:
                sprod = enc_flat // denom
                s = int(round(sprod ** (1.0 / 3.0)))
                if s > 0 and s * s * s == sprod:
                    g = s * (2 ** max(0, num_levels - 1))
                    inferred_grid = (g, g, g)

    return {
        'model_type': model_type,
        'use_lora': bool(hparams.get('use_lora', False)),
        'lora_preset': hparams.get('lora_preset', 'light'),
        'latent_dim': latent_dim,
        'base_channels': base_channels,
        'num_levels': num_levels,
        'grid_size': inferred_grid,
    }


def _load_state_dict_compatible(model, state_dict: dict) -> Tuple[int, int]:
    model_sd = model.state_dict()
    filtered = {}
    skipped_mismatch = 0

    for k, v in state_dict.items():
        if k not in model_sd:
            continue
        if model_sd[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped_mismatch += 1

    model.load_state_dict(filtered, strict=False)
    return len(filtered), skipped_mismatch


def _safe_torch_load(path: str, map_location='cpu'):
    """Documentation translated to English for open-source release."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)


def _load_ours_model(model_path: str, device: torch.device,
                     grid_size: Tuple[int, int, int]):
    ckpt = _safe_torch_load(model_path, map_location='cpu')

    state_dict, hparams = _resolve_state_dict(ckpt)
    inferred = _infer_model_cfg(state_dict, hparams, grid_size)

    candidates = []
    if hparams:
        candidates.append({
            'model_type': hparams.get('model_type', 'standard'),
            'use_lora': bool(hparams.get('use_lora', False)),
            'lora_preset': hparams.get('lora_preset', 'light'),
            'latent_dim': int(hparams.get('latent_dim', 256)),
            'base_channels': int(hparams.get('base_channels', 32)),
            'num_levels': int(hparams.get('octree_levels', 4)),
            'grid_size': tuple(hparams.get('grid_size', grid_size)) if isinstance(hparams.get('grid_size', None), (list, tuple)) else grid_size,
        })

    candidates.extend([
        {'model_type': 'standard', 'use_lora': False, 'lora_preset': 'light', 'latent_dim': inferred['latent_dim'], 'base_channels': inferred['base_channels'], 'num_levels': 4, 'grid_size': inferred['grid_size']},
        {'model_type': 'standard', 'use_lora': False, 'lora_preset': 'light', 'latent_dim': 1024, 'base_channels': 32, 'num_levels': 4, 'grid_size': grid_size},
        {'model_type': 'standard', 'use_lora': False, 'lora_preset': 'light', 'latent_dim': 256, 'base_channels': 32, 'num_levels': 4, 'grid_size': grid_size},
        {'model_type': 'octree',   'use_lora': False, 'lora_preset': 'light', 'latent_dim': inferred['latent_dim'], 'base_channels': inferred['base_channels'], 'num_levels': inferred['num_levels'], 'grid_size': inferred['grid_size']},
        {'model_type': 'octree',   'use_lora': False, 'lora_preset': 'light', 'latent_dim': 1024, 'base_channels': 32, 'num_levels': 4, 'grid_size': grid_size},
        {'model_type': 'octree',   'use_lora': False, 'lora_preset': 'light', 'latent_dim': 256, 'base_channels': 32, 'num_levels': 4, 'grid_size': grid_size},
    ])

    uniq_candidates = []
    seen = set()
    for cfg in candidates:
        sig = (
            cfg.get('model_type'), cfg.get('use_lora'), cfg.get('lora_preset'),
            cfg.get('latent_dim'), cfg.get('base_channels'), cfg.get('num_levels'),
            tuple(cfg.get('grid_size', grid_size)),
        )
        if sig in seen:
            continue
        seen.add(sig)
        uniq_candidates.append(cfg)

    last_err = None
    for cfg in uniq_candidates:
        try:
            model_result = create_model(
                model_type=cfg['model_type'],
                use_lora=cfg['use_lora'],
                lora_preset=cfg['lora_preset'],
                grid_size=tuple(cfg.get('grid_size', grid_size)),
                latent_dim=cfg['latent_dim'],
                base_channels=cfg['base_channels'],
                num_levels=cfg['num_levels'],
                device=device,
                log_mode='brief',
            )
            model = model_result[0] if isinstance(model_result, tuple) else model_result
            loaded_n, skipped_n = _load_state_dict_compatible(model, state_dict)
            model.to(device)
            model.eval()
            run_grid = tuple(cfg.get('grid_size', grid_size))

            with torch.no_grad():
                dummy = torch.zeros((1, 2, run_grid[0], run_grid[1], run_grid[2]), dtype=torch.float32, device=device)
                _ = model(dummy)

            print(
                f"[Ours] translated_text: type={cfg['model_type']}, latent={cfg['latent_dim']}, "
                f"base={cfg['base_channels']}, grid={run_grid}, loaded={loaded_n}, skipped_mismatch={skipped_n}"
            )
            return model, run_grid
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f'translated_text Ours translated_text: {last_err}')


def _load_or_train_unet(unet_ckpt: str,
                        train_data_dir: str,
                        unet_epochs: int,
                        unet_batch_size: int,
                        unet_lr: float) -> Optional[UNet3D]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = os.path.abspath(unet_ckpt)

    if not os.path.exists(ckpt_path):
        print(f"[U-Net] translated_text: {ckpt_path}")
        print(f"[U-Net] translated_text(epochs={unet_epochs})...")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        train_args = SimpleNamespace(
            epochs=int(unet_epochs),
            batch_size=int(unet_batch_size),
            lr=float(unet_lr),
            ply_dir=train_data_dir,
            output_dir=os.path.dirname(ckpt_path),
        )
        saved_path = train_unet(train_args)
        if saved_path and os.path.exists(saved_path):
            ckpt_path = saved_path

    if not os.path.exists(ckpt_path):
        print("[U-Net] translated_text, translated_text U-Net translated_text.")
        return None

    model = UNet3D(in_channels=2, out_channels=1).to(device)
    state = _safe_torch_load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
        model.load_state_dict(state['state_dict'], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[U-Net] translated_text: {ckpt_path}")
    return model


def _predict_ik(obs_np: np.ndarray, grid_size: Tuple[int, int, int]) -> np.ndarray:
    try:
        ik_model = KrigingModel(grid_size=grid_size)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                category=_SklearnConvergenceWarning,
            )
            pred = ik_model.fit_predict(obs_np)
        if pred.ndim == 4:
            pred = pred[0]
        return np.clip(pred.astype(np.float32), 0.0, 1.0)
    except Exception as e:
        print(f"[IK] translated_text, translated_text: {e}")
        return np.zeros(grid_size, dtype=np.float32)


def _predict_unet(model: Optional[UNet3D], obs_np: np.ndarray) -> Optional[np.ndarray]:
    if model is None:
        return None
    device = next(model.parameters()).device
    obs = torch.from_numpy(obs_np).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(obs).squeeze().detach().cpu().numpy().astype(np.float32)
    return np.clip(pred, 0.0, 1.0)


def _predict_ours(model, obs_np: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    obs = torch.from_numpy(obs_np).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(obs)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        if logits.ndim == 5 and logits.shape[1] == 1:
            logits = logits[:, 0]
        prob = torch.sigmoid(logits).squeeze().detach().cpu().numpy().astype(np.float32)
    return np.clip(prob, 0.0, 1.0)


def parse_args():
    repo_root = os.path.dirname(_SCRIPT_DIR)
    p = argparse.ArgumentParser(
        description='translated_text(GT / Sparse / IK / 3D U-Net / Ours)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--model-dir', type=str, default=os.path.join(repo_root, 'results', 'model'))
    p.add_argument('--model-name', type=str, default='best_model.pth')
    p.add_argument('--real-dir', type=str, default=os.path.join(repo_root, 'data', 'mining_ply'))
    p.add_argument('--pretrain-dir', type=str, default=os.path.join(repo_root, 'data', 'mining_ply_pretrain'))
    p.add_argument('--unet-ckpt', type=str, default=os.path.join(repo_root, 'results', 'comparisons', 'models', 'unet_best.pth'))
    p.add_argument('--out-dir', type=str, default=os.path.join(repo_root, 'results', 'figures'))

    p.add_argument('--rows', type=int, default=4, help='translated_text')
    p.add_argument('--real-rows', type=int, default=2, help='translated_text')
    p.add_argument('--virtual-rows', type=int, default=2, help='translated_text')
    p.add_argument('--fixed-samples', action='store_true',
                   help='translated_text, translated_text')
    p.add_argument('--sample-seed', type=int, default=None,
                   help='translated_text; translated_text')
    p.add_argument('--selected-virtual-files', type=str, nargs='*', default=None,
                   help='translated_text(translated_text)')
    p.add_argument('--selected-real-files', type=str, nargs='*', default=None,
                   help='translated_text(translated_text)')
    p.add_argument('--grid-size', type=int, default=32)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--num-holes', type=int, default=8)
    p.add_argument('--samples-per-hole', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dpi', type=int, default=300)

    p.add_argument('--unet-epochs', type=int, default=10)
    p.add_argument('--unet-batch-size', type=int, default=16)
    p.add_argument('--unet-lr', type=float, default=1e-3)
    p.add_argument('--zoom-radius', type=float, default=0.18,
                   help='Local zoom radius in normalized coordinate space')
    p.add_argument('--zoom-sample-rank', type=int, default=1,
                   help='Rank of ore body by file size for local zoom (1 means the largest)')
    p.add_argument('--uncertainty-mc-samples', type=int, default=24,
                   help='Monte Carlo forward passes for uncertainty variance estimation')
    return p.parse_args()


def generate_uncertainty_variance_figure(args):
    model_path = os.path.join(args.model_dir, args.model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'translated_text Ours translated_text: {model_path}')
    if not os.path.isdir(args.pretrain_dir):
        raise FileNotFoundError(f'translated_text: {args.pretrain_dir}')

    os.makedirs(args.out_dir, exist_ok=True)
    grid_size = (args.grid_size, args.grid_size, args.grid_size)

    requested_real = _expand_selected_items(getattr(args, 'selected_real_files', None))
    requested_virtual = _expand_selected_items(getattr(args, 'selected_virtual_files', None))
    custom_pool_mode = bool(requested_real or requested_virtual)

    use_fixed = bool(getattr(args, 'fixed_samples', False))
    sample_seed = getattr(args, 'sample_seed', None)

    if custom_pool_mode:
        real_pool = _resolve_selected_pool(args.real_dir, getattr(args, 'selected_real_files', None), 'Real')
        virtual_pool = _resolve_selected_pool(args.pretrain_dir, getattr(args, 'selected_virtual_files', None), 'Virtual')
        candidate_pool = list(dict.fromkeys(real_pool + virtual_pool))
        if not candidate_pool:
            raise RuntimeError('translated_text, translated_text.')
        candidate_pool.sort(key=lambda p: os.path.getsize(p), reverse=True)
        pool_size = len(candidate_pool)

        if use_fixed:
            rank = int(max(1, min(getattr(args, 'zoom_sample_rank', 1), len(candidate_pool))))
            target_file = candidate_pool[rank - 1]
            mode_text = f'fixed-custom(rank={rank})'
        else:
            rng = random.Random(sample_seed) if sample_seed is not None else random.Random()
            target_file = rng.choice(candidate_pool)
            mode_text = f'random-custom(seed={sample_seed})'
    else:
        virtual_files = _list_ply_files(args.pretrain_dir)
        if not virtual_files:
            raise RuntimeError(f'translated_text PLY: {args.pretrain_dir}')

        virtual_files.sort(key=lambda p: os.path.getsize(p), reverse=True)
        top_ratio = float(getattr(args, 'uncertainty_top_ratio', 0.35))
        top_ratio = float(np.clip(top_ratio, 0.05, 1.0))
        large_pool_size = max(1, int(np.ceil(len(virtual_files) * top_ratio)))
        large_pool = virtual_files[:large_pool_size]
        pool_size = len(large_pool)

        if use_fixed:
            rank = int(max(1, min(getattr(args, 'zoom_sample_rank', 1), len(large_pool))))
            target_file = large_pool[rank - 1]
            mode_text = f'fixed-largest(top {top_ratio:.0%}, rank={rank})'
        else:
            rng = random.Random(sample_seed) if sample_seed is not None else random.Random()
            target_file = rng.choice(large_pool)
            mode_text = f'random-from-top{int(round(top_ratio * 100))}%(seed={sample_seed})'

            print(f"[translated_text] Uncertainty translated_text: {mode_text}, pool={pool_size}, selected={os.path.basename(target_file)}")
    source_name = os.path.splitext(os.path.basename(target_file))[0]
    ore_label = source_name

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ours_model, effective_grid = _load_ours_model(model_path, device, grid_size)
    grid_size = tuple(effective_grid)

    verts, faces = read_ply(target_file)
    gt_vox, vmin, vmax = _voxelize_mesh(verts, faces, grid_size=grid_size)
    obs_np, holes_norm = _build_sparse_input(
        gt_vox, vmin, vmax,
        num_holes=args.num_holes,
        samples_per_hole=args.samples_per_hole,
        seed=_stable_seed(source_name, args.seed),
    )

    mc_samples = max(4, int(getattr(args, 'uncertainty_mc_samples', 24)))
    pred_list = []
    for _ in range(mc_samples):
        pred_list.append(_predict_ours(ours_model, obs_np))

    pred_stack = np.stack(pred_list, axis=0).astype(np.float32)
    pred_mean = np.mean(pred_stack, axis=0)
    pred_var = np.var(pred_stack, axis=0)

    fig = plt.figure(figsize=(13.5, 10.5), dpi=args.dpi)
    fig.patch.set_facecolor('white')
    ax_gt = fig.add_subplot(2, 2, 1, projection='3d')
    ax_var_a = fig.add_subplot(2, 2, 2, projection='3d')
    ax_var_b = fig.add_subplot(2, 2, 3, projection='3d')
    ax_var_c = fig.add_subplot(2, 2, 4, projection='3d')

    gt_verts_norm = _normalize_verts(verts)
    if faces is not None and len(faces) > 0:
        _render_mesh(ax_gt, gt_verts_norm, faces, color='#4c78a8', alpha=0.88, edge_alpha=0.25)
    else:
        gt_mesh_verts, gt_mesh_faces = _vox_to_mesh_norm(gt_vox, threshold=args.threshold)
        if gt_mesh_verts is not None and gt_mesh_faces is not None and len(gt_mesh_faces) > 0:
            _render_mesh(ax_gt, gt_mesh_verts, gt_mesh_faces, color='#4c78a8', alpha=0.88, edge_alpha=0.25)

    for hole in holes_norm:
        ax_gt.plot(hole[:, 0], hole[:, 1], hole[:, 2], color='#d62728', linewidth=1.0, alpha=0.9)
    _axes_unit(ax_gt)
    ax_gt.view_init(elev=22, azim=35)
    ax_gt.set_title('Ground Truth Ore Body', fontsize=11, pad=4, fontweight='bold')

    vmins = []
    vmaxs = []
    prob_axes = [ax_var_a, ax_var_b, ax_var_c]
    for ax in prob_axes:
        v0, v1 = _render_uncertainty_variance(
            ax,
            var_map=pred_var,
            mean_map=pred_mean,
            holes_norm=holes_norm,
            threshold=args.threshold,
        )
        vmins.append(v0)
        vmaxs.append(v1)

    ax_var_a.view_init(elev=22, azim=35)
    ax_var_b.view_init(elev=22, azim=125)
    ax_var_c.view_init(elev=72, azim=35)

    ax_var_a.set_title('Probability View A', fontsize=10, pad=4, fontweight='bold')
    ax_var_b.set_title('Probability View B', fontsize=10, pad=4, fontweight='bold')
    ax_var_c.set_title('Probability View C', fontsize=10, pad=4, fontweight='bold')

    fig.suptitle('Uncertainty-Aware Prospectivity Map (Multi-view)', fontsize=13, y=0.975, fontweight='bold')
    fig.text(
        0.5,
        0.945,
        f'Sample: {ore_label} | Top-left: GT ore body | Other panels: MC mean probability from different directions',
        ha='center',
        va='center',
        fontsize=9,
        color='#333333',
    )

    vmin_cb = float(min(vmins)) if vmins else 0.0
    vmax_cb = float(max(vmaxs)) if vmaxs else 1.0
    mappable = plt.cm.ScalarMappable(cmap='turbo')
    mappable.set_clim(vmin_cb, vmax_cb)
    cbar = fig.colorbar(mappable, ax=prob_axes, fraction=0.026, pad=0.02)
    cbar.set_label('Ore-body Probability P(occ | drillholes)', fontsize=10)

    plt.subplots_adjust(left=0.03, right=0.92, top=0.91, bottom=0.04, wspace=0.02, hspace=0.06)

    model_stem = os.path.splitext(args.model_name)[0]
    out_png = os.path.join(args.out_dir, f'uncertainty_variance_{model_stem}.png')
    out_pdf = os.path.join(args.out_dir, f'uncertainty_variance_{model_stem}.pdf')
    fig.savefig(out_png, dpi=args.dpi, bbox_inches='tight', facecolor='white', pad_inches=0.03)
    fig.savefig(out_pdf, format='pdf', bbox_inches='tight', facecolor='white', pad_inches=0.03)
    plt.close(fig)

    print('[translated_text] translated_text:')
    print(f'  - {out_png}')
    print(f'  - {out_pdf}')
    return out_png, out_pdf


def generate_local_detail_figure(args):
    model_path = os.path.join(args.model_dir, args.model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'translated_text Ours translated_text: {model_path}')
    if not os.path.isdir(args.real_dir):
        raise FileNotFoundError(f'translated_text: {args.real_dir}')

    os.makedirs(args.out_dir, exist_ok=True)
    grid_size = (args.grid_size, args.grid_size, args.grid_size)
    requested_real = _expand_selected_items(getattr(args, 'selected_real_files', None))
    requested_virtual = _expand_selected_items(getattr(args, 'selected_virtual_files', None))
    custom_pool_mode = bool(requested_real or requested_virtual)

    if custom_pool_mode:
        real_pool = _resolve_selected_pool(args.real_dir, getattr(args, 'selected_real_files', None), 'Real')
        virtual_pool = _resolve_selected_pool(args.pretrain_dir, getattr(args, 'selected_virtual_files', None), 'Virtual')
        candidate_files = list(dict.fromkeys(real_pool + virtual_pool))
    else:
        candidate_files = _pick_largest_ply(args.real_dir, 10 ** 9)

    rows = max(1, int(args.rows))
    use_fixed = bool(getattr(args, 'fixed_samples', False))
    sample_seed = getattr(args, 'sample_seed', None)

    candidate_files.sort(key=lambda p: os.path.getsize(p), reverse=True)

    if use_fixed:
        selected_files = candidate_files[:rows]
        mode_text = 'fixed-largest'
    else:
        rng = random.Random(sample_seed) if sample_seed is not None else random.Random()
        k = min(rows, len(candidate_files))
        selected_files = rng.sample(candidate_files, k)
        mode_text = f'random(seed={sample_seed})' if sample_seed is not None else 'random(seed=None)'

    if not selected_files:
        raise RuntimeError(f'translated_text PLY: {args.real_dir}')

    rank = int(max(1, min(getattr(args, 'zoom_sample_rank', 1), len(selected_files))))
    target_file = selected_files[rank - 1]
    source_name = os.path.splitext(os.path.basename(target_file))[0]
    ore_label = f"Sample {rank}"
    print(f"[translated_text] Local Detail translated_text: {mode_text}, pool={len(selected_files)}, selected={os.path.basename(target_file)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ours_model, effective_grid = _load_ours_model(model_path, device, grid_size)
    grid_size = tuple(effective_grid)
    unet_model = _load_or_train_unet(
        args.unet_ckpt,
        train_data_dir=args.pretrain_dir if os.path.isdir(args.pretrain_dir) else args.real_dir,
        unet_epochs=args.unet_epochs,
        unet_batch_size=args.unet_batch_size,
        unet_lr=args.unet_lr,
    )
    if unet_model is None:
        raise RuntimeError('U-Net model unavailable; local zoom figure cannot be generated.')

    verts, faces = read_ply(target_file)
    gt_vox, vmin, vmax = _voxelize_mesh(verts, faces, grid_size=grid_size)
    obs_np, _ = _build_sparse_input(
        gt_vox, vmin, vmax,
        num_holes=args.num_holes,
        samples_per_hole=args.samples_per_hole,
        seed=_stable_seed(source_name, args.seed),
    )

    pred_unet = _predict_unet(unet_model, obs_np)
    pred_ours = _predict_ours(ours_model, obs_np)
    if pred_unet is None:
        raise RuntimeError('U-Net prediction failed; local zoom figure cannot be generated.')

    zoom_center = _choose_zoom_center(pred_unet, pred_ours, threshold=args.threshold)
    zoom_radius = float(max(0.08, getattr(args, 'zoom_radius', 0.18)))

    gt_zoom_verts, gt_zoom_faces = _vox_to_mesh_norm(gt_vox, threshold=args.threshold)
    unet_verts, unet_faces = _vox_to_mesh_norm(pred_unet, threshold=args.threshold)
    ours_verts, ours_faces = _vox_to_mesh_norm(pred_ours, threshold=args.threshold)

    fig = plt.figure(figsize=(13.8, 8.2), dpi=args.dpi)
    fig.patch.set_facecolor('white')

    # English comment for public release.
    # English comment for public release.
    # English comment for public release.
    ax_gt_full = fig.add_subplot(2, 3, 1, projection='3d')
    ax_unet_full = fig.add_subplot(2, 3, 2, projection='3d')
    ax_ours_full = fig.add_subplot(2, 3, 3, projection='3d')

    ax_gt_zoom = fig.add_subplot(2, 3, 4, projection='3d')
    ax_unet_zoom = fig.add_subplot(2, 3, 5, projection='3d')
    ax_ours_zoom = fig.add_subplot(2, 3, 6, projection='3d')

    # English comment for public release.
    _render_mesh(ax_gt_full, _normalize_verts(verts), faces, color='#7f7f7f', alpha=0.90)

    # English comment for public release.
    if unet_verts is not None:
        _render_mesh(ax_unet_full, unet_verts, unet_faces, color='#d95f02', alpha=0.90)
    else:
        _render_voxel_points(ax_unet_full, pred_unet, color='#d95f02', threshold=args.threshold)

    if ours_verts is not None:
        _render_mesh(ax_ours_full, ours_verts, ours_faces, color='#1f77b4', alpha=0.90)
    else:
        _render_voxel_points(ax_ours_full, pred_ours, color='#1f77b4', threshold=args.threshold)

    # English comment for public release.
    if gt_zoom_verts is not None:
        _render_mesh(ax_gt_zoom, gt_zoom_verts, gt_zoom_faces, color='#7f7f7f', alpha=0.90)
    else:
        _render_voxel_points(ax_gt_zoom, gt_vox, color='#7f7f7f', threshold=args.threshold)

    if unet_verts is not None:
        _render_mesh(ax_unet_zoom, unet_verts, unet_faces, color='#d95f02', alpha=0.90)
    else:
        _render_voxel_points(ax_unet_zoom, pred_unet, color='#d95f02', threshold=args.threshold)

    if ours_verts is not None:
        _render_mesh(ax_ours_zoom, ours_verts, ours_faces, color='#1f77b4', alpha=0.90)
    else:
        _render_voxel_points(ax_ours_zoom, pred_ours, color='#1f77b4', threshold=args.threshold)

    for ax in (ax_gt_full, ax_unet_full, ax_ours_full):
        _axes_unit(ax)
        ax.view_init(elev=22, azim=35)

    for ax in (ax_gt_zoom, ax_unet_zoom, ax_ours_zoom):
        _set_local_zoom(ax, zoom_center, radius=zoom_radius)
        ax.view_init(elev=22, azim=35)

    ax_gt_full.set_title('Ground Truth (Original Mesh)', fontsize=11, pad=8, fontweight='bold')
    ax_unet_full.set_title('3D U-Net', fontsize=11, pad=8, fontweight='bold')
    ax_ours_full.set_title('Physics-Aware 3D-VAE (Ours)', fontsize=11, pad=8, fontweight='bold')

    ax_gt_zoom.set_title('GT Zoomed Region', fontsize=10, pad=6, fontweight='bold')
    ax_unet_zoom.set_title('3D U-Net Zoomed Region', fontsize=10, pad=6, fontweight='bold')
    ax_ours_zoom.set_title('Ours Zoomed Region', fontsize=10, pad=6, fontweight='bold')

    fig.suptitle('Local Detail Comparison with Region Zoom (2×3 Layout)',
                 fontsize=13, y=0.98, fontweight='bold')
    fig.text(0.5, 0.93,
             f'Sample: {ore_label} | Top row: global view, Bottom row: same local region zoom',
             ha='center', va='center', fontsize=10, color='#333333')

    fig.canvas.draw()
    _add_projected_circle(ax_gt_full, zoom_center, radius_axes=0.085, color='red')
    _add_projected_circle(ax_unet_full, zoom_center, radius_axes=0.085, color='red')
    _add_projected_circle(ax_ours_full, zoom_center, radius_axes=0.085, color='red')

    plt.subplots_adjust(left=0.03, right=0.99, top=0.89, bottom=0.04, wspace=0.03, hspace=0.10)

    model_stem = os.path.splitext(args.model_name)[0]
    out_png = os.path.join(args.out_dir, f'local_detail_zoom_{model_stem}.png')
    out_pdf = os.path.join(args.out_dir, f'local_detail_zoom_{model_stem}.pdf')
    fig.savefig(out_png, dpi=args.dpi, bbox_inches='tight', facecolor='white', pad_inches=0.03)
    fig.savefig(out_pdf, format='pdf', bbox_inches='tight', facecolor='white', pad_inches=0.03)
    plt.close(fig)

    print('[translated_text] translated_text:')
    print(f'  - {out_png}')
    print(f'  - {out_pdf}')
    return out_png, out_pdf


def generate_comparison_figure(args):
    model_path = os.path.join(args.model_dir, args.model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'translated_text Ours translated_text: {model_path}')
    if not os.path.isdir(args.real_dir):
        raise FileNotFoundError(f'translated_text: {args.real_dir}')
    if not os.path.isdir(args.pretrain_dir):
        raise FileNotFoundError(f'translated_text: {args.pretrain_dir}')

    os.makedirs(args.out_dir, exist_ok=True)
    grid_size = (args.grid_size, args.grid_size, args.grid_size)

    real_pool = _resolve_selected_pool(args.real_dir, getattr(args, 'selected_real_files', None), 'Real')
    virtual_pool = _resolve_selected_pool(args.pretrain_dir, getattr(args, 'selected_virtual_files', None), 'Virtual')

    selected_samples = _pick_mixed_largest_ply(
        args.real_dir,
        args.pretrain_dir,
        n_rows=args.rows,
        real_rows=getattr(args, 'real_rows', 2),
        virtual_rows=getattr(args, 'virtual_rows', 2),
        random_samples=not bool(getattr(args, 'fixed_samples', False)),
        sample_seed=getattr(args, 'sample_seed', None),
        real_files=real_pool,
        virtual_files=virtual_pool,
    )
    if not selected_samples:
        raise RuntimeError(f'translated_text PLY: real={args.real_dir}, virtual={args.pretrain_dir}')

    n_real = sum(1 for src, _ in selected_samples if src == 'real')
    n_virtual = len(selected_samples) - n_real
    mode_text = 'random' if not bool(getattr(args, 'fixed_samples', False)) else 'fixed-largest'
    print(f"[translated_text] translated_text({mode_text}): real={n_real}, virtual={n_virtual}, total={len(selected_samples)}")
    for src, p in selected_samples:
        tag = 'Real' if src == 'real' else 'Virtual'
        print(f"  - [{tag}] {os.path.basename(p)} ({os.path.getsize(p)/1024:.1f} KB)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ours_model, effective_grid = _load_ours_model(model_path, device, grid_size)
    if tuple(effective_grid) != tuple(grid_size):
        print(f"[Ours] translated_text: {effective_grid}(translated_text --grid-size={args.grid_size})")
    grid_size = tuple(effective_grid)
    unet_model = _load_or_train_unet(
        args.unet_ckpt,
        train_data_dir=args.pretrain_dir if os.path.isdir(args.pretrain_dir) else args.real_dir,
        unet_epochs=args.unet_epochs,
        unet_batch_size=args.unet_batch_size,
        unet_lr=args.unet_lr,
    )

    rows_data = []
    real_rank = 0
    virtual_rank = 0
    for row_idx, (source_type, file_path) in enumerate(selected_samples, start=1):
        source_name = os.path.splitext(os.path.basename(file_path))[0]
        if source_type == 'real':
            real_rank += 1
            display_name = f"Real Ore Body {real_rank}"
        else:
            virtual_rank += 1
            display_name = f"Virtual Ore Body {virtual_rank}"
        print(f"[translated_text] {display_name} ...")
        verts, faces = read_ply(file_path)
        verts_norm = _normalize_verts(verts)

        gt_vox, vmin, vmax = _voxelize_mesh(verts, faces, grid_size=grid_size)
        obs_np, holes_norm = _build_sparse_input(
            gt_vox, vmin, vmax,
            num_holes=args.num_holes,
            samples_per_hole=args.samples_per_hole,
            seed=_stable_seed(source_name, args.seed),
        )

        pred_ik = _predict_ik(obs_np, grid_size)
        pred_unet = _predict_unet(unet_model, obs_np)
        pred_ours = _predict_ours(ours_model, obs_np)

        rows_data.append({
            'name': display_name,
            'source': source_type,
            'source_name': source_name,
            'verts_norm': verts_norm,
            'faces': faces,
            'gt_vox': gt_vox,
            'holes_norm': holes_norm,
            'ik': pred_ik,
            'unet': pred_unet,
            'ours': pred_ours,
        })

    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#17becf', '#d62728']
    col_titles = ['Ground Truth', 'Sparse Input', 'IK', '3D U-Net', 'Ours']
    n_rows = len(rows_data)
    n_cols = len(col_titles)

    fig = plt.figure(figsize=(n_cols * 3.8, max(4.0, n_rows * 2.9 + 1.0)), dpi=args.dpi)
    fig.patch.set_facecolor('white')

    legend_patches = []

    for r, row in enumerate(rows_data):
        color = palette[r % len(palette)]
        legend_patches.append(Patch(facecolor=color, edgecolor='none', label=row['name']))

        for c in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1, projection='3d')
            ax.set_facecolor('white')

            if c == 0:
                _render_mesh(ax, row['verts_norm'], row['faces'], color=color, alpha=0.88)
            elif c == 1:
                _render_sparse(ax, row['holes_norm'], row.get('gt_vox'), ore_color=color)
            elif c == 2:
                ik_verts, ik_faces = _vox_to_mesh_norm(row['ik'], threshold=args.threshold)
                if ik_verts is not None:
                    _render_mesh(ax, ik_verts, ik_faces, color=color, alpha=0.82)
                else:
                    _render_voxel_points(ax, row['ik'], color=color, threshold=args.threshold)
            elif c == 3:
                if row['unet'] is None:
                    _axes_unit(ax)
                    ax.text(0.5, 0.5, 0.5, 'N/A', ha='center', va='center', color='gray')
                else:
                    unet_verts, unet_faces = _vox_to_mesh_norm(row['unet'], threshold=args.threshold)
                    if unet_verts is not None:
                        _render_mesh(ax, unet_verts, unet_faces, color=color, alpha=0.82)
                    else:
                        _render_voxel_points(ax, row['unet'], color=color, threshold=args.threshold)
            else:
                ours_verts, ours_faces = _vox_to_mesh_norm(row['ours'], threshold=args.threshold)
                if ours_verts is not None:
                    _render_mesh(ax, ours_verts, ours_faces, color=color, alpha=0.82)
                else:
                    _render_voxel_points(ax, row['ours'], color=color, threshold=args.threshold)

            if r == 0:
                ax.set_title(col_titles[c], fontsize=11, pad=8, fontweight='bold')

            if c == 0:
                ax.text2D(-0.18, 0.5, row['name'], transform=ax.transAxes,
                          fontsize=10, va='center', ha='right', color=color, fontweight='bold')

    fig.suptitle('Global 3D Reconstruction Comparison Matrix', fontsize=13, y=0.995, fontweight='bold')
    fig.legend(handles=legend_patches, loc='lower center', ncol=min(6, len(legend_patches)),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.01),
               title='Ore Body ID (row-wise color coding)')

    plt.subplots_adjust(left=0.08, right=0.995, top=0.92, bottom=0.10, wspace=0.02, hspace=0.08)

    model_stem = os.path.splitext(args.model_name)[0]
    out_png = os.path.join(args.out_dir, f'comparison_matrix_{model_stem}.png')
    out_pdf = os.path.join(args.out_dir, f'comparison_matrix_{model_stem}.pdf')

    fig.savefig(out_png, dpi=args.dpi, bbox_inches='tight', facecolor='white', pad_inches=0.05)
    fig.savefig(out_pdf, format='pdf', bbox_inches='tight', facecolor='white', pad_inches=0.05)
    plt.close(fig)

    print('[translated_text] translated_text:')
    print(f'  - {out_png}')
    print(f'  - {out_pdf}')
    return out_png, out_pdf


def main():
    args = parse_args()
    generate_comparison_figure(args)


if __name__ == '__main__':
    main()

