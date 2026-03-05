"""Documentation translated to English for open-source release."""

import os
import sys
import argparse
import datetime
import hashlib
import warnings
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from scipy.spatial.distance import directed_hausdorff

try:
    from sklearn.exceptions import ConvergenceWarning as _SklearnConvergenceWarning
except Exception:
    _SklearnConvergenceWarning = Warning

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from dataset import MiningDataset
from comparisons.models.interpolation import KrigingModel, RBFModel
from comparisons.models.unet import UNet3D
from comparisons.train_unet import train_unet
from plot_comparison import _load_ours_model, _safe_torch_load


def calculate_chamfer(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    pred_pts = np.argwhere(pred > threshold)
    target_pts = np.argwhere(target > 0.5)

    if len(pred_pts) == 0 or len(target_pts) == 0:
        return 999.0

    try:
        from scipy.spatial import KDTree
        tree_pred = KDTree(pred_pts)
        tree_target = KDTree(target_pts)
        d1, _ = tree_target.query(pred_pts, k=1)
        d2, _ = tree_pred.query(target_pts, k=1)
        return float(np.mean(d1 ** 2) + np.mean(d2 ** 2))
    except Exception:
        return 999.0


def calculate_hausdorff(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    pred_pts = np.argwhere(pred > threshold)
    target_pts = np.argwhere(target > 0.5)

    if len(pred_pts) == 0 or len(target_pts) == 0:
        return 999.0

    d1 = directed_hausdorff(pred_pts, target_pts)[0]
    d2 = directed_hausdorff(target_pts, pred_pts)[0]
    return float(max(d1, d2))


def calculate_metrics(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> Tuple[float, float, float, float]:
    pred_bin = (pred > threshold).astype(np.float32)
    target_bin = (target > 0.5).astype(np.float32)

    intersection = np.sum(pred_bin * target_bin)
    union = np.sum(pred_bin) + np.sum(target_bin) - intersection

    dice = (2.0 * intersection) / (np.sum(pred_bin) + np.sum(target_bin) + 1e-8)
    iou = intersection / (union + 1e-8)

    try:
        hd = calculate_hausdorff(pred, target, threshold)
        cd = calculate_chamfer(pred, target, threshold)
    except Exception:
        hd, cd = 999.0, 999.0

    return float(iou), float(dice), float(hd), float(cd)


def _find_latest_best_ckpt(results_3dvae_dir: str, exclude: Optional[str] = None) -> Optional[str]:
    if not os.path.isdir(results_3dvae_dir):
        return None

    candidates = []
    for run_name in os.listdir(results_3dvae_dir):
        p = os.path.join(results_3dvae_dir, run_name, 'checkpoints', 'best_model.pth')
        if os.path.isfile(p):
            if exclude and os.path.abspath(p) == os.path.abspath(exclude):
                continue
            candidates.append(p)

    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _auto_find_baseline_ckpt(repo_root: str, ours_ckpt: str) -> Optional[str]:
    ours_hash = _file_sha256(ours_ckpt) if os.path.isfile(ours_ckpt) else None

    probe_paths = [
        os.path.join(repo_root, 'results', 'ablations', 'No_Octree', 'checkpoints', 'best_model.pth'),
        os.path.join(repo_root, 'results', 'auto_analysis', 'Benchmarks', 'Standard_VAE', 'checkpoints', 'best_model.pth'),
    ]
    for p in probe_paths:
        if os.path.isfile(p):
            if ours_hash and _file_sha256(p) == ours_hash:
                continue
            return p

    results_3dvae_dir = os.path.join(repo_root, 'results', '3dvae')
    if not os.path.isdir(results_3dvae_dir):
        return None

    candidates = []
    for run_name in os.listdir(results_3dvae_dir):
        p = os.path.join(results_3dvae_dir, run_name, 'checkpoints', 'best_model.pth')
        if os.path.isfile(p) and os.path.abspath(p) != os.path.abspath(ours_ckpt):
            candidates.append(p)
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    for p in candidates:
        if ours_hash and _file_sha256(p) == ours_hash:
            continue
        return p

    return None


def _load_or_train_unet(unet_ckpt: str, train_data_dir: str,
                        unet_epochs: int, unet_batch_size: int, unet_lr: float,
                        device: torch.device) -> Optional[UNet3D]:
    ckpt_path = os.path.abspath(unet_ckpt)

    if not os.path.exists(ckpt_path):
        print(f"[U-Net] 未找到权重: {ckpt_path}")
        print(f"[U-Net] 自动训练基础模型（epochs={unet_epochs}）...")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        from types import SimpleNamespace
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
        print('[U-Net] 仍未获得可用权重，跳过 U-Net 评估。')
        return None

    model = UNet3D(in_channels=2, out_channels=1).to(device)
    state = _safe_torch_load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
        model.load_state_dict(state['state_dict'], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[U-Net] 已加载: {ckpt_path}")
    return model


def _predict_vae(model, obs: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        out = model(obs)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        if logits.ndim == 5 and logits.shape[1] == 1:
            logits = logits[:, 0]
        prob = torch.sigmoid(logits).squeeze().detach().cpu().numpy().astype(np.float32)
    return np.clip(prob, 0.0, 1.0)


def _predict_unet(model: Optional[UNet3D], obs: torch.Tensor) -> Optional[np.ndarray]:
    if model is None:
        return None
    with torch.no_grad():
        pred = model(obs).squeeze().detach().cpu().numpy().astype(np.float32)
    return np.clip(pred, 0.0, 1.0)


def _predict_ik(obs_np: np.ndarray, grid_size: Tuple[int, int, int]) -> np.ndarray:
    try:
        m = KrigingModel(grid_size=grid_size)
        pred = m.fit_predict(obs_np)
        if pred.ndim == 4:
            pred = pred[0]
        return np.clip(pred.astype(np.float32), 0.0, 1.0)
    except Exception as e:
        print(f"[IK] 预测失败，回退全零: {e}")
        return np.zeros(grid_size, dtype=np.float32)


def _predict_rbf(obs_np: np.ndarray, grid_size: Tuple[int, int, int], kernel: str = 'linear') -> np.ndarray:
    try:
        m = RBFModel(grid_size=grid_size, kernel=kernel)
        pred = m.fit_predict(obs_np)
        if pred.ndim == 4:
            pred = pred[0]
        return np.clip(pred.astype(np.float32), 0.0, 1.0)
    except Exception as e:
        print(f"[RBF] 预测失败，回退全零: {e}")
        return np.zeros(grid_size, dtype=np.float32)


def parse_args():
    repo_root = os.path.dirname(_SCRIPT_DIR)
    p = argparse.ArgumentParser(
        description='五方法统一定量评估（IK/RBF/U-Net/基础VAE/Physics-Aware VAE）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument('--real-dir', type=str, default=os.path.join(repo_root, 'data', 'mining_ply'))
    p.add_argument('--pretrain-dir', type=str, default=os.path.join(repo_root, 'data', 'mining_ply_pretrain'))

    p.add_argument('--model-dir', type=str, default=os.path.join(repo_root, 'results', 'model'))
    p.add_argument('--model-name', type=str, default='best_model.pth', help='Physics-Aware 3D-VAE 权重名（与 plot_dataset_overview 一致）')

    p.add_argument('--baseline-ckpt', type=str, default=None, help='基础 3D-VAE 权重路径；为空时自动探测')
    p.add_argument('--unet-ckpt', type=str, default=os.path.join(repo_root, 'results', 'comparisons', 'models', 'unet_best.pth'))

    p.add_argument('--grid-size', type=int, default=32)
    p.add_argument('--num-holes', type=int, default=8)
    p.add_argument('--samples-per-hole', type=int, default=16)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit', type=int, default=0, help='评估样本上限；0 表示全部')

    p.add_argument('--unet-epochs', type=int, default=10)
    p.add_argument('--unet-batch-size', type=int, default=16)
    p.add_argument('--unet-lr', type=float, default=1e-3)
    p.add_argument('--show-gp-warnings', action='store_true',
                   help='显示 IK(高斯过程) 收敛警告；默认隐藏以避免大量刷屏')

    p.add_argument('--out-dir', type=str, default=os.path.join(repo_root, 'results', 'figures'))
    return p.parse_args()


def run_evaluation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    repo_root = os.path.dirname(_SCRIPT_DIR)

    ours_ckpt = os.path.join(args.model_dir, args.model_name)
    if not os.path.isfile(ours_ckpt):
        raise FileNotFoundError(f'未找到 Physics-Aware 3D-VAE 权重: {ours_ckpt}')

    baseline_ckpt = args.baseline_ckpt
    if baseline_ckpt is None:
        baseline_ckpt = _auto_find_baseline_ckpt(repo_root, ours_ckpt)
    if baseline_ckpt is None or not os.path.isfile(baseline_ckpt):
        raise FileNotFoundError('未找到基础 3D-VAE 权重。请通过 --baseline-ckpt 显式指定。')

    print(f"[模型] Physics-Aware 3D-VAE: {ours_ckpt}")
    print(f"[模型] 基础 3D-VAE:         {baseline_ckpt}")

    show_gp_warnings = bool(getattr(args, 'show_gp_warnings', False))
    os.environ['SHOW_GP_WARNINGS'] = '1' if show_gp_warnings else '0'

    if not show_gp_warnings:
        warnings.filterwarnings(
            'ignore',
            category=_SklearnConvergenceWarning,
        )

    input_grid = (args.grid_size, args.grid_size, args.grid_size)
    ours_model, ours_grid = _load_ours_model(ours_ckpt, device, input_grid)
    baseline_model, baseline_grid = _load_ours_model(baseline_ckpt, device, input_grid)

    if tuple(ours_grid) != tuple(baseline_grid):
        print(f"[警告] Ours 网格 {ours_grid} 与 Baseline 网格 {baseline_grid} 不一致，评估使用 Ours 网格。")

    eval_grid = tuple(ours_grid)
    print(f"[评估] 使用网格: {eval_grid}")

    train_data_dir = args.pretrain_dir if os.path.isdir(args.pretrain_dir) else args.real_dir
    unet_model = _load_or_train_unet(
        args.unet_ckpt, train_data_dir,
        args.unet_epochs, args.unet_batch_size, args.unet_lr,
        device,
    )

    if not os.path.isdir(args.real_dir):
        raise FileNotFoundError(f'真实矿体目录不存在: {args.real_dir}')
    if not os.path.isdir(args.pretrain_dir):
        raise FileNotFoundError(f'虚拟矿体目录不存在: {args.pretrain_dir}')

    def _build_dataset(ply_dir: str, split_seed: int) -> MiningDataset:
        ds = MiningDataset(
            ply_dir=ply_dir,
            num_holes=args.num_holes,
            samples_per_hole=args.samples_per_hole,
            grid_size=eval_grid,
            augment_per_mesh=0,
            train_frac=0.9,
            min_samples_per_file=1,
            max_samples_per_file=1,
            split_seed=split_seed,
            force_regen_cache=False,
            load_mode='sequential',
            log_mode='brief',
        )
        ds.set_split('all')
        return ds

    dataset_real = _build_dataset(args.real_dir, args.seed)
    dataset_virtual = _build_dataset(args.pretrain_dir, args.seed + 131)

    total_real = len(dataset_real)
    total_virtual = len(dataset_virtual)
    total_all = total_real + total_virtual

    eval_indices: List[Tuple[str, int]] = (
        [('real', i) for i in range(total_real)] +
        [('virtual', i) for i in range(total_virtual)]
    )

    if args.limit and args.limit > 0:
        eval_indices = eval_indices[:args.limit]

    eval_n = len(eval_indices)
    eval_real = sum(1 for src, _ in eval_indices if src == 'real')
    eval_virtual = eval_n - eval_real

    print(f"[数据] 真实矿体样本数: {total_real}")
    print(f"[数据] 虚拟矿体样本数: {total_virtual}")
    print(f"[数据] 合计样本数: {total_all}")
    if args.limit and args.limit > 0:
        print(f"[数据] 按 --limit={args.limit} 截断后评估样本数: {eval_n} (real={eval_real}, virtual={eval_virtual})")
    else:
        print(f"[数据] 全量评估样本数: {eval_n} (real={eval_real}, virtual={eval_virtual})")

    method_order = [
        ('传统方法', '指示克里金', 'IK'),
        ('传统方法', '径向基函数', 'RBF'),
        ('深度学习', '3D U-Net', 'UNET'),
        ('深度学习', '基础 3D-VAE', 'BASE_VAE'),
        ('本文方法', 'Physics-Aware 3D-VAE', 'OURS_VAE'),
    ]

    metrics_map: Dict[str, Dict[str, List[float]]] = {
        key: {'iou': [], 'dice': [], 'hd': [], 'cd': []}
        for _, _, key in method_order
    }

    for src, idx in tqdm(eval_indices, desc='Quantitative Eval'):
        if src == 'real':
            obs, target, _, _, _ = dataset_real[idx]
        else:
            obs, target, _, _, _ = dataset_virtual[idx]
        obs = obs.unsqueeze(0).to(device)
        obs_np = obs.squeeze(0).detach().cpu().numpy().astype(np.float32)
        target_np = target.numpy().astype(np.float32)

        pred_ik = _predict_ik(obs_np, eval_grid)
        iou, dice, hd, cd = calculate_metrics(pred_ik, target_np, threshold=args.threshold)
        metrics_map['IK']['iou'].append(iou)
        metrics_map['IK']['dice'].append(dice)
        metrics_map['IK']['hd'].append(hd)
        metrics_map['IK']['cd'].append(cd)

        pred_rbf = _predict_rbf(obs_np, eval_grid, kernel='linear')
        iou, dice, hd, cd = calculate_metrics(pred_rbf, target_np, threshold=args.threshold)
        metrics_map['RBF']['iou'].append(iou)
        metrics_map['RBF']['dice'].append(dice)
        metrics_map['RBF']['hd'].append(hd)
        metrics_map['RBF']['cd'].append(cd)

        pred_unet = _predict_unet(unet_model, obs)
        if pred_unet is not None:
            iou, dice, hd, cd = calculate_metrics(pred_unet, target_np, threshold=args.threshold)
            metrics_map['UNET']['iou'].append(iou)
            metrics_map['UNET']['dice'].append(dice)
            metrics_map['UNET']['hd'].append(hd)
            metrics_map['UNET']['cd'].append(cd)

        pred_base = _predict_vae(baseline_model, obs)
        iou, dice, hd, cd = calculate_metrics(pred_base, target_np, threshold=args.threshold)
        metrics_map['BASE_VAE']['iou'].append(iou)
        metrics_map['BASE_VAE']['dice'].append(dice)
        metrics_map['BASE_VAE']['hd'].append(hd)
        metrics_map['BASE_VAE']['cd'].append(cd)

        pred_ours = _predict_vae(ours_model, obs)
        iou, dice, hd, cd = calculate_metrics(pred_ours, target_np, threshold=args.threshold)
        metrics_map['OURS_VAE']['iou'].append(iou)
        metrics_map['OURS_VAE']['dice'].append(dice)
        metrics_map['OURS_VAE']['hd'].append(hd)
        metrics_map['OURS_VAE']['cd'].append(cd)

    rows = []
    for category, method_name, key in method_order:
        vals = metrics_map[key]
        if len(vals['iou']) == 0:
            continue
        rows.append({
            'Category': category,
            'Methods': method_name,
            'IoU': float(np.mean(vals['iou'])),
            'Dice': float(np.mean(vals['dice'])),
            'HD': float(np.mean(vals['hd'])),
            'CD': float(np.mean(vals['cd'])),
            'Samples': int(len(vals['iou'])),
        })

    df = pd.DataFrame(rows)

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(args.out_dir, f'quantitative_eval_{ts}.csv')
    md_path = os.path.join(args.out_dir, f'quantitative_eval_{ts}.md')

    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('| 方法类别(Category) | 模型名称(Methods) | IoU(↑) | Dice(↑) | HD(↓) | CD(↓) |\n')
        f.write('|---|---:|---:|---:|---:|---:|\n')
        for _, r in df.iterrows():
            f.write(
                f"| {r['Category']} | {r['Methods']} | {r['IoU']:.4f} | {r['Dice']:.4f} | {r['HD']:.3f} | {r['CD']:.3f} |\n"
            )

    print('\n================ 定量评估结果 ================')
    for _, r in df.iterrows():
        print(f"{r['Category']:<8} | {r['Methods']:<24} | IoU={r['IoU']:.4f} | Dice={r['Dice']:.4f} | HD={r['HD']:.3f} | CD={r['CD']:.3f}")
    print('==============================================')
    print(f'[保存] CSV: {csv_path}')
    print(f'[保存] MD : {md_path}')

    return csv_path, md_path


def main():
    args = parse_args()
    run_evaluation(args)


if __name__ == '__main__':
    main()
