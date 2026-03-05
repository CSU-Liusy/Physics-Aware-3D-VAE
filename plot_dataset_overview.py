"""Documentation translated to English for open-source release."""

import os
import sys
import argparse
import random
from types import SimpleNamespace
from typing import Optional, List, Dict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# English comment for public release.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from dataset import read_ply  # English comment for public release.

try:
    from plot_comparison import generate_comparison_figure as _generate_comparison_matrix
except Exception:
    _generate_comparison_matrix = None

try:
    from plot_comparison import generate_local_detail_figure as _generate_local_detail_figure
except Exception:
    _generate_local_detail_figure = None

try:
    from plot_comparison import generate_uncertainty_variance_figure as _generate_uncertainty_variance_figure
except Exception:
    _generate_uncertainty_variance_figure = None

try:
    from quantitative_evaluation import run_evaluation as _run_quantitative_eval
except Exception:
    _run_quantitative_eval = None

# English comment for public release.
try:
    from skimage.measure import marching_cubes
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

# English comment for public release.
import matplotlib.font_manager as _fm


def _get_cjk_font_prop():
    """Documentation translated to English for open-source release."""
    from matplotlib.font_manager import FontProperties
    candidates = [
        'Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi',
        'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Arial Unicode MS',
    ]
    available = {f.name for f in _fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return FontProperties(family=name)
    return None


_CJK_FP = _get_cjk_font_prop()


# =============================================================================
# English comment for public release.
# =============================================================================

def _axes_equal(ax, verts: np.ndarray):
    """Documentation translated to English for open-source release."""
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    center = (lo + hi) / 2.0
    span = (hi - lo).max() / 2.0
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)


def _normalize_verts(verts: np.ndarray) -> np.ndarray:
    """Documentation translated to English for open-source release."""
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    span = hi - lo
    span[span < 1e-8] = 1.0  # English comment for public release.
    return (verts - lo) / span


def _scan_ply_sizes(ply_files: list[str], max_scan: int = 500, seed: int = 42) -> list[tuple[str, int]]:
    """Documentation translated to English for open-source release."""
    rng = random.Random(seed)
    sample = ply_files if len(ply_files) <= max_scan else rng.sample(ply_files, max_scan)
    sizes = []
    for p in sample:
        try:
            v, _ = read_ply(p)
            sizes.append((p, len(v)))
        except Exception:
            continue
    sizes.sort(key=lambda x: x[1])
    return sizes


def _select_diverse(ply_files: list[str], n: int, seed: int = 42) -> list[str]:
    """Documentation translated to English for open-source release."""
    if len(ply_files) <= n:
        return ply_files

    sizes = _scan_ply_sizes(ply_files, max_scan=500, seed=seed)
    if not sizes:
        rng = random.Random(seed)
        return rng.sample(ply_files, n)

    # English comment for public release.
    idxs = np.linspace(0, len(sizes) - 1, n, dtype=int)
    return [sizes[i][0] for i in idxs]


def _select_real_diverse(ply_files: list[str], n: int, seed: int = 42,
                          size_thresh: float = 0.8) -> list[str]:
    """Documentation translated to English for open-source release."""
    if len(ply_files) <= n:
        return ply_files

    # English comment for public release.
    sized = sorted(ply_files, key=lambda p: os.path.getsize(p))
    N = len(sized)

    size_thresh = float(np.clip(size_thresh, 0.0, 1.0))
    center_idx = size_thresh * (N - 1)

    # English comment for public release.
    half_w = max(int(np.ceil(3 * n / 2)), int(N * 0.2))
    lo = int(max(0,     center_idx - half_w))
    hi = int(min(N - 1, center_idx + half_w))

    pool = sized[lo: hi + 1]
    if len(pool) < n:
        pool = sized  # English comment for public release.

    sz_lo = os.path.getsize(pool[0])
    sz_hi = os.path.getsize(pool[-1])
    print(f"  [选样] 文件大小窗口：{sz_lo/1024:.1f} KB ~ {sz_hi/1024:.1f} KB"
          f"，共 {len(pool)} 个候选（size_thresh={size_thresh:.2f}）")

    idxs = np.linspace(0, len(pool) - 1, n, dtype=int)
    return [pool[i] for i in idxs]


def _simulate_drillholes(verts: np.ndarray,
                          num_holes: int = 8,
                          samples_per_hole: int = 12,
                          seed: int = 0) -> list[np.ndarray]:
    """Documentation translated to English for open-source release."""
    rng = np.random.RandomState(seed)
    holes = []
    for _ in range(num_holes):
        cx = rng.uniform(0.05, 0.95)
        cy = rng.uniform(0.05, 0.95)
        zs = np.linspace(0.02, 0.98, samples_per_hole)
        xs = cx + rng.uniform(-0.01, 0.01, samples_per_hole)
        ys = cy + rng.uniform(-0.01, 0.01, samples_per_hole)
        pts = np.stack([xs, ys, zs], axis=1).astype(np.float32)
        holes.append(pts)
    return holes


def _render_virtual_mesh(ax, verts: np.ndarray, faces: np.ndarray,
                          face_color: str = '#3a7abf',
                          title: str = '') -> None:
    """Documentation translated to English for open-source release."""
    if len(faces) == 0:
        return
    tri_verts = verts[faces]
    mesh = Poly3DCollection(
        tri_verts, alpha=0.92,
        facecolor=face_color, edgecolor='#1a3a5c',
        linewidths=0.15
    )
    ax.add_collection3d(mesh)
    _axes_equal(ax, verts)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=7, pad=2, color='#333333')


def _render_real_mesh_with_drillholes(ax, verts: np.ndarray, faces: np.ndarray,
                                       holes: list[np.ndarray],
                                       mesh_color: str = '#6aaa6a',
                                       title: str = '') -> None:
    """Documentation translated to English for open-source release."""
    if len(faces) > 0:
        tri_verts = verts[faces]
        mesh = Poly3DCollection(
            tri_verts, alpha=0.25,
            facecolor=mesh_color, edgecolor='none'
        )
        ax.add_collection3d(mesh)

    # English comment for public release.
    for hole in holes:
        ax.plot(hole[:, 0], hole[:, 1], hole[:, 2],
                color='#d62728', linewidth=1.2, alpha=0.85)
        ax.scatter(hole[:, 0], hole[:, 1], hole[:, 2],
                   c='#d62728', s=4, alpha=0.7, zorder=5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=7, pad=2, color='#333333')


# =============================================================================
# English comment for public release.
# =============================================================================

def parse_args():
    _repo_root = os.path.dirname(_SCRIPT_DIR)
    default_model_dir = os.path.join(_repo_root, 'results', 'model')
    default_virtual_dir = os.path.join(_repo_root, 'data', 'mining_ply_pretrain')
    default_real_dir = os.path.join(_repo_root, 'data', 'mining_ply')
    default_out_dir = os.path.join(_repo_root, 'results', 'figures')
    default_unet_ckpt = os.path.join(_repo_root, 'results', 'comparisons', 'models', 'unet_best.pth')

    p = argparse.ArgumentParser(
        description='生成数据集样例与分布展示图（虚拟矿体 + 真实钻孔轨迹）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--model-dir', type=str, default=default_model_dir,
                   help='模型权重目录')
    p.add_argument('--model-name', type=str, default='best_model.pth',
                   help='模型权重文件名（仅文件名，如 best_model.pth）；'
                        '若不存在会报错并列出目录中的 .pth 文件')
    p.add_argument('--virtual-dir', type=str, default=default_virtual_dir,
                   help='虚拟矿体 PLY 目录')
    p.add_argument('--real-dir', type=str, default=default_real_dir,
                   help='真实矿体 PLY 目录')
    p.add_argument('--out-dir', type=str, default=default_out_dir,
                   help='图像输出目录')
    p.add_argument('--n-virtual', type=int, default=9,
                   help='左侧展示的虚拟矿体数量（建议为完全平方数：4/9）')
    p.add_argument('--n-real', type=int, default=9,
                   help='右侧展示的真实矿体数量')
    p.add_argument('--num-holes', type=int, default=8,
                   help='每个真实矿体模拟的钻孔数')
    p.add_argument('--samples-per-hole', type=int, default=12,
                   help='每个钻孔的竖向采样点数')
    p.add_argument('--dpi', type=int, default=300,
                   help='输出图像 DPI')
    p.add_argument('--seed', type=int, default=42,
                   help='随机种子（控制样本选取和钻孔位置）')
    p.add_argument('--size-thresh', type=float, default=0.8,
                   help='真实矿体选样的文件大小分位数阈值 [0.0~1.0]：'
                        '越大越偏向大文件（复杂矿体），越小越偏向小文件（默认 0.8）')
    p.add_argument('--skip-comparison', action='store_true',
                   help='仅生成数据集概览图，不联动生成全局重建对比矩阵图')
    p.add_argument('--comparison-rows', type=int, default=4,
                   help='联动生成对比矩阵时的样本总行数（可由真实/虚拟行数配比控制）')
    p.add_argument('--comparison-real-rows', type=int, default=2,
                   help='联动生成对比矩阵时，优先展示的真实矿体行数')
    p.add_argument('--comparison-virtual-rows', type=int, default=2,
                   help='联动生成对比矩阵时，优先展示的虚拟矿体行数')
    p.add_argument('--comparison-fixed-samples', action='store_true',
                   help='联动生成对比矩阵时关闭随机选样，改为按文件大小固定选样')
    p.add_argument('--comparison-sample-seed', type=int, default=None,
                   help='联动生成对比矩阵的随机选样种子；不指定则每次运行随机变化')
    p.add_argument('--selected-virtual-files', type=str, nargs='*', default=None,
                   help='指定用于展示的虚拟矿体文件名（支持逗号分隔或多值输入）')
    p.add_argument('--selected-real-files', type=str, nargs='*', default=None,
                   help='指定用于展示的真实矿体文件名（支持逗号分隔或多值输入）')
    p.add_argument('--unet-ckpt', type=str, default=default_unet_ckpt,
                   help='3D U-Net 权重路径；若不存在将自动训练基础模型')
    p.add_argument('--comparison-unet-epochs', type=int, default=10,
                   help='联动生成对比矩阵时，若缺少 U-Net 权重则自动训练轮次')
    p.add_argument('--skip-local-zoom', action='store_true',
                   help='跳过局部细节与伪影抑制放大对比图生成')
    p.add_argument('--zoom-radius', type=float, default=0.18,
                   help='局部放大的归一化坐标半径')
    p.add_argument('--zoom-sample-rank', type=int, default=1,
                   help='局部放大样本在大文件排序中的序号（1 表示最大文件）')
    p.add_argument('--uncertainty-mc-samples', type=int, default=24,
                   help='认知不确定性方差估计的蒙特卡洛前向次数')
    p.add_argument('--skip-quant-eval', action='store_true',
                   help='跳过定量评估（默认会在图像生成后自动计算指标）')
    p.add_argument('--quant-limit', type=int, default=0,
                   help='定量评估样本上限；0 表示使用全部样本')
    p.add_argument('--baseline-vae-ckpt', type=str, default=None,
                   help='基础 3D-VAE 权重路径；不指定则自动探测')
    p.add_argument('--quant-out-dir', type=str, default=os.path.join(_repo_root, 'results', 'figures'),
                   help='定量评估结果输出目录（CSV/Markdown）')
    return p.parse_args()


def _check_model(model_dir: str, model_name: str) -> str:
    """Documentation translated to English for open-source release."""
    model_path = os.path.join(model_dir, model_name)
    if os.path.isfile(model_path):
        print(f"[模型] 找到权重文件: {model_path}")
        return model_path

    # English comment for public release.
    print(f"\n[错误] 未找到模型权重文件: {model_path}")
    if os.path.isdir(model_dir):
        available = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        if available:
            print(f"  目录 '{model_dir}' 中现有以下模型文件:")
            for f in sorted(available):
                print(f"    - {f}")
            print(f"  请通过 --model-name 指定其中一个，例如:")
            print(f"    python 3dvae/plot_dataset_overview.py --model-name {available[0]}")
        else:
            print(f"  目录 '{model_dir}' 中没有任何 .pth 文件。")
    else:
        print(f"  模型目录 '{model_dir}' 不存在，请通过 --model-dir 指定正确路径。")
    sys.exit(1)


def _collect_ply(directory: str, label: str) -> list[str]:
    """Documentation translated to English for open-source release."""
    if not os.path.isdir(directory):
        print(f"[错误] {label} 目录不存在: {directory}")
        sys.exit(1)
    files = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith('.ply')
    ])
    if not files:
        print(f"[错误] {label} 目录中没有 .ply 文件: {directory}")
        sys.exit(1)
    print(f"[数据] {label}: 共 {len(files)} 个 PLY 文件")
    return files


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


def _resolve_selected_candidates(all_files: List[str],
                                 selected_items: Optional[List[str]],
                                 label: str) -> List[str]:
    requested = _expand_selected_items(selected_items)
    if not requested:
        return all_files

    by_name: Dict[str, List[str]] = {}
    for p in all_files:
        by_name.setdefault(os.path.basename(p), []).append(p)

    resolved: List[str] = []
    missing: List[str] = []

    for item in requested:
        chosen = None
        if os.path.isabs(item) and os.path.isfile(item):
            chosen = os.path.abspath(item)
        else:
            matches = by_name.get(os.path.basename(item), [])
            if len(matches) > 1:
                print(f"  [警告] {label} 文件名 '{item}' 匹配到多个文件，默认取第一个。")
            if matches:
                chosen = matches[0]

        if chosen is None:
            missing.append(item)
            continue
        if chosen not in resolved:
            resolved.append(chosen)

    if missing:
        print(f"  [警告] {label} 指定文件未找到，已忽略: {missing}")
    if not resolved:
        print(f"[错误] {label} 指定文件均无效，请检查参数。")
        sys.exit(1)

    print(f"[样本池] {label}: 使用指定文件 {len(resolved)} 个")
    return resolved


def generate_figure(args) -> None:
    # English comment for public release.
    _check_model(args.model_dir, args.model_name)
    # English comment for public release.
    # English comment for public release.

    # English comment for public release.
    virtual_files = _collect_ply(args.virtual_dir, '虚拟矿体')
    real_files = _collect_ply(args.real_dir, '真实矿体')

    virtual_candidates = _resolve_selected_candidates(
        virtual_files, args.selected_virtual_files, '虚拟矿体'
    )
    real_candidates = _resolve_selected_candidates(
        real_files, args.selected_real_files, '真实矿体'
    )

    # English comment for public release.
    print(f"[选样] 从虚拟矿体中按形态多样性选取 {args.n_virtual} 个...")
    selected_virtual = _select_diverse(virtual_candidates, args.n_virtual, seed=args.seed)

    print(f"[选样] 从真实矿体中按立体性与表面积选取 {args.n_real} 个...")
    selected_real = _select_real_diverse(real_candidates, args.n_real, seed=args.seed,
                                          size_thresh=args.size_thresh)

    # English comment for public release.
    # English comment for public release.
    # English comment for public release.
    # English comment for public release.
    nv = args.n_virtual
    nr = args.n_real

    import math
    n_cols_v = math.ceil(math.sqrt(nv))
    n_rows_v = math.ceil(nv / n_cols_v)
    n_cols_r = math.ceil(math.sqrt(nr))
    n_rows_r = math.ceil(nr / n_cols_r)

    n_rows = max(n_rows_v, n_rows_r)

    fig_w = 18.0
    fig_h = max(4.0, n_rows * 3.2 + 1.2)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=args.dpi)
    fig.patch.set_facecolor('white')

    # English comment for public release.
    left_width = 0.58
    right_start = 0.62
    right_width = 0.36
    top_margin = 0.92
    bottom_margin = 0.10

    cell_h = (top_margin - bottom_margin) / n_rows

    # English comment for public release.
    colors_virtual = [
        '#3a7abf', '#e07b39', '#6aaa6a', '#9467bd',
        '#8c564b', '#17becf', '#bcbd22', '#d62728', '#7f7f7f'
    ]

    print("[渲染] 绘制虚拟矿体...")
    for i, ply_path in enumerate(selected_virtual):
        row = i // n_cols_v
        col = i % n_cols_v

        cell_w = left_width / n_cols_v
        left = col * cell_w + 0.004
        bottom_pos = top_margin - (row + 1) * cell_h + 0.004
        width = cell_w - 0.006
        height = cell_h - 0.008

        ax = fig.add_axes([left, bottom_pos, width, height], projection='3d')
        ax.set_facecolor('white')
        ax.dist = 7  # English comment for public release.

        try:
            verts, faces = read_ply(ply_path)
            color = colors_virtual[i % len(colors_virtual)]
            fname = os.path.splitext(os.path.basename(ply_path))[0]
            _render_virtual_mesh(ax, verts, faces, face_color=color)
        except Exception as e:
            ax.text(0.5, 0.5, 0.5, 'N/A', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='gray')
            print(f"  [警告] 无法加载 {ply_path}: {e}")

    # English comment for public release.
    fig.text(
        left_width / 2, top_margin + 0.015,
        '(a)  Virtual Ore Body Samples',
        ha='center', va='bottom', fontsize=11, fontweight='bold', color='#1a1a2e'
    )
    fig.text(
        left_width / 2, bottom_margin - 0.04,
        f'Selected {len(selected_virtual)} representative samples from {len(virtual_candidates):,} generated ore bodies',
        ha='center', va='top', fontsize=8, color='#555555', style='italic'
    )

    # English comment for public release.
    print("[渲染] 绘制真实矿体与钻孔轨迹...")
    for i, ply_path in enumerate(selected_real):
        row = i // n_cols_r
        col = i % n_cols_r

        cell_w = right_width / n_cols_r
        left = right_start + col * cell_w + 0.004
        bottom_pos = top_margin - (row + 1) * cell_h + 0.004
        width = cell_w - 0.006
        height = cell_h - 0.008

        ax = fig.add_axes([left, bottom_pos, width, height], projection='3d')
        ax.set_facecolor('#f8f8f8')
        ax.dist = 7  # English comment for public release.

        try:
            verts, faces = read_ply(ply_path)
            # English comment for public release.
            verts_norm = _normalize_verts(verts)
            holes = _simulate_drillholes(
                verts_norm,
                num_holes=args.num_holes,
                samples_per_hole=args.samples_per_hole,
                seed=args.seed + i
            )
            _render_real_mesh_with_drillholes(ax, verts_norm, faces, holes)
        except Exception as e:
            ax.text(0.5, 0.5, 0.5, 'N/A', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='gray')
            print(f"  [警告] 无法加载 {ply_path}: {e}")

    # English comment for public release.
    fig.text(
        right_start + right_width / 2, top_margin + 0.015,
        '(b)  Real Drillhole Observations',
        ha='center', va='bottom', fontsize=11, fontweight='bold', color='#1a1a2e'
    )
    fig.text(
        right_start + right_width / 2, bottom_margin - 0.04,
        f'Selected {len(selected_real)} samples from {len(real_candidates)} real ore bodies  '
        f'| {args.num_holes} simulated drillholes each',
        ha='center', va='top', fontsize=8, color='#555555', style='italic'
    )

    # English comment for public release.
    line_x = (left_width + right_start) / 2
    fig.add_artist(
        plt.Line2D(
            [line_x, line_x], [bottom_margin - 0.02, top_margin + 0.03],
            transform=fig.transFigure,
            color='#cccccc', linewidth=1.2, linestyle='--'
        )
    )

    # ── 8. Bottom note (English only) ──────────────────────────────────────
    fig.text(
        0.5, 0.005,
        '* Real mine spatial coordinates have been anonymized.',
        ha='center', va='bottom', fontsize=7.5, color='#888888', style='italic'
    )

    # English comment for public release.
    os.makedirs(args.out_dir, exist_ok=True)
    model_stem = os.path.splitext(args.model_name)[0]
    out_png = os.path.join(args.out_dir, f'dataset_overview_{model_stem}.png')
    out_pdf = os.path.join(args.out_dir, f'dataset_overview_{model_stem}.pdf')

    print(f"[保存] 正在保存 PNG ({args.dpi} DPI)...")
    fig.savefig(out_png, dpi=args.dpi, bbox_inches='tight',
                facecolor='white', pad_inches=0.1)
    print(f"  → {out_png}")

    print("[保存] 正在保存 PDF...")
    fig.savefig(out_pdf, format='pdf', bbox_inches='tight',
                facecolor='white', pad_inches=0.1)
    print(f"  → {out_pdf}")

    plt.close(fig)
    print("[完成] 数据集展示图已生成。")

    # English comment for public release.
    if args.skip_comparison:
        print('[跳过] 已按参数要求跳过对比矩阵图生成。')
    elif _generate_comparison_matrix is None:
        print('[警告] 无法导入 plot_comparison.py，已跳过对比矩阵图生成。')
    else:
        print('[联动] 开始生成全局三维重建对比矩阵图（GT / Sparse / IK / 3D U-Net / Ours）...')
        comp_args = SimpleNamespace(
            model_dir=args.model_dir,
            model_name=args.model_name,
            real_dir=args.real_dir,
            pretrain_dir=args.virtual_dir,
            unet_ckpt=args.unet_ckpt,
            out_dir=args.out_dir,
            rows=max(1, int(args.comparison_rows)),
            real_rows=max(0, int(args.comparison_real_rows)),
            virtual_rows=max(0, int(args.comparison_virtual_rows)),
            fixed_samples=bool(args.comparison_fixed_samples),
            sample_seed=args.comparison_sample_seed,
            selected_virtual_files=args.selected_virtual_files,
            selected_real_files=args.selected_real_files,
            grid_size=32,
            threshold=0.5,
            num_holes=max(1, int(args.num_holes)),
            samples_per_hole=max(2, int(args.samples_per_hole)),
            seed=int(args.seed),
            dpi=int(args.dpi),
            unet_epochs=max(1, int(args.comparison_unet_epochs)),
            unet_batch_size=16,
            unet_lr=1e-3,
            zoom_radius=float(args.zoom_radius),
            zoom_sample_rank=max(1, int(args.zoom_sample_rank)),
        )

        try:
            _generate_comparison_matrix(comp_args)
        except Exception as e:
            print(f'[警告] 对比矩阵图生成失败: {e}')

    # English comment for public release.
    if args.skip_local_zoom:
        print('[跳过] 已按参数要求跳过局部放大对比图生成。')
    elif _generate_local_detail_figure is None:
        print('[警告] 无法导入局部放大图函数，已跳过 Local Detail 图生成。')
    else:
        print('[联动] 开始生成局部细节与伪影抑制放大对比图（Local Detail & Artifact Suppression）...')
        local_args = SimpleNamespace(
            model_dir=args.model_dir,
            model_name=args.model_name,
            real_dir=args.real_dir,
            pretrain_dir=args.virtual_dir,
            unet_ckpt=args.unet_ckpt,
            out_dir=args.out_dir,
            rows=max(1, int(args.comparison_rows)),
            fixed_samples=bool(args.comparison_fixed_samples),
            sample_seed=args.comparison_sample_seed,
            selected_virtual_files=args.selected_virtual_files,
            selected_real_files=args.selected_real_files,
            grid_size=32,
            threshold=0.5,
            num_holes=max(1, int(args.num_holes)),
            samples_per_hole=max(2, int(args.samples_per_hole)),
            seed=int(args.seed),
            dpi=int(args.dpi),
            unet_epochs=max(1, int(args.comparison_unet_epochs)),
            unet_batch_size=16,
            unet_lr=1e-3,
            zoom_radius=float(args.zoom_radius),
            zoom_sample_rank=max(1, int(args.zoom_sample_rank)),
        )
        try:
            _generate_local_detail_figure(local_args)
        except Exception as e:
            print(f'[警告] 局部放大对比图生成失败: {e}')

    # English comment for public release.
    # English comment for public release.
    if _generate_uncertainty_variance_figure is None:
        print('[警告] 无法导入不确定性方差图函数，已跳过 Uncertainty Variance 图生成。')
    else:
        print('[联动] 开始生成认知不确定性方差热力图（Uncertainty Variance Heatmap）...')
        uncertainty_args = SimpleNamespace(
            model_dir=args.model_dir,
            model_name=args.model_name,
            real_dir=args.real_dir,
            pretrain_dir=args.virtual_dir,
            unet_ckpt=args.unet_ckpt,
            out_dir=args.out_dir,
            rows=max(1, int(args.comparison_rows)),
            fixed_samples=bool(args.comparison_fixed_samples),
            sample_seed=args.comparison_sample_seed,
            selected_virtual_files=args.selected_virtual_files,
            selected_real_files=args.selected_real_files,
            grid_size=32,
            threshold=0.5,
            num_holes=max(1, int(args.num_holes)),
            samples_per_hole=max(2, int(args.samples_per_hole)),
            seed=int(args.seed),
            dpi=int(args.dpi),
            unet_epochs=max(1, int(args.comparison_unet_epochs)),
            unet_batch_size=16,
            unet_lr=1e-3,
            zoom_radius=float(args.zoom_radius),
            zoom_sample_rank=max(1, int(args.zoom_sample_rank)),
            uncertainty_mc_samples=max(4, int(args.uncertainty_mc_samples)),
        )
        try:
            _generate_uncertainty_variance_figure(uncertainty_args)
        except Exception as e:
            print(f'[警告] 不确定性方差热力图生成失败: {e}')

    # English comment for public release.
    if args.skip_quant_eval:
        print('[跳过] 已按参数要求跳过定量评估。')
        return

    if _run_quantitative_eval is None:
        print('[警告] 无法导入 quantitative_evaluation.py，已跳过定量评估。')
        return

    print('[联动] 开始执行定量评估（IK/RBF/U-Net/基础VAE/Physics-Aware VAE）...')
    quant_args = SimpleNamespace(
        real_dir=args.real_dir,
        pretrain_dir=args.virtual_dir,
        model_dir=args.model_dir,
        model_name=args.model_name,
        baseline_ckpt=args.baseline_vae_ckpt,
        unet_ckpt=args.unet_ckpt,
        grid_size=32,
        num_holes=max(1, int(args.num_holes)),
        samples_per_hole=max(2, int(args.samples_per_hole)),
        threshold=0.5,
        seed=int(args.seed),
        limit=max(0, int(args.quant_limit)),
        unet_epochs=max(1, int(args.comparison_unet_epochs)),
        unet_batch_size=16,
        unet_lr=1e-3,
        out_dir=args.quant_out_dir,
    )

    try:
        _run_quantitative_eval(quant_args)
    except Exception as e:
        print(f'[警告] 定量评估执行失败: {e}')


def main():
    args = parse_args()
    generate_figure(args)


if __name__ == '__main__':
    main()
