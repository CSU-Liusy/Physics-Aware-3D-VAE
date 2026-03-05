"""Documentation translated to English for open-source release."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def _set_cn_font():
    """Pick an available CJK-capable font to avoid glyph warnings."""
    candidates = [
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Microsoft YaHei",
        "SimSun",
        "Arial Unicode MS",
    ]
    for name in candidates:
        try:
            path = font_manager.findfont(name, fallback_to_default=False)
            if os.path.exists(path):
                plt.rcParams["font.family"] = "sans-serif"
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                matplotlib.rcParams.update({
                    "font.family": "sans-serif",
                    "font.sans-serif": [name, "DejaVu Sans"],
                })
                return name
        except Exception:
            continue

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
    })
    print("⚠️ 未找到中文字体，已回退到 DejaVu Sans，可能出现缺字提示。建议安装 fonts-noto-cjk。")
    return "DejaVu Sans"


_CN_FONT = _set_cn_font()


def ensure_cn_font():
    """Ensure CJK-capable font is configured for matplotlib."""
    return _set_cn_font()

def write_ply_mesh(path, verts, faces, colors=None):
    """Documentation translated to English for open-source release."""
    try:
        with open(path, 'w') as f:
            f.write('ply\nformat ascii 1.0\n')
            f.write(f'element vertex {len(verts)}\n')
            f.write('property float x\nproperty float y\nproperty float z\n')
            if colors is not None:
                f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
            f.write(f'element face {len(faces)}\n')
            f.write('property list uchar int vertex_index\n')
            f.write('end_header\n')
            
            for i, v in enumerate(verts):
                x, y, z = float(v[0]), float(v[1]), float(v[2])
                if colors is not None:
                    r, g, b = colors[i]
                    f.write(f'{x} {y} {z} {int(r)} {int(g)} {int(b)}\n')
                else:
                    f.write(f'{x} {y} {z}\n')
            
            for face in faces:
                f.write(f'3 {int(face[0])} {int(face[1])} {int(face[2])}\n')
    except Exception as e:
        print(f'⚠️  写入PLY网格失败 {path}: {e}')


def save_mesh_ply(path, vox, vmin, vmax, threshold=0.5, color=None):
    """Documentation translated to English for open-source release."""
    if not HAS_SKIMAGE:
        print("⚠️  未安装 scikit-image，无法生成 Mesh PLY，回退到点云。")
        pts = vox_to_pointcloud(vox, threshold)
        # English comment for public release.
        pts_world = pts * (vmax - vmin) + vmin
        write_ply_points(path, pts_world, colors=np.tile(color, (len(pts), 1)) if color else None)
        return

    try:
        # Marching Cubes
        # English comment for public release.
        # English comment for public release.
        # English comment for public release.
        verts, faces, normals, values = marching_cubes(vox, level=threshold)
        
        # English comment for public release.
        # verts[:, 0] -> z (D), verts[:, 1] -> y (H), verts[:, 2] -> x (W)
        # English comment for public release.
        
        D, H, W = vox.shape
        z = verts[:, 0] / (D - 1)
        y = verts[:, 1] / (H - 1)
        x = verts[:, 2] / (W - 1)
        
        # English comment for public release.
        # world_x = x * (vmax[0] - vmin[0]) + vmin[0]
        wx = x * (vmax[0] - vmin[0]) + vmin[0]
        wy = y * (vmax[1] - vmin[1]) + vmin[1]
        wz = z * (vmax[2] - vmin[2]) + vmin[2]
        
        world_verts = np.stack([wx, wy, wz], axis=1)
        
        # English comment for public release.
        v_colors = None
        if color is not None:
            v_colors = np.tile(np.array(color), (len(world_verts), 1))
            
        write_ply_mesh(path, world_verts, faces, colors=v_colors)
        
    except Exception as e:
        # English comment for public release.
        # English comment for public release.
        pass


def save_diff_ply(path, recon_vox, gt_vox, vmin, vmax, threshold=0.5):
    """Documentation translated to English for open-source release."""
    recon_bin = (recon_vox >= threshold)
    gt_bin = (gt_vox >= threshold)
    
    # TP: Recon & GT
    tp = recon_bin & gt_bin
    # FP: Recon & !GT
    fp = recon_bin & (~gt_bin)
    # FN: !Recon & GT
    fn = (~recon_bin) & gt_bin
    
    # English comment for public release.
    def get_pts(mask):
        idx = np.array(np.nonzero(mask)).T
        if idx.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        D, H, W = mask.shape
        z = idx[:, 0].astype(np.float32) / (D - 1)
        y = idx[:, 1].astype(np.float32) / (H - 1)
        x = idx[:, 2].astype(np.float32) / (W - 1)
        
        wx = x * (vmax[0] - vmin[0]) + vmin[0]
        wy = y * (vmax[1] - vmin[1]) + vmin[1]
        wz = z * (vmax[2] - vmin[2]) + vmin[2]
        return np.stack([wx, wy, wz], axis=1)

    pts_tp = get_pts(tp)
    pts_fp = get_pts(fp)
    pts_fn = get_pts(fn)
    
    all_pts = []
    all_colors = []
    
    if len(pts_tp) > 0:
        all_pts.append(pts_tp)
        all_colors.append(np.tile([0, 255, 0], (len(pts_tp), 1))) # Green
    if len(pts_fp) > 0:
        all_pts.append(pts_fp)
        all_colors.append(np.tile([255, 0, 0], (len(pts_fp), 1))) # Red
    if len(pts_fn) > 0:
        all_pts.append(pts_fn)
        all_colors.append(np.tile([0, 0, 255], (len(pts_fn), 1))) # Blue
        
    if not all_pts:
        return
        
    final_pts = np.vstack(all_pts)
    final_colors = np.vstack(all_colors)
    
    write_ply_points(path, final_pts, colors=final_colors)


def write_ply_points(path, points, colors=None):
    """Documentation translated to English for open-source release."""
    try:
        with open(path, 'w') as f:
            f.write('ply\nformat ascii 1.0\n')
            f.write(f'element vertex {len(points)}\n')
            f.write('property float x\nproperty float y\nproperty float z\n')
            if colors is not None:
                f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
            f.write('end_header\n')
            for i, p in enumerate(points):
                x, y, z = float(p[0]), float(p[1]), float(p[2])
                if colors is not None:
                    r, g, b = colors[i]
                    f.write(f'{x} {y} {z} {int(r)} {int(g)} {int(b)}\n')
                else:
                    f.write(f'{x} {y} {z}\n')
    except Exception as e:
        print(f'⚠️  写入PLY文件失败 {path}: {e}')



def convex_hull_2d(points):
    """Documentation translated to English for open-source release."""
    pts = sorted(map(tuple, points.tolist()))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def alpha_shape(points, alpha=0.01):
    """Documentation translated to English for open-source release."""
    try:
        from scipy.spatial import Delaunay
    except ImportError:
        return convex_hull_2d(points)

    if len(points) < 4:
        return convex_hull_2d(points)

    coords = np.array(points)
    tri = Delaunay(coords)
    triangles = coords[tri.simplices]

    a = triangles[:, 0, :]
    b = triangles[:, 1, :]
    c = triangles[:, 2, :]
    la = np.linalg.norm(c - b, axis=1)
    lb = np.linalg.norm(c - a, axis=1)
    lc = np.linalg.norm(b - a, axis=1)
    s = (la + lb + lc) / 2.0
    area = np.sqrt(np.maximum(s * (s - la) * (s - lb) * (s - lc), 0.0))
    with np.errstate(divide='ignore', invalid='ignore'):
        R = (la * lb * lc) / (4.0 * area)
    mask = R < (1.0 / max(1e-12, alpha))
    edges = {}
    for tri_idx, keep in enumerate(mask):
        if not keep:
            continue
        simplex = tri.simplices[tri_idx]
        for i in range(3):
            e = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
            edges[e] = edges.get(e, 0) + 1

    boundary_edges = [e for e, cnt in edges.items() if cnt == 1]
    if not boundary_edges:
        return convex_hull_2d(points)

    adj = {}
    for ea, eb in boundary_edges:
        adj.setdefault(ea, []).append(eb)
        adj.setdefault(eb, []).append(ea)

    start = boundary_edges[0][0]
    for k, v in adj.items():
        if len(v) == 1:
            start = k
            break

    hull_idx = [start]
    prev, cur = None, start
    while True:
        nbrs = adj[cur]
        nxt = nbrs[0] if nbrs[0] != prev else (nbrs[1] if len(nbrs) > 1 else None)
        if nxt is None or nxt == start:
            break
        hull_idx.append(nxt)
        prev, cur = cur, nxt

    hull = [tuple(coords[i]) for i in hull_idx]
    return hull if len(hull) >= 3 else convex_hull_2d(points)


def vox_to_pointcloud(vox, threshold=0.5, max_points=None):
    """Convert voxels to Nx3 points in [0,1], with optional downsizing."""
    D, H, W = vox.shape
    grid = vox >= threshold
    idx = np.array(np.nonzero(grid)).T
    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if max_points is not None and max_points > 0 and idx.shape[0] > max_points:
        stride = int(np.ceil((idx.shape[0] / float(max_points)) ** (1.0 / 3.0)))
        stride = max(1, stride)
        if stride > 1:
            mask = ((idx[:, 0] % stride == 0) &
                    (idx[:, 1] % stride == 0) &
                    (idx[:, 2] % stride == 0))
            filtered = idx[mask]
            if filtered.size > 0:
                idx = filtered
        if idx.shape[0] > max_points:
            step = max(1, idx.shape[0] // max_points)
            idx = idx[::step]
            if idx.shape[0] > max_points:
                idx = idx[:max_points]

    y = idx[:, 0].astype(np.float32)
    x = idx[:, 1].astype(np.float32)
    z = idx[:, 2].astype(np.float32)
    if D > 1:
        y /= float(D - 1)
    if H > 1:
        x /= float(H - 1)
    if W > 1:
        z /= float(W - 1)
    return np.stack([x, y, z], axis=1)


def save_xy_comparison(recon_pts, gt_pts, vmin, vmax, output_dir, epoch, save_svg=True):
    """Documentation translated to English for open-source release."""
    xmin, ymin = vmin[0], vmin[1]
    xmax, ymax = vmax[0], vmax[1]
    dx = max(1e-6, xmax - xmin)
    dy = max(1e-6, ymax - ymin)
    margin_x = dx * 0.02
    margin_y = dy * 0.02

    # English comment for public release.
    if recon_pts.shape[0] > 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        nbins = 64
        hist, xedges, yedges = np.histogram2d(recon_pts[:, 0], recon_pts[:, 1], bins=nbins)
        xidx = np.clip(np.digitize(recon_pts[:, 0], xedges) - 1, 0, nbins - 1)
        yidx = np.clip(np.digitize(recon_pts[:, 1], yedges) - 1, 0, nbins - 1)
        dens = hist[xidx, yidx]
        sc = ax.scatter(recon_pts[:, 0], recon_pts[:, 1], c=dens, s=4, cmap='viridis', marker='o')
        ax.set_xlim(xmin - margin_x, xmax + margin_x)
        ax.set_ylim(ymin - margin_y, ymax + margin_y)
        ax.set_aspect('equal')
        ax.ticklabel_format(style='plain', axis='both')
        plt.colorbar(sc, ax=ax, label='密度')
        ax.set_title(f'重建结果 Epoch {epoch}')
        plt.savefig(os.path.join(output_dir, f'recon_epoch_{epoch}_xy.png'), dpi=150)
        if save_svg:
            plt.savefig(os.path.join(output_dir, f'recon_epoch_{epoch}_xy.svg'))
        plt.close(fig)

    # English comment for public release.
    if gt_pts.shape[0] > 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        nbins = 64
        hist, xedges, yedges = np.histogram2d(gt_pts[:, 0], gt_pts[:, 1], bins=nbins)
        xidx = np.clip(np.digitize(gt_pts[:, 0], xedges) - 1, 0, nbins - 1)
        yidx = np.clip(np.digitize(gt_pts[:, 1], yedges) - 1, 0, nbins - 1)
        dens = hist[xidx, yidx]
        sc = ax.scatter(gt_pts[:, 0], gt_pts[:, 1], c=dens, s=4, cmap='plasma', marker='o')
        ax.set_xlim(xmin - margin_x, xmax + margin_x)
        ax.set_ylim(ymin - margin_y, ymax + margin_y)
        ax.set_aspect('equal')
        ax.ticklabel_format(style='plain', axis='both')
        plt.colorbar(sc, ax=ax, label='密度')
        ax.set_title(f'GT 目标 Epoch {epoch}')
        plt.savefig(os.path.join(output_dir, f'GT_epoch_{epoch}_xy.png'), dpi=150)
        if save_svg:
            plt.savefig(os.path.join(output_dir, f'GT_epoch_{epoch}_xy.svg'))
        plt.close(fig)

    # English comment for public release.
    if recon_pts.shape[0] > 0 and gt_pts.shape[0] > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # English comment for public release.
        ax = axes[0]
        ax.scatter(recon_pts[:, 0], recon_pts[:, 1], s=2, c='blue', alpha=0.6)
        ax.set_xlim(xmin - margin_x, xmax + margin_x)
        ax.set_ylim(ymin - margin_y, ymax + margin_y)
        ax.set_aspect('equal')
        ax.ticklabel_format(style='plain', axis='both')
        ax.set_title(f'重建 Epoch {epoch}')
        # GT
        ax = axes[1]
        ax.scatter(gt_pts[:, 0], gt_pts[:, 1], s=2, c='red', alpha=0.6)
        ax.set_xlim(xmin - margin_x, xmax + margin_x)
        ax.set_ylim(ymin - margin_y, ymax + margin_y)
        ax.set_aspect('equal')
        ax.ticklabel_format(style='plain', axis='both')
        ax.set_title(f'GT Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'compare_epoch_{epoch}_xy.png'), dpi=150)
        if save_svg:
            plt.savefig(os.path.join(output_dir, f'compare_epoch_{epoch}_xy.svg'))
        plt.close(fig)


def save_area_plot(pts, vmin, vmax, output_dir, epoch, prefix='recon', save_svg=True):
    """Documentation translated to English for open-source release."""
    if pts.shape[0] < 3:
        return
    xmin, ymin = vmin[0], vmin[1]
    xmax, ymax = vmax[0], vmax[1]
    dx = max(1e-6, xmax - xmin)
    dy = max(1e-6, ymax - ymin)
    margin_x = dx * 0.02
    margin_y = dy * 0.02

    xy = pts[:, :2]
    try:
        hull = alpha_shape(xy, alpha=0.01)
    except Exception:
        hull = convex_hull_2d(xy)

    if hull and len(hull) >= 3:
        hx = [p[0] for p in hull]
        hy = [p[1] for p in hull]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.fill(hx, hy, alpha=0.4, facecolor='orange', edgecolor='red')
        ax.scatter(pts[:, 0], pts[:, 1], s=1, c='black')
        ax.set_xlim(xmin - margin_x, xmax + margin_x)
        ax.set_ylim(ymin - margin_y, ymax + margin_y)
        ax.set_aspect('equal')
        ax.ticklabel_format(style='plain', axis='both')
        ax.set_title(f'{prefix.upper()} Area Epoch {epoch}')
        plt.savefig(os.path.join(output_dir, f'{prefix}_epoch_{epoch}_area.png'), dpi=150)
        if save_svg:
            plt.savefig(os.path.join(output_dir, f'{prefix}_epoch_{epoch}_area.svg'))
        plt.close(fig)


def generate_epoch_outputs(recon_vox, gt_vox, vmin, vmax, output_dir, epoch, threshold=0.5, save_svg=False):
    """Documentation translated to English for open-source release."""
    try:
        vmin = np.asarray(vmin)
        vmax = np.asarray(vmax)
        
        # English comment for public release.
        if recon_vox.shape != gt_vox.shape:
            print(f'⚠️  体素形状不匹配: recon={recon_vox.shape}, gt={gt_vox.shape}')
            return None, None
        
        # English comment for public release.
        recon_pts_norm = vox_to_pointcloud(recon_vox, threshold)
        gt_pts_norm = vox_to_pointcloud(gt_vox, threshold)
        
        # English comment for public release.
        recon_pts_world = vmin + recon_pts_norm * (vmax - vmin)
        gt_pts_world = vmin + gt_pts_norm * (vmax - vmin)
        
        # English comment for public release.
        write_ply_points(os.path.join(output_dir, f'recon_epoch_{epoch}.ply'), recon_pts_world)
        write_ply_points(os.path.join(output_dir, f'GT_epoch_{epoch}.ply'), gt_pts_world)
        
        # English comment for public release.
        try:
            save_xy_comparison(recon_pts_world, gt_pts_world, vmin, vmax, output_dir, epoch, save_svg)
        except Exception as e:
            print(f'⚠️  XY对比图生成失败: {e}')
        
        # English comment for public release.
        try:
            save_area_plot(recon_pts_world, vmin, vmax, output_dir, epoch, prefix='recon', save_svg=save_svg)
            save_area_plot(gt_pts_world, vmin, vmax, output_dir, epoch, prefix='GT', save_svg=save_svg)
        except Exception as e:
            print(f'⚠️  区域图生成失败: {e}')

        return recon_pts_world, gt_pts_world
    except Exception as e:
        print(f'❌ Epoch 输出生成失败: {e}')
        return None, None


def generate_sample_report(recon_vox, gt_vox, vmin, vmax, output_dir, filename_prefix, threshold=0.3, orig_verts=None, orig_faces=None):
    """Documentation translated to English for open-source release."""
    # 1. Original (GT)
    if orig_verts is not None and orig_faces is not None:
        # English comment for public release.
        write_ply_mesh(
            os.path.join(output_dir, f'{filename_prefix}_original.ply'),
            orig_verts, orig_faces, colors=np.tile([200, 200, 200], (len(orig_verts), 1))
        )
    else:
        # English comment for public release.
        save_mesh_ply(
            os.path.join(output_dir, f'{filename_prefix}_original.ply'),
            gt_vox, vmin, vmax, threshold, color=[200, 200, 200] # Grey
        )
    
    # 2. Generated (Recon)
    save_mesh_ply(
        os.path.join(output_dir, f'{filename_prefix}_generated.ply'),
        recon_vox, vmin, vmax, threshold, color=[100, 150, 255] # Light Blue
    )
    
    # 3. Difference
    save_diff_ply(
        os.path.join(output_dir, f'{filename_prefix}_diff.ply'),
        recon_vox, gt_vox, vmin, vmax, threshold
    )


def get_cube_mesh(center, size):
    """Documentation translated to English for open-source release."""
    x, y, z = center
    s = size / 2.0
    # 8 vertices
    verts = np.array([
        [x-s, y-s, z-s], [x+s, y-s, z-s], [x+s, y+s, z-s], [x-s, y+s, z-s], # Bottom
        [x-s, y-s, z+s], [x+s, y-s, z+s], [x+s, y+s, z+s], [x-s, y+s, z+s]  # Top
    ])
    # 12 faces (2 per face of cube)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], # Bottom
        [4, 5, 6], [4, 6, 7], # Top
        [0, 1, 5], [0, 5, 4], # Front
        [2, 3, 7], [2, 7, 6], # Back
        [1, 2, 6], [1, 6, 5], # Right
        [3, 0, 4], [3, 4, 7]  # Left
    ])
    return verts, faces


def save_combined_scene(path, recon_vox, gt_vox, drill_traces, vmin, vmax, threshold=0.5, orig_verts=None, orig_faces=None):
    """Documentation translated to English for open-source release."""
    if not HAS_SKIMAGE:
        print("⚠️  未安装 scikit-image，无法生成综合场景 Mesh。")
        return

    all_verts = []
    all_faces = []
    all_edges = []
    all_colors = [] # RGBA
    
    current_v_idx = 0
    
    # --- 1. GT Mesh (Blue, Alpha=80) ---
    try:
        if orig_verts is not None and orig_faces is not None:
            # English comment for public release.
            all_verts.append(orig_verts)
            all_faces.append(orig_faces + current_v_idx)
            colors = np.tile([0, 0, 255, 80], (len(orig_verts), 1))
            all_colors.append(colors)
            current_v_idx += len(orig_verts)
        else:
            # English comment for public release.
            verts, faces, _, _ = marching_cubes(gt_vox, level=threshold)
            # Normalize & Map to World
            D, H, W = gt_vox.shape
            z = verts[:, 0] / (D - 1)
            y = verts[:, 1] / (H - 1)
            x = verts[:, 2] / (W - 1)
            wx = x * (vmax[0] - vmin[0]) + vmin[0]
            wy = y * (vmax[1] - vmin[1]) + vmin[1]
            wz = z * (vmax[2] - vmin[2]) + vmin[2]
            world_verts = np.stack([wx, wy, wz], axis=1)
            
            all_verts.append(world_verts)
            all_faces.append(faces + current_v_idx)
            
            # Color: Blue, Alpha 80
            colors = np.tile([0, 0, 255, 80], (len(world_verts), 1))
            all_colors.append(colors)
            
            current_v_idx += len(world_verts)
    except Exception:
        pass

    # --- 2. Recon Mesh (Green, Alpha=80) ---
    try:
        verts, faces, _, _ = marching_cubes(recon_vox, level=threshold)
        D, H, W = recon_vox.shape
        z = verts[:, 0] / (D - 1)
        y = verts[:, 1] / (H - 1)
        x = verts[:, 2] / (W - 1)
        wx = x * (vmax[0] - vmin[0]) + vmin[0]
        wy = y * (vmax[1] - vmin[1]) + vmin[1]
        wz = z * (vmax[2] - vmin[2]) + vmin[2]
        world_verts = np.stack([wx, wy, wz], axis=1)
        
        all_verts.append(world_verts)
        all_faces.append(faces + current_v_idx)
        
        # Color: Green, Alpha 80
        colors = np.tile([0, 255, 0, 80], (len(world_verts), 1))
        all_colors.append(colors)
        
        current_v_idx += len(world_verts)
    except Exception:
        pass

    # --- 3. Drill Traces (Red, Alpha=255) ---
    # English comment for public release.
    line_verts = []
    line_edges = []
    
    # English comment for public release.
    trace_start_indices = []
    
    for trace in drill_traces:
        if not trace: 
            trace_start_indices.append(None)
            continue
        
        start_idx = current_v_idx
        trace_start_indices.append(start_idx)
        
        for i, p in enumerate(trace):
            line_verts.append([p[0], p[1], p[2]])
            if i > 0:
                line_edges.append([start_idx + i - 1, start_idx + i])
        
        current_v_idx += len(trace)
    
    if line_verts:
        all_verts.append(np.array(line_verts))
        all_colors.append(np.tile([255, 0, 0, 255], (len(line_verts), 1)))
        all_edges.append(np.array(line_edges))

    # English comment for public release.
    # English comment for public release.
    diag = np.linalg.norm(vmax - vmin)
    point_size = diag * 0.015 
    
    for trace in drill_traces:
        for p in trace:
            c_verts, c_faces = get_cube_mesh(p, point_size)
            
            # English comment for public release.
            c_faces += current_v_idx
            
            all_verts.append(c_verts)
            all_faces.append(c_faces)
            all_colors.append(np.tile([255, 0, 0, 255], (len(c_verts), 1)))
            
            current_v_idx += len(c_verts)

    # --- Write PLY ---
    try:
        with open(path, 'w') as f:
            total_verts = sum(len(v) for v in all_verts)
            total_faces = sum(len(f) for f in all_faces)
            total_edges = sum(len(e) for e in all_edges)
            
            f.write('ply\nformat ascii 1.0\n')
            f.write(f'element vertex {total_verts}\n')
            f.write('property float x\nproperty float y\nproperty float z\n')
            f.write('property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n')
            
            f.write(f'element face {total_faces}\n')
            f.write('property list uchar int vertex_index\n')
            
            f.write(f'element edge {total_edges}\n')
            f.write('property int vertex1\nproperty int vertex2\n')
            f.write('property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n')
            
            f.write('end_header\n')
            
            # Vertices
            for verts, colors in zip(all_verts, all_colors):
                for v, c in zip(verts, colors):
                    f.write(f'{v[0]} {v[1]} {v[2]} {int(c[0])} {int(c[1])} {int(c[2])} {int(c[3])}\n')
            
            # Faces
            for faces in all_faces:
                for face in faces:
                    f.write(f'3 {int(face[0])} {int(face[1])} {int(face[2])}\n')
            
            # Edges
            # Edge color is Red (255, 0, 0, 255)
            for edges in all_edges:
                for edge in edges:
                    f.write(f'{int(edge[0])} {int(edge[1])} 255 0 0 255\n')
                    
    except Exception as e:
        print(f'⚠️  写入综合场景 PLY 失败 {path}: {e}')


def save_downsampled_points(path, vox, vmin, vmax, threshold=0.3, max_points=60000):
    """Export a reduced point cloud while keeping global shape."""
    try:
        vox = np.asarray(vox)
        vmin = np.asarray(vmin, dtype=np.float32)
        vmax = np.asarray(vmax, dtype=np.float32)
        if vox.ndim == 4:
            vox = np.squeeze(vox, axis=0)
        limit = None if max_points is None or max_points <= 0 else int(max_points)
        pts_norm = vox_to_pointcloud(vox, threshold=threshold, max_points=limit)
        if pts_norm.size == 0:
            print(f'⚠️  下采样后体素为空，跳过 {path}')
            return
        scale = (vmax - vmin).astype(np.float32)
        scale = np.where(scale == 0.0, 1e-8, scale)
        points = vmin + pts_norm * scale
        write_ply_points(path, points)
    except Exception as e:
        print(f'⚠️  保存下采样点云失败 {path}: {e}')
