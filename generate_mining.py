"""Documentation translated to English for open-source release."""

import argparse
import os
import sys
# English comment for public release.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import time
import math
import random
import numpy as np
import concurrent.futures # English comment for public release.

try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# English comment for public release.
try:
    from dataset import read_ply, point_in_mesh
    HAS_DATASET_UTILS = True
except ImportError:
    HAS_DATASET_UTILS = False



def write_ply_mesh(path, verts, faces, color):
    """Documentation translated to English for open-source release."""
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_index\n")
        f.write("end_header\n")
        
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        for v in verts:
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {r} {g} {b}\n")
        
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def get_target_face_count(base_target, temperature, shape_seed):
    """Documentation translated to English for open-source release."""
    if temperature >= 0.999:
        return base_target
    
    rng = np.random.default_rng(shape_seed)
    
    # English comment for public release.
    # English comment for public release.
    dev_range = (1.0 - temperature) * 0.5
    factor = 1.0 + rng.uniform(-dev_range, dev_range)
    
    return max(100, int(round(base_target * factor)))


def load_all_source_meshes(source_dir):
    """Documentation translated to English for open-source release."""
    if not HAS_DATASET_UTILS:
        raise ImportError("translated_text dataset.py translated_text read_ply translated_text, translated_text")
    
    meshes = []
    if not os.path.isdir(source_dir):
        print(f"translated_text: translated_text {source_dir} translated_text")
        return meshes

    files = [f for f in os.listdir(source_dir) if f.lower().endswith('.ply')]
    print(f"translated_text {len(files)} translated_text...")
    for f in files:
        try:
            verts, faces = read_ply(os.path.join(source_dir, f))
            # English comment for public release.
            vmin, vmax = verts.min(0), verts.max(0)
            center = (vmin + vmax) / 2
            scale = 1.0 / (np.linalg.norm(vmax - vmin) + 1e-8) * 1.8 # English comment for public release.
            verts = (verts - center) * scale
            meshes.append((verts, faces))
        except Exception:
            continue
    print(f"translated_text {len(meshes)} translated_text")
    return meshes


def generate_density_field(grid_res, shape_seed, disperse, source_meshes=None):
    """Documentation translated to English for open-source release."""
    rng = np.random.default_rng(shape_seed)
    
    # English comment for public release.
    xs = np.linspace(-1, 1, grid_res).astype(np.float32)
    ys = np.linspace(-1, 1, grid_res).astype(np.float32)
    zs = np.linspace(-1, 1, grid_res).astype(np.float32)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing='ij')
    
    # English comment for public release.
    field = np.zeros_like(X, dtype=np.float32) - 1.0 # English comment for public release.

    if source_meshes and len(source_meshes) > 0 and HAS_DATASET_UTILS and HAS_SCIPY:
        # English comment for public release.
        # English comment for public release.
        num_bodies = rng.integers(1, 4)
        
        # English comment for public release.
        # English comment for public release.
        # English comment for public release.
        grid_points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        
        combined_field = np.zeros(grid_points.shape[0], dtype=np.float32)

        for _ in range(num_bodies):
            verts_src, faces_src = source_meshes[rng.integers(0, len(source_meshes))]
            verts = verts_src.copy()

            # English comment for public release.
            theta = rng.uniform(0, 2*np.pi)
            phi = rng.uniform(0, np.pi)
            R_z = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            # English comment for public release.
            verts = verts @ R_z.T 

            # English comment for public release.
            # English comment for public release.
            scale_base = rng.uniform(0.5, 1.2)
            flatten = rng.uniform(0.3, 1.0) # English comment for public release.
            
            # English comment for public release.
            scale_vec = np.array([scale_base, scale_base, scale_base])
            axis = rng.integers(0, 3)
            scale_vec[axis] *= flatten
            verts = verts * scale_vec

            # English comment for public release.
            # English comment for public release.
            offset = rng.uniform(-0.5 * disperse, 0.5 * disperse, size=3)
            verts = verts + offset
            
            # English comment for public release.
            # English comment for public release.
            # English comment for public release.
            vmin, vmax = verts.min(0), verts.max(0)
            # English comment for public release.
            vmin -= 0.05
            vmax += 0.05
            
            mask_aabb = (
                (grid_points[:, 0] >= vmin[0]) & (grid_points[:, 0] <= vmax[0]) &
                (grid_points[:, 1] >= vmin[1]) & (grid_points[:, 1] <= vmax[1]) &
                (grid_points[:, 2] >= vmin[2]) & (grid_points[:, 2] <= vmax[2])
            )
            
            if np.any(mask_aabb):
                pts_to_check = grid_points[mask_aabb]
                inside = point_in_mesh(pts_to_check, verts, faces_src.astype(np.int32)) # English comment for public release.
                
                # English comment for public release.
                combined_field[mask_aabb] += inside.astype(np.float32)

        # English comment for public release.
        field_raw = combined_field.reshape(grid_res, grid_res, grid_res)
        
        # English comment for public release.
        # English comment for public release.
        sigma = 1.0 + rng.uniform(0, 1.0)
        field = gaussian_filter(field_raw, sigma=sigma)
        
        # English comment for public release.
        # English comment for public release.
        # English comment for public release.
        # English comment for public release.
        # English comment for public release.
        field = (field - 0.5) * 2.0
        
        # English comment for public release.
        scale_vec = np.array([1.0, 1.0, 1.0])
        X_scaled, Y_scaled, Z_scaled = X, Y, Z

    else:
        # English comment for public release.
        # English comment for public release.
        # English comment for public release.
        # English comment for public release.
        mode_rand = rng.random()
        if mode_rand < 0.60:
            morphology = "sheet"
        elif mode_rand < 0.85:
            morphology = "rod"
        else:
            morphology = "massive"

        scale_vec = np.array([1.0, 1.0, 1.0])
        
        if morphology == "sheet":
            # English comment for public release.
            thin_axis = rng.integers(0, 3) 
            # English comment for public release.
            flatten_factor = rng.uniform(4.0, 10.0)
            scale_vec[thin_axis] = flatten_factor
            # English comment for public release.
            other_axes = [i for i in range(3) if i != thin_axis]
            elongate_axis = rng.choice(other_axes)
            scale_vec[elongate_axis] = rng.uniform(1.2, 2.0)
            
        elif morphology == "rod":
            # English comment for public release.
            long_axis = rng.integers(0, 3)
            # English comment for public release.
            for i in range(3):
                if i != long_axis:
                    scale_vec[i] = rng.uniform(2.5, 4.5)
            scale_vec[long_axis] = 1.0
            
        else: # massive
            # English comment for public release.
            scale_vec = rng.uniform(0.8, 1.5, size=3)

        # English comment for public release.
        X_scaled = X * scale_vec[0]
        Y_scaled = Y * scale_vec[1]
        Z_scaled = Z * scale_vec[2]
        
        # English comment for public release.
        field = np.zeros_like(X, dtype=np.float32)
        
        # English comment for public release.
        if disperse < 0.1:
            num_blobs = rng.integers(2, 5)
        else:
            num_blobs = rng.integers(4, 9)

        # English comment for public release.
        if morphology == "sheet":
            # English comment for public release.
            spread_ranges = [0.6, 0.6, 0.6] # English comment for public release.
            spread_ranges[thin_axis] = 0.05 
        elif morphology == "rod":
            # English comment for public release.
            spread_ranges = [0.15, 0.15, 0.15]
            long_ax_idx = np.argmin(scale_vec)
            spread_ranges[long_ax_idx] = 0.7 
        else:
            # English comment for public release.
            r_spread = 0.2 + 0.3 * disperse
            spread_ranges = [r_spread, r_spread, r_spread]

        for _ in range(num_blobs):
            c_pos = [0.0, 0.0, 0.0]
            for ax in range(3):
                c_pos[ax] = rng.uniform(-spread_ranges[ax], spread_ranges[ax])
                
            cx, cy, cz = c_pos
            dx = X_scaled - cx * scale_vec[0]
            dy = Y_scaled - cy * scale_vec[1]
            dz = Z_scaled - cz * scale_vec[2]
            dist_sq = dx*dx + dy*dy + dz*dz
            radius = rng.uniform(0.4, 0.75)
            blob = np.exp(-dist_sq / (2 * radius * radius))
            field += blob * rng.uniform(0.8, 1.2)
            
        # English comment for public release.
        freq = rng.uniform(1.2, 2.5)
        phase = rng.uniform(0, 6.28)
        # English comment for public release.
        axis_opts = [X_scaled, Y_scaled, Z_scaled]
        warp_val = 0.15 * np.sin(axis_opts[rng.integers(0, 3)] * freq + phase) * np.cos(axis_opts[rng.integers(0, 3)] * freq)
        field += warp_val

        # English comment for public release.
        use_planes = rng.random() < 0.7 # English comment for public release.
        
        if use_planes:
            num_faults = rng.integers(1, 5)
            for _ in range(num_faults):
                fn = rng.standard_normal(3).astype(np.float32)
                fn /= np.linalg.norm(fn)
                
                # English comment for public release.
                # English comment for public release.
                # English comment for public release.
                # English comment for public release.
                safe_dist = rng.uniform(0.25, 0.65)
                
                dot_val = X * fn[0] + Y * fn[1] + Z * fn[2]
                d_plane = safe_dist - dot_val
                
                cut_mask = np.clip((d_plane + 0.02) / 0.04, 0.0, 1.0)
                field *= cut_mask
        else:
            # English comment for public release.
            erosion_noise = rng.uniform(0, 1, size=field.shape)
            field -= erosion_noise * 0.35

        # English comment for public release.
        noise = rng.uniform(-0.06, 0.06, size=field.shape).astype(np.float32)
        field += noise * np.clip(field, 0, 1)

    # English comment for public release.
    # English comment for public release.
    dist_to_edge = np.min(np.stack([
        1.0 - np.abs(X), 
        1.0 - np.abs(Y), 
        1.0 - np.abs(Z)
    ]), axis=0)
    # English comment for public release.
    # English comment for public release.
    # English comment for public release.
    # English comment for public release.
    dist_from_center_box = np.maximum(np.abs(X), np.maximum(np.abs(Y), np.abs(Z)))
    
    # English comment for public release.
    border_mask = np.clip((dist_from_center_box - 0.90) / 0.08, 0.0, 1.0)
    
    # English comment for public release.
    # English comment for public release.
    field -= border_mask * 10.0

    # English comment for public release.
    # English comment for public release.
    field[:2, :, :] = -10.0
    field[-2:, :, :] = -10.0
    field[:, :2, :] = -10.0
    field[:, -2:, :] = -10.0
    field[:, :, :2] = -10.0
    field[:, :, -2:] = -10.0

    return field


def solve_mesh_for_target(exact_target, shape_seed, disperse, source_meshes=None):
    """Documentation translated to English for open-source release."""
    # English comment for public release.
    # English comment for public release.
    # English comment for public release.
    # res = sqrt(target / C). Let's guess C ~= 0.7
    
    current_res = int(math.sqrt(exact_target / 0.8))
    current_res = max(32, min(current_res, 300))
    current_res = (current_res // 2) * 2 # English comment for public release.
    
    # English comment for public release.
    tolerance = 0.10 # 10%
    max_iter = 4
    
    best_result = None
    min_diff = float('inf')
    
    iso_level = 0.35  # English comment for public release.

    for i in range(max_iter):
        # English comment for public release.
        field = generate_density_field(current_res, shape_seed, disperse, source_meshes=source_meshes)
        
        # 2. Marching Cubes

        try:
            verts, faces, norm, val = marching_cubes(field, level=iso_level)
        except (ValueError, RuntimeError):
            # English comment for public release.
            verts, faces = [], []
            
        n_faces = len(faces)
        diff = abs(n_faces - exact_target)
        
        # English comment for public release.
        if diff < min_diff:
            min_diff = diff
            best_result = (verts, faces)
            
        # English comment for public release.
        if n_faces > 0 and diff / exact_target <= tolerance:
            break
            
        # English comment for public release.
        # English comment for public release.
        # Res_new = Res_old * sqrt(Target / N_old)
        if n_faces == 0:
            current_res += 16 # English comment for public release.
        else:
            ratio = math.sqrt(exact_target / n_faces)
            # English comment for public release.
            ratio = np.clip(ratio, 0.7, 1.4)
            new_res = int(current_res * ratio)
            # English comment for public release.
            if new_res == current_res:
                new_res += 2 if n_faces < exact_target else -2
            current_res = new_res
        
        current_res = max(32, min(current_res, 320))
        
    # English comment for public release.
    # English comment for public release.
    # English comment for public release.
    
    if best_result is None or len(best_result[1]) == 0:
        raise RuntimeError(f"translated_text (Target: {exact_target})")
        
    verts, faces = best_result
    
    # English comment for public release.
    # English comment for public release.
    # English comment for public release.
    # English comment for public release.
    
    # English comment for public release.
    # English comment for public release.
    # English comment for public release.
    # English comment for public release.
    # English comment for public release.
    
    # English comment for public release.
    
    # English comment for public release.
    # English comment for public release.
    # skimage docs: returns (N, 3) spatial coordinates for V vertices.
    # usually corresponds to indexing='ij' -> (0, 1, 2) matches (x, y, z) if passed correctly.
    # But meshgrid(ij) maps input array order.
    # Let's assume uniform scaling correction:
    
    # English comment for public release.
    # English comment for public release.
    # English comment for public release.
    # English comment for public release.
    # English comment for public release.
    # English comment for public release.
    # English comment for public release.
    
    real_res = field.shape[0]
    scale = 2.0 / (real_res - 1)
    # English comment for public release.
    verts = verts * scale - 1.0
    
    # English comment for public release.
    # English comment for public release.
    # English comment for public release.
    # English comment for public release.
    # English comment for public release.
    verts = verts[:, [2, 1, 0]]
    
    return verts, faces, n_faces


# =============================================================================
# English comment for public release.
# =============================================================================

# English comment for public release.
WORKER_SOURCE_MESHES = None
WORKER_SOURCE_DIR = None

def init_worker_process(source_dir, use_augment, rescale_only):
    """Documentation translated to English for open-source release."""
    global WORKER_SOURCE_MESHES
    global WORKER_SOURCE_DIR
    
    WORKER_SOURCE_DIR = source_dir
    
    # English comment for public release.
    unique_seed = (os.getpid() * int(time.time() * 1000)) % 123456789
    np.random.seed(unique_seed)
    
    # English comment for public release.
    # English comment for public release.
    if use_augment:
        # English comment for public release.
        time.sleep(random.uniform(0.1, 1.5))
        try:
             # English comment for public release.
             # English comment for public release.
             WORKER_SOURCE_MESHES = load_all_source_meshes(source_dir)
        except Exception as e:
             # English comment for public release.
             WORKER_SOURCE_MESHES = None
    else:
        WORKER_SOURCE_MESHES = None


def process_single_task(job_args):
    """Documentation translated to English for open-source release."""
    # English comment for public release.
    idx, total, target_faces, temperature, disperse, out_dir, base_seed, rescale_arg = job_args
    
    try:
        file_seed = base_seed + idx
        rng = np.random.default_rng(file_seed)
        
        # English comment for public release.
        # English comment for public release.
        is_rescale_mode = isinstance(rescale_arg, str) or (rescale_arg is True)
        
        if is_rescale_mode:
            
            src_verts, src_faces = None, None
            
            if isinstance(rescale_arg, str):
                # English comment for public release.
                # English comment for public release.
                # English comment for public release.
                # English comment for public release.
                # English comment for public release.
                # English comment for public release.
                
                # English comment for public release.
                if WORKER_SOURCE_DIR is None:
                     return False, f"ID: {idx} | Rescale translated_text: Worker translated_text", idx
                
                fname = rescale_arg
                fpath = os.path.join(WORKER_SOURCE_DIR, fname)
                try:
                    src_verts, src_faces = read_ply(fpath)
                except Exception as ex:
                    return False, f"ID: {idx} | translated_text {fname}: {ex}", idx
            
            elif WORKER_SOURCE_MESHES:
                # English comment for public release.
                src_verts, src_faces = WORKER_SOURCE_MESHES[rng.integers(0, len(WORKER_SOURCE_MESHES))]
            else:
                 return False, f"ID: {idx} | Rescale translated_text: translated_text", idx

            verts = src_verts.copy()
            faces = src_faces.copy()
            
            # English comment for public release.
            center = (verts.min(0) + verts.max(0)) / 2.0
            verts = verts - center
            
            # English comment for public release.
            # English comment for public release.
            # English comment for public release.
            # English comment for public release.
            
            # English comment for public release.
            max_span = np.linalg.norm(verts.max(0) - verts.min(0)) + 1e-8
            verts = verts / max_span 
            
            # English comment for public release.
            target_size = rng.uniform(100.0, 1500.0)
            verts = verts * target_size
            
            # English comment for public release.
            # English comment for public release.
            from scipy.stats import special_ortho_group
            if HAS_SCIPY:
                 R = special_ortho_group.rvs(3, random_state=rng)
                 verts = verts @ R.T
            
            final_count = len(faces)
            dt = 0.1
            
        else:
            # English comment for public release.
            
            # English comment for public release.
            this_target = get_target_face_count(target_faces, temperature, file_seed)
            
            # English comment for public release.
            t0 = time.time()
            
            verts, faces, final_count = solve_mesh_for_target(
                this_target, 
                file_seed, 
                disperse, 
                source_meshes=WORKER_SOURCE_MESHES
            )
            dt = time.time() - t0
        
        # English comment for public release.
        color = rng.integers(100, 255, size=3)
        
        # English comment for public release.
        if isinstance(rescale_arg, str):
            # English comment for public release.
            base = os.path.splitext(rescale_arg)[0]
            fname = f"{base}.ply"
        else:
            prefix = "rescale" if is_rescale_mode else "generate"
            fname = f"{prefix}_mine_j{idx}.ply"
            
        fpath = os.path.join(out_dir, fname)
        write_ply_mesh(fpath, verts, faces, color)
        
        if is_rescale_mode:
            return True, f"ID: {idx} | Rescale {fname} | Size: {np.ptp(verts, axis=0).max():.1f}m | Faces: {final_count}", idx
        else:
            error_rate = (final_count - this_target) / this_target * 100
            # English comment for public release.
            return True, f"ID: {idx} | Faces: {final_count} (Err {error_rate:+.1f}%) | Time: {dt:.2f}s", idx
        
    except Exception as e:
        return False, f"ID: {idx} | translated_text: {e}", idx


def main():
    parser = argparse.ArgumentParser(description="translated_text (translated_text/translated_text) PLY")
    parser.add_argument("--count", type=int, default=611, help="translated_text")
    parser.add_argument("--target-faces", type=int, default=7500, help="translated_text")
    parser.add_argument("--temperature", type=float, default=0.22, help="1.0=translated_text, translated_text")
    parser.add_argument("--disperse", type=float, default=0.75, help="0=translated_text, 1=translated_text")
    parser.add_argument("--out-dir", type=str, default=os.path.join("..", "data", "mining_ply_pretrains"))
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--use-augment", action='store_true',default=True, help="translated_text 2: translated_text (translated_text ply translated_text)")
    parser.add_argument("--rescale-only", action='store_true', default=False, help="translated_text 3: translated_text (translated_text, translated_text)")
    parser.add_argument("--source-dir", type=str, default=os.path.join(os.path.dirname(__file__), '..', 'data', 'mining_ply_pretrain'), help="translated_text")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1), help="translated_text")
    
    args = parser.parse_args()
    
    if not HAS_SKIMAGE:
        print("Error: scikit-image not installed. pip install scikit-image")
        return

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    # English comment for public release.
    if args.rescale_only and args.use_augment:
         print("translated_text: translated_text --rescale-only translated_text --use-augment, translated_text --rescale-only translated_text")

    print(f"=== translated_text ===")
    
    if args.rescale_only:
        print(f"translated_text: [Mode 3] translated_text (Rescale Only)")
    elif args.use_augment:
        print(f"translated_text: [Mode 2] translated_text (Composition + Deformation)")
    else:
        print(f"translated_text: [Mode 1] translated_text (Random Field Generation)")

    
    # English comment for public release.
    # English comment for public release.
    # English comment for public release.
    # English comment for public release.
    # English comment for public release.
    
    if args.rescale_only:
        if not os.path.exists(args.source_dir):
            print(f"Error: translated_text {args.source_dir} translated_text")
            return
            
        src_files = [f for f in os.listdir(args.source_dir) if f.lower().endswith('.ply')]
        if not src_files:
            print(f"Error: translated_text {args.source_dir} translated_text PLY translated_text")
            return
            
        real_total_cnt = len(src_files)
        args.count = real_total_cnt # English comment for public release.
        
        print(f"translated_text {real_total_cnt} translated_text, translated_text...")
        
        # English comment for public release.
        task_args = []
        for i, fname in enumerate(src_files):
            task_args.append((
                i + 1,        # idx (1-based)
                real_total_cnt,    # total
                args.target_faces, 
                args.temperature, 
                args.disperse, 
                out_dir, 
                args.seed,
                fname         # English comment for public release.
            ))
            
        # English comment for public release.
        # English comment for public release.
        # English comment for public release.
        # English comment for public release.
        
        # English comment for public release.
        # English comment for public release.
        init_args = (args.source_dir, False, False)

    else:
        # English comment for public release.
        # English comment for public release.
        task_args = []
        for i in range(1, args.count + 1):
            task_args.append((
                i, 
                args.count, 
                args.target_faces, 
                args.temperature, 
                args.disperse, 
                out_dir, 
                args.seed,
                args.rescale_only  # Bool: False or True (but only False reaches here logically if we split)
            ))
            
        init_args = (args.source_dir, args.use_augment, args.rescale_only)

    print(f"translated_text: {args.count}")

    
    start_time = time.time()
    
    # English comment for public release.
    max_workers = args.workers
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker_process, initargs=init_args) as executor:
        # English comment for public release.
        futures = {executor.submit(process_single_task, arg): arg[0] for arg in task_args}
        
        finished_cnt = 0
        total_cnt = args.count
        
        try:
            for future in concurrent.futures.as_completed(futures):
                # English comment for public release.
                # original_idx = futures[future]
                
                try:
                    success, msg, ret_idx = future.result()
                    finished_cnt += 1
                    
                    # English comment for public release.
                    progress = finished_cnt / total_cnt * 100
                    elapsed = time.time() - start_time
                    avg_time = elapsed / finished_cnt
                    eta = avg_time * (total_cnt - finished_cnt)
                    
                    status_prefix = f"[{finished_cnt}/{total_cnt} | {progress:.1f}%] "
                    
                    if success:
                        print(f"{status_prefix}{msg}")
                    else:
                        print(f"{status_prefix}ERROR: {msg}")
                        
                except Exception as exc:
                    print(f"translated_text: {exc}")
                
        except KeyboardInterrupt:
            print("\n!!! translated_text, translated_text ...")
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    total_time = time.time() - start_time
    print(f"=== translated_text, translated_text: {total_time:.2f}s ===")

if __name__ == "__main__":
    main()

