import os
import random
import math
import numpy as np
import torch
from torch.utils.data import Dataset
try:
    import scipy.ndimage
except ImportError:
    pass # augment will check availability

# =============================================================================
# English comment for public release.
# English comment for public release.
# English comment for public release.
# English comment for public release.
# English comment for public release.
# English comment for public release.
# English comment for public release.
# =============================================================================

def read_ply(path):
	"""Documentation translated to English for open-source release."""
	if not os.path.exists(path):
		raise FileNotFoundError(f'PLY file not found: {path}')
	
	verts = []
	faces = []
	try:
		with open(path, 'r', encoding='utf-8', errors='ignore') as f:
			line = f.readline()
			if not line.startswith('ply'):
				raise ValueError(f'Not a valid PLY file: {path}')
			header = True
			num_verts = 0
			num_faces = 0
			while header:
				line = f.readline()
				if line.startswith('element vertex'):
					num_verts = int(line.split()[-1])
				if line.startswith('element face'):
					num_faces = int(line.split()[-1])
				if line.strip() == 'end_header':
					header = False
					break
			# read vertices
			for i in range(num_verts):
				parts = f.readline().strip().split()
				if len(parts) < 3:
					continue
				x, y, z = map(float, parts[:3])
				verts.append((x, y, z))
			# read faces
			for i in range(num_faces):
				parts = f.readline().strip().split()
				if len(parts) < 4:
					continue
				cnt = int(parts[0])
				idx = list(map(int, parts[1:1+cnt]))
				if cnt >= 3:
					# triangulate if needed
					for j in range(1, cnt-1):
						faces.append((idx[0], idx[j], idx[j+1]))
	except Exception as e:
		raise RuntimeError(f'Error reading PLY file {path}: {str(e)}')
	
	if len(verts) == 0:
		raise ValueError(f'PLY file {path} contains no vertices')
	if len(faces) == 0:
		raise ValueError(f'PLY file {path} contains no faces')
	
	return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)
	


def point_in_mesh(points, verts, faces):
	"""Documentation translated to English for open-source release."""
	EPS = 1e-9
	P = points.shape[0]
	F = faces.shape[0]
	
	# precompute triangle vertices: (F, 3)
	v0 = verts[faces[:, 0]]  # (F, 3)
	v1 = verts[faces[:, 1]]
	v2 = verts[faces[:, 2]]
	
	# edges
	edge1 = v1 - v0  # (F, 3)
	edge2 = v2 - v0  # (F, 3)
	
	# ray direction (1, 0, 0)
	# h = cross(dir, edge2) = (0, -edge2_z, edge2_y)
	h = np.stack([np.zeros(F, dtype=np.float32), -edge2[:, 2], edge2[:, 1]], axis=1)  # (F, 3)
	
	# det = dot(edge1, h)
	det = np.sum(edge1 * h, axis=1)  # (F,)
	
	# English comment for public release.
	valid_tri = np.abs(det) > EPS
	inv_det = np.where(valid_tri, 1.0 / (det + EPS), 0.0)  # (F,)
	
	# English comment for public release.
	inside = np.zeros(P, dtype=bool)
	
	# English comment for public release.
	batch_size = min(500, P)
	for start in range(0, P, batch_size):
		end = min(start + batch_size, P)
		pts_batch = points[start:end]  # (B, 3)
		B = pts_batch.shape[0]
		
		# s = p - v0: (B, F, 3)
		s = pts_batch[:, None, :] - v0[None, :, :]
		
		# u = inv_det * dot(s, h): (B, F)
		u = inv_det[None, :] * np.sum(s * h[None, :, :], axis=2)
		
		# q = cross(s, edge1): (B, F, 3)
		q = np.cross(s, edge1[None, :, :])
		
		# v_param = inv_det * dot(dir, q) = inv_det * q[:,:,0]
		v_param = inv_det[None, :] * q[:, :, 0]
		
		# t = inv_det * dot(edge2, q): (B, F)
		t = inv_det[None, :] * np.sum(edge2[None, :, :] * q, axis=2)
		
		# English comment for public release.
		hit = (
			valid_tri[None, :] &
			(u >= 0.0) & (u <= 1.0) &
			(v_param >= 0.0) & (u + v_param <= 1.0) &
			(t > EPS)
		)  # (B, F)
		
		# English comment for public release.
		cnt = np.sum(hit, axis=1)  # (B,)
		inside[start:end] = (cnt % 2) == 1
	
	return inside


def _process_file_job(args):
	"""Documentation translated to English for open-source release."""
	(
		path,
		n_samples,
		grid_size,
		num_holes,
		samples_per_hole,
		cache_dir,
		force_regen_cache,
		size_kb,
		median_kb,
	) = args
	
	try:
		verts, faces = read_ply(path)
	except Exception as e:
		print(f"Error reading {path}: {e}")
		return path, None, []

	vmin = verts.min(axis=0)
	vmax = verts.max(axis=0)
	D, H, W = grid_size
	
	# Voxel Cache Logic
	ply_name = os.path.basename(path)
	cache_filename = f"{ply_name}.vox{D}x{H}x{W}.npy"
	cache_path = os.path.join(cache_dir, cache_filename)
	vox = None
	
	if not force_regen_cache and os.path.exists(cache_path):
		try:
			vox = np.load(cache_path)
			if vox.shape != (D, H, W):
				vox = None
		except (IOError, ValueError, OSError) as e:
			print(f"⚠️ translated_text {cache_path} translated_text: {e}")
			vox = None
	
	if vox is None:
		xs = np.linspace(vmin[0], vmax[0], W)
		ys = np.linspace(vmin[1], vmax[1], H)
		zs = np.linspace(vmin[2], vmax[2], D)
		
		gz, gy, gx = np.meshgrid(zs, ys, xs, indexing='ij')
		grid_centers = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
		
		occ = point_in_mesh(grid_centers, verts, faces).astype(np.uint8)
		vox = occ.reshape(D, H, W).astype(np.float32)
		try:
			np.save(cache_path, vox)
		except Exception as e:
			print(f"⚠️ translated_text {cache_path}: {e}")

	# English comment for public release.
	# English comment for public release.
	return path, vox, vmin, vmax, n_samples


class MiningDataset(Dataset):
	"""Documentation translated to English for open-source release."""
	def __init__(self, ply_dir, num_holes=8, samples_per_hole=16, grid_size=(32, 32, 32), 
			  num_samples=None, augment_per_mesh=4, file_list=None, train_frac=0.8, 
			  base_kb_per_sample=100, max_samples_per_file=100, min_samples_per_file=1,
			  force_regen_cache=False, log_mode='full', load_mode='parallel', split_seed=42):
		"""Documentation translated to English for open-source release."""
		self.ply_dir = ply_dir
		self.num_holes = num_holes
		self.samples_per_hole = samples_per_hole
		self.num_points = num_holes * samples_per_hole
		
		if isinstance(grid_size, int):
			self.grid_size = (grid_size, grid_size, grid_size)
		else:
			self.grid_size = tuple(grid_size)
		# English comment for public release.
		# English comment for public release.
		# xs (W), ys (H), zs (D)
		
		# num_samples: if provided (>0) will be used as a target (proportional allocation);
		# if None (default) we compute per-file counts from file sizes in KB using base_kb_per_sample
		self.num_samples = int(num_samples) if (num_samples is not None and int(num_samples) > 0) else None
		self.augment_per_mesh = augment_per_mesh
		self.train_frac = float(train_frac)
		self.base_kb_per_sample = float(base_kb_per_sample)
		self.max_samples_per_file = int(max_samples_per_file)
		self.min_samples_per_file = int(min_samples_per_file)
		self.force_regen_cache = force_regen_cache
		self.is_brief = (log_mode == 'brief')
		self.load_mode = 'parallel' if load_mode not in ('parallel', 'sequential') else load_mode
		self.split_seed = split_seed

		# build list of .ply files to use
		if file_list is not None:
			if isinstance(file_list, (list, tuple)):
				files = list(file_list)
			else:
				if os.path.isdir(file_list):
					files = [os.path.join(file_list, f) for f in os.listdir(file_list) if f.lower().endswith('.ply')]
				elif os.path.isfile(file_list):
					dirp = os.path.dirname(file_list)
					files = [os.path.join(dirp, f) for f in os.listdir(dirp) if f.lower().endswith('.ply')]
				else:
					files = [os.path.join(ply_dir, f) for f in os.listdir(ply_dir) if f.lower().endswith('.ply')]
		else:
			files = [os.path.join(ply_dir, f) for f in os.listdir(ply_dir) if f.lower().endswith('.ply')]

		# canonicalize and verify
		self.files = sorted(list(dict.fromkeys(files)))
		if split_seed is None or split_seed < 0:
			rng = np.random.default_rng()
		else:
			rng = np.random.default_rng(int(split_seed))
		perm = rng.permutation(len(self.files))
		self.files = [self.files[i] for i in perm]
		if len(self.files) == 0:
			raise RuntimeError(f'No .ply files found in {ply_dir}')

		# compute per-file sample counts
		sizes = np.array([os.path.getsize(p) for p in self.files], dtype=np.float64)
		sizes_kb = sizes / 1024.0
		median_kb = float(np.median(sizes_kb)) if len(sizes_kb) > 0 else 1.0
		if self.num_samples is None:
			# automatic per-file allocation based on file size (KB)
			raw_counts = np.round(sizes_kb / max(1.0, self.base_kb_per_sample)).astype(int)
			raw_counts = np.clip(raw_counts, self.min_samples_per_file, self.max_samples_per_file)
			samples_per_file = raw_counts.tolist()
		else:
			# proportional allocation to meet requested total num_samples
			total_size = float(sizes.sum())
			if total_size <= 0:
				raw = np.full(len(self.files), self.num_samples / max(1, len(self.files)))
			else:
				raw = sizes / total_size * float(self.num_samples)
			base = np.floor(raw).astype(int)
			remainder = int(self.num_samples - base.sum())
			if remainder > 0:
				fracs = raw - base
				order = np.argsort(-fracs)
				for i in range(remainder):
					base[order[i]] += 1
			base = np.maximum(base, 1)
			samples_per_file = base.tolist()

		# ---------------------------------------------------------
		# English comment for public release.
		# ---------------------------------------------------------
		# English comment for public release.
		# English comment for public release.
		# English comment for public release.
		# English comment for public release.
		self.vox_cache = []
		self.samples = []
		self.sample_meta = []
		file_sample_counters = {p: 0 for p in self.files}
		
		if self.is_brief:
			print(f'MiningDataset: files={len(self.files)}, samples={sum(samples_per_file)}')
		else:
			print(f'MiningDataset: translated_text {len(self.files)} translated_text .ply translated_text; translated_text(translated_text={sum(samples_per_file)})')
		
		# English comment for public release.
		cache_dir = os.path.join(os.path.dirname(ply_dir), 'cache')
		os.makedirs(cache_dir, exist_ok=True)
		
		# English comment for public release.
		if self.force_regen_cache:
			msg = f'translated_text, translated_text {cache_dir} ...'
			print(msg)
			for f in os.listdir(cache_dir):
				if f.endswith('.npy'):
					try:
						os.remove(os.path.join(cache_dir, f))
					except Exception as e:
						print(f'translated_text {f}: {e}')

		from tqdm import tqdm
		from multiprocessing import Pool, cpu_count
		
		total_samples = sum(samples_per_file)
		
		# English comment for public release.
		args_list = []
		for fi, path in enumerate(self.files):
			args_list.append((
				path, 
				samples_per_file[fi], 
				self.grid_size, 
				self.num_holes, 
				self.samples_per_hole, 
				cache_dir, 
				self.force_regen_cache,
				sizes_kb[fi],
				median_kb,
			))
		
		use_parallel = (self.load_mode == 'parallel')
		num_processes = min(cpu_count(), 64) if use_parallel else 1

		if use_parallel:
			with Pool(processes=num_processes) as pool:
				with tqdm(total=total_samples, desc='translated_text(translated_text)', unit='translated_text', disable=self.is_brief) as pbar:
					for path, vox, vmin, vmax, n_s in pool.imap(_process_file_job, args_list):
						if vox is None:
							continue
						self.vox_cache.append(vox)
						vox_idx = len(self.vox_cache) - 1
						
						# Use deterministic seeding based on split_seed and file path
						current_seed = (self.split_seed + sum(ord(c) for c in path)) if self.split_seed is not None else None
						local_rng = np.random.default_rng(current_seed)
						file_seeds = local_rng.integers(0, 2**32, size=n_s, dtype=np.int64)
						
						for i in range(n_s):
							self.samples.append((vox_idx, vmin.copy(), vmax.copy(), file_seeds[i]))
							self.sample_meta.append((path, file_sample_counters[path]))
							file_sample_counters[path] += 1
						pbar.update(n_s)
		else:
			with tqdm(total=total_samples, desc='translated_text(translated_text)', unit='translated_text', disable=self.is_brief) as pbar:
				for a in args_list:
					path, vox, vmin, vmax, n_s = _process_file_job(a)
					if vox is None:
						continue
					self.vox_cache.append(vox)
					vox_idx = len(self.vox_cache) - 1
					
					# Same deterministic logic for sequential mode
					current_seed = (self.split_seed + sum(ord(c) for c in path)) if self.split_seed is not None else None
					local_rng = np.random.default_rng(current_seed)
					file_seeds = local_rng.integers(0, 2**32, size=n_s, dtype=np.int64)
					
					for i in range(n_s):
						self.samples.append((vox_idx, vmin.copy(), vmax.copy(), file_seeds[i]))
						self.sample_meta.append((path, file_sample_counters[path]))
						file_sample_counters[path] += 1
					pbar.update(n_s)
					del vox

		# final counts and sample-level train/test split
		total = len(self.samples)
		all_indices = np.arange(total)
		split_rng = np.random.default_rng(self.split_seed if self.split_seed is not None else None)
		split_rng.shuffle(all_indices)
		train_cut = int(np.floor(total * self.train_frac))
		train_cut = min(max(train_cut, 1), total - 1) if total > 1 else total
		self.train_indices = all_indices[:train_cut].tolist()
		self.test_indices = all_indices[train_cut:].tolist()
		self.train_sample_count = len(self.train_indices)
		self.test_sample_count = len(self.test_indices)
		self.active_split = 'train'
		self.blacklist = set()
		self.valid_indices = list(self.train_indices)

		# English comment for public release.
		self.samples_per_file = samples_per_file
		self.file_train_counts = {p: 0 for p in self.files}
		self.file_test_counts = {p: 0 for p in self.files}
		for idx in self.train_indices:
			fname, _ = self.sample_meta[idx]
			self.file_train_counts[fname] = self.file_train_counts.get(fname, 0) + 1
		for idx in self.test_indices:
			fname, _ = self.sample_meta[idx]
			self.file_test_counts[fname] = self.file_test_counts.get(fname, 0) + 1

		if self.is_brief:
			print(f'MiningDataset translated_text: total={total}, train={self.train_sample_count}, test={self.test_sample_count} (translated_text)')
		else:
			print(f'MiningDataset translated_text: translated_text={total}, translated_text={self.train_sample_count}, translated_text={self.test_sample_count}(translated_text)')

	def _generate_obs(self, vox, vmin, vmax, seed):
		"""
		On-the-fly generation of observation grid using a deterministic seed.
		"""
		# Use int(seed) to ensure compatible type for RandomState
		rng = np.random.RandomState(int(seed))
		D, H, W = self.grid_size
		obs_grid = np.zeros((2, D, H, W), dtype=np.float32)
		
		# Parameters - ensure we use the instance config
		local_holes = max(1, int(self.num_holes))
		local_samples_per_hole = max(1, int(self.samples_per_hole))
		
		pts = []
		# Pre-calculate ranges
		x_range = vmax[0] - vmin[0]
		y_range = vmax[1] - vmin[1]
		z_range = vmax[2] - vmin[2]
		
		for h in range(local_holes):
			cx = rng.uniform(vmin[0], vmax[0])
			cy = rng.uniform(vmin[1], vmax[1])
			zs_line = np.linspace(vmin[2], vmax[2], local_samples_per_hole)
			
			for s in range(local_samples_per_hole):
				# Add noise
				x = cx + rng.uniform(-0.01, 0.01) * x_range
				y = cy + rng.uniform(-0.01, 0.01) * y_range
				z = zs_line[s] + rng.uniform(-0.002, 0.002) * z_range
				pts.append([x, y, z])
				
		if not pts:
			return obs_grid
			
		pts = np.array(pts, dtype=np.float32)
		
		# Normalize to [0, 1] relative to bounding box
		norm_pts = (pts - vmin) / (vmax - vmin + 1e-8)
		
		# Map to grid coordinates
		nx = norm_pts[:, 0]
		ny = norm_pts[:, 1]
		nz = norm_pts[:, 2]
		
		x_inds = np.clip(np.floor(nx * (W - 1)).astype(int), 0, W-1)
		y_inds = np.clip(np.floor(ny * (H - 1)).astype(int), 0, H-1)
		z_inds = np.clip(np.floor(nz * (D - 1)).astype(int), 0, D-1)
		
		# Sample from vox
		# vox is typically (D, H, W)
		# Note: We must ensure vox labels are correct.
		# Original code: labels = vox[z_inds, y_inds, x_inds]
		labels = vox[z_inds, y_inds, x_inds].astype(np.float32)
		
		# Fill obs grid
		# Channel 0: Value (1 if solid, 0 if empty) - but wait, vox is 1=Solid?
		# Original code: obs_grid[0] = labels
		# Channel 1: Mask (1 where sampled)
		obs_grid[0, z_inds, y_inds, x_inds] = labels
		obs_grid[1, z_inds, y_inds, x_inds] = 1.0
		
		return obs_grid

	def __len__(self):
		return len(self.valid_indices)

	def __getitem__(self, idx):
		if idx < 0:
			idx = len(self.valid_indices) + idx
		if idx < 0 or idx >= len(self.valid_indices):
			raise IndexError('index out of range')

		sample_idx = self.valid_indices[idx]
		# Updated unpacking: (vox_idx, vmin, vmax, seed)
		vox_idx, vmin_np, vmax_np, seed = self.samples[sample_idx]
		
		# English comment for public release.
		vox_np = self.vox_cache[vox_idx]
		
		# on-the-fly generation
		obs_np = self._generate_obs(vox_np, vmin_np, vmax_np, seed)
		
		# ---------------------------------------------------------
		# English comment for public release.
		# ---------------------------------------------------------
		# -----------------------------------------------------------
		# English comment for public release.
		# -----------------------------------------------------------
		if self.active_split == 'train' and self.augment_per_mesh > 0:
			# English comment for public release.
			if random.random() < 0.5:
				# English comment for public release.
				# obs_np: (2, D, H, W), vox_np: (D, H, W)
				# rotate expects (H, W) as last two dims by default for 2D, or axes parameter
				# Scipy rotate: input, angle, axes, reshape=False (keep grid size), order=0 (nearest for masks)
				
				# English comment for public release.
				angle = random.uniform(0, 360)
				# English comment for public release.
				# Dataset format: D, H, W. X corresponds to W, Y to H, Z to D in meshgrid previously?
				# Actually in getitem logic: z_inds, y_inds, x_inds. So dims are (Z, Y, X).
				# Rotating around Z axis means rotating in (Y, X) plane -> axes (1, 2).
				
				# English comment for public release.
				# English comment for public release.
				try:
					# English comment for public release.
					import scipy.ndimage
					
					# English comment for public release.
					# OBS channel 0 is value (0/1), channel 1 is mask (0/1). Order=0 is safe.
					obs_np = scipy.ndimage.rotate(obs_np, angle, axes=(2, 3), reshape=False, order=0, mode='constant', cval=0.0)
					vox_np = scipy.ndimage.rotate(vox_np, angle, axes=(1, 2), reshape=False, order=0, mode='constant', cval=0.0)
					
					# English comment for public release.
					if random.random() < 0.5:
						scale = random.uniform(0.8, 1.2)
						# zoom: input, zoom, order=0
						# Obs zoom: (1, 1, scale, scale) ? assuming isotropic generic scaling or just spatial
						# Let's do isotropic spatial scaling
						# obs: (2, D, H, W) -> zoom (1, scale, scale, scale)
						obs_np = scipy.ndimage.zoom(obs_np, (1, scale, scale, scale), order=0, mode='constant', cval=0.0)
						vox_np = scipy.ndimage.zoom(vox_np, (scale, scale, scale), order=0, mode='constant', cval=0.0)
						
						# Crop or Pad back to original size
						# This is tricky manually. Let's skip complex crop/pad for now and trust 'zoom' to change shape
						# But we need fixed shape for DataLoader...
						# If we zoom, the shape changes. VAE requires fixed input size.
						# So we must Crop Center or Pad Center.
						
						def crop_or_pad(arr, target_shape):
							# arr: (..., D', H', W')
							curr_shape = arr.shape
							# calculate start indices
							slices = []
							for i in range(len(curr_shape)):
								if i < len(curr_shape) - 3: # channel dims
									slices.append(slice(None))
									continue
								
								curr_dim = curr_shape[i]
								tgt_dim = target_shape[i - (len(curr_shape)-3)]
								
								if curr_dim > tgt_dim:
									# Crop
									start = (curr_dim - tgt_dim) // 2
									slices.append(slice(start, start + tgt_dim))
								else:
									# Pad (handled later? no, slice handles crop)
									# If smaller, we slice full and then need to pad.
									slices.append(slice(None))
							
							cropped = arr[tuple(slices)]
							
							# Now Pad if needed
							pad_width = []
							for i in range(len(curr_shape)):
								if i < len(curr_shape) - 3: 
									pad_width.append((0, 0))
									continue
								curr_dim = cropped.shape[i]
								tgt_dim = target_shape[i - (len(curr_shape)-3)]
								if curr_dim < tgt_dim:
									diff = tgt_dim - curr_dim
									pad_width.append((diff//2, diff - diff//2))
								else:
									pad_width.append((0, 0))
									
							if any(p != (0,0) for p in pad_width):
								cropped = np.pad(cropped, pad_width, mode='constant', constant_values=0)
							return cropped

						obs_np = crop_or_pad(obs_np, self.grid_size)
						vox_np = crop_or_pad(vox_np, self.grid_size)

				except ImportError:
					pass # fallback to simple flips

			# English comment for public release.
			# Flip H
			if random.random() < 0.5:
				obs_np = np.ascontiguousarray(np.flip(obs_np, axis=2))
				vox_np = np.ascontiguousarray(np.flip(vox_np, axis=1))
			
			# Flip W
			if random.random() < 0.5:
				obs_np = np.ascontiguousarray(np.flip(obs_np, axis=3))
				vox_np = np.ascontiguousarray(np.flip(vox_np, axis=2))
			
			# English comment for public release.
			if random.random() < 0.5:
				obs_np = np.ascontiguousarray(np.flip(obs_np, axis=1))
				vox_np = np.ascontiguousarray(np.flip(vox_np, axis=0))

		return (
			torch.from_numpy(obs_np),
			torch.from_numpy(vox_np),
			torch.from_numpy(vmin_np),
			torch.from_numpy(vmax_np),
			sample_idx,
		)

	def set_blacklist(self, indices):
		"""Remove noisy sample indices from iteration."""
		self.blacklist = set(indices)
		self.valid_indices = [i for i in (self.train_indices if self.active_split == 'train' else self.test_indices) if i not in self.blacklist]
		if len(self.valid_indices) == 0:
			raise RuntimeError('All samples were blacklisted; dataset empty')

	def set_split(self, split: str):
		"""Documentation translated to English for open-source release."""
		split = split.lower()
		if split not in ('train', 'test', 'all'):
			raise ValueError('split must be train/test/all')
		self.active_split = split
		if split == 'train':
			base_indices = self.train_indices
		elif split == 'test':
			base_indices = self.test_indices
		else:
			base_indices = list(range(len(self.samples)))
		self.valid_indices = [i for i in base_indices if i not in self.blacklist]


if __name__ == '__main__':
    # quick smoke test
    ds = MiningDataset(os.path.join(os.path.dirname(__file__), '..', 'data', 'mining_ply'), log_mode='brief')
    if len(ds) > 0:
        obs, vox, vmin, vmax, idx = ds[0]
        print('obs', obs.shape, 'vox', vox.shape, 'vmin', vmin.numpy() if isinstance(vmin, torch.Tensor) else vmin, 'vmax', vmax.numpy() if isinstance(vmax, torch.Tensor) else vmax)
    else:
        print("Dataset is empty.")
    print('Finish')
