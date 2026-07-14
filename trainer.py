"""
trainer.py - 3D VAE training main program (resource-optimized)

Main features:
- Execute training and evaluation (params from config_loader via YAML+CLI).
- Training loop (BCE + KLD + Dice) with dynamic gradient clipping & KL cyclical annealing.
- TensorBoard logging & checkpoint saving; generates report at training end.
"""

import os
import datetime
import time
import shutil
import torch
import traceback
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from concurrent.futures import ThreadPoolExecutor, wait
from collections import defaultdict

from dataset import MiningDataset, read_ply, point_in_mesh
from model import loss_function, vox_to_pointcloud
from model_factory import create_model, print_recommendations
from output_result import (
    generate_epoch_outputs,
    write_ply_points,
    generate_sample_report,
    save_combined_scene,
    save_downsampled_points,
    ensure_cn_font,
)
from showresult import export_spin


def render_outputs_to_vis(
    outputs_dir: str,
    vis_root: str,
    frames: int = 60,
    elev: float = 35.0,
    dpi: int = 600,
    workers: int = None,
):
    """Batch-render PLY files under outputs_dir as GIFs (no PNG retention), output to vis_root/timestamp dir, with multi-threaded acceleration."""
    if not os.path.isdir(outputs_dir):
        return None

    ply_list = [os.path.join(outputs_dir, f) for f in os.listdir(outputs_dir) if f.lower().endswith('.ply')]
    if not ply_list:
        return None

    # Take timestamp dir name (parent of outputs_dir) and build vis subdirectory
    ts_dir = os.path.basename(os.path.normpath(os.path.dirname(outputs_dir)))
    vis_dir = os.path.join(vis_root, ts_dir)
    os.makedirs(vis_dir, exist_ok=True)

    print(f'Found {len(ply_list)} PLY files; starting visualization...')

    def _job(ply_path: str):
        try:
            verts, faces = read_ply(ply_path)
            if faces.ndim != 2 or faces.shape[1] != 3:
                return None
            base_name = os.path.splitext(os.path.basename(ply_path))[0]
            export_spin(verts, faces, vis_dir, base_name, frames=frames, elev=elev, dpi=dpi, save_png=False)
        except Exception:
            return None
        return None

    if workers is None or workers <= 0:
        workers = os.cpu_count() or 4
    max_workers = max(1, min(workers, len(ply_list)))
    sorted_list = sorted(ply_list)

    if max_workers == 1:
        for _ in tqdm(sorted_list, desc='Vis', leave=False):
            _job(_)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(tqdm(ex.map(_job, sorted_list), total=len(sorted_list), desc='Vis', leave=False))

    return vis_dir


def train(args):
    loaded_state_dict = None
    loaded_hparams = None
    # Optional: load hyperparams & weights from checkpoint
    if getattr(args, 'checkpoint', None):
        ckpt_path = os.path.abspath(args.checkpoint)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Specified checkpoint not found: {ckpt_path}')
        try:
            # Explicitly set weights_only=False to support full model / complex dicts, compatible with older PyTorch
            try:
                obj = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            except TypeError:
                obj = torch.load(ckpt_path, map_location='cpu')

            if isinstance(obj, dict):
                if 'hparams' in obj and isinstance(obj['hparams'], dict):
                    loaded_hparams = obj['hparams']
                # Try reading the epoch saved in checkpoint
                _ckpt_saved_epoch = obj['epoch'] if isinstance(obj.get('epoch'), int) else 0
                # Common naming conventions
                for key in ['state_dict', 'model_state_dict', 'model']:
                    if key in obj and isinstance(obj[key], dict):
                        loaded_state_dict = obj[key]
                        break
                # If the entire obj is a state_dict
                if loaded_state_dict is None and all(isinstance(k, str) for k in obj.keys()):
                    loaded_state_dict = obj
            elif isinstance(obj, torch.nn.Module):
                loaded_state_dict = obj.state_dict()
                _ckpt_saved_epoch = 0

            # Auto-strip 'module.' prefix from DataParallel to prevent weight loading failures
            if loaded_state_dict is not None:
                new_state_dict = {}
                has_module_prefix = False
                for k, v in loaded_state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                        has_module_prefix = True
                    else:
                        new_state_dict[k] = v
                if has_module_prefix:
                    vprint("Detected 'module.' prefix; auto-stripped to match current model structure.")
                    loaded_state_dict = new_state_dict

        except Exception as e:
            raise RuntimeError(f'Failed to load checkpoint: {ckpt_path} ({e})')

        if loaded_hparams:
            # Override current params with checkpoint hyperparams
            for k, v in loaded_hparams.items():
                if hasattr(args, k):
                    setattr(args, k, v)
            vprint(f'Initialization params overridden by checkpoint hyperparams: {ckpt_path}')
        elif loaded_state_dict is not None:
            # When checkpoint lacks hparams, infer key hyperparams (at least latent_dim) from weights
            inferred_latent_dim = None
            if 'fc_mu.bias' in loaded_state_dict and getattr(loaded_state_dict['fc_mu.bias'], 'ndim', 0) == 1:
                inferred_latent_dim = int(loaded_state_dict['fc_mu.bias'].shape[0])
            elif 'fc_mu.weight' in loaded_state_dict and getattr(loaded_state_dict['fc_mu.weight'], 'ndim', 0) == 2:
                inferred_latent_dim = int(loaded_state_dict['fc_mu.weight'].shape[0])

            if inferred_latent_dim is not None and hasattr(args, 'latent_dim') and int(args.latent_dim) != inferred_latent_dim:
                print(f"Detected checkpoint latent_dim={inferred_latent_dim}; overriding current latent_dim={args.latent_dim}")
                args.latent_dim = inferred_latent_dim
        # If checkpoint recorded epoch count and user did not set start_epoch, auto-set offset
        if _ckpt_saved_epoch > 0 and getattr(args, 'start_epoch', 0) == 0:
            args.start_epoch = _ckpt_saved_epoch
            print(f"Detected checkpoint trained through epoch {_ckpt_saved_epoch}; KL annealing / Curriculum Learning auto-continues from epoch {_ckpt_saved_epoch + 1}")
    # Basic parameter validation
    if args.epochs <= 0:
        raise ValueError('--epochs must be > 0')
    if args.max_batch_size <= 0:
        raise ValueError('--max-batch-size must be > 0')

    # Normalize grid_size
    if isinstance(args.grid_size, int):
        args.grid_size = [args.grid_size] * 3
    elif len(args.grid_size) == 1:
        args.grid_size = [args.grid_size[0]] * 3
    elif len(args.grid_size) != 3:
        raise ValueError('--grid-size must provide 1 or 3 integers')
    if any(d <= 0 or d % 16 != 0 for d in args.grid_size):
        raise ValueError(f'--grid-size: each dimension must be positive and a multiple of 16 (current: {args.grid_size})')

    if args.latent_dim <= 0:
        raise ValueError('--latent-dim must be > 0')

    # Log mode (verbosity takes precedence over log_mode for backward compatibility)
    log_mode = getattr(args, 'verbosity', None) or getattr(args, 'log_mode', 'full')
    args.log_mode = log_mode
    is_brief = log_mode == 'brief'
    def vprint(*a, **k):
        if not is_brief:
            print(*a, **k)

    # Device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    vprint(f'Using device: {device}')
    if device.type == 'cuda':
        vprint(f'GPU: {torch.cuda.get_device_name(0)}')
        vprint(f'Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

    ply_dir = os.path.abspath(args.ply_dir)
    target_file = os.path.abspath(args.ply_file) if args.ply_file else None

    # Output directory
    if hasattr(args, 'output_dir') and args.output_dir:
         base_out = os.path.abspath(args.output_dir)
         ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') # Still needed for logs filename potentially
    else:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        base_out = os.path.join(os.path.dirname(__file__), '..', 'results', '3dvae', ts)
        
    checkpoints_dir = os.path.join(base_out, 'checkpoints')
    outputs_dir = os.path.join(base_out, 'outputs')
    logs_dir = os.path.join(base_out, 'logs')
    model_root_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'model')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(model_root_dir, exist_ok=True)
    vprint(f'Output directory: {base_out}')

    writer = SummaryWriter(log_dir=logs_dir)
    history = {'loss': [], 'bce': [], 'kld': [], 'dice': []}

    # Generate-only mode
    if args.generate_only:
        if target_file is None or not os.path.exists(target_file):
            raise RuntimeError('No valid --ply-file provided')
        vprint(f'Generating {args.num_samples} virtual drilling samples from {target_file}...')
        verts, faces = read_ply(target_file)
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        for si in tqdm(range(args.num_samples), desc='Generating drill samples'):
            pts = []
            for h in range(args.num_holes):
                cx = float(np.random.uniform(vmin[0], vmax[0]))
                cy = float(np.random.uniform(vmin[1], vmax[1]))
                zs_line = np.linspace(vmin[2], vmax[2], args.samples_per_hole)
                for s in range(args.samples_per_hole):
                    x = cx + np.random.uniform(-0.01, 0.01) * (vmax[0] - vmin[0])
                    y = cy + np.random.uniform(-0.01, 0.01) * (vmax[1] - vmin[1])
                    z = zs_line[s] + np.random.uniform(-0.002, 0.002) * (vmax[2] - vmin[2])
                    pts.append((x, y, z))
            pts = np.array(pts, dtype=np.float32)
            labels = point_in_mesh(pts, verts, faces).astype(np.uint8)
            colors = np.zeros((pts.shape[0], 3), dtype=np.uint8)
            colors[labels == 1] = [255, 0, 0]
            colors[labels == 0] = [0, 0, 255]
            outpath = os.path.join(outputs_dir, f'drill_sample_{si+1:02d}.ply')
            write_ply_points(outpath, pts, colors=colors)
        vprint(f'Drill samples saved to: {outputs_dir}')
        if not args.train_after_generate:
            return

    # Load dataset
    vprint('----- Loading data -----')
    try:
        ds = MiningDataset(
            ply_dir,
            num_holes=args.num_holes,
            samples_per_hole=args.samples_per_hole,
            grid_size=args.grid_size,
            num_samples=(args.num_samples if args.num_samples and args.num_samples > 0 else None),
            augment_per_mesh=args.augment,
            file_list=target_file,
            train_frac=args.train_frac,
            base_kb_per_sample=args.base_kb_per_sample,
            max_samples_per_file=args.max_samples_per_file,
            min_samples_per_file=args.min_samples_per_file,
            force_regen_cache=args.force_regen_cache,
            log_mode=args.log_mode,
            load_mode=args.load_mode,
            split_seed=args.split_seed,
        )
    except Exception as e:
        print(f'Data loading failed: {e}')
        raise

    # Record train/test sample list (sample-level random split)
    split_txt = os.path.join(base_out, 'train_test_split.txt')
    try:
        with open(split_txt, 'w', encoding='utf-8') as f:
            f.write(f"Training samples: {getattr(ds, 'train_sample_count', 0)}\n")
            for p in getattr(ds, 'files', []):
                cnt = getattr(ds, 'file_train_counts', {}).get(p, 0)
                if cnt > 0:
                    f.write(f"  - {os.path.basename(p)}: {cnt}\n")
            f.write(f"\nTesting samples: {getattr(ds, 'test_sample_count', 0)}\n")
            for p in getattr(ds, 'files', []):
                cnt = getattr(ds, 'file_test_counts', {}).get(p, 0)
                if cnt > 0:
                    f.write(f"  - {os.path.basename(p)}: {cnt}\n")
        
    except Exception as e:
        vprint(f'Failed to write train/test split: {e}')

    # DataLoader base config
    num_workers = args.num_workers if hasattr(args, 'num_workers') else 0

    # VRAM estimation: measure single-sample VRAM from model structure, then compute batch size from available VRAM
    mem_frac = min(max(args.batch_to_mem, 0.0), 1.0)
    
    # 2. Compute usable VRAM for current device
    if device.type == 'cuda':
        # For GPU: get total VRAM minus reserved
        current_device_id = torch.cuda.current_device()
        # total_mb = torch.cuda.get_device_properties(current_device_id).total_memory / (1024**2)
        # reserved_mb = torch.cuda.memory_reserved(current_device_id) / (1024**2)
        # free_bytes = torch.cuda.get_device_properties(current_device_id).total_memory - torch.cuda.memory_reserved(current_device_id)
        # mem_get_info is more accurate: free_mem, total_mem
        free_bytes, total_bytes = torch.cuda.mem_get_info(current_device_id)
        
        available_memo_bytes = free_bytes
        # Multi-GPU: only compute for current card. DataParallel would need more complex logic,
        # but current code is mainly single-card / primary-card oriented.
    else:
        # CPU: assume a large value or query system memory
        available_memo_bytes = 16 * (1024**3)  # 16 GB default
        total_bytes = 32 * (1024**3)

    # 3. Compute allocated VRAM
    allocated_memory_bytes = available_memo_bytes * mem_frac
    
    vprint(f"[VRAM Mgmt] Total device VRAM: {total_bytes/1024**3:.2f} GB, currently available: {available_memo_bytes/1024**3:.2f} GB")
    vprint(f"[VRAM Mgmt] Target occupancy ratio: {mem_frac:.2f}, planned allocation: {allocated_memory_bytes/1024**3:.2f} GB")

    def measure_single_batch_usage(grid_size):
        """Measure actual VRAM usage increment at batch_size=1"""
        if device.type != 'cuda':
            return 50 * 1024 * 1024  # CPU estimate: 50 MB
            
        # Create a temporary model to measure VRAM
        try:
            temp_model, _, _ = create_model(
                model_type=args.model_type,
                use_lora=args.use_lora,
                lora_preset=args.lora_preset,
                grid_size=tuple(grid_size),
                latent_dim=args.latent_dim,
                base_channels=args.base_channels,
                num_levels=args.octree_levels,
                lr=args.lr,
                device=device,
                log_mode='brief'  # Suppress verbose info
            )
            temp_model.train()  # Train mode to include gradient graph footprint
            
            # Build dummy input
            # obs: (B, 2, D, H, W)
            dummy_obs = torch.randn(1, 2, grid_size[0], grid_size[1], grid_size[2], device=device)
            dummy_vox = torch.randn(1, grid_size[0], grid_size[1], grid_size[2], device=device)
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base_mem = torch.cuda.memory_allocated()
            
            # Forward
            with torch.amp.autocast('cuda', enabled=args.amp):
                 logits, mu, logvar = temp_model(dummy_obs)
                 loss, _, _, _ = loss_function(logits, dummy_vox, mu, logvar, obs_mask=dummy_obs[:, 1], lambda_drill=args.lambda_drill)
            
            # Backward
            loss.backward()
            
            peak_mem = torch.cuda.max_memory_allocated()
            used_mem = peak_mem - base_mem
            
            # Cleanup
            del temp_model, dummy_obs, dummy_vox, mu, logvar, logits, loss
            torch.cuda.empty_cache()
            
            return used_mem
            
        except Exception as e:
            vprint(f"[VRAM Mgmt] Failed to measure single-batch VRAM: {e}; falling back to estimation")
            return None

    # 4. Execute measurement
    single_batch_bytes = measure_single_batch_usage(args.grid_size)
    
    if single_batch_bytes is None:
        # Measurement fallback: use a conservative estimate
        # For safety when falling back, use a conservative 500 MB value
        single_batch_bytes = 500 * 1024 * 1024 
    
    vprint(f"[VRAM Mgmt] Single-sample (BS=1) VRAM cost: {single_batch_bytes/1024**2:.2f} MB")

    # 4. Check if single batch exceeds limit
    if single_batch_bytes > allocated_memory_bytes:
        raise RuntimeError(
            f"Single-sample VRAM requirement ({single_batch_bytes/1024**3:.2f} GB) exceeds allocated limit ({allocated_memory_bytes/1024**3:.2f} GB)! "
            f"Please reduce --grid-size, shrink the network, or increase --batch-to-mem."
        )

    # 5. Compute optimal batch size
    # In practice VRAM is not just a linear multiple; it includes model weights (constant) + optimizer state (linear) + activations (linear).
    # The single_batch_bytes measured above mainly captures the dynamic portion (activations + gradients).
    # The static model portion was already allocated before base_mem in measure_single_batch_usage.
    # Simple linear estimate: Max_BS * Single_Mem <= Allocated
    # Be conservative, factor 0.95
    
    calc_bs = int(allocated_memory_bytes // single_batch_bytes)
    
    # 6. Safety guard
    final_bs = min(calc_bs, args.max_batch_size)
    
    # At least 1
    final_bs = max(1, final_bs)
    
    vprint(f"[VRAM Mgmt] Theoretical max BS: {calc_bs}, limit ceiling: {args.max_batch_size}")
    vprint(f"[VRAM Mgmt] Final batch size: {final_bs}")
    
    chosen_batch_size = final_bs
    estimated_batch_size = chosen_batch_size

    def build_loader(batch_size: int, pin: bool = True):
        drop_last = len(ds) >= batch_size
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == 'cuda' and pin),
            drop_last=drop_last,
        )

    def build_loader_safely(initial_bs: int):
        bs = initial_bs
        last_err = None
        # Try pin_memory=True; if OOM repeatedly, switch to False and retry
        for pin_flag in (True, False):
            bs = initial_bs
            while bs >= 1:
                try:
                    loader = build_loader(bs, pin=pin_flag)
                    fixed_batch = next(iter(loader))  # Verify one batch can be fetched, exposing OOM early
                    return bs, loader, fixed_batch
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        last_err = e
                        if device.type == 'cuda':
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                        new_bs = bs // 2
                        print(f"BS {bs} {('pin' if pin_flag else 'no-pin')} OOM, trying to halve to {new_bs}")
                        bs = new_bs
                        continue
                    raise
        raise RuntimeError(f'Cannot build DataLoader (minimum batch still OOM): {last_err}')

    # Build and verify loader
    chosen_batch_size, loader, fixed_batch = build_loader_safely(chosen_batch_size)
    if chosen_batch_size != estimated_batch_size:
        vprint(f'Actual BS {chosen_batch_size} (auto-reduced due to OOM)')
    data_grid_size = tuple(fixed_batch[0].shape[2:])
    if tuple(args.grid_size) != data_grid_size:
        args.grid_size = list(data_grid_size)

    if args.show_recommendations:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if device.type == 'cuda' else 8
        print_recommendations(len(ds), gpu_memory, args.grid_size, log_mode=args.log_mode)

    model, optimizer, model_info = create_model(
        model_type=args.model_type,
        use_lora=args.use_lora,
        lora_preset=args.lora_preset,
        grid_size=args.grid_size,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        num_levels=args.octree_levels,
        lr=args.lr,
        weight_decay=1e-4, # Increased L2 Regularization
        device=device,
        log_mode=args.log_mode,
    )

    # If checkpoint provided, load its weights
    if loaded_state_dict is not None:
        model_state = model.state_dict()
        compatible_state = {}
        skipped_shape_mismatch = []
        for k, v in loaded_state_dict.items():
            if k in model_state and model_state[k].shape == v.shape:
                compatible_state[k] = v
            elif k in model_state:
                skipped_shape_mismatch.append(k)

        missing, unexpected = model.load_state_dict(compatible_state, strict=False)
        print(
            f"Loaded checkpoint weights: loaded={len(compatible_state)}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"shape_mismatch_skipped={len(skipped_shape_mismatch)}"
        )
        if len(missing) > 0:
            print(f'Warning: {len(missing)} keys not found in checkpoint (possibly new layers or structural changes)')
            if len(missing) > 10:
                print(f'Example missing keys: {missing[:5]} ...')
        if len(skipped_shape_mismatch) > 0:
            print(f'Warning: {len(skipped_shape_mismatch)} keys skipped due to shape mismatch')
            if len(skipped_shape_mismatch) > 10:
                print(f'Example shape-mismatched keys: {skipped_shape_mismatch[:5]} ...')
    
    vprint(f'-----Data loading complete----- Total samples={len(ds)}  BS={chosen_batch_size}')





    # Optimizer scheduler
    if isinstance(optimizer, dict):
        opt_enc = optimizer['encoder']
        opt_dec = optimizer['decoder']
        scheduler_enc = optim.lr_scheduler.ReduceLROnPlateau(
            opt_enc, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        scheduler_dec = optim.lr_scheduler.ReduceLROnPlateau(
            opt_dec, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        using_multi_optimizer = True
    else:
        opt_enc = optimizer
        opt_dec = None
        scheduler_enc = optim.lr_scheduler.ReduceLROnPlateau(
            opt_enc, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        scheduler_dec = None
        using_multi_optimizer = False

    vprint('-----Starting training-----')
    # AMP training support
    use_amp = args.amp
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    log_file = open(os.path.join(logs_dir, 'train_log.txt'), 'w', encoding='utf-8')
    best_loss = float('inf')
    best_bce = float('inf')   # Used for early stopping (unaffected by beta scheduling)
    patience_counter = 0
    max_patience = args.patience
    grad_norm_ema = None
    noisy_indices = set()
    spike_factor = 3.0
    recent_klds = []
    _start_epoch = getattr(args, 'start_epoch', 0)
    _total_epochs = _start_epoch + args.epochs

    epoch_iter = tqdm(
        range(1, args.epochs + 1),
        desc='Epoch',
        total=args.epochs,
        disable=not is_brief,
        leave=False,
    )
    for epoch in epoch_iter:
        epoch_start = time.time()
        global_epoch = _start_epoch + epoch  # Global epoch accounting for resume offset

        # Beta-VAE logic
        max_beta = 2.0 # Beta > 1 encourages disentanglement

        cycle_idx = (global_epoch - 1) % args.kl_cycle
        cycle_progress = cycle_idx / args.kl_cycle
        if cycle_progress < args.kl_ratio:
            beta = (cycle_progress / args.kl_ratio) * max_beta
        else:
            beta = max_beta

        # KL cycle boundary: reset patience & best_bce baseline to avoid false early-stop from rising total loss due to beta ramp-up
        is_new_kl_cycle = (cycle_idx == 0) and (global_epoch > 1)
        if is_new_kl_cycle:
            patience_counter = 0
            best_bce = float('inf')  # New cycle with fresh baseline
            vprint(f'[KL cycle reset] Epoch {global_epoch} enters new KL annealing cycle; patience & BCE baseline reset')

        model.train()
        running_loss = 0.0
        running_bce = 0.0
        running_kld = 0.0
        running_dice = 0.0
        valid_batches = 0

        # === Dynamic constraint weight strategy (Curriculum Learning) ===
        # First 20% epochs: use base weight (1.0), let model freely learn global shape
        # Remaining 80% epochs: linearly increase weight to 50.0, enforcing drill-hole detail fitting
        # Note: use global_epoch and _total_epochs so curriculum progress is not reset on resume
        start_forcing_epoch = int(_total_epochs * 0.2)
        max_constraint_weight = args.lambda_drill

        if global_epoch <= start_forcing_epoch:
             current_constraint_w = 1.0
        else:
             progress = (global_epoch - start_forcing_epoch) / max(1, _total_epochs - start_forcing_epoch)
             # Linear transition from 1.0 to 50.0
             current_constraint_w = 1.0 + progress * (max_constraint_weight - 1.0)

        current_lr = opt_enc.param_groups[0]['lr']
        pbar = tqdm(loader, desc=f'Epoch {epoch}/{args.epochs} [beta={beta:.2f} cw={current_constraint_w:.1f} lr={current_lr:.1e}]', leave=False, disable=is_brief)
        for batch_idx, batch in enumerate(pbar):
            try:
                obs, vox, vmin_t, vmax_t, idx_batch = batch
                obs = obs.to(device, non_blocking=True)
                vox = vox.to(device, non_blocking=True)

                # --- 2. Pseudo-sparsity (Denoising VAE) ---
                # Randomly mask out 30%-60% of input drill holes, forcing the model to hallucinate
                # obs: (B, 2, D, H, W). Channel 1 is mask.
                if model.training:
                    # Create random drop mask
                    drop_prob = torch.rand(1, device=obs.device).item() * 0.3 + 0.3 # 0.3 ~ 0.6
                    
                    # Only dropout from existing mask (obs[:, 1] > 0)
                    real_mask = obs[:, 1] > 0.5
                    
                    # Generate random keep matrix (keep probability = 1 - drop_prob)
                    keep_mask = torch.rand_like(real_mask, dtype=torch.float32) > drop_prob
                    
                    # Apply mask: if dropped, set channel 1 to 0, channel 0 to 0 (the latter doesn't matter since mask=0)
                    # Should obs be cloned? Dataloader yields can usually be modified directly, but clone for safety
                    obs = obs.clone() 
                    
                    # Update mask channel
                    new_mask = real_mask & keep_mask
                    obs[:, 1] = new_mask.float()
                    # Update value channel (zero out masked positions to prevent leakage)
                    obs[:, 0] = obs[:, 0] * new_mask.float()


                with torch.amp.autocast('cuda', enabled=args.amp):
                    logits, mu, logvar = model(obs)
                    # Use channel 1 (mask) for weighted constraint
                    obs_mask = obs[:, 1]
                    loss, bce, kld, dice = loss_function(
                        logits, vox, mu, logvar,
                        beta=beta,
                        obs_mask=obs_mask,
                        lambda_drill=current_constraint_w
                    )

                kld_val = kld.item()

                recent_klds.append(kld_val)
                if len(recent_klds) > 100:
                    recent_klds.pop(0)
                median_kld = float(np.median(recent_klds)) if recent_klds else kld_val
                if len(recent_klds) > 10 and kld_val > spike_factor * max(1e-6, median_kld):
                    noisy_indices.update(idx_batch.cpu().tolist())
                    # Silently skip anomalous batch to avoid noisy training logs
                    continue

                if torch.isnan(loss) or torch.isinf(loss):
                    vprint(f'Batch {batch_idx} has abnormal loss (NaN/Inf), skipping')
                    continue

                if using_multi_optimizer:
                    opt_enc.zero_grad(set_to_none=True)
                    opt_dec.zero_grad(set_to_none=True)
                else:
                    opt_enc.zero_grad(set_to_none=True)
                if args.amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if grad_norm_ema is None:
                    curr_max_norm = 1e4
                else:
                    curr_max_norm = max(1.0, grad_norm_ema * 2.0)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=curr_max_norm)
                if torch.isfinite(total_norm):
                    norm_val = total_norm.item()
                    if grad_norm_ema is None:
                        grad_norm_ema = norm_val
                    else:
                        grad_norm_ema = 0.9 * grad_norm_ema + 0.1 * norm_val

                if args.amp:
                    scaler.step(opt_enc)
                    if using_multi_optimizer:
                        scaler.step(opt_dec)
                    scaler.update()
                else:
                    opt_enc.step()
                    if using_multi_optimizer:
                        opt_dec.step()

                running_loss += loss.item()
                running_bce += bce.item()
                running_kld += kld.item()
                running_dice += dice.item()
                valid_batches += 1

                pbar.set_postfix({'loss': f'{loss.item():.2f}', 'dice': f'{dice.item():.2f}'})

                del obs, vox, mu, logvar, logits, loss, bce, kld, dice
            except Exception as e:
                vprint(f'❌ Batch {batch_idx} processing failed: {e}')
                vprint(traceback.format_exc())
                continue

        if valid_batches > 0:
            avg_loss = running_loss / valid_batches
            avg_bce = running_bce / valid_batches
            avg_kld = running_kld / valid_batches
            avg_dice = running_dice / valid_batches
        else:
            avg_loss = float('inf')
            avg_bce = float('inf')
            avg_kld = float('inf')
            avg_dice = float('inf')
            print(f'Epoch {epoch}: no valid batches')

        writer.add_scalar('Train/Total_Loss', avg_loss, global_epoch)
        writer.add_scalar('Train/BCE_Loss', avg_bce, global_epoch)
        writer.add_scalar('Train/KLD_Loss', avg_kld, global_epoch)
        writer.add_scalar('Train/Dice_Loss', avg_dice, global_epoch)
        writer.add_scalar('Train/Beta', beta, global_epoch)
        if grad_norm_ema is not None:
            writer.add_scalar('Train/Grad_Norm_EMA', grad_norm_ema, epoch)
        history['loss'].append(avg_loss)
        history['bce'].append(avg_bce)
        history['kld'].append(avg_kld)
        history['dice'].append(avg_dice)

        scheduler_enc.step(avg_loss)
        if using_multi_optimizer and scheduler_dec is not None:
            scheduler_dec.step(avg_loss)

        current_lr_enc = opt_enc.param_groups[0]['lr']
        writer.add_scalar('Train/LR', current_lr_enc, global_epoch)
        if using_multi_optimizer and opt_dec is not None:
            current_lr_dec = opt_dec.param_groups[0]['lr']
            writer.add_scalar('Train/LR_Decoder', current_lr_dec, global_epoch)

        # Update total loss record (for saving best_model.pth)
        if avg_loss < best_loss:
            best_loss = avg_loss
        # Use BCE for early stopping: BCE measures pure reconstruction quality, unaffected by dynamic beta scheduling, avoiding false triggers at new KL cycles
        if avg_bce < best_bce:
            best_bce = avg_bce
            patience_counter = 0
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': global_epoch,
                'hparams': vars(args),
            }, os.path.join(checkpoints_dir, 'best_model.pth'))
        else:
            if global_epoch > args.patience_start:
                patience_counter += 1
            else:
                patience_counter = 0

        patience_str = f"{patience_counter}/{max_patience}"
        if global_epoch <= args.patience_start:
            patience_str += "(Wait)"

        epoch_elapsed = time.time() - epoch_start
        log_msg = (
            f'Epoch {global_epoch} time={epoch_elapsed:.1f}s avg_loss={avg_loss:.4f} '
            f'BCE={avg_bce:.2f}(best={best_bce:.2f}) KL={avg_kld:.2f} β={beta:.2f} [patience: {patience_str}]'
        )
        vprint(log_msg)
        log_file.write(log_msg + '\n')
        log_file.flush()

        if noisy_indices:
            try:
                ds.set_blacklist(noisy_indices)
                noisy_log = os.path.join(base_out, 'noisy_batches.txt')
                with open(noisy_log, 'a', encoding='utf-8') as f:
                    f.write(f'Epoch {epoch} spike batches (count={len(noisy_indices)}): {sorted(noisy_indices)}\n')
                chosen_batch_size, loader, fixed_batch = build_loader_safely(chosen_batch_size)
                noisy_indices.clear()
                recent_klds.clear()
            except Exception as e:
                print(f'Failed to handle noisy samples: {e}')

        if patience_counter >= max_patience:
            print(f'{max_patience}No improvement for')
            log_file.write(f'Early stopping at epoch {epoch}\n')
            break

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': global_epoch,
                'hparams': vars(args),
            }, os.path.join(checkpoints_dir, f'model_epoch_{global_epoch}.pth'))

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    writer.close()

    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Total Loss')
    plt.plot(history['bce'], label='BCE Loss')
    plt.plot(history['kld'], label='KLD Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(logs_dir, 'loss_curve.png'))
    plt.close()
    print(f'Loss curve saved to: {os.path.join(logs_dir, "loss_curve.png")}')

    log_file.close()

    # Final evaluation
    print("-----Generating final full evaluation results-----")
    best_model_path = os.path.join(checkpoints_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        print(f"Loading best model: {best_model_path}")
        try:
            payload = torch.load(best_model_path, map_location='cpu', weights_only=True)
        except TypeError:
            # Compatible with older torch lacking weights_only param
            payload = torch.load(best_model_path, map_location='cpu')

        if isinstance(payload, dict) and 'state_dict' in payload and isinstance(payload['state_dict'], dict):
            state_to_load = payload['state_dict']
        elif isinstance(payload, dict) and 'model_state_dict' in payload and isinstance(payload['model_state_dict'], dict):
            state_to_load = payload['model_state_dict']
        elif isinstance(payload, dict) and all(isinstance(k, str) for k in payload.keys()):
            # Case: entire payload is a state_dict
            state_to_load = payload
        elif isinstance(payload, torch.nn.Module):
            state_to_load = payload.state_dict()
        else:
            raise RuntimeError('best_model.pth format unrecognized; no loadable state_dict found')

        model.load_state_dict(state_to_load, strict=False)
        # Copy best model to global model directory
        try:
            global_best_path = os.path.join(model_root_dir, f'3dvae_best_{ts}.pth')
            shutil.copyfile(best_model_path, global_best_path)
            print(f'Best model copied to global directory: {global_best_path}')
        except Exception as e:
            print(f'Cannot copy best model to global directory: {e}')
    else:
        print("Best model not found; using current model weights")

    model.eval()
    metrics_records = []
    output_summaries = []  # Collect statistics for output samples
    eval_threshold = 0.3
    max_outputs_per_file = getattr(args, 'max_output_per_file', -1)
    output_workers = max(1, getattr(args, 'output_workers', 1))
    file_output_counts = defaultdict(int)
    executor = ThreadPoolExecutor(max_workers=output_workers) if output_workers > 1 else None
    futures = []

    def compute_metrics(recon_prob, gt):
        recon_bin = recon_prob >= eval_threshold
        gt_bin = gt >= eval_threshold
        tp = np.logical_and(recon_bin, gt_bin).sum()
        fp = np.logical_and(recon_bin, ~gt_bin).sum()
        fn = np.logical_and(~recon_bin, gt_bin).sum()
        tn = np.logical_and(~recon_bin, ~gt_bin).sum()
        eps = 1e-8
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        iou = tp / (tp + fp + fn + eps)
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        acc = (tp + tn) / (tp + tn + fp + fn + eps)
        return {
            'precision': float(precision),
            'recall': float(recall),
            'iou': float(iou),
            'dice': float(dice),
            'accuracy': float(acc),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
        }

    def extract_drill_traces(obs_tensor, vmin_arr, vmax_arr):
        """Recover drill points from observation mask; return per-point trace list."""
        if obs_tensor.ndim == 4:  # (C, D, H, W)
            mask = obs_tensor[1] if obs_tensor.shape[0] > 1 else obs_tensor[0]
        else:
            return []

        pts_idx = np.argwhere(mask > 0.5)  # (N, 3) -> z, y, x
        if pts_idx.size == 0:
            return []

        D, H, W = mask.shape
        z_idx, y_idx, x_idx = pts_idx[:, 0], pts_idx[:, 1], pts_idx[:, 2]
        wx = vmin_arr[0] + (x_idx.astype(np.float32) / max(1, W - 1)) * (vmax_arr[0] - vmin_arr[0])
        wy = vmin_arr[1] + (y_idx.astype(np.float32) / max(1, H - 1)) * (vmax_arr[1] - vmin_arr[1])
        wz = vmin_arr[2] + (z_idx.astype(np.float32) / max(1, D - 1)) * (vmax_arr[2] - vmin_arr[2])
        pts_world = np.stack([wx, wy, wz], axis=1)
        # Each point as a separate trace to avoid spurious connections
        return [[tuple(p)] for p in pts_world.tolist()]

    def eval_split(split_name):
        ds.set_split(split_name)
        loader_eval = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == 'cuda'),
            drop_last=False,
        )
        local_records = []
        pbar_eval = tqdm(loader_eval, desc=f'Eval-{split_name}', disable=is_brief)
        for batch in pbar_eval:
            try:
                obs, vox, vmin_t, vmax_t, idx_batch = batch
                obs = obs.to(device, non_blocking=True)
                vox = vox.to(device, non_blocking=True)
                vmin_np = vmin_t[0].numpy()
                vmax_np = vmax_t[0].numpy()
                sample_idx = int(idx_batch[0].item())
                fname, sample_id = ds.sample_meta[sample_idx]
                base_name = os.path.splitext(os.path.basename(fname))[0]
                name_prefix = f"{base_name}_s{sample_id:04d}_{split_name}"

                # Recover original mesh for original.ply & combined visualization output
                orig_verts = orig_faces = None
                try:
                    orig_verts, orig_faces = read_ply(fname)
                except Exception:
                    orig_verts = orig_faces = None

                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=args.amp and device.type == 'cuda'):
                        logits, mu, logvar = model(obs)
                    recon_probs = torch.sigmoid(logits).cpu().numpy()[0]
                gt_vox = vox.cpu().numpy()[0]
                obs_np = obs.cpu().numpy()[0]
                drill_traces = extract_drill_traces(obs_np, vmin_np, vmax_np)

                # Compute drill position accuracy
                # obs_np[1] is the mask channel, indicating drill/known data positions
                drill_mask = (obs_np[1] > 0.5)
                if drill_mask.sum() > 0:
                    recon_bin_drill = (recon_probs[drill_mask] >= eval_threshold)
                    gt_bin_drill = (gt_vox[drill_mask] >= eval_threshold)
                    drill_acc = np.mean(recon_bin_drill == gt_bin_drill)
                else:
                    drill_acc = 1.0  # Default perfect match when no drill data

                metrics = compute_metrics(recon_probs, gt_vox)
                local_records.append({
                    'file': base_name,
                    'sample_id': sample_id,
                    'split': split_name,
                    **metrics,
                })

                should_output = True
                if max_outputs_per_file >= 0 and file_output_counts[base_name] >= max_outputs_per_file:
                    should_output = False

                if should_output:
                    file_output_counts[base_name] += 1

                    # Record statistics (one-to-one with output files)
                    total_pts = int(recon_probs.size)
                    gt_pts = int(metrics['tp'] + metrics['fn'])
                    tp = int(metrics['tp'])
                    fp = int(metrics['fp'])
                    fn = int(metrics['fn'])
                    iou = float(metrics['iou'])
                    output_summaries.append(f"{name_prefix},{total_pts},{gt_pts},{tp},{fp},{fn},{iou:.4f},{drill_acc:.4f}")

                    def _output_job(recon_arr, gt_arr, vmin_arr, vmax_arr, prefix, drill_points, orig_v, orig_f):
                        generate_sample_report(
                            recon_arr,
                            gt_arr,
                            vmin_arr,
                            vmax_arr,
                            outputs_dir,
                            prefix,
                            threshold=eval_threshold,
                            orig_verts=orig_v,
                            orig_faces=orig_f,
                        )

                        save_combined_scene(
                            os.path.join(outputs_dir, f'{prefix}_combined.ply'),
                            recon_arr,
                            gt_arr,
                            drill_points,
                            vmin_arr,
                            vmax_arr,
                            threshold=eval_threshold,
                            orig_verts=orig_v,
                            orig_faces=orig_f,
                        )

                        if args.max_output_points != 0:
                            save_downsampled_points(
                                os.path.join(outputs_dir, f'{prefix}_simplified.ply'),
                                recon_arr,
                                vmin_arr,
                                vmax_arr,
                                threshold=eval_threshold,
                                max_points=args.max_output_points,
                            )

                    if executor is not None:
                        futures.append(
                            executor.submit(
                                _output_job,
                                recon_probs.copy(),
                                gt_vox.copy(),
                                vmin_np.copy(),
                                vmax_np.copy(),
                                name_prefix,
                                drill_traces,
                                orig_verts.copy() if isinstance(orig_verts, np.ndarray) else orig_verts,
                                orig_faces.copy() if isinstance(orig_faces, np.ndarray) else orig_faces,
                            )
                        )
                    else:
                        _output_job(recon_probs, gt_vox, vmin_np, vmax_np, name_prefix, drill_traces, orig_verts, orig_faces)

            except Exception as e:
                print(f"Processing sample {split_name}:{sample_idx} failed: {e}")
        return local_records

    metrics_records.extend(eval_split('train'))
    metrics_records.extend(eval_split('test'))

    if executor is not None and futures:
        wait(futures)
        executor.shutdown(wait=True)

    # Write output statistics summary (only for samples with generated outputs)
    if output_summaries:
        stats_path = os.path.join(base_out, 'output_stats.csv')
        try:
            with open(stats_path, 'w', encoding='utf-8') as f:
                f.write('name,total_points,gt_points,correct_gt(tp),over_pred(fp),miss(fn),iou,drill_acc\n')
                for line in output_summaries:
                    f.write(line + '\n')
            print(f'Output statistics written to: {stats_path}')
        except Exception as e:
            print(f'Failed to write output statistics: {e}')

    if metrics_records:
        metrics_csv = os.path.join(base_out, 'metrics_summary.csv')
        try:
            with open(metrics_csv, 'w', encoding='utf-8') as f:
                f.write('file,sample_id,split,dice,iou,precision,recall,accuracy,tp,fp,fn,tn\n')
                for m in metrics_records:
                    f.write(
                        f"{m['file']},{m['sample_id']},{m['split']},{m['dice']:.4f},{m['iou']:.4f},{m['precision']:.4f},{m['recall']:.4f},{m['accuracy']:.4f},{m['tp']},{m['fp']},{m['fn']},{m['tn']}\n"
                    )
            print(f'Prediction metrics written to: {metrics_csv}')
        except Exception as e:
            print(f'Failed to save metrics CSV: {e}')

        try:
            ensure_cn_font()
            idx = np.arange(len(metrics_records))
            plt.figure(figsize=(12, 6))
            for key, label in [
                ('dice', 'Dice'),
                ('iou', 'IoU'),
                ('precision', 'Precision'),
                ('recall', 'Recall'),
                ('accuracy', 'Accuracy'),
            ]:
                plt.plot(idx, [m[key] for m in metrics_records], marker='o', label=label)
            plt.xticks(idx, [f"{m['file']}-s{m['sample_id']:04d} ({'T' if m['split']=='train' else 'E'})" for m in metrics_records], rotation=45, ha='right')
            plt.ylim(0, 1.05)
            plt.ylabel('Score')
            plt.title('Prediction Accuracy Metrics Curve (per Sample)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            acc_plot_path = os.path.join(base_out, 'accuracy_multiline.png')
            plt.savefig(acc_plot_path, dpi=150)
            plt.close()
            print(f'Prediction metrics curve saved to: {acc_plot_path}')
        except Exception as e:
            print(f'Failed to save prediction metrics plot: {e}')

    # After training, generate visualization GIF/PNG to vis/timestamp dir
    try:
        # Release GPU after training/eval to free VRAM for rendering stage
        try:
            model.cpu()
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        if getattr(args, 'skip_vis', False):
            print('Configured to skip visualization (--skip-vis)')
        else:
            vis_root = os.path.join(os.path.dirname(__file__), '..', 'results', 'vis')
            # Use more CPU threads to accelerate GIF rendering (default: available cores)
            vis_dir = render_outputs_to_vis(outputs_dir, vis_root, frames=60, elev=35.0, dpi=600, workers=os.cpu_count())
            if vis_dir is None:
                print('No visualization generated: output PLY not found or directory missing')
            else:
                print(f'Spin visualization saved to: {vis_dir}')
    except Exception as e:
        print(f'Visualization generation failed: {e}')

    print(f'-----Training & evaluation complete----- Results saved to: {base_out}')


