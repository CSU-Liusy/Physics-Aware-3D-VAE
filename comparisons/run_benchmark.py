import os
import sys
import torch
import numpy as np
import datetime
import argparse
from tqdm import tqdm
import pandas as pd
from scipy.ndimage import label
from scipy.spatial.distance import directed_hausdorff

# Add 3dvae root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from dataset import MiningDataset
from model_factory import create_model
from output_result import write_ply_points
from comparisons.models.unet import UNet3D

def calculate_chamfer(pred, target, threshold=0.5):
    """
    Calculate Chamfer Distance between two voxel grids.
    Lower is better.
    """
    pred_pts = np.argwhere(pred > threshold)
    target_pts = np.argwhere(target > 0.5)
    
    if len(pred_pts) == 0 or len(target_pts) == 0:
        return 999.0 # Penalty
        
    try:
        from scipy.spatial import KDTree
        tree_pred = KDTree(pred_pts)
        tree_target = KDTree(target_pts)
        
        d1, _ = tree_target.query(pred_pts, k=1)
        d2, _ = tree_pred.query(target_pts, k=1)
        
        return np.mean(d1**2) + np.mean(d2**2)
    except Exception:
        return 999.0

def calculate_hausdorff(pred, target, threshold=0.5):
    """
    Calculate Hausdorff Distance (95% approximate or max).
    Lower is better.
    """
    pred_pts = np.argwhere(pred > threshold)
    target_pts = np.argwhere(target > 0.5)
    
    if len(pred_pts) == 0 or len(target_pts) == 0:
        return 999.0 # Penalty for empty
        
    d1 = directed_hausdorff(pred_pts, target_pts)[0]
    d2 = directed_hausdorff(target_pts, pred_pts)[0]
    
    return max(d1, d2)

def calculate_metrics(pred, target, threshold=0.5):
    """
    pred: (D, H, W) float [0, 1]
    target: (D, H, W) 0 or 1
    Returns: dice, iou, hd, cd
    """
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
        hd = 999.0
        cd = 999.0
        
    return dice, iou, hd, cd

def load_vae(checkpoint_path, device):
    try:
        # Try calling with keyword args
        model_result = create_model(
            model_type='octree', # Default to octree if proposed is octree, or standard? Try standard for compat? 
            # Actually, we should infer from checkpoint or args. 
            # For robust pipeline, we usually use 'octree' as proposed.
            # But let's try 'standard' first as fallback or check checkpoint.
            # Best way: Try to load state_dict and see if it fits.
            # Or just create default model (which is standard VAE in current factory defaults?).
            # No, factory defaults to standard.
            # Let's try creating Octree VAE first since that's likely the Proposed model.
            grid_size=(32, 32, 32),
            latent_dim=1024, # Updated to match checkpoint often seen
            base_channels=32,
            device=device,
            log_mode='brief'
        )
        if hasattr(model_result, '__len__') and len(model_result) >= 1:
            model = model_result[0]
        else:
            model = model_result
    except Exception:
        print("Model creation fallback...")
        return None
    
    model = model.to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading VAE from {checkpoint_path}")
        try:
            state = torch.load(checkpoint_path, map_location=device)
            if isinstance(state, dict) and 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'], strict=False)
            elif isinstance(state, dict) and 'state_dict' in state:
                 model.load_state_dict(state['state_dict'], strict=False)
            else:
                model.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"Failed to load VAE weights: {e}")
            return None
    
    model.eval()
    return model

def execute_benchmark(vae_ckpt=None, unet_ckpt=None, data_dir=None, output_dir=None, limit=10):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_dir:
         RESULT_DIR = output_dir
    else:
         RESULT_DIR = os.path.join(ROOT_DIR, f'../results/comparisons/{TIMESTAMP}')
         
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    print(f"Benchmark Results -> {RESULT_DIR}")

    # Data
    if data_dir:
        data_path = data_dir
    else:
        data_path = os.path.join(ROOT_DIR, '../data/mining_ply_pretrain')
        if not os.path.exists(data_path):
             data_path = os.path.join(ROOT_DIR, '../data/mining_ply')
    
    print(f"Loading data from {data_path}")
    dataset = MiningDataset(
        ply_dir=data_path,
        num_holes=8,
        samples_per_hole=64, 
        grid_size=(32, 32, 32), 
        augment_per_mesh=0, 
        force_regen_cache=False
    )
    
    # Load Models
    models = {}
    
    # 1. Proposed VAE
    if not vae_ckpt:
        proposed_dir = os.path.join(ROOT_DIR, '../results/auto_analysis/Proposed_Method')
        if os.path.exists(proposed_dir):
            all_pts = []
            for root, _, files in os.walk(proposed_dir):
                for f in files:
                    if f.endswith('.pth') and 'best' in f:
                        all_pts.append(os.path.join(root, f))
            if all_pts:
                vae_ckpt = all_pts[0]

    if vae_ckpt and os.path.exists(vae_ckpt):
        try:
             vae = load_vae(vae_ckpt, DEVICE)
             if vae: models['Proposed_VAE'] = vae
        except Exception as e:
             print(f"VAE Load Error: {e}")
    else:
        print("No VAE checkpoint found.")

    # 2. U-Net
    if not unet_ckpt:
        unet_ckpt = os.path.join(ROOT_DIR, '../results/comparisons/models/unet_best.pth')
            
    if unet_ckpt and os.path.exists(unet_ckpt):
        try:
            unet = UNet3D(in_channels=2, out_channels=1).to(DEVICE)
            state = torch.load(unet_ckpt, map_location=DEVICE)
            unet.load_state_dict(state)
            unet.eval()
            models['U-Net'] = unet
            print(f"Loaded U-Net from {unet_ckpt}")
        except Exception as e:
            print(f"U-Net Load Error: {e}")
    else:
        print("No U-Net checkpoint found.")
    
    model_metrics = {name: {'dice': [], 'iou': [], 'hd': [], 'cd': []} for name in models}
    
    indices = range(min(len(dataset), limit))
    
    for i in tqdm(indices, desc="Benchmarking"):
        try:
            obs, target, _, _, _ = dataset[i]
            obs_tensor = obs.unsqueeze(0).to(DEVICE)
            target_np = target.numpy()
            
            for name, model in models.items():
                try:
                    pred = None
                    if name == 'Proposed_VAE':
                        with torch.no_grad():
                            logits, _, _ = model(obs_tensor)
                            pred = torch.sigmoid(logits).squeeze().cpu().numpy()
                    elif name == 'U-Net':
                        with torch.no_grad():
                            out = model(obs_tensor)
                            pred = out.squeeze().cpu().numpy()
                    
                    if pred is not None:
                        dice, iou, hd, cd = calculate_metrics(pred, target_np, threshold=0.5)
                        model_metrics[name]['dice'].append(dice)
                        model_metrics[name]['iou'].append(iou)
                        model_metrics[name]['hd'].append(hd)
                        model_metrics[name]['cd'].append(cd)
                except Exception:
                    pass
        except Exception:
            continue

    print("\n" + "="*60)
    print("       MODEL COMPARISON RESULTS       ")
    print("="*60)
    print(f"{'Model':<15} | {'IoU':<8} | {'Dice':<8} | {'HD':<8} | {'CD':<8}")
    print("-" * 60)
    
    summary_data = []
    
    for name, metrics in model_metrics.items():
        if not metrics['iou']: continue
        
        m_iou = np.mean(metrics['iou'])
        m_dice = np.mean(metrics['dice'])
        m_hd = np.mean(metrics['hd']) 
        m_cd = np.mean(metrics['cd'])
        
        print(f"{name:<15} | {m_iou:.4f}   | {m_dice:.4f}   | {m_hd:.2f}     | {m_cd:.2f}")
        
        summary_data.append({
            'Model': name,
            'IoU': m_iou,
            'Dice': m_dice,
            'HD': m_hd,
            'CD': m_cd
        })
        
    if summary_data:
        df = pd.DataFrame(summary_data)
        csv_path = os.path.join(RESULT_DIR, 'benchmark_summary_final.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nDetailed results saved to: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive benchmark.')
    parser.add_argument('--vae-ckpt', type=str, default=None)
    parser.add_argument('--unet-ckpt', type=str, default=None)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    execute_benchmark(
        vae_ckpt=args.vae_ckpt,
        unet_ckpt=args.unet_ckpt,
        data_dir=args.data_dir,
        limit=args.limit,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()