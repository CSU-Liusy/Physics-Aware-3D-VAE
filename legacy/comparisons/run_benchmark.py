import os
import sys
import torch
import numpy as np
import datetime
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from scipy.ndimage import label
from scipy.spatial.distance import directed_hausdorff

# Add 3dvae root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from dataset import MiningDataset
from legacy_paths import resolve_output_directory, external_input
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
    """Load an explicit trusted historical checkpoint with its saved architecture."""
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError('Supply an existing trusted legacy VAE checkpoint')
    try:
        payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(payload, dict):
        raise ValueError('Legacy benchmark requires a checkpoint dictionary with hparams')
    if payload.get('format_version') == 2 and 'model_kind' in payload and 'manifest_sha256' in payload:
        raise ValueError('Use the revision workflow for a revision-protocol checkpoint')
    hparams = payload.get('hparams')
    if not isinstance(hparams, dict):
        raise ValueError('Legacy benchmark cannot infer a reliable architecture: checkpoint hparams are missing')
    state = next((payload[k] for k in ('state_dict','model_state_dict','model') if isinstance(payload.get(k),dict)),None)
    if state is None: raise ValueError('Legacy checkpoint has no model state dictionary')
    grid_size = tuple(hparams.get('grid_size',(32,32,32)))
    if grid_size != (32,32,32):
        raise ValueError('This historical benchmark expects a32cubed checkpoint; use an explicit matching evaluation design for other resolutions')
    model, _, _ = create_model(model_type=hparams.get('model_type','standard'), use_lora=hparams.get('use_lora',False), lora_preset=hparams.get('lora_preset','light'), grid_size=grid_size, latent_dim=int(hparams['latent_dim']), base_channels=int(hparams.get('base_channels',32)), num_levels=int(hparams.get('octree_levels',4)), device=device, log_mode='brief')
    state = {key.removeprefix('module.'):value for key,value in state.items()}
    model.load_state_dict(state,strict=True)
    model.eval()
    return model

def execute_benchmark(vae_ckpt=None, unet_ckpt=None, data_dir=None, output_dir=None, limit=10, use_cuda=True, file_list=None):
    if not data_dir or not os.path.isdir(data_dir):
        raise ValueError('Supply an explicit existing external data_dir; no private fallback is searched')
    if not vae_ckpt or not os.path.isfile(vae_ckpt):
        raise ValueError('Supply an explicit existing trusted vae_ckpt')
    if unet_ckpt and not os.path.isfile(unet_ckpt):
        raise FileNotFoundError('Explicit U-Net checkpoint does not exist')
    DEVICE = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    RESULT_DIR = output_dir
    data_dir = str(external_input(data_dir))
    vae_ckpt = str(external_input(vae_ckpt))
    if unet_ckpt: unet_ckpt = str(external_input(unet_ckpt))
         
    RESULT_DIR = str(resolve_output_directory(RESULT_DIR,[data_dir]))
    if any(os.path.exists(os.path.join(RESULT_DIR,name)) for name in ('legacy_protocol.json','benchmark_summary_final.csv','benchmark_execution.json')):
        raise FileExistsError('Choose a new legacy benchmark output directory; existing result records are not overwritten')
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    print(f"Benchmark Results -> {RESULT_DIR}")

    # Read only the external input directory chosen by the caller.
    data_path = data_dir

    print(f"Loading data from {data_path}")
    dataset = MiningDataset(
        ply_dir=data_path,
        num_holes=8,
        samples_per_hole=64, 
        grid_size=(32, 32, 32), 
        augment_per_mesh=0, 
        file_list=file_list,
        force_regen_cache=False,
        cache_dir=os.path.join(RESULT_DIR, 'cache')
    )
    
    # Load Models
    models = {}
    
    # Legacy architecture and all state keys must match explicitly.
    models['Proposed_VAE'] = load_vae(vae_ckpt, DEVICE)

    # 2. U-Net
            
    if unet_ckpt and os.path.exists(unet_ckpt):
        try:
            unet = UNet3D(in_channels=2, out_channels=1).to(DEVICE)
            state = torch.load(unet_ckpt, map_location=DEVICE)
            unet.load_state_dict(state)
            unet.eval()
            models['U-Net'] = unet
            print(f"Loaded U-Net from {unet_ckpt}")
        except Exception as e:
            raise RuntimeError(f"Explicit U-Net checkpoint failed to load: {e}") from e
    else:
        print("No U-Net checkpoint supplied; that optional model is not evaluated.")
    
    with open(os.path.join(RESULT_DIR,'legacy_protocol.json'),'w',encoding='utf-8') as stream:
        json.dump({'scope':'historical_mesh_benchmark_not_revision_evidence','known_differences':['Within-volume whole-grid overlap, not primary blind-region overlap','Squared bidirectional Chamfer sum, not revised unsquared surface mean','Empty distances use the original999penalty','Not the revised parent-disjoint validation/test design']},stream,indent=2)
    model_metrics = {name: {'dice': [], 'iou': [], 'hd': [], 'cd': []} for name in models}
    
    indices = range(min(len(dataset), limit))
    execution = {'scope':'historical_mesh_benchmark_not_revision_evidence',
                 'requested_samples':len(indices), 'successful_evaluations':0,
                 'failures':[], 'status':'running'}
    
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
                except Exception as exc:
                    execution['failures'].append({'sample_index':i,'model':name,
                                                  'error_type':type(exc).__name__,
                                                  'error':str(exc)})
        except Exception as exc:
            execution['failures'].append({'sample_index':i,'model':None,
                                          'error_type':type(exc).__name__,
                                          'error':str(exc)})

    execution['successful_evaluations'] = sum(len(metrics['iou']) for metrics in model_metrics.values())
    execution['status'] = ('failed_no_successful_evaluations' if not execution['successful_evaluations']
                           else 'completed_with_failures' if execution['failures'] else 'complete')
    with open(os.path.join(RESULT_DIR,'benchmark_execution.json'),'w',encoding='utf-8') as stream:
        json.dump(execution,stream,indent=2)
    if not execution['successful_evaluations']:
        raise RuntimeError('Legacy benchmark produced no successful evaluations; see benchmark_execution.json')

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
    parser.add_argument('--ply-file', type=str, default=None, help='Optional exact external PLY selection within data-dir')
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    execute_benchmark(
        vae_ckpt=args.vae_ckpt,
        unet_ckpt=args.unet_ckpt,
        data_dir=args.data_dir,
        limit=args.limit,
        output_dir=args.output_dir,
        file_list=args.ply_file
    )

if __name__ == "__main__":
    main()
