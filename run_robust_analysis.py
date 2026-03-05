
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

# Imports from project
from dataset import MiningDataset
from model_factory import create_model

def vprint(*args, **kwargs):
    print(*args, **kwargs)

def add_noise_to_obs(obs, noise_level=0.0):
    """
    Simulate inaccuracies by adding noise to drilling positions.
    obs: (B, 2, D, H, W) tensor
    noise_level: float, probability or intensity of noise
    
    Assumption: obs[1] is the mask channel (1 where data exists).
    We can't easily shift voxels for localized noise in a grid without complex resampling.
    A simpler "noise" model for Voxel Grid is random flipping of observed values (0->1 or 1->0) 
    or random addition of false positives/negatives.
    
    If noise_level is probability of bit flip in observed regions:
    """
    if noise_level <= 0:
        return obs
        
    obs_noise = obs.clone()
    # Mask of where we have data
    mask = obs_noise[:, 1, :, :, :] > 0.5
    
    # Values
    values = obs_noise[:, 0, :, :, :]
    
    # Generate random noise mask
    noise_mask = torch.rand_like(values) < noise_level
    
    # Only apply noise where we actually have observations (simulating sensor error)
    # Flip values: 0->1, 1->0
    target_indices = mask & noise_mask
    values[target_indices] = 1.0 - values[target_indices]
    
    obs_noise[:, 0, :, :, :] = values
    return obs_noise

def run_robustness_analysis(args):
    """
    Execute robustness analysis on a pre-trained model.
    Evaluates model performance under:
    1. Sparsity (Reduced number of drill holes)
    2. Noise (Data inaccuracy perturbation)
    """
    print(f"==================================================")
    print(f"启动鲁棒性分析 (Robustness Analysis)")
    print(f"==================================================")

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    
    if not args.checkpoint:
        print("错误: 鲁棒性分析模式必须提供 --checkpoint 参数指向已训练模型。")
        return

    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到模型文件 {checkpoint_path}")
        return

    # Load Model
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config from checkpoint if available, or use args
    # For simplicity, we assume args matches or we rebuild based on args
    # Ideally, we should use hparams from checkpoint
    hparams = checkpoint.get('hparams', {})
    
    # Construct model
    model = create_model(
        model_type=hparams.get('model_type', args.model_type),
        use_lora=hparams.get('use_lora', args.use_lora),
        lora_preset=hparams.get('lora_preset', args.lora_preset),
        grid_size=hparams.get('grid_size', args.grid_size),
        latent_dim=hparams.get('latent_dim', args.latent_dim),
        base_channels=hparams.get('base_channels', args.base_channels),
        num_levels=hparams.get('octree_levels', args.octree_levels),
        device=device
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    model.eval()

    # Define Robustness Experiments
    
    # 1. Sparsity (Num Holes)
    # Default is usually 8. We test [8, 4, 2, 1]
    sparsity_levels = [8, 6, 4, 2, 1]
    
    # 2. Noise (Bit flip probability in observations)
    # 0.0, 0.05, 0.10, 0.20
    noise_levels = [0.0, 0.05, 0.10, 0.20]

    results = []

    # Common Dataset Args
    ply_dir = args.ply_dir
    # Ensure we use test split or a specific file
    
    # --- Experiment A: Sparsity ---
    print("\n--- 实验 A: 数据稀疏性 (Sparsity) ---")
    for holes in sparsity_levels:
        print(f"测试钻孔数量: {holes}")
        
        # Re-initialize dataset with specific hole count
        # Force cache regeneration (?) No, dataset generation logic in __getitem__ 
        # uses self.num_holes. We can modify the dataset instance or re-init.
        # Since we use online generation (lazy loading fix), we can just re-init dataset.
        
        ds = MiningDataset(
            ply_dir=ply_dir,
            num_holes=holes,  # Controlled Variable
            samples_per_hole=args.samples_per_hole,
            grid_size=args.grid_size,
            num_samples=100, # Fixed number of separate test samples
            train_frac=0.0,  # All test
            load_mode='sequential', # Safe
            force_regen_cache=False 
        )
        
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        
        ious = []
        accuracies = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f'Holes={holes}'):
                obs, vox, _, _, _ = batch
                obs = obs.to(device)
                vox = vox.to(device)
                
                # Forward
                logits, _, _ = model(obs)
                probs = torch.sigmoid(logits)
                
                # Metrics
                preds = (probs > 0.5).float()
                gt = vox
                
                intersection = (preds * gt).sum()
                union = preds.sum() + gt.sum() - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                acc = (preds == gt).float().mean()
                
                ious.append(iou.item())
                accuracies.append(acc.item())
        
        avg_iou = np.mean(ious)
        avg_acc = np.mean(accuracies)
        print(f"  -> Holes={holes}: IoU={avg_iou:.4f}, Acc={avg_acc:.4f}")
        
        results.append({
            'Type': 'Sparsity',
            'Level': holes,
            'IoU': avg_iou,
            'Accuracy': avg_acc
        })

    # --- Experiment B: Noise ---
    # Fix holes to standard (e.g., 8) and vary noise
    default_holes = 8
    print(f"\n--- 实验 B: 数据噪声 (Noise, 固定 Holes={default_holes}) ---")
    
    ds_noise = MiningDataset(
        ply_dir=ply_dir,
        num_holes=default_holes,
        samples_per_hole=args.samples_per_hole,
        grid_size=args.grid_size,
        num_samples=100,
        train_frac=0.0,
        load_mode='sequential',
        force_regen_cache=False
    )
    loader_noise = DataLoader(ds_noise, batch_size=1, shuffle=False, num_workers=0)
    
    for nl in noise_levels:
        print(f"测试噪声水平: {nl*100}%")
        ious = []
        
        with torch.no_grad():
            for batch in tqdm(loader_noise, desc=f'Noise={nl}'):
                obs, vox, _, _, _ = batch
                obs = obs.to(device)
                vox = vox.to(device)
                
                # Inject Noise
                obs_noisy = add_noise_to_obs(obs, noise_level=nl)
                
                # Forward
                logits, _, _ = model(obs_noisy)
                probs = torch.sigmoid(logits)
                
                preds = (probs > 0.5).float()
                gt = vox
                
                intersection = (preds * gt).sum()
                union = preds.sum() + gt.sum() - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                ious.append(iou.item())
        
        avg_iou = np.mean(ious)
        print(f"  -> Noise={nl}: IoU={avg_iou:.4f}")
        
        results.append({
            'Type': 'Noise',
            'Level': nl,
            'IoU': avg_iou,
            'Accuracy': 0.0 # Placeholder
        })

    # Save Results
    df = pd.DataFrame(results)
    out_dir = os.path.dirname(checkpoint_path)
    csv_path = os.path.join(out_dir, 'robustness_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n鲁棒性数据已保存: {csv_path}")
    
    # Plotting
    plot_robustness(df, out_dir)

def plot_robustness(df, out_dir):
    try:
        import seaborn as sns
        sns.set(style="whitegrid")
    except ImportError:
        plt.style.use('ggplot')
        
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Sparsity Plot
    sparsity_df = df[df['Type'] == 'Sparsity']
    if not sparsity_df.empty:
        axes[0].plot(sparsity_df['Level'], sparsity_df['IoU'], marker='o', linewidth=2, color='tab:blue')
        axes[0].set_title("稀疏性分析: 钻孔数量 vs IoU")
        axes[0].set_xlabel("钻孔数量 (Number of Holes)")
        axes[0].set_ylabel("Mean IoU")
        axes[0].invert_xaxis() # 8 -> 1
        axes[0].grid(True)
        # annotations
        for x, y in zip(sparsity_df['Level'], sparsity_df['IoU']):
            axes[0].text(x, y, f'{y:.3f}', ha='center', va='bottom')

    # 2. Noise Plot
    noise_df = df[df['Type'] == 'Noise']
    if not noise_df.empty:
        axes[1].plot(noise_df['Level'], noise_df['IoU'], marker='s', linewidth=2, color='tab:red')
        axes[1].set_title("噪声敏感度: 输入噪声 vs IoU")
        axes[1].set_xlabel("噪声水平 (Noise Probability)")
        axes[1].set_ylabel("Mean IoU")
        axes[1].grid(True)
        for x, y in zip(noise_df['Level'], noise_df['IoU']):
            axes[1].text(x, y, f'{y:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'robustness_analysis_plot.png'), dpi=300)
    print(f"鲁棒性图表已生成: {os.path.join(out_dir, 'robustness_analysis_plot.png')}")
