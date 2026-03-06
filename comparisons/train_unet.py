import os
import sys
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime

# Add 3dvae root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import MiningDataset
from comparisons.models.unet import UNet3D

def train_unet(args=None):
    # Configuration
    BATCH_SIZE = getattr(args, 'batch_size', 16) if args else 16
    EPOCHS = getattr(args, 'epochs', 50) if args else 50
    LR = getattr(args, 'lr', 1e-3) if args else 1e-3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args and args.ply_dir:
        DATA_DIR = args.ply_dir
    else:
        DATA_DIR = os.path.join(ROOT_DIR, '../data/mining_ply_pretrain')

    if args and args.output_dir:
        SAVE_DIR = args.output_dir
    else:
        SAVE_DIR = os.path.join(ROOT_DIR, '../results/comparisons/models')
        
    os.makedirs(SAVE_DIR, exist_ok=True)
    SAVE_PATH = os.path.join(SAVE_DIR, 'unet_best.pth')

    print(f"Device: {DEVICE}")
    print(f"Data Dir: {DATA_DIR}")
    print(f"Save Path: {SAVE_PATH}")
    print(f"Epochs: {EPOCHS}")

    # Dataset
    if not os.path.exists(DATA_DIR):
        # Fallback
        default_mining = os.path.join(ROOT_DIR, '../data/mining_ply')
        if os.path.exists(default_mining):
             DATA_DIR = default_mining
             print(f"Warning: switched to {DATA_DIR}")
        else:
             print(f"Error: Data directory {DATA_DIR} not found.")
             return

    dataset = MiningDataset(
        ply_dir=DATA_DIR,
        num_holes=8,
        samples_per_hole=16,
        grid_size=(32, 32, 32), # Match VAE default (was 64 which caused cache miss)
        augment_per_mesh=4, 
        force_regen_cache=False
    )
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    # Model
    model = UNet3D(in_channels=2, out_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss() 

    best_loss = float('inf')

    # Train Loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"UNet Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                obs, vox, _, _, _ = batch
                obs = obs.to(DEVICE)
                vox = vox.to(DEVICE).unsqueeze(1).float() # (B, 1, D, H, W)
                
                optimizer.zero_grad()
                output = model(obs)
                
                loss = criterion(output, vox)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
            
        if len(loader) > 0:
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"Saved best model (Loss {best_loss:.4f})")
        else:
            print("Empty loader?")

    print("UNet Training Complete.")
    return SAVE_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ply-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()
    
    train_unet(args)

