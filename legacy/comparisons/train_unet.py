import os
import sys
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
import json

# Add 3dvae root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import MiningDataset
from legacy_paths import external_input, resolve_output_directory
from comparisons.models.unet import UNet3D

def train_unet(args=None):
    # Configuration
    BATCH_SIZE = getattr(args, 'batch_size', 16) if args else 16
    EPOCHS = getattr(args, 'epochs', 50) if args else 50
    LR = getattr(args, 'lr', 1e-3) if args else 1e-3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Explicit external inputs/outputs; no fallback into an author data tree.
    if args is None or not getattr(args,'ply_dir',None) or not getattr(args,'output_dir',None):
        raise ValueError('Legacy U-Net training requires external ply_dir and output_dir')
    DATA_DIR = str(external_input(args.ply_dir))
    if not os.path.isdir(DATA_DIR): raise FileNotFoundError(DATA_DIR)
    SAVE_DIR = str(resolve_output_directory(args.output_dir,[DATA_DIR]))

    os.makedirs(SAVE_DIR, exist_ok=True)
    SAVE_PATH = os.path.join(SAVE_DIR, 'unet_best.pth')
    EXECUTION_PATH = os.path.join(SAVE_DIR, 'unet_execution.json')
    if os.path.exists(SAVE_PATH) or os.path.exists(EXECUTION_PATH):
        raise FileExistsError('Choose a new U-Net output directory; existing checkpoints/records are not overwritten')

    print(f"Device: {DEVICE}")
    print(f"Data Dir: {DATA_DIR}")
    print(f"Save Path: {SAVE_PATH}")
    print(f"Epochs: {EPOCHS}")

    # Dataset inputs remain read-only.
    dataset = MiningDataset(
        ply_dir=DATA_DIR,
        num_holes=8,
        samples_per_hole=16,
        grid_size=(32, 32, 32), # Match VAE default (was 64 which caused cache miss)
        augment_per_mesh=4, 
        force_regen_cache=False,
        cache_dir=os.path.join(SAVE_DIR, 'cache')
    )
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    # Model
    model = UNet3D(in_channels=2, out_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss() 

    best_loss = float('inf')
    execution = {'scope':'historical_unet_training_not_revision_evidence','epochs':[],
                 'status':'running'}

    # Train Loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        successful_batches = 0
        failures = []
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
                successful_batches += 1
                pbar.set_postfix({'loss': loss.item()})
            except Exception as e:
                print(f"Error in batch: {e}")
                failures.append({'batch_index':batch_idx,'error_type':type(e).__name__,'error':str(e)})
                continue
            
        epoch_record = {'epoch':epoch+1,'successful_batches':successful_batches,
                        'failures':failures}
        execution['epochs'].append(epoch_record)
        if successful_batches:
            avg_loss = epoch_loss / successful_batches
            epoch_record['mean_successful_batch_loss'] = avg_loss
            print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"Saved best model (Loss {best_loss:.4f})")
        else:
            execution['status'] = 'failed_no_successful_batches'
            with open(EXECUTION_PATH,'w',encoding='utf-8') as stream:
                json.dump(execution,stream,indent=2)
            raise RuntimeError('U-Net epoch had no successful batches; no best checkpoint was selected for that epoch')

    if not execution['epochs']:
        execution['status'] = 'failed_no_epochs'
        with open(EXECUTION_PATH,'w',encoding='utf-8') as stream:
            json.dump(execution,stream,indent=2)
        raise RuntimeError('U-Net requires at least one completed training epoch')
    execution['status'] = 'completed_with_failures' if any(item['failures'] for item in execution['epochs']) else 'complete'
    with open(EXECUTION_PATH,'w',encoding='utf-8') as stream:
        json.dump(execution,stream,indent=2)

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

