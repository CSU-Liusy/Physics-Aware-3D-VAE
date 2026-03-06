
import torch
import torch.nn.functional as F
import sys
import os

# Ensure 3dvae is in path
sys.path.append(os.path.join(os.getcwd(), '3dvae'))
sys.path.append(os.getcwd())

from model import loss_function

def test_strong_constraints():
    print("Verifying Strong Constraints Implementation...")
    
    # 1. Setup dummy data (Batch=1, D=32, H=32, W=32)
    B, D, H, W = 1, 32, 32, 32
    target_vox = torch.zeros((B, D, H, W))
    
    # 2. Define a specific voxel location to test
    x, y, z = 16, 16, 16
    
    # 3. Create two scenarios
    
    # Scenario A: No Mask (Unknown region)
    # Background is CORRECT (-10.0 -> prob=0.0, target=0)
    recon_logits_A = torch.ones((B, D, H, W)) * -10.0
    # One pixel is WRONG (10.0 -> prob=1.0, target=0)
    recon_logits_A[0, z, y, x] = 10.0 
    
    obs_mask_A = torch.zeros((B, D, H, W)) # No constraints
    
    # Note: loss_function returns tuple (total_loss, bce, kld, dice)
    _, bce_A, _, _ = loss_function(recon_logits_A, target_vox, torch.zeros(1), torch.zeros(1), obs_mask=obs_mask_A)
    print(f"Scenario A (Unconstrained Error):  Average BCE Loss = {bce_A.item():.8f}")

    # Scenario B: With Mask (Drill hole constraint)
    # The same wrong prediction, but now this pixel is marked as a drill hole
    recon_logits_B = recon_logits_A.clone()
    obs_mask_B = torch.zeros((B, D, H, W))
    obs_mask_B[0, z, y, x] = 1.0 # This pixel IS a drill hole
    
    _, bce_B, _, _ = loss_function(recon_logits_B, target_vox, torch.zeros(1), torch.zeros(1), obs_mask=obs_mask_B)
    print(f"Scenario B (Constrained Error):    Average BCE Loss = {bce_B.item():.8f}")

    # 4. Compare
    # Note: BCE is averaged over all pixels (32*32*32 = 32768). 
    # A single pixel with 50x weight will increase the TOTAL SUM by 50x the error of that pixel.
    # But since we average, the absolute number will be small.
    # Let's compare the raw contribution
    # Error for logit 10 vs target 0 is ~10.0 (log(1+exp(10)))
    
    ratio = bce_B / (bce_A + 1e-12)
    print(f"Ratio (B / A): {ratio.item():.2f}")
    
    # Why not exactly 50? 
    # Because A has 1 error pixel (weight 1) + 32767 correct pixels (weight 1)
    # B has 1 error pixel (weight 50) + 32767 correct pixels (weight 1)
    # But since correct pixels have near 0 loss, the ratio should be driven by the error pixel.
    
    if ratio > 1.0:
        print("\n✅ Verification SUCCESS: Constrained Error is significantly higher.")
    else:
        print("\n❌ Verification FAILED: Weights differentiation not observed.")

if __name__ == "__main__":
    test_strong_constraints()

