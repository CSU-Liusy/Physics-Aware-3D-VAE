"""U-Net loss must work with the same autocast setup as formal training."""
import pytest
import torch

from comparisons.models.unet import UNet3D
from repro_pipeline import compute_loss, deterministic_probability


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_unet_logits_loss_autocast_backward_step(device_type):
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this host")
    device = torch.device(device_type)
    torch.manual_seed(17)
    model = UNet3D(in_channels=2, out_channels=1, init_features=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device_type == "cuda")
    obs = torch.rand(1, 2, 16, 16, 16, device=device)
    obs[:, 1] = (obs[:, 1] > 0.75).float()
    target = (torch.rand(1, 16, 16, 16, device=device) > 0.7).float()

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device_type,
                        dtype=torch.float16 if device_type == "cuda" else torch.bfloat16):
        loss, components = compute_loss(
            "unet", model, obs, target, {"lambda_observed": 5.0}, beta=0.0)
    assert bool(torch.isfinite(loss))
    assert all(value >= 0 for value in components.values())
    # Formal training divides by eight microbatches before backward.
    scaler.scale(loss / 8).backward()
    scaler.unscale_(optimizer)
    assert all(bool(torch.isfinite(param.grad).all())
               for param in model.parameters() if param.grad is not None)
    scaler.step(optimizer)
    scaler.update()

    model.eval()
    with torch.no_grad():
        logits = model(obs, return_logits=True).squeeze(1)
        probabilities = deterministic_probability("unet", model, obs)
    assert torch.allclose(probabilities, torch.sigmoid(logits), atol=1e-6, rtol=1e-6)
