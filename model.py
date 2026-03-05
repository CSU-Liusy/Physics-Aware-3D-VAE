import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
# English comment for public release.
# English comment for public release.
# English comment for public release.


def _safe_gn_groups(channels: int, max_groups: int = 8) -> int:
    """Return a GroupNorm group count that divides channels."""
    g = min(max_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return max(1, g)
# English comment for public release.
# English comment for public release.
# English comment for public release.
# =============================================================================

class SELayer3D(nn.Module):
    """
    Squeeze-and-Excitation Block for 3D.
    """
    def __init__(self, channel, reduction=16):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock3D(nn.Module):
    """Documentation translated to English for open-source release."""
    def __init__(self, channels, groups=8):
        super().__init__()
        gn_groups = _safe_gn_groups(channels, groups)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(gn_groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(gn_groups, channels)
        self.se = SELayer3D(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.se(out)
        out += residual
        out = self.act(out)
        return out


class UpsampleConv3D(nn.Module):
    """Documentation translated to English for open-source release."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)



# ============================================================================
# GeoFormer Block (Transformer for 3D Geology)
# ============================================================================

class GeoTransformerBlock(nn.Module):
    """Documentation translated to English for open-source release."""
    def __init__(self, channels, num_heads=4, ff_dim_scale=2, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(channels)
        # Multi-head Attention
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True, dropout=dropout)
        
        self.norm2 = nn.LayerNorm(channels)
        # Feed Forward
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * ff_dim_scale),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * ff_dim_scale, channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Input x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        # Flatten spatial dims: (B, C, S) where S = D*H*W
        x_flat = x.view(B, C, -1).permute(0, 2, 1) # (B, S, C)
        
        # 1. Attention with Residual
        res = x_flat
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = res + attn_out
        
        # 2. Feed Forward with Residual
        res = x_flat
        x_norm = self.norm2(x_flat)
        ff_out = self.ff(x_norm)
        x_flat = res + ff_out
        
        # Reshape back to 3D: (B, C, D, H, W)
        return x_flat.permute(0, 2, 1).view(B, C, D, H, W)


class ConvVAE3D(nn.Module):
    """Documentation translated to English for open-source release."""
    def __init__(self, in_channels=2, grid_size=(32, 32, 32), latent_dim=256, base_channels=32, use_transformer=True):
        super().__init__()
        self.in_channels = in_channels
        if isinstance(grid_size, int):
            self.grid_size = (grid_size, grid_size, grid_size)
        else:
            self.grid_size = tuple(grid_size)
            
        self.latent_dim = latent_dim
        self.base = base_channels
        self.use_transformer = use_transformer

        # --- Encoder ---
        self.enc_conv_in = nn.Conv3d(in_channels, self.base, kernel_size=3, padding=1, bias=False)
        self.enc_bn_in = nn.GroupNorm(_safe_gn_groups(self.base), self.base)
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout3d(p=0.2) # Dropout for regularization

        # Layer 1: D -> D/2
        self.enc_res1 = ResidualBlock3D(self.base)
        self.enc_down1 = nn.Conv3d(self.base, self.base*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc_bn1 = nn.GroupNorm(_safe_gn_groups(self.base*2), self.base*2)

        # Layer 2: D/2 -> D/4
        self.enc_res2 = ResidualBlock3D(self.base*2)
        self.enc_down2 = nn.Conv3d(self.base*2, self.base*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc_bn2 = nn.GroupNorm(_safe_gn_groups(self.base*4), self.base*4)

        # Layer 3: D/4 -> D/8
        self.enc_res3 = ResidualBlock3D(self.base*4)
        self.enc_down3 = nn.Conv3d(self.base*4, self.base*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc_bn3 = nn.GroupNorm(_safe_gn_groups(self.base*8), self.base*8)

        # Layer 4: D/8 -> D/16
        self.enc_res4 = ResidualBlock3D(self.base*8)
        self.enc_down4 = nn.Conv3d(self.base*8, self.base*16, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc_bn4 = nn.GroupNorm(_safe_gn_groups(self.base*16), self.base*16)
        
        # Bottleneck: Residual + Optional Transformer
        self.enc_res_final = ResidualBlock3D(self.base*16)
        if self.use_transformer:
            # English comment for public release.
            self.enc_transformer = GeoTransformerBlock(self.base*16, num_heads=4)
        
        # Flatten
        s_d = max(1, self.grid_size[0] // 16)
        s_h = max(1, self.grid_size[1] // 16)
        s_w = max(1, self.grid_size[2] // 16)
        
        self._enc_out_shape = (self.base*16, s_d, s_h, s_w)
        enc_feats = self._enc_out_shape[0] * self._enc_out_shape[1] * self._enc_out_shape[2] * self._enc_out_shape[3]

        self.fc_mu = nn.Linear(enc_feats, latent_dim)
        self.fc_logvar = nn.Linear(enc_feats, latent_dim)

        # --- Decoder ---
        self.fc_dec = nn.Linear(latent_dim, enc_feats)
        
        # Bottleneck Decoder
        if self.use_transformer:
             self.dec_transformer = GeoTransformerBlock(self.base*16, num_heads=4)
        
        # Layer 4: D/16 -> D/8
        self.dec_res4 = ResidualBlock3D(self.base*16)
        # Replace ConvTranspose3d with UpsampleConv3D
        self.dec_up4 = UpsampleConv3D(self.base*16, self.base*8, kernel_size=3, padding=1)
        self.dec_bn4 = nn.GroupNorm(_safe_gn_groups(self.base*8), self.base*8)

        # Layer 3: D/8 -> D/4
        self.dec_res3 = ResidualBlock3D(self.base*8)
        self.dec_up3 = UpsampleConv3D(self.base*8, self.base*4, kernel_size=3, padding=1)
        self.dec_bn3 = nn.GroupNorm(_safe_gn_groups(self.base*4), self.base*4)

        # Layer 2: D/4 -> D/2
        self.dec_res2 = ResidualBlock3D(self.base*4)
        self.dec_up2 = UpsampleConv3D(self.base*4, self.base*2, kernel_size=3, padding=1)
        self.dec_bn2 = nn.GroupNorm(_safe_gn_groups(self.base*2), self.base*2)

        # Layer 1: D/2 -> D
        self.dec_res1 = ResidualBlock3D(self.base*2)
        self.dec_up1 = UpsampleConv3D(self.base*2, self.base, kernel_size=3, padding=1)
        self.dec_bn1 = nn.GroupNorm(_safe_gn_groups(self.base), self.base)

        # Final output
        self.dec_res_out = ResidualBlock3D(self.base)
        self.dec_final = nn.Conv3d(self.base, 1, kernel_size=3, padding=1)

    def encode(self, x):
        h = self.act(self.enc_bn_in(self.enc_conv_in(x)))
        h = self.drop(h) # Apply Dropout
        
        h = self.enc_res1(h)
        h = self.act(self.enc_bn1(self.enc_down1(h)))
        
        h = self.enc_res2(h)
        h = self.act(self.enc_bn2(self.enc_down2(h)))
        
        h = self.enc_res3(h)
        h = self.act(self.enc_bn3(self.enc_down3(h)))
        
        h = self.enc_res4(h)
        h = self.act(self.enc_bn4(self.enc_down4(h)))
        
        h = self.enc_res_final(h)
        if self.use_transformer:
            h = self.enc_transformer(h)
        
        B = h.shape[0]
        h_flat = h.view(B, -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        B = z.shape[0]
        h = self.fc_dec(z)
        h = h.view(B, *self._enc_out_shape)

        if self.use_transformer:
            h = self.dec_transformer(h)
        
        h = self.dec_res4(h)
        h = self.act(self.dec_bn4(self.dec_up4(h)))
        
        h = self.dec_res3(h)
        h = self.act(self.dec_bn3(self.dec_up3(h)))
        
        h = self.dec_res2(h)
        h = self.act(self.dec_bn2(self.dec_up2(h)))
        
        h = self.dec_res1(h)
        h = self.act(self.dec_bn1(self.dec_up1(h)))
        
        h = self.dec_res_out(h)
        out = self.dec_final(h)
        return out.squeeze(1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar


# ============================================================================
# English comment for public release.
# ============================================================================


class SparseVoxelEncoder(nn.Module):
    """Documentation translated to English for open-source release."""

    def __init__(self, in_channels=2, base_channels=32, num_levels=4):
        super().__init__()
        self.num_levels = num_levels
        self.base = base_channels
        self.sparse_convs = nn.ModuleList([
            nn.Conv3d(
                in_channels if i == 0 else base_channels * (2 ** (i - 1)),
                base_channels * (2 ** i),
                kernel_size=3,
                padding=1,
                bias=False,
            )
            for i in range(num_levels)
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(_safe_gn_groups(base_channels * (2 ** i)), base_channels * (2 ** i))
            for i in range(num_levels)
        ])
        self.act = nn.SiLU(inplace=True)

    def forward(self, x, sparse_mask=None):
        features = []
        masks = []
        if sparse_mask is None:
            sparse_mask = (x[:, 1:2, :, :, :] > 0).float()

        h = x
        current_mask = sparse_mask
        for level in range(self.num_levels):
            h = self.sparse_convs[level](h)
            h = self.act(self.norms[level](h))
            features.append(h)
            masks.append(current_mask)
            if level < self.num_levels - 1:
                h = F.avg_pool3d(h, kernel_size=2, stride=2)
                current_mask = F.max_pool3d(current_mask, kernel_size=2, stride=2)
        return features, masks


class SparseVoxelDecoder(nn.Module):
    """Documentation translated to English for open-source release."""

    def __init__(self, latent_dim=512, base_channels=32, num_levels=4, grid_size=(64, 64, 64)):
        super().__init__()
        self.num_levels = num_levels
        self.base = base_channels
        self.grid_size = grid_size
        self.min_res = tuple(g // (2 ** (num_levels - 1)) for g in grid_size)
        self.fc = nn.Linear(latent_dim, base_channels * (2 ** (num_levels - 1)) * np.prod(self.min_res))
        self.up_convs = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                nn.Conv3d(
                    base_channels * (2 ** (num_levels - i))
                    + (base_channels * (2 ** (num_levels - i - 1)) if i > 0 else 0),
                    base_channels * (2 ** (num_levels - i - 1)),
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
            )
            for i in range(1, num_levels)
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(
                _safe_gn_groups(base_channels * (2 ** (num_levels - i - 1))),
                base_channels * (2 ** (num_levels - i - 1)),
            )
            for i in range(1, num_levels)
        ])
        self.final_conv = nn.Conv3d(base_channels, 1, kernel_size=3, padding=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, z, encoder_features=None):
        batch = z.shape[0]
        h = self.fc(z).view(batch, -1, *self.min_res)
        for i, (up_conv, norm) in enumerate(zip(self.up_convs, self.norms)):
            # Decompose sequential to inject skip connection after upsampling
            upsampler = up_conv[0]
            conv = up_conv[1]
            h = upsampler(h)

            if encoder_features is not None and i < len(encoder_features):
                skip = encoder_features[-(i + 2)]
                # Ensure spatial dimensions match before concatenation
                if h.shape[2:] != skip.shape[2:]:
                    h = F.interpolate(h, size=skip.shape[2:], mode='trilinear', align_corners=True)
                h = torch.cat([h, skip], dim=1)
            
            h = self.act(norm(conv(h)))
        return self.final_conv(h).squeeze(1)


class OctreeVAE3D(nn.Module):
    """Documentation translated to English for open-source release."""

    def __init__(self, in_channels=2, grid_size=(64, 64, 64), latent_dim=512, base_channels=32, num_levels=4):
        super().__init__()
        self.in_channels = in_channels
        self.grid_size = tuple(grid_size) if isinstance(grid_size, (list, tuple)) else (grid_size,) * 3
        self.latent_dim = latent_dim
        self.base = base_channels
        self.num_levels = num_levels
        self.encoder = SparseVoxelEncoder(in_channels, base_channels, num_levels)
        # English comment for public release.
        enc_res = tuple(max(1, g // (2 ** (num_levels - 1))) for g in self.grid_size)
        enc_channels = base_channels * (2 ** (num_levels - 1))
        enc_flat = enc_channels * np.prod(enc_res)
        self.fc_mu = nn.Linear(enc_flat, latent_dim)
        self.fc_logvar = nn.Linear(enc_flat, latent_dim)
        self.decoder = SparseVoxelDecoder(latent_dim, base_channels, num_levels, self.grid_size)

    def encode(self, x, return_features=False):
        sparse_mask = (x[:, 1:2, :, :, :] > 0).float()
        features, _ = self.encoder(x, sparse_mask)
        h = features[-1]
        batch = h.shape[0]
        h_flat = h.view(batch, -1)

        # English comment for public release.
        if h_flat.shape[1] != self.fc_mu.in_features:
            enc_flat = h_flat.shape[1]
            enc_channels = h.shape[1]
            enc_res = h.shape[2:]
            self.fc_mu = nn.Linear(enc_flat, self.latent_dim, device=h.device)
            self.fc_logvar = nn.Linear(enc_flat, self.latent_dim, device=h.device)
            # English comment for public release.
            self.decoder.min_res = enc_res
            self.decoder.fc = nn.Linear(self.latent_dim, enc_channels * np.prod(enc_res), device=h.device)
            # English comment for public release.
            nn.init.kaiming_uniform_(self.fc_mu.weight, a=math.sqrt(5))
            nn.init.zeros_(self.fc_mu.bias)
            nn.init.kaiming_uniform_(self.fc_logvar.weight, a=math.sqrt(5))
            nn.init.zeros_(self.fc_logvar.bias)
            nn.init.kaiming_uniform_(self.decoder.fc.weight, a=math.sqrt(5))
            nn.init.zeros_(self.decoder.fc.bias)

        mu = self.fc_mu(h_flat)
        logvar = torch.clamp(self.fc_logvar(h_flat), min=-10, max=10)
        if return_features:
            return mu, logvar, features
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, encoder_features=None):
        return self.decoder(z, encoder_features)

    def forward(self, x, use_skip_connections=True):
        if use_skip_connections:
            mu, logvar, features = self.encode(x, return_features=True)
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z, features)
        else:
            mu, logvar = self.encode(x, return_features=False)
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z, None)
        return recon, mu, logvar


def dice_loss(pred_logits, target, smooth=1e-5):
    """
    Dice Loss for 3D segmentation.
    """
    pred = torch.sigmoid(pred_logits)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 2. * intersection / (union + smooth)
    return 1 - dice.mean()


def total_variation_loss(x):
    """
    Total Variation Loss to encourage spatial smoothness (physical constraint).
    Calculates L1 differences between adjacent voxels.
    """
    # x: (B, D, H, W) or (B, 1, D, H, W)
    if x.ndim == 5:
        x = x.squeeze(1)
    
    tv_d = torch.abs(x[:, 1:, :, :] - x[:, :-1, :, :]).mean()
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return (tv_d + tv_h + tv_w) / 3.0

def gradient_loss(pred, target):
    # Auto-unsqueeze for 4D inputs (B, D, H, W) -> (B, 1, D, H, W)
    if pred.ndim == 4:
        pred = pred.unsqueeze(1)
    if target.ndim == 4:
        target = target.unsqueeze(1)
        
    diff_h = torch.abs((pred[:, :, 1:, :, :]-pred[:, :, :-1, :, :]) - (target[:, :, 1:, :, :]-target[:, :, :-1, :, :]))
    diff_w = torch.abs((pred[:, :, :, 1:, :]-pred[:, :, :, :-1, :]) - (target[:, :, :, 1:, :]-target[:, :, :, :-1, :]))
    diff_d = torch.abs((pred[:, :, :, :, 1:]-pred[:, :, :, :, :-1]) - (target[:, :, :, :, 1:]-target[:, :, :, :, :-1]))
    return diff_h.mean() + diff_w.mean() + diff_d.mean()

def loss_function(recon_logits, target_vox, mu, logvar, beta=1.0, free_bits=0.0, obs_mask=None, lambda_drill=1.0):
    """Documentation translated to English for open-source release."""
    # English comment for public release.
    # English comment for public release.
    if obs_mask is not None:
        if obs_mask.ndim == 5:
            obs_mask = obs_mask.squeeze(1)
            
        # English comment for public release.
        # English comment for public release.
        extra_weight = max(0.0, lambda_drill - 1.0)
        pos_weight = torch.ones_like(target_vox) + obs_mask * extra_weight
        
        bce_raw = F.binary_cross_entropy_with_logits(recon_logits, target_vox, weight=pos_weight, reduction='none')
    else:
        bce_raw = F.binary_cross_entropy_with_logits(recon_logits, target_vox, reduction='none')
    
    bce = bce_raw.mean()


    # English comment for public release.
    dice = dice_loss(recon_logits, target_vox)
    dice_weight = 0.5  # English comment for public release.

    # English comment for public release.
    kld_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0.0:
        free_bits_tensor = torch.tensor(free_bits, device=recon_logits.device)
        kld_elem = torch.max(kld_elem, free_bits_tensor)
    kld = kld_elem.mean()

    # English comment for public release.
    recon_prob = torch.sigmoid(recon_logits)
    grad_loss_val = gradient_loss(recon_prob, target_vox)
    grad_weight = 0.15 # Stronger weight for gradients
    
    tv = total_variation_loss(recon_prob)
    tv_weight = 0.005 # Slight smoothness

    total_loss = bce + dice * dice_weight + beta * kld + grad_loss_val * grad_weight + tv * tv_weight
    return total_loss, bce, kld, dice


def optimize_latent_for_observation(
    model, 
    obs, 
    steps=50, 
    lr=0.05, 
    lambda_drill=100.0, 
    lambda_prior=0.01,
    verbose=False
):
    """Documentation translated to English for open-source release."""
    # English comment for public release.
    # obs: (B, 2, D, H, W)
    drill_vals = obs[:, 0, ...].unsqueeze(1)  # (B, 1, D, H, W)
    drill_mask = obs[:, 1, ...].unsqueeze(1)  # (B, 1, D, H, W)
    
    model.eval()
    
    # English comment for public release.
    # English comment for public release.
    with torch.no_grad():
        mu, logvar = model.encode(obs)
        z_init = model.reparameterize(mu, logvar)
    
    # English comment for public release.
    z_opt = z_init.clone().detach().requires_grad_(True)
    
    # English comment for public release.
    optimizer = torch.optim.Adam([z_opt], lr=lr)
    
    if verbose:
        print(f"Start Latent Search... (Steps={steps}, LR={lr})")

    for i in range(steps):
        optimizer.zero_grad()
        
        # English comment for public release.
        logits = model.decode(z_opt)
        
        # English comment for public release.
        # English comment for public release.
        # English comment for public release.
        probs = torch.sigmoid(logits)
        
        # English comment for public release.
        # English comment for public release.
        # L2 Loss on probability: (probs - vals)^2 * mask
        fitting_loss = ( (probs - drill_vals) ** 2 * drill_mask ).sum() / (drill_mask.sum() + 1e-8)
        
        # English comment for public release.
        # English comment for public release.
        prior_loss = (z_opt ** 2).mean()
        
        # English comment for public release.
        tv = total_variation_loss(probs)
        
        loss = lambda_drill * fitting_loss + lambda_prior * prior_loss + 0.1 * tv
        
        loss.backward()
        optimizer.step()
        
        if verbose and i % 10 == 0:
            print(f"  Step {i}: Loss={loss.item():.4f} (Fit={fitting_loss.item():.4f})")
            
    # English comment for public release.
    with torch.no_grad():
        logits_final = model.decode(z_opt)
        
    return z_opt, logits_final

def vox_to_pointcloud(vox, threshold=0.5):
    """Documentation translated to English for open-source release."""

    if isinstance(vox, torch.Tensor):
        vox_np = vox.detach().cpu().numpy()
    else:
        vox_np = vox

    if vox_np.ndim == 4:
        vox_np = vox_np.squeeze(0)

    D = vox_np.shape[0]
    grid = vox_np >= threshold
    idx = np.array(np.nonzero(grid)).T
    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.stack([idx[:, 1], idx[:, 0], idx[:, 2]], axis=1).astype(np.float32) / float(D - 1)
    return pts


# ============================================================================
# English comment for public release.
# ============================================================================


class LoRALayer(nn.Module):
    def __init__(self, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank


class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        for param in self.original_layer.parameters():
            param.requires_grad = False
        d_in = original_layer.in_features
        d_out = original_layer.out_features
        
        self.in_features = d_in
        self.out_features = d_out
        
        self.lora_A = nn.Parameter(torch.zeros(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = self.original_layer(x)
        if self.rank > 0:
            lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
            result = result + lora_out * self.scaling
        return result

    def merge_weights(self):
        if self.rank > 0:
            delta_W = (self.lora_B @ self.lora_A) * self.scaling
            self.original_layer.weight.data += delta_W
            self.lora_A.data.zero_()
            self.lora_B.data.zero_()


class LoRAConv3d(nn.Module):
    def __init__(self, original_layer, rank=4, alpha=8, dropout=0.0):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        for param in self.original_layer.parameters():
            param.requires_grad = False
        out_channels = original_layer.out_channels
        in_channels = original_layer.in_channels
        k = original_layer.kernel_size
        d_out = out_channels
        d_in = in_channels * k[0] * k[1] * k[2]
        self.lora_A = nn.Parameter(torch.zeros(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.kernel_size = k
        self.stride = original_layer.stride
        self.padding = original_layer.padding
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        result = self.original_layer(x)
        if self.rank > 0:
            delta_weight_2d = self.lora_B @ self.lora_A
            delta_weight = delta_weight_2d.view(
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
                self.kernel_size[2],
            )
            lora_out = F.conv3d(
                self.dropout(x),
                delta_weight * self.scaling,
                bias=None,
                stride=self.stride,
                padding=self.padding,
            )
            result = result + lora_out
        return result

    def merge_weights(self):
        if self.rank > 0:
            delta_weight_2d = (self.lora_B @ self.lora_A) * self.scaling
            delta_weight = delta_weight_2d.view(
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
                self.kernel_size[2],
            )
            self.original_layer.weight.data += delta_weight
            self.lora_A.data.zero_()
            self.lora_B.data.zero_()


def apply_lora_to_model(model, rank=8, alpha=16, target_modules=None, dropout=0.0):
    lora_params = []

    def should_apply(name):
        if target_modules is None:
            return True
        return any(target in name for target in target_modules)

    named = dict(model.named_modules())
    for name, module in list(model.named_modules()):
        if not should_apply(name):
            continue
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        parent = named.get(parent_name, model) if parent_name else model
        if isinstance(module, nn.Linear):
            lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
            lora_layer.to(module.weight.device) # Ensure LoRA params are on same device
            setattr(parent, child_name, lora_layer)
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
            # English comment for public release.
        elif isinstance(module, nn.Conv3d):
            lora_layer = LoRAConv3d(module, rank=rank, alpha=alpha, dropout=dropout)
            lora_layer.to(module.weight.device) # Ensure LoRA params are on same device
            setattr(parent, child_name, lora_layer)
            # English comment for public release.
    return model, lora_params


def get_lora_state_dict(model):
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv3d)):
            lora_state[f"{name}.lora_A"] = module.lora_A.data
            lora_state[f"{name}.lora_B"] = module.lora_B.data
    return lora_state


def load_lora_state_dict(model, lora_state_dict):
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv3d)):
            if f"{name}.lora_A" in lora_state_dict:
                module.lora_A.data = lora_state_dict[f"{name}.lora_A"]
            if f"{name}.lora_B" in lora_state_dict:
                module.lora_B.data = lora_state_dict[f"{name}.lora_B"]


def merge_all_lora_weights(model):
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAConv3d)):
            module.merge_weights()
    print("✓ 所有 LoRA 权重已合并到原始层")


def print_lora_statistics(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print("\n" + "=" * 60)
    print("LoRA 参数统计")
    print("=" * 60)
    print(f"总参数量:        {total_params:,}")
    print(f"可训练参数:      {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")
    print(f"冻结参数:        {frozen_params:,} ({frozen_params / total_params * 100:.2f}%)")
    print(f"参数减少比例:    {(1 - trainable_params / total_params) * 100:.1f}%")
    print("=" * 60 + "\n")


LORA_CONFIGS = {
    "minimal": {
        "rank": 4,
        "alpha": 8,
        "target_modules": ["fc_mu", "fc_logvar", "fc_dec"],
        "dropout": 0.0,
        "description": "极简配置 - 仅微调潜在空间映射层",
    },
    "light": {
        "rank": 8,
        "alpha": 16,
        "target_modules": ["enc_down", "dec_up", "fc"],
        "dropout": 0.05,
        "description": "轻量配置 - 微调下采样/上采样层和FC层",
    },
    "standard": {
        "rank": 16,
        "alpha": 32,
        "target_modules": None,
        "dropout": 0.1,
        "description": "标准配置 - 微调所有 Linear 和 Conv3d 层",
    },
    "aggressive": {
        "rank": 32,
        "alpha": 64,
        "target_modules": None,
        "dropout": 0.1,
        "description": "激进配置 - 高秩 LoRA，适合复杂任务",
    },
}


def apply_lora_preset(model, preset="light"):
    if preset not in LORA_CONFIGS:
        raise ValueError(f"未知预设: {preset}。可选: {list(LORA_CONFIGS.keys())}")
    config = LORA_CONFIGS[preset]
    print(f"\n📌 应用 LoRA 预设: '{preset}'")
    print(f"   描述: {config['description']}")
    print(f"   Rank={config['rank']}, Alpha={config['alpha']}, Dropout={config['dropout']}\n")
    model, lora_params = apply_lora_to_model(
        model,
        rank=config["rank"],
        alpha=config["alpha"],
        target_modules=config["target_modules"],
        dropout=config["dropout"],
    )

    # English comment for public release.
    if len(lora_params) == 0:
        print("⚠️  预设未匹配到任何层，回退为全部 Linear/Conv3d 应用 LoRA")
        model, lora_params = apply_lora_to_model(
            model,
            rank=config["rank"],
            alpha=config["alpha"],
            target_modules=None,
            dropout=config["dropout"],
        )

    print_lora_statistics(model)
    return model, lora_params


# English comment for public release.
def create_octree_vae(config):
    return OctreeVAE3D(
        in_channels=2,
        grid_size=config.get("grid_size", (64, 64, 64)),
        latent_dim=config.get("latent_dim", 512),
        base_channels=config.get("base_channels", 32),
        num_levels=config.get("num_levels", 4),
    )

