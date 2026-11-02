import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def _safe_gn_groups(channels: int, max_groups: int = 8) -> int:
    """Return a GroupNorm group count that divides channels."""
    g = min(max_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return max(1, g)

class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(1, channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock3D(nn.Module):
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
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


class GeoTransformerBlock(nn.Module):
    """Bottleneck self-attention with a learned, additive 3D relative-position bias.

    The bias is indexed by signed voxel offsets along depth, height and width,
    independently for every attention head. At grid sizes above max_distance,
    offsets are clamped; normal bottlenecks (2-8 voxels per axis) are exact.
    """

    def __init__(self, channels, num_heads=4, ff_dim_scale=2, dropout=0.1,
                 use_relative_position=True, max_distance=8):
        super().__init__()
        if channels % num_heads:
            raise ValueError("Attention channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.use_relative_position = bool(use_relative_position)
        self.max_distance = int(max_distance)
        if self.max_distance < 1:
            raise ValueError("max_distance must be positive")
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        if self.use_relative_position:
            # Zero initialization preserves ordinary attention at step zero.
            self.relative_bias = nn.Parameter(
                torch.zeros(3, num_heads, 2 * self.max_distance + 1)
            )
        else:
            self.register_parameter("relative_bias", None)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * ff_dim_scale),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * ff_dim_scale, channels),
            nn.Dropout(dropout),
        )

    def _attention_bias(self, shape, batch_size, device):
        if self.relative_bias is None:
            return None
        axes = [torch.arange(n, device=device) for n in shape]
        coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        coordinates = coordinates.reshape(-1, 3)
        offsets = coordinates[:, None, :] - coordinates[None, :, :]
        offsets = offsets.clamp(-self.max_distance, self.max_distance)
        offsets = offsets + self.max_distance
        head_bias = sum(
            self.relative_bias[axis][:, offsets[..., axis]]
            for axis in range(3)
        )
        # MultiheadAttention expects (batch * heads, tokens, tokens).
        return head_bias.repeat(batch_size, 1, 1)

    def forward(self, x):
        batch, channels, depth, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        normalized = self.norm1(tokens)
        bias = self._attention_bias((depth, height, width), batch, x.device)
        attended, _ = self.attn(
            normalized, normalized, normalized,
            attn_mask=bias, need_weights=False,
        )
        tokens = tokens + attended
        tokens = tokens + self.ff(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, depth, height, width)


class ConvVAE3D(nn.Module):
    def __init__(self, in_channels=2, grid_size=(32, 32, 32), latent_dim=256, base_channels=32, use_transformer=True, use_relative_position=True):
        super().__init__()
        self.in_channels = in_channels
        if isinstance(grid_size, int):
            self.grid_size = (grid_size, grid_size, grid_size)
        else:
            self.grid_size = tuple(grid_size)
            
        self.latent_dim = latent_dim
        self.base = base_channels
        self.use_transformer = bool(use_transformer)
        self.use_relative_position = bool(use_relative_position)

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
            self.enc_transformer = GeoTransformerBlock(
                self.base * 16, num_heads=4,
                use_relative_position=self.use_relative_position,
            )
        
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
             self.dec_transformer = GeoTransformerBlock(
                self.base * 16, num_heads=4,
                use_relative_position=self.use_relative_position,
            )
        
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

    def forward(self, x, sample_latent=None):
        mu, logvar = self.encode(x)
        # Training remains stochastic; evaluation uses the posterior mean.
        # Callers can explicitly override either choice for sampling studies.
        if sample_latent is None:
            sample_latent = self.training
        z = self.reparameterize(mu, logvar) if sample_latent else mu
        logits = self.decode(z)
        return logits, mu, logvar


class SparseVoxelEncoder(nn.Module):

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

    def __init__(self, in_channels=2, grid_size=(64, 64, 64), latent_dim=512, base_channels=32, num_levels=4):
        super().__init__()
        self.in_channels = in_channels
        self.grid_size = tuple(grid_size) if isinstance(grid_size, (list, tuple)) else (grid_size,) * 3
        self.latent_dim = latent_dim
        self.base = base_channels
        self.num_levels = num_levels
        self.encoder = SparseVoxelEncoder(in_channels, base_channels, num_levels)
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

        if h_flat.shape[1] != self.fc_mu.in_features:
            enc_flat = h_flat.shape[1]
            enc_channels = h.shape[1]
            enc_res = h.shape[2:]
            self.fc_mu = nn.Linear(enc_flat, self.latent_dim, device=h.device)
            self.fc_logvar = nn.Linear(enc_flat, self.latent_dim, device=h.device)
            self.decoder.min_res = enc_res
            self.decoder.fc = nn.Linear(self.latent_dim, enc_channels * np.prod(enc_res), device=h.device)
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
    if obs_mask is not None:
        if obs_mask.ndim == 5:
            obs_mask = obs_mask.squeeze(1)
            
        extra_weight = max(0.0, lambda_drill - 1.0)
        pos_weight = torch.ones_like(target_vox) + obs_mask * extra_weight
        
        bce_raw = F.binary_cross_entropy_with_logits(recon_logits, target_vox, weight=pos_weight, reduction='none')
    else:
        bce_raw = F.binary_cross_entropy_with_logits(recon_logits, target_vox, reduction='none')
    
    bce = bce_raw.mean()

    dice = dice_loss(recon_logits, target_vox)
    dice_weight = 0.5 

    kld_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0.0:
        free_bits_tensor = torch.tensor(free_bits, device=recon_logits.device)
        kld_elem = torch.max(kld_elem, free_bits_tensor)
    kld = kld_elem.mean()

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

    # obs: (B, 2, D, H, W)
    drill_vals = obs[:, 0, ...].unsqueeze(1)  # (B, 1, D, H, W)
    drill_mask = obs[:, 1, ...].unsqueeze(1)  # (B, 1, D, H, W)
    
    model.eval()
    
    with torch.no_grad():
        mu, logvar = model.encode(obs)
        z_init = model.reparameterize(mu, logvar)
    
    z_opt = z_init.clone().detach().requires_grad_(True)
    
    optimizer = torch.optim.Adam([z_opt], lr=lr)
    
    if verbose:
        print(f"Start Latent Search... (Steps={steps}, LR={lr})")

    for i in range(steps):
        optimizer.zero_grad()
        
        logits = model.decode(z_opt)
        
        probs = torch.sigmoid(logits)
        
        fitting_loss = ( (probs - drill_vals) ** 2 * drill_mask ).sum() / (drill_mask.sum() + 1e-8)
        
        prior_loss = (z_opt ** 2).mean()
        
        tv = total_variation_loss(probs)
        
        loss = lambda_drill * fitting_loss + lambda_prior * prior_loss + 0.1 * tv
        
        loss.backward()
        optimizer.step()
        
        if verbose and i % 10 == 0:
            print(f"  Step {i}: Loss={loss.item():.4f} (Fit={fitting_loss.item():.4f})")
            
    with torch.no_grad():
        logits_final = model.decode(z_opt)
        
    return z_opt, logits_final

def vox_to_pointcloud(vox, threshold=0.5):

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
        if original_layer.padding_mode != "zeros":
            raise NotImplementedError("LoRAConv3d requires zero padding")
        d_in = (in_channels // original_layer.groups) * k[0] * k[1] * k[2]
        self.lora_A = nn.Parameter(torch.zeros(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.kernel_size = k
        self.stride = original_layer.stride
        self.padding = original_layer.padding
        self.dilation = original_layer.dilation
        self.groups = original_layer.groups
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        result = self.original_layer(x)
        if self.rank > 0:
            delta_weight_2d = self.lora_B @ self.lora_A
            delta_weight = delta_weight_2d.view(
                self.out_channels,
                self.in_channels // self.groups,
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
                dilation=self.dilation,
                groups=self.groups,
            )
            result = result + lora_out
        return result

    def merge_weights(self):
        if self.rank > 0:
            delta_weight_2d = (self.lora_B @ self.lora_A) * self.scaling
            delta_weight = delta_weight_2d.view(
                self.out_channels, 
                self.in_channels // self.groups,
                self.kernel_size[0],
                self.kernel_size[1],
                self.kernel_size[2],
            )
            self.original_layer.weight.data += delta_weight
            self.lora_A.data.zero_()
            self.lora_B.data.zero_()


def apply_lora_to_model(model, rank=8, alpha=16, target_modules=None, dropout=0.0):
    """Freeze the base network and insert LoRA into selected Linear/Conv3d layers.

    MultiheadAttention consumes out_proj.weight directly, so wrapping that
    submodule would break its projection. It is excluded. The returned list
    contains both Linear and Conv3d adapter parameters.
    """
    if rank < 1:
        raise ValueError("LoRA rank must be positive")
    if any(isinstance(m, (LoRALinear, LoRAConv3d)) for m in model.modules()):
        raise ValueError("Model already contains LoRA layers; use a fresh base model")

    def should_apply(name):
        if name.endswith("attn.out_proj"):
            return False
        if target_modules is None:
            return True
        return any(target in name for target in target_modules)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    lora_params = []
    named = dict(model.named_modules())
    for name, module in list(model.named_modules()):
        if not should_apply(name):
            continue
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = named.get(parent_name, model) if parent_name else model
        if isinstance(module, nn.Linear):
            wrapper = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        elif isinstance(module, nn.Conv3d):
            wrapper = LoRAConv3d(module, rank=rank, alpha=alpha, dropout=dropout)
        else:
            continue
        wrapper.to(device=module.weight.device, dtype=module.weight.dtype)
        setattr(parent, child_name, wrapper)
        lora_params.extend((wrapper.lora_A, wrapper.lora_B))
    return model, lora_params



def adapt_to_observation(
    model,
    obs,
    mode="latent",
    steps=50,
    lora_rank=8,
    lr=0.05,
    lora_lr=0.001,
    prior_weight=0.01,
    lora_weight=0.0001,
    patience=25,
    min_delta=1e-6,
    max_grad_norm=5.0,
    project_observed=False,
    seed=0,
):
    """Adapt one trained ConvVAE3D to one sparse observation without target voxels.

    Input channels are [observed binary label, observation mask]. Only mask
    locations enter the data-fit BCE. Latent-only inference shares frozen
    checkpoint weights; joint LoRA inference creates a fresh model copy.
    Neither mode leaves state changes on ``model``. The
    returned probability is unprojected unless ``project_observed=True``.
    Projection, if requested, is an exact output operation rather than a
    claimed consequence of the soft optimization loss.
    """
    if not isinstance(model, ConvVAE3D):
        raise TypeError("adapt_to_observation currently supports ConvVAE3D only")
    if mode not in {"latent", "joint"}:
        raise ValueError("mode must be 'latent' or 'joint'")
    if not isinstance(obs, torch.Tensor) or obs.ndim != 5 or obs.shape[1] != 2:
        raise ValueError("obs must have shape [1, 2, D, H, W]")
    if obs.shape[0] != 1:
        raise ValueError("Adapt one orebody at a time to prevent case leakage")
    if tuple(obs.shape[-3:]) != tuple(model.grid_size):
        raise ValueError("Observation grid does not match model.grid_size")
    if steps < 0 or lr <= 0 or lora_lr <= 0:
        raise ValueError("steps must be nonnegative and learning rates positive")
    if prior_weight < 0 or lora_weight < 0:
        raise ValueError("Regularization weights must be nonnegative")
    if mode == "joint" and lora_rank < 1:
        raise ValueError("Joint adaptation requires lora_rank >= 1")
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Call adaptation outside torch.inference_mode()")

    reference_parameter = next(model.parameters())
    device, dtype = reference_parameter.device, reference_parameter.dtype
    obs = obs.detach().to(device=device, dtype=dtype)
    if not bool(torch.isfinite(obs).all()):
        raise ValueError("Observation contains nonfinite values")
    mask = obs[:, 1]
    labels = obs[:, 0]
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("Observation mask must be binary")
    if not bool((((labels == 0) | (labels == 1)) | (mask == 0)).all()):
        raise ValueError("Observed labels must be binary")
    observed_count = int(mask.sum().item())
    if observed_count == 0:
        raise ValueError("Observation mask is empty")

    def _run(working):
        for parameter in working.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            mu, _ = working.encode(obs)
        z_initial = mu.detach().clone()
        z = z_initial.clone().requires_grad_(True)

        lora_parameters = []
        if mode == "joint":
            if any(isinstance(m, (LoRALinear, LoRAConv3d)) for m in working.modules()):
                raise ValueError("Joint adaptation requires a checkpoint without training-time LoRA")
            rng_devices = [device] if device.type == "cuda" else []
            # Fixed local initialization makes repeated per-case evaluation stable
            # without changing the caller's global RNG state.
            with torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(int(seed))
                working, lora_parameters = apply_lora_to_model(
                    working,
                    rank=lora_rank,
                    alpha=lora_rank,
                    target_modules=["fc_dec", "dec_res", "dec_up", "dec_final"],
                    dropout=0.0,
                )
            if not lora_parameters:
                raise RuntimeError("No decoder LoRA layers were installed")
        initial_lora = [parameter.detach().clone() for parameter in lora_parameters]

        def objective(logits):
            per_voxel = F.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            )
            fit = (per_voxel * mask).sum() / mask.sum()
            prior = 0.5 * z.square().mean()
            if lora_parameters:
                adapter_penalty = sum(
                    (parameter - initial).square().mean()
                    for parameter, initial in zip(lora_parameters, initial_lora)
                )
            else:
                adapter_penalty = fit.new_zeros(())
            total = fit + prior_weight * prior + lora_weight * adapter_penalty
            return total, fit

        optimizer_groups = [{"params": [z], "lr": lr}]
        if lora_parameters:
            optimizer_groups.append({"params": lora_parameters, "lr": lora_lr})
        optimizer = torch.optim.Adam(optimizer_groups)
        best_value = float("inf")
        best_step = 0
        best_z = z_initial.clone()
        best_lora = [parameter.detach().clone() for parameter in lora_parameters]
        steps_run = 0
        first_z_grad_norm = 0.0
        first_lora_grad_norm = 0.0

        with torch.enable_grad():
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                logits = working.decode(z)
                total, _ = objective(logits)
                value = float(total.detach().item())
                if not math.isfinite(value):
                    raise FloatingPointError("Nonfinite test-time adaptation objective")
                if value < best_value - min_delta:
                    best_value = value
                    best_step = step
                    best_z = z.detach().clone()
                    best_lora = [p.detach().clone() for p in lora_parameters]
                total.backward()
                if step == 0:
                    first_z_grad_norm = float(z.grad.detach().norm().item())
                    first_lora_grad_norm = math.sqrt(sum(
                        float(p.grad.detach().square().sum().item())
                        for p in lora_parameters if p.grad is not None
                    ))
                torch.nn.utils.clip_grad_norm_([z] + lora_parameters, max_grad_norm)
                optimizer.step()
                steps_run = step + 1
                if patience is not None and patience > 0 and steps_run - best_step >= patience:
                    break

            # Include the state after the final optimizer update in the comparison.
            with torch.no_grad():
                final_total, _ = objective(working.decode(z))
                value = float(final_total.item())
                if math.isfinite(value) and value < best_value - min_delta:
                    best_value = value
                    best_step = steps_run
                    best_z = z.detach().clone()
                    best_lora = [p.detach().clone() for p in lora_parameters]
                if steps == 0:
                    best_value = value

                z.copy_(best_z)
                for parameter, saved in zip(lora_parameters, best_lora):
                    parameter.copy_(saved)
                logits = working.decode(z)
                raw_probability = torch.sigmoid(logits)
                observed_bce = (
                    F.binary_cross_entropy_with_logits(
                        logits, labels, reduction="none"
                    ) * mask
                ).sum() / mask.sum()
                observed_mse = (
                    (raw_probability - labels).square() * mask
                ).sum() / mask.sum()
                observed_error = (
                    ((raw_probability >= 0.5) != (labels >= 0.5)).float() * mask
                ).sum() / mask.sum()
                z_shift = (z - z_initial).norm()
                adapter_shift = math.sqrt(sum(
                    float((p - initial).square().sum().item())
                    for p, initial in zip(lora_parameters, initial_lora)
                ))
                if project_observed:
                    probability = torch.where(mask.bool(), labels, raw_probability)
                else:
                    probability = raw_probability
                post_mse = (
                    (probability - labels).square() * mask
                ).sum() / mask.sum()

        metadata = {
            "mode": mode,
            "seed": int(seed),
            "steps_requested": int(steps),
            "steps_run": int(steps_run),
            "best_step": int(best_step),
            "best_objective": float(best_value),
            "observed_voxels": observed_count,
            "obs_bce_pre_projection": float(observed_bce.item()),
            "obs_mse_pre_projection": float(observed_mse.item()),
            "obs_binary_error_pre_projection": float(observed_error.item()),
            "obs_mse_post_projection": float(post_mse.item()),
            "projection_applied": bool(project_observed),
            "z_shift_l2": float(z_shift.item()),
            "lora_shift_l2": float(adapter_shift),
            "lora_trainable_parameters": sum(p.numel() for p in lora_parameters),
            "z_grad_norm_first_step": first_z_grad_norm,
            "lora_grad_norm_first_step": first_lora_grad_norm,
        }
        return probability.detach(), metadata

    if mode == "joint":
        # Decoder adapters must never affect the next case.
        return _run(copy.deepcopy(model).eval())

    # Only z is optimized here. Sharing frozen decoder weights avoids a full
    # 3D-model copy per case while preserving all caller flags and gradients.
    modules = [(module, module.training) for module in model.modules()]
    parameters = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    try:
        model.eval()
        for parameter, _ in parameters:
            parameter.requires_grad_(False)
        return _run(model)
    finally:
        for parameter, requires_grad in parameters:
            parameter.requires_grad_(requires_grad)
        for module, training in modules:
            module.training = training


def get_lora_state_dict(model):
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv3d)):
            lora_state[f"{name}.lora_A"] = module.lora_A.detach().clone()
            lora_state[f"{name}.lora_B"] = module.lora_B.detach().clone()
    return lora_state


def load_lora_state_dict(model, lora_state_dict):
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv3d)):
            with torch.no_grad():
                if f"{name}.lora_A" in lora_state_dict:
                    module.lora_A.copy_(lora_state_dict[f"{name}.lora_A"])
                if f"{name}.lora_B" in lora_state_dict:
                    module.lora_B.copy_(lora_state_dict[f"{name}.lora_B"])


def merge_all_lora_weights(model):
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAConv3d)):
            module.merge_weights()
    print("✓ All LoRA weights merged into base layers")


def print_lora_statistics(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print("\n" + "=" * 60)
    print("LoRA Parameter Statistics")
    print("=" * 60)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")
    print(f"Frozen params:    {frozen_params:,} ({frozen_params / total_params * 100:.2f}%)")
    print(f"Param reduction:  {(1 - trainable_params / total_params) * 100:.1f}%")
    print("=" * 60 + "\n")


LORA_CONFIGS = {
    "minimal": {
        "rank": 4,
        "alpha": 8,
        "target_modules": ["fc_mu", "fc_logvar", "fc_dec"],
        "dropout": 0.0,
        "description": "Minimal - fine-tune only latent space projection layers",
    },
    "light": {
        "rank": 8,
        "alpha": 16,
        "target_modules": ["enc_down", "dec_up", "fc"],
        "dropout": 0.05,
        "description": "Lightweight - fine-tune down/up-sample and FC layers",
    },
    "standard": {
        "rank": 16,
        "alpha": 32,
        "target_modules": None,
        "dropout": 0.1,
        "description": "Standard - fine-tune all Linear and Conv3d layers",
    },
    "aggressive": {
        "rank": 32,
        "alpha": 64,
        "target_modules": None,
        "dropout": 0.1,
        "description": "Aggressive - high-rank LoRA for complex tasks",
    },
}


def apply_lora_preset(model, preset="light"):
    if preset not in LORA_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Options: {list(LORA_CONFIGS.keys())}")
    config = LORA_CONFIGS[preset]
    print(f"\n📌 Applying LoRA preset: '{preset}'")
    print(f"   Description: {config['description']}")
    print(f"   Rank={config['rank']}, Alpha={config['alpha']}, Dropout={config['dropout']}\n")
    model, lora_params = apply_lora_to_model(
        model,
        rank=config["rank"],
        alpha=config["alpha"],
        target_modules=config["target_modules"],
        dropout=config["dropout"],
    )

    if not lora_params:
        raise ValueError(
            f"LoRA preset {preset!r} matched no layers; choose explicit targets"
        )

    print_lora_statistics(model)
    return model, lora_params


# Convenience factory: maintain compatibility
def create_octree_vae(config):
    return OctreeVAE3D(
        in_channels=2,
        grid_size=config.get("grid_size", (64, 64, 64)),
        latent_dim=config.get("latent_dim", 512),
        base_channels=config.get("base_channels", 32),
        num_levels=config.get("num_levels", 4),
    )

