"""Documentation translated to English for open-source release."""

import torch
import torch.optim as optim
from model import (
    ConvVAE3D,
    OctreeVAE3D,
    apply_lora_preset,
    print_lora_statistics,
)


def create_model(
    model_type='standard',
    use_lora=False,
    lora_preset='light',
    grid_size=(64, 64, 64),
    latent_dim=512,
    base_channels=32,
    num_levels=4,
    lr=1e-4,
    weight_decay=1e-5,
    device='cuda',
    log_mode='full'
):
    """Documentation translated to English for open-source release."""
    
    is_brief = (log_mode == 'brief')
    if not is_brief:
        print("\n" + "="*70)
        print("🚀 模型配置向导")
        print("="*70)
    
    # English comment for public release.
    if model_type.lower() == 'octree':
        if not is_brief:
            print(f"📦 模型类型: 八叉树 VAE (稀疏优化)")
        model = OctreeVAE3D(
            in_channels=2,
            grid_size=grid_size,
            latent_dim=latent_dim,
            base_channels=base_channels,
            num_levels=num_levels
        )
    elif model_type.lower() == 'standard':
        if not is_brief:
            print(f"📦 模型类型: 标准 3D VAE")
        model = ConvVAE3D(
            in_channels=2,
            grid_size=grid_size,
            latent_dim=latent_dim,
            base_channels=base_channels
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}。可选: 'standard', 'octree'")
    
    model = model.to(device)
    
    # English comment for public release.
    base_params = sum(p.numel() for p in model.parameters())
    if not is_brief:
        print(f"   - 网格分辨率: {grid_size}")
        print(f"   - 潜在维度: {latent_dim}")
        print(f"   - 基础通道数: {base_channels}")
        if model_type.lower() == 'octree':
            print(f"   - 八叉树层数: {num_levels}")
        print(f"   - 基础参数量: {base_params:,}")
    
    # English comment for public release.
    lora_params = None
    encoder_param_count = 0
    decoder_param_count = 0
    if use_lora:
        if not is_brief:
            print(f"\n⚡ 应用 LoRA 优化: 预设 '{lora_preset}'")
        model, lora_params = apply_lora_preset(model, preset=lora_preset)
        if not is_brief:
            print_lora_statistics(model)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        reduction = (1 - trainable_params / base_params) * 100
        
        if not is_brief:
            print(f"✅ LoRA 已应用:")
            print(f"   - 可训练参数: {trainable_params:,} (减少 {reduction:.1f}%)")
            print(f"   - 预计训练速度提升: 2-3x")
            print(f"   - 预计显存节省: 40-60%")
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if not is_brief:
            print(f"\n📊 未使用 LoRA 优化 (全量训练)")
            print(f"   - 可训练参数: {trainable_params:,}")
    
    # English comment for public release.
    if use_lora and lora_params is not None:
        # English comment for public release.
        optimizer = optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)
        if not is_brief:
            print(f"\n🔧 优化器: AdamW (仅 LoRA 参数)")
    else:
        # English comment for public release.
        encoder_params = []
        decoder_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if any(x in name for x in ['enc_', 'fc_mu', 'fc_logvar', 'encoder']):
                    encoder_params.append(param)
                else:
                    decoder_params.append(param)
        
        # English comment for public release.
        optimizer = {
            'encoder': optim.AdamW(encoder_params, lr=lr, weight_decay=weight_decay),
            'decoder': optim.AdamW(decoder_params, lr=lr, weight_decay=weight_decay)
        }
        encoder_param_count = sum(p.numel() for p in encoder_params)
        decoder_param_count = sum(p.numel() for p in decoder_params)
        if not is_brief:
            print(f"\n🔧 优化器: 分离优化 (Encoder + Decoder)")
            print(f"   - Encoder 参数: {encoder_param_count:,}")
            print(f"   - Decoder 参数: {decoder_param_count:,}")
    
    # English comment for public release.
    model_info = {
        'model_type': model_type,
        'use_lora': use_lora,
        'lora_preset': lora_preset if use_lora else None,
        'grid_size': grid_size,
        'latent_dim': latent_dim,
        'base_channels': base_channels,
        'num_levels': num_levels if model_type == 'octree' else None,
        'total_params': base_params,
        'trainable_params': trainable_params,
        'using_multi_optimizer': isinstance(optimizer, dict)
    }
    
    return model, optimizer, model_info


def get_model_recommendations(num_samples, gpu_memory_gb, grid_size):
    """Documentation translated to English for open-source release."""
    recommendations = []
    
    # English comment for public release.
    grid_volume = grid_size[0] * grid_size[1] * grid_size[2]
    
    # English comment for public release.
    if num_samples < 1000 or gpu_memory_gb < 6:
        recommendations.append({
            'config': {
                'model_type': 'octree',
                'use_lora': True,
                'lora_preset': 'light',
                'base_channels': 32
            },
            'reason': '数据集小或显存受限，推荐轻量配置',
            'expected_memory': '~4GB',
            'expected_speed': '快 (3-5x)',
            'priority': 'high'
        })
    
    # English comment for public release.
    if 1000 <= num_samples < 3000:
        recommendations.append({
            'config': {
                'model_type': 'octree',
                'use_lora': False,
                'base_channels': 32
            },
            'reason': '中等数据集，八叉树稀疏优化',
            'expected_memory': '~5GB',
            'expected_speed': '中等 (2-3x)',
            'priority': 'medium'
        })
        
        if gpu_memory_gb >= 8:
            recommendations.append({
                'config': {
                    'model_type': 'standard',
                    'use_lora': True,
                    'lora_preset': 'light',
                    'base_channels': 48
                },
                'reason': '标准模型 + LoRA，平衡精度与效率',
                'expected_memory': '~6GB',
                'expected_speed': '中等 (2x)',
                'priority': 'medium'
            })
    
    # English comment for public release.
    if num_samples >= 3000 and gpu_memory_gb >= 12:
        recommendations.append({
            'config': {
                'model_type': 'standard',
                'use_lora': False,
                'base_channels': 64
            },
            'reason': '大数据集，追求最高精度',
            'expected_memory': '~8-10GB',
            'expected_speed': '基准',
            'priority': 'low'
        })
    
    # English comment for public release.
    if grid_volume >= 128**3:
        recommendations.insert(0, {
            'config': {
                'model_type': 'octree',
                'use_lora': True,
                'lora_preset': 'minimal',
                'base_channels': 16
            },
            'reason': '大网格分辨率，必须使用优化',
            'expected_memory': '~6GB',
            'expected_speed': '快 (4-6x)',
            'priority': 'critical'
        })
    
    return recommendations


def print_recommendations(num_samples, gpu_memory_gb, grid_size, log_mode='full'):
    """Documentation translated to English for open-source release."""
    recs = get_model_recommendations(num_samples, gpu_memory_gb, grid_size)
    is_brief = (log_mode == 'brief')
    if is_brief:
        if not recs:
            return
        best = recs[0]
        cfg = best['config']
        print(
            f"推荐配置: {cfg} | 理由: {best['reason']} | 显存: {best['expected_memory']} | 速度: {best['expected_speed']}"
        )
        return
    
    print("\n" + "="*70)
    print("💡 模型配置推荐")
    print("="*70)
    print(f"数据集信息: {num_samples} 个样本, 网格 {grid_size}, GPU {gpu_memory_gb}GB")
    print()
    
    for i, rec in enumerate(recs, 1):
        priority = rec['priority']
        emoji = '🔥' if priority == 'critical' else '⭐' if priority == 'high' else '✓'
        
        print(f"{emoji} 推荐 {i}: {rec['reason']}")
        print(f"   配置: {rec['config']}")
        print(f"   预计显存: {rec['expected_memory']}")
        print(f"   预计速度: {rec['expected_speed']}")
        print()
    
    print("使用方法:")
    print("  model, optimizer, info = create_model(**recommendations[0]['config'])")
    print("="*70 + "\n")


# English comment for public release.

def create_lightweight_model(grid_size=(64, 64, 64), latent_dim=512, lr=1e-4, device='cuda'):
    """Documentation translated to English for open-source release."""
    return create_model(
        model_type='octree',
        use_lora=True,
        lora_preset='minimal',
        grid_size=grid_size,
        latent_dim=latent_dim,
        base_channels=16,
        lr=lr,
        device=device
    )


def create_balanced_model(grid_size=(64, 64, 64), latent_dim=512, lr=1e-4, device='cuda'):
    """Documentation translated to English for open-source release."""
    return create_model(
        model_type='octree',
        use_lora=True,
        lora_preset='light',
        grid_size=grid_size,
        latent_dim=latent_dim,
        base_channels=32,
        lr=lr,
        device=device
    )


def create_highperf_model(grid_size=(64, 64, 64), latent_dim=512, lr=1e-4, device='cuda'):
    """Documentation translated to English for open-source release."""
    return create_model(
        model_type='octree',
        use_lora=False,
        grid_size=grid_size,
        latent_dim=latent_dim,
        base_channels=48,
        lr=lr,
        device=device
    )


def create_standard_model(grid_size=(64, 64, 64), latent_dim=512, lr=1e-4, device='cuda'):
    """Documentation translated to English for open-source release."""
    return create_model(
        model_type='standard',
        use_lora=True,
        lora_preset='light',
        grid_size=grid_size,
        latent_dim=latent_dim,
        base_channels=48,
        lr=lr,
        device=device
    )


if __name__ == '__main__':
    print("=== 模型工厂测试 ===\n")
    
    # English comment for public release.
    print("1. 测试轻量配置 (Octree + LoRA minimal)")
    model, optimizer, info = create_lightweight_model(device='cpu')
    
    # English comment for public release.
    print("\n2. 测试推荐配置 (Octree + LoRA light)")
    model, optimizer, info = create_balanced_model(device='cpu')
    
    # English comment for public release.
    print("\n3. 测试配置推荐系统")
    print_recommendations(num_samples=1510, gpu_memory_gb=8, grid_size=(64, 64, 64))
    
    print("\n✅ 所有测试通过！")
