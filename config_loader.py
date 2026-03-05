import argparse
import os
from typing import Any, Dict, Optional, Sequence

import yaml


DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'config', 'default.yaml')
)


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Documentation translated to English for open-source release."""
    if config_path is None:
        return {}

    path = os.path.abspath(config_path)
    if not os.path.exists(path):
        # English comment for public release.
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError('配置文件内容必须是 key-value 映射')
    return data


def build_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='3D VAE 矿体重建训练 (配置驱动)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    get = defaults.get
    default_ply = os.path.join(os.path.dirname(__file__), '..', 'data', 'mining_ply', 'test_b10.ply')

    # English comment for public release.
    p.add_argument('--lr', type=float, default=float(get('lr', 1e-4)), help='学习率')
    p.add_argument('--max-batch-size', type=int, default=int(get('max_batch_size', 128)), help='Batch Size 上限（防止过大导致性能瓶颈）')
    p.add_argument('--batch-to-mem', type=float, default=float(get('batch_to_mem', 0.8)), help='显存分配比例 (0-1)，实际BS由 (显存可使用量 * 此比例) 动态计算')
    p.add_argument('--epochs', type=int, default=int(get('epochs', 300)), help='训练轮数')
    p.add_argument('--patience', type=int, default=int(get('patience', 30)), help='无效果停止轮数（早停机制）')
    p.add_argument('--patience-start', type=int, default=int(get('patience_start', 80)), help='早停机制启动轮数（在此轮数之前不进行早停检测）')
    p.add_argument('--start-epoch', type=int, default=int(get('start_epoch', 0)), help='恢复训练时的起始 epoch 偏移（加载 checkpoint 时自动填充，0 = 从头训练）')
    p.add_argument('--kl-cycle', type=int, default=int(get('kl_cycle', 80)), help='KL退火周期长度(epoch)')
    p.add_argument('--kl-ratio', type=float, default=float(get('kl_ratio', 0.5)), help='KL退火增加比例(0-1)')
    p.add_argument('--cuda', action='store_true', default=bool(get('cuda', True)), help='使用GPU')
    p.add_argument('--split-seed', type=int, default=int(get('split_seed', 42)), help='样本级训练/测试划分的随机种子（<0 表示每次随机）')

    # English comment for public release.
    p.add_argument('--grid-size', type=int, nargs='+', default=list(get('grid_size', [32, 32, 32])), help='体素网格分辨率 (D H W)，16 的倍数')
    p.add_argument('--latent-dim', type=int, default=int(get('latent_dim', 1024)), help='潜在空间维度')
    p.add_argument('--base-channels', type=int, default=int(get('base_channels', 32)), help='基础通道数')
    p.add_argument('--lambda-drill', type=float, default=float(get('lambda_drill', 50.0)), help='钻孔掩码 BCE 权重上限 (λ_drill)')

    # English comment for public release.
    p.add_argument('--mode', type=str, default='train', choices=['train', 'benchmark', 'ablation', 'robust_eval'], help='运行模式: train(训练), benchmark(对比评测), ablation(消融实验), robust_eval(鲁棒性分析)')
    p.add_argument('--model-type', type=str, default=str(get('model_type', 'standard')), choices=['standard', 'octree'], help='模型类型')
    p.add_argument('--use-lora', action='store_true', default=bool(get('use_lora', False)), help='启用LoRA')
    p.add_argument('--lora-preset', type=str, default=str(get('lora_preset', 'light')), choices=['minimal', 'light', 'standard', 'aggressive'], help='LoRA预设')
    p.add_argument('--octree-levels', type=int, default=int(get('octree_levels', 4)), help='八叉树层数（octree 模型）')
    p.add_argument('--show-recommendations', action='store_true', default=bool(get('show_recommendations', False)), help='显示配置推荐')

    # English comment for public release.
    p.add_argument('--num-holes', type=int, default=int(get('num_holes', 8)), help='每个样本的钻孔数')
    p.add_argument('--samples-per-hole', type=int, default=int(get('samples_per_hole', 12)), help='每个钻孔的采样点数')
    p.add_argument('--augment', type=int, default=int(get('augment', 0)), help='每个网格的增强倍数')
    p.add_argument('--num-samples', type=int, default=int(get('num_samples', 0)), help='手动指定样本数(0表示自动分配)')

    # English comment for public release.
    p.add_argument('--train-frac', type=float, default=float(get('train_frac', 1)), help='训练集比例')
    p.add_argument('--base-kb-per-sample', type=float, default=float(get('base_kb_per_sample', 20.0)), help='每样本基准KB')
    p.add_argument('--max-samples-per-file', type=int, default=int(get('max_samples_per_file', 30)), help='单文件最大样本数')
    p.add_argument('--min-samples-per-file', type=int, default=int(get('min_samples_per_file', 1)), help='单文件最小样本数')

    # English comment for public release.
    p.add_argument('--output-dir', type=str, default=get('output_dir', None), help='指定结果输出根目录 (默认自动按时间戳生成)')
    p.add_argument('--save-every', type=int, default=int(get('save_every', 50)), help='每N轮保存一次检查点')
    p.add_argument('--vis-every', type=int, default=int(get('vis_every', 1000)), help='每N轮生成一次可视化')
    p.add_argument('--skip-vis', action='store_true', default=bool(get('skip_vis', True)), help='跳过训练结束后的可视化GIF生成（减少冗余操作）')
    p.add_argument('--save-svg', action='store_true', default=bool(get('save_svg', False)), help='同时保存SVG格式')
    p.add_argument('--max-output-points', type=int, default=int(get('max_output_points', 0)), help='生成结果点云的最大点数(0表示不限制)')
    p.add_argument('--max-output-per-file', type=int, default=int(get('max_output_per_file', 3)), help='每个矿体输出的钻井数据对上限，<0 表示全部输出')
    p.add_argument('--output-workers', type=int, default=int(get('output_workers', 12)), help='结果输出线程数，>1 启用多线程并发写盘')
    p.add_argument('--checkpoint', type=str, default=get('checkpoint', r"E:\PyCharm\Project_VAE\results\model\best_model.pth"), help='可选的预训练模型路径；提供时优先加载其超参与权重')
    p.add_argument('--amp', action='store_true', default=bool(get('amp', False)), help='启用 AMP 半精度训练')
    p.add_argument('--log-mode', type=str, default=str(get('log_mode', 'full')), choices=['full', 'brief'], help='日志输出模式（将被 --verbosity 覆盖）')
    p.add_argument('--verbosity', type=str, default=get('verbosity', None), choices=['full', 'brief'], help='日志详细程度开关，优先于 --log-mode')

    # English comment for public release.
    p.add_argument('--num-workers', type=int, default=int(get('num_workers', 0)), help='DataLoader工作进程数')
    p.add_argument('--force-regen-cache', action='store_true', default=bool(get('force_regen_cache', False)), help='强制重新生成体素缓存')
    p.add_argument('--load-mode', type=str, default=str(get('load_mode', 'parallel')), choices=['parallel', 'sequential'], help='数据预处理模式：parallel 使用多进程并行，sequential 逐文件缓存后释放内存')

    # English comment for public release.
    p.add_argument('--generate-only', action='store_true', default=bool(get('generate_only', False)), help='仅生成钻井样本')
    p.add_argument('--ply-file', type=str, default=str(get('ply_file', default_ply)), help='目标PLY文件路径')
    p.add_argument('--ply-dir', type=str, default=str(get('ply_dir', os.path.join(os.path.dirname(__file__), '..', 'data', 'mining_ply_pretrain'))), help='PLY数据集目录')
    p.add_argument('--train-after-generate', action='store_true', default=bool(get('train_after_generate', False)), help='生成后继续训练')

    return p


def parse_args(argv: Optional[Sequence[str]] = None):
    base = argparse.ArgumentParser(add_help=False)
    # English comment for public release.
    base_args, remaining = base.parse_known_args(argv)

    config = load_config(getattr(base_args, 'config', None))
    config_path = os.path.abspath(base_args.config) if getattr(base_args, 'config', None) else '<inline-defaults>'

    parser = build_parser(config)
    parser.add_argument('--config', type=str, default=config_path, help='配置文件路径 (已加载)')
    parser.set_defaults(config=config_path)

    return parser.parse_args(remaining)
