
import os
import subprocess
import sys
import shutil
import time

# English comment for public release.

# English comment for public release.
PYTHON_EXE = sys.executable

# English comment for public release.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(PROJECT_ROOT, 'main.py')

# English comment for public release.
RESULTS_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, '../results'))
AUTO_DIR = os.path.join(RESULTS_ROOT, 'auto_analysis') # English comment for public release.

# English comment for public release.
# English comment for public release.
PROPOSED_EPOCHS = 1        # English comment for public release.
PROPOSED_PATIENCE = 20        # English comment for public release.
PROPOSED_CONFIG = [
    '--model-type', 'octree', 
    '--use-lora',              # English comment for public release.
    '--lambda-drill', '50.0'
]

# English comment for public release.
BENCHMARK_EPOCHS = 1         # English comment for public release.
BENCHMARKS = {
    'Standard_VAE': ['--model-type', 'standard', '--lambda-drill', '50.0'],
    'No_Constraint': ['--model-type', 'octree', '--lambda-drill', '0.0'],
    'No_Data_Augmentation': ['--model-type', 'octree', '--augment', '0']
}

# 3. U-Net Baseline
UNET_EPOCHS = 1
UNET_SCRIPT = os.path.join(PROJECT_ROOT, 'comparisons', 'train_unet.py')

# English comment for public release.
# English comment for public release.
ROBUSTNESS_CHECKPOINT = None # English comment for public release.

# English comment for public release.

def run_command(desc, cmd_list, check=True):
    print(f"\n[{time.strftime('%H:%M:%S')}] >>> 开始阶段: {desc}")
    print(f"执行命令: {' '.join(cmd_list)}")
    try:
        subprocess.run(cmd_list, check=check, cwd=PROJECT_ROOT)
        print(f"[{time.strftime('%H:%M:%S')}] >>> 阶段完成: {desc}")
    except subprocess.CalledProcessError as e:
        print(f"[{time.strftime('%H:%M:%S')}] !!! 阶段失败: {desc}")
        print(f"错误信息: {e}")
        if check:
            sys.exit(1)

def find_best_model(result_dir):
    """Documentation translated to English for open-source release."""
    for root, dirs, files in os.walk(result_dir):
        if 'best_model.pth' in files:
            return os.path.join(root, 'best_model.pth')
    return None

def main():
    print("==================================================")
    print("      3D VAE 自动分析流水线 (Auto-Analysis)       ")
    print("==================================================")
    print(f"结果输出目录: {AUTO_DIR}")
    
    # English comment for public release.
    if os.path.exists(AUTO_DIR):
        print(f"清理旧数据: {AUTO_DIR}")
        shutil.rmtree(AUTO_DIR)
    os.makedirs(AUTO_DIR, exist_ok=True)

    # English comment for public release.
    proposed_dir = os.path.join(AUTO_DIR, 'Proposed_Method')
    # English comment for public release.
    os.makedirs(proposed_dir, exist_ok=True)
    best_model_path = os.path.join(proposed_dir, 'checkpoints', 'best_model.pth')

    if not os.path.exists(best_model_path):
        cmd_proposed = [PYTHON_EXE, MAIN_SCRIPT] + [
            '--mode', 'train',
            '--epochs', str(PROPOSED_EPOCHS),
            '--patience', str(PROPOSED_PATIENCE),
            '--save-every', '50',
            '--batch-to-mem', '0.4', # English comment for public release.
            '--load-mode', 'sequential',
            '--num-workers', '0',
            '--output-dir', proposed_dir # English comment for public release.
        ] + PROPOSED_CONFIG
        
        run_command("训练核心模型 (Proposed Method)", cmd_proposed)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] >>> 跳过阶段: 训练核心模型 (模型已存在)")

    # English comment for public release.
    if not os.path.exists(best_model_path):
        print("警告: 未在 Proposed 目录找到 best_model.pth，尝试查找 checkpoints 子目录...")
        best_model_path = find_best_model(os.path.join(proposed_dir, 'checkpoints'))
        
    if best_model_path:
        print(f"锁定最佳模型用于分析: {best_model_path}")
    else:
        print("严重错误: 无法找到训练好的模型文件！")

    # English comment for public release.
    unet_out_dir = os.path.join(AUTO_DIR, 'UNet')
    unet_path = os.path.join(unet_out_dir, 'unet_best.pth')
    if os.path.exists(UNET_SCRIPT):
        if not os.path.exists(unet_path):
            cmd_unet = [PYTHON_EXE, UNET_SCRIPT, 
                        '--epochs', str(UNET_EPOCHS), 
                        '--batch-size', '16',
                        '--output-dir', unet_out_dir]
            run_command("训练 U-Net Baseline", cmd_unet, check=False) # Non-critical if fails
        else:
             print(f"[{time.strftime('%H:%M:%S')}] >>> 跳过阶段: 训练 U-Net Baseline (模型已存在)")
    else:
        print(f"警告: 找不到 U-Net 脚本 {UNET_SCRIPT}，跳过 U-Net 训练")


    # English comment for public release.
    benchmark_root = os.path.join(AUTO_DIR, 'Benchmarks')
    os.makedirs(benchmark_root, exist_ok=True)

    for name, args in BENCHMARKS.items():
        model_dir = os.path.join(benchmark_root, name)
        
        if not os.path.exists(os.path.join(model_dir, 'checkpoints', 'best_model.pth')):
            cmd_bench = [PYTHON_EXE, MAIN_SCRIPT] + [
                '--mode', 'train',
                '--epochs', str(BENCHMARK_EPOCHS),
                '--patience', '10', # English comment for public release.
                '--batch-to-mem', '0.4',
                '--load-mode', 'sequential',
                '--num-workers', '0',
                '--output-dir', model_dir
            ] + args
            
            run_command(f"训练对比模型: {name}", cmd_bench)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] >>> 跳过阶段: 训练对比模型: {name} (模型已存在)")
        
    # English comment for public release.
    if best_model_path:
        # English comment for public release.
        # English comment for public release.
        
        cmd_robust = [PYTHON_EXE, MAIN_SCRIPT] + [
            '--mode', 'robust_eval',
            '--checkpoint', best_model_path,
            '--ply-dir', os.path.join(PROJECT_ROOT, '../data/mining_ply_pretrain'),
            '--batch-to-mem', '0.2',
            '--num-workers', '0'
        ]
        
        run_command("执行鲁棒性分析 (Proposed Method)", cmd_robust)
    else:
        print("跳过鲁棒性分析 (未找到最佳模型)")

    # English comment for public release.
    print("\n[Stage 4] 生成综合对比报表...")
    benchmark_script = os.path.join(PROJECT_ROOT, 'comparisons', 'run_benchmark.py')
    benchmark_out_dir = os.path.join(AUTO_DIR, 'Benchmark_Summary')
    
    cmd_summary = [PYTHON_EXE, benchmark_script]
    # Pass output dir
    cmd_summary += ['--output-dir', benchmark_out_dir]
    
    if best_model_path:
        cmd_summary += ['--vae-ckpt', best_model_path]
    
    # U-Net path (updated explicitly)
    unet_path = os.path.join(AUTO_DIR, 'UNet', 'unet_best.pth')
    if os.path.exists(unet_path):
        cmd_summary += ['--unet-ckpt', unet_path]
    else:
        # Fallback to old path if not found (e.g. if skipped)
        unet_old = os.path.join(RESULTS_ROOT, 'comparisons', 'models', 'unet_best.pth')
        if os.path.exists(unet_old):
            cmd_summary += ['--unet-ckpt', unet_old]
        
    cmd_summary += ['--limit', '100'] # Test 100 samples
    
    run_command("综合对比基准测试 (All Models)", cmd_summary, check=False)
    
    print("\n==================================================")
    print("      自动流水线完成! (Auto-Analysis Protocol)    ")
    print("==================================================")
    print(f"1. 核心模型结果: {proposed_dir}")
    print(f"2. 对比模型结果: {benchmark_root}")
    print(f"3. 鲁棒性分析:   查看 {proposed_dir} 下的 CSV/PNG") 
    print(f"4. 最终报表:     {benchmark_out_dir}")
    print("\n建议: 您可以手动运行 plot_ablation_summary.py 并修改 RESULTS_DIR 指向:")
    print(f"{benchmark_root}")

if __name__ == '__main__':
    main()
