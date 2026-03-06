
import os
import subprocess
import sys
import pandas as pd
import glob

import shutil

# Configuration
PYTHON_EXE = sys.executable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 3dvae/
MAIN_SCRIPT = os.path.join(ROOT_DIR, 'main.py')
RESULT_BASE = os.path.join(ROOT_DIR, '../results/ablations')
RESULTS_3DVAE_ROOT = os.path.join(ROOT_DIR, '../results/3dvae')

# Common Arguments
COMMON_ARGS = [
    '--mode', 'train',        # Explicitly set mode to train
    '--epochs', '50',         # Reduced for ablation speed (normally 300)
    '--patience', '10',
    '--num-workers', '0',     # Set to 0 for Windows stability (avoids pickling MemoryError)
    '--load-mode', 'sequential', # Use sequential loading to avoid multiprocessing errors on Windows
    '--batch-to-mem', '0.3',
    '--cuda',
    '--save-every', '50',
    '--skip-vis'
]

EXPERIMENTS = {
    'Baseline': {
        'args': ['--model-type', 'octree', '--lambda-drill', '50.0', '--kl-ratio', '0.5']
    },
    'No_Constraint': {
        'args': ['--model-type', 'octree', '--lambda-drill', '0.0', '--kl-ratio', '0.5']
    },
    'No_Octree': {
        'args': ['--model-type', 'standard', '--lambda-drill', '50.0', '--kl-ratio', '0.5']
    },
    'No_KL': {
        'args': ['--model-type', 'octree', '--lambda-drill', '50.0', '--kl-ratio', '0.0']
    }
}

def run_experiment(name, specific_args):
    print(f"==================================================")
    print(f"translated_text: {name}")
    print(f"==================================================")
    
    # 1. Snapshot existing output directories
    os.makedirs(RESULTS_3DVAE_ROOT, exist_ok=True)
    existing_dirs = set(os.listdir(RESULTS_3DVAE_ROOT))
    
    # 2. Run Training
    cmd = [PYTHON_EXE, MAIN_SCRIPT] + COMMON_ARGS + specific_args
    print(f"translated_text: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd, cwd=ROOT_DIR)
        
        # 3. Find and Move Result
        current_dirs = set(os.listdir(RESULTS_3DVAE_ROOT))
        new_dirs = current_dirs - existing_dirs
        
        if new_dirs:
            # Sort by creation time just in case multiple appeared
            newest_dir = max([os.path.join(RESULTS_3DVAE_ROOT, d) for d in new_dirs], key=os.path.getmtime)
            
            dst_path = os.path.join(RESULT_BASE, name)
            if os.path.exists(dst_path):
                print(f"translated_text: {dst_path}")
                shutil.rmtree(dst_path)
                
            print(f"translated_text {newest_dir} translated_text {dst_path}")
            shutil.move(newest_dir, dst_path)
        else:
            print("translated_text: translated_text results/3dvae/ translated_text")
            
        print(f"translated_text {name} translated_text.")
        
    except subprocess.CalledProcessError as e:
        print(f"translated_text {name} translated_text, translated_text: {e}")

def main():
    print("translated_text (Ablation Studies)...")
    print(f"translated_text: {RESULT_BASE}")
    
    os.makedirs(RESULT_BASE, exist_ok=True)
    
    for name, config in EXPERIMENTS.items():
        run_experiment(name, config['args'])
        
    print("translated_text.")

    # Call the plotting script to generate the summary plot automatically
    try:
        from . import plot_ablation_summary
        print("\ntranslated_text...")
        plot_ablation_summary.main()
    except ImportError:
        # Fallback if relative import fails when running as script
        try:
            import plot_ablation_summary
            print("\ntranslated_text...")
            plot_ablation_summary.main()
        except Exception as e:
            print(f"\ntranslated_text: {e}")
            print("translated_text: python 3dvae/comparisons/plot_ablation_summary.py")

if __name__ == "__main__":
    main()

