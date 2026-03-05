"""Documentation translated to English for open-source release."""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes
from matplotlib import cm, patches
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

# English comment for public release.
# English comment for public release.
try:
    from output_result import _set_cn_font
    _set_cn_font()
except ImportError:
    pass

class PaperPlotter:
    def __init__(self, dpi=300, font_family='sans-serif'):
        """Documentation translated to English for open-source release."""
        self.dpi = dpi
        plt.rcParams['font.family'] = font_family
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = dpi
        
        # English comment for public release.
        # English comment for public release.
        self.color_gt = '#1f77b4'  # Steel Blue
        self.color_pred = '#ff7f0e'  # Safety Orange
        self.color_drill = '#d62728' # Brick Red
        self.cmap_diff = 'RdBu_r'    # English comment for public release.
        self.cmap_conf = 'viridis'   # English comment for public release.

    def _normalize_coords(self, verts, shape, vmin, vmax):
        """Documentation translated to English for open-source release."""
        D, H, W = shape
        z = verts[:, 0] / (D - 1)
        y = verts[:, 1] / (H - 1)
        x = verts[:, 2] / (W - 1)
        
        wx = x * (vmax[0] - vmin[0]) + vmin[0]
        wy = y * (vmax[1] - vmin[1]) + vmin[1]
        wz = z * (vmax[2] - vmin[2]) + vmin[2]
        return np.stack([wx, wy, wz], axis=1)

    def render_3d_iso(self, ax, volume, vmin, vmax, threshold=0.5, color='gray', alpha=0.6, title=None):
        """Documentation translated to English for open-source release."""
        try:
            verts, faces, _, _ = marching_cubes(volume, level=threshold)
            verts_world = self._normalize_coords(verts, volume.shape, vmin, vmax)
            
            mesh = ax.plot_trisurf(
                verts_world[:, 0], verts_world[:, 1], verts_world[:, 2],
                triangles=faces,
                color=color,
                alpha=alpha,
                shade=True,
                edgecolor='none',
                linewidth=0,
                antialiased=True
            )
            
            # English comment for public release.
            ax.set_box_aspect((
                vmax[0]-vmin[0], 
                vmax[1]-vmin[1], 
                vmax[2]-vmin[2]
            ))
            
            # English comment for public release.
            ax.grid(False)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('w')
            ax.yaxis.pane.set_edgecolor('w')
            ax.zaxis.pane.set_edgecolor('w')
            
            if title:
                ax.set_title(title, fontsize=10, y=0.95)
                
            return mesh
        except Exception as e:
            # print(f"ISO Surface render failed: {e}")
            return None

    def render_slices(self, ax, volume, vmin, vmax, axis='z', slice_idx=None, cmap='gray', title=None):
        """Documentation translated to English for open-source release."""
        D, H, W = volume.shape
        
        if axis == 'z':
            idx = slice_idx if slice_idx is not None else D // 2
            slice_data = volume[idx, :, :]
            extent = [vmin[0], vmax[0], vmin[1], vmax[1]] # x, y
            xlabel, ylabel = 'X (m)', 'Y (m)'
        elif axis == 'y':
            idx = slice_idx if slice_idx is not None else H // 2
            slice_data = volume[:, idx, :]
            extent = [vmin[0], vmax[0], vmin[2], vmax[2]] # x, z
            xlabel, ylabel = 'X (m)', 'Z (m)'
        else: # x
            idx = slice_idx if slice_idx is not None else W // 2
            slice_data = volume[:, :, idx]
            extent = [vmin[1], vmax[1], vmin[2], vmax[2]] # y, z
            xlabel, ylabel = 'Y (m)', 'Z (m)'
            
        im = ax.imshow(
            slice_data, 
            cmap=cmap, 
            origin='lower',
            extent=extent,
            vmin=0, vmax=1,
            aspect='auto',
            interpolation='nearest' 
        )
        
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        if title:
            ax.set_title(title, fontsize=10)
        return im

    def render_diff_map(self, ax, pred, gt, vmin, vmax, axis='z', slice_idx=None, title="Difference Error"):
        """Documentation translated to English for open-source release."""
        D, H, W = pred.shape
        diff = pred - gt
        
        if axis == 'z':
            idx = slice_idx if slice_idx is not None else D // 2
            slice_data = diff[idx, :, :]
            extent = [vmin[0], vmax[0], vmin[1], vmax[1]]
            xlabel, ylabel = 'X (m)', 'Y (m)'
        
        # English comment for public release.
        im = ax.imshow(
            slice_data,
            cmap=self.cmap_diff,
            origin='lower',
            extent=extent,
            vmin=-1, vmax=1, # Error is between -1 and 1
            aspect='auto'
        )
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Error')
        return im

    def render_drill_holes(self, ax, obs_volume, vmin, vmax, color='red', size=20):
        """Documentation translated to English for open-source release."""
        if obs_volume is None: 
            return
            
        if obs_volume.ndim == 4:
            mask = obs_volume[1] # Channel 1 is mask
        else:
            mask = obs_volume

        indices = np.argwhere(mask > 0.5)
        if len(indices) == 0:
            return

        D, H, W = mask.shape
        z_idx, y_idx, x_idx = indices[:, 0], indices[:, 1], indices[:, 2]
        
        wx = vmin[0] + (x_idx / max(1, W - 1)) * (vmax[0] - vmin[0])
        wy = vmin[1] + (y_idx / max(1, H - 1)) * (vmax[1] - vmin[1])
        wz = vmin[2] + (z_idx / max(1, D - 1)) * (vmax[2] - vmin[2])

        ax.scatter(wx, wy, wz, c=color, s=size, depthshade=False, label='Drill Constraint', alpha=1.0, edgecolor='white', linewidth=0.5)

def generate_comprehensive_report(gt_vol, pred_vol, obs_vol, vmin, vmax, save_path, epoch_info=""):
    """Documentation translated to English for open-source release."""
    plotter = PaperPlotter()
    
    # GridSpec layout
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.5, 1])
    
    # 1. 3D Ground Truth
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    plotter.render_3d_iso(ax1, gt_vol, vmin, vmax, color=plotter.color_gt, title="Ground Truth (3D)")
    
    # 2. 3D Prediction with Drills
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    plotter.render_3d_iso(ax2, pred_vol, vmin, vmax, color=plotter.color_pred, title=f"Ours Prediction (3D)\n{epoch_info}")
    plotter.render_drill_holes(ax2, obs_vol, vmin, vmax, color=plotter.color_drill)
    
    # 3. 3D Prediction (Angle 2 or Pure Drills) - Let's show Drills Only or Another Angle
    # English comment for public release.
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    # Use empty volume for iso to just setup axis, but here we just scatter
    # ax3.set_title("Input Sparse Drills", fontsize=12)
    plotter.render_drill_holes(ax3, obs_vol, vmin, vmax, color='black', size=15)
    ax3.set_box_aspect((vmax[0]-vmin[0], vmax[1]-vmin[1], vmax[2]-vmin[2]))
    ax3.set_xlim(vmin[0], vmax[0])
    ax3.set_ylim(vmin[1], vmax[1])
    ax3.set_zlim(vmin[2], vmax[2])
    ax3.axis('off')
    ax3.set_title("Input: Sparse Drills", fontsize=12, y=0.95)

    # 4. GT Slice
    ax4 = fig.add_subplot(gs[1, 0])
    plotter.render_slices(ax4, gt_vol, vmin, vmax, axis='z', title="Ground Truth (Slice Z-mid)", cmap='Greys_r')
    
    # 5. Pred Slice
    ax5 = fig.add_subplot(gs[1, 1])
    plotter.render_slices(ax5, pred_vol, vmin, vmax, axis='z', title="Prediction (Slice Z-mid)", cmap='Greys_r')
    
    # 6. Difference Map
    ax6 = fig.add_subplot(gs[1, 2])
    plotter.render_diff_map(ax6, pred_vol, gt_vol, vmin, vmax, axis='z', title="Error Map (Blue=Miss, Red=Over)")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Report saved to: {save_path}")

if __name__ == '__main__':
    # CLI Mode
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    # Add dummy test generation
    args = parser.parse_args()
    
    # Mock data for quick CLI test
    grid = (32, 32, 32)
    vmin, vmax = np.array([0,0,0]), np.array([100,100,100])
    gt = np.random.rand(*grid) > 0.8
    pred = np.random.rand(*grid)
    obs = np.zeros((2, *grid))
    obs[1, 16, 16, :] = 1 # Fake drill
    
    generate_comprehensive_report(gt.astype(float), pred, obs, vmin, vmax, args.out)