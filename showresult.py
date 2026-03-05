"""Documentation translated to English for open-source release."""

import os
import argparse
import datetime
import numpy as np
import matplotlib

# English comment for public release.
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from dataset import read_ply


def _make_axes_equal(ax, verts: np.ndarray):
	"""Documentation translated to English for open-source release."""
	vmin = verts.min(axis=0)
	vmax = verts.max(axis=0)
	center = (vmin + vmax) / 2.0
	span = (vmax - vmin).max()
	half = span / 2.0
	ax.set_xlim(center[0] - half, center[0] + half)
	ax.set_ylim(center[1] - half, center[1] + half)
	ax.set_zlim(center[2] - half, center[2] + half)


def _render_static(ax, verts: np.ndarray, faces: np.ndarray, face_color='#1f77b4'):
	"""Documentation translated to English for open-source release."""
	mesh = Poly3DCollection(verts[faces], alpha=0.85, facecolor=face_color, edgecolor='k', linewidths=0.2)
	ax.add_collection3d(mesh)
	ax.set_axis_off()
	_make_axes_equal(ax, verts)
	return mesh


def export_spin(verts: np.ndarray, faces: np.ndarray, out_dir: str, base_name: str, frames: int = 60, elev: float = 20.0, dpi: int = 120, save_png: bool = True):
	"""Documentation translated to English for open-source release."""
	os.makedirs(out_dir, exist_ok=True)
	gif_path = os.path.join(out_dir, f"{base_name}_spin.gif")
	png_path = os.path.join(out_dir, f"{base_name}_view.png")

	# English comment for public release.
	screen_dpi = 100
	fig = Figure(figsize=(6, 6), dpi=screen_dpi)
	canvas = FigureCanvasAgg(fig)
	ax = fig.add_subplot(111, projection='3d')
	
	mesh = _render_static(ax, verts, faces)
	if save_png:
		fig.tight_layout()
		fig.savefig(png_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)

	# English comment for public release.
	angles = np.linspace(0, 360, frames, endpoint=False)
	images = []
	for azim in angles:
		ax.view_init(elev=elev, azim=azim)
		canvas.draw()
		# English comment for public release.
		rgba = np.asarray(canvas.buffer_rgba())  # shape: (h, w, 4)
		rgb = rgba[:, :, :3].copy()  # English comment for public release.
		images.append(rgb)

	# English comment for public release.

	# English comment for public release.
	try:
		import imageio

		imageio.mimsave(gif_path, images, format='GIF', duration=0.06)
	except Exception:
		# English comment for public release.
		import matplotlib.animation as animation

		fig2 = Figure(figsize=(6, 6), dpi=screen_dpi)
		FigureCanvasAgg(fig2)
		ax2 = fig2.add_subplot(111, projection='3d')
		_render_static(ax2, verts, faces)

		def _update(angle):
			ax2.view_init(elev=elev, azim=angle)
			return ax2.collections

		anim = animation.FuncAnimation(fig2, _update, frames=angles, interval=60, blit=False)
		writer = animation.PillowWriter(fps=1 / 0.06)
		anim.save(gif_path, writer=writer, dpi=dpi, savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0.05})

	return gif_path, (png_path if save_png else None)


def main():
	default_ply = os.path.join(os.path.dirname(__file__), '..', 'data', 'mining_ply', 'test_b10.ply')
	parser = argparse.ArgumentParser(description='为 PLY 生成旋转展示 GIF（可单文件或目录批量）')
	parser.add_argument('--ply-file', type=str, default=default_ply, help='输入 PLY 文件路径，或包含 .ply 的目录')
	parser.add_argument('--frames', type=int, default=60, help='旋转帧数')
	parser.add_argument('--elev', type=float, default=35.0, help='视角仰角')
	parser.add_argument('--dpi', type=int, default=600, help='输出分辨率 DPI')
	args = parser.parse_args()

	ply_path = os.path.abspath(args.ply_file)
	out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'vis')

	if os.path.isdir(ply_path):
		ply_list = [os.path.join(ply_path, f) for f in os.listdir(ply_path) if f.lower().endswith('.ply')]
		if not ply_list:
			raise FileNotFoundError(f'目录中未找到 PLY 文件: {ply_path}')
		print(f'发现 {len(ply_list)} 个 PLY，批量生成...')
		for p in sorted(ply_list):
			try:
				verts, faces = read_ply(p)
			except ValueError as e:
				print(f'跳过（读入失败）: {p} ({e})')
				continue
			except Exception as e:
				print(f'跳过（异常）: {p} ({e})')
				continue
			if faces.ndim != 2 or faces.shape[1] != 3:
				print(f'跳过（非三角面）: {p}')
				continue
			base_name = os.path.splitext(os.path.basename(p))[0]
			gif_path, png_path = export_spin(verts, faces, out_dir, base_name, frames=args.frames, elev=args.elev, dpi=args.dpi)
			print(f'完成: {base_name} -> GIF: {gif_path}, PNG: {png_path}')
	else:
		if not os.path.isfile(ply_path):
			raise FileNotFoundError(f'未找到 PLY 文件: {ply_path}')
		verts, faces = read_ply(ply_path)
		if faces.ndim != 2 or faces.shape[1] != 3:
			raise ValueError('PLY 文件需要三角形面片 (faces Nx3)')
		base_name = os.path.splitext(os.path.basename(ply_path))[0]
		gif_path, png_path = export_spin(verts, faces, out_dir, base_name, frames=args.frames, elev=args.elev, dpi=args.dpi)
		print(f'旋转 GIF: {gif_path}')
		print(f'静态视图: {png_path}')


if __name__ == '__main__':
	main()
