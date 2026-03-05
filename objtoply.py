import os
import sys
from typing import List
from tqdm import tqdm


def _parse_face_indices(tokens: List[str], v_count: int) -> List[int]:
	"""Documentation translated to English for open-source release."""
	idx = []
	for tok in tokens:
		parts = tok.split('/')
		raw = int(parts[0])
		pos = raw + v_count if raw < 0 else raw
		idx.append(pos - 1)
	return idx


def _triangulate(indices: List[int]) -> List[List[int]]:
	"""Documentation translated to English for open-source release."""
	if len(indices) < 3:
		return []
	if len(indices) == 3:
		return [indices]
	tris = []
	for i in range(1, len(indices) - 1):
		tris.append([indices[0], indices[i], indices[i + 1]])
	return tris


def convert_obj_to_ply(obj_path: str, ply_path: str) -> bool:
	"""Documentation translated to English for open-source release."""
	vertices: List[List[float]] = []
	faces: List[List[int]] = []

	try:
		with open(obj_path, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip()
				if not line or line.startswith('#'):
					continue
				if line.startswith('v '):
					parts = line.split()
					if len(parts) < 4:
						continue
					vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
				elif line.startswith('f '):
					parts = line.split()[1:]
					idx = _parse_face_indices(parts, len(vertices))
					faces.extend(_triangulate(idx))
	except Exception as e:
		print(f"❌ 读取 OBJ 失败 {obj_path}: {e}")
		return False

	if not vertices or not faces:
		print(f"⚠️  跳过（无有效顶点或面）: {obj_path}")
		return False

	try:
		with open(ply_path, 'w', encoding='utf-8') as f:
			f.write('ply\n')
			f.write('format ascii 1.0\n')
			f.write(f'element vertex {len(vertices)}\n')
			f.write('property float x\n')
			f.write('property float y\n')
			f.write('property float z\n')
			f.write(f'element face {len(faces)}\n')
			f.write('property list uchar int vertex_index\n')
			f.write('end_header\n')

			for v in vertices:
				f.write(f"{v[0]} {v[1]} {v[2]}\n")

			for tri in faces:
				f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")
	except Exception as e:
		print(f"❌ 写入 PLY 失败 {ply_path}: {e}")
		return False

	return True


if __name__ == '__main__':
	current_dir = os.path.dirname(os.path.abspath(__file__))
	data_dir = os.path.join(current_dir, '..', 'data', '3dtest')
	output_dir = os.path.join(current_dir, '..', 'data', 'mining_ply')

	if not os.path.exists(data_dir):
		print(f"❌ 数据目录不存在: {data_dir}")
		sys.exit(1)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
		print(f"📂 创建输出目录: {output_dir}")

	obj_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.obj')]
	print(f"📂 在 {data_dir} 中发现 {len(obj_files)} 个 OBJ 文件")
	if not obj_files:
		print('没有需要转换的 OBJ。')
		sys.exit(0)

	success = 0
	for fname in tqdm(sorted(obj_files), desc='正在转换 OBJ'):
		obj_path = os.path.join(data_dir, fname)
		stem = os.path.splitext(fname)[0]
		ply_name = f"test_{stem}.ply"
		ply_path = os.path.join(output_dir, ply_name)
		if convert_obj_to_ply(obj_path, ply_path):
			success += 1

	print(f"✅ 转换完成: {success}/{len(obj_files)}")
	print(f"📄 输出目录: {output_dir}")
