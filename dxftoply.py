import os
import sys
import ezdxf
from tqdm import tqdm

# =============================================================================
# English comment for public release.
# English comment for public release.
# English comment for public release.
# English comment for public release.
# English comment for public release.
# English comment for public release.
# 
# English comment for public release.
# English comment for public release.
# =============================================================================

def convert_dxf_to_ply(dxf_path, ply_path):
    """Documentation translated to English for open-source release."""
    try:
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
    except Exception as e:
        print(f"❌ 读取失败 {dxf_path}: {e}")
        return False

    vertices = []
    faces = []
    
    # English comment for public release.
    vert_map = {} 
    next_idx = 0

    def get_vert_idx(v):
        nonlocal next_idx
        # English comment for public release.
        v_tuple = (float(v[0]), float(v[1]), float(v[2]))
        if v_tuple not in vert_map:
            vert_map[v_tuple] = next_idx
            vertices.append(v_tuple)
            next_idx += 1
        return vert_map[v_tuple]

    # English comment for public release.
    # English comment for public release.
    count_3dface = 0
    for e in msp.query('3DFACE'):
        v1 = get_vert_idx(e.dxf.vtx0)
        v2 = get_vert_idx(e.dxf.vtx1)
        v3 = get_vert_idx(e.dxf.vtx2)
        v4 = get_vert_idx(e.dxf.vtx3)
        
        # English comment for public release.
        if v3 == v4:
            faces.append([3, v1, v2, v3])
        else:
            # English comment for public release.
            faces.append([4, v1, v2, v3, v4])
        count_3dface += 1

    # English comment for public release.
    if count_3dface == 0:
        for e in msp.query('POLYLINE'):
            if e.is_poly_face_mesh:
                # English comment for public release.
                # English comment for public release.
                # English comment for public release.
                pass 
                # English comment for public release.

    if len(vertices) == 0:
        print(f"⚠️  {os.path.basename(dxf_path)} 中未发现几何体 (仅支持 3DFACE)")
        return False

    # English comment for public release.
    try:
        with open(ply_path, 'w', encoding='utf-8') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_index\n")
            f.write("end_header\n")
            
            for v in vertices:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")
            
            for face in faces:
                # English comment for public release.
                indices = " ".join(map(str, face[1:]))
                f.write(f"{face[0]} {indices}\n")
    except Exception as e:
        print(f"❌ 写入 PLY 失败: {e}")
        return False
            
    return True

if __name__ == "__main__":
    # English comment for public release.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data', 'ms')
    output_dir = os.path.join(current_dir, '..', 'data', 'mining_ply_pretrains')
    
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📂 创建输出目录: {output_dir}")
        
    files = [f for f in os.listdir(data_dir) if f.lower().endswith('.dxf')]
    print(f"📂 在 {data_dir} 中发现 {len(files)} 个 DXF 文件")
    
    if len(files) == 0:
        print("没有需要转换的文件。")
        sys.exit(0)
    
    success_count = 0
    for f in tqdm(files, desc="正在转换"):
        dxf_path = os.path.join(data_dir, f)
        ply_name = os.path.splitext(f)[0] + '.ply'
        ply_path = os.path.join(output_dir, ply_name)
        
        # English comment for public release.
        if convert_dxf_to_ply(dxf_path, ply_path):
            success_count += 1
            
    print(f"✅ 转换完成! 成功: {success_count}/{len(files)}")
    print(f"📄 输出目录: {output_dir}")
