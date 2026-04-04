import os
import shutil
import xml.etree.ElementTree as ET
from xacrodoc import XacroDoc

def main():
    print("[convert_urdf.py] Bắt đầu convert xacro sang URDF cho UR5e")
    
    repo_dir = "Universal_Robots_ROS2_Description"
    xacro_file = os.path.join(repo_dir, "urdf", "ur.urdf.xacro")
    output_urdf = os.path.join("urdf", "ur5e_final.urdf")
    meshes_dest_dir = os.path.join("urdf", "meshes")
    meshes_src_dir = os.path.join(repo_dir, "meshes")
    
    print(f"[convert_urdf.py] Đọc file xacro: {xacro_file}")
    
    doc = XacroDoc.from_file(
        xacro_file, 
        subargs={"name": "ur", "ur_type": "ur5e"}
    )
    urdf_string = doc.to_urdf_string()
    
    print("[convert_urdf.py] Chỉnh sửa đường dẫn mesh")
    try:
        root = ET.fromstring(urdf_string)
    except ET.ParseError as e:
        print(f"[convert_urdf.py] Lỗi parse URDF sinh ra: {e}")
        return

    import re
    count_mesh = 0
    for mesh in root.iter('mesh'):
        filename = mesh.get('filename')
        if filename and ('package://' in filename or 'file://' in filename):
            new_filename = re.sub(r'(package|file)://.*?(?:/|\\\\)meshes(?:/|\\\\)', 'meshes/', filename)
            new_filename = new_filename.replace('\\\\', '/')
            mesh.set('filename', new_filename)
            count_mesh += 1
            
    print(f"[convert_urdf.py] Đã cập nhật {count_mesh} đường dẫn mesh")
    
    print(f"[convert_urdf.py] Lưu URDF tại {output_urdf}")
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    
    print(f"[convert_urdf.py] Copy meshes từ {meshes_src_dir} sang {meshes_dest_dir}")
    if os.path.exists(meshes_dest_dir):
        shutil.rmtree(meshes_dest_dir)
    shutil.copytree(meshes_src_dir, meshes_dest_dir)
    print(f"[convert_urdf.py] Copy mesh hoàn tất")
    
    print("[convert_urdf.py] Validate file XML...")
    try:
        ET.parse(output_urdf)
        print("[convert_urdf.py] File URDF hợp lệ, cấu trúc XML đúng.")
    except ET.ParseError as e:
        print(f"[convert_urdf.py] LỖI: URDF không hợp lệ - {e}")
        
    print("[convert_urdf.py] Xong!")

if __name__ == "__main__":
    main()
