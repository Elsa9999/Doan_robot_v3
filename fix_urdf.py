import re

with open('urdf/ur5e_final.urdf', 'r', encoding='utf-8') as f:
    content = f.read()

# Thay: file://D:\do_an_robot_v2\...\meshes\ur5e\visual\base.dae
# Thanh: meshes/ur5e/visual/base.dae  (relative, PyBullet tim tu thu muc chua URDF)
def replace_path(m):
    full = m.group(0)
    normalized = full.replace('\\', '/')
    idx = normalized.find('meshes/')
    if idx == -1:
        return full
    return normalized[idx:]

new_content = re.sub(r'file://[^"]+', replace_path, content)

with open('urdf/ur5e_final.urdf', 'w', encoding='utf-8') as f:
    f.write(new_content)

print('Done! Kiem tra:')
for line in new_content.split('\n'):
    if 'mesh' in line.lower() and 'filename' in line.lower():
        print(' ', line.strip())
