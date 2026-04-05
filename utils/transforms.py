"""
Utility transformations array.
Chuyển đổi tọa độ giữa Robot local frame (DH) và PyBullet world frame.
"""
import math

# Transformation từ Local (FK) ra PyBullet World
# Robot base rotated 180 deg (PI) around Z, offset Z by 0.42
def local_to_world(pos: list, euler: list = None) -> tuple:
    w_pos = [-pos[0], -pos[1], pos[2] + 0.42]
    if euler is not None:
        w_eul = [euler[0], euler[1], euler[2] - math.pi]
        return w_pos, w_eul
    return w_pos

# Transformation từ PyBullet World về Local (để IK)
def world_to_local(w_pos: list, w_euler: list = None) -> tuple:
    l_pos = [-w_pos[0], -w_pos[1], w_pos[2] - 0.42]
    if w_euler is not None:
        l_eul = [w_euler[0], w_euler[1], w_euler[2] + math.pi]
        return l_pos, l_eul
    return l_pos
