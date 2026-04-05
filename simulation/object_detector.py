"""
simulation/object_detector.py — Detect vật thể và tính pick/place poses.
"""
import pybullet as p


class ObjectDetector:
    """
    Phát hiện vật thể bằng getBasePosition + raycast visualization.
    """

    def __init__(self, env):
        self._env     = env
        self._ray_ids = []   # debug line IDs ray hiện tại (max 1)

    # ─── Pose query ───────────────────────────────────────────────────────────

    def get_object_pose(self, object_id: int) -> dict:
        """Lấy vị trí + orientation vật trực tiếp từ PyBullet."""
        pos, orn = p.getBasePositionAndOrientation(object_id)
        euler    = p.getEulerFromQuaternion(orn)
        return {
            'pos':   list(pos),
            'orn':   list(orn),
            'euler': list(euler)
        }

    # ─── Raycast ──────────────────────────────────────────────────────────────

    def raycast_detect(self, ee_pos: list, max_dist: float = 0.5) -> dict:
        """
        Bắn tia từ EE thẳng xuống, visualize kết quả.
        Return dict nếu hit, None nếu không.
        """
        ray_start = list(ee_pos)
        ray_end   = [ee_pos[0], ee_pos[1], ee_pos[2] - max_dist]

        result        = p.rayTest(ray_start, ray_end)[0]
        hit_object_id = result[0]
        hit_pos       = list(result[3]) if result[3] else ray_end

        hit = hit_object_id > 0
        self._draw_ray(ray_start, ray_end,
                       color=[0, 1, 0] if hit else [1, 0, 0])

        if not hit:
            return None

        return {
            'object_id': hit_object_id,
            'hit_pos':   hit_pos,
            'distance':  ee_pos[2] - hit_pos[2]
        }

    def _draw_ray(self, start, end, color):
        """Xóa ray cũ, vẽ mới."""
        for lid in self._ray_ids:
            try:
                p.removeUserDebugItem(lid)
            except Exception:
                pass
        self._ray_ids.clear()
        lid = p.addUserDebugLine(start, end, color, lineWidth=1)
        self._ray_ids.append(lid)

    def clear_ray(self):
        for lid in self._ray_ids:
            try: p.removeUserDebugItem(lid)
            except: pass
        self._ray_ids.clear()

    # ─── Pose computation ─────────────────────────────────────────────────────

    def compute_pick_poses(self,
                           object_pos: list,
                           approach_height: float = 0.15,
                           pick_clearance: float  = 0.01) -> dict:
        """Tính approach / pick / lift poses từ vị trí vật."""
        x, y, z = object_pos
        return {
            'approach': [x, y, z + approach_height],
            'pick':     [x, y, z + pick_clearance],
            'lift':     [x, y, z + 0.20],
        }

    def compute_place_poses(self,
                            bin_center: list,
                            place_height: float = 0.15) -> dict:
        """Tính above_bin / place poses từ tâm bin."""
        x, y, z = bin_center
        return {
            'above_bin': [x, y, z + place_height],
            'place':     [x, y, z + 0.08],
        }
