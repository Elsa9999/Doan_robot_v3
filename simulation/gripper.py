"""
simulation/gripper.py — VacuumGripper simulation bằng PyBullet constraint.
"""
import pybullet as p
import math


class VacuumGripper:
    """
    Mô phỏng gripper hút chân không.
    Dùng JOINT_FIXED constraint để gắn vật vào EE.
    """

    _INDICATOR_LINES = 8    # số điểm vẽ vòng tròn
    _INDICATOR_RADIUS = 0.03

    def __init__(self, robot_id: int, ee_link_index: int):
        self._robot_id   = robot_id
        self._ee_link    = ee_link_index
        self._constraint = None
        self._object_id  = None
        self._activated  = False
        self._debug_lines = []

    # ─── Gripper actions ──────────────────────────────────────────────────────

    def activate(self, object_id: int) -> bool:
        """
        Gắn vật vào EE bằng JOINT_FIXED constraint.
        Tính offset cục bộ để vật không bị nhảy vị trí.
        """
        if self._activated:
            self.release()

        # Lấy pose EE hiện tại (world frame, từ link state index 4,5)
        link_state  = p.getLinkState(self._robot_id, self._ee_link,
                                     computeForwardKinematics=True)
        ee_pos = link_state[4]   # worldLinkFramePosition
        ee_orn = link_state[5]   # worldLinkFrameOrientation

        # Lấy pose vật (world frame)
        obj_pos, obj_orn = p.getBasePositionAndOrientation(object_id)

        # Tính offset vật so với EE trong frame EE
        inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
        obj_local_pos, obj_local_orn = p.multiplyTransforms(
            inv_ee_pos, inv_ee_orn,
            obj_pos,    obj_orn
        )

        self._constraint = p.createConstraint(
            parentBodyUniqueId    = self._robot_id,
            parentLinkIndex       = self._ee_link,
            childBodyUniqueId     = object_id,
            childLinkIndex        = -1,
            jointType             = p.JOINT_FIXED,
            jointAxis             = [0, 0, 0],
            parentFramePosition   = obj_local_pos,
            childFramePosition    = [0, 0, 0],
            parentFrameOrientation= obj_local_orn
        )
        p.changeConstraint(self._constraint, maxForce=500)

        self._object_id = object_id
        self._activated = True
        print(f"[GRIPPER] Activated — holding object {object_id}")
        return True

    def release(self) -> bool:
        """Xóa constraint, vật rơi tự do."""
        if self._constraint is not None:
            p.removeConstraint(self._constraint)
            self._constraint = None

        held = self._object_id
        self._object_id = None
        self._activated = False
        print(f"[GRIPPER] Released — object {held} dropped")
        return True

    # ─── State ────────────────────────────────────────────────────────────────

    def is_activated(self) -> bool:
        return self._activated

    def get_held_object(self):
        return self._object_id

    # ─── Visual indicator ─────────────────────────────────────────────────────

    def draw_indicator(self):
        """Vẽ vòng tròn xanh (active) hoặc đỏ (inactive) quanh EE."""
        # Lấy EE position
        link_state = p.getLinkState(self._robot_id, self._ee_link,
                                    computeForwardKinematics=True)
        ee_pos = list(link_state[4])

        color  = [0, 1, 0] if self._activated else [1, 0, 0]
        r      = self._INDICATOR_RADIUS
        n      = self._INDICATOR_LINES
        pts    = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            pts.append([
                ee_pos[0] + r * math.cos(angle),
                ee_pos[1] + r * math.sin(angle),
                ee_pos[2]
            ])

        # Xóa lines cũ
        for lid in self._debug_lines:
            p.removeUserDebugItem(lid)
        self._debug_lines.clear()

        # Vẽ lines mới
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            lid = p.addUserDebugLine(p1, p2, color, lineWidth=2)
            self._debug_lines.append(lid)

    def clear_indicator(self):
        for lid in self._debug_lines:
            p.removeUserDebugItem(lid)
        self._debug_lines.clear()
