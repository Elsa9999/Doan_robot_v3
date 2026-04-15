"""
Tệp environment.py: 
Đóng vai trò là "Thế giới Vật lý Ảo" (Simulation Engine) cho toàn bộ đồ án.
Mục đích:
- Khởi tạo và thiết lập môi trường PyBullet (Trọng lực, Tốc độ khung hình 240Hz).
- Tạo ra các đối tượng vật lý gồm: Sàn nhà, Bàn làm việc, Thùng rác, Cục mút di động.
- Nạp (Load) mô hình động học Robot UR5e và kết nối các khớp (joints).
- Cung cấp các hàm tương tác vật lý (Điều khiển tốc độ/vị trí khớp, di chuyển bằng Cartesian).
Note: Không dùng `resetJointState` ở thời gian thực (runtime), mà dùng `POSITION_CONTROL` để robot có khả năng va chạm vật lý (Collision).
"""
from pathlib import Path
import numpy as np
import pybullet as p
import pybullet_data
import time
import random
import math

# Đường dẫn cố định dựa trên vị trí file này, không phụ thuộc working directory
_SIM_DIR  = Path(__file__).resolve().parent
_URDF_DIR = _SIM_DIR.parent / "urdf"

# ─── CẤU HÌNH THÔNG SỐ VẬT LÝ VÀ ĐỊA HÌNH ───
TABLE_HEIGHT  = 0.40      # Chiều cao của chân bàn (mét)
TABLE_THICK   = 0.02      # Độ dày của mặt bàn 
TABLE_SURFACE = 0.42      # Mặt phẳng làm việc (Chiều cao z = 0.40 + 0.02)

ROBOT_BASE    = [0.0, 0.0, TABLE_SURFACE]  # Tọa độ gốc đặt Robot (ngay trên mặt bàn)
# Tư thế nghỉ (Home Pose) của robot tính bằng góc radian ở 6 khớp
HOME_POSE     = [0, -1.5708, 1.5708, -1.5708, -1.5708, 0]

# Khu vực rải ngẫu nhiên khối mút (x: 0.3 -> 0.7, y: -0.15 -> 0.15)
WORK_ZONE = {'x': (0.3, 0.7), 'y': (-0.15, 0.15)}

# Thùng chứa mục tiêu (Sọt rác)
BIN_CENTER = [0.65, -0.28, TABLE_SURFACE]   # Trọng tâm tọa độ Thùng (x, y, z trên mặt bàn)
BIN_HALF   = [0.096, 0.071]                  # Nửa chiều rộng / nửa chiều sâu để tính toán giới hạn va chạm

# Workspace bounds cho EE (Cartesian control)
EE_WORKSPACE = {
    'x': (0.20, 0.75),
    'y': (-0.30, 0.30),
    'z': (0.44, 0.95),   # trên mặt bàn (TABLE_SURFACE=0.42) đến tầm tay
}

CART_DELTA_MAX = 0.05   # m/step — 5cm tối đa mỗi bước

CAMERA = {
    'distance': 1.2,
    'yaw':      45,
    'pitch':    -35,
    'target':   [0.5, 0.0, 0.3]
}

# PD gains + max force per joint
JOINT_PD = [
    {'kp': 0.3, 'kd': 1.0, 'force': 150},  # J1 Shoulder Pan
    {'kp': 0.3, 'kd': 1.0, 'force': 150},  # J2 Shoulder Lift
    {'kp': 0.3, 'kd': 1.0, 'force': 150},  # J3 Elbow
    {'kp': 0.2, 'kd': 0.8, 'force':  28},  # J4 Wrist 1
    {'kp': 0.2, 'kd': 0.8, 'force':  28},  # J5 Wrist 2
    {'kp': 0.2, 'kd': 0.8, 'force':  28},  # J6 Wrist 3
]


class UR5eEnvironment:
    """Lớp đối tượng chịu trách nhiệm xây dựng và quản lý toàn bộ Thế giới Vật lý của Robot UR5e."""

    def __init__(self, gui=True):
        # 1. Khởi tạo PyBullet Client
        # Nếu chạy có giao diện (GUI) sẽ tạo cửa sổ đồ họa, nếu không sẽ chạy ngầm (DIRECT) - tốt cho training RL
        if gui:
            self._physics_client = p.connect(p.GUI)
        else:
            self._physics_client = p.connect(p.DIRECT)

        # 2. Tuỳ chỉnh Vật lý cơ bản
        p.setGravity(0, 0, -9.81)    # Lực hấp dẫn của Trái Đất (Z hướng xuống)
        p.setTimeStep(1 / 240)       # Tần số mô phỏng chuẩn Mỹ: 240 khung hình / 1 giây
        p.setRealTimeSimulation(0)   # Chạy Step by Step (thay vì tự động Realtime) để ta có toàn quyền kiểm soát vòng lặp
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.resetDebugVisualizerCamera(
            cameraDistance=CAMERA['distance'],
            cameraYaw=CAMERA['yaw'],
            cameraPitch=CAMERA['pitch'],
            cameraTargetPosition=CAMERA['target']
        )

        print("[ENV] Init PyBullet Environment")
        self._table_ids = []
        self._bin_ids = []
        self._object_id = -1
        self._constraint_id = -1
        self._work_zone_lines = []

        self._load_ground()
        self._load_table()
        self._load_robot()
        self._load_bin()
        self._spawn_object()
        self.draw_work_zone()

    # ─── Scene loading ────────────────────────────────────────────────────────

    def _load_ground(self):
        print("[ENV] Loading ground...")
        self._ground_id = p.loadURDF("plane.urdf")

    def _load_table(self):
        print("[ENV] Loading table...")
        color = [0.55, 0.35, 0.10, 1.0]

        shape_surface = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.45, 0.30, TABLE_THICK / 2])
        vis_surface   = p.createVisualShape(p.GEOM_BOX,   halfExtents=[0.45, 0.30, TABLE_THICK / 2], rgbaColor=color)
        surface_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=shape_surface,
            baseVisualShapeIndex=vis_surface,
            basePosition=[0.25, 0.0, TABLE_HEIGHT + TABLE_THICK / 2]
        )
        p.changeDynamics(surface_id, -1, lateralFriction=1.0)
        self._table_ids.append(surface_id)

        leg_half = [0.02, 0.02, TABLE_HEIGHT / 2]
        shape_leg = p.createCollisionShape(p.GEOM_BOX, halfExtents=leg_half)
        vis_leg   = p.createVisualShape(p.GEOM_BOX,   halfExtents=leg_half, rgbaColor=color)
        for ox, oy in [(-0.18, 0.27), (-0.18, -0.27), (0.68, 0.27), (0.68, -0.27)]:
            leg_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=shape_leg,
                baseVisualShapeIndex=vis_leg,
                basePosition=[ox, oy, TABLE_HEIGHT / 2]
            )
            p.changeDynamics(leg_id, -1, lateralFriction=1.0)
            self._table_ids.append(leg_id)

    def _load_robot(self):
        print("[ENV] Loading robot...")
        _urdf = _URDF_DIR / "ur5e_final.urdf"
        if not _urdf.exists():
            raise FileNotFoundError(
                f"[ENV] URDF not found: {_urdf}\n"
                f"  Expected in: {_URDF_DIR}"
            )
        self._robot_id = p.loadURDF(
            str(_urdf),
            basePosition=ROBOT_BASE,
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION
        )

        self._joint_indices = []
        joint_names = []
        for i in range(p.getNumJoints(self._robot_id)):
            info = p.getJointInfo(self._robot_id, i)
            if info[2] == p.JOINT_REVOLUTE:
                self._joint_indices.append(i)
                joint_names.append(info[1].decode('utf-8'))

        print(f"[ENV] Found {len(self._joint_indices)} revolute joints: {joint_names}")

        # Init tại home pose (teleport OK lúc init)
        self.teleport_joints(HOME_POSE)

    def _load_bin(self):
        print("[ENV] Loading bin...")
        color = [0.25, 0.25, 0.25, 1.0]

        def create_wall(halfExt, local_pos):
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=halfExt)
            vis   = p.createVisualShape(p.GEOM_BOX,   halfExtents=halfExt, rgbaColor=color)
            world_pos = [BIN_CENTER[0] + local_pos[0],
                         BIN_CENTER[1] + local_pos[1],
                         BIN_CENTER[2] + local_pos[2]]
            body_id = p.createMultiBody(0, shape, vis, world_pos)
            p.changeDynamics(body_id, -1, lateralFriction=0.5)
            self._bin_ids.append(body_id)

        create_wall([0.096, 0.071, 0.004], [0, 0, 0.004])
        create_wall([0.004, 0.071, 0.035], [-0.096, 0, 0.039])
        create_wall([0.004, 0.071, 0.035], [ 0.096, 0, 0.039])
        create_wall([0.096, 0.004, 0.035], [0,  0.071, 0.039])
        create_wall([0.096, 0.004, 0.035], [0, -0.071, 0.039])

    def _spawn_object(self, pos=None, difficulty=2):
        """Spawn object với curriculum difficulty.
        difficulty=0: gần EE home (r ≤ 0.15m) — dễ
        difficulty=1: vừa (r ≤ 0.25m)           — vừa
        difficulty=2: full random trong WORK_ZONE  — khó
        """
        if pos is None:
            if difficulty == 0:
                # Spawn gần EE home position (~0.45, 0, table) trong bán kính 0.12~0.25m
                # r phải > grasp_threshold (0.035) rất nhiều để tránh reward hacking
                EE_HOME_XY = [0.45, 0.0]
                r   = random.uniform(0.12, 0.25)  # minimum 12cm, agent PHẢI di chuyển
                ang = random.uniform(0, 2 * 3.14159)
                x   = EE_HOME_XY[0] + r * math.cos(ang)
                y   = EE_HOME_XY[1] + r * math.sin(ang)
                x   = max(WORK_ZONE['x'][0] + 0.05, min(WORK_ZONE['x'][1] - 0.05, x))
                y   = max(WORK_ZONE['y'][0] + 0.05, min(WORK_ZONE['y'][1] - 0.05, y))
            elif difficulty == 1:
                # Spawn trong bán kính 0.25m từ EE home
                EE_HOME_XY = [0.45, 0.0]
                r   = random.uniform(0.05, 0.25)
                ang = random.uniform(0, 2 * 3.14159)
                x   = EE_HOME_XY[0] + r * math.cos(ang)
                y   = EE_HOME_XY[1] + r * math.sin(ang)
                x   = max(WORK_ZONE['x'][0] + 0.05, min(WORK_ZONE['x'][1] - 0.05, x))
                y   = max(WORK_ZONE['y'][0] + 0.05, min(WORK_ZONE['y'][1] - 0.05, y))
            else:
                # Full random — giống bản cũ
                x = random.uniform(WORK_ZONE['x'][0] + 0.05, WORK_ZONE['x'][1] - 0.05)
                y = random.uniform(WORK_ZONE['y'][0] + 0.05, WORK_ZONE['y'][1] - 0.05)
            pos = [x, y, TABLE_SURFACE + 0.033]

        # Nâng cấp 17D: Random dáng nằm của vật thể (Đứng hoặc Lăn lóc ngang)
        # Nằm ngang: Roll = 90 độ (pi/2) hoặc Pitch = 90 độ (pi/2). Yaw random.
        quat = [0, 0, 0, 1] # Đứng thẳng (mặc định)
        if difficulty >= 1 and random.random() < 0.5:
            # 50% cơ hội đổ ngang trên bàn
            roll = math.pi / 2
            pitch = 0
            yaw = random.uniform(0, 2 * math.pi)
            quat = p.getQuaternionFromEuler([roll, pitch, yaw])
            # Khi đổ ngang, trung tâm Z sẽ thấp hơn (bằng radius thay vì half-height)
            pos[2] = TABLE_SURFACE + 0.02

        shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.02, height=0.06)
        vis   = p.createVisualShape(p.GEOM_CYLINDER,   radius=0.02, length=0.06,
                                    rgbaColor=[0.1, 0.3, 0.9, 1.0])
        self._object_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=shape,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            baseOrientation=quat
        )
        p.changeDynamics(self._object_id, -1,
                         lateralFriction=0.8, spinningFriction=0.05, rollingFriction=0.01)
        self.step(50)

    def draw_work_zone(self):
        for line_id in self._work_zone_lines:
            p.removeUserDebugItem(line_id)
        self._work_zone_lines.clear()

        z = TABLE_SURFACE + 0.002
        A = [WORK_ZONE['x'][0], WORK_ZONE['y'][0], z]
        B = [WORK_ZONE['x'][1], WORK_ZONE['y'][0], z]
        C = [WORK_ZONE['x'][1], WORK_ZONE['y'][1], z]
        D = [WORK_ZONE['x'][0], WORK_ZONE['y'][1], z]
        for a, b_ in [(A, B), (B, C), (C, D), (D, A)]:
            self._work_zone_lines.append(p.addUserDebugLine(a, b_, [1, 0, 0], 2))

    # ─── Joint control ────────────────────────────────────────────────────────

    def teleport_joints(self, q):
        """Instant teleport — CHỈ dùng lúc init/reset, không có collision."""
        for i, idx in enumerate(self._joint_indices):
            p.resetJointState(self._robot_id, idx, q[i])
            p.setJointMotorControl2(
                self._robot_id, idx,
                p.POSITION_CONTROL,
                targetPosition=q[i],
                force=500
            )

    def set_joint_positions(self, q, max_velocity=1.0):
        """Runtime dynamics — có collision, mượt."""
        for i, idx in enumerate(self._joint_indices):
            pd = JOINT_PD[i]
            p.setJointMotorControl2(
                self._robot_id, idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=q[i],
                targetVelocity=0,
                positionGain=pd['kp'],
                velocityGain=pd['kd'],
                force=pd['force'],
                maxVelocity=max_velocity
            )

    def get_joint_positions(self) -> list:
        return [p.getJointState(self._robot_id, idx)[0]
                for idx in self._joint_indices]

    def get_joint_velocities(self) -> list:
        return [p.getJointState(self._robot_id, idx)[1]
                for idx in self._joint_indices]

    def move_ee_cartesian(self, delta_xyz, delta_euler=None) -> bool:
        """Cartesian EE control dùng PyBullet built-in IK.
        Hỗ trợ 6D (Vị trí + Lật góc).
        Returns: True nếu IK thành công.
        """
        ee_link  = self._joint_indices[-1]
        link_state = p.getLinkState(self._robot_id, ee_link)
        cur_pos  = list(link_state[0])
        cur_quat = list(link_state[1])

        # Tính target position + clip vào workspace
        target = [
            float(np.clip(cur_pos[0] + delta_xyz[0],
                          EE_WORKSPACE['x'][0], EE_WORKSPACE['x'][1])),
            float(np.clip(cur_pos[1] + delta_xyz[1],
                          EE_WORKSPACE['y'][0], EE_WORKSPACE['y'][1])),
            float(np.clip(cur_pos[2] + delta_xyz[2],
                          EE_WORKSPACE['z'][0], EE_WORKSPACE['z'][1])),
        ]

        # UR5e joint limits (radians) — giới hạn vật lý của robot
        _ll = [-2*np.pi, -2*np.pi, -np.pi,   -2*np.pi, -2*np.pi, -2*np.pi]
        _ul = [ 2*np.pi,  2*np.pi,  np.pi,    2*np.pi,  2*np.pi,  2*np.pi]
        _jr = [ 4*np.pi,  4*np.pi,  2*np.pi,  4*np.pi,  4*np.pi,  4*np.pi]  # range = ul-ll

        target_quat = None
        if delta_euler is not None:
            cur_euler = p.getEulerFromQuaternion(cur_quat)
            target_euler = [
                cur_euler[0] + delta_euler[0],
                cur_euler[1] + delta_euler[1],
                cur_euler[2] + delta_euler[2],
            ]
            target_quat = p.getQuaternionFromEuler(target_euler)

        # PyBullet built-in IK với joint limit constraints
        if target_quat is not None:
            ik = p.calculateInverseKinematics(
                self._robot_id, ee_link, target,
                targetOrientation=target_quat,
                lowerLimits=_ll,
                upperLimits=_ul,
                jointRanges=_jr,
                restPoses=list(HOME_POSE),
                maxNumIterations=100,
                residualThreshold=1e-4,
            )
        else:
            ik = p.calculateInverseKinematics(
                self._robot_id, ee_link, target,
                lowerLimits=_ll,
                upperLimits=_ul,
                jointRanges=_jr,
                restPoses=list(HOME_POSE),   # Bias về home pose → tránh singularity
                maxNumIterations=100,
                residualThreshold=1e-4,
            )

        # PyBullet trả về tuple length = số DOF, thứ tự theo _joint_indices
        # IK đôi khi trả về equivalent angle rất lớn (VD: 25 rad thay vì ~0.9 rad)
        # → Normalize về góc tương đương gần nhất với joint hiện tại
        current_q = [p.getJointState(self._robot_id, idx)[0]
                     for idx in self._joint_indices]
        q_target = []
        for i, cur in enumerate(current_q):
            raw = float(ik[i])
            # Map raw → góc equiv gần cur nhất (tránh unwinding nhiều vòng)
            diff = (raw - cur + np.pi) % (2 * np.pi) - np.pi
            q_target.append(float(np.clip(cur + diff, -2*np.pi, 2*np.pi)))

        self.set_joint_positions(q_target)
        return True

    def get_ee_position(self) -> list:
        """Trả về EE position (xyz) — shortcut không cần unpack tuple."""
        ee_link = self._joint_indices[-1]
        return list(p.getLinkState(self._robot_id, ee_link)[0])

    # ─── Public API ───────────────────────────────────────────────────────────

    def reset(self, difficulty=2):
        self.release_gripper()  # Ensure object is dropped before resetting

        if self._object_id != -1:
            p.removeBody(self._object_id)
            self._object_id = -1
        self.teleport_joints(HOME_POSE)
        self.step(10)
        self._spawn_object(difficulty=difficulty)
        self.draw_work_zone()
        self.step(100)

    def get_robot_id(self)    -> int:   return self._robot_id
    def get_joint_indices(self) -> list: return self._joint_indices
    def get_object_id(self)   -> int:   return self._object_id

    def get_object_pose(self) -> tuple:
        return p.getBasePositionAndOrientation(self._object_id)

    def get_ee_pose(self) -> tuple:
        ee_link = self._joint_indices[-1]
        state = p.getLinkState(self._robot_id, ee_link)
        return state[0], state[1]

    # ─── Virtual Gripper API ──────────────────────────────────────────────────
    
    def activate_gripper(self) -> bool:
        """Create a fixed constraint between EE and Object if close enough."""
        if self._constraint_id != -1:
            return True  # Already gripping

        ee_pos = self.get_ee_position()
        obj_pos = self.get_object_pose()[0]
        dist = np.linalg.norm(np.array(ee_pos) - np.array(obj_pos))
        
        # Mở rộng vùng hút lên 4.5cm. Do khi lật ngang 17D, điểm trung tâm của xylanh và EE 
        # có thể bị chệch một chút (do thành trụ cấn vật lý cản lại). 4.5cm mô phỏng độ nén của giác hút cao su.
        if dist < 0.045:
            ee_link = self._joint_indices[-1]
            
            # [HOTFIX VĂNG VẬT] Tính toán ma trận biến đổi tương đối để bù trừ sai số vị trí.
            # Tránh việc ép 2 tâm vật lý phải đè lên nhau (gây Physics Explosion bắn vật thể bay đi)
            ee_pos, ee_quat = self.get_ee_pose()
            obj_pos_tup, obj_quat_tup = self.get_object_pose()
            
            # Di chuyển Tọa độ của Vật về Hệ quy chiếu cục bộ (Local Frame) của Đầu Hút (EE)
            inv_ee_pos, inv_ee_quat = p.invertTransform(ee_pos, ee_quat)
            local_obj_pos, local_obj_quat = p.multiplyTransforms(
                inv_ee_pos, inv_ee_quat, 
                obj_pos_tup, obj_quat_tup
            )
            
            self._constraint_id = p.createConstraint(
                parentBodyUniqueId=self._robot_id,
                parentLinkIndex=ee_link,
                childBodyUniqueId=self._object_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=local_obj_pos,
                childFramePosition=[0, 0, 0],
                parentFrameOrientation=local_obj_quat
            )
            return True
        return False

    def release_gripper(self):
        """Remove the constraint if gripping."""
        if self._constraint_id != -1:
            p.removeConstraint(self._constraint_id)
            self._constraint_id = -1

    def is_gripping(self) -> bool:
        return self._constraint_id != -1

    def get_object_height(self) -> float:
        """Height of object base relative to the table surface."""
        obj_z = self.get_object_pose()[0][2]
        return max(0.0, obj_z - TABLE_SURFACE)

    def get_bin_center(self) -> list:
        """Return the (x, y, z) center of the bin drop target."""
        # Target is slightly above bin interior for carry navigation
        return [BIN_CENTER[0], BIN_CENTER[1], BIN_CENTER[2] + 0.08]

    def is_in_bin(self) -> bool:
        """Check if object has been placed squarely inside the bin."""
        obj_pos = self.get_object_pose()[0]
        # Bắt buộc vật thể phải bay sát vào khoảng trống lõi của Thùng (Cách tâm < 5cm) 
        # Chứ không được phép thả ngay khi vừa sượt qua mép viền ngoài của thành thùng.
        in_x = abs(obj_pos[0] - BIN_CENTER[0]) < 0.050
        in_y = abs(obj_pos[1] - BIN_CENTER[1]) < 0.050
        above_floor = obj_pos[2] < (TABLE_SURFACE + 0.13)
        return in_x and in_y and above_floor

    def step(self, n=1):
        for _ in range(n):
            p.stepSimulation()

    def close(self):
        p.disconnect()


if __name__ == "__main__":
    env = UR5eEnvironment(gui=True)
    print("\n[TEST] Scene info:")
    print(f"  Robot ID:     {env.get_robot_id()}")
    print(f"  Joint idx:    {env.get_joint_indices()}")
    print(f"  Object pose:  {env.get_object_pose()}")
    print(f"  EE pose:      {env.get_ee_pose()}")

    print("\n[TEST] Running 8s dynamics loop...")
    for _ in range(1920):
        env.step()
        time.sleep(1 / 240)

    env.close()
    print("[ENV] Done.")
