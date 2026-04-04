"""
Environment wrapper for PyBullet — dynamics-based control.
Không dùng resetJointState ở runtime, dùng POSITION_CONTROL để có collision.
"""
import pybullet as p
import pybullet_data
import time
import random

TABLE_HEIGHT  = 0.40
TABLE_THICK   = 0.02
TABLE_SURFACE = 0.42

ROBOT_BASE    = [0.0, 0.0, TABLE_SURFACE]
HOME_POSE     = [0, -1.5708, 1.5708, -1.5708, -1.5708, 0]

WORK_ZONE = {'x': (0.3, 0.7), 'y': (-0.15, 0.15)}

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
    def __init__(self, gui=True):
        if gui:
            self._physics_client = p.connect(p.GUI)
        else:
            self._physics_client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)
        p.setRealTimeSimulation(0)
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
        self._robot_id = p.loadURDF(
            "urdf/ur5e_final.urdf",
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
        BIN_CENTER = [0.65, -0.28, TABLE_SURFACE]
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
        create_wall([0.004, 0.071, 0.056], [-0.096, 0, 0.060])
        create_wall([0.004, 0.071, 0.056], [ 0.096, 0, 0.060])
        create_wall([0.096, 0.004, 0.056], [0,  0.071, 0.060])
        create_wall([0.096, 0.004, 0.056], [0, -0.071, 0.060])

    def _spawn_object(self, pos=None):
        print("[ENV] Spawning object...")
        if pos is None:
            x = random.uniform(WORK_ZONE['x'][0] + 0.05, WORK_ZONE['x'][1] - 0.05)
            y = random.uniform(WORK_ZONE['y'][0] + 0.05, WORK_ZONE['y'][1] - 0.05)
            pos = [x, y, TABLE_SURFACE + 0.033]

        shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.02, height=0.06)
        vis   = p.createVisualShape(p.GEOM_CYLINDER,   radius=0.02, length=0.06,
                                    rgbaColor=[0.1, 0.3, 0.9, 1.0])
        self._object_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=shape,
            baseVisualShapeIndex=vis,
            basePosition=pos
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

    # ─── Public API ───────────────────────────────────────────────────────────

    def reset(self):
        if self._object_id != -1:
            p.removeBody(self._object_id)
            self._object_id = -1
        self.teleport_joints(HOME_POSE)
        self.step(10)
        self._spawn_object()
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
