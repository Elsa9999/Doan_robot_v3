"""
kinematics/trajectory.py — Trajectory planning engine.
Hỗ trợ: Trapezoid velocity profile, Joint Space, Cartesian Space.
"""
import numpy as np
from scipy.interpolate import CubicSpline

from kinematics.workspace_validator import WorkspaceValidator
from kinematics.forward_kinematics import forward_kinematics


# ─────────────────────────────────────────────────────────────────────────────
# A. Trapezoid velocity profile
# ─────────────────────────────────────────────────────────────────────────────

def trapezoid_profile(
    distance: float,
    v_max: float,
    a_max: float,
    dt: float = 1 / 240
) -> np.ndarray:
    """
    Tính mảng position theo thời gian với vận tốc hình thang.
    Luôn trả về mảng đi từ 0.0 đến |distance|, sau đó scale dấu (chiều đi).
    """
    dist_abs = abs(distance)
    if dist_abs < 1e-9:
        return np.array([0.0])

    # Thời gian tăng tốc/giảm tốc
    t_accel = v_max / a_max

    # Quãng đường trong giai đoạn tăng + giảm tốc
    d_accel = 0.5 * a_max * t_accel ** 2

    if 2 * d_accel >= dist_abs:
        # Profile tam giác (không đủ quãng đường để đạt v_max)
        t_accel = np.sqrt(dist_abs / a_max)
        t_total = 2 * t_accel
    else:
        d_const = dist_abs - 2 * d_accel
        t_const = d_const / v_max
        t_total = 2 * t_accel + t_const

    N = max(2, int(np.ceil(t_total / dt)) + 1)
    t = np.linspace(0, t_total, N)
    pos = np.zeros(N)

    for i, ti in enumerate(t):
        if ti <= t_accel:
            pos[i] = 0.5 * a_max * ti ** 2
        elif ti <= t_total - t_accel:
            d_phase1 = 0.5 * a_max * t_accel ** 2
            v_at_end_accel = a_max * t_accel
            pos[i] = d_phase1 + v_at_end_accel * (ti - t_accel)
        else:
            tr = t_total - ti
            pos[i] = dist_abs - 0.5 * a_max * tr ** 2

    # Clamp cuối về đúng distance
    pos = np.clip(pos, 0, dist_abs)
    pos[-1] = dist_abs

    # Khôi phục dấu
    return pos if distance >= 0 else -pos


# ─────────────────────────────────────────────────────────────────────────────
# B. Joint Space Trajectory
# ─────────────────────────────────────────────────────────────────────────────

class JointTrajectory:
    def __init__(
        self,
        timestamps: np.ndarray,   # (N,)
        positions:  np.ndarray,   # (N, 6)
        velocities: np.ndarray    # (N, 6)
    ):
        self.timestamps = timestamps
        self.positions  = positions
        self.velocities = velocities
        self.duration   = float(timestamps[-1])

    # ── Factory methods ───────────────────────────────────────────────────────

    @staticmethod
    def from_two_points(
        q_start: list,
        q_end:   list,
        duration: float = None,
        v_max_joint: float = 1.0,
        a_max_joint: float = 0.5,
        dt: float = 1 / 240
    ) -> 'JointTrajectory':
        q_start = np.array(q_start, dtype=float)
        q_end   = np.array(q_end,   dtype=float)
        delta   = q_end - q_start

        # Tính duration tự động nếu không cho trước
        if duration is None:
            t_per_joint = []
            for d in delta:
                d_abs = abs(d)
                if d_abs < 1e-9:
                    t_per_joint.append(0.0)
                    continue
                t_a = v_max_joint / a_max_joint
                d_a = 0.5 * a_max_joint * t_a ** 2
                if 2 * d_a >= d_abs:
                    t = 2 * np.sqrt(d_abs / a_max_joint)
                else:
                    t = 2 * t_a + (d_abs - 2 * d_a) / v_max_joint
                t_per_joint.append(t)
            duration = max(t_per_joint) + 0.1 if max(t_per_joint) > 0 else 0.5

        # Nội suy: mỗi joint scale để đến đích cùng lúc
        N = max(2, int(np.ceil(duration / dt)) + 1)
        timestamps = np.linspace(0, duration, N)

        positions  = np.zeros((N, 6))
        velocities = np.zeros((N, 6))

        for j in range(6):
            d = delta[j]
            if abs(d) < 1e-9:
                positions[:, j] = q_start[j]
                continue

            # Scale v_max/a_max cho joint này
            v_j = v_max_joint * abs(d) / (max(abs(delta)) + 1e-12)
            v_j = max(v_j, 1e-4)

            profile = trapezoid_profile(d, v_j, a_max_joint, dt)

            # Resample profile lên/xuống N điểm
            t_prof = np.linspace(0, duration, len(profile))
            positions[:, j] = q_start[j] + np.interp(timestamps, t_prof, profile)

        # Tính velocity bằng gradient
        for j in range(6):
            velocities[:, j] = np.gradient(positions[:, j], timestamps)
        velocities[0,  :] = 0.0
        velocities[-1, :] = 0.0

        return JointTrajectory(timestamps, positions, velocities)

    @staticmethod
    def from_waypoints(
        waypoints: list,
        v_max: float = 1.0,
        a_max: float = 0.5,
        dt: float = 1 / 240
    ) -> 'JointTrajectory':
        """Nội suy CubicSpline qua tất cả waypoints, v=0 đầu và cuối."""
        wps = [np.array(w, dtype=float) for w in waypoints]
        n   = len(wps)

        # Ước tính thời gian mỗi đoạn bằng khoảng cách joint tối đa
        t_list = [0.0]
        for k in range(1, n):
            d_max = np.max(np.abs(wps[k] - wps[k - 1]))
            t_seg = max(d_max / v_max + 0.1, 0.3)
            t_list.append(t_list[-1] + t_seg)

        t_knots = np.array(t_list)
        pos_knots = np.vstack(wps)   # (n, 6)

        N = max(2, int(np.ceil(t_knots[-1] / dt)) + 1)
        timestamps = np.linspace(0, t_knots[-1], N)
        positions  = np.zeros((N, 6))
        velocities = np.zeros((N, 6))

        for j in range(6):
            # Điều kiện biên: v=0 đầu và cuối (not-a-knot fallback)
            bc = ((1, 0.0), (1, 0.0))   # clamped spline
            cs = CubicSpline(t_knots, pos_knots[:, j], bc_type=bc)
            positions[:, j]  = cs(timestamps)
            velocities[:, j] = cs(timestamps, 1)

        return JointTrajectory(timestamps, positions, velocities)

    # ── Playback ──────────────────────────────────────────────────────────────

    def get_point(self, t: float) -> dict:
        t = float(np.clip(t, 0, self.duration))
        q  = np.array([np.interp(t, self.timestamps, self.positions[:, j])
                        for j in range(6)])
        dq = np.array([np.interp(t, self.timestamps, self.velocities[:, j])
                        for j in range(6)])
        return {'q': q.tolist(), 'dq': dq.tolist()}

    def is_done(self, t: float) -> bool:
        return t >= self.duration


# ─────────────────────────────────────────────────────────────────────────────
# C. Cartesian Space Trajectory
# ─────────────────────────────────────────────────────────────────────────────

class CartesianTrajectory:
    def __init__(self, timestamps, positions, eulers):
        self.timestamps = timestamps   # (N,)
        self.positions  = positions    # (N, 3)
        self.eulers     = eulers       # (N, 3)
        self.duration   = float(timestamps[-1])

    @staticmethod
    def from_two_points(
        pos_start:   list,
        pos_end:     list,
        euler_start: list,
        euler_end:   list,
        v_max: float = 0.1,
        a_max: float = 0.2,
        dt: float = 1 / 240
    ) -> 'CartesianTrajectory':
        pos_s = np.array(pos_start,   dtype=float)
        pos_e = np.array(pos_end,     dtype=float)
        eul_s = np.array(euler_start, dtype=float)
        eul_e = np.array(euler_end,   dtype=float)

        distance = float(np.linalg.norm(pos_e - pos_s))
        if distance < 1e-6:
            # Không di chuyển, chỉ giữ nguyên
            ts = np.linspace(0, 0.1, 5)
            pos_arr = np.tile(pos_s, (5, 1))
            eul_arr = np.tile(eul_s, (5, 1))
            return CartesianTrajectory(ts, pos_arr, eul_arr)

        # Trapezoid profile theo khoảng cách
        prof = trapezoid_profile(distance, v_max, a_max, dt)
        N    = len(prof)
        t_total = N * dt
        timestamps = np.linspace(0, t_total, N)

        direction = (pos_e - pos_s) / distance
        positions = np.outer(prof, direction) + pos_s      # (N, 3)
        alphas    = prof / distance                         # 0→1
        eulers    = np.outer(alphas, eul_e - eul_s) + eul_s  # linear interp

        # Validate workspace
        validator = WorkspaceValidator()
        for i in range(N):
            ok, reason = validator.is_valid_ee(positions[i])
            if not ok:
                raise ValueError(
                    f"Trajectory passes through invalid workspace at step {i}/{N}: {reason}. pos_s={pos_s}, pos_e={pos_e}, at={positions[i]}"
                )

        return CartesianTrajectory(timestamps, positions, eulers)

    def get_pose(self, t: float) -> dict:
        t = float(np.clip(t, 0, self.duration))
        pos   = np.array([np.interp(t, self.timestamps, self.positions[:, j])
                          for j in range(3)])
        euler = np.array([np.interp(t, self.timestamps, self.eulers[:, j])
                          for j in range(3)])
        return {'pos': pos.tolist(), 'euler': euler.tolist()}

    def is_done(self, t: float) -> bool:
        return t >= self.duration

    def to_joint_trajectory(
        self,
        ik_func,
        q_start: list,
        dt: float = 1 / 240
    ) -> JointTrajectory:
        from kinematics.inverse_kinematics import inverse_kinematics
        from utils.transforms import world_to_local

        N = len(self.timestamps)
        positions  = np.zeros((N, 6))
        velocities = np.zeros((N, 6))

        q_prev = list(q_start)

        for i, t in enumerate(self.timestamps):
            pose   = self.get_pose(float(t))
            
            # Trajectory runs in WORLD coords, IK needs LOCAL coords
            local_pos, local_eul = world_to_local(pose['pos'], pose['euler'])
            
            result = ik_func(local_pos, local_eul, q_current=q_prev)
            best   = result.get('best')
            if best is None:
                raise ValueError(
                    f"IK failed at t={t:.3f}s, pos={pose['pos']}"
                )
            positions[i] = best
            q_prev = best

        for j in range(6):
            velocities[:, j] = np.gradient(positions[:, j], self.timestamps)
        velocities[0,  :] = 0.0
        velocities[-1, :] = 0.0

        return JointTrajectory(self.timestamps, positions, velocities)
