"""
simulation/trajectory_executor.py — Thực thi trajectory trong simulation loop.
"""
import time


class TrajectoryExecutor:
    def __init__(self, env, bridge=None):
        self._env           = env
        self._bridge        = bridge
        self._traj          = None
        self._t0            = None
        self._running       = False
        self._done_callback = None
        self._speed_scale   = 1.0

    # ─── Public API ───────────────────────────────────────────────────────────

    def execute(self, trajectory, done_callback=None, speed_scale: float = 1.0):
        """Bắt đầu chạy trajectory. Gọi trong simulation thread."""
        self._traj          = trajectory
        self._t0            = time.time()
        self._running       = True
        self._done_callback = done_callback
        self._speed_scale   = max(0.1, min(2.0, speed_scale))

    def stop(self):
        """Dừng ngay, giữ nguyên vị trí hiện tại."""
        self._running = False
        self._traj    = None

    def set_speed(self, speed_scale: float):
        """Thay đổi tốc độ real-time (0.1 → 2.0)."""
        self._speed_scale = max(0.1, min(2.0, speed_scale))

    @property
    def is_running(self) -> bool:
        return self._running

    # ─── Update (gọi mỗi simulation step) ────────────────────────────────────

    def update(self) -> dict:
        """
        Gọi trong simulation loop (240Hz).
        Trả về {'running', 'progress', 't', 'q'}.
        """
        if not self._running or self._traj is None:
            return {'running': False, 'progress': 0.0, 't': 0.0, 'q': None}

        t = (time.time() - self._t0) * self._speed_scale
        point = self._traj.get_point(t)

        # Set joint với max_velocity cao hơn manual vì trajectory đã smooth
        self._env.set_joint_positions(point['q'], max_velocity=3.0)

        progress = min(t / self._traj.duration, 1.0)

        if self._traj.is_done(t):
            self._running = False
            if self._done_callback:
                self._done_callback()

        return {
            'running':  self._running,
            'progress': progress,
            't':        t,
            'q':        point['q']
        }
