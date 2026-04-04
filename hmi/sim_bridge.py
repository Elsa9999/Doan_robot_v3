import time
import queue
from queue import Queue
from threading import Thread

from simulation.environment import UR5eEnvironment, HOME_POSE
from simulation.trajectory_executor import TrajectoryExecutor
from kinematics.forward_kinematics import forward_kinematics
from kinematics.inverse_kinematics import inverse_kinematics
from kinematics.workspace_validator import WorkspaceValidator
from kinematics.trajectory import JointTrajectory, CartesianTrajectory

LOOP_HZ      = 240
PUBLISH_EVERY = 24      # Publish state every N steps = 10 Hz


class SimBridge:
    def __init__(self):
        self.command_queue  = Queue()
        self.state_queue    = Queue()
        self._env           = None
        self._executor      = None
        self._running       = False
        self._ready         = False
        self._estop         = False
        self._q_target      = list(HOME_POSE)
        self._validator     = WorkspaceValidator()
        self._log_queue     = Queue()
        self._step_counter  = 0
        self._traj_progress = 0.0

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._ready

    def start(self, gui=True):
        self._running  = True
        self._env      = UR5eEnvironment(gui=gui)
        self._executor = TrajectoryExecutor(self._env)
        self._ready    = True

        while self._running:
            # Trajectory execution takes priority over manual commands
            if self._executor.is_running:
                status = self._executor.update()
                self._traj_progress = status['progress']
                if status.get('q'):
                    self._q_target = list(status['q'])
            else:
                self._traj_progress = 1.0
                self._process_commands()

            if not self._estop:
                self._env.step(1)

            self._step_counter += 1
            if self._step_counter >= PUBLISH_EVERY:
                self._publish_state()
                self._step_counter = 0

            time.sleep(1.0 / LOOP_HZ)

        self._env.close()

    def stop(self):
        self._running = False

    # ─── Command processing ───────────────────────────────────────────────────

    def _process_commands(self):
        while not self.command_queue.empty():
            try:
                cmd = self.command_queue.get_nowait()
            except queue.Empty:
                break

            t = cmd.get('type')

            if t == 'set_joints' and not self._estop:
                self._handle_set_joints(cmd.get('q'))

            elif t == 'set_cartesian' and not self._estop:
                self._handle_set_cartesian(cmd.get('pos'), cmd.get('euler'))

            elif t == 'jog_cartesian' and not self._estop:
                self._handle_jog_cartesian(cmd.get('axis'), cmd.get('step'))

            elif t == 'run_joint_traj' and not self._estop:
                self._handle_joint_traj(cmd)

            elif t == 'run_cartesian_traj' and not self._estop:
                self._handle_cartesian_traj(cmd)

            elif t == 'stop_traj':
                self._executor.stop()
                self._push_log("Trajectory stopped", 'WARN')

            elif t == 'set_speed':
                if self._executor:
                    self._executor.set_speed(cmd.get('speed_scale', 1.0))

            elif t == 'go_home' and not self._estop:
                self._q_target = list(HOME_POSE)
                self._env.set_joint_positions(self._q_target)
                self._push_log("Go home", 'INFO')

            elif t == 'reset':
                if self._executor:
                    self._executor.stop()
                self._env.reset()
                self._q_target = list(HOME_POSE)
                self._estop = False
                self._push_log("Scene reset", 'INFO')

            elif t == 'emergency_stop':
                self._estop = True
                if self._executor:
                    self._executor.stop()
                self._push_log("EMERGENCY STOP ACTIVATED!", 'ESTOP')
                print("[BRIDGE] EMERGENCY STOP ACTIVATED!")

            elif t == 'clear_estop':
                self._estop = False
                self._push_log("E-Stop cleared", 'INFO')
                print("[BRIDGE] EMERGENCY STOP CLEARED!")

    def _handle_set_joints(self, q):
        if q is None: return
        fk_res = forward_kinematics(q)
        ok, reason = self._validator.is_valid_ee(fk_res['position'])
        if not ok:
            self._push_log(f"BLOCKED set_joints: {reason}", 'WARN')
            return
        self._q_target = list(q)
        self._env.set_joint_positions(self._q_target)

    def _handle_set_cartesian(self, pos, euler):
        if pos is None or euler is None: return
        pos_safe = self._validator.clamp_to_workspace(pos)
        if pos_safe != list(pos):
            self._push_log(f"Pos clamped to workspace", 'WARN')
        q_now = self._env.get_joint_positions()
        ik_res = inverse_kinematics(pos_safe, euler, q_current=q_now)
        best = ik_res.get('best')
        if best is None:
            self._push_log("IK failed — no solution", 'WARN')
            return
        self._q_target = list(best)
        self._env.set_joint_positions(self._q_target)

    def _handle_jog_cartesian(self, axis, step):
        if axis is None or step is None: return
        q_now  = self._env.get_joint_positions()
        fk_res = forward_kinematics(q_now)
        pos    = list(fk_res['position'])
        euler  = list(fk_res['euler'])
        delta_map = {
            'x+': (0,  step), 'x-': (0, -step),
            'y+': (1,  step), 'y-': (1, -step),
            'z+': (2,  step), 'z-': (2, -step),
        }
        if axis in delta_map:
            i, d = delta_map[axis]
            pos[i] += d
        pos_safe = self._validator.clamp_to_workspace(pos)
        if pos_safe != pos:
            self._push_log(f"Jog clamped on {axis}", 'WARN')
        ik_res = inverse_kinematics(pos_safe, euler, q_current=q_now)
        best = ik_res.get('best')
        if best is None:
            self._push_log(f"Jog {axis} IK failed", 'WARN')
            return
        self._q_target = list(best)
        self._env.set_joint_positions(self._q_target)

    def _handle_joint_traj(self, cmd):
        q_end       = cmd.get('q_end')
        duration    = cmd.get('duration', None)
        speed_scale = cmd.get('speed_scale', 1.0)
        if q_end is None: return

        q_now = self._env.get_joint_positions()
        try:
            traj = JointTrajectory.from_two_points(
                q_now, q_end, duration=duration,
                v_max_joint=1.0, a_max_joint=0.5
            )
            self._executor.execute(traj, speed_scale=speed_scale)
            self._push_log(f"Joint traj started ({traj.duration:.2f}s)", 'IK')
        except Exception as e:
            self._push_log(f"Joint traj error: {e}", 'WARN')

    def _handle_cartesian_traj(self, cmd):
        pos_end     = cmd.get('pos_end')
        euler_end   = cmd.get('euler_end')
        v_max       = cmd.get('v_max', 0.1)
        speed_scale = cmd.get('speed_scale', 1.0)
        if pos_end is None: return

        q_now   = self._env.get_joint_positions()
        fk_res  = forward_kinematics(q_now)
        pos_now = list(fk_res['position'])
        eul_now = list(fk_res['euler'])

        if euler_end is None:
            euler_end = eul_now

        try:
            cart_traj = CartesianTrajectory.from_two_points(
                pos_now, pos_end, eul_now, euler_end, v_max=v_max, a_max=0.2
            )
            joint_traj = cart_traj.to_joint_trajectory(inverse_kinematics, q_now)
            self._executor.execute(joint_traj, speed_scale=speed_scale)
            self._push_log(f"Cart traj started ({joint_traj.duration:.2f}s)", 'IK')
        except ValueError as e:
            self._push_log(f"Cart traj blocked: {str(e)[:60]}", 'WARN')
        except Exception as e:
            self._push_log(f"Cart traj error: {e}", 'WARN')

    # ─── State publishing ──────────────────────────────────────────────────────

    def _publish_state(self):
        if not self._env: return
        q_actual = self._env.get_joint_positions()
        fk_res   = forward_kinematics(q_actual)
        pos      = list(fk_res['position'])
        euler    = list(fk_res['euler'])
        ok, _    = self._validator.is_valid_ee(pos)

        state = {
            'q':             q_actual,
            'q_target':      self._q_target,
            'at_target':     self._is_motion_complete(q_actual),
            'ee_pos':        pos,
            'ee_euler':      euler,
            'estop':         self._estop,
            'workspace_ok':  ok,
            'traj_running':  self._executor.is_running if self._executor else False,
            'traj_progress': self._traj_progress,
            'timestamp':     time.time(),
        }

        logs = []
        while not self._log_queue.empty():
            try: logs.append(self._log_queue.get_nowait())
            except queue.Empty: break
        if logs:
            state['logs'] = logs

        while self.state_queue.qsize() > 5:
            try: self.state_queue.get_nowait()
            except queue.Empty: break

        self.state_queue.put(state)

    def _is_motion_complete(self, q_actual=None, tol=0.02) -> bool:
        if self._q_target is None: return True
        q = q_actual if q_actual else self._env.get_joint_positions()
        return max(abs(a - t) for a, t in zip(q, self._q_target)) < tol

    def _push_log(self, msg: str, level: str = 'INFO'):
        self._log_queue.put({'msg': msg, 'level': level})

    # ─── External API ──────────────────────────────────────────────────────────

    def send_command(self, cmd: dict):
        self.command_queue.put(cmd)

    def get_state(self) -> dict:
        latest = None
        while not self.state_queue.empty():
            try: latest = self.state_queue.get_nowait()
            except queue.Empty: break
        return latest


if __name__ == "__main__":
    print("-" * 50)
    bridge = SimBridge()
    t = Thread(target=bridge.start, kwargs={'gui': False}, daemon=True)
    t.start()

    timeout = time.time() + 5
    while not bridge.is_ready() and time.time() < timeout:
        time.sleep(0.1)

    print("[TEST] Bridge ready.")
    time.sleep(0.5)

    state = bridge.get_state()
    print(f"[TEST] State: ee_pos={[round(v, 3) for v in state['ee_pos']]}")

    bridge.send_command({'type': 'run_joint_traj',
                         'q_end': [0.5, -1.2, 1.0, -1.5, -1.5, 0.3]})
    time.sleep(4)
    state = bridge.get_state()
    print(f"[TEST] After traj: traj_running={state['traj_running']}  progress={state['traj_progress']:.2f}")

    bridge.stop()
    print("[TEST] Done.")
