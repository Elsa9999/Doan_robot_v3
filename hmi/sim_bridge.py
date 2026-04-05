import time
import queue
from queue import Queue
from threading import Thread

from simulation.environment import UR5eEnvironment, HOME_POSE
from simulation.trajectory_executor import TrajectoryExecutor
from simulation.gripper import VacuumGripper
from simulation.object_detector import ObjectDetector
from simulation.pick_place_sm import PickPlaceStateMachine, State

from kinematics.forward_kinematics import forward_kinematics
from kinematics.inverse_kinematics import inverse_kinematics
from kinematics.workspace_validator import WorkspaceValidator
from kinematics.trajectory import JointTrajectory, CartesianTrajectory
from utils.transforms import local_to_world, world_to_local

LOOP_HZ      = 240
PUBLISH_EVERY = 24      # 10 Hz state publish


class SimBridge:
    def __init__(self):
        self.command_queue  = Queue()
        self.state_queue    = Queue()
        self._env           = None
        self._executor      = None
        self._gripper       = None
        self._detector      = None
        self._sm            = None
        self._running       = False
        self._ready         = False
        self._estop         = False
        self._mode          = 'manual'   # 'manual' | 'auto'
        self._q_target      = list(HOME_POSE)
        self._validator     = WorkspaceValidator()
        self._log_queue     = Queue()
        self._step_counter  = 0
        self._traj_progress = 0.0
        self._sm_status     = {}

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._ready

    def start(self, gui=True):
        self._running = True
        self._env     = UR5eEnvironment(gui=gui)

        # Build subsystems
        robot_id  = self._env.get_robot_id()
        ee_link   = self._env.get_joint_indices()[-1]

        self._executor = TrajectoryExecutor(self._env)
        self._gripper  = VacuumGripper(robot_id, ee_link)
        self._detector = ObjectDetector(self._env)
        self._sm       = PickPlaceStateMachine(
            self._env, self._executor, self._gripper, self._detector
        )
        self._ready = True

        while self._running:
            # Luôn xử lý GUI Command trước
            self._process_commands()

            # ── Update Auto SM ──
            if self._mode == 'auto' and not self._estop:
                self._sm_status = self._sm.update()
            elif self._mode != 'auto':
                self._traj_progress = 1.0

            # ── Update Trajectory Executor ──
            if self._executor.is_running and not self._estop:
                status = self._executor.update()
                self._traj_progress = status['progress']
                if status.get('q'):
                    self._q_target = list(status['q'])

            # ── Step physics ──
            if not self._estop:
                self._env.step(1)

            # ── Gripper indicator every 10 steps ──
            if self._step_counter % 10 == 0 and gui:
                self._gripper.draw_indicator()

            # ── Publish state every PUBLISH_EVERY steps ──
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

            # ── Auto mode commands ──
            if t == 'start_auto':
                obj_id = self._env.get_object_id()
                if obj_id < 0:
                    self._push_log("No object in scene", 'WARN')
                    continue
                
                # Tránh tình trạng gắp lại vật đã thả vào bin
                pos, _ = self._env.get_object_pose()
                x, y = pos[0], pos[1]
                if 0.45 <= x <= 0.85 and -0.50 <= y <= -0.10:
                    self._push_log("Object already in bin, resetting scene...", 'INFO')
                    self._env.reset()
                    self._q_target = list(HOME_POSE)
                    obj_id = self._env.get_object_id()

                self._mode = 'auto'
                self._sm.start(obj_id, auto_repeat=cmd.get('auto_repeat', False))
                self._push_log("Auto mode started", 'INFO')

            elif t == 'stop_auto':
                self._sm.stop()
                self._mode = 'manual'
                self._push_log("Auto mode stopped", 'INFO')

            elif t == 'reset_error':
                self._sm.stop()
                self._env.reset()
                self._mode = 'manual'
                self._q_target = list(HOME_POSE)
                self._push_log("Error reset, scene reloaded", 'INFO')

            # ── Manual commands ──
            elif self._mode == 'manual':
                self._process_manual_command(t, cmd)

    def _process_manual_command(self, t, cmd):
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
            if self._executor: self._executor.stop()
            self._env.reset()
            self._q_target = list(HOME_POSE)
            self._estop = False
            self._push_log("Scene reset", 'INFO')

        elif t == 'emergency_stop':
            self._estop = True
            if self._sm: self._sm.stop()
            if self._executor: self._executor.stop()
            self._mode = 'manual'
            self._push_log("EMERGENCY STOP!", 'ESTOP')

        elif t == 'clear_estop':
            self._estop = False
            self._push_log("E-Stop cleared", 'INFO')

    def _handle_set_joints(self, q):
        if q is None: return
        fk_res = forward_kinematics(q)
        w_pos = local_to_world(fk_res['position'])
        ok, reason = self._validator.is_valid_ee(w_pos)
        if not ok:
            self._push_log(f"BLOCKED: {reason}", 'WARN')
            return
        self._q_target = list(q)
        self._env.set_joint_positions(self._q_target)

    def _handle_set_cartesian(self, pos, euler):
        if pos is None or euler is None: return
        pos_safe = self._validator.clamp_to_workspace(pos)
        q_now = self._env.get_joint_positions()
        l_pos, l_eul = world_to_local(pos_safe, euler)
        ik_res = inverse_kinematics(l_pos, l_eul, q_current=q_now)
        best = ik_res.get('best')
        if best is None:
            self._push_log("IK failed", 'WARN')
            return
        self._q_target = list(best)
        self._env.set_joint_positions(self._q_target)

    def _handle_jog_cartesian(self, axis, step):
        if axis is None or step is None: return
        q_now  = self._env.get_joint_positions()
        fk_res = forward_kinematics(q_now)
        pos, euler = local_to_world(fk_res['position'], fk_res['euler'])
        delta_map = {'x+': (0, step), 'x-': (0,-step),
                     'y+': (1, step), 'y-': (1,-step),
                     'z+': (2, step), 'z-': (2,-step)}
        if axis in delta_map:
            i, d = delta_map[axis]
            pos[i] += d
        pos_safe = self._validator.clamp_to_workspace(pos)
        l_pos, l_eul = world_to_local(pos_safe, euler)
        ik_res = inverse_kinematics(l_pos, l_eul, q_current=q_now)
        best = ik_res.get('best')
        if best:
            self._q_target = list(best)
            self._env.set_joint_positions(self._q_target)

    def _handle_joint_traj(self, cmd):
        q_end = cmd.get('q_end')
        if q_end is None: return
        q_now = self._env.get_joint_positions()
        try:
            traj = JointTrajectory.from_two_points(q_now, q_end)
            self._executor.execute(traj, speed_scale=cmd.get('speed_scale', 1.0))
            self._push_log(f"Joint traj {traj.duration:.2f}s", 'IK')
        except Exception as e:
            self._push_log(f"Joint traj error: {e}", 'WARN')

    def _handle_cartesian_traj(self, cmd):
        pos_end = cmd.get('pos_end')
        euler_end = cmd.get('euler_end')
        if pos_end is None: return
        q_now  = self._env.get_joint_positions()
        fk_res = forward_kinematics(q_now)
        pos, euler = local_to_world(fk_res['position'], fk_res['euler'])
        try:
            cart = CartesianTrajectory.from_two_points(
                list(pos), pos_end,
                list(euler), euler_end or list(euler),
                v_max=cmd.get('v_max', 0.1)
            )
            jtraj = cart.to_joint_trajectory(inverse_kinematics, q_now)
            self._executor.execute(jtraj, speed_scale=cmd.get('speed_scale', 1.0))
            self._push_log(f"Cart traj {jtraj.duration:.2f}s", 'IK')
        except ValueError as e:
            self._push_log(f"Cart traj blocked: {str(e)[:50]}", 'WARN')

    # ─── State publishing ──────────────────────────────────────────────────────

    def _publish_state(self):
        if not self._env: return
        q_actual = self._env.get_joint_positions()
        fk_res   = forward_kinematics(q_actual)
        pos, euler = local_to_world(fk_res['position'], fk_res['euler'])
        ok, _    = self._validator.is_valid_ee(pos)

        sm = self._sm_status
        state = {
            'q':             q_actual,
            'q_target':      self._q_target,
            'at_target':     self._is_motion_complete(q_actual),
            'ee_pos':        pos,
            'ee_euler':      euler,
            'estop':         self._estop,
            'workspace_ok':  ok,
            'traj_running':  self._executor.is_running,
            'traj_progress': self._traj_progress,
            'mode':          self._mode,
            'sm_state':      sm.get('state', 'IDLE'),
            'sm_cycle':      sm.get('cycle', 0),
            'sm_error':      sm.get('error_msg', ''),
            'gripper':       self._gripper.is_activated() if self._gripper else False,
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
        q = q_actual or self._env.get_joint_positions()
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
