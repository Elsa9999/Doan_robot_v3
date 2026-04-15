"""
Tệp sim_bridge.py
Đóng vai trò là "Cầu mương" (Bridge) hoặc "Tủy sống" truyền tín hiệu thần kinh.
Nhiệm vụ:
- Chạy một luồng vòng lặp riêng biệt (QThread) ở 240Hz độc lập với giao diện PyQt (UI) để không bị lag màn hình.
- Lắng nghe yêu cầu từ người bấm nút (Manual, Auto, AI).
- Triệu hồi (import) PyBullet Environment, giải phương trình động học (IK) rồi đẩy tín hiệu xuống cho Robot chạy thật.
"""
import time
import queue
from queue import Queue
from threading import Thread
from PyQt5.QtCore import QThread, pyqtSignal

# Fix "WinError 1114" on Windows when loading PyTorch in a worker thread
import torch
from stable_baselines3 import SAC

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


class SimBridge(QThread):
    """
    Luồng chạy ngầm điều khiển toàn bộ logic phần cứng mô phỏng.
    Các Signal (pyqtSignal) bên dưới là cách nó "hét" lên cho giao diện (Main Window) biết để cập nhật số má trên màn hình (vd: Tọa độ X=..., Tốc độ...).
    """
    state_updated = pyqtSignal(dict)   # Đẩy thông tin: Joint angles, Cartesian pos
    log_msg       = pyqtSignal(str, str) # Đẩy dòng text Log: Nội dung, Loại log (INFO/ERROR)
    
    # [!] Quản lý State Machine Cổ Điển
    sm_state_updated = pyqtSignal(str) 
    sm_error         = pyqtSignal(str)
    sm_finished      = pyqtSignal()
    
    # [!] Thống kê điểm số AI
    ai_stats_updated = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
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
        self._ai_model      = None
        self._ai_success    = 0

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
        
        # Load AI Model (Hỗ trợ cả 13D cũ và 17D mới)
        import os
        model_17d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models_rl_17d", "seed42", "best_model.zip")
        model_13d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models_rl_place", "seed42", "best_model.zip")
        
        try:
            if os.path.exists(model_17d):
                self._ai_model = SAC.load(model_17d, device="cpu", custom_objects={"learning_rate": 0.0})
                self._push_log("Loaded SAC 17D Model (Rotation)", "INFO")
                self._ai_mode = "17D"
            elif os.path.exists(model_13d):
                self._ai_model = SAC.load(model_13d, device="cpu", custom_objects={"learning_rate": 0.0})
                self._push_log("Loaded SAC 13D Model (Place)", "INFO")
                self._ai_mode = "13D"
        except Exception as e:
            self._push_log(f"Fail to load SAC: {e}", "WARN")

        self._ready = True

        while self._running:
            # Luôn xử lý GUI Command trước
            self._process_commands()

            # ── Update Auto SM ──
            if self._mode == 'auto' and not self._estop:
                self._sm_status = self._sm.update()
            elif self._mode == 'ai' and not self._estop and self._ai_model:
                self._update_ai()
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

            # ── AI mode commands ──
            elif t == 'start_ai':
                if self._ai_model:
                    self._mode = 'ai'
                    self._push_log("AI Mode Started", "INFO")
                else:
                    self._push_log("Cannot start AI - Model missing", "WARN")

            elif t == 'stop_ai':
                if self._mode == 'ai':
                    self._mode = 'manual'
                    self._push_log("AI Mode Stopped", "INFO")

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

    def _update_ai(self):
        import numpy as np
        from simulation.environment import CART_DELTA_MAX
        try:
            from train_17d import EULER_DELTA_MAX
        except ImportError:
            EULER_DELTA_MAX = 0.08
            
        # NẾU đang trong giai đoạn thu tay về sau khi thả thành công HOẶC tự gỡ kẹt
        state_retract = getattr(self, '_ai_returning', 0)
        if state_retract > 0:
            if state_retract == 1 or state_retract == 3:
                # Phase 1: Kéo thẳng tay lên trời (Z) để thoát khỏi thành thùng rác
                ee_cur = self._env.get_ee_position()
                if ee_cur[2] > 0.25:
                    self._ai_returning = 2 if state_retract == 1 else 4 # Chuyển sang Phase 2
                else:
                    self._env.move_ee_cartesian([0.0, 0.0, 0.02]) # Kéo lên
                return
                
            elif state_retract == 2 or state_retract == 4:
                # Phase 2: Tính nội suy đưa tay về HOME_POSE qua Joint Space
                q_cur = np.array(self._env.get_joint_positions())
                q_home = np.array(self._q_target) # Đã được set là HOME_POSE
                
                error = np.linalg.norm(q_cur - q_home)
                if error < 0.05:
                    # Đã về đến nhà an toàn
                    self._ai_returning = 0
                    if state_retract == 2:
                        # Luồng Thành Công -> Bỏ rác xong -> Reset sinh cục rác mới
                        self._env.reset()
                    else:
                        # Luồng Khôi Phục Kẹt (Jam) -> KHÔNG xóa rác -> Quét camera bay đi gắp lại!
                        pass
                    return

                # Di chuyển mượt mà các khớp dần về HOME (Tốc độ x2 cho mượt)
                step_q = q_cur + (q_home - q_cur) * 0.05 
                self._env.set_joint_positions(step_q.tolist())
                return

        # NẾU đang trong quá trình chờ đợi vật rơi xuống đáy
        if hasattr(self, '_ai_wait_frames') and self._ai_wait_frames > 0:
            self._ai_wait_frames -= 1
            if self._ai_wait_frames == 0:
                # Bắt đầu thu tay về (Bắt đầu từ Phase 1: kéo lên)
                self._ai_returning = 1
                self._q_target = list(HOME_POSE)
            return

        # Build obs (13D hoặc 17D tùy model)
        ee_pos = np.array(self._env.get_ee_position(), dtype=np.float32)
        
        # [FAIL-SAFE TỰ ĐỘNG GỠ RỐI] Cảm biến kẹt (Jam Detector)
        if not hasattr(self, '_ai_last_ee'):
            self._ai_last_ee = ee_pos
            self._ai_stuck_frames = 0
            
        dist_moved = np.linalg.norm(ee_pos - self._ai_last_ee)
        if dist_moved < 0.002: # Khớp bị khóa, nhúc nhích chưa tới 2mm
            self._ai_stuck_frames += 1
        else:
            self._ai_stuck_frames = 0
            self._ai_last_ee = ee_pos
            
        if self._ai_stuck_frames > 60: # Nếu khựng đơ liên tục 2 Giây
            self._push_log("Cảnh báo: Robot bị kẹt/vặn sai khớp. Kích hoạt Auto-Home!", 'WARN')
            self._env.release_gripper()
            self._ai_returning = 3 # Trạng thái 3: Gỡ kẹt (Không Xóa Rác)
            self._q_target = list(HOME_POSE)
            self._ai_stuck_frames = 0
            return
            
        obj_pose = self._env.get_object_pose()
        obj_pos = np.array(obj_pose[0], dtype=np.float32)
        bin_pos = np.array(self._env.get_bin_center(), dtype=np.float32)
        
        # [Visual Cadence HACK] Dừng nhịp (Dwell) nửa giây khi vừa chạm để hút
        is_gripping = self._env.is_gripping()
        was_gripping = getattr(self, '_ai_was_gripping', False)
        
        if is_gripping and not was_gripping:
            self._ai_pause_frames = 15 
            self._push_log("Chạm mục tiêu! Dừng 0.5s nén áp suất chân không...", "INFO")
        self._ai_was_gripping = is_gripping
        
        if getattr(self, '_ai_pause_frames', 0) > 0:
            self._ai_pause_frames -= 1
            return
        
        # ── AI điều hướng + kiểm soát gripper ──────────────────────────────
        rel_obj = obj_pos - ee_pos
        rel_bin = bin_pos - obj_pos
        grip = np.array([1.0 if is_gripping else 0.0], dtype=np.float32)

        if getattr(self, '_ai_mode', '13D') == '17D':
            obj_quat = np.array(obj_pose[1], dtype=np.float32)
            obs = np.concatenate([ee_pos, obj_pos, rel_obj, rel_bin, obj_quat, grip])
        else: # 13D
            obs = np.concatenate([ee_pos, obj_pos, rel_obj, rel_bin, grip])

        action, _ = self._ai_model.predict(obs, deterministic=True)
        action = np.clip(action, -1.0, 1.0)

        # Damping 0.5 — mượt giữa UI 30Hz và Training 240Hz
        delta_xyz = action[:3] * CART_DELTA_MAX * 0.5

        if len(action) == 7:
            delta_euler = action[3:6] * EULER_DELTA_MAX * 0.5
            self._env.move_ee_cartesian(delta_xyz, delta_euler)
            grip_cmd = action[6]
        else:
            self._env.move_ee_cartesian(delta_xyz)
            grip_cmd = action[3]

        # AI (Phase-Based) tự điều khiển gripper đúng theo phase
        if grip_cmd > 0:
            self._env.activate_gripper()
        else:
            self._env.release_gripper()

        # Check success
        wait_frames = getattr(self, '_ai_wait_frames', 0)
        if self._env.is_in_bin() and wait_frames <= 0:
            self._ai_success += 1
            self._push_log(f"AI thả thành công (Đợi nửa giây)! Tổng: {self._ai_success}", "INFO")
            
            # [HOTFIX] Bắt buộc nhả giác hút ra để vật thể rơi thụp xuống đáy thùng nhìn cho logic
            self._env.release_gripper()
            
            # Liền lập tức chuyển form sang chế độ thu tay mượt mà (Delay 30 khung hình cho vật rơi)
            self._ai_wait_frames = 30 

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
            'ai_loaded':     self._ai_model is not None,
            'ai_success_count': self._ai_success,
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
