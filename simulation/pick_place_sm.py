"""
Tệp pick_place_sm.py
Cỗ máy trạng thái hữu hạn (Finite State Machine - FSM) cho hệ thống Auto Cổ điển.
Nguyên lý:
Trái ngược với AI tự mò đường, chế độ Auto cổ điển hoạt động dựa trên các bước được LẬP TRÌNH CỨNG theo thứ tự tuần tự:
IDLE -> DETECT -> APPROACH -> DESCEND -> PICK -> LIFT -> MOVE_TO_BIN -> PLACE -> RELEASE -> RETREAT -> DONE.
Nếu ở bất kỳ bước nào mà Robot đi lệch quỹ đạo hoặc dãn khớp, FSM sẽ chuyển sang trạng thái ERROR và khoá hệ thống.
"""
import time
from enum import Enum, auto

from kinematics.trajectory import JointTrajectory, CartesianTrajectory
from kinematics.inverse_kinematics import inverse_kinematics
from kinematics.forward_kinematics import forward_kinematics
from kinematics.workspace_validator import WorkspaceValidator
from simulation.environment import HOME_POSE

BIN_CENTER    = [0.65, -0.28, 0.42]   # bin center (trên mặt bàn)
EE_EULER_DOWN = [3.14159, 0.0, 0.0]   # EE hướng thẳng xuống
STATE_TIMEOUT = 15.0                   # timeout mỗi state (giây)


class State(Enum):
    IDLE        = auto()
    DETECT      = auto()
    APPROACH    = auto()
    DESCEND     = auto()
    PICK        = auto()
    LIFT        = auto()
    MOVE_TO_BIN = auto()
    PLACE       = auto()
    RELEASE     = auto()
    RETREAT     = auto()
    DONE        = auto()
    ERROR       = auto()


# Thứ tự các state để hiển thị progress trong HMI
STATE_ORDER = [
    State.IDLE, State.DETECT, State.APPROACH, State.DESCEND,
    State.PICK, State.LIFT, State.MOVE_TO_BIN, State.PLACE,
    State.RELEASE, State.RETREAT, State.DONE
]


class PickPlaceStateMachine:
    """
    Bộ điều khiển Cỗ máy trạng thái (FSM Controller).
    Nó là thủ kho chứa các Tọa độ mục tiêu tính sẵn (pick_poses, place_poses),
    và theo dõi xem Robot đã chạy xong quỹ đạo Trajectory của bước hiện tại hay chưa để chuyển sang bước sau.
    """
    def __init__(self, env, executor, gripper, detector):
        self._env       = env         # Trỏ tới Thế giới Vật lý
        self._executor  = executor    # Trỏ tới Bộ kích hoạt Quỹ đạo (Người bóp lẫy)
        self._gripper   = gripper     # Trỏ tới Nam châm hút
        self._detector  = detector    # Trỏ tới Camera ảo (Quét vật thể)
        self._validator = WorkspaceValidator() # Cảnh sát vùng cấm (Ngăn robot đập tay vào tường)

        self._state       = State.IDLE
        self._object_id   = None
        self._pick_poses  = None
        self._place_poses = None
        self._cycle_count = 0
        self._error_msg   = ""
        self._auto_repeat = False

        self._state_start_time = None   # timeout tracking
        self._wait_start       = None   # dwell wait (PICK/RELEASE)

    # ─── Control ──────────────────────────────────────────────────────────────

    def start(self, object_id: int, auto_repeat: bool = False):
        self._object_id  = object_id
        self._auto_repeat = auto_repeat
        self._state      = State.DETECT
        self._error_msg  = ""
        self._state_start_time = time.time()
        print(f"[SM] Started Pick & Place (repeat={auto_repeat})")

    def stop(self):
        self._executor.stop()
        self._gripper.release()
        self._state = State.IDLE
        self._error_msg = ""
        print("[SM] Stopped")

    def reset(self):
        self.stop()

    @property
    def state(self) -> State:
        return self._state

    # ─── Main update (240 Hz) ─────────────────────────────────────────────────

    def update(self) -> dict:
        if self._state == State.IDLE:
            pass

        elif self._state == State.DETECT:
            self._do_detect()

        elif self._state in (State.APPROACH, State.DESCEND,
                             State.LIFT, State.MOVE_TO_BIN,
                             State.PLACE, State.RETREAT):
            self._do_traj_state()

        elif self._state == State.PICK:
            self._do_pick()

        elif self._state == State.RELEASE:
            self._do_release()

        elif self._state == State.DONE:
            self._do_done()

        elif self._state == State.ERROR:
            pass   # hold until user resets

        return self._get_status()

    # ─── State handlers ───────────────────────────────────────────────────────

    def _do_detect(self):
        object_id = self._object_id
        if object_id is None or object_id < 0:
            self._set_error("No object to detect")
            return

        pose       = self._detector.get_object_pose(object_id)
        obj_pos    = pose['pos']
        self._pick_poses  = self._detector.compute_pick_poses(obj_pos)
        
        # Override place_poses để đảm bảo rớt tự do thay vì đâm quá sâu (tránh va chạm vật lý = gấp tay)
        self._place_poses = {
            'above_bin': [BIN_CENTER[0], BIN_CENTER[1], 0.75],
            'place':     [BIN_CENTER[0], BIN_CENTER[1], 0.68] 
        }

        # Validate approach pose
        approach = self._pick_poses['approach']
        ok, reason = self._validator.is_valid_ee(approach)
        if not ok:
            self._set_error(f"Approach blocked: {reason}")
            return

        print(f"[SM] DETECT → object at {[f'{v:.3f}' for v in obj_pos]}")
        self._transition(State.APPROACH)
        self._start_traj_cartesian(self._pick_poses['approach'], v_max=0.12)

    def _do_traj_state(self):
        # Check timeout
        if self._state_start_time and \
                time.time() - self._state_start_time > STATE_TIMEOUT:
            self._set_error(f"Timeout in state {self._state.name}")
            return

        # Chờ executor xong
        if self._executor.is_running:
            return

        # Transition khi xong
        if self._state == State.APPROACH:
            print("[SM] APPROACH done → DESCEND")
            self._transition(State.DESCEND)
            self._start_traj_cartesian(self._pick_poses['pick'], v_max=0.05)

        elif self._state == State.DESCEND:
            print("[SM] DESCEND done → PICK")
            self._transition(State.PICK)
            self._wait_start = time.time()

        elif self._state == State.LIFT:
            print("[SM] LIFT done → MOVE_TO_BIN")
            self._transition(State.MOVE_TO_BIN)
            self._start_traj_cartesian(self._place_poses['above_bin'], v_max=0.15)

        elif self._state == State.MOVE_TO_BIN:
            print("[SM] MOVE_TO_BIN done → PLACE")
            self._transition(State.PLACE)
            self._start_traj_cartesian(self._place_poses['place'], v_max=0.05)

        elif self._state == State.PLACE:
            print("[SM] PLACE done → RELEASE")
            self._transition(State.RELEASE)
            self._wait_start = time.time()

        elif self._state == State.RETREAT:
            print("[SM] RETREAT done → DONE")
            self._transition(State.DONE)

    def _do_pick(self):
        # Dwell 0.3 giây để ổn định
        if self._wait_start and time.time() - self._wait_start < 0.3:
            return

        # Activate gripper
        self._gripper.activate(self._object_id)

        # Raycast verify (optional — just log nếu fail)
        q_now  = self._env.get_joint_positions()
        fk_res = forward_kinematics(q_now)
        ee_pos = list(fk_res['position'])
        ray    = self._detector.raycast_detect(ee_pos, max_dist=0.3)
        if ray is None:
            print("[SM] PICK — raycast miss (EE might be far, continuing)")
        else:
            print(f"[SM] PICK — raycast hit obj={ray['object_id']} d={ray['distance']:.3f}")

        print("[SM] PICK done → LIFT")
        self._transition(State.LIFT)
        self._start_traj_cartesian(self._pick_poses['lift'], v_max=0.08)

    def _do_release(self):
        # Dwell 0.5 giây
        if self._wait_start and time.time() - self._wait_start < 0.5:
            return

        self._gripper.release()
        print("[SM] RELEASE done → RETREAT")
        self._transition(State.RETREAT)
        self._start_traj_joint(HOME_POSE)

    def _do_done(self):
        self._cycle_count += 1
        print(f"[SM] Cycle {self._cycle_count} complete!")

        if self._auto_repeat:
            # Spawn object mới và lặp lại
            try:
                self._env.reset()
                self._object_id = self._env.get_object_id()
            except Exception as e:
                self._set_error(f"Reset failed: {e}")
                return
            self._transition(State.DETECT)
        else:
            self._state = State.IDLE

    # ─── Trajectory helpers ───────────────────────────────────────────────────

    def _start_traj_cartesian(self, target_pos, euler=None, v_max=0.08):
        if euler is None:
            euler = EE_EULER_DOWN

        import pybullet as p
        q_now  = self._env.get_joint_positions()

        # Lấy EE pose thực tế từ PyBullet world frame
        robot_id = self._env.get_robot_id()
        ee_link  = self._env.get_joint_indices()[-1]
        link_state = p.getLinkState(robot_id, ee_link,
                                    computeForwardKinematics=True)
        pos_now = list(link_state[4])   # worldLinkFramePosition
        orn_now = link_state[5]
        eul_now = list(p.getEulerFromQuaternion(orn_now))

        try:
            cart_traj = CartesianTrajectory.from_two_points(
                pos_now, target_pos, eul_now, euler, v_max=v_max, a_max=0.2
            )
            joint_traj = cart_traj.to_joint_trajectory(inverse_kinematics, q_now)
            self._executor.execute(joint_traj, speed_scale=1.0)
        except Exception as e:
            self._set_error(f"Cart traj failed: {e}")

    def _start_traj_joint(self, q_end, speed=1.0):
        q_now = self._env.get_joint_positions()
        try:
            traj = JointTrajectory.from_two_points(
                q_now, q_end, v_max_joint=1.0, a_max_joint=0.5
            )
            self._executor.execute(traj, speed_scale=speed)
        except Exception as e:
            self._set_error(f"Joint traj failed: {e}")

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _transition(self, new_state: State):
        print(f"[SM] {self._state.name} → {new_state.name}")
        self._state = new_state
        self._state_start_time = time.time()

    def _set_error(self, msg: str):
        print(f"[SM] ERROR: {msg}")
        self._error_msg = msg
        self._gripper.release()
        self._executor.stop()
        self._state = State.ERROR

    def _get_status(self) -> dict:
        return {
            'state':       self._state.name,
            'cycle':       self._cycle_count,
            'progress':    self._executor._traj.duration
                           if self._executor.is_running and self._executor._traj
                           else 0.0,
            'gripper':     self._gripper.is_activated(),
            'error_msg':   self._error_msg,
            'auto_repeat': self._auto_repeat,
        }
