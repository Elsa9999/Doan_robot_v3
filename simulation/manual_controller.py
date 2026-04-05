import pybullet as p
import copy
from kinematics.forward_kinematics import forward_kinematics
from kinematics.inverse_kinematics import inverse_kinematics
from kinematics.workspace_validator import WorkspaceValidator
from simulation.environment import UR5eEnvironment, HOME_POSE
from utils.transforms import local_to_world, world_to_local

JOINT_LIMITS = {
    'lower': [-6.28, -6.28, -3.14, -6.28, -6.28, -6.28],
    'upper': [ 6.28,  6.28,  3.14,  6.28,  6.28,  6.28]
}


class ManualController:
    def __init__(self, env: UR5eEnvironment):
        self._env        = env
        self._q_current  = list(HOME_POSE)
        self._mode       = 'joint'
        self._step_joint = 0.05
        self._step_cart  = 0.01
        self._running    = True
        self._validator  = WorkspaceValidator()

        self._text_ids = []
        self._keys = self._define_keys()
        self._setup_debug_text()

    def _define_keys(self) -> dict:
        return {
            ord('q'): 'joint_0_minus',
            ord('w'): 'joint_0_plus',
            ord('a'): 'joint_1_minus',
            ord('s'): 'joint_1_plus',
            ord('z'): 'joint_2_minus',
            ord('x'): 'joint_2_plus',
            ord('e'): 'joint_3_minus',
            ord('r'): 'joint_3_plus',
            ord('d'): 'joint_4_minus',
            ord('f'): 'joint_4_plus',
            ord('c'): 'joint_5_minus',
            ord('v'): 'joint_5_plus',
            p.B3G_UP_ARROW:    'cart_x_plus',
            p.B3G_DOWN_ARROW:  'cart_x_minus',
            p.B3G_LEFT_ARROW:  'cart_y_plus',
            p.B3G_RIGHT_ARROW: 'cart_y_minus',
            p.B3G_PAGE_UP:     'cart_z_plus',
            p.B3G_PAGE_DOWN:   'cart_z_minus'
        }

    def _setup_debug_text(self):
        self._text_ids = [
            p.addUserDebugText("MODE: JOINT", [-0.8, -0.6, 1.4], textColorRGB=[1, 1, 0], textSize=1.2),
            p.addUserDebugText("J1:0 J2:0 J3:0 J4:0 J5:0 J6:0", [-0.8, -0.6, 1.3], textColorRGB=[1, 1, 1], textSize=1.0),
            p.addUserDebugText("EE: x=0 y=0 z=0", [-0.8, -0.6, 1.2], textColorRGB=[0, 1, 1], textSize=1.0),
            p.addUserDebugText("ENTER=toggle MODE, SPACE=home, F1=reset", [-0.8, -0.6, 1.1], textColorRGB=[0.8, 0.8, 0.8], textSize=0.8)
        ]
        self._update_debug_text()

    def _update_debug_text(self):
        q = self._q_current
        fk_result = forward_kinematics(q)
        pos = fk_result['position']
        ok, reason = self._validator.is_valid_ee(pos)
        ws_status = "OK" if ok else f"WARN: {reason[:20]}"

        txt1 = f"MODE: {self._mode.upper()} | WS: {ws_status}"
        txt2 = f"J1:{q[0]:.2f} J2:{q[1]:.2f} J3:{q[2]:.2f} J4:{q[3]:.2f} J5:{q[4]:.2f} J6:{q[5]:.2f}"
        txt3 = f"EE: x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f}"
        txt4 = "ENTER=toggle mode   SPACE=home   F1=reset"

        p.addUserDebugText(txt1, [-0.8, -0.6, 1.4], textColorRGB=[1, 1, 0], textSize=1.2, replaceItemUniqueId=self._text_ids[0])
        p.addUserDebugText(txt2, [-0.8, -0.6, 1.3], textColorRGB=[1, 1, 1], textSize=1.0, replaceItemUniqueId=self._text_ids[1])
        p.addUserDebugText(txt3, [-0.8, -0.6, 1.2], textColorRGB=[0, 1, 1], textSize=1.0, replaceItemUniqueId=self._text_ids[2])
        p.addUserDebugText(txt4, [-0.8, -0.6, 1.1], textColorRGB=[0.8, 0.8, 0.8], textSize=0.8, replaceItemUniqueId=self._text_ids[3])

    def _clamp_joints(self, q) -> list:
        q_clamped = []
        for i in range(6):
            val = q[i]
            lo, hi = JOINT_LIMITS['lower'][i], JOINT_LIMITS['upper'][i]
            if val < lo:
                print(f"[WARN] Joint {i+1} clamped to lower ({lo})")
                val = lo
            elif val > hi:
                print(f"[WARN] Joint {i+1} clamped to upper ({hi})")
                val = hi
            q_clamped.append(val)
        return q_clamped

    def _apply_joints(self, q):
        # Validate workspace trước khi apply
        fk_res = forward_kinematics(q)
        w_pos = local_to_world(fk_res['position'])
        ok, reason = self._validator.is_valid_ee(w_pos)
        if not ok:
            print(f"[CTRL] Blocked by workspace: {reason}")
            return

        self._q_current = self._clamp_joints(q)
        self._env.set_joint_positions(self._q_current)
        self._env.step(5)
        self._update_debug_text()

    def handle_joint_mode(self, action: str):
        if not action.startswith('joint_'): return
        parts = action.split('_')
        idx = int(parts[1])
        direction = 1 if parts[2] == 'plus' else -1
        q_new = copy.copy(self._q_current)
        q_new[idx] += direction * self._step_joint
        self._apply_joints(q_new)

    def handle_cartesian_mode(self, action: str):
        if not action.startswith('cart_'): return
        fk_res = forward_kinematics(self._q_current)
        pos, euler = local_to_world(fk_res['position'], fk_res['euler'])
        pos_list = list(pos)
        parts = action.split('_')
        axis  = parts[1]
        direction = 1 if parts[2] == 'plus' else -1
        pos_list[{'x': 0, 'y': 1, 'z': 2}[axis]] += direction * self._step_cart

        # Workspace check trước khi gọi IK
        ok, reason = self._validator.is_valid_ee(pos_list)
        if not ok:
            print(f"[CTRL] Blocked: {reason}")
            return

        l_pos, l_eul = world_to_local(pos_list, euler)
        res = inverse_kinematics(l_pos, l_eul, q_current=self._q_current)
        best = res['best']
        if best is not None:
            self._apply_joints(best)
        else:
            print(f"[CTRL] IK failed at pos: {pos_list}")

    def go_home(self):
        self._q_current = list(HOME_POSE)
        self._env.set_joint_positions(self._q_current)
        self._env.step(10)
        self._update_debug_text()
        print("[CTRL] Go home")

    def toggle_mode(self):
        self._mode = 'cartesian' if self._mode == 'joint' else 'joint'
        print(f"[CTRL] Mode: {self._mode.upper()}")
        self._update_debug_text()

    def process_keys(self):
        keys = p.getKeyboardEvents()
        for key, state in keys.items():
            if state & p.KEY_WAS_TRIGGERED:
                if key == ord(' '):
                    self.go_home()
                elif key == p.B3G_RETURN:
                    self.toggle_mode()
                elif key == p.B3G_F1:
                    self._env.reset()
                    self.go_home()
                elif key in self._keys:
                    action = self._keys[key]
                    if self._mode == 'joint' and action.startswith('joint_'):
                        self.handle_joint_mode(action)
                    elif self._mode == 'cartesian' and action.startswith('cart_'):
                        self.handle_cartesian_mode(action)
        return self._running
