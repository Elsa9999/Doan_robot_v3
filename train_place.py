"""
Tệp train_place.py:
Trung tâm huấn luyện Trí tuệ Nhân tạo (Reinforcement Learning - SAC) - Giai đoạn 2.
============= LÝ THUYẾT CURRICULUM LEARNING =============
Đầu vào 13 Chiều (13D): Toạ độ Tay(3) + Vật(3) + Tay_với_Vật(3) + Vật_với_Thùng(3) + Cảm biến kẹp(1).
Đầu ra 4 Chiều (4D Action): Di chuyển X, Di chuyển Y, Di chuyển Z, Công tắc Kẹp.

Kỹ thuật Tensor Surgery (Phẫu thuật Nơ-ron):
Lấy tệp tủy (bộ não) của giai đoạn 1 (Chỉ biết gắp - 10D) lắp vào mô hình mới (13D),
bằng cách "bơm" thêm 3 giá trị 0.01 vào các khớp nối Nơ-ron bị thiếu để nó học tiếp bước "Thả vào thùng".
"""
import os, sys, random, zipfile, io
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback,
    StopTrainingOnRewardThreshold, CallbackList
)

from simulation.environment import UR5eEnvironment, CART_DELTA_MAX, BIN_CENTER

# ── Config ─────────────────────────────────────────────────────────────────────
SEEDS       = [42]
PLACE_STEPS = 3_000_000
N_ENVS      = 1
GRASP_MODEL = os.path.join(ROOT, "models_rl_grasp", "sac", "seed42", "best_model.zip")
LOG_DIR     = os.path.join(ROOT, "logs_rl_place")
MODEL_DIR   = os.path.join(ROOT, "models_rl_place")
os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_PARAMS = dict(lr=3e-4, gamma=0.98, batch_size=256, tau=0.02)
_CURRICULUM_DIFFICULTY = [0]

# ── Lớp Định Nghĩa Bài Toán RL (Gymnasium Environment) ──
class UR5ePlaceEnv(gym.Env):
    """
    Kịch bản: Đưa tay tới -> Bật giác hút gắp -> Nhấc vật dời đi -> Canh lọt xuống lỗ -> Thả.
    Hoàn thành toàn bộ được thưởng +50 điểm (Thưởng siêu to).
    """
    MAX_STEPS      = 200   # Place cần nhiều hơn grasping
    PICK_THRESHOLD = 0.035 # Giữ ngưỡng grasping chặt

    def __init__(self):
        super().__init__()
        os.chdir(ROOT)
        self._env    = UR5eEnvironment(gui=False)
        self._steps  = 0
        self._placed = False

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(13,), dtype=np.float32)

    def _get_obs(self):
        ee_pos    = np.array(self._env.get_ee_position(),     dtype=np.float32)  # 3
        obj_pos   = np.array(self._env.get_object_pose()[0],  dtype=np.float32)  # 3
        rel_obj   = obj_pos - ee_pos                                              # 3
        bin_pos   = np.array(self._env.get_bin_center(),      dtype=np.float32)  # 3
        rel_bin   = bin_pos - obj_pos                                             # 3 — khoảng cách vật tới thùng
        grip      = np.array([1.0 if self._env.is_gripping() else 0.0],
                              dtype=np.float32)                                    # 1
        return np.concatenate([ee_pos, obj_pos, rel_obj, rel_bin, grip])          # 13

    def _compute_reward(self):
        ee_pos  = np.array(self._env.get_ee_position())
        obj_pos = np.array(self._env.get_object_pose()[0])
        bin_pos = np.array(self._env.get_bin_center())  # [x, y, TABLE+0.08]
        
        dist_ee  = float(np.linalg.norm(ee_pos - obj_pos))  # EE → vật
        dist_bin = float(np.linalg.norm(obj_pos - bin_pos)) # Vật → điểm thả (3D)

        # 1. Luôn ưu tiên khoảng cách từ Vật -> Thùng (3D)
        # Điểm thả cao 8cm, nên tự động việc nhấc lên 8cm sẽ giảm dist_bin!
        reward = -dist_bin * 4.0

        if not self._env.is_gripping():
            # 2. Phase 1: Chưa gắp
            # Phạt tiếp vì EE chưa chạm vào vật
            reward -= dist_ee * 4.0
        else:
            # 3. Phase 2: Đã kẹp được vật
            # Thưởng cố định +1.0. 
            # Toán học: Nếu gắp, mất phạt dist_ee (-0.14) và DƯỢC +1.0.
            # -> Luôn chắc chắn GẮP có lợi hơn KHÔNG GẮP!
            reward += 1.0

        if self._env.is_in_bin():
            self._placed = True
            reward += 50.0  # Bonus vào lưới (tăng lên 50 để tạo tín hiệu siêu mạnh)

        return reward, dist_ee

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._steps  = 0
        self._placed = False
        self._env.reset(difficulty=_CURRICULUM_DIFFICULTY[0])
        return self._get_obs(), {}

    def step(self, action):
        self._steps += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        delta  = action[:3] * CART_DELTA_MAX
        self._env.move_ee_cartesian(delta)
        if action[3] > 0:
            self._env.activate_gripper()
        else:
            self._env.release_gripper()
        self._env.step(10)
        obs            = self._get_obs()
        reward, dist   = self._compute_reward()
        done  = self._placed
        trunc = self._steps >= self.MAX_STEPS
        return obs, reward, done, trunc, {'is_success': self._placed, 'dist': dist}


# ── Kỹ thuật Transfer Learning: Phẫu Thuật Tensor (Tensor Surgery) ──
def load_grasp_weights_into_place_model(place_model, grasp_zip_path):
    """
    Load mô hình SAC Giai đoạn 1 (Nạp input 10D, nhả output 4D) 
    → Can thiệp vào cấp độ ma trận Pytorch (Tensors) để khoét lỗ, đắp 3 chiều mới thành 13D.
    3 chiều mới (Khoảng cách tĩnh Vật -> Thùng, dims 9-11) được nhét vào với giá trị rất nhỏ (0.01) để không làm sốc mô hình cũ.
    """
    print(f"\n[Transfer] Loading grasp weights from: {grasp_zip_path}")
    
    OLD_OBS_DIM = 10  # grasp model: ee(3)+obj(3)+rel(3)+grip(1)
    NEW_OBS_DIM = 13  # place model: ee(3)+obj(3)+rel(3)+bin_rel(3)+grip(1)
    
    # Load old state dict từ zip (SB3 lưu policy.pth bên trong)
    old_whole_sd = torch.load(
        zipfile.ZipFile(grasp_zip_path).open("policy.pth"),
        map_location='cpu', weights_only=False
    )

    new_sd = place_model.policy.state_dict()
    patched = 0
    for key, new_val in new_sd.items():
        if key not in old_whole_sd:
            continue
        old_val = old_whole_sd[key]
        if old_val.shape == new_val.shape:
            # Giống hệt → copy thẳng
            new_sd[key] = old_val.clone()
            patched += 1
        elif old_val.dim() == 2:
            # Ma trận weight có chiều input khác nhau
            old_cols = old_val.shape[1]
            new_cols = new_val.shape[1]
            if new_cols > old_cols:
                # Mở rộng: copy cột cũ, cột mới gần 0
                padded = torch.zeros_like(new_val)
                # Xác định vị trí 3 chiều rel_bin mới nằm ở đâu trong obs
                # Obs: [ee(3), obj(3), rel(3)] giữ nguyên vị trí 0-8
                # Sau đó thêm rel_bin(3) ở vị trí 9-11
                # grip(1) lúc cũ ở pos 9, giờ ở pos 12
                # Critic input = obs + action: xử lý tương tự
                if old_cols == OLD_OBS_DIM:       # actor first layer (obs only)
                    padded[:, :9]  = old_val[:, :9]  # ee+obj+rel copy
                    padded[:, 9:12] = torch.randn(new_val.shape[0], 3) * 0.01  # rel_bin mới
                    padded[:, 12:]  = old_val[:, 9:]  # grip
                elif old_cols == OLD_OBS_DIM + 4:   # critic first layer (obs + action)
                    old_act_start = OLD_OBS_DIM
                    # Copy obs part
                    padded[:, :9]         = old_val[:, :9]
                    padded[:, 9:12]       = torch.randn(new_val.shape[0], 3) * 0.01
                    padded[:, 12]         = old_val[:, 9]  # grip
                    # Copy action part (last 4 dims)
                    padded[:, NEW_OBS_DIM:] = old_val[:, old_act_start:]
                else:
                    padded[:, :min(old_cols, new_cols)] = old_val[:, :min(old_cols, new_cols)]
                new_sd[key] = padded
                patched += 1

    place_model.policy.load_state_dict(new_sd)
    print(f"[Transfer] Patched {patched} layers OK | 3 new bin_rel dims initialized near-zero")
    return place_model


# ── Vec Env factory ────────────────────────────────────────────────────────────
def make_env(seed):
    def _init():
        env = UR5ePlaceEnv()
        env.reset(seed=seed)
        return env
    return _init

def make_vec_env(seed):
    print(f"  SubprocVecEnv({N_ENVS})")
    return SubprocVecEnv([make_env(seed + i) for i in range(N_ENVS)])


# ── Training Loop ──────────────────────────────────────────────────────────────
def main():
    for seed in SEEDS:
        _CURRICULUM_DIFFICULTY[0] = 0
        print(f"\n{'='*57}")
        print(f"Training SAC PLACE | seed={seed} | {PLACE_STEPS:,} steps")
        print(f"  Transfer from: {GRASP_MODEL}")
        print(f"  Obs: 13D | Action: 4D (Cartesian + Gripper)")
        print(f"{'='*57}")

        vec_env  = make_vec_env(seed)
        ckpt_dir = os.path.join(MODEL_DIR, f"seed{seed}", "ckpt")
        log_dir  = os.path.join(LOG_DIR, f"seed{seed}")
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_cb = CheckpointCallback(
            save_freq=max(25_000 // N_ENVS, 1),
            save_path=ckpt_dir, name_prefix=f"sac_place_s{seed}")

        eval_env = DummyVecEnv([make_env(seed + 100)])
        stop_cb  = StopTrainingOnRewardThreshold(reward_threshold=10.0, verbose=1)
        # Threshold 10.0 phù hợp với reward mới: max 0/step khi carry, +20 khi thả vào thùng
        eval_cb  = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(MODEL_DIR, f"seed{seed}"),
            log_path=log_dir,
            eval_freq=max(25_000 // N_ENVS, 1),  # Dài hơn ƒ train nhiều hơn giữa 2 lần eval
            n_eval_episodes=5,
            deterministic=True,
            callback_after_eval=stop_cb)

        # Tạo SAC mới với obs 13D
        model = SAC(
            policy='MlpPolicy', env=vec_env,
            learning_rate=BEST_PARAMS['lr'],
            gamma=BEST_PARAMS['gamma'],
            batch_size=BEST_PARAMS['batch_size'],
            tau=BEST_PARAMS['tau'],
            buffer_size=200_000,
            learning_starts=5_000,
            train_freq=1, gradient_steps=1,
            use_sde=True,
            # Fix entropy collapse: giữ cố ent_coef=0.1 trong 50k bước đầu
            # Sau đó mới để SAC tự điều chỉnh. Transferred weights rất deterministic
            # nên cần giữ entropy cưỡng bức để khám phá hành vi carry+place.
            ent_coef=0.3,       # Cao hơn (0.3 thay vì 0.1) — cần explore mạnh hơn
                                  # và break ra khỏi local min "hover" của grasp model
            policy_kwargs=dict(
                log_std_init=-1,    # Rộng hơn nữa — tạo độ ngẫu nhiên lớn ngay từ đầu
                net_arch=[256, 256]
            ),
            verbose=1, seed=seed, device='auto',
            tensorboard_log=LOG_DIR
        )

        # LUON dùng tensor surgery từ grasp model (không resume place checkpoint cũ).
        # Lý do: place checkpoint cũ có thể bị kẹt ở local min. Grasp model + entropy cao hơn
        # thì tốt hơn để khám phá carry-to-bin behavior.
        if os.path.exists(GRASP_MODEL):
            print(f"[Transfer] Tensor surgery từ grasp model: {GRASP_MODEL}")
            model = load_grasp_weights_into_place_model(model, GRASP_MODEL)
        else:
            print("[WARNING] Grasp model không tìm thấy, train từ đầu")

        model.learn(
            total_timesteps=PLACE_STEPS,
            callback=CallbackList([ckpt_cb, eval_cb]),
            progress_bar=True
        )

        save_dir = os.path.join(MODEL_DIR, f"seed{seed}")
        os.makedirs(save_dir, exist_ok=True)
        model.save(os.path.join(save_dir, "sac_place_final"))
        print(f"\nSaved PLACE model → {save_dir}")
        vec_env.close()

    print("\n=== Training Pick & Place hoàn tất! ===")
    print(f"Models: {MODEL_DIR}")
    print(f"TensorBoard: tensorboard --logdir {LOG_DIR}")


if __name__ == '__main__':
    main()
