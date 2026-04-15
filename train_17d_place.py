"""
Tệp train_17d_place.py — Phase-Based Reward Architecture

KIẾN TRÚC MỚI: Chia rõ 3 pha huấn luyện tách biệt:
  Pha 0 (APPROACH+GRASP): Tiếp cận vật, thưởng MẠNH khi gắp thành công
  Pha 1 (CARRY): Nâng cao ≥ 0.20m, bay ngang tới bin ở độ cao an toàn
  Pha 2 (PLACE): Hạ xuống thả chính xác vào bin

Mục tiêu: Loại bỏ hoàn toàn Reward Hacking gripper từ phiên bản cũ.

Observation (17D):
  - EE Position (3)
  - Obj Position (3)
  - Rel Obj (3)
  - Rel Bin (3)
  - Obj Quaternion (4)
  - Gripper (1)

Action (7D):
  - dx, dy, dz (3)
  - dRoll, dPitch, dYaw (3)
  - Grip (1)  → > 0 = cố gắp/giữ, <= 0 = nhả
"""
import os
import sys
import numpy as np
import math

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
TRAIN_STEPS = 10_000_000
N_ENVS      = 20
LOG_DIR     = os.path.join(ROOT, "logs_rl_17d")
MODEL_DIR   = os.path.join(ROOT, "models_rl_17d")
os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_PARAMS = dict(lr=3e-4, gamma=0.99, batch_size=512, tau=0.02)
_CURRICULUM_DIFFICULTY = [1]

EULER_DELTA_MAX = 0.08

# ── Phase-Based Gymnasium Environment ──────────────────────────────────────────
class UR5e17DEnv(gym.Env):
    """
    17D AI Environment với Phase-Based Reward.
    Chia rõ 3 giai đoạn để loại bỏ Reward Hacking.
    """
    MAX_STEPS   = 300
    GRASP_BONUS = 50.0    # Thưởng khi gắp lần đầu thành công
    LIFT_BONUS  = 30.0    # Thưởng khi nâng vật lên >= 0.20m
    PLACE_BONUS = 500.0   # Thưởng khi thả vào bin
    CARRY_Z_MIN = 0.62    # Độ cao tối thiểu khi vận chuyển (m)

    def __init__(self):
        super().__init__()
        os.chdir(ROOT)
        self._env     = UR5eEnvironment(gui=False)
        self._steps   = 0
        self._placed  = False
        self._phase   = 0   # 0=approach/grasp, 1=carry, 2=place
        self._lifted  = False

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(17,), dtype=np.float32)

    def _get_obs(self):
        ee_pos   = np.array(self._env.get_ee_position(),  dtype=np.float32)   # 3
        obj_pose = self._env.get_object_pose()
        obj_pos  = np.array(obj_pose[0],                  dtype=np.float32)   # 3
        obj_quat = np.array(obj_pose[1],                  dtype=np.float32)   # 4
        rel_obj  = obj_pos - ee_pos                                            # 3
        bin_pos  = np.array(self._env.get_bin_center(),   dtype=np.float32)   # 3
        rel_bin  = bin_pos - obj_pos                                           # 3
        grip     = np.array([1.0 if self._env.is_gripping() else 0.0],
                            dtype=np.float32)                                  # 1
        return np.concatenate([ee_pos, obj_pos, rel_obj, rel_bin, obj_quat, grip])  # 17

    def _compute_reward(self, action=None):
        ee_pos   = np.array(self._env.get_ee_position())
        obj_pos  = np.array(self._env.get_object_pose()[0])
        bin_pos  = np.array(self._env.get_bin_center())
        gripping = self._env.is_gripping()
        reward   = 0.0

        # ── PHA 0: APPROACH + GRASP ──────────────────────────────────────────
        if self._phase == 0:
            dist_ee = float(np.linalg.norm(ee_pos - obj_pos))

            # Thưởng tỉ lệ: càng gần vật càng nhiều điểm
            reward += max(0.0, 1.5 - dist_ee * 6.0)

            # Penalty nếu bay quá xa
            if dist_ee > 0.5:
                reward -= 3.0

            # Thưởng MẠNH khi GẮP THÀNH CÔNG (chuyển pha)
            if gripping:
                reward += self.GRASP_BONUS
                self._phase = 1

        # ── PHA 1: CARRY (Nâng lên + bay cao ngang tới bin) ─────────────────
        elif self._phase == 1:
            if not gripping:
                # Rớt rác → phạt nặng, quay về tìm lại
                reward -= 25.0
                self._phase  = 0
                self._lifted = False
            else:
                obj_height  = self._env.get_object_height()
                dist_xy_bin = float(np.hypot(
                    obj_pos[0] - bin_pos[0], obj_pos[1] - bin_pos[1]))

                # Thưởng khi nâng đủ cao (>= 0.20m trên bàn)
                if not self._lifted and obj_height >= 0.20:
                    reward += self.LIFT_BONUS
                    self._lifted = True

                # Thưởng nâng cao — khuyến khích bay lên CARRY_Z_MIN
                reward += min(obj_height, 0.25) * 4.0

                # Phạt bay thấp trong khi vận chuyển (nguy cơ quệt thành thùng)
                if self._lifted and ee_pos[2] < self.CARRY_Z_MIN:
                    reward -= 6.0

                # Thưởng tiến lại gần bin theo XY
                reward += max(0.0, 1.5 - dist_xy_bin * 4.0)

                # Khi đủ gần bin theo XY → chuyển sang Phase Place
                if self._lifted and dist_xy_bin < 0.15:
                    self._phase = 2

        # ── PHA 2: PLACE (Hạ xuống thả vào bin) ─────────────────────────────
        elif self._phase == 2:
            if not gripping:
                # Đã nhả → kiểm tra vào bin chưa
                if self._env.is_in_bin():
                    reward += self.PLACE_BONUS
                    self._placed = True
                else:
                    # Nhả lạc → phạt nặng, reset về Pha 0
                    reward -= 40.0
                    self._phase  = 0
                    self._lifted = False
            else:
                # Đang giữ rác → khuyến khích hạ xuống hướng tâm bin
                dist_bin_3d = float(np.linalg.norm(ee_pos - bin_pos))
                reward += max(0.0, 2.5 - dist_bin_3d * 6.0)

        # ── BONUS THÀNH CÔNG (fallback) ──────────────────────────────────────
        if self._env.is_in_bin() and not self._placed:
            reward += self.PLACE_BONUS
            self._placed = True

        # Phạt nhẹ action norm (chống giật cục, KHÔNG ảnh hưởng gripper)
        if action is not None:
            reward -= float(np.linalg.norm(action[:3])) * 0.03

        dist_ee = float(np.linalg.norm(ee_pos - obj_pos))
        return reward, dist_ee

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._steps  = 0
        self._placed = False
        self._phase  = 0
        self._lifted = False
        self._env.reset(difficulty=_CURRICULUM_DIFFICULTY[0])
        return self._get_obs(), {}

    def step(self, action):
        self._steps += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        delta_xyz   = action[:3] * CART_DELTA_MAX
        delta_euler = action[3:6] * EULER_DELTA_MAX

        self._env.move_ee_cartesian(delta_xyz, delta_euler)

        # action[6] > 0 → cố gắp/giữ; <= 0 → nhả
        if action[6] > 0:
            self._env.activate_gripper()
        else:
            self._env.release_gripper()

        self._env.step(10)

        obs          = self._get_obs()
        reward, dist = self._compute_reward(action)
        done  = self._placed
        trunc = self._steps >= self.MAX_STEPS

        return obs, reward, done, trunc, {
            'is_success': self._placed, 'dist': dist, 'phase': self._phase
        }

# ── Vec Env factory ────────────────────────────────────────────────────────────
def make_env(seed):
    def _init():
        env = UR5e17DEnv()
        env.reset(seed=seed)
        return env
    return _init

def make_vec_env(seed):
    print(f"  SubprocVecEnv({N_ENVS})")
    return SubprocVecEnv([make_env(seed + i) for i in range(N_ENVS)])

# ── Training Loop ──────────────────────────────────────────────────────────────
def main():
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Training SAC 17D PHASE 2 (PLACE) — Phase-Based Reward")
        print(f"  seed={seed} | {TRAIN_STEPS:,} steps | {N_ENVS} envs")
        print(f"{'='*60}")

        vec_env  = make_vec_env(seed)
        ckpt_dir = os.path.join(MODEL_DIR, f"seed{seed}", "ckpt")
        log_dir  = os.path.join(LOG_DIR,   f"seed{seed}")
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_cb = CheckpointCallback(
            save_freq=max(200_000 // N_ENVS, 1),
            save_path=ckpt_dir,
            name_prefix=f"sac_17d_s{seed}")

        eval_env = DummyVecEnv([make_env(seed + 100)])

        # Ngưỡng tốt nghiệp nâng cao: phải đạt gần PLACE_BONUS liên tục
        stop_cb = StopTrainingOnRewardThreshold(reward_threshold=480.0, verbose=1)
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(MODEL_DIR, f"seed{seed}"),
            log_path=log_dir,
            eval_freq=max(200_000 // N_ENVS, 1),
            n_eval_episodes=20,
            deterministic=True,
            callback_after_eval=stop_cb)

        # Ưu tiên load Phase 2 hiện tại → Nếu không có thì load Phase 1
        phase2_path = os.path.join(MODEL_DIR, f"seed{seed}", "best_model.zip")
        grasp_path  = os.path.join(ROOT, "models_rl_17d_grasp", f"seed{seed}", "best_model.zip")

        if os.path.exists(phase2_path):
            print(f"[RESUME] Tiếp tục từ Phase 2 model: {phase2_path}")
            model = SAC.load(
                phase2_path, env=vec_env, device='auto',
                custom_objects={"learning_rate": BEST_PARAMS['lr'],
                                "tensorboard_log": LOG_DIR})
            model.replay_buffer.reset()
        elif os.path.exists(grasp_path):
            print(f"[TRANSFER] Load từ Phase 1 Grasp model: {grasp_path}")
            model = SAC.load(
                grasp_path, env=vec_env, device='auto',
                custom_objects={"learning_rate": BEST_PARAMS['lr'],
                                "tensorboard_log": LOG_DIR})
            model.replay_buffer.reset()
        else:
            print("[SCRATCH] Không tìm thấy model cũ, train từ đầu...")
            model = SAC(
                policy='MlpPolicy', env=vec_env,
                learning_rate=BEST_PARAMS['lr'],
                gamma=BEST_PARAMS['gamma'],
                batch_size=BEST_PARAMS['batch_size'],
                tau=BEST_PARAMS['tau'],
                buffer_size=500_000,
                learning_starts=20_000,
                train_freq=1,
                gradient_steps=2,   # 2 gradient steps per env step — tốt hơn cho SAC
                use_sde=True,
                ent_coef='auto_0.1',
                policy_kwargs=dict(
                    log_std_init=-1,
                    net_arch=[512, 512, 256]  # Mạng lớn hơn cho 3-phase task
                ),
                verbose=1, seed=seed, device='auto',
                tensorboard_log=LOG_DIR
            )

        model.learn(
            total_timesteps=TRAIN_STEPS,
            callback=CallbackList([ckpt_cb, eval_cb]),
            progress_bar=True
        )

        save_dir = os.path.join(MODEL_DIR, f"seed{seed}")
        os.makedirs(save_dir, exist_ok=True)
        model.save(os.path.join(save_dir, "sac_17d_final"))
        print(f"\nSaved Phase-Based model → {save_dir}")
        vec_env.close()

    print("\n=== Training Phase-Based 17D hoàn tất! ===")

if __name__ == '__main__':
    main()
