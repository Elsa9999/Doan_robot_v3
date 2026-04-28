"""
train_17d_place.py — PHIÊN BẢN TỐI ƯU CUỐI CÙNG

Kiến trúc: Hybrid Gripper + Phase-Based Reward + VecNormalize

Tham khảo:
  - Haarnoja et al. (2019): "Soft Actor-Critic Algorithms and Applications"
  - SB3 Best Practices: VecNormalize, auto-entropy, proper tau
  - robosuite benchmarks: observation normalization, reward scaling

Nguyên lý Hybrid Gripper:
  AI KHÔNG điều khiển gripper. AI chỉ cần học bay đi đâu.
  - Phase 0: Gripper TỰ ĐỘNG gắp khi EE gần vật < 4.5cm
  - Phase 1: Gripper GIỮ CHẶT, không nhả
  - Phase 2: Gripper TỰ ĐỘNG nhả khi EE gần bin < 8cm

→ Loại bỏ hoàn toàn Reward Hacking. AI chỉ học navigation.

Observation (20D):
  [0:3]   EE xyz         [3:6]  Obj xyz
  [6:9]   Rel EE→Obj     [9:12] Rel Obj→Bin
  [12:16] Obj quaternion  [16]   Grip state
  [17:20] EE euler (roll, pitch, yaw) ← UPRIGHT REWARD

Action (7D): dx dy dz dRoll dPitch dYaw grip_unused
  → action[6] KHÔNG ĐƯỢC SỬ DỤNG (Hybrid Gripper handles it)
"""
import os
import sys
import numpy as np
import pybullet as p

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList, BaseCallback
)

from simulation.environment import UR5eEnvironment, CART_DELTA_MAX, BIN_CENTER

# ═══════════════════════════════════════════════════════════════════════════════
# CẤU HÌNH TRAINING — Tối ưu cho Core i7, 16GB RAM
# ═══════════════════════════════════════════════════════════════════════════════
SEEDS        = [42]
TRAIN_STEPS  = 10_000_000      # 10M steps
N_ENVS       = 16              # 16 song song (ổn định hơn 20 trên máy yếu)
LOG_DIR      = os.path.join(ROOT, "logs_rl_17d")
MODEL_DIR    = os.path.join(ROOT, "models_rl_17d")
os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

EULER_DELTA_MAX = 0.08         # rad/step — xoay cổ tay tối đa ~4.5°/step

# Hyperparameters chuẩn SAC cho manipulation (Haarnoja 2019 + SB3 Zoo)
SAC_PARAMS = dict(
    learning_rate  = 3e-4,     # Adam LR chuẩn (paper SAC)
    gamma          = 0.99,     # Discount — task dài (~300 steps)
    batch_size     = 256,      # Chuẩn SAC (512 cũ quá lớn → chậm gradient)
    tau            = 0.005,    # Polyak averaging — chuẩn paper (0.02 cũ quá nhanh → unstable)
    buffer_size    = 1_000_000,# 1M — đủ lưu trữ cho 10M steps off-policy
    learning_starts= 10_000,   # Warm-up exploration trước khi học
    train_freq     = 1,        # Update mỗi step
    gradient_steps = 1,        # 1 gradient/step — ổn định (2 cũ quá aggressive)
    ent_coef       = 'auto_0.1', # Khởi tạo entropy cao hơn → explore nhiều hơn
)

# Curriculum: difficulty=1 → spawn trong 25cm, 50% nằm ngang
_DIFFICULTY = 1

# ═══════════════════════════════════════════════════════════════════════════════
# GYMNASIUM ENVIRONMENT — Phase-Based + Hybrid Gripper
# ═══════════════════════════════════════════════════════════════════════════════
class UR5ePickPlaceEnv(gym.Env):
    """
    20D Observation, 7D Action (action[6] unused — Hybrid Gripper).
    Phase-Based Reward — AI chỉ học navigation, không học gripper timing.
    """
    MAX_STEPS   = 300          # ~12.5 giây simulation (300 steps × 10 physics × 1/240s)
    GRASP_BONUS = 50.0         # Thưởng gắp thành công
    LIFT_BONUS  = 30.0         # Thưởng nâng ≥ 20cm
    PLACE_BONUS = 500.0        # Thưởng thả vào bin — PHẢI lớn hơn tổng per-step rewards
    CARRY_Z_MIN = 0.60         # Độ cao tối thiểu khi carry (m, world frame)

    def __init__(self):
        super().__init__()
        os.chdir(ROOT)
        self._env     = UR5eEnvironment(gui=False)
        self._steps   = 0
        self._placed  = False
        self._phase   = 0      # 0=approach, 1=carry, 2=place
        self._lifted  = False

        # Action space: 7D liên tục [-1, 1]
        # action[0:3] = delta xyz, action[3:6] = delta euler, action[6] = unused
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # Observation space: 20D (17D cũ + 3D EE euler để học giữ thẳng)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(20,), dtype=np.float32)

    # ── Observation builder ──────────────────────────────────────────────
    def _get_ee_euler(self):
        """Lấy EE euler (roll, pitch, yaw) trong PyBullet world frame.
        Tham khảo Auto FSM: EE thẳng đứng = Roll≈0, Pitch≈π.
        """
        _, quat = self._env.get_ee_pose()
        return list(p.getEulerFromQuaternion(quat))

    def _get_obs(self):
        ee_pos   = np.array(self._env.get_ee_position(),  dtype=np.float32)
        obj_pose = self._env.get_object_pose()
        obj_pos  = np.array(obj_pose[0],                  dtype=np.float32)
        obj_quat = np.array(obj_pose[1],                  dtype=np.float32)
        rel_obj  = obj_pos - ee_pos
        bin_pos  = np.array(self._env.get_bin_center(),   dtype=np.float32)
        rel_bin  = bin_pos - obj_pos
        grip     = np.array([1.0 if self._env.is_gripping() else 0.0],
                            dtype=np.float32)
        ee_euler = np.array(self._get_ee_euler(), dtype=np.float32)
        return np.concatenate([ee_pos, obj_pos, rel_obj, rel_bin, obj_quat, grip, ee_euler])

    # ── Phase-Based Reward ───────────────────────────────────────────────
    def _compute_reward(self):
        ee_pos   = np.array(self._env.get_ee_position())
        obj_pos  = np.array(self._env.get_object_pose()[0])
        bin_pos  = np.array(self._env.get_bin_center())
        gripping = self._env.is_gripping()
        reward   = 0.0

        # Time penalty nhẹ: khuyến khích hoàn thành nhanh
        reward -= 0.05

        # ── PHASE 0: APPROACH ────────────────────────────────────────────
        if self._phase == 0:
            dist = float(np.linalg.norm(ee_pos - obj_pos))
            # Dense reward: tỉ lệ nghịch khoảng cách (max +2.0 khi dist=0)
            reward += max(0.0, 2.0 - dist * 8.0)

            if gripping:
                reward += self.GRASP_BONUS
                self._phase = 1

        # ── PHASE 1: CARRY ───────────────────────────────────────────────
        elif self._phase == 1:
            height = self._env.get_object_height()
            dist_xy = float(np.hypot(
                obj_pos[0] - bin_pos[0], obj_pos[1] - bin_pos[1]))

            # One-time bonus khi nâng ≥ 20cm
            if not self._lifted and height >= 0.20:
                reward += self.LIFT_BONUS
                self._lifted = True

            # Phạt bay thấp khi đang vận chuyển
            if self._lifted and ee_pos[2] < self.CARRY_Z_MIN:
                reward -= 3.0

            # Dense: tiến gần bin XY
            reward += max(0.0, 2.0 - dist_xy * 5.0)

            # Chuyển phase khi gần bin (không cần HARD GATE — clamp ±15° ép thẳng rồi)
            if self._lifted and dist_xy < 0.15:
                self._phase = 2

        # ── PHASE 2: PLACE ───────────────────────────────────────────────
        elif self._phase == 2:
            dist_3d = float(np.linalg.norm(ee_pos - bin_pos))
            # Dense: hạ xuống gần bin center (max +3.0)
            reward += max(0.0, 3.0 - dist_3d * 10.0)

        # ── BONUS: Thả thành công ────────────────────────────────────────
        if self._env.is_in_bin() and not self._placed:
            reward += self.PLACE_BONUS
            self._placed = True

        return reward

    # ── Step: Hybrid Gripper + Physics ───────────────────────────────────
    def step(self, action):
        self._steps += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Di chuyển EE (action[0:6])
        delta_xyz   = action[:3] * CART_DELTA_MAX
        delta_euler = action[3:6] * EULER_DELTA_MAX
        self._env.move_ee_cartesian(delta_xyz, delta_euler)

        # ═══ HYBRID GRIPPER — AI không can thiệp ═══
        if self._phase == 0:
            self._env.activate_gripper()          # Luôn cố gắp
        elif self._phase == 1:
            pass                                   # Giữ chặt
        elif self._phase == 2:
            ee  = np.array(self._env.get_ee_position())
            tgt = np.array(self._env.get_bin_center())
            if np.linalg.norm(ee - tgt) < 0.05:
                self._env.release_gripper()        # Siết 5cm → rơi đúng tâm bin

        # Chạy vật lý
        self._env.step(10)

        obs    = self._get_obs()
        reward = self._compute_reward()
        done   = self._placed
        trunc  = self._steps >= self.MAX_STEPS

        return obs, reward, done, trunc, {
            'is_success': self._placed, 'phase': self._phase
        }

    # ── Reset ────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._steps  = 0
        self._placed = False
        self._phase  = 0
        self._lifted = False
        self._straightened = False
        self._env.reset(difficulty=_DIFFICULTY)
        return self._get_obs(), {}

    def close(self):
        self._env.close()


# ═══════════════════════════════════════════════════════════════════════════════
# VEC ENV FACTORY + VecNormalize (Best Practice #1)
# ═══════════════════════════════════════════════════════════════════════════════
def make_env(seed):
    def _init():
        env = UR5ePickPlaceEnv()
        env.reset(seed=seed)
        return env
    return _init


def create_normalized_env(seed, n_envs, norm_reward=True):
    """Tạo VecEnv + VecNormalize — chuẩn hóa observation & reward tự động."""
    vec_env = SubprocVecEnv([make_env(seed + i) for i in range(n_envs)])

    # VecNormalize: chuẩn hóa obs (mean=0, std=1) và reward
    # clip_obs=10: clip outlier, gamma=0.99: running mean discount
    normed = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=norm_reward,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=SAC_PARAMS['gamma'],
    )
    return normed


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    for seed in SEEDS:
        tag = f"seed{seed}"
        ckpt_dir = os.path.join(MODEL_DIR, tag, "ckpt")
        best_dir = os.path.join(MODEL_DIR, tag)
        log_dir  = os.path.join(LOG_DIR,   tag)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(best_dir, exist_ok=True)

        print(f"\n{'='*65}")
        print(f" SAC Pick & Place — Hybrid Gripper + VecNormalize")
        print(f" seed={seed} | {TRAIN_STEPS:,} steps | {N_ENVS} envs")
        print(f" tau={SAC_PARAMS['tau']} | batch={SAC_PARAMS['batch_size']}"
              f" | grad_steps={SAC_PARAMS['gradient_steps']}")
        print(f"{'='*65}")

        # ── Tạo môi trường chuẩn hóa ──
        train_env = create_normalized_env(seed, N_ENVS, norm_reward=True)

        # Eval env: KHÔNG normalize reward (để đọc reward thật)
        eval_env  = create_normalized_env(seed + 1000, 1, norm_reward=False)

        # ── Callbacks ──
        ckpt_cb = CheckpointCallback(
            save_freq=max(250_000 // N_ENVS, 1),
            save_path=ckpt_dir,
            name_prefix=f"sac_{tag}",
            save_vecnormalize=True,           # Lưu cả VecNormalize stats!
        )

        # Custom callback: lưu VecNormalize mỗi khi best model được lưu
        class SaveVecNormOnBest(BaseCallback):
            """Lưu VecNormalize stats cùng thời điểm best_model.zip."""
            def __init__(self, save_path, train_env, verbose=0):
                super().__init__(verbose)
                self._save_path = save_path
                self._train_env = train_env
                self._prev_best = None
            def _on_step(self):
                # Check if best_model.zip just got updated
                bm = os.path.join(self._save_path, 'best_model.zip')
                if os.path.exists(bm):
                    mtime = os.path.getmtime(bm)
                    if self._prev_best is None or mtime > self._prev_best:
                        self._prev_best = mtime
                        vn_path = os.path.join(self._save_path, 'vecnormalize.pkl')
                        self._train_env.save(vn_path)
                        if self.verbose:
                            print(f"[CB] Saved VecNormalize alongside best_model")
                return True

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=best_dir,
            log_path=log_dir,
            eval_freq=max(200_000 // N_ENVS, 1),
            n_eval_episodes=20,
            deterministic=True,
        )

        save_vn_cb = SaveVecNormOnBest(best_dir, train_env, verbose=1)

        # ── Khởi tạo SAC từ đầu ──
        print("[SCRATCH] Training từ đầu (clean start)...")
        model = SAC(
            policy='MlpPolicy',
            env=train_env,
            learning_rate  = SAC_PARAMS['learning_rate'],
            gamma          = SAC_PARAMS['gamma'],
            batch_size     = SAC_PARAMS['batch_size'],
            tau            = SAC_PARAMS['tau'],
            buffer_size    = SAC_PARAMS['buffer_size'],
            learning_starts= SAC_PARAMS['learning_starts'],
            train_freq     = SAC_PARAMS['train_freq'],
            gradient_steps = SAC_PARAMS['gradient_steps'],
            ent_coef       = SAC_PARAMS['ent_coef'],
            use_sde        = False,
            policy_kwargs  = dict(net_arch=[256, 256]),
            verbose  = 1,
            seed     = seed,
            device   = 'auto',
            tensorboard_log = LOG_DIR,
        )

        # ── Train! ──
        print(f"\n>>> Training {TRAIN_STEPS:,} steps (clean)...")
        print(f">>> tensorboard --logdir {LOG_DIR}")
        print(f">>> Model → {best_dir}\n")

        model.learn(
            total_timesteps=TRAIN_STEPS,
            callback=CallbackList([ckpt_cb, eval_cb, save_vn_cb]),
            progress_bar=True,
        )

        # ── Lưu model cuối (sac_final + vecnormalize_final — cặp khớp) ──
        model.save(os.path.join(best_dir, "sac_final"))
        train_env.save(os.path.join(best_dir, "vecnormalize_final.pkl"))
        # KHÔNG ghi đè vecnormalize.pkl! Nó đã được SaveVecNormOnBest
        # callback lưu khớp với best_model.zip rồi.
        print(f"\n[DONE] sac_final.zip + vecnormalize_final.pkl saved")
        print(f"[DONE] best_model.zip + vecnormalize.pkl (matched by callback)")

        train_env.close()
        eval_env.close()

    print(f"\n{'='*65}")
    print(f" TRAINING HOÀN TẤT!")
    print(f" Chạy inference: python -m hmi.app")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
