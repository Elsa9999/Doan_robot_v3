"""
Tệp train_17d_grasp.py:
HUẤN LUYỆN GIAI ĐOẠN 1: Bắt robot chỉ loanh quanh đi gắp chai (Không mang ra thùng).
Observation (17D) & Action (7D) ĐƯỢC GIỮ NGUYÊN để không lệch form model.
"""
import os, sys, random
import numpy as np
import torch
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

from simulation.environment import UR5eEnvironment, CART_DELTA_MAX

# ── Config ─────────────────────────────────────────────────────────────────────
SEEDS       = [42]
TRAIN_STEPS = 3_000_000   # Vì có 4 luồng chạy siêu tốc nên mình ép nó học tới 3 triệu bước cho xương khớp lật mượt mà!
N_ENVS      = 4
LOG_DIR     = os.path.join(ROOT, "logs_rl_17d_grasp")
MODEL_DIR   = os.path.join(ROOT, "models_rl_17d_grasp")
os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_PARAMS = dict(lr=3e-4, gamma=0.98, batch_size=256, tau=0.02)
_CURRICULUM_DIFFICULTY = [1]  # Start with difficulty 1 (can spawn sideways)

EULER_DELTA_MAX = 0.08  # Max rotation per step (~4.5 degrees)

# ── Gymnasium Environment ──────────────────────────────────────────────────────
class UR5e17DGraspEnv(gym.Env):
    """
    17D AI Environment - CHỈ TẬP TRUNG GẮP VÀ NÂNG LÊN.
    """
    MAX_STEPS      = 150 

    def __init__(self):
        super().__init__()
        os.chdir(ROOT)
        self._env    = UR5eEnvironment(gui=False)
        self._steps  = 0
        self._picked = False

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(17,), dtype=np.float32)

    def _get_obs(self):
        ee_pos    = np.array(self._env.get_ee_position(),     dtype=np.float32)  # 3
        obj_pose  = self._env.get_object_pose()
        obj_pos   = np.array(obj_pose[0],  dtype=np.float32)                     # 3
        obj_quat  = np.array(obj_pose[1],  dtype=np.float32)                     # 4
        
        rel_obj   = obj_pos - ee_pos                                              # 3
        bin_pos   = np.array(self._env.get_bin_center(),      dtype=np.float32)  # 3
        rel_bin   = bin_pos - obj_pos                                             # 3
        grip      = np.array([1.0 if self._env.is_gripping() else 0.0],
                              dtype=np.float32)                                   # 1
                              
        return np.concatenate([ee_pos, obj_pos, rel_obj, rel_bin, obj_quat, grip]) # 17

    def _compute_reward(self):
        ee_pos  = np.array(self._env.get_ee_position())
        obj_pos = np.array(self._env.get_object_pose()[0])
        dist    = float(np.linalg.norm(ee_pos - obj_pos))
        
        rew_dist = -dist * 2.0
        rew_lift = 0.0
        success_bonus = 0.0
        
        if self._env.is_gripping():
            lift_height = self._env.get_object_height()
            # Thưởng nhẹ khi nhấc lên
            rew_lift = lift_height * 10.0
            if lift_height >= 0.15:
                self._picked = True
                # THƯỞNG CỰC LỚN ĐỂ NÓ MUỐN CHUỒN SỚM (1 Episode = max 150 step)
                success_bonus = 200.0
                
        # Phạt quay lưng bỏ chạy xa
        if dist > 0.4:
            rew_dist -= 5.0

        return rew_dist + rew_lift + success_bonus, dist

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._steps  = 0
        self._picked = False
        self._env.reset(difficulty=_CURRICULUM_DIFFICULTY[0])
        return self._get_obs(), {}

    def step(self, action):
        self._steps += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        delta_xyz   = action[:3] * CART_DELTA_MAX
        delta_euler = action[3:6] * EULER_DELTA_MAX
        
        self._env.move_ee_cartesian(delta_xyz, delta_euler)
        
        if action[6] > 0:
            self._env.activate_gripper()
        else:
            self._env.release_gripper()
            
        self._env.step(10)
        
        obs            = self._get_obs()
        reward, dist   = self._compute_reward()
        done  = self._picked
        trunc = self._steps >= self.MAX_STEPS
        
        return obs, reward, done, trunc, {'is_success': self._picked, 'dist': dist}

# ── Vec Env factory ────────────────────────────────────────────────────────────
def make_env(seed):
    def _init():
        env = UR5e17DGraspEnv()
        env.reset(seed=seed)
        return env
    return _init

def make_vec_env(seed):
    print(f"  SubprocVecEnv({N_ENVS})")
    return SubprocVecEnv([make_env(seed + i) for i in range(N_ENVS)])

# ── Training Loop ──────────────────────────────────────────────────────────────
def main():
    for seed in SEEDS:
        print(f"\n{'='*57}")
        print(f"Training SAC 17D PHASE 1 (GRASP) | seed={seed} | {TRAIN_STEPS:,} steps")
        print(f"{'='*57}")

        vec_env  = make_vec_env(seed)
        ckpt_dir = os.path.join(MODEL_DIR, f"seed{seed}", "ckpt")
        log_dir  = os.path.join(LOG_DIR, f"seed{seed}")
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_cb = CheckpointCallback(
            save_freq=max(50_000 // N_ENVS, 1),
            save_path=ckpt_dir, name_prefix=f"sac_17d_grasp_s{seed}")

        eval_env = DummyVecEnv([make_env(seed + 100)])
        # Ngưỡng phải lên tới 150.0 mới được tính là "Tốt nghiệp"
        stop_cb  = StopTrainingOnRewardThreshold(reward_threshold=150.0, verbose=1)
        eval_cb  = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(MODEL_DIR, f"seed{seed}"),
            log_path=log_dir,
            eval_freq=max(50_000 // N_ENVS, 1), 
            n_eval_episodes=5,
            deterministic=True,
            callback_after_eval=stop_cb)

        model = SAC(
            policy='MlpPolicy', env=vec_env,
            learning_rate=BEST_PARAMS['lr'],
            gamma=BEST_PARAMS['gamma'],
            batch_size=BEST_PARAMS['batch_size'],
            tau=BEST_PARAMS['tau'],
            buffer_size=300_000,
            learning_starts=10_000,
            train_freq=1, gradient_steps=1,
            use_sde=True,
            ent_coef='auto_0.1', 
            policy_kwargs=dict(
                log_std_init=-1,    
                net_arch=[256, 256, 256] 
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
        model.save(os.path.join(save_dir, "sac_17d_grasp_final"))
        print(f"\nSaved 17D Grasp Model → {save_dir}")
        vec_env.close()

if __name__ == '__main__':
    main()
