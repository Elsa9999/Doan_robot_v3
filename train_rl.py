"""
Train SAC & PPO cho UR5e Pick & Place — Joint-Space Control
============================================================
Thiết kế lại hoàn toàn:
- Bỏ IK solver → điều khiển trực tiếp 6 khớp (joint-space)
- Lấy vị trí end-effector trực tiếp từ PyBullet (không qua FK tự viết)
- Reward đơn giản: tiến gần = thưởng, xa = phạt

Dùng: set PYTHONUTF8=1 && python train_rl.py
"""
import os, sys, random
import numpy as np
import torch

# ── Paths ──────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# ── Config ─────────────────────────────────────────────────
SEEDS     = [42]
SAC_STEPS = 3_000_000
PPO_STEPS = 3_000_000
N_ENVS    = 1   # PyBullet DIRECT không thread-safe với DummyVecEnv multi-env
                # N_ENVS>1 → 2 physics clients, tất cả p.* gọi client cuối → dist=47m!
LOG_DIR   = os.path.join(ROOT, "logs_rl_grasp")
MODEL_DIR = os.path.join(ROOT, "models_rl_grasp")

os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ── Imports ─────────────────────────────────────────────────
import gymnasium as gym
from gymnasium import spaces
from simulation.environment import UR5eEnvironment

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback,
    StopTrainingOnRewardThreshold, CallbackList)
from stable_baselines3.common.monitor import Monitor


# ════════════════════════════════════════════════════════════
#  Gym Environment — Cartesian EE Control (giống FetchReach)
# ════════════════════════════════════════════════════════════
# ── Curriculum difficulty (shared across envs, tăng qua callback) ──
_CURRICULUM_DIFFICULTY = [0]   # 0=gần, 1=vừa, 2=full random

# CART_DELTA_MAX import từ environment
from simulation.environment import CART_DELTA_MAX


class UR5eGymEnv(gym.Env):
    """UR5e Reaching — Cartesian EE control.

    Action : (dx, dy, dz) ∈ [-1, +1]^3  → scale bởi CART_DELTA_MAX (0.05m)
    Obs    : ee_pos(3) + obj_pos(3) + rel(3) = 9D   (giống FetchReach)
    Reward : -dist * 2.0                             (giống FetchReach dense)
    Done   : dist < 0.05m  hoặc  steps >= MAX_STEPS
    """

    metadata        = {'render_modes': ['rgb_array']}
    MAX_STEPS       = 150         # Pick & Lift cần thời gian dài hơn
    PICK_THRESHOLD  = 0.06        # 6cm cho grasping

    def __init__(self):
        super().__init__()
        os.chdir(ROOT)
        self._env    = UR5eEnvironment(gui=False)
        self._steps  = 0
        self._picked = False

        # Action: 3D Cartesian delta + 1D gripper id
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation: ee_pos(3) + obj_pos(3) + rel(3) + grip(1) = 10D
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(10,), dtype=np.float32)

    def _get_obs(self):
        ee_pos  = np.array(self._env.get_ee_position(),      dtype=np.float32)  # 3
        obj_pos = np.array(self._env.get_object_pose()[0],   dtype=np.float32)  # 3
        rel     = obj_pos - ee_pos                                              # 3
        grip    = np.array([1.0 if self._env.is_gripping() else 0.0], dtype=np.float32) # 1
        return np.concatenate([ee_pos, obj_pos, rel, grip])                     # 10

    def _compute_reward(self):
        """Reward system cho Pick & Lift."""
        ee_pos  = np.array(self._env.get_ee_position())
        obj_pos = np.array(self._env.get_object_pose()[0])
        dist    = float(np.linalg.norm(ee_pos - obj_pos))
        
        rew_dist = -dist * 2.0
        rew_lift = 0.0
        success_bonus = 0.0
        
        if self._env.is_gripping():
            lift_height = self._env.get_object_height()
            rew_lift = lift_height * 10.0
            if lift_height >= 0.15:
                self._picked = True
                success_bonus = 5.0
                
        return rew_dist + rew_lift + success_bonus, dist

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._steps  = 0
        self._picked = False
        self._env.reset(difficulty=_CURRICULUM_DIFFICULTY[0])
        return self._get_obs(), {}

    def step(self, action):
        self._steps += 1
        action  = np.clip(action, -1.0, 1.0).astype(np.float32)
        delta   = action[:3] * CART_DELTA_MAX
        self._env.move_ee_cartesian(delta)
        
        if action[3] > 0:
            self._env.activate_gripper()
        else:
            self._env.release_gripper()
            
        self._env.step(10)
        obs              = self._get_obs()
        reward, cur_dist = self._compute_reward()
        done  = self._picked
        trunc = self._steps >= self.MAX_STEPS
        return obs, reward, done, trunc, {'is_success': self._picked, 'dist': cur_dist}

    def render(self): pass

    def close(self):
        try: self._env.close()
        except Exception: pass


# ── Helpers ──────────────────────────────────────────────────
def make_env(seed):
    def _init():
        env = Monitor(UR5eGymEnv(),
                      filename=os.path.join(LOG_DIR, f"monitor_{seed}"))
        env.reset(seed=seed)
        return env
    return _init


def make_vec_env(seed):
    fns = [make_env(seed + i) for i in range(N_ENVS)]
    print(f"  SubprocVecEnv({N_ENVS})")
    return SubprocVecEnv(fns)


# [Phase 3] Curriculum callback — tự động tăng difficulty khi agent giỏi lên
from stable_baselines3.common.callbacks import BaseCallback

class CurriculumCallback(BaseCallback):
    """
    Theo dõi success rate mỗi CHECK_FREQ steps.
    Khi success rate >= THRESHOLD → tăng _CURRICULUM_DIFFICULTY lên 1 level.
    Stage 0: object spawn trong r=0.15m từ EE home  (dễ)
    Stage 1: object spawn trong r=0.25m             (vừa)
    Stage 2: full random theo WORK_ZONE             (khó — giống như cũ)
    """
    CHECK_FREQ  = 10_000   # check mỗi 10k steps
    THRESHOLD   = 0.50     # 50% success rate → lên level

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._ep_successes = []
        self._last_check   = 0

    def _on_step(self) -> bool:
        # Thu thập is_success từ info dict
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'is_success' in info:
                self._ep_successes.append(float(info['is_success']))

        if self.num_timesteps - self._last_check >= self.CHECK_FREQ:
            self._last_check = self.num_timesteps
            if len(self._ep_successes) >= 20:     # cần đủ mẫu
                rate = np.mean(self._ep_successes[-100:])
                lvl  = _CURRICULUM_DIFFICULTY[0]
                if rate >= self.THRESHOLD and lvl < 2:
                    _CURRICULUM_DIFFICULTY[0] = lvl + 1
                    print(f"\n[Curriculum] Step {self.num_timesteps:,} | "
                          f"success={rate:.1%} → difficulty {lvl}→{lvl+1}")
        return True


def build_callbacks(algo, seed):
    ckpt_dir = os.path.join(MODEL_DIR, algo, f"seed{seed}", "ckpt")
    log_dir  = os.path.join(LOG_DIR, algo, f"seed{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    ckpt_cb = CheckpointCallback(
        save_freq=max(25_000 // N_ENVS, 1),
        save_path=ckpt_dir,
        name_prefix=f"{algo}_s{seed}")

    eval_env = DummyVecEnv([make_env(seed + 100)]) # Đánh giá chỉ 1 env nên Dummy là đủ

    # Threshold cho Pick & Lift: ep_rew > 0 nghĩa là lift_reward > dist_penalty
    # dist_penalty ~-45 | lift_reward lên đến +15 | success_bonus +5
    # Chỉ stop khi agent THỰC SỰ thành thạo lift (ep_rew > 5.0)
    stop_cb  = StopTrainingOnRewardThreshold(reward_threshold=5.0, verbose=1)
    eval_cb  = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_DIR, algo, f"seed{seed}"),
        log_path=log_dir,
        eval_freq=max(10_000 // N_ENVS, 1),
        n_eval_episodes=5,
        deterministic=True,
        callback_after_eval=stop_cb)

    curriculum_cb = CurriculumCallback(verbose=1)   # [Phase 3]
    return CallbackList([ckpt_cb, eval_cb, curriculum_cb])


# ═════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═════════════════════════════════════════════════════════════
# Conservative SAC — mục tiêu: ổn định trước, tối ưu sau
# Reward = -dist*2 ∈ [-1, 0] → Q-value bounded → critic không diverge
BEST_PARAMS = {
    'lr':         3e-4,   # Safe default, không quá lớn
    'batch_size': 256,
    'gamma':      0.99,
    'tau':        0.005,  # Default — soft update chậm, ổn
}

def main():
    for algo, AlgoCls, total_steps in [
        ('sac', SAC, SAC_STEPS),
        ('ppo', PPO, PPO_STEPS),
    ]:
        for seed in SEEDS:
            _CURRICULUM_DIFFICULTY[0] = 0   # reset difficulty mỗi run
            print(f"\n{'='*55}")
            print(f"Training {algo.upper()} | seed={seed} | {total_steps:,} steps")
            print(f"  Obs: 21D (q+dq+ee+obj+rel) | Curriculum: ON")
            print(f"{'='*55}")

            vec_env = make_vec_env(seed)
            cbs     = build_callbacks(algo, seed)

            kw = dict(policy='MlpPolicy', env=vec_env,
                      learning_rate=BEST_PARAMS['lr'],
                      gamma=BEST_PARAMS['gamma'],
                      verbose=1, seed=seed, device='auto',
                      tensorboard_log=LOG_DIR)

            if algo == 'sac':
                model = SAC(**kw,
                            batch_size=BEST_PARAMS['batch_size'],
                            tau=BEST_PARAMS['tau'],
                            buffer_size=200_000,
                            learning_starts=5_000,   # Warmup vừa đủ
                            train_freq=1,             # Simple — update mỗi step
                            gradient_steps=1,         # Simple
                            use_sde=True,             # Giữ — fix entropy collapse
                            policy_kwargs=dict(
                                log_std_init=-3,
                                net_arch=[256, 256],  # Vừa phải, không quá lớn
                            ))
            else:
                model = PPO(**kw,
                            n_steps=2048,
                            batch_size=BEST_PARAMS['batch_size'],
                            n_epochs=10, clip_range=0.2, ent_coef=0.01)

            model.learn(total_timesteps=total_steps,
                        callback=cbs, progress_bar=True)
            
            save_dir = os.path.join(MODEL_DIR, algo, f"seed{seed}")
            os.makedirs(save_dir, exist_ok=True)
            model.save(os.path.join(save_dir, f"{algo}_final"))
            # vec_env.save(os.path.join(MODEL_DIR, algo, f"vecnorm_s{seed}.pkl"))
            print(f"Saved {algo.upper()} seed={seed} -> {save_dir}")
            vec_env.close()

    print("\n=== Training hoan tat! ===")
    print(f"Models: {MODEL_DIR}")
    print(f"Logs  : {LOG_DIR}")
    print("TensorBoard: tensorboard --logdir logs_rl")

if __name__ == '__main__':
    main()
