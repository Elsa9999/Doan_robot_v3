"""
SANITY CHECK — Chạy 5 phút để biết reward đi đúng hướng không.
Dùng: set PYTHONUTF8=1 && python check_env.py
"""
import os, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ['PYTHONUTF8'] = '1'

import gymnasium as gym
from gymnasium import spaces
from simulation.environment import UR5eEnvironment

# ── Dùng CÙNG class UR5eGymEnv từ train_rl.py ──────────────
class UR5eGymEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array']}
    MAX_STEPS       = 200
    PICK_THRESHOLD  = 0.05
    JOINT_DELTA_MAX = 0.05
    JOINT_LIMITS_LOW  = [-2*np.pi]*3 + [-2*np.pi]*3
    JOINT_LIMITS_HIGH = [ 2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi]
    HOME_POSE = [0, -1.5708, 1.5708, -1.5708, -1.5708, 0]

    def __init__(self):
        super().__init__()
        os.chdir(ROOT)
        self._env = UR5eEnvironment(gui=False)
        self._steps = 0
        self._picked = False
        self._prev_dist = None
        self.action_space = spaces.Box(-1., 1., shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(-10., 10., shape=(15,), dtype=np.float32)

    def _get_obs(self):
        q       = np.array(self._env.get_joint_positions(), dtype=np.float32)
        ee_pos  = np.array(self._env.get_ee_pose()[0], dtype=np.float32)
        obj_pos = np.array(self._env.get_object_pose()[0], dtype=np.float32)
        return np.concatenate([q, ee_pos, obj_pos, obj_pos - ee_pos])

    def _compute_reward(self):
        ee_pos  = np.array(self._env.get_ee_pose()[0])
        obj_pos = np.array(self._env.get_object_pose()[0])
        dist = float(np.linalg.norm(ee_pos - obj_pos))
        r_progress = (self._prev_dist - dist) * 20.0
        r_success = 0.0
        if dist < self.PICK_THRESHOLD:
            r_success = 100.0
            self._picked = True
        r_proximity = 0.0
        if dist < 0.15:
            r_proximity = (0.15 - dist) * 5.0
        self._prev_dist = dist
        return float(r_progress + r_success + r_proximity - 0.01)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0; self._picked = False
        self._env.reset()
        obs = self._get_obs()
        ee  = np.array(self._env.get_ee_pose()[0])
        obj = np.array(self._env.get_object_pose()[0])
        self._prev_dist = float(np.linalg.norm(ee - obj))
        return obs, {}

    def step(self, action):
        self._steps += 1
        action = np.clip(action, -1., 1.).astype(np.float32)
        deltas = action * self.JOINT_DELTA_MAX
        current_q = self._env.get_joint_positions()
        new_q = [float(np.clip(current_q[i]+deltas[i],
                 self.JOINT_LIMITS_LOW[i], self.JOINT_LIMITS_HIGH[i]))
                 for i in range(6)]
        self._env.set_joint_positions(new_q)
        self._env.step(10)
        return (self._get_obs(), self._compute_reward(),
                self._picked, self._steps >= self.MAX_STEPS,
                {'picked': self._picked})

    def close(self):
        try: self._env.close()
        except: pass


# ══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  SANITY CHECK — Joint-Space Control")
print("="*55)

# ── TEST 1: Random Policy ────────────────────────────────
print("\n[TEST 1] Random Policy — 3 episodes...")
env = UR5eGymEnv()
ep_rewards = []

for ep in range(3):
    obs, _ = env.reset(seed=ep)
    total_r = 0
    for _ in range(200):
        action = env.action_space.sample()
        obs, r, done, trunc, info = env.step(action)
        total_r += r
        if done or trunc: break
    ep_rewards.append(total_r)
    print(f"  Episode {ep+1}: reward = {total_r:.1f}")

env.close()
avg_random = np.mean(ep_rewards)
print(f"\n  Avg random reward: {avg_random:.1f}")

# ── TEST 2: SAC mini-train 10k steps ─────────────────────
print("\n[TEST 2] SAC mini-train 10,000 steps...")

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

def make_env(seed):
    def _init():
        e = Monitor(UR5eGymEnv())
        e.reset(seed=seed)
        return e
    return _init

vec = VecNormalize(DummyVecEnv([make_env(42), make_env(43)]),
                   norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)

model = SAC('MlpPolicy', vec, learning_rate=3e-4, gamma=0.99,
            buffer_size=50_000, learning_starts=1_000,
            batch_size=256, tau=0.005,
            train_freq=1, gradient_steps=1,
            verbose=1, seed=42)

model.learn(total_timesteps=10_000, progress_bar=True)

# Lấy ent_coef
try:
    ent = model.ent_coef_tensor.item()
except:
    ent = 0.0

vec.close()

# ── Kết luận ─────────────────────────────────────────────
print("\n" + "="*55)
print("  KET LUAN")
print("="*55)
ent_ok = ent > 0.05
print(f"  Random reward : {avg_random:.1f}  {'OK' if avg_random > -100 else 'QUA AM'}")
print(f"  ent_coef cuoi : {ent:.4f}  {'OK' if ent_ok else 'QUA THAP'}")

ok = avg_random > -100 and ent_ok
print()
if ok:
    print("  >>> GO TRAIN! Co the chay 3 trieu step an toan.")
else:
    print("  >>> CAN XEM LAI reward truoc khi train nang.")
print("="*55 + "\n")
