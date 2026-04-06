"""Mini test: chạy UR5eGymEnv qua Monitor+DummyVecEnv và đo ep_rew_mean thực tế"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT); sys.path.insert(0, ROOT)

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from simulation.environment import UR5eEnvironment, CART_DELTA_MAX

_CURRICULUM_DIFFICULTY = [0]

class UR5eGymEnv(gym.Env):
    MAX_STEPS = 50
    PICK_THRESHOLD = 0.05
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self):
        super().__init__()
        os.chdir(ROOT)
        self._env    = UR5eEnvironment(gui=False)
        self._steps  = 0
        self._picked = False
        self.action_space = spaces.Box(-1., 1., shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-10., 10., shape=(9,), dtype=np.float32)

    def _get_obs(self):
        ee  = np.array(self._env.get_ee_position(), dtype=np.float32)
        obj = np.array(self._env.get_object_pose()[0], dtype=np.float32)
        return np.concatenate([ee, obj, obj-ee])

    def _compute_reward(self):
        ee  = np.array(self._env.get_ee_position())
        obj = np.array(self._env.get_object_pose()[0])
        dist = float(np.linalg.norm(ee-obj))
        if dist < self.PICK_THRESHOLD:
            self._picked = True
        return -dist * 2.0, dist

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0; self._picked = False
        self._env.reset(difficulty=0)
        return self._get_obs(), {}

    def step(self, action):
        self._steps += 1
        action = np.clip(action, -1., 1.).astype(np.float32)
        self._env.move_ee_cartesian(action * CART_DELTA_MAX)
        self._env.step(10)
        obs = self._get_obs()
        reward, dist = self._compute_reward()
        done  = self._picked
        trunc = self._steps >= self.MAX_STEPS
        return obs, reward, done, trunc, {'dist': dist}

    def render(self): pass
    def close(self):
        try: self._env.close()
        except: pass

print("=== TEST: Monitor + DummyVecEnv reward tracking ===\n")

ep_rewards = []

# Test WITHOUT VecNormalize
def make():
    return Monitor(UR5eGymEnv())

vec = DummyVecEnv([make])

rng = np.random.default_rng(42)
total_ep = 0
total_steps = 0
ep_sum = 0.0
step_rewards = []

obs = vec.reset()
for step in range(500):
    action = rng.uniform(-1, 1, (1, 3)).astype(np.float32)
    obs, rewards, dones, infos = vec.step(action)
    ep_sum += float(rewards[0])
    step_rewards.append(float(rewards[0]))
    if dones[0]:
        ep_rewards.append(ep_sum)
        total_ep += 1
        ep_sum = 0.0
    total_steps += 1

print(f"Steps: {total_steps}, Episodes completed: {total_ep}")
if ep_rewards:
    print(f"ep_rew_mean (raw): {np.mean(ep_rewards):.2f}")
    print(f"Per-step reward avg: {np.mean(step_rewards):.4f}")
else:
    print(f"No completed episodes! ep_sum so far: {ep_sum:.2f}")
    print(f"Accumulated over {total_steps} steps without episode end")
    print(f"Per-step reward avg: {np.mean(step_rewards):.4f}")

vec.close()
