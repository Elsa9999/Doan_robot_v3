"""Final eval 100 episodes — sac_final + matched vecnormalize."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from train_17d_place import UR5ePickPlaceEnv

N_EVAL = 100

eval_env = DummyVecEnv([lambda: UR5ePickPlaceEnv()])
eval_env = VecNormalize.load("models_rl_17d/seed42/vecnormalize.pkl", eval_env)
eval_env.training = False
eval_env.norm_reward = False

model = SAC.load("models_rl_17d/seed42/best_model.zip", device="cpu")
print(f"Loaded sac_final + matched vecnormalize")

successes = 0
rewards = []
lengths = []

for ep in range(N_EVAL):
    obs = eval_env.reset()
    total_r = 0
    for step in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        total_r += reward[0]
        if done[0]:
            break
    s = info[0].get('is_success', False)
    if s: successes += 1
    rewards.append(total_r)
    lengths.append(step + 1)
    if (ep+1) % 20 == 0:
        print(f"  {ep+1:3d}/{N_EVAL} | running={successes/(ep+1):.0%}")

eval_env.close()

rate = successes / N_EVAL
from math import sqrt
z = 1.96; p = rate; n = N_EVAL
denom = 1 + z**2/n
center = (p + z**2/(2*n)) / denom
margin = z * sqrt((p*(1-p) + z**2/(4*n)) / n) / denom

print(f"\n{'='*50}")
print(f" FINAL: {successes}/{N_EVAL} = {rate:.1%}")
print(f" 95% CI: [{max(0,center-margin):.1%} - {min(1,center+margin):.1%}]")
print(f" Reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
print(f" Length: {np.mean(lengths):.1f}")
print(f"{'='*50}")
