"""Eval NO normalization — test if model works with raw obs."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import SAC
from train_17d_place import UR5ePickPlaceEnv

N_EVAL = 100
env = UR5ePickPlaceEnv()
model = SAC.load("models_rl_17d/seed42/best_model.zip", device="cpu")
print("Loaded model (NO VecNormalize)")

successes = 0
for ep in range(N_EVAL):
    obs, _ = env.reset()
    for step in range(300):
        # RAW obs — no normalization
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        if done or trunc:
            break
    if info.get('is_success', False):
        successes += 1
    if (ep + 1) % 10 == 0:
        print(f"  {ep+1:3d}/{N_EVAL} success={successes/(ep+1):.0%}")

env.close()
print(f"\nFINAL: {successes}/{N_EVAL} = {successes/N_EVAL:.0%}")
