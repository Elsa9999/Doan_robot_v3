"""Eval both pairs: best_model+matched_vn AND sac_final+final_vn."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from train_17d_place import UR5ePickPlaceEnv

N = 100
pairs = [
    ("best_model.zip", "vecnormalize.pkl",       "best_model + ckpt_vn (4:36-4:40)"),
    ("sac_final.zip",  "vecnormalize_final.pkl", "sac_final + final_vn (7:24-7:24)"),
]

for model_f, vn_f, label in pairs:
    mpath = f"models_rl_17d/seed42/{model_f}"
    vpath = f"models_rl_17d/seed42/{vn_f}"
    
    env = DummyVecEnv([lambda: UR5ePickPlaceEnv()])
    env = VecNormalize.load(vpath, env)
    env.training = False
    env.norm_reward = False
    
    model = SAC.load(mpath, device="cpu")
    
    ok = 0
    for ep in range(N):
        obs = env.reset()
        for step in range(300):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, info = env.step(action)
            if done[0]: break
        if info[0].get('is_success', False): ok += 1
    
    env.close()
    print(f"{label}: {ok}/{N} = {ok/N:.0%}")
