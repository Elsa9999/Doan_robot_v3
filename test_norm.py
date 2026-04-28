"""Verify manual normalization matches SB3 wrapper - same obs."""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from train_17d_place import UR5ePickPlaceEnv

VN_PATH = "models_rl_17d/seed42/vecnormalize.pkl"

# Load SAME stats both ways
# Way 1: SB3 wrapper
env = DummyVecEnv([lambda: UR5ePickPlaceEnv()])
wrapped = VecNormalize.load(VN_PATH, env)
wrapped.training = False
wrapped.norm_reward = False

# Way 2: Manual pickle  
with open(VN_PATH, "rb") as f:
    vn = pickle.load(f)

# Print stats comparison
print("SB3 wrapper obs_rms.mean:", np.round(wrapped.obs_rms.mean, 4))
print("Manual pickle obs_rms.mean:", np.round(vn.obs_rms.mean, 4))
print("Same?", np.allclose(wrapped.obs_rms.mean, vn.obs_rms.mean))
print()
print("SB3 wrapper obs_rms.var:", np.round(wrapped.obs_rms.var, 4))
print("Manual pickle obs_rms.var:", np.round(vn.obs_rms.var, 4))
print("Same?", np.allclose(wrapped.obs_rms.var, vn.obs_rms.var))
print()
print("clip_obs:", wrapped.clip_obs, "vs", vn.clip_obs)
print("epsilon:", wrapped.epsilon)

# Test same raw obs
raw_obs = np.array([0.3, 0.0, 0.65, 0.5, 0.1, 0.45, -0.2, -0.1, 0.2, 
                     0.2, 0.1, -0.2, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

sb3_norm = wrapped.normalize_obs(raw_obs)
manual_norm = np.clip(
    (raw_obs - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + 1e-8),
    -vn.clip_obs, vn.clip_obs
).astype(np.float32)

print("\nSame raw obs test:")
print("SB3:   ", np.round(sb3_norm, 4))
print("Manual:", np.round(manual_norm, 4))
print("Match?", np.allclose(sb3_norm, manual_norm, atol=1e-5))

wrapped.close()
