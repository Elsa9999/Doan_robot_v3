"""Debug: simulate EXACTLY what HMI sim_bridge does for AI mode."""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import SAC
from simulation.environment import UR5eEnvironment, CART_DELTA_MAX

EULER_DELTA_MAX = 0.08
MODEL_PATH = "models_rl_17d/seed42/best_model.zip"
VN_PATH = "models_rl_17d/seed42/vecnormalize.pkl"

# Load model + VecNormalize stats (manual, like sim_bridge)
model = SAC.load(MODEL_PATH, device="cpu")
with open(VN_PATH, "rb") as f:
    vn = pickle.load(f)
obs_rms = vn.obs_rms
clip_obs = vn.clip_obs
print(f"Loaded model + VecNormalize stats")
print(f"obs_rms.mean: {np.round(obs_rms.mean, 4)}")
print(f"obs_rms.var:  {np.round(obs_rms.var, 4)}")

# Create env (like HMI)
env = UR5eEnvironment(gui=False)
env.reset()

N_EPISODES = 20
successes = 0

for ep in range(N_EPISODES):
    env.reset()
    placed = False
    
    for step in range(300):
        # Build obs EXACTLY like sim_bridge
        ee_pos = np.array(env.get_ee_position(), dtype=np.float32)
        obj_pose = env.get_object_pose()
        obj_pos = np.array(obj_pose[0], dtype=np.float32)
        obj_quat = np.array(obj_pose[1], dtype=np.float32)
        bin_pos = np.array(env.get_bin_center(), dtype=np.float32)
        is_gripping = env.is_gripping()
        
        rel_obj = obj_pos - ee_pos
        rel_bin = bin_pos - obj_pos
        grip = np.array([1.0 if is_gripping else 0.0], dtype=np.float32)
        
        obs = np.concatenate([ee_pos, obj_pos, rel_obj, rel_bin, obj_quat, grip])
        
        # Normalize (manual, like sim_bridge)
        obs = (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
        obs = np.clip(obs, -clip_obs, clip_obs).astype(np.float32)
        
        # Predict
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(action, -1.0, 1.0)
        
        # Apply action (like sim_bridge with fix)
        delta_xyz = action[:3] * CART_DELTA_MAX
        delta_euler = action[3:6] * EULER_DELTA_MAX
        env.move_ee_cartesian(delta_xyz, delta_euler)
        
        # Hybrid gripper (like sim_bridge)
        if not is_gripping:
            env.activate_gripper()
        elif is_gripping:
            bin_target = np.array(env.get_bin_center())
            if np.linalg.norm(ee_pos - bin_target) < 0.05:
                env.release_gripper()
        
        # Step physics (10 substeps like training)
        env.step(10)
        
        # Check success
        if env.is_in_bin():
            placed = True
            break
    
    if placed:
        successes += 1
    print(f"  Ep {ep+1}: {'OK' if placed else 'FAIL'} (step={step+1}) | running={successes}/{ep+1}")

env.close()
print(f"\nFINAL: {successes}/{N_EPISODES} = {successes/N_EPISODES:.0%}")
