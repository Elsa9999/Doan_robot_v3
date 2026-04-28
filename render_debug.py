import os, sys, time
import numpy as np
import imageio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import SAC
import pybullet as p
import pickle

from train_17d_place import UR5ePickPlaceEnv

MODEL_PATH = "models_rl_17d/seed42/sac_final.zip"
VN_PATH = "models_rl_17d/seed42/vecnormalize_final.pkl"

def main():
    model = SAC.load(MODEL_PATH, device="cpu")
    # Tắt GUI nhưng sẽ chụp ảnh camera PyBullet
    env = UR5ePickPlaceEnv() 
    
    with open(VN_PATH, "rb") as f:
        vn = pickle.load(f)
    
    frames = []
    
    # Setup camera
    cam_dist = 1.5
    cam_yaw = 45
    cam_pitch = -30
    cam_target = [0.0, 0.0, 0.2]
    
    # Chỉ chạy 3 episodes để nhìn rõ behavior
    for ep in range(3):
        obs, _ = env.reset()
        success = False
        
        for step in range(250):
            # Normalize obs
            obs_norm = (obs - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + 1e-8)
            obs_norm = np.clip(obs_norm, -vn.clip_obs, vn.clip_obs).astype(np.float32)
            
            action, _ = model.predict(obs_norm, deterministic=True)
            
            # --- Logic giống y hệt sim_bridge.py hiện tại ---
            # Chỉ cho dịch chuyển XYZ, khoá Euler = [0,0,0]
            delta_xyz = action[:3] * 0.05
            env._env.move_ee_cartesian(delta_xyz, [0.0, 0.0, 0.0])
            
            # Hybrid Gripper logic
            if env._phase == 0:
                env._env.activate_gripper()
            elif env._phase == 2:
                bin_target = np.array(env._env.get_bin_center())
                ee_pos = np.array(env._env.get_ee_position())
                if np.linalg.norm(ee_pos - bin_target) < 0.05:
                    env._env.release_gripper()
                    
            env._env.step(10)
            obs = env._get_obs()
            
            done = env._phase == 2 and env._env.is_in_bin()
            trunc = step >= 249
            
            # Chụp màn hình (mỗi 5 steps chụp 1 lần cho nhẹ)
            if step % 5 == 0:
                view_matrix = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=cam_target, distance=cam_dist,
                    yaw=cam_yaw, pitch=cam_pitch, roll=0, upAxisIndex=2)
                proj_matrix = p.computeProjectionMatrixFOV(
                    fov=60, aspect=float(640)/480, nearVal=0.1, farVal=100.0)
                
                w, h, rgb_px, _, _ = p.getCameraImage(
                    width=640, height=480, 
                    viewMatrix=view_matrix, projectionMatrix=proj_matrix,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL)
                
                # Convert RBGA to RGB
                rgb_arr = np.array(rgb_px, dtype=np.uint8)
                rgb_img = np.reshape(rgb_arr, (480, 640, 4))[:, :, :3]
                frames.append(rgb_img)
            
            if done:
                print(f"Ep {ep}: SUCCESS at step {step}")
                break
            if trunc:
                print(f"Ep {ep}: FAIL (timeout)")
                break

    env.close()
    
    print(f"Saving {len(frames)} frames to gif...")
    imageio.mimsave('debug_behavior.gif', frames, fps=10)
    print("Done!")

if __name__ == "__main__":
    main()
