"""Eval 50 episodes: đo orientation thực tế ở mỗi phase."""
import os, sys, pickle, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pybullet as p

from train_17d_place import UR5ePickPlaceEnv

MODEL_PATH = "models_rl_17d/seed42/best_model.zip"
VN_PATH = "models_rl_17d/seed42/vecnormalize.pkl"

N_EPISODES = 50

def main():
    model = SAC.load(MODEL_PATH, device="cpu")
    
    # Dùng raw env (không qua VecNormalize) để đo trực tiếp
    env = UR5ePickPlaceEnv()
    
    # Load VecNormalize stats cho normalization
    with open(VN_PATH, "rb") as f:
        vn = pickle.load(f)
    obs_rms = vn.obs_rms
    clip_obs = vn.clip_obs
    
    results = []
    
    for ep in range(N_EPISODES):
        obs, _ = env.reset()
        
        phase_orient = {0: [], 1: [], 2: []}  # orientation errors per phase
        phase_steps = {0: 0, 1: 0, 2: 0}
        success = False
        max_orient_carry = 0.0
        max_orient_place = 0.0
        
        for step in range(300):
            # Normalize obs
            obs_norm = (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
            obs_norm = np.clip(obs_norm, -clip_obs, clip_obs).astype(np.float32)
            
            action, _ = model.predict(obs_norm, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            
            # Đo orientation
            ee_euler = env._get_ee_euler()
            roll_dev = abs(abs(ee_euler[0]) - np.pi)
            pitch_dev = abs(ee_euler[1])
            orient_err = roll_dev + pitch_dev
            
            phase = env._phase
            phase_orient[phase].append(orient_err)
            phase_steps[phase] = phase_steps.get(phase, 0) + 1
            
            if phase == 1:
                max_orient_carry = max(max_orient_carry, orient_err)
            elif phase == 2:
                max_orient_place = max(max_orient_place, orient_err)
            
            if done:
                success = info.get('is_success', False)
                break
        
        # Stats cho episode này
        ep_data = {
            'success': success,
            'steps': step + 1,
            'phase_steps': dict(phase_steps),
            'mean_orient_p0': np.mean(phase_orient[0]) if phase_orient[0] else 0,
            'mean_orient_p1': np.mean(phase_orient[1]) if phase_orient[1] else 0,
            'mean_orient_p2': np.mean(phase_orient[2]) if phase_orient[2] else 0,
            'max_orient_carry': max_orient_carry,
            'max_orient_place': max_orient_place,
        }
        results.append(ep_data)
        
        status = "OK" if success else "FAIL"
        orient_str = f"P0={ep_data['mean_orient_p0']:.2f} P1={ep_data['mean_orient_p1']:.2f} P2={ep_data['mean_orient_p2']:.2f}"
        print(f"  Ep {ep+1:2d}: {status} steps={step+1:3d} | orient: {orient_str} | max_carry={max_orient_carry:.2f} max_place={max_orient_place:.2f}")
    
    env.close()
    
    # ═══ TỔNG KẾT ═══
    successes = [r for r in results if r['success']]
    fails = [r for r in results if not r['success']]
    
    print(f"\n{'='*70}")
    print(f"TỔNG KẾT {N_EPISODES} EPISODES")
    print(f"{'='*70}")
    print(f"Success rate: {len(successes)}/{N_EPISODES} = {len(successes)/N_EPISODES*100:.0f}%")
    print(f"Mean steps (success): {np.mean([r['steps'] for r in successes]):.1f}" if successes else "N/A")
    
    if successes:
        print(f"\n--- Orientation Error (thấp = thẳng, 0 = hoàn hảo) ---")
        print(f"  Phase 0 (approach): mean={np.mean([r['mean_orient_p0'] for r in successes]):.3f}")
        print(f"  Phase 1 (carry):    mean={np.mean([r['mean_orient_p1'] for r in successes]):.3f}  max={np.max([r['max_orient_carry'] for r in successes]):.3f}")
        print(f"  Phase 2 (place):    mean={np.mean([r['mean_orient_p2'] for r in successes]):.3f}  max={np.max([r['max_orient_place'] for r in successes]):.3f}")
        
        # Phân loại: upright = orient_err < 0.15
        upright_carry = sum(1 for r in successes if r['mean_orient_p1'] < 0.15)
        upright_place = sum(1 for r in successes if r['mean_orient_p2'] < 0.15)
        print(f"\n  Carry thẳng đứng (<0.15): {upright_carry}/{len(successes)} = {upright_carry/len(successes)*100:.0f}%")
        print(f"  Place thẳng đứng (<0.15): {upright_place}/{len(successes)} = {upright_place/len(successes)*100:.0f}%")
        
        # Đánh giá
        mean_carry = np.mean([r['mean_orient_p1'] for r in successes])
        if mean_carry < 0.15:
            print(f"\n  [PASS] DANH GIA: Robot giu thang dung tot khi carry")
        elif mean_carry < 0.5:
            print(f"\n  [WARN] DANH GIA: Robot hoi nghieng khi carry (chap nhan duoc)")
        else:
            print(f"\n  [FAIL] DANH GIA: Robot NGHIENG NHIEU khi carry -- can tang orient penalty")
            print(f"     Khuyến nghị: tăng hệ số từ 0.5 lên 2.0-3.0 và retrain")

if __name__ == "__main__":
    main()
