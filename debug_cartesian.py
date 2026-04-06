"""Debug script: kiểm tra IK + reward thực tế của Cartesian env"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from simulation.environment import UR5eEnvironment, CART_DELTA_MAX

print("=== DEBUG: Cartesian EE Control ===\n")
env = UR5eEnvironment(gui=False)

print(f"Joint indices: {env.get_joint_indices()}")
print(f"IK target link: {env.get_joint_indices()[-1]}")

# Reset với difficulty=0
env.reset(difficulty=0)
env.step(100)

ee  = env.get_ee_position()
obj = env.get_object_pose()[0]
dist = np.linalg.norm(np.array(ee) - np.array(obj))

print(f"\nAFTER RESET (difficulty=0):")
print(f"  EE  pos : {[f'{x:.3f}' for x in ee]}")
print(f"  Obj pos : {[f'{x:.3f}' for x in obj]}")
print(f"  dist    : {dist:.4f} m  (reward = {-dist*2:.4f})")

# Test 10 bước với delta ngẫu nhiên
print(f"\nTEST 10 random Cartesian steps:")
total_reward = 0.0
for i in range(10):
    delta = np.random.uniform(-1, 1, 3) * CART_DELTA_MAX
    env.move_ee_cartesian(delta)
    env.step(10)
    ee_new  = env.get_ee_position()
    obj_new = env.get_object_pose()[0]
    dist_new = float(np.linalg.norm(np.array(ee_new) - np.array(obj_new)))
    r = -dist_new * 2
    total_reward += r
    print(f"  step {i+1:2d}: EE={[f'{x:.3f}' for x in ee_new]}  dist={dist_new:.4f}  r={r:.4f}")

print(f"\nTotal reward 10 steps: {total_reward:.4f}")
print(f"Expected ep_rew_mean (50 steps) ≈ {total_reward / 10 * 50:.1f}")

# So sánh với số quan sát được -4000
print(f"\n{'='*40}")
print(f"Observed ep_rew_mean = -4000")
print(f"If correct: avg dist should be {4000/2/50:.1f}m per step")
print(f"Actual avg dist: {total_reward/-2/10:.3f}m per step")

env.close()
print("\nDone.")
