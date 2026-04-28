"""Smoke test: verify training script runs without errors."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from train_17d_place import UR5ePickPlaceEnv
import numpy as np

print("=== SMOKE TEST: Full Episode ===")
env = UR5ePickPlaceEnv()
obs, _ = env.reset()
print(f"Obs shape: {obs.shape} (expect 17)")

total_reward = 0
for i in range(100):
    action = np.zeros(7, dtype=np.float32)
    action[0] = np.clip(obs[6] * 5, -1, 1)
    action[1] = np.clip(obs[7] * 5, -1, 1)
    action[2] = np.clip(obs[8] * 5, -1, 1)

    obs, reward, done, trunc, info = env.step(action)
    total_reward += reward

    if i % 20 == 0:
        grip = "GRIP" if obs[16] > 0.5 else "free"
        print(f"  Step {i:3d} | phase={info.get('phase','?')} | {grip} | r={reward:+.2f} | total={total_reward:.1f}")

    if done:
        print(f"  >>> SUCCESS at step {i}! Total: {total_reward:.1f}")
        break

if not done:
    print(f"  End: phase={info.get('phase','?')} total={total_reward:.1f}")

env.close()
print("=== SMOKE TEST PASSED ===")
