"""Debug: Run AI in GUI mode to VISUALIZE what model does."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pybullet as p

from train_17d_place import UR5ePickPlaceEnv
from simulation.environment import UR5eEnvironment

MODEL_PATH = "models_rl_17d/seed42/best_model.zip"
VN_PATH = "models_rl_17d/seed42/vecnormalize.pkl"

def make_env():
    env = UR5ePickPlaceEnv()
    # Replace the DIRECT physics backend with a GUI one
    p.disconnect(env._env._physics_client)
    env._env = UR5eEnvironment(gui=True)
    return env

def main():
    model = SAC.load(MODEL_PATH, device="cpu")
    
    # Use exact same VecEnv logic as training
    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(VN_PATH, vec_env)
    
    # Critical: Turn off updating running stats during inference!
    vec_env.training = False
    vec_env.norm_reward = False

    print("Watching AI run... Close PyBullet window to stop.")
    
    for ep in range(5):
        obs = vec_env.reset()
        placed = False
        
        for step in range(300):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            
            # Slow down so you can see
            time.sleep(0.03)

            # DummyVecEnv returns done as a batched list/array
            if done[0]:
                placed = info[0].get('is_success', False)
                break
                
        result = "SUCCESS" if placed else "FAIL"
        print(f"Episode {ep+1}: {result} (steps={step+1})")
        time.sleep(1)

    vec_env.close()

if __name__ == "__main__":
    main()
