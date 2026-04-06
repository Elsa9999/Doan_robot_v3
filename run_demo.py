"""
🤖 UR5e Pick & Place: Auto-Demo
Chạy inference (kiểm thử) mô hình Pick & Place hoàn hảo nhất (100% Success Rate).
"""
import os, sys, time
import numpy as np
from stable_baselines3 import SAC
from simulation.environment import UR5eEnvironment, CART_DELTA_MAX

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

MODEL_PATH = os.path.join(ROOT, "models_rl_place", "seed42", "best_model.zip")

def get_obs_13d(env: UR5eEnvironment) -> np.ndarray:
    ee_pos    = np.array(env.get_ee_position(),     dtype=np.float32)  
    obj_pos   = np.array(env.get_object_pose()[0],  dtype=np.float32)  
    rel_obj   = obj_pos - ee_pos                                              
    bin_pos   = np.array(env.get_bin_center(),      dtype=np.float32)  
    rel_bin   = bin_pos - obj_pos                                             
    grip      = np.array([1.0 if env.is_gripping() else 0.0], dtype=np.float32)
    return np.concatenate([ee_pos, obj_pos, rel_obj, rel_bin, grip])

def main():
    print("=" * 50)
    print("🚀 LOADING PERFECT PICK & PLACE MODEL 🚀")
    print("=" * 50)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Không tìm thấy model tại: {MODEL_PATH}")
        return

    # Khởi tạo mô trường có Bật Giao Diện (GUI=True)
    env = UR5eEnvironment(gui=True)
    
    # Load trọng số vinh quang
    # Custom_objects để tránh warning nếu gym/gymnasium version mis-match
    model = SAC.load(MODEL_PATH, device="cpu")
    print("✅ Model Loaded!")

    for episode in range(5):
        print(f"\n--- Episode {episode + 1} ---")
        env.reset(difficulty=0) # Sinh ngẫu nhiên đồ vật trên bàn
        
        step = 0
        success = False
        
        while step < 200:
            obs = get_obs_13d(env)
            
            # Predict hành động (đã được train)
            action, _ = model.predict(obs, deterministic=True)
            action = np.clip(action, -1.0, 1.0)
            
            # Thực thi Action (move cartesian)
            delta = action[:3] * CART_DELTA_MAX
            env.move_ee_cartesian(delta)
            
            # Thực thi Action (Grip)
            if action[3] > 0:
                env.activate_gripper()
            else:
                env.release_gripper()
                
            env.step(10)
            time.sleep(0.04) # Slower (tương đương 25 khung hình/giây) để nhìn rõ hơn
            step += 1
            
            # Kiểm tra xem vô lưới chưa
            if env.is_in_bin():
                print(f"🎉 THÀNH CÔNG! Trúng đích tại bước {step}")
                success = True
                time.sleep(1.0) # Dừng một giây để ngắm thành quả
                break
                
        if not success:
            print("❌ Thất bại (hết thời gian).")
        
    print("\n[DONE] Hoàn tất quá trình Test.")
    
if __name__ == "__main__":
    main()
