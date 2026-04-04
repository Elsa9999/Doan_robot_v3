import time
import sys
from simulation.environment import UR5eEnvironment
from simulation.manual_controller import ManualController
from kinematics.forward_kinematics import forward_kinematics

def main():
    print("="*50)
    print("  UR5e Manual Controller")
    print("="*50)
    print("  JOINT mode  : Q/W A/S Z/X E/R D/F C/V")
    print("  CART mode   : Arrow keys + PgUp/PgDn")
    print("  ENTER       : Toggle Joint/Cartesian")
    print("  SPACE       : Go Home")
    print("  F1          : Reset Scene")
    print("  Ctrl+C      : Thoát")
    print("="*50)

    env  = UR5eEnvironment(gui=True)
    ctrl = ManualController(env)

    print("\n[MAIN] Bắt đầu vòng lặp điều khiển...")
    print("[MAIN] Mode hiện tại: JOINT\n")

    last_print = time.time()

    try:
        while True:
            ctrl.process_keys()
            env.step(1)

            # In EE pose ra terminal mỗi 1 giây
            if time.time() - last_print > 1.0:
                fk_result = forward_kinematics(ctrl._q_current)
                pos = fk_result['position']
                print(f"[EE] x={pos[0]:.4f}  y={pos[1]:.4f}  z={pos[2]:.4f}"
                      f"  | mode={ctrl._mode.upper()}"
                      f"  | q=[{', '.join(f'{v:.2f}' for v in ctrl._q_current)}]")
                last_print = time.time()

            time.sleep(1/60)

    except KeyboardInterrupt:
        print("\n[MAIN] Dừng chương trình.")
        env.close()
        sys.exit(0)

if __name__ == "__main__":
    main()
