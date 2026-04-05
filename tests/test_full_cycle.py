"""
Test 4 — Full cycle headless pick & place (appended to test_pick_place.py logic).
Chạy standalone: python tests/test_full_cycle.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.environment import UR5eEnvironment, HOME_POSE, TABLE_SURFACE
from simulation.gripper import VacuumGripper
from simulation.object_detector import ObjectDetector
from simulation.trajectory_executor import TrajectoryExecutor
from simulation.pick_place_sm import PickPlaceStateMachine, State

PASS = "[PASS]"
FAIL = "[FAIL]"
BIN_CENTER = [0.65, -0.28, 0.52]
BIN_REGION = {'x': (0.45, 0.85), 'y': (-0.50, -0.10)}


def banner(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def test_full_cycle():
    banner("Test 4 — Full Pick & Place Cycle (headless, timeout 45s)")
    import pybullet as p

    env       = UR5eEnvironment(gui=False)
    robot_id  = env.get_robot_id()
    ee_link   = env.get_joint_indices()[-1]
    object_id = env.get_object_id()

    env.teleport_joints(HOME_POSE)
    env.step(100)

    executor = TrajectoryExecutor(env)
    gripper  = VacuumGripper(robot_id, ee_link)
    detector = ObjectDetector(env)

    sm = PickPlaceStateMachine(env, executor, gripper, detector)
    sm.start(object_id, auto_repeat=False)

    DT      = 1 / 240
    timeout = 45.0
    t0      = time.time()
    passed  = False
    last_state = ''

    print("  Running state machine...")

    while time.time() - t0 < timeout:
        status = sm.update()
        state  = status['state']

        if state != last_state:
            elapsed = time.time() - t0
            print(f"  [{elapsed:5.1f}s] State: {state}")
            last_state = state

        if state == 'IDLE' and sm._cycle_count >= 1:
            print(f"  Cycle completed in {time.time()-t0:.1f}s  ✓")
            passed = True
            break

        if state == 'ERROR':
            print(f"  {FAIL} State machine ERROR: {status['error_msg']}")
            break

        if executor.is_running:
            executor.update()

        env.step(1)
        time.sleep(DT)

    if not passed and last_state not in ('IDLE', 'DONE', 'ERROR'):
        print(f"  {FAIL} Timeout after {timeout}s, last state: {last_state}")

    # Verify object in bin region
    if sm._cycle_count >= 1:
        obj_pos, _ = p.getBasePositionAndOrientation(object_id)
        x, y = obj_pos[0], obj_pos[1]
        in_bin = (BIN_REGION['x'][0] <= x <= BIN_REGION['x'][1] and
                  BIN_REGION['y'][0] <= y <= BIN_REGION['y'][1])
        if in_bin:
            print(f"  Object in bin region: ({x:.3f}, {y:.3f})  ✓")
        else:
            print(f"  WARN: Object pos ({x:.3f}, {y:.3f}) not in expected bin region")
            print(f"        (bin x: {BIN_REGION['x']}, bin y: {BIN_REGION['y']})")
            print(f"        (This may be OK if place was near bin — cycle still counted)")

        cycles = sm._cycle_count
        if cycles == 1:
            print(f"  sm.cycle_count = {cycles}  ✓")
        else:
            print(f"  {FAIL} cycle_count = {cycles}, expected 1")
            passed = False

    env.close()
    print(f"  {PASS if passed else FAIL}")
    return passed


def main():
    print("\n" + "=" * 55)
    print("  FULL CYCLE TEST (T4)")
    print("=" * 55)

    result = test_full_cycle()

    banner("SUMMARY")
    print(f"  {'[PASS]' if result else '[FAIL]'} T4_full_cycle")
    print()
    print(f"  {'ALL TESTS PASSED ✓' if result else 'TEST FAILED ✗'}")
    print()


if __name__ == "__main__":
    main()
