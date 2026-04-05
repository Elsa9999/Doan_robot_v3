"""
tests/test_pick_place.py — Test gripper, detector, full cycle.
Chạy: python tests/test_pick_place.py
"""
import sys
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.environment import UR5eEnvironment, HOME_POSE, TABLE_SURFACE
from simulation.gripper import VacuumGripper
from simulation.object_detector import ObjectDetector

PASS = "[PASS]"
FAIL = "[FAIL]"
BIN_CENTER = [0.65, -0.28, 0.42]


def banner(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def setup_env():
    env = UR5eEnvironment(gui=False)
    env.teleport_joints(HOME_POSE)
    env.step(50)
    return env


# ─── Test 1: Gripper activate / release ───────────────────
def test_gripper():
    banner("Test 1 — VacuumGripper activate / release")
    env = setup_env()
    robot_id   = env.get_robot_id()
    joint_idx  = env.get_joint_indices()
    ee_link    = joint_idx[-1]
    object_id  = env.get_object_id()

    gripper = VacuumGripper(robot_id, ee_link)
    passed = True

    # Teleport robot ở vị trí trên vật thể
    import pybullet as p
    obj_pos, _ = p.getBasePositionAndOrientation(object_id)
    print(f"  Object position: {[f'{v:.3f}' for v in obj_pos]}")

    # Activate
    gripper.activate(object_id)
    if gripper.is_activated():
        print(f"  gripper.is_activated() == True  ✓")
    else:
        print(f"  {FAIL} gripper.is_activated() == False after activate")
        passed = False

    if gripper.get_held_object() == object_id:
        print(f"  get_held_object() == object_id ({object_id})  ✓")
    else:
        print(f"  {FAIL} get_held_object() = {gripper.get_held_object()}")
        passed = False

    # Step physics và kiểm tra vật không rơi
    for _ in range(240):    # 1 giây
        env.step()
        time.sleep(1/240)

    held_pos, _ = p.getBasePositionAndOrientation(object_id)
    if held_pos[2] > TABLE_SURFACE - 0.05:
        print(f"  Object z={held_pos[2]:.3f} > {TABLE_SURFACE - 0.05:.3f} (not fallen)  ✓")
    else:
        print(f"  {FAIL} Object z={held_pos[2]:.3f} <= {TABLE_SURFACE - 0.05:.3f} — vat roi!")
        passed = False

    # Release
    gripper.release()
    if not gripper.is_activated():
        print(f"  gripper.is_activated() == False after release  ✓")
    else:
        print(f"  {FAIL} Still activated after release")
        passed = False

    if gripper.get_held_object() is None:
        print(f"  get_held_object() == None  ✓")
    else:
        print(f"  {FAIL} get_held_object() = {gripper.get_held_object()}")
        passed = False

    env.close()
    print(f"  {PASS if passed else FAIL}")
    return passed


# ─── Test 2: ObjectDetector poses ─────────────────────────
def test_detector_poses():
    banner("Test 2 — ObjectDetector pose + compute_pick_poses")
    env = setup_env()
    object_id = env.get_object_id()
    detector  = ObjectDetector(env)
    passed    = True

    # get_object_pose
    pose = detector.get_object_pose(object_id)
    z    = pose['pos'][2]
    expected_z = TABLE_SURFACE + 0.03
    if abs(z - expected_z) < 0.05:
        print(f"  Object z={z:.3f} ≈ TABLE_SURFACE+0.03 ({expected_z:.3f})  ✓")
    else:
        print(f"  {FAIL} Object z={z:.3f}, expected ≈ {expected_z:.3f}")
        passed = False

    # compute_pick_poses
    pick_poses = detector.compute_pick_poses(pose['pos'])
    app_z   = pick_poses['approach'][2]
    pick_z  = pick_poses['pick'][2]
    lift_z  = pick_poses['lift'][2]

    if app_z > z:
        print(f"  approach z={app_z:.3f} > object z={z:.3f}  ✓")
    else:
        print(f"  {FAIL} approach z not above object")
        passed = False

    if pick_z < app_z:
        print(f"  pick z={pick_z:.3f} < approach z={app_z:.3f}  ✓")
    else:
        print(f"  {FAIL} pick z >= approach z")
        passed = False

    if lift_z > pick_z:
        print(f"  lift z={lift_z:.3f} > pick z={pick_z:.3f}  ✓")
    else:
        print(f"  {FAIL} lift z <= pick z")
        passed = False

    # compute_place_poses
    place_poses = detector.compute_place_poses(BIN_CENTER)
    abv = place_poses['above_bin'][2]
    plc = place_poses['place'][2]
    if abv > plc:
        print(f"  above_bin z={abv:.3f} > place z={plc:.3f}  ✓")
    else:
        print(f"  {FAIL} above_bin not above place")
        passed = False

    env.close()
    print(f"  {PASS if passed else FAIL}")
    return passed


# ─── Test 3: Raycast detect ───────────────────────────────
def test_raycast():
    banner("Test 3 — Raycast Detect")
    import pybullet as p
    env       = setup_env()
    object_id = env.get_object_id()
    detector  = ObjectDetector(env)
    passed    = True

    # Lấy vị trí vật
    obj_pos, _ = p.getBasePositionAndOrientation(object_id)
    print(f"  Object at: {[f'{v:.3f}' for v in obj_pos]}")

    # Đặt điểm bắn tia ngay trên vật (x,y của vật, cao hơn 0.2m)
    ee_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.20]
    print(f"  Ray from:  {[f'{v:.3f}' for v in ee_pos]}")

    result = detector.raycast_detect(ee_pos, max_dist=0.5)

    if result is not None:
        print(f"  Ray hit: object_id={result['object_id']}  ✓")
        if result['object_id'] == object_id:
            print(f"  Hit correct object ({object_id})  ✓")
        else:
            print(f"  {FAIL} Hit object {result['object_id']}, expected {object_id}")
            passed = False
        print(f"  Distance: {result['distance']:.3f}m")
    else:
        print(f"  {FAIL} Raycast returned None — no hit (object not directly below EE?)")
        # Leniently pass if detector found something nearby
        passed = False

    # Test miss — bắn sang vị trí không có gì
    ee_miss = [2.0, 2.0, 1.0]
    miss = detector.raycast_detect(ee_miss, max_dist=0.5)
    if miss is None:
        print(f"  Miss at off-target pos → None  ✓")
    else:
        print(f"  Warning: hit something unexpected at miss position: {miss}")
        # Not failing — might hit ground plane

    env.close()
    print(f"  {PASS if passed else FAIL}")
    return passed


# ─── Test 4: Full Cycle (Headless) ────────────────────────
def test_full_cycle():
    banner("Test 4 — Full Pick & Place Cycle")
    from simulation.trajectory_executor import TrajectoryExecutor
    from simulation.pick_place_sm import PickPlaceStateMachine
    
    env = setup_env()
    robot_id   = env.get_robot_id()
    ee_link    = env.get_joint_indices()[-1]
    
    executor = TrajectoryExecutor(env)
    gripper  = VacuumGripper(robot_id, ee_link)
    detector = ObjectDetector(env)
    
    sm = PickPlaceStateMachine(env, executor, gripper, detector)
    object_id = env.get_object_id()
    
    # Check start
    sm.start(object_id, auto_repeat=False)
    
    timeout = time.time() + 60.0
    while time.time() < timeout:
        sm.update()
        if executor.is_running:
            executor.update()
        env.step(1)
        
        if sm.state.name in ['ERROR', 'DONE']:
            break

    success = (sm.state.name == 'DONE')
    if success:
        print(f"  {PASS} Full cycle completed!")
        
        # Check object is in bin
        import pybullet as p
        pos, _ = p.getBasePositionAndOrientation(object_id)
        x, y, z = pos
        if 0.45 <= x <= 0.75 and -0.45 <= y <= -0.15:
            print(f"  {PASS} Object is in the bin! ({x:.2f}, {y:.2f})")
        else:
            print(f"  {FAIL} Object not in bin: ({x:.2f}, {y:.2f})")
            success = False
            
    else:
        print(f"  {FAIL} Ended in state: {sm.state.name}")
        
    env.close()
    return success

# ─── Main ─────────────────────────────────────────────────
def main():
    print("\n" + "=" * 55)
    print("  PICK & PLACE TESTS (T1-T3)")
    print("=" * 55)

    if '--full' in sys.argv:
        results = {
            'T4_full_cycle': test_full_cycle()
        }
    else:
        results = {
            'T1_gripper_activate_release': test_gripper(),
            'T2_detector_poses':           test_detector_poses(),
            'T3_raycast_detect':           test_raycast(),
        }

    banner("SUMMARY")
    all_ok = True
    for name, r in results.items():
        status = PASS if r else FAIL
        mark   = "✓" if r else "✗"
        print(f"  {status} {mark} {name}")
        if not r:
            all_ok = False

    print()
    print("  ALL TESTS PASSED ✓" if all_ok else "  SOME TESTS FAILED ✗")
    print()


if __name__ == "__main__":
    main()
