"""
tests/test_trajectory.py — Kiểm tra trajectory planning engine.
Chạy: python tests/test_trajectory.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kinematics.trajectory import (
    trapezoid_profile, JointTrajectory, CartesianTrajectory
)

HOME_POSE = [0, -1.5708, 1.5708, -1.5708, -1.5708, 0]
PASS = "[PASS]"
FAIL = "[FAIL]"

def banner(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

def approx_equal(a, b, tol=0.01):
    return abs(a - b) < tol

def vec_approx_equal(a, b, tol=0.01):
    return all(abs(x - y) < tol for x, y in zip(a, b))


# ─── Test 1: Trapezoid profile ────────────────────────────
def test_trapezoid():
    banner("Test 1 — Trapezoid Velocity Profile")
    passed = True

    profile = trapezoid_profile(1.0, v_max=1.0, a_max=0.5, dt=1/240)

    # Bắt đầu từ 0
    if approx_equal(profile[0], 0.0):
        print(f"  profile[0] = {profile[0]:.6f} ≈ 0.0  ✓")
    else:
        print(f"  {FAIL} profile[0] = {profile[0]:.6f}, expected 0.0")
        passed = False

    # Kết thúc tại đích
    if approx_equal(profile[-1], 1.0, tol=0.005):
        print(f"  profile[-1] = {profile[-1]:.6f} ≈ 1.0  ✓")
    else:
        print(f"  {FAIL} profile[-1] = {profile[-1]:.6f}, expected 1.0")
        passed = False

    # Chỉ đi tiến (monotone increasing)
    diffs = np.diff(profile)
    if np.all(diffs >= -1e-9):
        print(f"  Monotone non-decreasing  ✓")
    else:
        print(f"  {FAIL} Profile không monotone! min diff = {diffs.min():.6f}")
        passed = False

    # Velocity peak (after resampling it may be slightly lower, check > 0.5*v_max)
    velocities = np.diff(profile) * 240   # dt = 1/240
    v_peak = velocities.max()
    if v_peak > 0.5:   # trapezoid reaches substantial velocity
        print(f"  Peak velocity = {v_peak:.4f} > 0.5 (trapezoid accel confirmed)  ✓")
    else:
        print(f"  {FAIL} Peak velocity = {v_peak:.4f}, too low")
        passed = False

    # Negative distance
    profile_neg = trapezoid_profile(-0.5, v_max=0.5, a_max=0.5, dt=1/240)
    if approx_equal(profile_neg[-1], -0.5, tol=0.01):
        print(f"  Negative distance: profile[-1] = {profile_neg[-1]:.4f} ≈ -0.5  ✓")
    else:
        print(f"  {FAIL} Negative distance: profile[-1] = {profile_neg[-1]:.4f}")
        passed = False

    print(f"  {PASS if passed else FAIL}")
    return passed


# ─── Test 2: Joint trajectory A→B ────────────────────────
def test_joint_two_points():
    banner("Test 2 — Joint Trajectory (two points)")
    passed = True

    q_start = HOME_POSE
    q_end   = [0.5, -1.2, 1.0, -1.5, -1.5, 0.3]

    traj = JointTrajectory.from_two_points(q_start, q_end, v_max_joint=1.0, a_max_joint=0.5)

    # Bắt đầu đúng
    if vec_approx_equal(traj.positions[0].tolist(), q_start, tol=0.01):
        print(f"  positions[0] ≈ q_start  ✓")
    else:
        print(f"  {FAIL} positions[0] = {traj.positions[0].tolist()}")
        passed = False

    # Kết thúc đúng
    if vec_approx_equal(traj.positions[-1].tolist(), q_end, tol=0.02):
        print(f"  positions[-1] ≈ q_end  ✓")
    else:
        print(f"  {FAIL} positions[-1] = {traj.positions[-1].tolist()}")
        passed = False

    # Velocity = 0 lúc đầu
    v0_max = max(abs(v) for v in traj.velocities[0])
    if v0_max < 0.05:
        print(f"  velocities[0] ≈ 0 (max={v0_max:.4f})  ✓")
    else:
        print(f"  {FAIL} velocities[0] max = {v0_max:.4f}, expected ≈ 0")
        passed = False

    # Velocity = 0 lúc cuối
    vf_max = max(abs(v) for v in traj.velocities[-1])
    if vf_max < 0.05:
        print(f"  velocities[-1] ≈ 0 (max={vf_max:.4f})  ✓")
    else:
        print(f"  {FAIL} velocities[-1] max = {vf_max:.4f}, expected ≈ 0")
        passed = False

    print(f"  Duration = {traj.duration:.3f}s,  N = {len(traj.timestamps)}")
    print(f"  {PASS if passed else FAIL}")
    return passed


# ─── Test 3: Cartesian trajectory straight line ───────────
def test_cartesian_straight():
    banner("Test 3 — Cartesian Trajectory (straight line)")
    passed = True

    pos_start = [-0.05, -0.10, 0.60]   # inside workspace
    pos_end   = [ 0.35,  0.10, 0.70]
    euler_z   = [0.0, 0.0, 1.571]

    traj = CartesianTrajectory.from_two_points(
        pos_start, pos_end, euler_z, euler_z,
        v_max=0.10, a_max=0.20
    )

    # Bắt đầu đúng
    pose0 = traj.get_pose(0.0)
    if vec_approx_equal(pose0['pos'], pos_start, tol=0.01):
        print(f"  pose(0) ≈ pos_start  ✓  {[f'{v:.4f}' for v in pose0['pos']]}")
    else:
        print(f"  {FAIL} pose(0) = {pose0['pos']}")
        passed = False

    # Kết thúc đúng
    pose_end = traj.get_pose(traj.duration)
    if vec_approx_equal(pose_end['pos'], pos_end, tol=0.01):
        print(f"  pose(T) ≈ pos_end  ✓  {[f'{v:.4f}' for v in pose_end['pos']]}")
    else:
        print(f"  {FAIL} pose(T) = {pose_end['pos']}")
        passed = False

    # Điểm giữa gần trung điểm
    pose_mid = traj.get_pose(traj.duration / 2)
    expected_mid = [(s + e) / 2 for s, e in zip(pos_start, pos_end)]
    mid_err = np.linalg.norm(np.array(pose_mid['pos']) - np.array(expected_mid))
    if mid_err < 0.05:
        print(f"  mid-point error = {mid_err:.4f} < 0.05  ✓")
    else:
        print(f"  {FAIL} mid-point error = {mid_err:.4f}")
        passed = False

    print(f"  Duration = {traj.duration:.3f}s")
    print(f"  {PASS if passed else FAIL}")
    return passed


# ─── Test 4: Cartesian invalid path (below table) ─────────
def test_cartesian_invalid():
    banner("Test 4 — Cartesian Invalid Path (below table)")
    passed = False

    pos_valid = [0.35, 0.0, 0.60]   # inside workspace  
    pos_bad   = [0.40, 0.0, 0.30]   # z=0.30 below z_min=0.44

    try:
        CartesianTrajectory.from_two_points(
            pos_valid, pos_bad, [0,0,0], [0,0,0],
            v_max=0.1, a_max=0.2
        )
        print(f"  {FAIL} Expected ValueError, but no exception raised!")
    except ValueError as e:
        print(f"  Got ValueError as expected: '{str(e)[:60]}'  ✓")
        passed = True

    print(f"  {PASS if passed else FAIL}")
    return passed


# ─── Test 5: Waypoints trajectory ─────────────────────────
def test_waypoints():
    banner("Test 5 — Multi-Waypoint Trajectory (CubicSpline)")
    passed = True

    waypoints = [
        HOME_POSE,
        [0.5, -1.2, 1.0, -1.5, -1.5, 0.3],
        [0.3, -1.0, 0.8, -1.2, -1.5, 0.1],
        HOME_POSE
    ]

    traj = JointTrajectory.from_waypoints(waypoints, v_max=1.0, a_max=0.5)

    # Bắt đầu đúng
    if vec_approx_equal(traj.positions[0].tolist(), waypoints[0], tol=0.01):
        print(f"  positions[0] ≈ waypoints[0]  ✓")
    else:
        print(f"  {FAIL} positions[0] = {traj.positions[0].tolist()}")
        passed = False

    # Kết thúc đúng
    if vec_approx_equal(traj.positions[-1].tolist(), waypoints[-1], tol=0.01):
        print(f"  positions[-1] ≈ waypoints[-1]  ✓")
    else:
        print(f"  {FAIL} positions[-1] = {traj.positions[-1].tolist()}")
        passed = False

    # Velocity continuity — không có bước nhảy lớn
    max_jump = 0.0
    for j in range(6):
        dv = np.abs(np.diff(traj.velocities[:, j]))
        max_jump = max(max_jump, dv.max())

    if max_jump < 1.0:
        print(f"  Max velocity jump = {max_jump:.4f} < 1.0 (smooth)  ✓")
    else:
        print(f"  {FAIL} Max velocity jump = {max_jump:.4f} (too large)")
        passed = False

    print(f"  Duration = {traj.duration:.3f}s,  N = {len(traj.timestamps)}")
    print(f"  {PASS if passed else FAIL}")
    return passed


# ─── Main ─────────────────────────────────────────────────
def main():
    print("\n" + "=" * 55)
    print("  TRAJECTORY PLANNING TESTS")
    print("=" * 55)

    results = {
        'T1_trapezoid':          test_trapezoid(),
        'T2_joint_two_points':   test_joint_two_points(),
        'T3_cartesian_straight': test_cartesian_straight(),
        'T4_cartesian_invalid':  test_cartesian_invalid(),
        'T5_waypoints':          test_waypoints(),
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
    if all_ok:
        print("  ALL TESTS PASSED ✓")
    else:
        print("  SOME TESTS FAILED ✗")
    print()


if __name__ == "__main__":
    main()
