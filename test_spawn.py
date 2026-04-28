"""Kiểm tra spawn logic: vị trí vật, EE reach, bin overlap."""
import sys, os, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from simulation.environment import (
    UR5eEnvironment, WORK_ZONE, BIN_CENTER, TABLE_SURFACE, EE_WORKSPACE, HOME_POSE
)

env = UR5eEnvironment(gui=False)
ee_home = env.get_ee_position()

print("=" * 60)
print(" SPAWN LOGIC ANALYSIS")
print("=" * 60)

# 1. EE Home position
print(f"\n[1] EE HOME (from HOME_POSE joints):")
print(f"    x={ee_home[0]:.3f}  y={ee_home[1]:.3f}  z={ee_home[2]:.3f}")
print(f"    Spawn uses EE_HOME_XY = [0.45, 0.0]")
print(f"    Offset: dx={abs(ee_home[0]-0.45):.3f}  dy={abs(ee_home[1]):.3f}")

# 2. 20 random spawns
print(f"\n[2] Object spawn test (difficulty=1, 20 samples):")
print(f"    WORK_ZONE x={WORK_ZONE['x']} y={WORK_ZONE['y']}")
print(f"    BIN_CENTER = {BIN_CENTER}")

random.seed(42)
overlaps = 0
unreachable = 0
for i in range(20):
    EE_HOME_XY = [0.45, 0.0]
    r   = random.uniform(0.05, 0.25)
    ang = random.uniform(0, 2 * 3.14159)
    x   = EE_HOME_XY[0] + r * math.cos(ang)
    y   = EE_HOME_XY[1] + r * math.sin(ang)
    x   = max(WORK_ZONE['x'][0] + 0.05, min(WORK_ZONE['x'][1] - 0.05, x))
    y   = max(WORK_ZONE['y'][0] + 0.05, min(WORK_ZONE['y'][1] - 0.05, y))

    dist_ee  = math.hypot(x - ee_home[0], y - ee_home[1])
    dist_bin = math.hypot(x - BIN_CENTER[0], y - BIN_CENTER[1])
    near_bin = abs(x - BIN_CENTER[0]) < 0.10 and abs(y - BIN_CENTER[1]) < 0.10
    in_ws    = EE_WORKSPACE['x'][0] <= x <= EE_WORKSPACE['x'][1] and \
               EE_WORKSPACE['y'][0] <= y <= EE_WORKSPACE['y'][1]

    if near_bin:
        overlaps += 1
    if not in_ws:
        unreachable += 1

    tag = " !!BIN!!" if near_bin else ""
    print(f"    #{i:2d}: pos=({x:.3f},{y:.3f}) dist_ee={dist_ee:.2f} dist_bin={dist_bin:.2f} ws={in_ws}{tag}")

print(f"\n    Overlaps with bin area: {overlaps}/20")
print(f"    Outside EE workspace:  {unreachable}/20")

# 3. Z-axis reach
print(f"\n[3] Z-axis reach check:")
print(f"    TABLE_SURFACE = {TABLE_SURFACE}")
print(f"    Object Z (upright) = {TABLE_SURFACE + 0.033:.3f}")
print(f"    Object Z (on side) = {TABLE_SURFACE + 0.020:.3f}")
print(f"    EE Z min = {EE_WORKSPACE['z'][0]}")
print(f"    EE can touch object? {EE_WORKSPACE['z'][0] <= TABLE_SURFACE + 0.06}")
print(f"    Gap: {EE_WORKSPACE['z'][0] - (TABLE_SURFACE + 0.02):.3f}m")

# 4. Gripper reach test
print(f"\n[4] Gripper reach distance:")
print(f"    activate_gripper threshold: 0.045m")
print(f"    EE Z min ({EE_WORKSPACE['z'][0]}) - Object Z ({TABLE_SURFACE+0.02:.2f}) = {EE_WORKSPACE['z'][0]-(TABLE_SURFACE+0.02):.3f}m")
can_reach = EE_WORKSPACE['z'][0] - (TABLE_SURFACE + 0.02) < 0.045
print(f"    Can grip on-side object? {can_reach}")

env.close()
print("\n" + "=" * 60)
