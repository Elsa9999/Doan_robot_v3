import numpy as np
from kinematics.workspace_validator import WorkspaceValidator
from kinematics.trajectory import CartesianTrajectory

print("WORKSPACE BIN: ", WorkspaceValidator()._lim['bin_forbidden'])

pos_s = [0.65, -0.28, 0.70]
pos_e = [0.65, -0.28, 0.62]
eul_s = [3.14159, 0.0, 0.0]
eul_e = [3.14159, 0.0, 0.0]

try:
    cart_traj = CartesianTrajectory.from_two_points(
        pos_s, pos_e, eul_s, eul_e, v_max=0.05, a_max=0.2
    )
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
