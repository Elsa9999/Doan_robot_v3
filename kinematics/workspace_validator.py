"""
Workspace Validator — kiểm tra vị trí EE có hợp lệ không.
"""

WORKSPACE_LIMITS = {
    'x_min': -0.20,
    'x_max':  0.75,
    'y_min': -0.40,
    'y_max':  0.40,
    'z_min':  0.44,   # TABLE_SURFACE (0.42) + 0.02 safety margin
    'z_max':  1.20,

    # Vùng cấm — bin box
    'bin_forbidden': {
        'x': (0.45, 0.75),
        'y': (-0.45, -0.15),
        'z': (0.42, 0.60),
    }
}


class WorkspaceValidator:
    def __init__(self, limits=None):
        self._lim = limits if limits is not None else WORKSPACE_LIMITS

    def is_valid_ee(self, pos) -> tuple:
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        L = self._lim

        if x < L['x_min']: return False, f"X={x:.3f} < x_min={L['x_min']}"
        if x > L['x_max']: return False, f"X={x:.3f} > x_max={L['x_max']}"
        if y < L['y_min']: return False, f"Y={y:.3f} < y_min={L['y_min']}"
        if y > L['y_max']: return False, f"Y={y:.3f} > y_max={L['y_max']}"
        if z < L['z_min']: return False, f"Z={z:.3f} < z_min (duoi ban!)"
        if z > L['z_max']: return False, f"Z={z:.3f} > z_max"

        b = L['bin_forbidden']
        if (b['x'][0] <= x <= b['x'][1] and
                b['y'][0] <= y <= b['y'][1] and
                b['z'][0] <= z <= b['z'][1]):
            return False, "EE trong vung cam bin!"

        return True, "OK"

    def clamp_to_workspace(self, pos) -> list:
        L = self._lim
        x = max(L['x_min'], min(L['x_max'], float(pos[0])))
        y = max(L['y_min'], min(L['y_max'], float(pos[1])))
        z = max(L['z_min'], min(L['z_max'], float(pos[2])))
        return [x, y, z]

    def is_near_limit(self, pos, margin=0.05) -> bool:
        """True nếu EE gần biên giới (yellow warning)."""
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        L = self._lim
        return (x < L['x_min'] + margin or x > L['x_max'] - margin or
                y < L['y_min'] + margin or y > L['y_max'] - margin or
                z < L['z_min'] + margin or z > L['z_max'] - margin)

    def validate_ik_solutions(self, solutions, fk_func) -> list:
        valid = []
        for q in solutions:
            result = fk_func(q)
            pos = result['position']
            ok, _ = self.is_valid_ee(pos)
            if ok:
                valid.append(q)
        return valid
