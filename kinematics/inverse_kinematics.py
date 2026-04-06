"""
Tệp inverse_kinematics.py:
Lõi Toán học giải bài toán Đoán góc xoay ngược từ Tọa độ đích (Inverse Kinematics).
Hệ thống sử dụng cả 2 phương pháp:
1. Analytical (Giải tích): Tính toán nhanh siêu tốc thông qua ma trận DH và lượng giác.
2. Numerical (Số học cận biên): Dùng thuật toán L-BFGS-B khi vật vượt ngoài tầm giải tích đơn thuần (dự phòng).
"""
import numpy as np
import math
import scipy.optimize
from kinematics.forward_kinematics import forward_kinematics, DH_TABLE, dh_transform

def euler_matrix(euler):
    """
    Tạo ma trận xoay 3x3 từ 3 góc Euler ZYX (Roll, Pitch, Yaw).
    Góc Euler là cách để định nghĩa góc chéo của vật thể khi bị nghiêng trong không gian.
    """
    x, y, z = euler
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def validate_limits(q):
    """
    Kiểm tra joint limits từ URDF:
    joint 1,2,4,5,6: [-6.28, 6.28]
    joint 3:         [-3.14, 3.14]
    """
    limits = [6.28, 6.28, 3.14, 6.28, 6.28, 6.28]
    for i in range(6):
        if abs(q[i]) > limits[i]:
            return False
    return True

def analytical_ik(T_target):
    """
    Tính Inverse Kinematics (IK) bằng phương pháp Cổ điển Giải tích (Analytical).
    Đây là cách tính toán hình học trực diện, cực kỳ nhẹ và nhanh nhưng giới hạn trong không gian lý tưởng.
    Đầu vào: T_target (Ma trận 4x4 đại diện cho vị trí và góc xoay mong muốn của End-Effector).
    """
    # Lấy thông số trực tiếp từ bảng DH_TABLE
    d1 = DH_TABLE[0]['d'] 
    a2 = DH_TABLE[1]['a'] 
    a3 = DH_TABLE[2]['a'] 
    d4 = DH_TABLE[3]['d']
    d5 = DH_TABLE[4]['d']
    d6 = DH_TABLE[5]['d']
    
    px, py, pz = T_target[0,3], T_target[1,3], T_target[2,3]
    
    # 1. Tính vị trí Wrist Center (P05)
    P05 = T_target[:3, 3] - d6 * T_target[:3, 2]
    
    r = math.hypot(P05[0], P05[1])
    if r < abs(d4):
        return [] # Nếu quá sát tâm, không có solution (out of reach)
        
    phi = math.atan2(P05[1], P05[0])
    asin_val = math.asin(d4 / r)
    
    # 2. theta1 (2 solutions: Left/Right)
    th1_sols = [phi + asin_val, phi + math.pi - asin_val]
    
    sols = []
    for th1 in th1_sols:
        th1 = math.atan2(math.sin(th1), math.cos(th1))
        
        # 3. theta5
        num = T_target[0,3] * math.sin(th1) - T_target[1,3] * math.cos(th1) - d4
        c5 = num / d6
        if abs(c5) > 1.0:
            if abs(c5) < 1.001: c5 = np.sign(c5) * 1.0
            else: continue
            
        th5_val = math.acos(c5)
        
        # Có 2 solutions cho th5 ứng với mỗi th1
        for th5 in [th5_val, -th5_val]:
            if abs(math.sin(th5)) < 1e-5:
                th6_sols = [0.0]
            else:
                A = T_target[0,0] * math.sin(th1) - T_target[1,0] * math.cos(th1)
                B = T_target[0,1] * math.sin(th1) - T_target[1,1] * math.cos(th1)
                th6_sols = [math.atan2( -B / math.sin(th5), A / math.sin(th5) )]
                
            for th6 in th6_sols:
                T1 = dh_transform(0, d1, np.pi/2, th1)
                T5 = dh_transform(0, d5, -np.pi/2, th5)
                T6 = dh_transform(0, d6, 0, th6)
                
                T56 = T5 @ T6
                # Dùng inv để truy ngược lại T14
                T14 = np.linalg.inv(T1) @ T_target @ np.linalg.inv(T56)
                
                P14x, P14y = T14[0,3], T14[1,3]
                dist_sq = P14x**2 + P14y**2
                
                # 4. Tính theta3 từ định lý hàm cos
                c3 = (dist_sq - a2**2 - a3**2) / (2 * a2 * a3)
                if abs(c3) > 1.0:
                    if abs(c3) < 1.001: c3 = np.sign(c3) * 1.0
                    else: continue
                        
                th3_val = math.acos(c3)
                
                # Có 2 solutions cho th3 ứng với mỗi tổ hợp (Elbow up/down)
                for th3 in [th3_val, -th3_val]:
                    s3 = math.sin(th3)
                    
                    # Tính theta2
                    th2 = math.atan2(P14y, P14x) - math.atan2(a3 * s3, a2 + a3 * c3)
                    
                    # Lấy th4 từ tổng góc (do rục joint 2, 3, 4 song song)
                    th_sum = math.atan2(T14[1,0], T14[0,0])
                    th4 = th_sum - th2 - th3
                    
                    q = [th1, th2, th3, th4, th5, th6]
                    q = [(x + math.pi) % (2*math.pi) - math.pi for x in q] # Normalize
                    
                    if validate_limits(q):
                        sols.append(q)
    return sols

def numerical_ik(T_target, q_current=None):
    """
    Tính IK bằng Thuật toán tối ưu hóa (Numerical). Dùng làm phương án dự phòng (Fallback).
    Khi mục tiêu gắp nằm ở góc cực hẹp khiến Giải tích báo lỗi, Numerical sẽ tính đạo hàm
    để ép các khớp nghiêng nhẹ tới điểm gần đúng nhất mà không vi phạm quy tắc xoay.
    """
    if q_current is None:
        q_current = [0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0]
        
    def cost(q):
        T_curr = forward_kinematics(q)['T']
        pos_err = np.sum((T_curr[:3, 3] - T_target[:3, 3])**2)
        
        R_curr = T_curr[:3, :3]
        R_target = T_target[:3, :3]
        rot_err = 3.0 - np.trace(R_curr @ R_target.T)
        
        return pos_err + rot_err
        
    bounds = [(-6.28, 6.28)] * 6
    bounds[2] = (-3.14, 3.14)
    
    res = scipy.optimize.minimize(cost, q_current, bounds=bounds, method='L-BFGS-B', options={'maxiter': 1000})
    if res.success and res.fun < 1e-4:
        return [res.x.tolist()]
    return []

def inverse_kinematics(target_pos, target_euler, q_current=None, method='auto') -> dict:
    """
    Hàm tổng - Tính động học ngược cho End-Effector của UR5e.
    Quá trình:
    1. Lắp ráp ma trận quay và tịnh tiến vào ma trận 4x4.
    2. Chạy hàm giải tích (Analytical). 
    3. Nếu vô nghiệm, hệ thống tự động nhảy sang chạy Số học (Numerical).
    
    Tùy chọn: method có thể ép cứng về 'analytical' hoặc 'numerical', mặc định là 'auto'.
    Trả về Dictionary chứa góc xoay của 6 khớp an toàn nhất.
    """
    T_target = np.eye(4)
    T_target[:3, :3] = euler_matrix(target_euler)
    T_target[:3, 3] = target_pos
    
    sols = []
    used_method = method
    
    if method in ['auto', 'analytical']:
        sols = analytical_ik(T_target)
        if len(sols) > 0:
            used_method = 'analytical'
            
    if len(sols) == 0 and method in ['auto', 'numerical']:
        sols = numerical_ik(T_target, q_current)
        used_method = 'numerical'
        
    best_sol = None
    if len(sols) > 0:
        if q_current is None:
            best_sol = sols[0]
        else:
            # Chọn solution gần với q_current nhất
            best_sol = min(sols, key=lambda q: np.linalg.norm(np.array(q) - np.array(q_current)))
            
    # Tính Errors cho từng solution
    errors = []
    for sol in sols:
        T_sol = forward_kinematics(sol)['T']
        pos_err = np.linalg.norm(T_sol[:3, 3] - T_target[:3, 3])
        errors.append(float(pos_err))
        
    if best_sol is None:
        print("[WARNING] Không tìm được IK hợp lệ cho target pos:", target_pos)
        
    return {
        'solutions': sols,
        'best': best_sol,
        'method': used_method,
        'n_solutions': len(sols),
        'errors': errors
    }

if __name__ == "__main__":
    print("-" * 50)
    print("TESTING INVERSE KINEMATICS")
    print("-" * 50)
    
    def run_test(name, q_ref, expected_sols_min=1):
        print(f"\n{name}")
        fk_ref = forward_kinematics(q_ref)
        pos = fk_ref['position']
        euler = fk_ref['euler']
        
        ik_res = inverse_kinematics(pos, euler, q_current=q_ref)
        
        n_sols = ik_res['n_solutions']
        best = ik_res['best']
        
        print(f"Target Pos  : x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
        print(f"Target Euler: {euler[0]:.4f}, {euler[1]:.4f}, {euler[2]:.4f}")
        print(f"Số solutions: {n_sols} (Method: {ik_res['method']})")
        
        if n_sols >= expected_sols_min and best is not None:
            fk_check = forward_kinematics(best)
            err = np.linalg.norm(np.array(fk_check['position']) - np.array(pos))
            print(f"Best Error  : {err:.6f}m")
            for i, (sol, sol_err) in enumerate(zip(ik_res['solutions'], ik_res['errors'])):
                print(f"  Sol {i+1}: {np.array(sol).round(4)} -> Err: {sol_err:.6f}m")
            
            if err < 0.001:
                print("=> PASS")
                return True
            else:
                print("=> FAIL (Lỗi vượt ngưỡng 0.001m)")
                return False
        else:
            print("=> FAIL (Không đủ số lượng solution hoặc không tìm thấy)")
            return False

    pass_all = True
    
    # Test 1
    q_test1 = [0, -1.5708, 1.5708, -1.5708, -1.5708, 0]
    pass_all &= run_test("Test 1 — Round-trip từ home pose", q_test1)
    
    # Test 2
    q_test2 = [0.5, -1.2, 1.0, -1.5, -1.5, 0.3]
    pass_all &= run_test("Test 2 — Round-trip từ pose khác", q_test2)
    
    # Test 3
    print("\nTest 3 — Kiểm tra số solutions từ home pose")
    ik_res_3 = inverse_kinematics(forward_kinematics(q_test1)['position'], forward_kinematics(q_test1)['euler'])
    if ik_res_3['n_solutions'] >= 2:
        print(f"=> PASS ({ik_res_3['n_solutions']} solutions)")
    else:
        print(f"=> FAIL (Chỉ tìm thấy {ik_res_3['n_solutions']} < 2 solutions)")
        pass_all = False
        
    # Test 4
    print("\nTest 4 — Out of reach")
    ik_res_4 = inverse_kinematics((5.0, 5.0, 5.0), (0, 0, 0))
    if ik_res_4['best'] is None:
        print("=> PASS ('best' = None đúng mong đợi)")
    else:
        print(f"=> FAIL (Đã tìm ra nghiệm sai: {ik_res_4['best']})")
        pass_all = False
        
    print("-" * 50)
    print("KẾT QUẢ TỔNG THỂ:", "PASS" if pass_all else "FAIL")
