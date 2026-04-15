# 🤖 Hệ thống Điều khiển & Mô phỏng Robot UR5e  
## Ứng dụng Học Tăng Cường (Reinforcement Learning) cho bài toán Pick & Place

> **Đồ án Tốt nghiệp** | Platform: PyBullet | Thuật toán RL: SAC (Soft Actor-Critic)  
> Ngôn ngữ: Python 3.10+ | Thư viện: Stable-Baselines3, PyQt5, NumPy, SciPy

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/RL-Stable_Baselines3-orange)
![Engine](https://img.shields.io/badge/Physics-PyBullet-red)
![Status](https://img.shields.io/badge/Status-In_Progress-yellow)

---

## 1. Bài Toán Giải Quyết

**Bài toán:** Điều khiển cánh tay robot công nghiệp UR5e (6-DOF) thực hiện thao tác **Pick & Place** — gắp vật thể từ vị trí ngẫu nhiên trên bàn và thả vào thùng chứa mục tiêu.

**Điểm khó của bài toán:**
- Vật thể sinh ra **ngẫu nhiên** về cả vị trí (XY) lẫn hướng nằm (đứng thẳng hoặc lăn ngang).
- Robot phải **xoay cổ tay (6D)** để tiếp cận đúng góc gắp.
- Quỹ đạo di chuyển phải **tránh va chạm** với thành thùng và bề mặt bàn.
- Hệ thống phải hoạt động **real-time** và phục hồi được khi xảy ra lỗi singularity.

**Hướng tiếp cận được so sánh:**

| Hướng tiếp cận | Phương pháp | Ưu điểm | Nhược điểm |
|---|---|---|---|
| **Cổ điển** | FSM + IK + Trajectory Planning | Ổn định, tiên đoán được | Phải lập trình cứng từng tình huống |
| **Học tăng cường** | SAC Deep RL | Tự thích nghi, không cần lập trình quỹ đạo | Cần thời gian train, reward design phức tạp |

---

## 2. Kiến Trúc Tổng Thể

```
┌─────────────────────────────────────────────────────────┐
│                    HMI (PyQt5)                          │
│  [Manual] [Trajectory] [Auto/FSM] [AI/RL]               │
└──────────────────────┬──────────────────────────────────┘
                       │ Thread-safe Queue
┌──────────────────────▼──────────────────────────────────┐
│               SimBridge (Control Thread)                 │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Manual   │  │  FSM + IK    │  │   SAC RL Agent   │  │
│  │ Jog XYZ  │  │  Trajectory  │  │   (best_model)   │  │
│  └──────────┘  └──────────────┘  └──────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              PyBullet Physics Engine (240 Hz)            │
│    [UR5e URDF] [Object] [Bin] [Vacuum Gripper]          │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Động Học Robot UR5e

### 3.1 Động Học Thuận (Forward Kinematics — FK)

**Mục đích:** Tính vị trí & hướng của đầu gắp (EE) khi biết 6 góc khớp.

**Phương pháp:** Sử dụng quy ước **Denavit-Hartenberg (DH)** — nhân chuỗi 6 ma trận biến đổi thuần nhất 4×4:

$$T_i^{i-1} = R_z(\theta_i) \cdot T_z(d_i) \cdot T_x(a_i) \cdot R_x(\alpha_i)$$

$$T_{EE}^{base} = T_1^0 \cdot T_2^1 \cdot T_3^2 \cdot T_4^3 \cdot T_5^4 \cdot T_6^5$$

**Bảng thông số DH của UR5e:**

| Khớp | `a` (m) | `d` (m) | `α` (rad) | Chức năng |
|------|---------|---------|-----------|-----------|
| 1 | 0.0000 | 0.1625 | π/2 | Shoulder Pan |
| 2 | −0.4250 | 0.0000 | 0 | Shoulder Lift |
| 3 | −0.3922 | 0.0000 | 0 | Elbow |
| 4 | 0.0000 | 0.1333 | π/2 | Wrist 1 |
| 5 | 0.0000 | 0.0997 | −π/2 | Wrist 2 |
| 6 | 0.0000 | 0.0996 | 0 | Wrist 3 |

**File thực thi:** `kinematics/forward_kinematics.py`

---

### 3.2 Động Học Nghịch (Inverse Kinematics — IK)

**Mục đích:** Tính 6 góc khớp cần thiết khi biết tọa độ XYZ + hướng mong muốn của EE.

**Bộ giải IK Hybrid 2 lớp:**

```
Đầu vào: Tọa độ EE (xyz) + hướng (euler/quaternion)
         │
         ▼
┌─────────────────────────────┐
│  Lớp 1: Analytical Solver   │  → Tính giải tích dựa trên hình học + lượng giác
│  (Tốc độ: < 1ms)            │  → Trả về tối đa 8 nghiệm, chọn nghiệm gần HOME nhất
└────────────┬────────────────┘
             │ Nếu thất bại (Singularity / Out-of-workspace)
             ▼
┌─────────────────────────────┐
│  Lớp 2: Numerical Solver    │  → Tối ưu hóa L-BFGS-B (SciPy)
│  (Tốc độ: < 10ms)           │  → Minimize lỗi vị trí + lỗi hướng
└─────────────────────────────┘
         │
         ▼
Đầu ra: [q1, q2, q3, q4, q5, q6] (radian)
```

**Tiêu chí chọn nghiệm tối ưu:**
- Gần nhất với cấu hình Home Pose hiện tại (tránh unwinding).
- Không vi phạm giới hạn khớp ±2π.
- Không gây singularity (det(Jacobian) > ε).

**File thực thi:** `kinematics/inverse_kinematics.py`

---

### 3.3 Quy Hoạch Quỹ Đạo (Trajectory Planning)

**2 loại quỹ đạo được triển khai:**

**a) Joint Space Trajectory** — Nội suy trực tiếp các góc khớp:
- Dùng velocity profile hình thang (Trapezoidal): tăng tốc → đều → giảm tốc.
- Đảm bảo gia tốc và vận tốc không vượt giới hạn cơ học của UR5e.

**b) Cartesian Space Trajectory** — Nội suy tọa độ EE trong không gian:
- Chia đoạn thẳng P_start → P_end thành N điểm trung gian.
- Mỗi điểm trung gian gọi IK để tính góc khớp → ghép thành Joint Trajectory.
- Hiệu quả để thực hiện đường thẳng chính xác (tiếp cận vật theo trục Z).

**File thực thi:** `kinematics/trajectory.py`

---

## 4. Chế Độ Auto (FSM — Finite State Machine)

### 4.1 Sơ Đồ Trạng Thái

```
IDLE ──► DETECT ──► APPROACH ──► DESCEND ──► PICK
                                               │
DONE ◄── RETREAT ◄── RELEASE ◄── PLACE ◄── LIFT ◄── MOVE_TO_BIN
```

### 4.2 Mô Tả Từng Trạng Thái

| Trạng thái | Hành động | Điều kiện chuyển |
|---|---|---|
| **DETECT** | Lấy tọa độ vật từ PyBullet, tính pick/place poses | Tọa độ hợp lệ → APPROACH |
| **APPROACH** | Chạy Cartesian traj đến điểm 15cm trên đầu vật | Executor hoàn thành → DESCEND |
| **DESCEND** | Hạ thẳng đứng xuống vị trí gắp (Z = obj_z + 1cm) | Executor hoàn thành → PICK |
| **PICK** | Dwell 0.3s → Kích hoạt vacuum constraint | Tức thì → LIFT |
| **LIFT** | Nâng vật lên 20cm | Executor hoàn thành → MOVE_TO_BIN |
| **MOVE_TO_BIN** | Bay ngang tới điểm 30cm trên đầu bin | Executor hoàn thành → PLACE |
| **PLACE** | Hạ xuống vị trí thả | Executor hoàn thành → RELEASE |
| **RELEASE** | Dwell 0.5s → Tháo vacuum constraint | Tức thì → RETREAT |
| **RETREAT** | Về HOME_POSE bằng Joint Trajectory | Executor hoàn thành → DONE |
| **ERROR** | Giữ nguyên, chờ user reset | Nút Reset Scene |

**Cơ chế Timeout:** Mỗi state có giới hạn 15 giây — nếu executor không xong → chuyển ERROR tự động.

**File thực thi:** `simulation/pick_place_sm.py`

---

## 5. Chế Độ AI — Deep Reinforcement Learning (SAC)

### 5.1 Thuật Toán: Soft Actor-Critic (SAC)

**Lý do chọn SAC:**
- Không gian hành động **liên tục** (continuous) → loại DQN ngay.
- **Sample-efficient** hơn PPO (cần ít data hơn 5-10 lần để hội tụ).
- **Entropy regularization** tự động cân bằng exploit/explore → ổn định hơn DDPG.
- Đã được chứng minh trong nhiều bài toán robot manipulation.

**Kiến trúc mạng:**

```
Observation (17D) ──► Actor MLP [512 → 512 → 256] ──► Action (7D) liên tục [-1, 1]
                  ──► Critic 1 MLP [512 → 512 → 256] ──► Q-value
                  ──► Critic 2 MLP [512 → 512 → 256] ──► Q-value (chống overestimate)
```

**Hyperparameters:**

| Tham số | Giá trị | Lý do |
|---|---|---|
| Learning rate | 3×10⁻⁴ | Chuẩn Adam cho MLP |
| Gamma (discount) | 0.99 | Task dài, cần nhìn xa |
| Batch size | 512 | GPU throughput tốt |
| Buffer size | 500,000 | Off-policy, tái dùng data |
| Gradient steps | 2/step | Tăng sample efficiency |
| Entropy coef | auto (init 0.1) | Tự điều chỉnh entropy |

---

### 5.2 Không Gian Quan Sát (Observation Space — 17D)

| Index | Ý nghĩa | Chiều |
|---|---|---|
| 0-2 | Vị trí EE (x, y, z) | 3D |
| 3-5 | Vị trí vật thể (x, y, z) | 3D |
| 6-8 | Vector EE → Vật (rel_obj) | 3D |
| 9-11 | Vector Vật → Bin (rel_bin) | 3D |
| 12-15 | Quaternion hướng vật (qx, qy, qz, qw) | 4D |
| 16 | Trạng thái gripper (0/1) | 1D |

*Quaternion ở index 12-15 là điểm khác biệt so với các thiết kế 13D thông thường — cho phép AI biết vật đang đứng hay nằm ngang để xoay cổ tay đúng góc.*

---

### 5.3 Không Gian Hành Động (Action Space — 7D)

| Index | Ý nghĩa | Đơn vị |
|---|---|---|
| 0-2 | Δx, Δy, Δz (dịch chuyển EE) | ±5 cm/step |
| 3-5 | ΔRoll, ΔPitch, ΔYaw (xoay cổ tay) | ±4.5°/step |
| 6 | Gripper (> 0 = gắp, ≤ 0 = nhả) | [-1, 1] |

---

### 5.4 Thiết Kế Hàm Reward — Phase-Based Architecture

**Vấn đề của reward thông thường:** AI bị *Reward Hacking* — tìm cách gian lận để ăn điểm mà không làm đúng nhiệm vụ. Ví dụ: đứng yên không gắp để tránh bị trừ điểm action_norm.

**Giải pháp Phase-Based Reward:** Chia rõ 3 giai đoạn, mỗi giai đoạn có reward riêng:

```
Pha 0 — APPROACH & GRASP
├── +1.5 × (1 - dist_ee) : Thưởng tỉ lệ khi tiến gần vật
├── −3.0 nếu dist_ee > 0.5m : Phạt bay quá xa
└── +50.0 khi gắp thành công → chuyển Pha 1

Pha 1 — CARRY (Vận chuyển)
├── +30.0 khi nâng vật ≥ 20cm : Thưởng nâng thành công
├── +4 × height : Thưởng càng nâng cao
├── −6.0 nếu EE < 0.62m trong khi vận chuyển : Phạt bay thấp
├── +1.5 × (1 - dist_xy_bin) : Thưởng tiến về bin
└── dist_xy_bin < 15cm → chuyển Pha 2

Pha 2 — PLACE (Thả)
├── +2.5 × (1 - dist_bin_3d) : Thưởng hạ xuống đúng vị trí
└── +500.0 khi vật rơi vào bin → DONE ✓
```

**Kết quả:** AI học được hành vi: Gắp → Nâng cao → Lướt ngang → Hạ xuống thả — tự nhiên tạo ra quỹ đạo parabol mà không cần lập trình cứng.

---

### 5.5 Curriculum Learning (Học theo giáo trình)

**Phase 1 — Học Gắp** (`train_17d_grasp.py`):
- Chỉ học tiếp cận & kích hoạt gripper (không cần mang ra bin).
- Spawn vật trong bán kính 25cm từ HOME.
- Reward: phạt khoảng cách + bonus +200 khi gắp & nâng 15cm.
- Kết quả: ~3 triệu steps, 100% success rate.

**Phase 2 — Học Pick & Place** (`train_17d_place.py`):
- Load weights từ Phase 1, train thêm phần vận chuyển & thả.
- Phase-Based reward để tránh reward hacking.
- 10 triệu steps, 20 parallel environments, ~4-5 tiếng trên Core i7.

---

### 5.6 Inference — Kết Hợp AI + Safety Layer

```python
# sim_bridge.py — vòng lặp AI ở 30 Hz
obs = build_observation_17D()        # Lấy state từ PyBullet
action = sac_model.predict(obs)      # AI ra quyết định
apply_cartesian_delta(action[:6])    # Di chuyển EE
control_gripper(action[6])           # Bật/tắt giác hút

# Safety layers:
# - Jam Detector: nếu EE đứng yên > 2s → Auto-Home
# - Workspace Validator: clip tọa độ vào giới hạn an toàn  
# - Dwell Time: dừng 0.5s sau khi gắp (mô phỏng áp suất khí nén)
```

---

## 6. Kết Cấu Thư Mục

```
do_an_robot_v2/
├── hmi/                        # Giao diện HMI (PyQt5)
│   ├── app.py                  # Điểm khởi động
│   ├── main_window.py          # Layout chính
│   ├── sim_bridge.py           # Bridge thread GUI ↔ PyBullet
│   └── widgets/                # Các panel con
├── kinematics/                 # Động học
│   ├── forward_kinematics.py   # FK (DH convention)
│   ├── inverse_kinematics.py   # IK (Analytical + L-BFGS-B)
│   ├── trajectory.py           # Trapezoid + Cartesian trajectory
│   └── workspace_validator.py  # Giới hạn workspace
├── simulation/                 # Vật lý mô phỏng
│   ├── environment.py          # PyBullet world setup
│   ├── gripper.py              # Vacuum gripper (constraint-based)
│   ├── pick_place_sm.py        # FSM Auto mode
│   └── object_detector.py      # Object pose + raycast
├── models_rl_17d/              # Weights AI Phase 2
│   └── seed42/best_model.zip
├── models_rl_17d_grasp/        # Weights AI Phase 1
│   └── seed42/best_model.zip
├── train_17d_grasp.py          # Train Phase 1
├── train_17d_place.py          # Train Phase 2 (Phase-Based Reward)
└── README.md
```

---

## 7. Hướng Dẫn Chạy

```bash
# Cài thư viện
pip install pybullet numpy scipy PyQt5 stable-baselines3 torch gymnasium

# Mở HMI
python -m hmi.app

# Train Phase 1 (nếu cần train lại từ đầu)
python train_17d_grasp.py

# Train Phase 2
python train_17d_place.py

# Theo dõi quá trình train
tensorboard --logdir logs_rl_17d/
```

---

## 8. Giới Hạn & Hướng Phát Triển

| Giới hạn hiện tại | Giải pháp tương lai |
|---|---|
| AI chỉ hoạt động tốt trong vùng train (WORK_ZONE) | Domain Randomization mở rộng vùng spawn |
| Tọa độ vật lấy trực tiếp từ PyBullet (Omniscient) | Tích hợp Camera RealSense + Point Cloud |
| Mô phỏng, chưa giao tiếp robot thật | Thêm `ur_rtde` để kết nối UR5e thật |
| Reward Hacking vẫn có thể xảy ra | RLHF (Reinforcement Learning from Human Feedback) |

---

*Đồ án Tốt nghiệp — Khoa Cơ Điện Tử / Tự Động Hóa*
