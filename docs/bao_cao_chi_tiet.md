# NỘI DUNG CHI TIẾT ĐỒ ÁN TỐT NGHIỆP
# Hệ thống Điều khiển & Mô phỏng Robot UR5e — Ứng dụng Học Tăng Cường cho bài toán Pick & Place

---

## CHƯƠNG 1: GIỚI THIỆU

### 1.1 Đặt vấn đề
Trong sản xuất công nghiệp hiện đại, robot cánh tay (manipulator) đóng vai trò then chốt trong các dây chuyền lắp ráp, đóng gói và phân loại sản phẩm. Thao tác Pick & Place — gắp vật thể từ vị trí này và đặt vào vị trí khác — là một trong những tác vụ cơ bản nhất nhưng cũng đặt ra nhiều thách thức:

- **Vật thể nằm ở vị trí ngẫu nhiên**, có thể đứng thẳng hoặc lăn ngang trên bề mặt.
- **Robot cần tính toán quỹ đạo** di chuyển tránh va chạm với các vật cản xung quanh.
- **Cánh tay 6 bậc tự do (6-DOF)** yêu cầu giải bài toán Động học Nghịch (Inverse Kinematics) phức tạp.

Phương pháp truyền thống sử dụng lập trình cứng (hard-coded trajectory) giải quyết ổn định nhưng thiếu tính linh hoạt khi môi trường thay đổi. Học Tăng Cường (Reinforcement Learning — RL) là hướng tiếp cận hiện đại cho phép robot tự học từ kinh nghiệm tương tác với môi trường, không cần lập trình trước quỹ đạo.

### 1.2 Mục tiêu đồ án
1. Xây dựng môi trường mô phỏng 3D robot UR5e trên PyBullet.
2. Triển khai bộ giải Động học Thuận (FK) và Nghịch (IK) theo quy ước Denavit-Hartenberg.
3. Thiết kế 3 chế độ điều khiển: Manual (tay), Auto (FSM + Trajectory Planning), AI (SAC RL).
4. Huấn luyện agent AI thực hiện Pick & Place bằng Curriculum Learning hai giai đoạn.
5. So sánh hiệu suất giữa chế độ Auto (deterministic) và AI (adaptive).

### 1.3 Phạm vi đồ án
- **Robot:** Universal Robots UR5e — 6 bậc tự do (DOF), tải trọng 5kg.
- **Môi trường:** Mô phỏng PyBullet 240Hz, URDF model chuẩn.
- **Vật thể:** Hình trụ (cylinder), bán kính 2cm, chiều cao 6cm, khối lượng 100g.
- **Gripper:** Giác hút chân không mô phỏng bằng Fixed Constraint.
- **Giao diện:** HMI PyQt5 với 4 tab: Manual, Trajectory, Auto, AI.

### 1.4 Công nghệ sử dụng
| Thành phần | Công nghệ | Phiên bản |
|---|---|---|
| Ngôn ngữ lập trình | Python | 3.10+ |
| Engine vật lý | PyBullet | 3.2+ |
| Giao diện người dùng | PyQt5 | 5.15+ |
| Thuật toán RL | Stable-Baselines3 (SAC) | 2.0+ |
| Deep Learning backend | PyTorch | 2.0+ |
| Giải phương trình phi tuyến | SciPy (L-BFGS-B) | 1.10+ |
| Trực quan hóa training | TensorBoard | 2.0+ |

---

## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

### 2.1 Động học Robot — Quy ước Denavit-Hartenberg (DH)

#### 2.1.1 Động học Thuận (Forward Kinematics — FK)
Động học Thuận là bài toán xác định vị trí và hướng của đầu công tác (End-Effector — EE) khi biết giá trị các góc khớp.

**Công thức:**
Ma trận biến đổi thuần nhất 4×4 cho mỗi khớp:

T_i = Rot_z(θ_i) × Trans_z(d_i) × Trans_x(a_i) × Rot_x(α_i)

Ma trận tổng hợp từ gốc đến EE:

T_EE = T_1 × T_2 × T_3 × T_4 × T_5 × T_6

**Bảng thông số DH của UR5e:**

| Khớp | a (m) | d (m) | α (rad) | Chức năng |
|------|-------|-------|---------|-----------|
| 1 | 0.0000 | 0.1625 | π/2 | Shoulder Pan |
| 2 | −0.4250 | 0.0000 | 0 | Shoulder Lift |
| 3 | −0.3922 | 0.0000 | 0 | Elbow |
| 4 | 0.0000 | 0.1333 | π/2 | Wrist 1 |
| 5 | 0.0000 | 0.0997 | −π/2 | Wrist 2 |
| 6 | 0.0000 | 0.0996 | 0 | Wrist 3 |

#### 2.1.2 Động học Nghịch (Inverse Kinematics — IK)
Bài toán ngược: tìm 6 góc khớp [q1...q6] khi biết vị trí (x,y,z) và hướng mong muốn của EE.

**Bộ giải Hybrid 2 lớp được triển khai:**
- **Lớp 1 — Analytical Solver:** Tính nghiệm giải tích kín từ hình học lượng giác. Tốc độ < 1ms. Trả về tối đa 8 nghiệm, chọn nghiệm gần HOME nhất. Ưu tiên vì nhanh và chính xác.
- **Lớp 2 — Numerical Solver (L-BFGS-B):** Khi Lớp 1 thất bại (singularity, out-of-workspace), thuật toán tối ưu hóa L-BFGS-B tiếp quản. Minimize đồng thời lỗi vị trí (position error) và lỗi hướng (orientation error). Tốc độ < 10ms.

**Tiêu chí chọn nghiệm:**
- Gần nhất với cấu hình hiện tại (tránh unwinding nhiều vòng).
- Nằm trong giới hạn khớp (±2π rad).
- Không gây singularity (det(Jacobian) > ε).

### 2.2 Quy hoạch Quỹ đạo (Trajectory Planning)

#### 2.2.1 Joint Space Trajectory
- Nội suy trực tiếp các góc khớp từ cấu hình bắt đầu đến cấu hình kết thúc.
- Velocity profile hình thang (Trapezoidal): giai đoạn tăng tốc → vận tốc đều → giảm tốc.
- Giới hạn vận tốc tối đa v_max = 1.0 rad/s và gia tốc a_max = 0.5 rad/s².
- Đảm bảo chuyển động mượt mà, không gây giật cơ (jerk).

#### 2.2.2 Cartesian Space Trajectory
- Chia đoạn thẳng từ điểm bắt đầu → kết thúc thành N điểm trung gian trong không gian Cartesian.
- Mỗi điểm trung gian được giải IK để tìm góc khớp tương ứng.
- Ghép các góc khớp thành Joint Trajectory liên tục.
- Ưu điểm: EE di chuyển theo đường thẳng chính xác (quan trọng khi tiếp cận vật theo trục Z).

### 2.3 Học Tăng Cường (Reinforcement Learning)

#### 2.3.1 Khái niệm cơ bản
Học Tăng Cường là phương pháp học máy trong đó Agent tương tác với Môi trường (Environment) theo vòng lặp:
1. Agent quan sát trạng thái s_t từ môi trường.
2. Agent chọn hành động a_t theo chính sách π(a|s).
3. Môi trường trả về phần thưởng r_t và trạng thái mới s_{t+1}.
4. Agent cập nhật chính sách để tối đa hóa tổng phần thưởng tương lai.

#### 2.3.2 Soft Actor-Critic (SAC)
SAC là thuật toán Off-Policy thuộc họ Actor-Critic, tối ưu hóa entropy-regularized objective:

π* = argmax_π E[Σ γ^t (r_t + α H(π(·|s_t)))]

Trong đó:
- γ: hệ số chiết khấu (discount factor) = 0.99
- α: hệ số entropy, tự điều chỉnh (auto-tuning)
- H(π): entropy của chính sách — khuyến khích đa dạng hành động

**Tại sao chọn SAC?**
- Không gian hành động liên tục → loại DQN (chỉ discrete).
- Sample-efficient hơn PPO 5-10 lần (ít data, hội tụ nhanh hơn).
- Entropy regularization → tự cân bằng exploit/explore, ổn định hơn DDPG.
- Đã được chứng minh hiệu quả trong robot manipulation (Haarnoja et al., 2018).

**Kiến trúc mạng:**
- Actor Network: MLP [512 → 512 → 256] → Action (7D)
- Critic 1: MLP [512 → 512 → 256] → Q-value
- Critic 2: MLP [512 → 512 → 256] → Q-value (twin critic chống overestimation)
- Entropy coef: tự điều chỉnh từ giá trị khởi tạo 0.1

---

## CHƯƠNG 3: THIẾT KẾ HỆ THỐNG

### 3.1 Kiến trúc tổng thể
Hệ thống chia thành 4 tầng:
1. **Tầng Giao diện (HMI):** PyQt5, chạy trên Main Thread. 4 tab chức năng.
2. **Tầng Điều khiển (SimBridge):** Thread riêng, xử lý lệnh từ HMI và điều phối 3 chế độ.
3. **Tầng Thuật toán:** FK/IK Solver, Trajectory Planner, FSM Controller, SAC Agent.
4. **Tầng Vật lý (PyBullet):** Mô phỏng 240Hz, collision detection, constraint-based gripper.

### 3.2 Chế độ Manual
- Người dùng điều khiển trực tiếp EE qua giao diện HMI.
- Jog Cartesian: dịch chuyển EE theo từng trục X, Y, Z với bước tùy chỉnh.
- Jog Joint: thay đổi trực tiếp từng góc khớp q1-q6.
- Workspace Validator kiểm tra giới hạn an toàn (X: 0.20-0.75m, Y: -0.30-0.30m, Z: 0.44-0.95m).

### 3.3 Chế độ Auto (FSM + Trajectory Planning)
Cỗ máy trạng thái hữu hạn (Finite State Machine) điều phối 10 trạng thái:

IDLE → DETECT → APPROACH → DESCEND → PICK → LIFT → MOVE_TO_BIN → PLACE → RELEASE → RETREAT → DONE

| Trạng thái | Hành động | Loại trajectory | Tốc độ |
|---|---|---|---|
| APPROACH | Di chuyển đến 15cm phía trên vật | Cartesian | 0.12 m/s |
| DESCEND | Hạ thẳng đứng xuống vị trí gắp | Cartesian | 0.05 m/s |
| PICK | Dwell 0.3s + kích hoạt giác hút | - | - |
| LIFT | Nâng vật lên 20cm | Cartesian | 0.08 m/s |
| MOVE_TO_BIN | Bay ngang đến phía trên bin | Cartesian | 0.15 m/s |
| PLACE | Hạ xuống vị trí thả | Cartesian | 0.05 m/s |
| RELEASE | Dwell 0.5s + tháo giác hút | - | - |
| RETREAT | Về HOME_POSE | Joint | 1.0 rad/s |

Timeout mỗi state: 15 giây. Nếu quá thời gian → ERROR.

### 3.4 Chế độ AI (SAC Reinforcement Learning)

#### 3.4.1 Không gian Quan sát (Observation — 17D)
| Index | Ý nghĩa | Chiều |
|---|---|---|
| 0-2 | Vị trí EE (x, y, z) | 3 |
| 3-5 | Vị trí vật thể (x, y, z) | 3 |
| 6-8 | Vector tương đối EE → Vật | 3 |
| 9-11 | Vector tương đối Vật → Bin | 3 |
| 12-15 | Quaternion hướng vật (qx, qy, qz, qw) | 4 |
| 16 | Trạng thái gripper (0 hoặc 1) | 1 |

Quaternion (4D) cho phép AI nhận biết vật đang đứng hay nằm ngang → xoay cổ tay đúng góc tiếp cận.

#### 3.4.2 Không gian Hành động (Action — 7D)
| Index | Ý nghĩa | Phạm vi | Đơn vị |
|---|---|---|---|
| 0-2 | Δx, Δy, Δz (dịch chuyển EE) | [-1, 1] × 5cm | cm/step |
| 3-5 | ΔRoll, ΔPitch, ΔYaw (xoay cổ tay) | [-1, 1] × 4.5° | °/step |
| 6 | Gripper (> 0 = gắp, ≤ 0 = nhả) | [-1, 1] | - |

#### 3.4.3 Thiết kế Hàm Reward — Phase-Based Architecture
Đây là phần quan trọng nhất và đã trải qua 3 lần thiết kế lại:

**Lần 1 — Sparse Reward:** Chỉ thưởng +500 khi rác vào bin. Kết quả: AI mò mẫm quá lâu, không hội tụ.

**Lần 2 — Dense Reward (Parabolic Shaping):** Phạt bay thấp, phạt xa bin. Kết quả: AI bị Reward Hacking — không bật gripper để tránh bị phạt.

**Lần 3 — Phase-Based Reward (phiên bản hiện tại):**
Chia rõ 3 giai đoạn:

PHA 0 — APPROACH & GRASP:
  - Thưởng tiến gần vật: +1.5 × (1 - dist × 6)
  - Phạt bay quá xa: -3.0 nếu dist > 0.5m
  - Thưởng gắp thành công: +50 → chuyển sang Pha 1

PHA 1 — CARRY:
  - Thưởng nâng cao: +30 khi nâng ≥ 20cm
  - Thưởng chiều cao: +4 × height
  - Phạt bay thấp (< 0.62m): -6.0 (tránh va chạm thành thùng)
  - Thưởng tiến về bin: +1.5 × (1 - dist_xy_bin × 4)
  - Khi gần bin (< 15cm XY) → chuyển Pha 2

PHA 2 — PLACE:
  - Thưởng hạ chính xác: +2.5 × (1 - dist_bin_3d × 6)
  - Thưởng thả thành công vào bin: +500 → KẾT THÚC

Phạt action norm nhẹ: -0.03 × ||action[:3]|| (chống giật cục, không ảnh hưởng gripper).

#### 3.4.4 Curriculum Learning (Học theo giáo trình)

**Phase 1 — Học Gắp (train_17d_grasp.py):**
- Mục tiêu: Chỉ học tiếp cận + kích hoạt giác hút. Không cần mang tới bin.
- Observation: 17D (17D ngay từ đầu để transition sang Phase 2 không bị lệch).
- Reward: phạt khoảng cách (-dist × 2.0) + thưởng nâng (height × 10.0) + bonus +200 khi gắp & nâng 15cm.
- Training: ~3 triệu steps, 4 envs, ~1 tiếng. Kết quả: 100% success rate.

**Phase 2 — Học Pick & Place (train_17d_place.py):**
- Load weights từ Phase 1 (Transfer Learning) → AI đã biết gắp, chỉ học thêm vận chuyển & thả.
- Phase-Based Reward (mô tả ở mục 3.4.3).
- Training: 10 triệu steps, 20 parallel envs (SubprocVecEnv), ~4-5 tiếng trên Core i7.
- Tốc độ: ~600 FPS (nhờ song song hóa 20 luồng CPU).

#### 3.4.5 Safety Layers (Lớp bảo vệ khi Inference)
- **Jam Detector:** Nếu EE đứng yên > 2 giây → tự động release gripper + quay về HOME.
- **Workspace Validator:** Clip tọa độ EE vào giới hạn an toàn.
- **Dwell Time (0.5s):** Dừng nhịp sau khi gắp, mô phỏng thời gian bơm áp suất chân không.
- **Retract Logic:** Sau khi thả xong → nâng tay lên → về HOME bằng Joint interpolation.

---

## CHƯƠNG 4: TRIỂN KHAI & KẾT QUẢ

### 4.1 Môi trường mô phỏng
- PyBullet 240 Hz, GUI với camera 3D có thể xoay.
- Bàn gỗ (0.42m cao), bin thùng rác (4 thành + 1 đáy), vật cylinder xanh dương.
- Robot UR5e load từ URDF chuẩn, 6 khớp quay với PD controller.
- Gripper mô phỏng bằng JOINT_FIXED constraint, khoảng cách gắp < 4.5cm.
- Workspace đánh dấu bằng đường viền đỏ trên mặt bàn.

### 4.2 Kết quả Training
**Phase 1 (Grasp):**
- Steps: 3,000,000 | Envs: 4 | Thời gian: ~60 phút
- Success rate: 100% | Mean reward: ~150

**Phase 2 (Pick & Place):**
- Steps: 10,000,000 | Envs: 20 | Thời gian: ~4.5 tiếng
- Success rate: 100% (trong training) | FPS: ~600

### 4.3 So sánh Auto vs AI

| Tiêu chí | Auto (FSM) | AI (SAC RL) |
|---|---|---|
| Tỉ lệ thành công | 100% | ~95% (trong vùng train) |
| Cần lập trình quỹ đạo | Có | Không |
| Thích nghi vật mới | Không | Có tiềm năng |
| Tốc độ cycle | Nhanh (tối ưu) | Phụ thuộc AI |
| Khi vật ngoài vùng train | Không ảnh hưởng | Giảm hiệu suất (OOD) |
| Mượt mà | Rất mượt | Đôi khi có jitter |

### 4.4 Hạn chế
1. AI chỉ hoạt động tốt trong vùng WORK_ZONE đã train (Out-of-Distribution failure).
2. Tọa độ vật lấy trực tiếp từ PyBullet (omniscient) — đời thực cần Camera 3D.
3. Chưa test Sim-to-Real trên robot UR5e thật.
4. Reward Hacking vẫn có thể xảy ra nếu thay đổi môi trường.

---

## CHƯƠNG 5: KẾT LUẬN & HƯỚNG PHÁT TRIỂN

### 5.1 Kết luận
Đồ án đã triển khai thành công:
- Hệ thống mô phỏng 3D robot UR5e hoàn chỉnh trên PyBullet.
- Bộ giải IK Hybrid (Analytical + Numerical) tự lập trình bằng NumPy/SciPy.
- 3 chế độ điều khiển trên cùng một nền tảng: Manual, Auto (FSM), AI (SAC).
- Agent RL đạt 100% success rate trong training thông qua Phase-Based Reward và Curriculum Learning.
- Giao diện HMI chuyên nghiệp với PyQt5 hỗ trợ vận hành và giám sát.

### 5.2 Hướng phát triển
1. Domain Randomization: mở rộng vùng spawn, random kích thước/hình dạng vật.
2. Camera Integration: thay omniscient bằng RealSense D435 + Point Cloud Segmentation.
3. Sim-to-Real Transfer: kết nối robot thật qua thư viện ur_rtde.
4. Multi-Object Sorting: gắp nhiều vật, phân loại theo màu sắc/hình dáng.
5. RLHF: tinh chỉnh reward bằng phản hồi từ người vận hành.
