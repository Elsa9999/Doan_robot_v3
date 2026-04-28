# 🔬 GIẢI THÍCH CHI TIẾT MÃ NGUỒN — ĐỒ ÁN UR5e PICK & PLACE

> **Tài liệu dành cho:** Người mới đọc code lần đầu, thành viên nhóm muốn hiểu tổng thể, hoặc Hội đồng muốn kiểm tra logic.
>
> **Cấu trúc:** Đọc từ trên xuống dưới = Đi từ nền tảng Toán học → Vật lý → Điều khiển → Giao diện.

---

## 📁 Sơ đồ thư mục dự án

```
do_an_robot_v2/
├── kinematics/                 # 🧮 LÕI TOÁN HỌC (Động học Thuận, Nghịch, Quỹ đạo)
│   ├── forward_kinematics.py   #    Tính tọa độ XYZ từ 6 góc khớp
│   ├── inverse_kinematics.py   #    Tính 6 góc khớp từ tọa độ XYZ
│   ├── trajectory.py           #    Nội suy quỹ đạo hình thang (Trapezoid Profile)
│   └── workspace_validator.py  #    Cảnh sát vùng cấm (ngăn robot đập tay vào bàn)
│
├── simulation/                 # 🎮 THẾ GIỚI VẬT LÝ (PyBullet)
│   ├── environment.py          #    Dựng bàn, robot, thùng rác, cục mút
│   ├── gripper.py              #    Giác hút chân không (Vacuum Gripper)
│   ├── pick_place_sm.py        #    Máy trạng thái FSM (chế độ Auto)
│   ├── object_detector.py      #    Camera ảo quét tọa độ vật thể
│   ├── trajectory_executor.py  #    Bơm từng điểm quỹ đạo xuống PyBullet
│   └── manual_controller.py    #    Logic điều khiển bằng tay (Jog)
│
├── hmi/                        # 🖥️ GIAO DIỆN NGƯỜI DÙNG (PyQt5)
│   ├── app.py                  #    Điểm khởi chạy chương trình
│   ├── main_window.py          #    Cửa sổ chính (ghép các panel lại)
│   ├── sim_bridge.py           #    Cầu nối giữa GUI và PyBullet (chạy thread riêng)
│   └── widgets/                #    Các bảng điều khiển con
│       ├── joint_panel.py      #       Tab điều khiển từng khớp
│       ├── cartesian_panel.py  #       Tab điều khiển tọa độ XYZ
│       ├── trajectory_panel.py #       Tab quy hoạch quỹ đạo
│       ├── auto_panel.py       #       Tab chế độ Auto (FSM)
│       ├── ai_panel.py         #       Tab chế độ AI (SAC RL)
│       ├── status_panel.py     #       Hiển thị trạng thái robot
│       └── log_panel.py        #       Console log
│
├── train_17d_grasp.py          # 🧠 Huấn luyện AI giai đoạn 1 (Học gắp)
├── train_17d_place.py          #    Huấn luyện AI giai đoạn 2 (Học gắp + thả)
├── urdf/                       #    Bản vẽ 3D robot (file XML mô tả khớp, link)
└── models_rl_17d/              #    Bộ não AI đã huấn luyện xong (file .zip)
```

---

## 1. KHỐI TOÁN HỌC — `kinematics/`

Đây là phần quan trọng nhất, là "bộ não toán học" giúp robot biết cách di chuyển.

### 1.1 Động học Thuận — `forward_kinematics.py`

**Bài toán:** Cho 6 góc quay của 6 mô-tơ → Tính ra tọa độ (X, Y, Z) của mũi kẹp gắp đang ở đâu trong không gian.

**Hàm quan trọng:**

#### `dh_transform(a, d, alpha, theta)`
Tính ma trận biến đổi 4×4 từ 4 thông số DH (Denavit-Hartenberg).
```python
# Ma trận 4×4 kết quả có dạng:
# ┌                                       ┐
# │ cos(θ)  -sin(θ)cos(α)  sin(θ)sin(α)  a·cos(θ) │  ← Hàng 1: Trục X mới
# │ sin(θ)   cos(θ)cos(α) -cos(θ)sin(α)  a·sin(θ) │  ← Hàng 2: Trục Y mới
# │ 0        sin(α)        cos(α)         d        │  ← Hàng 3: Trục Z mới
# │ 0        0             0              1        │  ← Hàng 4: Hệ số tỷ lệ
# └                                       ┘
```
Trong đó:
- `a` = chiều dài cánh tay (link length) — đo dọc trục X
- `d` = độ dời dọc trục Z (link offset) — chiều cao khớp
- `α` = góc xoắn giữa 2 trục Z liên tiếp
- `θ` = góc quay thực tế của mô-tơ (biến số)

#### `parse_dh_from_urdf()`
Đọc file URDF (bản vẽ 3D) của hãng Universal Robots, tự động trích xuất ra bảng DH 6 hàng:
```
Khớp │ a (m)    │ d (m)   │ α (rad)
─────┼──────────┼─────────┼────────
  1  │  0.0000  │ 0.1625  │  π/2     ← Vai xoay (Base → Shoulder)
  2  │ -0.4250  │ 0.0000  │  0       ← Bắp tay (Shoulder → Elbow)
  3  │ -0.3922  │ 0.0000  │  0       ← Cẳng tay (Elbow → Wrist1)
  4  │  0.0000  │ 0.1333  │  π/2     ← Cổ tay 1
  5  │  0.0000  │ 0.0997  │ -π/2     ← Cổ tay 2
  6  │  0.0000  │ 0.0996  │  0       ← Cổ tay 3 (End-Effector)
```
> **Lưu ý:** Số liệu này KHÔNG phải gõ tay. Nó được đọc tự động từ file `urdf/ur5e_final.urdf` do hãng cung cấp, đảm bảo khớp 100% với robot thực.

#### `forward_kinematics(q)`
Hàm chính. Nhân dồn 6 ma trận 4×4 lại với nhau:
```
T_kết_quả = T₁ × T₂ × T₃ × T₄ × T₅ × T₆
```
Cột cuối cùng của ma trận kết quả chính là tọa độ `(X, Y, Z)` của mũi kẹp.

---

### 1.2 Động học Nghịch — `inverse_kinematics.py`

**Bài toán:** Ngược lại với FK. Cho tọa độ (X, Y, Z) mong muốn → Tính ra 6 góc quay cần bẻ.

**Hệ thống Hybrid 2 lớp:**

#### Lớp 1: `analytical_ik(T_target)` — Giải tích hình học
Dùng lượng giác thuần túy (sin, cos, atan2) để giải ra **tối đa 8 cấu hình nghiệm** trong < 1 mili-giây.

**Quy trình 5 bước:**
1. **Tìm Tâm Cổ Tay (Wrist Center):** Lấy tọa độ mũi kẹp, lùi lại một đoạn `d₆ = 0.0996m` dọc theo hướng trục Z của kẹp. Đây là kỹ thuật **Kinematic Decoupling** — tách bài toán 6 biến thành 2 bài toán 3 biến.
2. **Giải θ₁ (Vai):** Nhìn từ trên xuống mặt phẳng XY, dùng `atan2` → Ra 2 nghiệm (Vai trái / Vai phải).
3. **Giải θ₅ (Cổ tay):** Từ θ₁ suy ra → Ra 2 nghiệm (Cổ tay sấp / ngửa).
4. **Giải θ₃ (Khuỷu tay):** Áp dụng **Định lý Hàm số Cosin** cho tam giác bắp tay + cẳng tay → Ra 2 nghiệm (Elbow Up / Down).
5. **Giải θ₂, θ₄, θ₆:** Suy ra từ các góc đã tìm ở trên.

> **Tổng cộng:** 2 × 2 × 2 = **8 cấu hình nghiệm** (Left/Right × Up/Down × Flip/NoFlip).

#### Lớp 2: `numerical_ik(T_target)` — Tối ưu hóa số học
Chỉ chạy khi Lớp 1 thất bại (ví dụ: robot bị kéo căng gần giới hạn). Dùng thuật toán **L-BFGS-B** (quasi-Newton) để tìm nghiệm xấp xỉ tốt nhất.

#### `inverse_kinematics(target_pos, target_euler)` — Hàm tổng
Tự động thử Lớp 1 trước, nếu không ra nghiệm → Chuyển sang Lớp 2. Sau đó chọn nghiệm gần nhất với tư thế hiện tại của robot (tránh giật cục).

**Kiểm chứng (ở cuối file):** Hệ thống tự động chạy **Round-Trip Verification** — Nạp góc vào FK, lấy tọa độ ra, nhét ngược vào IK, rồi so sánh sai lệch. Nếu sai số < 1mm → `PASS`.

---

### 1.3 Quy hoạch Quỹ đạo — `trajectory.py`

**Bài toán:** Robot cần đi từ điểm A sang điểm B, nhưng KHÔNG ĐƯỢC giật cục (gia tốc vô hạn sẽ phá hỏng mô-tơ). Cần phải tăng tốc → giữ tốc → giảm tốc mượt mà.

#### `trapezoid_profile(distance, v_max, a_max)`
Tạo biểu đồ vận tốc hình thang:
```
vận tốc
  │     ___________
  │    /           \          ← v_max (tốc độ tối đa)
  │   /             \
  │  /               \
  │ /                 \
  └──────────────────── thời gian
  Tăng tốc   Đều ga   Giảm tốc
```
Nếu quãng đường quá ngắn (không kịp đạt v_max), tự động chuyển sang **biên dạng tam giác** (chỉ có tăng + giảm, không có đều ga).

#### `JointTrajectory` — Nội suy trong không gian khớp
Nội suy trực tiếp 6 góc quay từ giá trị bắt đầu đến giá trị kết thúc. Robot sẽ đi theo đường cong trong không gian khớp (có thể không phải đường thẳng trong không gian XYZ).

#### `CartesianTrajectory` — Nội suy trong không gian tọa độ
Nội suy tọa độ XYZ theo đường thẳng, sau đó tại mỗi điểm nhỏ li ti trên đường thẳng, gọi hàm **IK** để chuyển đổi ngược ra 6 góc khớp. Kết quả: robot đi thẳng tắp trong không gian (như máy CNC).

> **Hàm `to_joint_trajectory()`**: Đây là nơi kết nối Quỹ đạo Cartesian với IK. Nó chia đường thẳng ra N điểm, tại mỗi điểm gọi `inverse_kinematics()` để chuyển XYZ → 6 góc.

---

## 2. KHỐI VẬT LÝ — `simulation/`

### 2.1 Thế giới ảo — `environment.py`

File lớn nhất, tạo ra toàn bộ cảnh vật lý 3D:

**Các hằng số quan trọng:**
```python
TABLE_SURFACE = 0.42        # Chiều cao mặt bàn (mét)
HOME_POSE = [0, -π/2, π/2, -π/2, -π/2, 0]  # Tư thế nghỉ 6 khớp
CART_DELTA_MAX = 0.05       # AI di chuyển tối đa 5cm/bước
```

**Các hàm chính:**
- `_load_robot()`: Nạp file URDF vào PyBullet, xoay 6 khớp về tư thế Home.
- `_spawn_object()`: Tạo cục mút hình trụ (cylinder) rải ngẫu nhiên trên bàn. Hỗ trợ **Curriculum Difficulty** (dễ → khó).
- `set_joint_positions(q)`: Điều khiển 6 mô-tơ bằng **PD Controller** (Position Control) — có va chạm vật lý, mượt.
- `teleport_joints(q)`: Dịch chuyển tức thời (chỉ dùng lúc khởi tạo, không có vật lý).
- `move_ee_cartesian(delta_xyz)`: Di chuyển đầu kẹp theo tọa độ XYZ tương đối. Bên trong gọi IK của PyBullet.
- `activate_gripper()`: Tạo ràng buộc vật lý (`JOINT_FIXED`) giữa đầu hút và vật thể khi khoảng cách < 4.5cm.
- `is_in_bin()`: Kiểm tra vật thể đã nằm trong thùng rác chưa.

> **Euler Clamp (Dòng 329-342):** Kẹp cứng góc Roll/Pitch của kẹp gắp trong phạm vi ±15° quanh tư thế thẳng đứng. Đây là bí quyết giúp AI luôn giữ tư thế gắp chuẩn công nghiệp mà không cần huấn luyện thêm.

### 2.2 Giác hút chân không — `gripper.py`

Mô phỏng giác hút bằng cách tạo **Constraint vật lý** (ràng buộc cứng) giữa mũi robot và vật thể. Khi "hút", 2 vật thể bị khóa dính lại. Khi "nhả", xóa ràng buộc → vật rơi tự do theo trọng lực.

### 2.3 Máy trạng thái Auto — `pick_place_sm.py`

Cỗ máy **Finite State Machine (FSM)** điều khiển chế độ Auto với 11 trạng thái:

```
IDLE → DETECT → APPROACH → DESCEND → PICK → LIFT → MOVE_TO_BIN → PLACE → RELEASE → RETREAT → DONE
```

Mỗi trạng thái tạo ra 1 quỹ đạo (Trajectory), khi robot chạy xong quỹ đạo → tự động chuyển sang trạng thái kế tiếp. Nếu bất kỳ bước nào bị treo quá 15 giây → Chuyển sang `ERROR` và khóa hệ thống.

### 2.4 Camera ảo — `object_detector.py`

Dùng **Raycast** (bắn tia laser ảo) từ trên trời xuống mặt bàn để dò tìm vật thể. Khi tia laser chạm vào vật → trả về tọa độ (X, Y, Z) chính xác của vật.

### 2.5 Bộ thực thi quỹ đạo — `trajectory_executor.py`

Nhận vào một đối tượng `JointTrajectory`, mỗi bước vật lý (1/240 giây) lấy ra 1 điểm trên quỹ đạo, bơm xuống `set_joint_positions()`. Giống như kim đĩa than lướt trên rãnh nhạc — đọc từng nốt nhạc và phát ra âm thanh liên tục.

---

## 3. KHỐI GIAO DIỆN — `hmi/`

### 3.1 Cầu nối — `sim_bridge.py`

**File quan trọng nhất** của hệ thống HMI. Chạy trên 1 **Thread riêng biệt** (QThread) ở tần số 240Hz, tách biệt hoàn toàn khỏi giao diện PyQt5 (để không bị lag khi click chuột).

**Luồng hoạt động:**
```
[Người dùng bấm nút]
       │
       ▼
[command_queue]  ──►  SimBridge Thread (240Hz)  ──►  [state_queue]
                          │                              │
                     PyBullet Engine                 [Giao diện đọc]
                     FK / IK / Traj                  cập nhật số liệu
```

**3 chế độ hoạt động:**
1. **Manual:** Người dùng bấm nút Jog hoặc nhập tọa độ XYZ → SimBridge gọi IK → Đẩy góc khớp xuống PyBullet.
2. **Auto (FSM):** SimBridge gọi `PickPlaceStateMachine.update()` mỗi vòng lặp → FSM tự quản lý quy trình gắp/thả.
3. **AI (SAC):** SimBridge đọc trạng thái (vị trí EE, vật, thùng) → Nạp vào mô hình Neural Network → Nhận lệnh di chuyển → Thực thi trên PyBullet.

**Tính năng đặc biệt của chế độ AI:**
- **Observation Normalization:** Chuẩn hóa input bằng Running Mean/Std từ VecNormalize (dòng 482-484).
- **Hybrid Gripper:** AI không cần học "khi nào bật giác hút". Hệ thống tự gắp khi đủ gần, tự nhả khi vào thùng (dòng 505-510).
- **Jam Detector:** Nếu robot bị kẹt > 3.3 giây → Tự động kích hoạt Auto-Home để gỡ rối (dòng 437-443).

### 3.2 Cửa sổ chính — `main_window.py`

Ghép 7 widget/panel vào 1 cửa sổ PyQt5. Mỗi panel là 1 tab riêng (Joint, Cartesian, Trajectory, Auto, AI...).

### 3.3 Các Panel điều khiển — `hmi/widgets/`

| File | Chức năng |
|------|-----------|
| `joint_panel.py` | 6 thanh trượt điều khiển từng góc khớp |
| `cartesian_panel.py` | Nhập tọa độ XYZ + góc Roll/Pitch/Yaw → Bấm "Go To Pose" |
| `trajectory_panel.py` | Tạo danh sách waypoints, chạy quỹ đạo nội suy |
| `auto_panel.py` | Bấm Start/Stop chế độ Auto FSM, hiển thị tiến trình 11 bước |
| `ai_panel.py` | Bấm Start/Stop chế độ AI, đếm số lần gắp thành công |
| `status_panel.py` | Hiển thị real-time: Tọa độ XYZ, góc khớp, trạng thái gripper |
| `log_panel.py` | Console log hiển thị mọi sự kiện (IK thành công, lỗi, AI reward) |

---

## 4. KHỐI TRÍ TUỆ NHÂN TẠO — `train_*.py`

### 4.1 Giai đoạn 1: Học Gắp — `train_17d_grasp.py`
- **Thuật toán:** SAC (Soft Actor-Critic)
- **Input (17D):** Vị trí EE(3) + Vật(3) + Vector tương đối(3) + Thùng(3) + Quaternion vật(4) + Gripper(1)
- **Output (7D):** Δx, Δy, Δz, ΔRoll, ΔPitch, ΔYaw, Gripper ON/OFF
- **Kết quả:** ~3 triệu bước, 1 tiếng, tỷ lệ gắp 100%

### 4.2 Giai đoạn 2: Học Gắp + Thả — `train_17d_place.py`
- **Kỹ thuật Transfer Learning:** Lấy bộ não giai đoạn 1, "phẫu thuật tensor" (Tensor Surgery) mở rộng input thêm 3 chiều `rel_bin` → Tiếp tục huấn luyện.
- **Kết quả:** ~5.5 triệu bước, 2.5 tiếng, tỷ lệ Pick & Place 100%

### 4.3 Euler Clamp — Bí quyết giữ tư thế
Thay vì để AI tự học cách giữ kẹp gắp thẳng đứng (rất khó, tốn hàng triệu bước), hệ thống **kẹp cứng góc nghiêng ở tầng Vật lý** (±15°). AI chỉ cần lo bay XYZ, còn tư thế luôn được đảm bảo tự động.

---

## 5. CÁCH CHẠY THỬ

### Chạy giao diện HMI (đầy đủ):
```bash
python -m hmi.app
```

### Kiểm chứng Động học Thuận:
```bash
python kinematics/forward_kinematics.py
```

### Kiểm chứng Động học Nghịch (Round-Trip Verification):
```bash
python kinematics/inverse_kinematics.py
```

### Huấn luyện AI (nếu muốn train lại):
```bash
python train_17d_grasp.py    # Giai đoạn 1
python train_17d_place.py    # Giai đoạn 2
```

---

## 6. TÀI LIỆU THAM KHẢO

1. **Hawkins, K. P. (2013).** *"Analytic Inverse Kinematics for the Universal Robots UR-5/UR-10 Arms"*. Georgia Institute of Technology.
2. **Andersen, R. S. (2018).** *"Kinematics of a UR5"*. Aalborg University.
3. **Haarnoja, T., et al. (2018).** *"Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"*. ICML 2018.
4. **Universal Robots (2024).** *UR5e Technical Specifications & URDF*. [universal-robots.com](https://www.universal-robots.com/)
