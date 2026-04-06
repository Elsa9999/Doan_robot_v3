# 🤖 UR5e Robot Controller — Smart Pick & Place System

> Đồ án Tốt nghiệp: Hệ thống Điều khiển & Mô phỏng Robot UR5e ứng dụng State Machine và Học tăng cường (Reinforcement Learning - SAC).

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/RL-Stable_Baselines3-orange)
![Engine](https://img.shields.io/badge/Physics-PyBullet-red)

Dự án này cung cấp một bộ công cụ toàn diện để mô phỏng và thực thi quy trình "Gắp và Thả" (Pick & Place) của cánh tay robot công nghiệp UR5e. Điểm nhấn của dự án là việc tích hợp **3 chế độ điều khiển** thay vì chỉ một:
1. **Manual Mode (Cơ bản):** Điều khiển tay qua giao diện HMI (Jog trục Cartesian & Joints).
2. **Auto Mode (Cổ điển):** Chạy tự động hóa bằng kiến trúc Finite State Machine (FSM) và Quy hoạch quỹ đạo hình thang (Trapezoidal Trajectory).
3. **AI Mode (Hiện đại):** Trí tuệ nhân tạo Reinforcement Learning (SAC) tự động tìm kiếm đường đi tối ưu theo kiểu Bang-Bang Control mà không cần lập trình trước quỹ đạo.

---

## 📂 Tổ chức Thư mục (Project Structure)

```text
📦 do_an_robot_v2
 ┣ 📂 hmi/                  # Nhánh Giao diện người dùng (PyQt5)
 ┃ ┣ 📜 app.py              # File chạy chính - Khởi động phần mềm 🚀
 ┃ ┣ 📜 main_window.py      # Bố cục cửa sổ HMI (Manual, Trajectory, Auto, AI)
 ┃ ┣ 📜 sim_bridge.py       # Cầu nối (Thread) liên kết GUI với PyBullet engine
 ┃ ┗ 📂 widgets/            # Các mảng giao diện nhỏ (Status, AI Panel, Log)
 ┣ 📂 kinematics/           # Khối Toán học Động học
 ┃ ┣ 📜 forward_kinematics.py   # Tính tọa độ từ góc quay (FK)
 ┃ ┣ 📜 inverse_kinematics.py   # Tính góc khớp từ tọa độ xyz (IK Solver)
 ┃ ┣ 📜 trajectory.py           # Tính toán Spline và Hình thang (Trapezoid velocity)
 ┃ ┗ 📜 workspace_validator.py  # Giới hạn vùng không gian làm việc an toàn
 ┣ 📂 simulation/           # Lõi Vật lý Mô phỏng (PyBullet Engine)
 ┃ ┣ 📜 environment.py      # Môi trường 3D (Sinh ra bàn, hộp, sọt rác, robot)
 ┃ ┣ 📜 gripper.py          # Tay gắp giác hút (Vacuum Gripper Constraint)
 ┃ ┗ 📜 pick_place_sm.py    # Máy trạng thái hữu hạn cho Auto Mode cổ điển
 ┣ 📂 models_rl_place/      # Chứa "Bộ não AI" (Mạng Neural) đã train thành công
 ┃ ┗ 📂 seed42/best_model.zip   # Trọng số của đỉnh cao AI (Thành công 100%)
 ┣ 📜 train_place.py        # File chứa Thuật toán huấn luyện AI (Run here to train)
 ┣ 📜 train_rl.py           # Thuật toán huấn luyện Giai đoạn 1 (Curriculum phase 1)
 ┗ 📜 run_demo.py           # Kịch bản trình diễn mô hình chỉ bằng lệnh Console
```

---

## 🧮 Lõi Toán học: Động học Thuận (FK) và Nghịch (IK)

Điểm nổi bật của đồ án là không phụ thuộc vào thư viện IK đen (Black-box) của PyBullet mà hoàn toàn lập trình bằng thư viện Toán `numpy` & `scipy`, bám ráp chính xác theo catalogue của dòng UR5e.

### 1. Bảng Thông số Denavit-Hartenberg (DH Table)
Bảng thông số cấu hình dưới đây được trích xuất (Parse) tự động từ file URDF của bản UR5e:

| Khớp (Joint) | `a` (m) | `d` (m) | `alpha` ( rad) | `θ_offset` | Chức năng (Function) |
|---|---|---|---|---|---|
| **1** | 0.0000 | 0.1625 | π/2 | 0 | Shoulder Pan |
| **2** | -0.4250 | 0.0000 | 0 | 0 | Shoulder Lift |
| **3** | -0.3922 | 0.0000 | 0 | 0 | Elbow |
| **4** | 0.0000 | 0.1333 | π/2 | 0 | Wrist 1 |
| **5** | 0.0000 | 0.0997 | -π/2 | 0 | Wrist 2 |
| **6** | 0.0000 | 0.0996 | 0 | 0 | Wrist 3 |

**Công thức Động học Thuận (Forward Kinematics):**
$$ T_i^{i-1} = Rot_z(\theta_i) \cdot Trans_z(d_i) \cdot Trans_x(a_i) \cdot Rot_x(\alpha_i) $$
Sau đó nhân chuỗi 6 ma trận biến đổi thuần nhất (Homogeneous Transformation Matrix 4x4) lại với nhau để ra vị trí điểm cuối của tay gắp (End-Effector).

### 2. Thuật giải Động học Nghịch (Inverse Kinematics)
Quy trình đoán góc khớp khi biết điểm tọa độ X, Y, Z muốn gắp đến được vận hành theo cơ chế 2 lớp (Hybrid Solver):
- **Lớp 1 (Analytical):** Tính nghiệm giải tích kín từ hình học lượng giác và ma trận nghịch đảo. Tốc độ thực thi trong vòng vài phần ngàn giây. Trả về đến 8 nghiệm và lựa chọn cấu hình an toàn nhất so với Home Pose.
- **Lớp 2 (Numerical):** Nếu điểm cần với quá khắt khe khiến Lớp 1 thất bại (ma trận dị thường Singularity), thuật toán tối ưu hóa vô hướng `L-BFGS-B` của thư viện Scipy sẽ tiếp quản, tìm bộ góc gần tiệm cận mục tiêu nhất dựa trên hàm lỗi xoay (Rotation Error) và lỗi tịnh tiến (Translation Error).

---

## 🧠 Chế Độ Trí Tuệ Nhân Tạo (Curriculum Learning)

Dự án này không huấn luyện AI bằng cách ném robot vào một bài toán khó ngay từ đầu. Thay vào đó, chúng mình áp dụng **Curriculum Learning (Học theo giáo trình)** gồm 2 giai đoạn:

### Giai đoạn 1: Học Gắp (Grasping) - Tập tin `train_rl.py`
- **Mục tiêu:** Chỉ học cách tiếp cận vật thể và kết nối giác hút.
- **Không gian quan sát (10D):** `Tọa độ Kẹp_EndEffector (3)` + `Tọa độ Vật_Object (3)` + `Vector khoảng cách (3)` + `Trạng thái kẹp (1)`.
- **Thưởng phạt (Reward):** Phạt nếu rời xa vật, thưởng +10 nếu kẹp được.

### Giai đoạn 2: Học Gắp và Thả (Pick & Place) - Tập tin `train_place.py`
- **Mục tiêu:** Giữ nguyên khả năng gắp, nhưng học thêm cách mang vật tới sọt rác và buông tay.
- **Kỹ thuật Surgery Tensor:** Vá thêm input tensor vào mạng Neural cũ để nâng mức quan sát lên **13D** (Thêm 3 vector khoảng cách từ vật -> thùng).
- **Thưởng phạt:** Sửa lỗ hổng toán học (Reward Hacking) khiến AI thích đứng yên "bú điểm". Chỉ khi mang lọt vào thùng (`is_in_bin()`) mới được +50 điểm.

---

## ⚙️ Hướng dẫn Sử dụng (How to Run)

### 1. Cài đặt thư viện
Hãy chắc chắn bạn đã cài đặt đủ các gói phụ thuộc (Môi trường Python >= 3.10):
```bash
pip install pybullet numpy scipy PyQt5 stable-baselines3 torch tensorboard gymnasium
```

### 2. Mở Phần mềm Điều khiển Trung tâm (HMI)
Để bắt đầu thao tác với Robot và xài toàn bộ các chế độ chức năng của dự án:
```bash
python -m hmi.app
```
*Lưu ý: Trên Windows, AI sử dụng `PyTorch` nên có thể sẽ mất 2-3 giây chững lại lúc mở app để Nạp lõi DLL Tensor (C10.dll).*

### 3. Huấn luyện lại AI (Training) / Vẽ biểu đồ
Nếu bạn muốn chứng minh cách AI học hoặc thay đổi hàm Reward và tự train lại:
```bash
# Xóa thư mục models_rl_place cũ nếu muốn train lại
python train_place.py
```
Để xem biểu đồ hiệu suất (Actor Loss, Critic Loss, Episode Reward):
```bash
tensorboard --logdir logs_rl_place/
```
---

## 🚀 Lộ trình Tương lai (Roadmap)
Dự án giới hạn trong phạm vi Proof of Concept (Khái niệm có thể ứng dụng được). Để đem con Robot này ra nhặt rác ngoài đời thực (Sim-to-Real), cần nâng cấp những điểm sau:
1. **Nâng cấp Đầu vào thành 17D:** Cung cấp cho AI biết góc lật (Quaternions - 4D) của vật thể để nó biết xoay cổ tay đón lõng vật bị đổ hoặc ngang.
2. **Hệ thống Nhận diện Ảnh (OpenCV/YOLO):** Hiện tại Robot lấy tọa độ vật theo kiểu "Toàn tri" từ Engine vật lý. Ở đời thực, cần gắn Camera RealSense lấy tọa độ 3D.
3. **Thư viện Truyền thông Công nghiệp (`ur_rtde`):** Thay vì giả lập mô tơ bằng `pybullet.setJointMotorControl2`, sẽ đẩy lệnh nội suy xuống mâm điện con UR5e thật.
