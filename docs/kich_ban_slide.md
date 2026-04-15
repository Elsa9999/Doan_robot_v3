# KỊCH BẢN SLIDE THUYẾT TRÌNH ĐỒ ÁN TỐT NGHIỆP
# Robot UR5e — Pick & Place với Học Tăng Cường (SAC)
# Tổng cộng: ~18 slides

---

## SLIDE 1 — TRANG BÌA
**Nội dung:**
- Tên đề tài: "Hệ thống Điều khiển & Mô phỏng Robot UR5e — Ứng dụng Học Tăng Cường cho bài toán Pick & Place"
- Họ tên sinh viên
- GVHD
- Khoa / Trường
- Năm 2026

**Hình ảnh:** Ảnh chụp giao diện HMI + hình 3D robot UR5e trong PyBullet

---

## SLIDE 2 — NỘI DUNG TRÌNH BÀY
**Nội dung:**
1. Đặt vấn đề & Mục tiêu
2. Cơ sở lý thuyết (FK, IK, RL)
3. Thiết kế hệ thống
4. Chế độ Auto (FSM)
5. Chế độ AI (SAC RL)
6. Kết quả & Demo
7. Kết luận & Hướng phát triển

---

## SLIDE 3 — ĐẶT VẤN ĐỀ
**Nội dung:**
- Robot công nghiệp cần thực hiện Pick & Place trong sản xuất
- Thách thức: vật thể nằm ngẫu nhiên, cần tính quỹ đạo tránh va chạm, 6 bậc tự do
- Câu hỏi nghiên cứu: "Liệu AI (RL) có thể tự học điều khiển robot Pick & Place mà không cần lập trình cứng quỹ đạo?"

**Hình ảnh:** Sơ đồ so sánh 2 hướng:
```
┌────────────────────┐     ┌────────────────────┐
│  TRUYỀN THỐNG      │     │  HỌC TĂNG CƯỜNG    │
│  FSM + IK + Traj   │ VS  │  SAC Deep RL       │
│  ✓ Ổn định         │     │  ✓ Tự thích nghi   │
│  ✗ Cứng nhắc       │     │  ✗ Training lâu    │
└────────────────────┘     └────────────────────┘
```

---

## SLIDE 4 — MỤC TIÊU ĐỒ ÁN
**Nội dung (Bullet points):**
1. ✅ Xây dựng mô phỏng 3D robot UR5e (PyBullet)
2. ✅ Tự lập trình FK/IK theo DH convention (không dùng black-box)
3. ✅ 3 chế độ điều khiển: Manual · Auto · AI
4. ✅ Train AI bằng SAC + Curriculum Learning
5. ✅ Giao diện HMI chuyên nghiệp (PyQt5)

**Hình ảnh:** Screenshot giao diện HMI với 4 tab

---

## SLIDE 5 — CÔNG NGHỆ SỬ DỤNG
**Nội dung (Bảng/Icons):**

| Thành phần | Công nghệ |
|---|---|
| 🐍 Ngôn ngữ | Python 3.10+ |
| 🎮 Vật lý | PyBullet |
| 🖥️ Giao diện | PyQt5 |
| 🧠 RL | Stable-Baselines3 (SAC) |
| 🔥 Deep Learning | PyTorch |
| 📐 Giải IK | SciPy (L-BFGS-B) |

---

## SLIDE 6 — KIẾN TRÚC HỆ THỐNG
**Nội dung:** Sơ đồ block diagram

```
┌─────────────────────────────────────────────────┐
│               HMI (PyQt5)                       │
│  [Manual] [Trajectory] [Auto/FSM] [AI/RL]       │
└──────────────────┬──────────────────────────────┘
                   │ Queue
┌──────────────────▼──────────────────────────────┐
│           SimBridge (Thread)                     │
│  Manual Jog │ FSM+IK+Traj │ SAC RL Agent        │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│         PyBullet Physics (240 Hz)                │
│   UR5e URDF · Object · Bin · Vacuum Gripper      │
└─────────────────────────────────────────────────┘
```

---

## SLIDE 7 — ĐỘNG HỌC THUẬN (FK)
**Nội dung:**
- Quy ước DH: nhân chuỗi 6 ma trận 4×4
- Công thức: T_EE = T₁ × T₂ × T₃ × T₄ × T₅ × T₆
- Input: 6 góc khớp [q1...q6]
- Output: vị trí (x,y,z) + hướng (R3×3) của EE

**Hình ảnh:**
- Bảng DH (6 dòng)
- Hình minh họa robot UR5e với các trục khớp được đánh số

---

## SLIDE 8 — ĐỘNG HỌC NGHỊCH (IK)
**Nội dung:**
- Bài toán: biết (x,y,z) + hướng → tìm [q1...q6]
- Hybrid Solver 2 lớp:
  - Lớp 1: Analytical (giải tích) → < 1ms, 8 nghiệm
  - Lớp 2: Numerical (L-BFGS-B) → < 10ms, dùng khi Lớp 1 fail

**Hình ảnh:** Flowchart:
```
Input (xyz, euler) → Analytical Solver
                         │
                    Thành công? ──Yes──→ Chọn nghiệm gần HOME nhất
                         │
                        No
                         │
                    Numerical Solver (L-BFGS-B)
                         │
                    Output: [q1...q6]
```

---

## SLIDE 9 — QUY HOẠCH QUỸ ĐẠO
**Nội dung:**
- Joint Space: nội suy góc khớp, velocity profile hình thang
- Cartesian Space: nội suy XYZ → gọi IK mỗi điểm → ghép Joint Traj
- Đảm bảo v_max, a_max, chuyển tiếp mượt

**Hình ảnh:** Đồ thị velocity profile hình thang (trapezoidal):
```
velocity
  │     ___________
  │    /           \
  │   /             \
  │  /               \
  │ /                 \
  └──────────────────── time
  accelerate  cruise  decelerate
```

---

## SLIDE 10 — CHẾ ĐỘ AUTO (FSM)
**Nội dung:**
- 10 trạng thái tuần tự
- Mỗi state chạy 1 trajectory, khi xong → chuyển state tiếp
- Timeout 15s/state → ERROR nếu kẹt

**Hình ảnh:** Sơ đồ FSM:
```
IDLE → DETECT → APPROACH → DESCEND → PICK
                                       │
DONE ← RETREAT ← RELEASE ← PLACE ← LIFT ← MOVE_TO_BIN
```

---

## SLIDE 11 — GIỚI THIỆU HỌC TĂNG CƯỜNG
**Nội dung:**
- Định nghĩa: Agent tương tác với Environment, nhận Reward, tự học tối ưu
- Vòng lặp: State → Action → Reward → State' → Update Neural Network
- Khác với Deep Learning truyền thống: không cần dữ liệu gán nhãn

**Hình ảnh:** Sơ đồ vòng lặp RL:
```
       ┌──── action a_t ────►┐
       │                      │
   [Agent]              [Environment]
       │                      │
       └◄── reward r_t ──────┘
       └◄── state s_{t+1} ───┘
```

---

## SLIDE 12 — THUẬT TOÁN SAC
**Nội dung:**
- SAC = Soft Actor-Critic (Off-Policy, Continuous Action)
- 3 mạng: Actor (ra hành động) + 2 Critics (đánh giá)
- Đặc biệt: Entropy regularization — tự cân bằng khai thác/khám phá
- Tại sao chọn SAC? Sample-efficient, ổn định, liên tục

**Hình ảnh:** Kiến trúc mạng:
```
Obs (17D) ──► Actor [512→512→256] ──► Action (7D)
          ──► Critic₁ [512→512→256] ──► Q₁
          ──► Critic₂ [512→512→256] ──► Q₂
          ──► α (auto-tuned entropy)
```

---

## SLIDE 13 — OBSERVATION & ACTION SPACE
**Nội dung:**

**Observation (17D):**
| 0-2 | Vị trí EE (xyz) |
| 3-5 | Vị trí vật (xyz) |
| 6-8 | Vector EE→Vật |
| 9-11 | Vector Vật→Bin |
| 12-15 | Quaternion hướng vật |
| 16 | Trạng thái gripper |

**Action (7D):**
| 0-2 | Δxyz (±5cm/step) |
| 3-5 | ΔRoll/Pitch/Yaw (±4.5°/step) |
| 6 | Gripper ON/OFF |

---

## SLIDE 14 — THIẾT KẾ HÀM REWARD (Phần quan trọng nhất!)
**Nội dung:**
- Vấn đề: Reward Hacking — AI gian lận ăn điểm
- Giải pháp: Phase-Based Reward

**Hình ảnh:** Bảng 3 pha:
```
┌─────────────────────────────────┐
│  PHA 0 — APPROACH & GRASP      │
│  Thưởng gần vật: +1.5          │
│  ★ GẮP ĐƯỢC: +50 → Pha 1      │
├─────────────────────────────────┤
│  PHA 1 — CARRY                  │
│  Nâng ≥ 20cm: +30              │
│  Bay thấp: −6 (tránh va chạm)  │
│  Gần bin XY < 15cm → Pha 2     │
├─────────────────────────────────┤
│  PHA 2 — PLACE                  │
│  Hạ đúng vị trí: +2.5          │
│  ★ VÀO BIN: +500 → DONE ✓     │
└─────────────────────────────────┘
```

---

## SLIDE 15 — CURRICULUM LEARNING
**Nội dung:**
- Phase 1 (Học Gắp): 3M steps, 1 tiếng → AI biết gắp 100%
- Phase 2 (Học Pick&Place): 10M steps, 4.5 tiếng → AI biết cả quy trình
- Transfer Learning: load weights Phase 1 → train tiếp Phase 2

**Hình ảnh:**
```
[Phase 1: Gắp]         [Phase 2: Pick & Place]
  3M steps    ──────►    10M steps
  4 envs      weights    20 envs
  1 tiếng     transfer   4.5 tiếng
  100%                   100% (training)
```

---

## SLIDE 16 — KẾT QUẢ TRAINING
**Nội dung:**

| Metric | Phase 1 | Phase 2 |
|---|---|---|
| Steps | 3,000,000 | 10,000,000 |
| Envs song song | 4 | 20 |
| Thời gian | ~60 phút | ~4.5 tiếng |
| Success rate | 100% | 100% (training) |
| FPS | ~200 | ~600 |
| Hardware | Core i7, 16GB RAM | Core i7, 16GB RAM |

**Hình ảnh:** Screenshot TensorBoard (nếu có) hoặc bảng log training cuối cùng

---

## SLIDE 17 — SO SÁNH AUTO vs AI
**Nội dung:**

| Tiêu chí | Auto (FSM) | AI (SAC) |
|---|---|---|
| Thành công | 100% ✓ | ~95% |
| Cần lập trình quỹ đạo | Có ✗ | Không ✓ |
| Thích nghi | Không ✗ | Có ✓ |
| Khi vật ngoài vùng train | OK | Giảm (OOD) ✗ |
| Kết luận | Ổn định, cứng | Linh hoạt, cần train |

**Hình ảnh:** 2 ảnh so sánh quỹ đạo:
- Auto: đường thẳng vuông góc (approach → descend → lift → move)
- AI: đường cong mượt parabol tự nhiên

---

## SLIDE 18 — DEMO TRỰC TIẾP
**Nội dung:**
- Chạy `python -m hmi.app`
- Demo 3 chế độ: Manual → Auto → AI
- Nhấn mạnh: AI tự tìm đường, không lập trình trước

**Hành động:** Mở phần mềm, chạy live demo cho hội đồng xem

---

## SLIDE 19 — HẠN CHẾ & HƯỚNG PHÁT TRIỂN
**Nội dung:**

**Hạn chế:**
- AI yếu ngoài vùng train (Out-of-Distribution)
- Tọa độ vật từ PyBullet (chưa dùng camera thật)
- Chưa Sim-to-Real

**Hướng phát triển:**
1. Domain Randomization → mở rộng vùng hoạt động
2. Camera RealSense + Point Cloud → thay thế omniscient
3. ur_rtde → kết nối robot UR5e thật
4. Multi-Object → phân loại theo màu/hình

---

## SLIDE 20 — CẢM ƠN
**Nội dung:**
- Cảm ơn GVHD
- Cảm ơn hội đồng
- Q&A

---

# GHI CHÚ CHO NGƯỜI LÀM SLIDE:
1. Tông màu đề xuất: Nền tối (dark theme) + accent xanh dương/tím
2. Font: Roboto hoặc Inter (hiện đại, dễ đọc)
3. Mỗi slide nên có ít nhất 1 hình ảnh/sơ đồ
4. Các sơ đồ ASCII ở trên nên được vẽ lại bằng shape/diagram chuyên nghiệp
5. Slide quan trọng nhất: Slide 14 (Reward Design) — đây là phần nghiên cứu cốt lõi
6. Nên thêm animation cho các sơ đồ FSM và RL loop
