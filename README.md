# UR5e Robot Simulation — Đồ án mô phỏng robot

Dự án mô phỏng robot UR5e hoàn chỉnh với PyBullet + PyQt5 HMI.

## Cấu trúc dự án

```
├── urdf/                    # URDF robot model
├── kinematics/              # FK, IK, Trajectory, Workspace Validator
├── simulation/              # PyBullet environment + controllers
├── hmi/                     # PyQt5 HMI (app, bridge, widgets)
├── tests/                   # Test suite
├── requirements.txt
└── main.py                  # PyBullet keyboard mode (standalone)
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy chương trình

### HMI đầy đủ (PyBullet + GUI song song):
```bash
python hmi/app.py
```

### Keyboard mode (chỉ PyBullet):
```bash
python main.py
```

## Giai đoạn đã hoàn thành

| Giai đoạn | Nội dung | Status |
|-----------|----------|--------|
| 1 | FK/IK + PyBullet Environment + Manual Controller | ✅ |
| 2 | HMI PyQt5 + Dynamics Control + Workspace Validator | ✅ |
| 3 | Trajectory Planning (Trapezoid + Spline) + Executor | ✅ |

## Tính năng

- **Forward Kinematics**: DH parameters chuẩn UR5e, sai số < 0.000002m
- **Inverse Kinematics**: Analytical (8 solutions) + Numerical fallback
- **Trajectory**: Trapezoid velocity profile, Joint Space, Cartesian Space, Multi-waypoint Spline
- **Workspace Validator**: Giới hạn không gian làm việc + vùng cấm bin
- **HMI**: Dark theme, Joint sliders, Cartesian jog, Trajectory panel, E-STOP, Log panel

## Yêu cầu

- Python 3.10+
- pybullet
- numpy, scipy
- PyQt5
