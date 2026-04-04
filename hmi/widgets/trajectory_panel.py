import sys
from PyQt5.QtWidgets import (QApplication, QGroupBox, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QLabel, QDoubleSpinBox, QRadioButton,
                             QButtonGroup, QPushButton, QProgressBar, QSlider)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont

import math

HOME_POSE_DEG = [0, -90.0, 90.0, -90.0, -90.0, 0]
HOME_POSE_RAD = [0, -1.5708, 1.5708, -1.5708, -1.5708, 0]


class TrajectoryPanel(QGroupBox):
    traj_requested = pyqtSignal(str, list, float)    # (type, target, speed_scale)
    traj_stop      = pyqtSignal()
    speed_changed  = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__("Trajectory Control", parent)
        self._ee_pos   = [0.0, 0.0, 0.9]
        self._ee_euler = [0.0, 0.0, 0.0]
        self._q_current = list(HOME_POSE_RAD)
        self._done_timer = None

        main_layout = QVBoxLayout(self)
        self._setup_move_section(main_layout)
        self._setup_progress_section(main_layout)
        self._setup_speed_section(main_layout)

    # ─── Section 1: Move To Pose ──────────────────────────────────────────────

    def _setup_move_section(self, parent_layout):
        group = QGroupBox("Move To (Trajectory)")
        vbox = QVBoxLayout(group)

        # Radio buttons — mode selection
        radio_hbox = QHBoxLayout()
        self.rb_joint = QRadioButton("Joint Space")
        self.rb_cart  = QRadioButton("Cartesian Space")
        self.rb_joint.setChecked(True)
        radio_hbox.addWidget(self.rb_joint)
        radio_hbox.addWidget(self.rb_cart)
        radio_hbox.addStretch()
        vbox.addLayout(radio_hbox)

        self.rb_joint.toggled.connect(self._on_mode_changed)

        # Joint inputs (shown in joint mode)
        self.joint_group = QGroupBox("Joint Angles")
        jg = QGridLayout(self.joint_group)
        names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
        self._joint_spins = []
        for i, name in enumerate(names):
            lbl  = QLabel(name)
            spin = QDoubleSpinBox()
            spin.setRange(-360, 360)
            spin.setDecimals(1)
            spin.setSuffix("°")
            spin.setSingleStep(5.0)
            spin.setValue(HOME_POSE_DEG[i])
            jg.addWidget(lbl, i // 2, (i % 2) * 2)
            jg.addWidget(spin, i // 2, (i % 2) * 2 + 1)
            self._joint_spins.append(spin)

        btn_copy_j = QPushButton("📍 Copy từ hiện tại")
        btn_copy_j.clicked.connect(self._copy_joint_current)
        btn_copy_j.setStyleSheet("background: #444; color: white; border-radius: 3px; padding: 3px;")
        jg.addWidget(btn_copy_j, 3, 0, 1, 4)
        vbox.addWidget(self.joint_group)

        # Cartesian inputs (shown in cartesian mode)
        self.cart_group = QGroupBox("Cartesian Pose")
        cg = QGridLayout(self.cart_group)
        cart_defs = [
            ('X', -1.5, 1.5, 0.001, ' m', 0.40),
            ('Y', -1.5, 1.5, 0.001, ' m', 0.00),
            ('Z', -1.5, 1.5, 0.001, ' m', 0.70),
            ('R', -3.14, 3.14, 0.01, ' rad', 0.0),
            ('P', -3.14, 3.14, 0.01, ' rad', 0.0),
            ('Yaw', -3.14, 3.14, 0.01, ' rad', 1.571),
        ]
        self._cart_spins = []
        for i, (name, lo, hi, step, suf, default) in enumerate(cart_defs):
            lbl  = QLabel(name)
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setDecimals(3)
            spin.setSingleStep(step)
            spin.setSuffix(suf)
            spin.setValue(default)
            cg.addWidget(lbl, i // 2, (i % 2) * 2)
            cg.addWidget(spin, i // 2, (i % 2) * 2 + 1)
            self._cart_spins.append(spin)

        btn_copy_c = QPushButton("📍 Copy từ EE hiện tại")
        btn_copy_c.clicked.connect(self._copy_cart_current)
        btn_copy_c.setStyleSheet("background: #444; color: white; border-radius: 3px; padding: 3px;")
        cg.addWidget(btn_copy_c, 3, 0, 1, 4)
        vbox.addWidget(self.cart_group)
        self.cart_group.setVisible(False)   # hidden initially

        # Execute button
        self.btn_execute = QPushButton("▶  Execute Trajectory")
        self.btn_execute.setFixedHeight(36)
        self.btn_execute.setStyleSheet("""
            QPushButton { background: #0055cc; color: white; font-weight: bold;
                          border-radius: 4px; font-size: 11pt; }
            QPushButton:hover { background: #0077ff; }
        """)
        self.btn_execute.clicked.connect(self._on_execute)
        vbox.addWidget(self.btn_execute)

        parent_layout.addWidget(group)

    # ─── Section 2: Progress ──────────────────────────────────────────────────

    def _setup_progress_section(self, parent_layout):
        group = QGroupBox("Execution")
        vbox = QVBoxLayout(group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #444; border-radius: 3px; text-align: center; }
            QProgressBar::chunk { background: #00aa44; border-radius: 3px; }
        """)
        vbox.addWidget(self.progress_bar)

        hbox = QHBoxLayout()
        self.lbl_status = QLabel("Idle")
        self.lbl_status.setStyleSheet("color: #888888;")
        hbox.addWidget(self.lbl_status)
        hbox.addStretch()

        self.btn_stop = QPushButton("⏹  Stop")
        self.btn_stop.setFixedWidth(80)
        self.btn_stop.setStyleSheet("""
            QPushButton { background: #cc4400; color: white; font-weight: bold;
                          border-radius: 3px; padding: 4px; }
            QPushButton:hover { background: #ff5500; }
        """)
        self.btn_stop.clicked.connect(self.traj_stop.emit)
        hbox.addWidget(self.btn_stop)
        vbox.addLayout(hbox)

        parent_layout.addWidget(group)

    # ─── Section 3: Speed Override ────────────────────────────────────────────

    def _setup_speed_section(self, parent_layout):
        group = QGroupBox("Speed Override")
        hbox = QHBoxLayout(group)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(10, 200)
        self.speed_slider.setValue(100)
        self.speed_slider.setTickInterval(10)
        self.speed_slider.setStyleSheet("""
            QSlider::groove:horizontal { background: #444; height: 4px; }
            QSlider::handle:horizontal { background: #00aaff; width: 12px; border-radius: 6px; margin: -4px 0; }
        """)

        self.lbl_speed = QLabel("100%")
        self.lbl_speed.setFixedWidth(40)
        self.lbl_speed.setAlignment(Qt.AlignCenter)

        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        hbox.addWidget(self.speed_slider, 1)
        hbox.addWidget(self.lbl_speed)

        parent_layout.addWidget(group)

    # ─── Slots ────────────────────────────────────────────────────────────────

    def _on_mode_changed(self, joint_checked):
        self.joint_group.setVisible(joint_checked)
        self.cart_group.setVisible(not joint_checked)

    def _copy_joint_current(self):
        for i, spin in enumerate(self._joint_spins):
            spin.setValue(math.degrees(self._q_current[i]))

    def _copy_cart_current(self):
        for i, spin in enumerate(self._cart_spins[:3]):
            spin.setValue(self._ee_pos[i])
        for i, spin in enumerate(self._cart_spins[3:]):
            spin.setValue(self._ee_euler[i])

    def _on_execute(self):
        speed = self.speed_slider.value() / 100.0
        if self.rb_joint.isChecked():
            q_deg = [spin.value() for spin in self._joint_spins]
            q_rad = [math.radians(v) for v in q_deg]
            self.traj_requested.emit('joint', q_rad, speed)
        else:
            target = [spin.value() for spin in self._cart_spins]
            self.traj_requested.emit('cartesian', target, speed)

    def _on_speed_changed(self, val):
        self.lbl_speed.setText(f"{val}%")
        self.speed_changed.emit(val / 100.0)

    # ─── Public update methods ────────────────────────────────────────────────

    def update_progress(self, running: bool, progress: float):
        pct = int(progress * 100)
        self.progress_bar.setValue(pct)

        if running:
            self.lbl_status.setText(f"Running... {pct}%")
            self.lbl_status.setStyleSheet("color: #00aaff; font-weight: bold;")
            self.progress_bar.setStyleSheet("""
                QProgressBar { border: 1px solid #444; border-radius: 3px; text-align: center; }
                QProgressBar::chunk { background: #00aaff; border-radius: 3px; }
            """)
        elif pct >= 100:
            self.lbl_status.setText("Done ✓")
            self.lbl_status.setStyleSheet("color: #00ff88; font-weight: bold;")
            self.progress_bar.setStyleSheet("""
                QProgressBar { border: 1px solid #444; border-radius: 3px; text-align: center; }
                QProgressBar::chunk { background: #00aa44; border-radius: 3px; }
            """)
            # Đặt về Idle sau 2 giây
            if self._done_timer is None or not self._done_timer.isActive():
                self._done_timer = QTimer(self)
                self._done_timer.setSingleShot(True)
                self._done_timer.timeout.connect(self._reset_to_idle)
                self._done_timer.start(2000)
        else:
            self._reset_to_idle()

    def _reset_to_idle(self):
        self.lbl_status.setText("Idle")
        self.lbl_status.setStyleSheet("color: #888888;")
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #444; border-radius: 3px; text-align: center; }
            QProgressBar::chunk { background: #00aa44; border-radius: 3px; }
        """)

    def set_current_state(self, q, ee_pos, ee_euler):
        self._q_current = list(q)
        self._ee_pos    = list(ee_pos)
        self._ee_euler  = list(ee_euler)


if __name__ == "__main__":
    import threading, time

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet("QWidget { background-color: #2d2d2d; color: #ffffff; }")

    panel = TrajectoryPanel()
    panel.show()

    panel.traj_requested.connect(lambda t, target, s: print(f"[SIG] traj={t} speed={s:.2f} target[0]={target[0]:.3f}"))
    panel.traj_stop.connect(lambda: print("[SIG] Stop!"))
    panel.speed_changed.connect(lambda s: print(f"[SIG] Speed={s:.2f}"))

    # Simulate running trajectory
    def sim():
        time.sleep(1)
        for i in range(101):
            panel.update_progress(i < 100, i / 100.0)
            time.sleep(0.03)
        time.sleep(2)
        app.quit()

    threading.Thread(target=sim, daemon=True).start()
    sys.exit(app.exec_())
