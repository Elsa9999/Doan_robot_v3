import sys
import math
import random
from PyQt5.QtWidgets import (QApplication, QGroupBox, QVBoxLayout, QHBoxLayout,
                             QLabel, QProgressBar, QGridLayout)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QTimer, Qt

class StatusPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Robot Status", parent)
        self.layout = QVBoxLayout(self)
        
        # Section 1: Connection + E-STOP
        self._setup_section_1()
        
        # Section 2: EE Pose
        self._setup_section_2()
        
        # Section 3: Joint Angles
        self._setup_section_3()
        
        self.set_connected(False)
        self.set_estop(False)

    def _setup_section_1(self):
        hbox = QHBoxLayout()
        
        self.conn_indicator = QLabel("●")
        self.conn_indicator.setFont(QFont("Arial", 14))
        self.conn_label = QLabel("Simulation: Stopped")
        
        self.estop_label = QLabel("⚠ EMERGENCY STOP")
        self.estop_label.setStyleSheet("color: #ff3333; font-weight: bold;")
        self.estop_label.setVisible(False)
        
        hbox.addWidget(self.conn_indicator)
        hbox.addWidget(self.conn_label)
        hbox.addStretch()
        hbox.addWidget(self.estop_label)
        self.layout.addLayout(hbox)

    def _setup_section_2(self):
        group = QGroupBox("End-Effector Pose")
        grid = QGridLayout(group)
        
        mono_font = QFont("Consolas", 10)
        mono_font.setStyleHint(QFont.Monospace)
        
        self.ee_labels = {}
        labels = ['X', 'Y', 'Z', 'R', 'P', 'Yaw']
        
        for i, name in enumerate(labels):
            lbl_title = QLabel(f"{name}:")
            lbl_val = QLabel("0.0000")
            lbl_val.setFont(mono_font)
            lbl_val.setStyleSheet("color: #00ccff;")
            
            row = i % 3
            col = (i // 3) * 2
            
            grid.addWidget(lbl_title, row, col)
            grid.addWidget(lbl_val, row, col + 1)
            
            self.ee_labels[name] = lbl_val
            
        self.layout.addWidget(group)

    def _setup_section_3(self):
        group = QGroupBox("Joint Angles")
        vbox = QVBoxLayout(group)
        
        self.joint_bars = []
        self.joint_labels = []
        
        # Limits từ manual_controller, dùng giá trị gần đúng để map progress
        # limits range: 6.28 * 2 = 12.56
        self.joint_limits = [
            (-6.28, 6.28), (-6.28, 6.28), (-3.14, 3.14),
            (-6.28, 6.28), (-6.28, 6.28), (-6.28, 6.28)
        ]
        
        mono_font = QFont("Consolas", 9)
        mono_font.setStyleHint(QFont.Monospace)
        
        for i in range(6):
            hbox = QHBoxLayout()
            lbl_j = QLabel(f"J{i+1}")
            lbl_j.setFixedWidth(20)
            
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            bar.setFixedHeight(10)
            
            lbl_val = QLabel("0.00 rad / 0.0°")
            lbl_val.setFont(mono_font)
            lbl_val.setFixedWidth(130)
            lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            
            hbox.addWidget(lbl_j)
            hbox.addWidget(bar)
            hbox.addWidget(lbl_val)
            vbox.addLayout(hbox)
            
            self.joint_bars.append(bar)
            self.joint_labels.append(lbl_val)
            
        self.layout.addWidget(group)

    def update_state(self, state: dict):
        if 'ee_pos' in state and 'ee_euler' in state:
            pos = state['ee_pos']
            euler = state['ee_euler']
            self.ee_labels['X'].setText(f"{pos[0]:.4f} m")
            self.ee_labels['Y'].setText(f"{pos[1]:.4f} m")
            self.ee_labels['Z'].setText(f"{pos[2]:.4f} m")
            self.ee_labels['R'].setText(f"{euler[0]:.3f} rad")
            self.ee_labels['P'].setText(f"{euler[1]:.3f} rad")
            self.ee_labels['Yaw'].setText(f"{euler[2]:.3f} rad")
            
        if 'q' in state:
            for i, val in enumerate(state['q']):
                deg = math.degrees(val)
                self.joint_labels[i].setText(f"{val:+.2f} rad / {deg:+.1f}°")
                
                lower, upper = self.joint_limits[i]
                span = upper - lower
                zero_offset = val - lower
                pct = int(max(0, min(100, (zero_offset / span) * 100)))
                
                self.joint_bars[i].setValue(pct)
                
                # Cảnh báo đỏ nếu > 90% hoặc < 10%
                if pct > 90 or pct < 10:
                    self.joint_bars[i].setStyleSheet("""
                        QProgressBar::chunk { background-color: #ff3333; }
                    """)
                else:
                    self.joint_bars[i].setStyleSheet("""
                        QProgressBar::chunk { background-color: #00aaff; }
                    """)
                    
        if 'estop' in state:
            self.set_estop(state['estop'])

    def set_connected(self, connected: bool):
        if connected:
            self.conn_indicator.setStyleSheet("color: #00ff88;")
            self.conn_label.setText("Simulation: Running")
        else:
            self.conn_indicator.setStyleSheet("color: #ff3333;")
            self.conn_label.setText("Simulation: Stopped")

    def set_estop(self, active: bool):
        self.estop_label.setVisible(active)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    panel = StatusPanel()
    panel.set_connected(True)
    panel.show()
    
    home_q = [0, -1.5708, 1.5708, -1.5708, -1.5708, 0]
    
    def simulate_noise():
        # Thêm noise để thấy slider nhúc nhích
        noisy_q = [v + random.uniform(-0.02, 0.02) for v in home_q]
        state = {
            'q': noisy_q,
            'ee_pos': [-0.4919, -0.1333, 0.4879],
            'ee_euler': [-3.141, 0.0, 1.571],
            'estop': random.random() > 0.8
        }
        panel.update_state(state)
        
    timer = QTimer()
    timer.timeout.connect(simulate_noise)
    timer.start(500)
    simulate_noise()
    
    import threading
    def auto_close():
        import time
        time.sleep(5)
        app.quit()
    threading.Thread(target=auto_close, daemon=True).start()
    
    sys.exit(app.exec_())
