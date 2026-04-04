import sys
from PyQt5.QtWidgets import (QApplication, QGroupBox, QGridLayout, QVBoxLayout, 
                             QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

class CartesianPanel(QGroupBox):
    cartesian_go = pyqtSignal(list, list)
    cartesian_jog = pyqtSignal(str, float)
    
    def __init__(self, parent=None):
        super().__init__("Cartesian Control", parent)
        self.layout = QVBoxLayout(self)
        
        self._setup_section1()
        self._setup_section2()
        self._setup_section3()
        
    def _setup_section1(self):
        group = QGroupBox("Target Pose")
        grid = QGridLayout(group)
        
        self.target_spins = {}
        
        configs = [
            ('X', 0, 0, ' m', 1.5), ('Y', 0, 1, ' m', 1.5),
            ('Z', 1, 0, ' m', 1.5), 
            ('R', 2, 0, ' rad', 3.14), ('P', 2, 1, ' rad', 3.14),
            ('Yaw', 3, 0, ' rad', 3.14)
        ]
        
        for name, row, col, suffix, limit in configs:
            hbox = QHBoxLayout()
            lbl = QLabel(name)
            lbl.setFixedWidth(30)
            
            spin = QDoubleSpinBox()
            spin.setRange(-limit, limit)
            spin.setDecimals(3)
            # 0.005 step for pos, 0.01 for rot
            spin.setSingleStep(0.005 if 'm' in suffix else 0.01)
            spin.setSuffix(suffix)
            spin.setFixedWidth(90)
            
            hbox.addWidget(lbl)
            hbox.addWidget(spin)
            grid.addLayout(hbox, row, col)
            self.target_spins[name] = spin
            
        self.btn_sync = QPushButton("📍 Sync từ EE hiện tại")
        self.btn_sync.setStyleSheet("background-color: #555555; color: white; border-radius: 2px;")
        self.btn_sync.clicked.connect(self.sync_target_to_current)
        grid.addWidget(self.btn_sync, 1, 1)

        self.btn_go = QPushButton("▶  Go To Pose")
        self.btn_go.setFixedHeight(32)
        self.btn_go.setStyleSheet("""
            QPushButton { background-color: #00aa44; color: white; font-weight: bold; border-radius: 4px; }
            QPushButton:hover { background-color: #00cc55; }
        """)
        self.btn_go.clicked.connect(self._on_go_clicked)
        grid.addWidget(self.btn_go, 3, 1)
        
        self.layout.addWidget(group)

    def _setup_section2(self):
        group = QGroupBox("Jog")
        vbox = QVBoxLayout(group)
        
        hbox_step = QHBoxLayout()
        hbox_step.addWidget(QLabel("Step size:"))
        self.spin_step = QDoubleSpinBox()
        self.spin_step.setRange(0.001, 0.05)
        self.spin_step.setValue(0.01)
        self.spin_step.setSingleStep(0.001)
        self.spin_step.setSuffix(" m")
        hbox_step.addWidget(self.spin_step)
        hbox_step.addStretch()
        vbox.addLayout(hbox_step)
        
        jog_layout = QHBoxLayout()
        
        # Grid X, Y
        grid_xy = QGridLayout()
        buttons_xy = [
            ('', 0, 0), ('X+', 0, 1), ('', 0, 2),
            ('Y+', 1, 0), ('', 1, 1), ('Y-', 1, 2),
            ('', 2, 0), ('X-', 2, 1), ('', 2, 2)
        ]
        
        for name, r, c in buttons_xy:
            if name:
                btn = self._create_jog_btn(name)
                grid_xy.addWidget(btn, r, c)
            else:
                grid_xy.addWidget(QLabel(""), r, c)
                
        jog_layout.addLayout(grid_xy)
        
        # VBox Z
        vbox_z = QVBoxLayout()
        vbox_z.addWidget(self._create_jog_btn("Z+"))
        vbox_z.addWidget(self._create_jog_btn("Z-"))
        jog_layout.addLayout(vbox_z)
        
        vbox.addLayout(jog_layout)
        self.layout.addWidget(group)
        
    def _create_jog_btn(self, name):
        btn = QPushButton(name)
        btn.setFixedSize(40, 40)
        btn.setStyleSheet("""
            QPushButton { background-color: #333333; color: white; font-weight: bold; border: none; border-radius: 4px;}
            QPushButton:hover { background-color: #555555; border: 1px solid #00aaff; }
        """)
        btn.clicked.connect(lambda _, n=name: self._on_jog_clicked(n))
        return btn

    def _setup_section3(self):
        group = QGroupBox("Current EE Pose")
        grid = QGridLayout(group)
        
        mono_font = QFont("Consolas", 9)
        mono_font.setStyleHint(QFont.Monospace)
        
        self.current_labels = {}
        for i, name in enumerate(['X', 'Y', 'Z', 'R', 'P', 'Yaw']):
            lbl = QLabel(f"{name}: 0.000")
            lbl.setFont(mono_font)
            lbl.setStyleSheet("color: #00ccff;")
            grid.addWidget(lbl, i % 3, i // 3)
            self.current_labels[name] = lbl
            
        self.layout.addWidget(group)
        self._current_ee_pos = [0, 0, 0]
        self._current_ee_euler = [0, 0, 0]

    def update_ee_display(self, pos, euler):
        self._current_ee_pos = list(pos)
        self._current_ee_euler = list(euler)
        
        self.current_labels['X'].setText(f"X: {pos[0]:.4f}")
        self.current_labels['Y'].setText(f"Y: {pos[1]:.4f}")
        self.current_labels['Z'].setText(f"Z: {pos[2]:.4f}")
        
        self.current_labels['R'].setText(f"R: {euler[0]:.4f}")
        self.current_labels['P'].setText(f"P: {euler[1]:.4f}")
        self.current_labels['Yaw'].setText(f"Yaw: {euler[2]:.4f}")

    def sync_target_to_current(self):
        self.target_spins['X'].setValue(self._current_ee_pos[0])
        self.target_spins['Y'].setValue(self._current_ee_pos[1])
        self.target_spins['Z'].setValue(self._current_ee_pos[2])
        self.target_spins['R'].setValue(self._current_ee_euler[0])
        self.target_spins['P'].setValue(self._current_ee_euler[1])
        self.target_spins['Yaw'].setValue(self._current_ee_euler[2])
        print("[CARTESIAN] Synced target to current EE params")

    def _on_go_clicked(self):
        pos = [
            self.target_spins['X'].value(),
            self.target_spins['Y'].value(),
            self.target_spins['Z'].value()
        ]
        euler = [
            self.target_spins['R'].value(),
            self.target_spins['P'].value(),
            self.target_spins['Yaw'].value()
        ]
        self.cartesian_go.emit(pos, euler)
        
    def _on_jog_clicked(self, name):
        step = self.spin_step.value()
        axis = name[0].lower()
        sign = '+' if name[1] == '+' else '-'
        axis_str = f"{axis}{sign}"
        self.cartesian_jog.emit(axis_str, step)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet("QWidget { background-color: #2d2d2d; color: #ffffff; }")
    
    panel = CartesianPanel()
    panel.show()
    
    panel.cartesian_go.connect(lambda p, e: print(f"[SIGNAL] Go to: pos={p} euler={e}"))
    panel.cartesian_jog.connect(lambda axis, step: print(f"[SIGNAL] Jog: {axis} step={step}"))
    
    panel.update_ee_display([-0.4919, -0.1333, 0.4879], [-3.141, 0.0, 1.571])
    
    # automated test
    import threading, time
    def auto_test():
        time.sleep(2)
        # Click Go
        print("[TEST] Clicking Go...")
        panel.btn_go.clicked.emit()
        time.sleep(2)
        # Click jog X+
        print("[TEST] Clicking Jog X+...")
        panel._on_jog_clicked("X+")
        time.sleep(2)
        # Click Sync
        print("[TEST] Clicking Sync...")
        panel.btn_sync.clicked.emit()
        time.sleep(2)
        app.quit()
        
    threading.Thread(target=auto_test, daemon=True).start()
    
    sys.exit(app.exec_())
