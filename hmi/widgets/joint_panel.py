import sys
import math
from PyQt5.QtWidgets import (QApplication, QGroupBox, QGridLayout, 
                             QLabel, QSlider, QDoubleSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont

class JointPanel(QGroupBox):
    joints_changed = pyqtSignal(list)
    
    JOINT_CONFIG = [
        {'name': 'J1 Shoulder Pan',  'lower': -6.28, 'upper': 6.28},
        {'name': 'J2 Shoulder Lift', 'lower': -6.28, 'upper': 6.28},
        {'name': 'J3 Elbow',         'lower': -3.14, 'upper': 3.14},
        {'name': 'J4 Wrist 1',       'lower': -6.28, 'upper': 6.28},
        {'name': 'J5 Wrist 2',       'lower': -6.28, 'upper': 6.28},
        {'name': 'J6 Wrist 3',       'lower': -6.28, 'upper': 6.28},
    ]

    def __init__(self, parent=None):
        super().__init__("Joint Control", parent)
        
        self._updating = False
        self._q = [0.0] * 6
        
        self.layout = QGridLayout(self)
        
        self.sliders = []
        self.spinboxes = []
        self.rad_labels = []
        
        mono_font = QFont("Consolas", 9)
        mono_font.setStyleHint(QFont.Monospace)
        
        for i, config in enumerate(self.JOINT_CONFIG):
            lbl_name = QLabel(config['name'])
            lbl_name.setFixedWidth(120)
            
            slider = QSlider(Qt.Horizontal)
            slider.setRange(int(config['lower'] * 100), int(config['upper'] * 100))
            slider.setTickInterval(50)
            slider.setStyleSheet("""
                QSlider::groove:horizontal { background: #444; height: 4px; }
                QSlider::handle:horizontal { background: #00aaff; width: 12px; border-radius: 6px; margin: -4px 0; }
            """)
            
            spinbox = QDoubleSpinBox()
            spinbox.setRange(math.degrees(config['lower']), math.degrees(config['upper']))
            spinbox.setDecimals(1)
            spinbox.setSuffix("°")
            spinbox.setSingleStep(1.0)
            spinbox.setFixedWidth(70)
            
            lbl_rad = QLabel("0.00 rad")
            lbl_rad.setFont(mono_font)
            lbl_rad.setStyleSheet("color: #888888;")
            lbl_rad.setFixedWidth(70)
            
            self.layout.addWidget(lbl_name, i, 0)
            self.layout.addWidget(slider, i, 1)
            self.layout.addWidget(spinbox, i, 2)
            self.layout.addWidget(lbl_rad, i, 3)
            
            self.sliders.append(slider)
            self.spinboxes.append(spinbox)
            self.rad_labels.append(lbl_rad)
            
            slider.valueChanged.connect(lambda val, idx=i: self._on_slider_changed(idx, val))
            spinbox.valueChanged.connect(lambda val, idx=i: self._on_spinbox_changed(idx, val))

    def _on_slider_changed(self, idx, val):
        if self._updating: return
        self._updating = True
        
        rad_val = val / 100.0
        deg_val = math.degrees(rad_val)
        
        self.spinboxes[idx].setValue(deg_val)
        self.rad_labels[idx].setText(f"{rad_val:+.2f} rad")
        self._q[idx] = rad_val
        
        self._updating = False
        self.joints_changed.emit(self.get_q())
        
    def _on_spinbox_changed(self, idx, val):
        if self._updating: return
        self._updating = True
        
        rad_val = math.radians(val)
        slider_val = int(rad_val * 100)
        
        self.sliders[idx].setValue(slider_val)
        self.rad_labels[idx].setText(f"{rad_val:+.2f} rad")
        self._q[idx] = rad_val
        
        self._updating = False
        self.joints_changed.emit(self.get_q())

    def update_from_state(self, q: list):
        if self._updating: return
        self._updating = True
        
        self._q = list(q)
        for i, val in enumerate(q):
            slider_val = int(val * 100)
            deg_val = math.degrees(val)
            
            self.sliders[i].blockSignals(True)
            self.spinboxes[i].blockSignals(True)
            
            self.sliders[i].setValue(slider_val)
            self.spinboxes[i].setValue(deg_val)
            self.rad_labels[i].setText(f"{val:+.2f} rad")
            
            self.sliders[i].blockSignals(False)
            self.spinboxes[i].blockSignals(False)
            
        self._updating = False

    def get_q(self) -> list:
        return list(self._q)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet("QWidget { background-color: #2d2d2d; color: #ffffff; }")
    
    panel = JointPanel()
    panel.show()
    
    panel.joints_changed.connect(lambda q: print(f"[SIGNAL] Joints changed: {[round(v, 2) for v in q]}"))
    
    # Automated testing 8 seconds
    def apply_home():
        print("[TEST] Trở về Home Pose từ trạng thái hiện tại (update_from_state)")
        panel.update_from_state([0, -1.5708, 1.5708, -1.5708, -1.5708, 0])
        
    QTimer.singleShot(2000, apply_home)
    
    import threading, time
    def auto_close():
        time.sleep(4)
        # simulate user input
        print("[TEST] User kéo slider joint 0 đến 1.0 rad")
        panel.sliders[0].setValue(100)
        time.sleep(4)
        app.quit()
        
    threading.Thread(target=auto_close, daemon=True).start()
    
    sys.exit(app.exec_())
