import sys
from PyQt5.QtWidgets import (QApplication, QGroupBox, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

from simulation.pick_place_sm import STATE_ORDER, State

STATE_LABELS = {
    'IDLE':        'Idle',
    'DETECT':      'Detect Object',
    'APPROACH':    'Approach',
    'DESCEND':     'Descend',
    'PICK':        'Pick (Grasp)',
    'LIFT':        'Lift',
    'MOVE_TO_BIN': 'Move To Bin',
    'PLACE':       'Place',
    'RELEASE':     'Release',
    'RETREAT':     'Retreat Home',
    'DONE':        'Done',
    'ERROR':       'ERROR',
}


class AutoPanel(QGroupBox):
    auto_start  = pyqtSignal(bool)    # True = repeat
    auto_stop   = pyqtSignal()
    reset_error = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Auto Pick & Place", parent)
        self._current_state = 'IDLE'
        self._done_states   = set()

        layout = QVBoxLayout(self)
        self._setup_control(layout)
        self._setup_state_display(layout)
        self._setup_stats(layout)

    # ─── Section 1: Control buttons ───────────────────────────────────────────

    def _setup_control(self, parent):
        group = QGroupBox("Control")
        vbox = QVBoxLayout(group)

        self.btn_start_once = QPushButton("▶  Start Auto (1 cycle)")
        self.btn_start_once.setFixedHeight(40)
        self.btn_start_once.setStyleSheet("""
            QPushButton { background: #006600; color: white; font-weight: bold;
                          border-radius: 4px; font-size: 10pt; }
            QPushButton:hover { background: #008800; }
        """)
        self.btn_start_once.clicked.connect(lambda: self.auto_start.emit(False))
        vbox.addWidget(self.btn_start_once)

        self.btn_start_repeat = QPushButton("🔁  Start Auto (repeat)")
        self.btn_start_repeat.setFixedHeight(40)
        self.btn_start_repeat.setStyleSheet("""
            QPushButton { background: #004488; color: white; font-weight: bold;
                          border-radius: 4px; font-size: 10pt; }
            QPushButton:hover { background: #0055aa; }
        """)
        self.btn_start_repeat.clicked.connect(lambda: self.auto_start.emit(True))
        vbox.addWidget(self.btn_start_repeat)

        self.btn_stop = QPushButton("⏹  Stop Auto")
        self.btn_stop.setFixedHeight(40)
        self.btn_stop.setStyleSheet("""
            QPushButton { background: #880000; color: white; font-weight: bold;
                          border-radius: 4px; font-size: 10pt; }
            QPushButton:hover { background: #aa0000; }
        """)
        self.btn_stop.clicked.connect(self.auto_stop.emit)
        vbox.addWidget(self.btn_stop)

        parent.addWidget(group)

    # ─── Section 2: State diagram ──────────────────────────────────────────────

    def _setup_state_display(self, parent):
        group = QGroupBox("State Machine")
        vbox = QVBoxLayout(group)

        self._state_labels = {}
        for s in STATE_ORDER:
            name  = s.name
            label = QLabel(f"  ○  {STATE_LABELS.get(name, name)}")
            label.setFont(QFont("Segoe UI", 9))
            label.setStyleSheet("color: #666666; padding: 1px 4px;")
            vbox.addWidget(label)
            self._state_labels[name] = label

        parent.addWidget(group)

    # ─── Section 3: Stats ──────────────────────────────────────────────────────

    def _setup_stats(self, parent):
        group = QGroupBox("Status")
        vbox = QVBoxLayout(group)

        self.lbl_cycles = QLabel("Cycles completed: 0")
        self.lbl_cycles.setStyleSheet("color: #cccccc;")
        vbox.addWidget(self.lbl_cycles)

        self.lbl_gripper = QLabel("⬤  Gripper: Released")
        self.lbl_gripper.setStyleSheet("color: #ff4444; font-weight: bold;")
        vbox.addWidget(self.lbl_gripper)

        self.lbl_error = QLabel("")
        self.lbl_error.setStyleSheet("color: #ff2222; font-weight: bold;")
        self.lbl_error.setWordWrap(True)
        self.lbl_error.hide()
        vbox.addWidget(self.lbl_error)

        self.btn_reset_error = QPushButton("🔄  Reset Error")
        self.btn_reset_error.setStyleSheet("""
            QPushButton { background: #663300; color: white; font-weight: bold;
                          border-radius: 3px; padding: 4px; }
            QPushButton:hover { background: #994400; }
        """)
        self.btn_reset_error.clicked.connect(self.reset_error.emit)
        self.btn_reset_error.hide()
        vbox.addWidget(self.btn_reset_error)

        parent.addWidget(group)

    # ─── Public update ────────────────────────────────────────────────────────

    def update_state(self, state_dict: dict):
        current   = state_dict.get('state', 'IDLE')
        cycle     = state_dict.get('cycle', 0)
        gripper   = state_dict.get('gripper', False)
        error_msg = state_dict.get('error_msg', '')
        mode      = state_dict.get('mode', 'manual')

        # Track done states
        if current != self._current_state:
            if self._current_state not in ('IDLE', 'ERROR', 'DONE'):
                self._done_states.add(self._current_state)
            if current in ('IDLE', 'DONE', 'DETECT'):
                self._done_states.clear()
            self._current_state = current

        # Update state labels
        for name, lbl in self._state_labels.items():
            label_text = STATE_LABELS.get(name, name)
            if name == current:
                if current == 'ERROR':
                    lbl.setText(f"  ● ⚠ {label_text}")
                    lbl.setStyleSheet("color: #ff3333; font-weight: bold; "
                                      "background: #330000; border-radius: 3px; padding: 2px 4px;")
                else:
                    lbl.setText(f"  ● {label_text}")
                    lbl.setStyleSheet("color: #00ddff; font-weight: bold; "
                                      "background: #003344; border-radius: 3px; padding: 2px 4px;")
            elif name in self._done_states:
                lbl.setText(f"  ✓ {label_text}")
                lbl.setStyleSheet("color: #00cc66; padding: 1px 4px;")
            else:
                lbl.setText(f"  ○ {label_text}")
                lbl.setStyleSheet("color: #666666; padding: 1px 4px;")

        # Cycles
        self.lbl_cycles.setText(f"Cycles completed: {cycle}")

        # Gripper
        if gripper:
            self.lbl_gripper.setText("⬤  Gripper: Active")
            self.lbl_gripper.setStyleSheet("color: #00ff88; font-weight: bold;")
        else:
            self.lbl_gripper.setText("⬤  Gripper: Released")
            self.lbl_gripper.setStyleSheet("color: #ff4444; font-weight: bold;")

        # Error
        if current == 'ERROR' and error_msg:
            self.lbl_error.setText(f"Error: {error_msg}")
            self.lbl_error.show()
            self.btn_reset_error.show()
        else:
            self.lbl_error.hide()
            self.btn_reset_error.hide()

        # Button state
        in_auto = (mode == 'auto')
        self.btn_start_once.setEnabled(not in_auto)
        self.btn_start_repeat.setEnabled(not in_auto)
        self.btn_stop.setEnabled(in_auto)


if __name__ == "__main__":
    import threading, time

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet("QWidget { background-color: #2d2d2d; color: #ffffff; }")

    panel = AutoPanel()
    panel.resize(300, 600)
    panel.show()

    panel.auto_start.connect(lambda r: print(f"[SIG] auto_start repeat={r}"))
    panel.auto_stop.connect(lambda: print("[SIG] auto_stop"))
    panel.reset_error.connect(lambda: print("[SIG] reset_error"))

    # Simulate state machine progress
    states = ['DETECT', 'APPROACH', 'DESCEND', 'PICK',
              'LIFT', 'MOVE_TO_BIN', 'PLACE', 'RELEASE', 'RETREAT', 'DONE']

    def sim():
        time.sleep(0.5)
        panel.update_state({'state': 'IDLE', 'cycle': 0, 'gripper': False,
                            'mode': 'auto', 'error_msg': ''})
        time.sleep(0.5)
        for i, s in enumerate(states):
            panel.update_state({'state': s, 'cycle': 0,
                                'gripper': (i >= 4 and i < 8),
                                'mode': 'auto', 'error_msg': ''})
            time.sleep(0.4)
        panel.update_state({'state': 'IDLE', 'cycle': 1, 'gripper': False,
                            'mode': 'manual', 'error_msg': ''})
        time.sleep(1)
        app.quit()

    threading.Thread(target=sim, daemon=True).start()
    sys.exit(app.exec_())
