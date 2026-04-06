import time
from datetime import datetime
from PyQt5.QtWidgets import (QMainWindow, QWidget, QSplitter, QScrollArea, 
                             QAction, QToolBar, QStatusBar, QLabel, QTabWidget)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon

from hmi.widgets.joint_panel import JointPanel
from hmi.widgets.cartesian_panel import CartesianPanel
from hmi.widgets.status_panel import StatusPanel
from hmi.widgets.log_panel import LogPanel
from hmi.widgets.trajectory_panel import TrajectoryPanel
from hmi.widgets.auto_panel import AutoPanel
from hmi.widgets.ai_panel import AIPanel
from kinematics.forward_kinematics import forward_kinematics
from kinematics.workspace_validator import WorkspaceValidator
from utils.transforms import local_to_world

class MainWindow(QMainWindow):
    def __init__(self, bridge):
        super().__init__()
        self._bridge    = bridge
        self._estop     = False
        self._estop_state = False
        self._validator = WorkspaceValidator()
        
        self.setWindowTitle("UR5e Robot Controller — Manual Mode")
        self.resize(1280, 780)
        self.setMinimumSize(1024, 600)
        
        self._apply_style()
        self._setup_toolbar()
        self._setup_central_widget()
        self._setup_statusbar()
        self._connect_signals()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_ui)
        self.timer.start(50)
        
    def _apply_style(self):
        self.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #1e1e1e; color: #ffffff;
            font-family: Segoe UI; font-size: 10pt;
        }
        QGroupBox {
            border: 1px solid #444; border-radius: 4px;
            margin-top: 8px; padding-top: 8px;
            font-weight: bold; color: #00aaff;
        }
        QGroupBox::title {
            subcontrol-origin: margin; left: 8px;
        }
        QDoubleSpinBox, QSpinBox {
            background: #2d2d2d; border: 1px solid #555;
            border-radius: 3px; padding: 2px; color: #fff;
        }
        QScrollBar:vertical {
            background: #2d2d2d; width: 8px;
        }
        QScrollBar::handle:vertical {
            background: #555; border-radius: 4px;
        }
        """)
        
    def _setup_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setStyleSheet("QToolBar { spacing: 8px; padding: 5px; }")
        
        act_home = QAction("🏠  Home", self)
        act_home.triggered.connect(self._cmd_go_home)
        toolbar.addAction(act_home)
        
        act_reset = QAction("🔄  Reset Scene", self)
        act_reset.triggered.connect(self._cmd_reset)
        toolbar.addAction(act_reset)
        
        toolbar.addSeparator()
        
        self.act_estop = QAction("🛑  EMERGENCY STOP", self)
        self.act_estop.triggered.connect(self._cmd_toggle_estop)
        # MUST addAction BEFORE calling widgetForAction
        toolbar.addAction(self.act_estop)
        self.estop_btn = toolbar.widgetForAction(self.act_estop)
        if self.estop_btn:
            self._apply_estop_style(False)
            self.estop_btn.setMinimumHeight(32)
            self.estop_btn.setMinimumWidth(160)
            
    def _apply_estop_style(self, active: bool):
        if not self.estop_btn: return
        if active:
            self.estop_btn.setStyleSheet("""
                QToolButton { background-color: #ff6600; color: white; font-weight: bold; border-radius: 4px; padding: 5px; }
            """)
        else:
            self.estop_btn.setStyleSheet("""
                QToolButton { background-color: #cc0000; color: white; font-weight: bold; border-radius: 4px; padding: 5px; }
            """)
        
    def _cmd_go_home(self):
        if self._estop: return
        self._bridge.send_command({'type': 'go_home'})
        self.log_panel.log("Go Home", 'CMD')
        
    def _cmd_reset(self):
        self._bridge.send_command({'type': 'reset'})
        self.log_panel.log("Reset scene", 'CMD')
        self._estop = False
        self.act_estop.setText("🛑  EMERGENCY STOP")
        self._apply_estop_style(False)
        
    def _cmd_toggle_estop(self):
        if not self._estop:
            self._estop = True
            self._bridge.send_command({'type': 'emergency_stop'})
            self.act_estop.setText("⚠  RESUME")
            self._apply_estop_style(True)
            self.log_panel.log("EMERGENCY STOP!", 'ESTOP')
        else:
            self._estop = False
            self._bridge.send_command({'type': 'clear_estop'})
            self.act_estop.setText("🛑  EMERGENCY STOP")
            self._apply_estop_style(False)
            self.log_panel.log("E-Stop cleared", 'INFO')

    def _setup_central_widget(self):
        main_splitter = QSplitter(Qt.Vertical)
        top_splitter  = QSplitter(Qt.Horizontal)

        # ── Tab widget for left/mid controls ──────────────────────────────
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabBar::tab { background:#2d2d2d; color:#aaa; padding:6px 14px; border-radius:3px 3px 0 0; }
            QTabBar::tab:selected { background:#1e1e1e; color:#00aaff; border-bottom:2px solid #00aaff; }
        """)

        # Tab 1: Manual
        manual_splitter = QSplitter(Qt.Horizontal)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.joint_panel = JointPanel()
        scroll_area.setWidget(self.joint_panel)
        manual_splitter.addWidget(scroll_area)
        self.cartesian_panel = CartesianPanel()
        manual_splitter.addWidget(self.cartesian_panel)
        manual_splitter.setStretchFactor(0, 1)
        manual_splitter.setStretchFactor(1, 1)
        self.tab_widget.addTab(manual_splitter, "🕹  Manual")

        # Tab 2: Trajectory
        self.traj_panel = TrajectoryPanel()
        self.tab_widget.addTab(self.traj_panel, "📈  Trajectory")

        # Tab 3: Auto
        self.auto_panel = AutoPanel()
        self.tab_widget.addTab(self.auto_panel, "⚙️  Auto (Cổ điển)")

        # Tab 4: AI Mode
        self.ai_panel = AIPanel()
        self.tab_widget.addTab(self.ai_panel, "🧠  AI (Học Tăng Cường)")

        # Disable manual/traj when Auto is active
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        top_splitter.addWidget(self.tab_widget)

        # Right: Status panel
        self.status_panel = StatusPanel()
        top_splitter.addWidget(self.status_panel)
        top_splitter.setStretchFactor(0, 4)
        top_splitter.setStretchFactor(1, 1)

        main_splitter.addWidget(top_splitter)

        # Bottom: Log
        self.log_panel = LogPanel()
        main_splitter.addWidget(self.log_panel)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)

        self.setCentralWidget(main_splitter)

    def _on_tab_changed(self, idx):
        tab_name = self.tab_widget.tabText(idx)
        if 'Auto' in tab_name or 'AI' in tab_name:
            # Switch to auto/ai tab — do nothing, user presses Start
            pass
        else:
            # Switch away from auto/ai — stop auto if running
            self._bridge.send_command({'type': 'stop_auto'})
            self._bridge.send_command({'type': 'stop_ai'})

    def _setup_statusbar(self):
        self.statusBar()
        self.lbl_conn = QLabel("● Connected")
        self.lbl_conn.setStyleSheet("color: #00ff88; font-weight: bold; margin-left: 10px;")
        
        self.lbl_ws = QLabel("⬤ Workspace: OK")
        self.lbl_ws.setStyleSheet("color: #00ff88; font-weight: bold; margin-left: 8px;")
        
        self.lbl_time = QLabel("Last update: --:--:--")
        self.lbl_time.setStyleSheet("margin-right: 10px;")
        
        self.statusBar().addWidget(self.lbl_conn)
        self.statusBar().addWidget(self.lbl_ws, 1)
        self.statusBar().addPermanentWidget(self.lbl_time)
        
    def _connect_signals(self):
        self.joint_panel.joints_changed.connect(self._on_joints_changed)
        self.cartesian_panel.cartesian_go.connect(self._on_cartesian_go)
        self.cartesian_panel.cartesian_jog.connect(self._on_cartesian_jog)
        self.traj_panel.traj_requested.connect(self._on_traj_requested)
        self.traj_panel.traj_stop.connect(lambda: self._bridge.send_command({'type': 'stop_traj'}))
        self.traj_panel.speed_changed.connect(
            lambda s: self._bridge.send_command({'type': 'set_speed', 'speed_scale': s}))
        # Auto panel
        self.auto_panel.auto_start.connect(self._on_auto_start)
        self.auto_panel.auto_stop.connect(
            lambda: self._bridge.send_command({'type': 'stop_auto'}))
        self.auto_panel.reset_error.connect(
            lambda: self._bridge.send_command({'type': 'reset_error'}))
        
        # AI panel
        self.ai_panel.ai_start.connect(lambda: self._bridge.send_command({'type': 'start_ai'}))
        self.ai_panel.ai_stop.connect(lambda: self._bridge.send_command({'type': 'stop_ai'}))
        
    def _on_joints_changed(self, q):
        if self._estop: return
        # Workspace pre-check
        fk_res = forward_kinematics(q)
        w_pos = local_to_world(fk_res['position'])
        ok, reason = self._validator.is_valid_ee(w_pos)
        if not ok:
            self.log_panel.log(f"Blocked: {reason}", 'WARN')
            return
        self._bridge.send_command({'type': 'set_joints', 'q': q})
        self.log_panel.log(f"Set joints: {[f'{v:.2f}' for v in q]}", 'CMD')
        
    def _on_cartesian_go(self, pos, euler):
        if self._estop: return
        self._bridge.send_command({
            'type': 'set_cartesian',
            'pos': pos, 'euler': euler
        })
        self.log_panel.log(f"Go to XYZ={[f'{v:.3f}' for v in pos]}", 'CMD')
        
    def _on_cartesian_jog(self, axis, step):
        if self._estop: return
        self._bridge.send_command({'type': 'jog_cartesian', 'axis': axis, 'step': step})
        self.log_panel.log(f"Jog {axis} {step:+.3f}m", 'CMD')

    def _on_traj_requested(self, traj_type, target, speed):
        if self._estop: return
        if traj_type == 'joint':
            self._bridge.send_command({
                'type': 'run_joint_traj',
                'q_end': target,
                'speed_scale': speed
            })
            self.log_panel.log(f"Joint traj -> {[f'{v:.2f}' for v in target]}", 'CMD')
        elif traj_type == 'cartesian':
            self._bridge.send_command({
                'type': 'run_cartesian_traj',
                'pos_end': target[:3],
                'euler_end': target[3:],
                'speed_scale': speed
            })
            self.log_panel.log(f"Cart traj -> XYZ={[f'{v:.3f}' for v in target[:3]]}", 'CMD')

    def _on_auto_start(self, repeat: bool):
        if self._estop: return
        self._bridge.send_command({'type': 'start_auto', 'auto_repeat': repeat})
        mode = 'repeat' if repeat else '1 cycle'
        self.log_panel.log(f"Auto mode started ({mode})", 'CMD')
        
    def _refresh_ui(self):
        state = self._bridge.get_state()
        if state is None: return
        
        self.joint_panel.update_from_state(state['q'])
        self.cartesian_panel.update_ee_display(state['ee_pos'], state['ee_euler'])
        self.status_panel.update_state(state)
        
        # Trajectory panel
        self.traj_panel.update_progress(
            state.get('traj_running', False),
            state.get('traj_progress', 0.0)
        )
        self.traj_panel.set_current_state(state['q'], state['ee_pos'], state['ee_euler'])
        
        if state['estop'] != self._estop_state:
            self._estop_state = state['estop']
            self.log_panel.log("E-Stop state changed", 'WARN')

        # Forward bridge logs
        for entry in state.get('logs', []):
            self.log_panel.log(entry['msg'], entry.get('level', 'INFO'))

        # Auto panel
        self.auto_panel.update_state({
            'state':     state.get('sm_state', 'IDLE'),
            'cycle':     state.get('sm_cycle', 0),
            'gripper':   state.get('gripper', False),
            'mode':      state.get('mode', 'manual'),
            'error_msg': state.get('sm_error', ''),
        })

        # AI panel
        self.ai_panel.update_state({
            'mode': state.get('mode', 'manual'),
            'ai_loaded': state.get('ai_loaded', False),
            'ai_success_count': state.get('ai_success_count', 0)
        })

        # Workspace indicator
        ws_ok = state.get('workspace_ok', True)
        ee_pos = state['ee_pos']
        near = self._validator.is_near_limit(ee_pos)
        if not ws_ok:
            self.lbl_ws.setText("⬤ Workspace: VIOLATED")
            self.lbl_ws.setStyleSheet("color: #ff3333; font-weight: bold; margin-left: 8px;")
        elif near:
            self.lbl_ws.setText("⬤ Workspace: Near limit")
            self.lbl_ws.setStyleSheet("color: #ffcc00; font-weight: bold; margin-left: 8px;")
        else:
            self.lbl_ws.setText("⬤ Workspace: OK")
            self.lbl_ws.setStyleSheet("color: #00ff88; font-weight: bold; margin-left: 8px;")
            
        current_time = datetime.now().strftime("%H:%M:%S")
        self.lbl_time.setText(f"Last update: {current_time}")
        self.status_panel.set_connected(True)
