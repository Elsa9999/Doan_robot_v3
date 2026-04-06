from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QGroupBox, QGridLayout
from PyQt5.QtCore import pyqtSignal

class AIPanel(QWidget):
    ai_start = pyqtSignal()
    ai_stop  = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.lbl_title = QLabel("🤖 Trí Tuệ Nhân Tạo (SAC Model)")
        self.lbl_title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #ff00ff; padding-bottom: 10px;")
        self.layout.addWidget(self.lbl_title)
        
        self.btn_start = QPushButton("🚀 TỰ ĐỘNG LÀM VIỆC (AI)")
        self.btn_start.setMinimumHeight(50)
        self.btn_start.setStyleSheet("background-color: #aa00ff; color: white; font-weight: bold;")
        self.btn_start.clicked.connect(self.ai_start.emit)
        
        self.btn_stop = QPushButton("🛑 TẠM DỪNG MÔ HÌNH")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setStyleSheet("background-color: #555555;")
        self.btn_stop.clicked.connect(self.ai_stop.emit)
        self.btn_stop.setEnabled(False)
        
        self.layout.addWidget(self.btn_start)
        self.layout.addWidget(self.btn_stop)
        
        grp_status = QGroupBox("Tình trạng Huấn Luyện")
        g_layout = QGridLayout()
        self.lbl_model_status = QLabel("❌ Model chưa được nạp")
        self.lbl_model_status.setStyleSheet("color: red;")
        self.lbl_success = QLabel("📦 Số lượng thả vào thùng: 0")
        self.lbl_success.setStyleSheet("color: #00ff88;")
        
        g_layout.addWidget(self.lbl_model_status, 0, 0)
        g_layout.addWidget(self.lbl_success, 1, 0)
        grp_status.setLayout(g_layout)
        self.layout.addWidget(grp_status)
        self.layout.addStretch()
        
    def update_state(self, state):
        mode = state.get('mode', 'manual')
        if mode == 'ai':
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_stop.setStyleSheet("background-color: #ff3333; font-weight: bold;")
        else:
            self.btn_start.setEnabled(state.get('ai_loaded', False))
            self.btn_stop.setEnabled(False)
            self.btn_stop.setStyleSheet("background-color: #555555;")
            
        if state.get('ai_loaded'):
            self.lbl_model_status.setText("✅ SAC Model (100% Success) Đã Load!")
            self.lbl_model_status.setStyleSheet("color: #00ff88; font-weight: bold;")
        else:
            self.lbl_model_status.setText("❌ Không tìm thấy Model")
            self.lbl_model_status.setStyleSheet("color: red;")
            
        self.lbl_success.setText(f"📦 Số vật đã thu gom: {state.get('ai_success_count', 0)}")
