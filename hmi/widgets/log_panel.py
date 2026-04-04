import sys
import datetime
from PyQt5.QtWidgets import (QApplication, QGroupBox, QVBoxLayout, 
                             QTextEdit, QPushButton, QHBoxLayout)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt

class LogPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Command Log", parent)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 10, 5, 5)
        
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        # Font monospace
        font = QFont("Consolas", 9)
        font.setStyleHint(QFont.Monospace)
        self.text_edit.setFont(font)
        
        # Style
        self.text_edit.setStyleSheet("background-color: #1a1a1a; color: #ffffff; border: 1px solid #333333;")
        self.layout.addWidget(self.text_edit)
        
        # Button layout
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.clear_btn = QPushButton("🗑 Clear", self)
        self.clear_btn.clicked.connect(self.clear)
        btn_layout.addWidget(self.clear_btn)
        self.layout.addLayout(btn_layout)
        
        self.max_lines = 200
        self.setMinimumHeight(150)
        
        self._colors = {
            'INFO': '#ffffff',
            'CMD': '#00ccff',
            'IK': '#00ff88',
            'WARN': '#ffcc00',
            'ESTOP': '#ff4444'
        }

    def log(self, message: str, level: str = 'INFO'):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        color = self._colors.get(level, '#ffffff')
        
        html_msg = f'<span style="color: {color};">[{timestamp}] [{level}] {message}</span>'
        self.text_edit.append(html_msg)
        
        # Auto-scroll
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # Trim history
        doc = self.text_edit.document()
        if doc.blockCount() > self.max_lines:
            cursor = self.text_edit.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar() # remove newline
            
    def clear(self):
        self.text_edit.clear()

if __name__ == "__main__":
    import time
    app = QApplication(sys.argv)
    
    panel = LogPanel()
    panel.show()
    
    panel.log("Về home pose", "INFO")
    panel.log("Set joints: [0.00, -1.57, 1.57, -1.57, -1.57, 0.00]", "CMD")
    panel.log("Cartesian → 4 solutions found", "IK")
    panel.log("Joint 3 clamped to limit", "WARN")
    panel.log("Emergency stop activated", "ESTOP")
    
    # Hide after 3 seconds for automated testing 
    import threading
    def auto_close():
        time.sleep(3)
        app.quit()
    threading.Thread(target=auto_close, daemon=True).start()
    
    sys.exit(app.exec_())
