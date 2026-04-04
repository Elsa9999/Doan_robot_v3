import sys
import time
from threading import Thread
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QColor, QPainter, QFont

from hmi.sim_bridge import SimBridge
from hmi.main_window import MainWindow


def _make_splash_pixmap():
    """Tạo Splash Screen đẹp bằng QPainter."""
    w, h = 480, 200
    pix = QPixmap(w, h)
    pix.fill(QColor("#1e1e1e"))
    painter = QPainter(pix)

    font_title = QFont("Segoe UI", 18, QFont.Bold)
    painter.setFont(font_title)
    painter.setPen(QColor("#00aaff"))
    painter.drawText(0, 60, w, 50, Qt.AlignCenter, "UR5e Robot Controller")

    font_sub = QFont("Segoe UI", 10)
    painter.setFont(font_sub)
    painter.setPen(QColor("#aaaaaa"))
    painter.drawText(0, 110, w, 40, Qt.AlignCenter, "Đang khởi động simulation...")

    painter.end()
    return pix


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # ── Splash Screen ──────────────────────────────────────────
    from PyQt5.QtWidgets import QSplashScreen
    splash = QSplashScreen(_make_splash_pixmap())
    splash.show()
    app.processEvents()

    # ── Khởi động SimBridge (PyBullet) trong background thread ─
    bridge = SimBridge()
    sim_thread = Thread(target=bridge.start, kwargs={'gui': True}, daemon=True)
    sim_thread.start()

    # ── Chờ bridge sẵn sàng (max 10 giây) ─────────────────────
    timeout = time.time() + 10
    while not bridge.is_ready() and time.time() < timeout:
        time.sleep(0.1)
        app.processEvents()

    if bridge.is_ready():
        splash.showMessage("  Simulation sẵn sàng!", Qt.AlignBottom | Qt.AlignLeft, QColor("#00ff88"))
    else:
        print("[APP] Bridge timeout — GUI sẽ chạy không có sim")
        splash.showMessage("  Khởi động GUI (không có sim)...", Qt.AlignBottom | Qt.AlignLeft, QColor("#ffcc00"))

    app.processEvents()
    time.sleep(0.3)
    splash.close()

    # ── Tạo và hiện MainWindow ─────────────────────────────────
    window = MainWindow(bridge)
    window.show()

    # ── Dark title bar trên Windows 10/11 ──────────────────────
    try:
        from ctypes import windll, c_int, byref, sizeof
        HWND = int(window.winId())
        windll.dwmapi.DwmSetWindowAttribute(HWND, 20, byref(c_int(1)), sizeof(c_int))
    except Exception:
        pass

    print("[APP] MainWindow đã hiển thị. Vào event loop...")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
