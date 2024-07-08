import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QMessageBox
import subprocess

class DAQmxLauncher(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        btn_original = QPushButton('Launch Original UI', self)
        btn_original.clicked.connect(self.launch_original)
        layout.addWidget(btn_original)

        btn_multichannel = QPushButton('Launch Multichannel UI', self)
        btn_multichannel.clicked.connect(self.launch_multichannel)
        layout.addWidget(btn_multichannel)

        self.setLayout(layout)
        self.setGeometry(300, 300, 300, 150)
        self.setWindowTitle('DAQmx Interface Launcher')
        self.show()

    def launch_original(self):
        try:
            subprocess.Popen([sys.executable, 'UI.py'])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch Original UI: {str(e)}")

    def launch_multichannel(self):
        try:
            subprocess.Popen([sys.executable, 'MultichannelCombined.py'])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch Multichannel UI: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DAQmxLauncher()
    sys.exit(app.exec_())