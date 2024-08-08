import sys
import os
import asyncio
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QMessageBox, QScrollArea
from PyQt5.QtCore import QThread, pyqtSignal
import subprocess

# Set the event loop policy to WindowsSelectorEventLoopPolicy
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Base directory path
BASE_DIR = r"C:\Users\bovta\Desktop\Aaron (Intern)\Aaron (Intern)\VS code Stuff\Final UI"

# Specify the full paths to your Python files here
SCRIPT_PATHS = {
    "IndividualChannel UI": os.path.join(BASE_DIR, "IndividualChannel UI.py"),
    "MultichannelCombined": os.path.join(BASE_DIR, "MultichannelCombined.py"),
    "Single Channel Sampling (Untied)": os.path.join(BASE_DIR, "single channel sampling (untied).py"),
    "Single Channel Sampling (Untied and Multichannel)": os.path.join(BASE_DIR, "single channel sampling (untied and multichannel sampling allowed).py"),
    "Optical Power Oscilloscope UI": os.path.join(BASE_DIR, "optical power oscilloscope UI.py")
}

class PythonScriptExecutor(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, script_path):
        super().__init__()
        self.script_path = script_path

    def run(self):
        try:
            subprocess.Popen([sys.executable, self.script_path], 
                             creationflags=subprocess.CREATE_NEW_CONSOLE)
            self.finished.emit(os.path.basename(self.script_path))
        except Exception as e:
            self.error.emit(str(e))

class DAQmxLauncher(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Create a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        for script_name, script_path in SCRIPT_PATHS.items():
            btn = QPushButton(script_name, self)
            btn.clicked.connect(lambda checked, path=script_path: self.launch_script(path))
            scroll_layout.addWidget(btn)

        scroll_content.setLayout(scroll_layout)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        self.setLayout(layout)
        self.setGeometry(300, 300, 400, 300)
        self.setWindowTitle('DAQmx Interface Launcher')
        self.show()

    def launch_script(self, script_path):
        if not os.path.exists(script_path):
            QMessageBox.critical(self, "Error", f"The file {script_path} does not exist.")
            return

        self.executor = PythonScriptExecutor(script_path)
        self.executor.finished.connect(self.on_script_finished)
        self.executor.error.connect(self.on_script_error)
        self.executor.start()

        QMessageBox.information(self, "Launching", f"Launching {os.path.basename(script_path)}. This may take a moment...")

    def on_script_finished(self, script_name):
        QMessageBox.information(self, "Success", f"Launched {script_name} successfully")

    def on_script_error(self, error_message):
        QMessageBox.critical(self, "Error", f"Failed to launch script: {error_message}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DAQmxLauncher()
    sys.exit(app.exec_())
