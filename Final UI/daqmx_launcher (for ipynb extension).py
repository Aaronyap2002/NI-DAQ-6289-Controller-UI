import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QMessageBox
import subprocess
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

class DAQmxLauncher(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        btn_original = QPushButton('Launch Original UI', self)
        btn_original.clicked.connect(lambda: self.launch_notebook('UI.ipynb'))
        layout.addWidget(btn_original)

        btn_multichannel = QPushButton('Launch Multichannel UI', self)
        btn_multichannel.clicked.connect(lambda: self.launch_notebook('MultichannelCombined.ipynb'))
        layout.addWidget(btn_multichannel)

        self.setLayout(layout)
        self.setGeometry(300, 300, 300, 150)
        self.setWindowTitle('DAQmx Interface Launcher')
        self.show()

    def launch_notebook(self, notebook_path):
        try:
            with open(notebook_path) as f:
                nb = nbformat.read(f, as_version=4)
            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
            ep.preprocess(nb, {'metadata': {'path': '.'}})
            
            QMessageBox.information(self, "Success", f"Launched {notebook_path} successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch {notebook_path}: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DAQmxLauncher()
    sys.exit(app.exec_())
