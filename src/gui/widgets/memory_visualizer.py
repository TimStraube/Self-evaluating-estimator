from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np

class MemoryVisualizer(QtWidgets.QWidget):
    def __init__(self, gedächtnis):
        super().__init__()
        self.gedächtnis = gedächtnis
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Memory Visualizer')
        self.setMinimumSize(400, 300)

        self.layout = QtWidgets.QVBoxLayout()

        self.memoryList = QtWidgets.QListWidget()
        self.layout.addWidget(self.memoryList)

        self.updateButton = QtWidgets.QPushButton('Update Memory')
        self.updateButton.clicked.connect(self.update_memory_display)
        self.layout.addWidget(self.updateButton)

        self.setLayout(self.layout)
        self.update_memory_display()

    def update_memory_display(self):
        self.memoryList.clear()
        for i in range(self.gedächtnis.getKapazität()):
            try:
                belohnung = self.gedächtnis.getReward(i)
                self.memoryList.addItem(f'Erinnerung {i}: Belohnung = {belohnung:.4f}')
            except Exception as e:
                self.memoryList.addItem(f'Erinnerung {i}: Fehler beim Laden - {str(e)}')