import sys
import cv2
import numpy as np
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
import GraphMaker
from WeightCalculation import *

class UI:

    def __init__(self):
        self.graph_maker = GraphMaker.GraphMaker(self.open_dialogue_box)
        self.a = QApplication(sys.argv)
        self.window = QMainWindow()
        self.window.setWindowTitle("ImageSegmentation")
        self.seed_num = self.graph_maker.foreground

        mainMenu = self.window.menuBar()
        fileMenu = mainMenu.addMenu('&File')

        openButton = QAction(QIcon('exit24.png'), 'Open Image', self.window)
        openButton.setShortcut('Ctrl+O')
        openButton.setStatusTip('Open a file for segmenting.')
        openButton.triggered.connect(self.on_open)
        fileMenu.addAction(openButton)

        saveButton = QAction(QIcon('exit24.png'), 'Save Image', self.window)
        saveButton.setShortcut('Ctrl+S')
        saveButton.setStatusTip('Save file to disk.')
        saveButton.triggered.connect(self.on_save)
        fileMenu.addAction(saveButton)

        closeButton = QAction(QIcon('exit24.png'), 'Exit', self.window)
        closeButton.setShortcut('Ctrl+Q')
        closeButton.setStatusTip('Exit application')
        closeButton.triggered.connect(self.on_close)
        fileMenu.addAction(closeButton)

        mainWidget = QWidget()
        mainBox = QVBoxLayout()

        buttonLayout = QHBoxLayout()
        self.foregroundButton = QPushButton('Add Foreground Seeds')
        self.foregroundButton.clicked.connect(self.on_foreground)
        self.foregroundButton.setStyleSheet("background-color: #545354")

        self.backGroundButton = QPushButton('Add Background Seeds')
        self.backGroundButton.clicked.connect(self.on_background)
        self.backGroundButton.setStyleSheet("background-color: gray")

        clearButton = QPushButton('Clear All Seeds')
        clearButton.clicked.connect(self.on_clear)

        segmentButton = QPushButton('Segment Image')
        segmentButton.clicked.connect(self.on_segment)

        onlyForegroundButton = QPushButton('Only Foreground')
        onlyForegroundButton.clicked.connect(self.on_only_foreground)

        buttonLayout.addWidget(self.foregroundButton)
        buttonLayout.addWidget(self.backGroundButton)
        buttonLayout.addWidget(clearButton)
        buttonLayout.addWidget(segmentButton)
        buttonLayout.addWidget(onlyForegroundButton)
        buttonLayout.addStretch()

        mainBox.addLayout(buttonLayout)

        self.stateLayout = QHBoxLayout()
        self.states = {}
        seedState = 'Foreground'
        seedStateLabel = QLabel(f'Marking seeds for: {seedState}')
        self.states['seed'] = seedStateLabel
        self.stateLayout.addWidget(seedStateLabel)
        hoLayout = QHBoxLayout()
        functionLabel = QLabel('Weight Function to be used')
        hoLayout.addWidget(functionLabel)
        functionComboBox = QComboBox()
        self.weight_fns = [
            Default(),
            Paramless()
        ]
        functionComboBox.addItems([v.get_name() for v in self.weight_fns])
        functionComboBox.currentIndexChanged.connect(self.functionSet)
        hoLayout.addWidget(functionComboBox)
        self.states['function'] = functionComboBox
        self.stateLayout.addLayout(hoLayout)


        if self.graph_maker.parameters is not None:
            self.paramLayout = QHBoxLayout()
            for k, v in self.graph_maker.parameters.items():
                textLayout = QLabel(f'{k}: {v}')
                self.paramLayout.addWidget(textLayout)
            self.stateLayout.addLayout(self.paramLayout)

        mainBox.addLayout(self.stateLayout)

        imageLayout = QHBoxLayout()

        self.seedLabel = QLabel()
        self.seedLabel.mousePressEvent = self.mouse_down
        self.seedLabel.mouseMoveEvent = self.mouse_drag

        self.segmentLabel = QLabel()

        imageLayout.addWidget(self.seedLabel)
        imageLayout.addWidget(self.segmentLabel)
        imageLayout.addStretch()
        mainBox.addLayout(imageLayout)
        mainBox.addStretch()
        mainWidget.setLayout(mainBox)
        self.window.setCentralWidget(mainWidget)

    def run(self):
        self.window.show()
        sys.exit(self.a.exec())
    
    def addStateLayout(self):
        if self.graph_maker.parameters is not None:
            self.paramLayout = QHBoxLayout()
            for k, v in self.graph_maker.parameters.items():
                textLayout = QLabel(f'{k}: {v}')
                self.paramLayout.addWidget(textLayout)
            self.stateLayout.addLayout(self.paramLayout)

    def removeStateLayout(self):
        while self.paramLayout.count():
            child = self.paramLayout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self.stateLayout.removeItem(self.paramLayout)

    def functionSet(self, v):
        self.graph_maker.set_parameters(None)
        self.removeStateLayout()
        self.graph_maker.weight_function_code = self.weight_fns[v]

    def open_dialogue_box(self):
        class CustomDialog(QDialog):
            def __init__(self, parent=self):
                super().__init__()

                self.setWindowTitle("Provide Parameters")
                self.parent = parent
                QBtn = QDialogButtonBox.StandardButton.Ok

                self.buttonBox = QDialogButtonBox(QBtn)
                self.buttonBox.accepted.connect(self.__set_params)
                self.buttonBox.rejected.connect(self.reject)

                self.layout = QVBoxLayout()
                message = QLabel("Provide parameters for graph-cut")

                formLayout = QGridLayout()
                labels = {}
                self.lineEdits = {}

                labels['K'] = QLabel('K (kappa value) similar pixels have weight close to kappa')
                labels['s'] = QLabel('s (Sigma value) determines how fast the values decay towards zero with increasing dissimilarity.')

                self.lineEdits['K'] = QLineEdit()
                self.lineEdits['s'] = QLineEdit()

                formLayout.addWidget(labels['K'], 0, 0, 1, 1)
                formLayout.addWidget(self.lineEdits['K'], 0, 1, 1, 3)

                formLayout.addWidget(labels['s'], 1, 0, 1, 1)
                formLayout.addWidget(self.lineEdits['s'], 1, 1, 1, 3)

                self.layout.addWidget(message)
                self.layout.addLayout(formLayout)
                self.layout.addWidget(self.buttonBox)
                self.setLayout(self.layout)
            
            def __set_params(self):
                self.parent.graph_maker.set_parameters({
                    'K': int(self.lineEdits['K'].text()),
                    's': int(self.lineEdits['s'].text())
                })
                self.parent.addStateLayout()
                self.accept()
        dlg = CustomDialog()
        if dlg.exec():
            print("Success!")
        else:
            print("Cancel!")
        return self.graph_maker.parameters

    @staticmethod
    def get_qimage(cvimage):
        height, width, bytes_per_pix = cvimage.shape
        bytes_per_line = width * bytes_per_pix
        cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB, cvimage)
        return QImage(cvimage.data, width, height, QImage.Format.Format_RGB888)

    def on_foreground(self):
        self.seed_num = self.graph_maker.foreground
        self.foregroundButton.setStyleSheet("background-color: #545354")
        self.backGroundButton.setStyleSheet("background-color: gray")
        self.states['seed'].setText('Marking seeds for: Foreground')

    def on_background(self):
        self.seed_num = self.graph_maker.background
        self.foregroundButton.setStyleSheet("background-color: gray")
        self.backGroundButton.setStyleSheet("background-color: #545354")
        self.states['seed'].setText('Marking seeds for: Background')

    def on_clear(self):
        self.graph_maker.clear_seeds()
        self.seedLabel.setPixmap(QPixmap.fromImage(
                self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.seeds))))

    def on_segment(self):
        self.graph_maker.create_graph()
        self.segmentLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.segmented))))

    def on_only_foreground(self):
        self.segmentLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.graph_maker.get_image_with_only_overlay())))


    def on_open(self):
        f = QFileDialog.getOpenFileName()
        if f is not None and f != "":
            self.graph_maker.load_image(str(f))
            self.seedLabel.setPixmap(QPixmap.fromImage(
                self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.seeds))))
            self.segmentLabel.setPixmap(QPixmap.fromImage(
                self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.segmented))))

    def on_save(self):
        f = QFileDialog.getSaveFileName()
        print('Saving')
        if f is not None and f != "":
            self.graph_maker.save_image(f)

    def on_close(self):
        self.window.close()

    def mouse_down(self, event):
        self.graph_maker.add_seed(event.position().x(), event.position().y(), self.seed_num)
        self.seedLabel.setPixmap(QPixmap.fromImage(
                self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.seeds))))

    def mouse_drag(self, event):
        self.graph_maker.add_seed(event.position().x(), event.position().y(), self.seed_num)
        self.seedLabel.setPixmap(QPixmap.fromImage(
                self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.seeds))))

