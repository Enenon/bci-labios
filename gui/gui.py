from keras.models import load_model
from tensorflow.keras.optimizers import Adam
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import sys
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np
import keyboard  # para detectar ESC
import time
from pylsl import StreamInlet, resolve_stream, resolve_byprop
from time import sleep
import matplotlib.pyplot as plt
from random import choice


class JanelaInicial(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(672, 446)
        self.setWindowTitle('BCI Labios')
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)


        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(429, 0, 231, 401))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")

        self.Layout_visualizacao = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.Layout_visualizacao.setContentsMargins(0, 0, 0, 0)
        self.Layout_visualizacao.setObjectName("Layout_visualizacao")

        # aqui ficará o gráfico da serie temporal
        self.frameGrafico = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.frameGrafico.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameGrafico.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameGrafico.setObjectName("frameGrafico")

        self.Layout_visualizacao.addWidget(self.frameGrafico)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.saida = QtWidgets.QLabel('Saída:', self.verticalLayoutWidget)
        self.saida.setObjectName("saida")
        self.horizontalLayout.addWidget(self.saida)
        self.label_4 = QtWidgets.QLabel('0', self.verticalLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout.addWidget(self.label_4)
        self.Layout_visualizacao.addLayout(self.horizontalLayout)
        
        # aqui ficará a imagem do cérebro
        self.frameCerebro = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.frameCerebro.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameCerebro.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameCerebro.setObjectName("frameCerebro")
        self.Layout_visualizacao.addWidget(self.frameCerebro)

        # Widget com status # fazer os outros widgets ficarem como esse aqui
        self.formLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(50, 30, 223, 80))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.label_3 = QtWidgets.QLabel('Status do modelo: ', self.formLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.label_5 = QtWidgets.QLabel('Nenhum modelo carregado', self.formLayoutWidget)
        self.label_5.setObjectName("label_5")
        self.label_5.setWordWrap(True) # habilita quebra automática de linha
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.label_5)
        self.label = QtWidgets.QLabel('Status do LSL: ', self.formLayoutWidget)
        self.label.setGeometry(QtCore.QRect(40, 100, 80, 16))
        self.label.setObjectName("label_status")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label)
        self.label_2 = QtWidgets.QLabel('Desconectado', self.formLayoutWidget)
        self.label_2.setGeometry(115,100,90,16)
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.label_2)

        # widget que guarda os botões
        self.buttonLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.buttonLayoutWidget.setGeometry(QtCore.QRect(50, 120, 223, 60))
        self.buttonLayoutWidget.setObjectName("buttonLayoutWidget")
        self.buttonLayout = QtWidgets.QVBoxLayout(self.buttonLayoutWidget)
        self.buttonLayout.setContentsMargins(0, 0, 0, 0)

        self.button = QtWidgets.QPushButton('Conectar LSL', self.buttonLayoutWidget)
        self.button.setObjectName("Conectar LSL")
        self.buttonLayout.addWidget(self.button)

        self.button_iniciarBCI = QtWidgets.QPushButton('Iniciar BCI', self.buttonLayoutWidget)
        self.buttonLayout.addWidget(self.button_iniciarBCI)

        # Paleta vermelha para 'Desconectado'
        self.palette_vermelha = QtGui.QPalette()
        brush_vermelho = QtGui.QBrush(QtGui.QColor(255, 0, 4))
        self.palette_vermelha.setBrush(QtGui.QPalette.All, QtGui.QPalette.WindowText, brush_vermelho)
        
        # Paleta amarela para 'Procurando...'
        self.palette_amarela = QtGui.QPalette()
        brush_amarelo = QtGui.QBrush(QtGui.QColor(150, 150, 0))
        self.palette_amarela.setBrush(QtGui.QPalette.All, QtGui.QPalette.WindowText, brush_amarelo)

        # Paleta verde para 'Conectado'
        self.palette_verde = QtGui.QPalette()
        brush_verde = QtGui.QBrush(QtGui.QColor(4, 150, 0))
        self.palette_verde.setBrush(QtGui.QPalette.All, QtGui.QPalette.WindowText, brush_verde)
        self.label_2.setPalette(self.palette_vermelha)

        
        self.setStatusBar(QtWidgets.QStatusBar(self))

        self.menubar = self.menuBar()

        self.menubar.arquivo = self.menubar.addMenu("Arquivo")
        self.menubar.modelo = self.menubar.arquivo.addMenu("Modelo")
        self.menubar.addmodelo = self.menubar.modelo.addAction("Importar modelo")
        self.treinar_modelo = self.menubar.addAction("Treinar modelo")
        
        self.menubar.addmodelo.triggered.connect(self.abrir_modelo)
        self.treinar_modelo.triggered.connect(self.abrir_janela_teste)
        self.button.clicked.connect(self.conectar_LSL)
        self.button_iniciarBCI.clicked.connect(self.abrir_janela_teste)
        QtCore.QMetaObject.connectSlotsByName(self)
        #self.show()
    

    def abrir_janela_teste(self):
        self.janela_teste = JanelaTeste()

    def abrir_modelo(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 
   '../',"Model files (*.h5)")
        print(fname)
        try:
            model = load_model(r"C:\Users\Enenon\Documents\GitHub\bci-labios\modelos\modelo rede c.h5")
            model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
            self.label_5.setText('shape:'+str(model.input_shape)+'\noutput shape:'+str(model.output_shape)+'\nmetric names:'+
                                 str(model.metrics_names))
            self.label_5.setPalette(self.palette_verde)
        except:
            self.label_5.setText('Modelo incompatível.')
            self.label_5.setPalette(self.palette_vermelha)
        #model.summary()

    def conectar_LSL(self):
        print("Aguardando stream EEG...")
        print('mudando a cor da paleta')

        self.label_2.setText('Procurando...')
        self.label_2.setPalette(self.palette_amarela)
        QApplication.processEvents() # <--- isso aplica as mudanças antes da função acabar
        self.streams = resolve_byprop('type', 'EEG',timeout=3)
        if self.streams:
            palette = QtGui.QPalette()
            palette.setBrush(QtGui.QPalette.All, QtGui.QPalette.WindowText,QtGui.QBrush(QtGui.QColor(4,150,0)))
            self.label_2.setText('Conectado!')
            self.label_2.setPalette(self.palette_verde)
        else:
            print('Não achou conexão')
            self.label_2.setPalette(self.palette_vermelha)
            self.label_2.setText('Desconectado')
            
        
class JanelaTeste(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(200,200,250,50)
        self.setWindowTitle('Teste')
        self.label = QtWidgets.QLabel("Recurso indisponível!",self)
        self.label.setGeometry(QtCore.QRect(1,8,100,23))
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.mudar_cor)
        self.timer.start(50)
        self.t = 0
        self.palette = QtGui.QPalette()
        font = self.label.font()
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.adjustSize()
        self.show()

    def mudar_cor(self):
        brush = QtGui.QBrush(QtGui.QColor(int(255*abs(np.sin(self.t))), 0, int(255*abs(np.cos(self.t)))))
        self.palette.setBrush(QtGui.QPalette.All, QtGui.QPalette.WindowText, brush)
        self.t += self.timer.interval()/1000
        if self.t > np.pi: self.t = 0
        self.label.setPalette(self.palette)
        




class Janela1(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(200,200,300,300)
        self.setWindowTitle('Janela 1')

        self.button = QtWidgets.QPushButton('Clique aqui',self)
        self.button.move(50,50)
        self.button.clicked.connect(self.open_new_window)
        self.show()

    def open_new_window(self):
        self.second_window = Janela2()

class Janela2(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(300,300,350,350)
        self.setWindowTitle('Janela 2')

        widget1 = QtWidgets.QWidget()
        self.setCentralWidget(widget1)
        layout = QtWidgets.QVBoxLayout(widget1)

        self.canvas = FigureCanvas(Figure(figsize=(5,4)))
        layout.addWidget(self.canvas)

        self.ax = self.canvas.figure.add_subplot(111)
        self.t = np.linspace(0, 10, 100)
        self.ax.plot(self.t, np.sin(self.t))
        self.ax.set_title('Exemplo de Gráfico')
        self.ax.set_xlabel('Tempo')
        self.ax.set_ylabel('Amplitude')

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1)

        self.show()

    def update_plot(self):
        self.t += self.timer.interval()/100
        self.ax.clear()
        self.ax.plot(self.t,np.sin(self.t))
        self.canvas.draw()

def window():
    app = QApplication(sys.argv)
    win = JanelaInicial()
    win.show()

    sys.exit(app.exec_())

window()