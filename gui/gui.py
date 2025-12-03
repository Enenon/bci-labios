from keras.models import load_model
from tensorflow.keras.optimizers import Adam
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QLabel, QSizePolicy
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
import mne

class JanelaInicial(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(672, 446)
        self.setWindowTitle('BCI Labios')
        # --- Configurações de Dados ---
        self.canais = ['C3..', 'C4..', 'Fp1.', 'Fp2.', 'F7..', 'F3..', 'F4..', 'F8..','T7..', 'T8..', 'P7..', 'P3..', 'P4..', 'P8..', 'O1..', 'O2..']
        self.n_channels = len(self.canais)
        self.x_size = 1000
        self.epochsize = 721
        self.current_data = np.zeros((self.x_size, self.n_channels))
        self.limiar = 0.4

        # Setup da UI com layouts (coluna esquerda = controles, coluna direita = visualização expansível)
        self.centralwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralwidget)
        self.main_layout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(8, 8, 8, 8)

        # coluna esquerda (controles) - não expande
        self.left_widget = QtWidgets.QWidget(self.centralwidget)
        self.left_layout = QtWidgets.QVBoxLayout(self.left_widget)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.left_widget, 0)

        # coluna direita (visualização) - expande com a janela
        self.visual_widget = QtWidgets.QWidget(self.centralwidget)
        self.Layout_visualizacao = QtWidgets.QVBoxLayout(self.visual_widget)
        self.Layout_visualizacao.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.visual_widget, 1)
 
        # --- CONFIGURAÇÃO OTIMIZADA DO MATPLOTLIB ---
        self.frameGrafico = QtWidgets.QFrame(self.visual_widget)
        self.frameGrafico_layout = QtWidgets.QVBoxLayout(self.frameGrafico)
        
        # Criamos a figura uma única vez
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Configuração estética do gráfico
        self.ax.set_title('Série Temporal EEG (Real-time)')
        self.ax.set_xlabel('Amostras')
        self.ax.set_yticks([]) # Remove eixo Y numérico para limpar
        self.ax.set_xlim(0, self.x_size)
        # Ajusta limite Y para caber todos os canais empilhados (waterfall)
        self.escala_visual = 750 # Fator para separar as linhas visualmente
        self.ax.set_ylim(-self.escala_visual, self.n_channels * self.escala_visual + self.escala_visual)
        
        # CRUCIAL: Criamos as linhas (artistas) vazias agora e guardamos as referências
        self.lines = []
        for i in range(self.n_channels):
           # Plotamos uma linha vazia para cada canal
           line, = self.ax.plot([], [], lw=1) 
           self.lines.append(line)

        self.frameGrafico_layout.addWidget(self.canvas)
        # permitir expansão do canvas/frame
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.frameGrafico.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.Layout_visualizacao.addWidget(self.frameGrafico)
 
        self.ax.set_title('Série Temporal EEG')
        self.ax.set_xlabel('Tempo (s)')
        self.frameGrafico.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameGrafico.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameGrafico.setObjectName("frameGrafico")
 
        # área de saída abaixo do gráfico
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.saida = QtWidgets.QLabel('Saída:', self.visual_widget)
        self.saida.setObjectName("saida")
        self.horizontalLayout.addWidget(self.saida)
        self.label_4 = QtWidgets.QLabel('0', self.visual_widget)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout.addWidget(self.label_4)
        self.Layout_visualizacao.addLayout(self.horizontalLayout)
        
        # aqui ficará a imagem do cérebro
        self.frameCerebro = QtWidgets.QFrame(self.visual_widget)
        self.frameCerebro.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameCerebro.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameCerebro.setObjectName("frameCerebro")
        self.Layout_visualizacao.addWidget(self.frameCerebro)
 
        # Widget com status (na coluna esquerda)
        self.formLayoutWidget = QtWidgets.QWidget(self.left_widget)
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.label_3 = QtWidgets.QLabel('Status do modelo: ', self.formLayoutWidget)
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.label_5 = QtWidgets.QLabel('Nenhum modelo carregado', self.formLayoutWidget)
        self.label_5.setWordWrap(True)
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.label_5)
        self.label = QtWidgets.QLabel('Status do LSL: ', self.formLayoutWidget)
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label)
        self.label_2 = QtWidgets.QLabel('Desconectado', self.formLayoutWidget)
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.label_2)
        self.left_layout.addWidget(self.formLayoutWidget)
 
        # botões (embaixo do form na coluna esquerda)
        self.buttonLayoutWidget = QtWidgets.QWidget(self.left_widget)
        self.buttonLayout = QtWidgets.QVBoxLayout(self.buttonLayoutWidget)
        self.button = QtWidgets.QPushButton('Conectar LSL', self.buttonLayoutWidget)
        self.buttonLayout.addWidget(self.button)
        self.button_iniciarBCI = QtWidgets.QPushButton('Iniciar BCI', self.buttonLayoutWidget)
        self.buttonLayout.addWidget(self.button_iniciarBCI)
        self.left_layout.addWidget(self.buttonLayoutWidget)
        self.left_layout.addStretch()
 
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
        #self.button_iniciarBCI.clicked.connect(self.abrir_janela_teste)
        self.button_iniciarBCI.clicked.connect(self.iniciar_bci)
        QtCore.QMetaObject.connectSlotsByName(self)
        #self.show()
    


    def iniciar_bci(self):
        # Inicia ao detectar dados não-zero
        self.timer_plot = QtCore.QTimer(self)
        self.timer_plot.timeout.connect(self.update_plot)
        self.timer_plot.start(5)

    def predict(self,arr):
        return self.model(arr, training=False)

    #data_info = mne.create_info(canais,sfreq=60)

    def update_plot(self):
        # Puxa chunk de amostras
        chunk, _ = self.inlet.pull_chunk(timeout=0.0)
        if not chunk:
           return 




        # 2. Atualização do Buffer Circular (Numpy é mais rápido que lista append/pop)
        # Desloca os dados antigos para a esquerda e insere os novos no final
        new_len = len(chunk)
        self.current_data = np.roll(self.current_data, -new_len, axis=0)
        self.current_data[-new_len:, :] = chunk
        print(self.current_data.shape)
        print(np.array([self.current_data]).shape)
    
        try:
            pred = self.predict(np.array([self.current_data[self.x_size - self.epochsize:]])).numpy()[0][0]
            if pred < self.limiar:
                label = 'T1'
            elif pred > 1 - self.limiar:
                label = 'T2'
            else:
                label = 'T0'
        except: pred = 'Modelo incompatível'


        self.label_4.setText(str(pred))

        # 4. ATUALIZAÇÃO DO GRÁFICO (A mágica acontece aqui)
        # Eixo X é sempre o mesmo (0 a x_size)
        x_data = np.arange(self.x_size)
        
        for i, line in enumerate(self.lines):
           # Pegamos os dados do canal i
           channel_data = self.current_data[:, i]
           
           # Adicionamos um offset para criar o efeito "Waterfall" (um canal em cima do outro)
           # Sem isso, todos os 16 canais ficariam misturados no zero.
           offset = i * self.escala_visual
           
           line.set_data(x_data, channel_data + offset)

        # Redesenha apenas o canvas
        self.canvas.draw_idle()


    def abrir_janela_teste(self):   
        self.janela_teste = JanelaTeste()

    def abrir_modelo(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 
   '../',"Model files (*.h5)")
        print(fname)
        try:
           self.model = load_model(fname[0])
           self.model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
           self.label_5.setText('shape:'+str(self.model.input_shape)+'\noutput shape:'+str(self.model.output_shape)+'\nmetric names:'+
                              str(self.model.metrics_names))
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
           self.inlet = StreamInlet(self.streams[0])
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
        self.setGeometry(200,200,350,50)
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