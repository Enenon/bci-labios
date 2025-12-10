from PyQt5 import QtCore, QtGui
# from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtWidgets import * 
import sys
import numpy as np
from scipy.fft import fft
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np
from pylsl import StreamInlet, resolve_stream, resolve_byprop
from time import sleep
import matplotlib.pyplot as plt
from random import choice

usar_modelo = False
if usar_modelo:
    from keras.models import load_model
    from tensorflow.keras.optimizers import Adam
    import os
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



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
        self.centralwidget = QWidget(self)
        self.setCentralWidget(self.centralwidget)
        self.main_layout = QHBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(8, 8, 8, 8)

        # coluna esquerda (controles) - não expande
        self.left_widget = QWidget(self.centralwidget)
        self.left_layout = QVBoxLayout(self.left_widget)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.left_widget, 0)

        # coluna direita (visualização) - expande com a janela
        self.visual_widget = QWidget(self.centralwidget)
        self.Layout_visualizacao = QVBoxLayout(self.visual_widget)
        self.Layout_visualizacao.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.visual_widget, 1)
 
        # --- CONFIGURAÇÃO OTIMIZADA DO MATPLOTLIB ---
        self.frameGrafico = QFrame(self.visual_widget)
        self.frameGrafico_layout = QVBoxLayout(self.frameGrafico)
        
        # Criamos a figura uma única vez
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax_seriet = self.figure.add_subplot(111)
        
        # Configuração estética do gráfico
        self.ax.set_title('Série Temporal EEG (Real-time)')
        self.ax.set_xlabel('Amostras')
        self.ax.set_yticks([]) # Remove eixo Y numérico para limpar
        self.ax.set_xlim(0, self.x_size)
        # Ajusta limite Y para caber todos os canais empilhados (waterfall)
        self.escala_visual = 750 # Fator para separar as linhas visualmente
        self.ax_seriet.set_ylim(-self.escala_visual, self.n_channels * self.escala_visual + self.escala_visual - 700) #botei esse 700 pra ficar mais encaixado no grafico
        
        # CRUCIAL: Criamos as linhas (artistas) vazias agora e guardamos as referências
        self.lines = []
        for i in range(self.n_channels):
           # Plotamos uma linha vazia para cada canal
           line, = self.ax_seriet.plot([0,self.epochsize], 2*[i*self.escala_visual], lw=1) 
           self.lines.append(line)

        self.frameGrafico_layout.addWidget(self.canvas)
        # permitir expansão do canvas/frame
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.frameGrafico.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.Layout_visualizacao.addWidget(self.frameGrafico)
 
        self.ax_seriet.set_title('Série Temporal EEG')
        self.ax_seriet.set_xlabel('Tempo (s)')
        self.frameGrafico.setFrameShape(QFrame.StyledPanel)
        self.frameGrafico.setFrameShadow(QFrame.Raised)
        self.frameGrafico.setObjectName("frameGrafico")
 
        # área de saída abaixo do gráfico
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.comboBoxPlots = QComboBox(self.visual_widget)
        self.comboBoxPlots.addItem('Série temporal')
        self.comboBoxPlots.addItem('FFT')
        self.horizontalLayout.addWidget(self.comboBoxPlots)
        self.button_abrirjanela = QPushButton('Abir em nova janela',self.visual_widget)
        self.horizontalLayout.addWidget(self.button_abrirjanela)
        self.saida = QLabel('Saída:', self.visual_widget)
        self.saida.setObjectName("saida")
        self.horizontalLayout.addWidget(self.saida)
        self.label_previsao = QLabel('0', self.visual_widget)
        self.label_previsao.setObjectName("label_previsao")
        
        self.button_abrirjanela.clicked.connect(self.abrir_janela_plot)
        self.horizontalLayout.addWidget(self.label_previsao)
        
        self.Layout_visualizacao.addLayout(self.horizontalLayout)
        
        # aqui ficará a imagem do cérebro
        self.frameCerebro = QFrame(self.visual_widget)
        self.frameCerebro.setFrameShape(QFrame.StyledPanel)
        self.frameCerebro.setFrameShadow(QFrame.Raised)
        self.frameCerebro.setObjectName("frameCerebro")
        self.Layout_visualizacao.addWidget(self.frameCerebro)
 
        # Widget com status (na coluna esquerda)
        self.formLayoutWidget = QWidget(self.left_widget)
        self.formLayout = QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.label_status_modelo = QLabel('Status do modelo: ', self.formLayoutWidget)
        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label_status_modelo)
        self.modelo_infos = QLabel('Nenhum modelo carregado', self.formLayoutWidget)
        self.modelo_infos.setWordWrap(True)
        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.modelo_infos)
        self.label_status_lsl = QLabel('Status do LSL: ', self.formLayoutWidget)
        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.label_status_lsl)
        self.status_lsl = QLabel('Desconectado', self.formLayoutWidget)
        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.status_lsl)
        self.left_layout.addWidget(self.formLayoutWidget)
 
        # botões (embaixo do form na coluna esquerda)
        self.buttonLayoutWidget = QWidget(self.left_widget)
        self.buttonLayout = QVBoxLayout(self.buttonLayoutWidget)
        self.button = QPushButton('Conectar LSL', self.buttonLayoutWidget)
        self.buttonLayout.addWidget(self.button)
        self.button_iniciarBCI = QPushButton('Iniciar BCI', self.buttonLayoutWidget)
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
        self.status_lsl.setPalette(self.palette_vermelha)

        
        self.setStatusBar(QStatusBar(self))

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

        # O plot da FFT é idêntico ao da série temporal

        self.figure_FFT = Figure(figsize=(5, 3), dpi=100)
        self.canvas_FFT = FigureCanvas(self.figure_FFT)

        self.ax_fft = self.figure_FFT.add_subplot(111) # subplot com 1 linha e 1 coluna no índice 1
        
        self.escala_FFT = 200
        self.normalizacaoFFT = 1
        self.ax_fft.set_title('Transformada de Fourier')
        self.ax_fft.set_xlabel('Frequencia')
        self.ax_fft.set_yticks([])
        self.ax_fft.set_xlim(0, self.epochsize)

        self.ax_fft.set_ylim(-self.escala_visual, self.n_channels * self.escala_FFT + self.escala_FFT - 700) #botei esse 700 pra ficar mais encaixado no grafico
        

        self.lines_fft = []
        for i in range(self.n_channels):
           # Plotamos uma linha vazia para cada canal
           line_fft, = self.ax_fft.plot([0,self.epochsize], 2*[i*self.escala_FFT], lw=1) 
           self.lines_fft.append(line_fft)

        
    



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


        self.label_previsao.setText(str(pred))

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

            fft_data = fft(channel_data)/self.normalizacaoFFT
            self.lines_fft[i].set_data([i for i in range(len(fft_data))],fft_data + i * self.escala_FFT)

        # Redesenha apenas o canvas
        self.canvas.draw_idle()

        self.canvas_FFT.draw_idle()



    def abrir_janela_teste(self):   
        self.janela_teste = JanelaTeste()

    def abrir_modelo(self):
        if usar_modelo:
            fname = QFileDialog.getOpenFileName(self, 'Open file', 
    '../',"Model files (*.h5)")
            print(fname)
            try:
                self.model = load_model(fname[0])
                self.model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
                self.modelo_infos.setText('shape:'+str(self.model.input_shape)+'\noutput shape:'+str(self.model.output_shape)+'\nmetric names:'+
                                    str(self.model.metrics_names))
                self.modelo_infos.setPalette(self.palette_verde)
            except:
                self.modelo_infos.setText('Modelo incompatível.')
                self.modelo_infos.setPalette(self.palette_vermelha)
        else: return
        #model.summary()

    def conectar_LSL(self):
        print("Aguardando stream EEG...")
        print('mudando a cor da paleta')

        self.status_lsl.setText('Procurando...')
        self.status_lsl.setPalette(self.palette_amarela)
        QApplication.processEvents() # <--- isso aplica as mudanças antes da função acabar
        self.streams = resolve_byprop('type', 'EEG',timeout=3)
        if self.streams:
           self.inlet = StreamInlet(self.streams[0])
           palette = QtGui.QPalette()
           palette.setBrush(QtGui.QPalette.All, QtGui.QPalette.WindowText,QtGui.QBrush(QtGui.QColor(4,150,0)))
           self.status_lsl.setText('Conectado!')
           self.status_lsl.setPalette(self.palette_verde)
        else:
           print('Não achou conexão')
           self.status_lsl.setPalette(self.palette_vermelha)
           self.status_lsl.setText('Desconectado')

    def abrir_janela_plot(self):
        widget_escolhido = self.comboBoxPlots.currentIndex()

        self.janela_plot = JanelaPlot(self,widget_escolhido)


    def update_FFT(self):
        self.t += self.timer.interval()/100
        self.ax_FFT.clear()
        self.ax_FFT.plot(self.t,np.sin(self.t))
        self.canvas_FFT.draw_idle()
           
class JanelaPlot(QMainWindow):
    def __init__(self,principal,index):
        super().__init__()
        self.janelaPrincipal = principal
        self.setGeometry(200,200,350,350)
        self.setWindowTitle('Plot')
        self.centralwidget = QWidget(self)
        self.setCentralWidget(self.centralwidget)
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvas(Figure(figsize=(5,4)))
        self.layout1 = QVBoxLayout(self.centralwidget)
        match index:
            case 0:
                #self.canvas = principal.canvas
                self.layout1.addWidget(principal.canvas)
            case 1:
                #self.canvas = principal.canvas_FFT
                self.layout1.addWidget(principal.canvas_FFT,99)
                self.normtext = QPlainTextEdit(self.centralwidget)
                #self.layout1.addWidget(self.normtext,1)

                self.layoutConfigs = QHBoxLayout(self.centralwidget)
                self.layout1.addLayout(self.layoutConfigs,1)
                self.normtext.setPlainText(str(principal.normalizacaoFFT))
                self.normButton = QPushButton('Normalizar',self.centralwidget)
                self.layoutConfigs.addWidget(self.normtext,1)
                self.layoutConfigs.addWidget(self.normButton,1)
                self.layoutConfigs.addWidget(QWidget(self),8)

                self.normButton.clicked.connect(self.normalizar)

        
        self.show()

    def normalizar(self):
        num = self.normtext.toPlainText()
        print(num)
        self.janelaPrincipal.normalizacaoFFT = float(num)





class JanelaTeste(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(200,200,350,50)
        self.setWindowTitle('Teste')
        self.label = QLabel("Recurso indisponível!",self)
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

        self.button = QPushButton('Clique aqui',self)
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

        widget1 = QWidget()
        self.setCentralWidget(widget1)
        layout = QVBoxLayout(widget1)

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