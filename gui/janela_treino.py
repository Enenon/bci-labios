from dependencias import *
from aquisicao import Aquisicao
class JanelaTreino(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(200,200,350,350)
        self.setWindowTitle('Janela Treino')
        self.centralwidget = QWidget(self)
        self.setCentralWidget(self.centralwidget)
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.layout1 = QVBoxLayout(self.centralwidget)

        self.titleLabel = QLabel('Treinamento da Rede Neural', self)
        self.titleLabel.setFont(QtGui.QFont('Arial', 16))
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.layout1.addWidget(self.titleLabel)

        self.botao_abrir_modelo = QPushButton('Abrir Modelo', self)
        self.botao_abrir_modelo.clicked.connect(self.abrir_modelo)
        self.layout1.addWidget(self.botao_abrir_modelo)

        self.modelo_infos = QLabel('', self)
        self.layout1.addWidget(self.modelo_infos)

        self.botao_iniciar_treino = QPushButton('Iniciar Treino', self)
        self.botao_iniciar_treino.clicked.connect(self.iniciar_treino)
        self.layout1.addWidget(self.botao_iniciar_treino)

        self.palette_verde = QtGui.QPalette()
        self.palette_verde.setColor(QtGui.QPalette.WindowText, QtGui.QColor(0, 150, 0))  # Verde
        self.palette_vermelha = QtGui.QPalette()
        self.palette_vermelha.setColor(QtGui.QPalette.WindowText, QtGui.QColor(150, 0, 0))  # Vermelho

    def abrir_modelo(self):
        if usar_modelo:
            try:
                fname = QFileDialog.getOpenFileName(self, 'Open file', 
        '../',"Model files (*.h5)")
                print(fname)

                self.model = load_model(fname[0])
                self.model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
                self.modelo_infos.setText('shape:'+str(self.model.input_shape)+'\noutput shape:'+str(self.model.output_shape)+'\nmetric names:'+
                                    str(self.model.metrics_names))
                self.modelo_infos.setPalette(self.palette_verde)

                self.aquisicao = Aquisicao(len_data=self.model.input_shape[1], num_canais=self.model.input_shape[2])
            except:
                self.modelo_infos.setText('Modelo incompatível.')
                self.modelo_infos.setPalette(self.palette_vermelha)
        else: return

    def iniciar_treino(self):
        if not hasattr(self, 'model'):
            QMessageBox.warning(self, 'Aviso', 'Carregue um modelo primeiro.')
            return
        duration, ok = QInputDialog.getInt(self, 'Duração do Treino', 'Digite a duração em segundos:', 60, 1, 3600, 1)
        if ok:
            self.training_window = TrainingWindow(self.model, duration)
            self.training_window.show()




class TrainingWindow(QDialog):
    def __init__(self, model, duration):
        super().__init__()
        self.model = model
        self.duration = duration
        self.output_binario = len(self.model.outputs) == 1
        if self.output_binario:
            self.outputs = 2 # para binário
        else:
            self.outputs = len(self.model.outputs) if hasattr(self.model, 'outputs') and isinstance(self.model.outputs, list) else 1
        self.current_output = 0
        self.setWindowTitle('Janela de Treino')
        self.resize(600, 400)
        self.layout = QVBoxLayout()
        self.label = QLabel(f"Treinando output {self.current_output + 1} de {self.outputs}")
        self.layout.addWidget(self.label)
        self.countdown_label = QLabel("3")
        self.layout.addWidget(self.countdown_label)
        self.setLayout(self.layout)
        self.start_countdown()

    def start_countdown(self):
        self.count = 3
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_countdown)
        self.timer.start(1000)

    def update_countdown(self):
        self.countdown_label.setText(str(self.count))
        self.count -= 1
        if self.count < 0:
            self.timer.stop()
            self.start_training()

    def start_training(self):
        self.countdown_label.setText("Treinando...")
        QtCore.QTimer.singleShot(self.duration * 1000, self.training_done)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.training)
        self.timer.start(20)


    def training_done(self):
        self.current_output += 1
        if self.current_output < self.outputs:
            self.label.setText(f"Treinando output {self.current_output + 1} de {self.outputs}")
            self.countdown_label.setText("3")
            self.start_countdown()
        else:
            self.label.setText("Treino concluído")
            self.countdown_label.setText("")
            # Optionally close after a delay
            QtCore.QTimer.singleShot(2000, self.close)

    def training(self):
        # Aqui você pode adicionar a lógica de treinamento usando self.model e self.aquisicao
        self.aquisicao.adquirir()
        pred = self.aquisicao.predict(self.model)
        if self.output_binario:
            pred = np.argmax(pred)  # Converte para classe binária
        else:
            pred = np.argmax(pred, axis=1)  # Converte para classe multi-classe
        print(pred)
        pass



if __name__ == '__main__':
    def window():
        app = QApplication(sys.argv)
        win = JanelaTreino()
        win.show()

        sys.exit(app.exec_())
    window()