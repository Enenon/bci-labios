from pyexpat import model

from dependencias import *
from aquisicao import Aquisicao

class GaugeWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(250, 130)
        self.current_angle = 0.0
        self.base_angle = 0.0   
        self.target_angle = 0.0
        self.incerteza = 0.0    
        
        self.anim_timer = QtCore.QTimer()
        self.anim_timer.timeout.connect(self.update_animation)
        self.anim_timer.start(30) 

    def set_probabilities(self, prob_left, prob_right, prob_rest):
        self.target_angle = (prob_right * 60) + (prob_left * -60)
        confianca_maxima = max(prob_left, prob_right, prob_rest)
        self.incerteza = 1.0 - confianca_maxima

    def update_animation(self):
        diff = self.target_angle - self.base_angle
        if abs(diff) > 0.1:
            self.base_angle += diff * 0.15
            
        tremor_maximo_graus = 15.0 
        tremor_atual = uniform(-1.0, 1.0) * (self.incerteza * tremor_maximo_graus)
        self.current_angle = self.base_angle + tremor_atual
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        center_x, center_y = w / 2, h - 20
        radius = min(w / 2, h) - 25

        rect = QRectF(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        painter.setPen(QPen(QColor("#00bcd4"), 15, Qt.SolidLine, Qt.FlatCap))
        painter.drawArc(rect, 120 * 16, 60 * 16) 
        
        painter.setPen(QPen(QColor("#888888"), 15, Qt.SolidLine, Qt.FlatCap))
        painter.drawArc(rect, 60 * 16, 60 * 16)
        
        painter.setPen(QPen(QColor("#ff4081"), 15, Qt.SolidLine, Qt.FlatCap))
        painter.drawArc(rect, 0 * 16, 60 * 16)

        painter.setPen(QColor("#ffffff"))
        font = QtGui.QFont("Arial", 8, QtGui.QFont.Bold)
        painter.setFont(font)
        #painter.drawText(int(center_x - radius - 20), int(center_y), "ESQ")
        #painter.drawText(int(center_x + radius + 0), int(center_y), "DIR")
        #painter.drawText(int(center_x - 15), int(center_y - radius - 15), "REP")

        painter.translate(center_x, center_y)
        painter.rotate(self.current_angle)
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#ffffff")))
        poly = QPolygon([QPoint(-4, 0), QPoint(4, 0), QPoint(0, int(-radius + 5))])
        painter.drawPolygon(poly)
        
        painter.setBrush(QBrush(QColor("#ffc107")))
        painter.drawEllipse(QPoint(0, 0), 6, 6)


class JanelaTreino(QMainWindow):
    def __init__(self,aquisicao=None):
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

        self.botao_guardar_dados = QCheckBox('Guardar Dados', self)
        self.layout1.addWidget(self.botao_guardar_dados)

        self.botao_iniciar_treino = QPushButton('Iniciar Treino', self)
        self.botao_iniciar_treino.clicked.connect(self.iniciar_treino)
        self.layout1.addWidget(self.botao_iniciar_treino)

        self.palette_verde = QtGui.QPalette()
        self.palette_verde.setColor(QtGui.QPalette.WindowText, QtGui.QColor(0, 150, 0))  # Verde
        self.palette_vermelha = QtGui.QPalette()
        self.palette_vermelha.setColor(QtGui.QPalette.WindowText, QtGui.QColor(150, 0, 0))  # Vermelho

        if aquisicao:
            self.aquisicao = aquisicao
        else:
            self.aquisicao = Aquisicao(len_data=512, num_canais=16)

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
            self.training_window = TrainingWindow(self.model, duration,aquisicao=self.aquisicao)
            self.aquisicao.conectar()
            self.training_window.show()
            # substituindo a janela de treino pela janela de fim de treino, para testar a nova janela
            #self.training_window = EndTrainingWindow(self.model)
            #self.training_window.show()




class TrainingWindow(QDialog):
    def __init__(self, model, duration,aquisicao):
        super().__init__()
        self.model = model
        self.model_weights = model.get_weights() # Armazena os pesos iniciais do modelo
        self.duration = duration
        self.aquisicao = aquisicao
        self.output_binario = len(self.model.outputs) == 1
        if self.output_binario:
            self.outputs = 2 # para binário
        else:
            self.outputs = len(self.model.outputs) if hasattr(self.model, 'outputs') and isinstance(self.model.outputs, list) else 1
        self.ponteiro = GaugeWidget()
        self.current_output = 0
        self.setWindowTitle('Janela de Treino')
        self.resize(600, 400)
        self.layout = QVBoxLayout()
        self.label = QLabel(f"Treinando output {self.current_output + 1} de {self.outputs}")
        self.layout.addWidget(self.label)
        self.countdown_label = QLabel("3")
        self.layout.addWidget(self.countdown_label)
        self.layout.addWidget(self.ponteiro)
        self.setLayout(self.layout)
        self.start_countdown()

    def start_countdown(self):
        self.count = 3
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_countdown) # atualiza a contagem de 1 em 1 segundo
        self.timer.start(1000)

    def update_countdown(self):
        self.countdown_label.setText(str(self.count))
        self.count -= 1
        if self.count < 0: # quando chega a 0, inicia o treino
            self.timer.stop()
            self.start_training()

    def start_training(self):
        self.countdown_label.setText("Treinando...")
        QtCore.QTimer.singleShot(self.duration * 1000, self.training_done) # programa o fim do treino para daqui a "duration" segundos
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.training) # chama a função de treino a cada 200ms (5 vezes por segundo)
        self.timer.start(200)


    def training_done(self):
        self.timer.stop()
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
            if self.model_weights[0] !=self.model.get_weights()[0]:
                print("Pesos foram atualizados durante o treino.")

            '''for layer in self.model.layers:
                print(f"Nome da Camada: {layer.name}")
                
                # weights[0] são os pesos (W), weights[1] são os bias (b)
                weights, biases = layer.get_weights()
                print(f"Pesos: \n{weights}")
                #print(f"Bias: \n{biases}")
                print("-" * 20)'''


    def training(self):
        # Aqui você pode adicionar a lógica de treinamento usando self.model e self.aquisicao
        if len(self.aquisicao.current_data) != self.model.input_shape[1]:
            print(self.aquisicao.current_data.shape, self.model.input_shape)
        self.aquisicao.adquirir()
        #pred = self.aquisicao.predict(self.model)
        pred = self.model(self.aquisicao.current_data[np.newaxis, :, :])
        if self.output_binario:
            pred = np.argmax(pred)  # Converte para classe binária
        else:
            pred = np.argmax(pred, axis=1)  # Converte para classe multi-classe
        print(pred)
        if pred == self.current_output:
            self.label.setText(f"Treinando output {self.current_output + 1} de {self.outputs} - Acertou!")
            #self.label.setPalette(self.palette_verde)
            self.model.fit(self.aquisicao.current_data[np.newaxis, :, :], np.array([self.current_output]), epochs=1, verbose=0)
        else:
            self.label.setText(f"Treinando output {self.current_output + 1} de {self.outputs} - Errou!")
            #self.label.setPalette(self.palette_vermelha)
        self.ponteiro.set_probabilities(1-pred,pred,0)

class EndTrainingWindow(QDialog):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.current_output = 0
        self.setWindowTitle('Finalização do Treino')
        self.resize(600, 400)
        self.layout = QVBoxLayout(self)
        self.label = QLabel("Treino concluído!"); self.label.setFont(QtGui.QFont('Arial', 16)); self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.label)
        layoutshape = QFormLayout()
        self.plainTextEdit = QPlainTextEdit()
        self.plainTextEdit.setPlaceholderText("Digite o nome do paciente aqui..."); self.plainTextEdit.setFixedHeight(30)
        layoutshape.addRow('Nome do Paciente:', self.plainTextEdit)
        self.layout.addLayout(layoutshape)
        self.button_salvar = QPushButton("Salvar Modelo")
        self.button_salvar.clicked.connect(self.salvar_modelo)
        self.layout.addWidget(self.button_salvar)

    def salvar_modelo(self):
        fname = QFileDialog.getSaveFileName(self, 'Salvar Modelo', '../',"Model files (*.h5)")
        if fname[0]:
            self.model.save(fname[0])
        
        
class Ponteiro(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(500,500)
        self.triangle = QPolygonF([QPointF(0, -20), QPointF(5, 60), QPointF(-5, 60)])
        self.angle = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.rotate)
        self.timer.start(100)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self.angle)

        painter.setBrush(QtGui.QColor(255, 0, 0))
        painter.drawPolygon(self.triangle)
    def rotate(self):
        self.angle = (self.angle + 10) % 360
        self.update()



if __name__ == '__main__':
    def window():
        app = QApplication(sys.argv)
        win = JanelaTreino()
        win.show()

        sys.exit(app.exec_())
    window()