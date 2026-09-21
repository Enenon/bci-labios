import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
# é possível que essa substitua a janela_main futuramente
from dependencias import *
from aquisicao import Aquisicao
from unitysender import UnitySender
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QPolygonF
from time import sleep
import pandas as pd
import random
from datetime import datetime
import numpy as np

# =============================================================================
# WIDGET DO VELOCÍMETRO (GaugeWidget Otimizado - Fixo e com Textos)
# =============================================================================
class GaugeWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(250, 160) 
        self.current_angle = 0.0
        self.target_angle = 0.0
        self.probs = [0.0, 0.0, 0.0]  
        
        self.anim_timer = QtCore.QTimer()
        self.anim_timer.timeout.connect(self.update_animation)
        self.anim_timer.start(30) 

    def set_probabilities(self, prob_left, prob_right, prob_rest):
        self.target_angle = (prob_right * 60) + (prob_left * -60)
        self.probs = [prob_left, prob_right, prob_rest]

    def update_animation(self):
        diff = self.target_angle - self.current_angle
        if abs(diff) > 0.1:
            self.current_angle += diff * 0.15
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        center_x, center_y = w / 2, h - 45 
        radius = min(w / 2, h) - 45

        rect = QtCore.QRectF(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        # Cores dos Arcos: Azul (Esq), Cinza (Repouso), Rosa (Dir)
        painter.setPen(QPen(QColor("#00bcd4"), 15, QtCore.Qt.SolidLine, QtCore.Qt.FlatCap))
        painter.drawArc(rect, 120 * 16, 60 * 16) 
        painter.setPen(QPen(QColor("#888888"), 15, QtCore.Qt.SolidLine, QtCore.Qt.FlatCap))
        painter.drawArc(rect, 60 * 16, 60 * 16)
        painter.setPen(QPen(QColor("#ff4081"), 15, QtCore.Qt.SolidLine, QtCore.Qt.FlatCap))
        painter.drawArc(rect, 0 * 16, 60 * 16)

        painter.save()
        painter.translate(center_x, center_y)
        painter.rotate(self.current_angle)
        
        # Agulha
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QBrush(QColor("#ffffff")))
        poly = QPolygonF([QtCore.QPointF(-4, 0), QtCore.QPointF(4, 0), QtCore.QPointF(0, -radius + 5)])
        painter.drawPolygon(poly)
        painter.setBrush(QBrush(QColor("#ffc107")))
        painter.drawEllipse(QtCore.QPointF(-3, -3), 6, 6)
        painter.restore()

        # Texto de Probabilidades
        painter.setPen(QPen(QtCore.Qt.white))
        painter.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        
        texto_prob = f"Cls 0: {self.probs[0]*100:.1f}%  |  Cls 1: {self.probs[1]*100:.1f}%  |  Cls 2: {self.probs[2]*100:.1f}%"
        rect_text = QtCore.QRectF(0, h - 35, w, 35)
        painter.drawText(rect_text, QtCore.Qt.AlignCenter, texto_prob)

# =============================================================================
# WORKER IA (Processamento Paralelo da Inteligência Artificial)
# =============================================================================
class WorkerIA(QThread):
    sinal_predicao = pyqtSignal(int, list)
    sinal_treino_concluido = pyqtSignal()

    def __init__(self, model, is_binary):
        super().__init__()
        self.model = model
        self.is_binary = is_binary
        self.rodando = True
        self.modo_treino = False
        self.dados_predicao = None
        self.dados_treino = None
        self.labels_treino = None

    def run(self):
        while self.rodando:
            if self.modo_treino and self.dados_treino is not None:
                try:
                    self.model.train_on_batch(self.dados_treino, self.labels_treino)
                except Exception as e: 
                    print(f"Erro no treino IA: {e}")
                self.modo_treino = False
                self.dados_treino = None
                self.sinal_treino_concluido.emit()
            
            elif not self.modo_treino and self.dados_predicao is not None:
                try:
                    res = self.model.predict(self.dados_predicao, verbose=0)[0]
                    if self.is_binary:
                        val = res[0]
                        pred = 0 if val < 0.5 else 1
                        prob = [1.0 - val, val]
                    else:
                        pred = int(np.argmax(res))
                        prob = res.tolist()
                    self.sinal_predicao.emit(pred, prob)
                except Exception: 
                    pass
                self.dados_predicao = None 
            
            sleep(0.01)

    def pedir_predicao(self, dados):
        if not self.modo_treino: 
            self.dados_predicao = dados

    def iniciar_transfer_learning(self, dados, labels):
        self.dados_treino = dados
        self.labels_treino = labels
        self.modo_treino = True 

    def parar(self):
        self.rodando = False


# =============================================================================
# JANELAS DE CONFIGURAÇÃO E EXECUÇÃO DE PROTOCOLO BCI
# =============================================================================
class JanelaConfiguracaoParadigma(QDialog):
    def __init__(self, unity_conectado, nomes_classes, playback_mode=False):
        super().__init__()
        self.nomes_classes = nomes_classes
        self.num_classes = len(nomes_classes)
        self.playback_mode = playback_mode
        self.setWindowTitle("Configuração da Sessão")
        self.resize(500, 550)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        layout = QVBoxLayout(self)

        lbl_titulo = QLabel(f"Sessão BCI ({self.num_classes} Classes Detectadas)")
        lbl_titulo.setStyleSheet("font-size: 16px; font-weight: bold; color: #00bcd4;")
        lbl_titulo.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(lbl_titulo)

        group_modo = QGroupBox("Modo de Operação")
        layout_modo = QVBoxLayout()
        self.combo_modo_sessao = QComboBox()
        self.combo_modo_sessao.addItems(["Treino + Teste Prático (Com Transfer Learning)", "Apenas Teste / Avaliação (Sem Transfer Learning)"])
        self.combo_modo_sessao.currentIndexChanged.connect(self.toggle_modo_sessao)
        layout_modo.addWidget(self.combo_modo_sessao)
        group_modo.setLayout(layout_modo)
        layout.addWidget(group_modo)

        group_tempos = QGroupBox("Tempos do Relógio Principal")
        form_tempos = QFormLayout()
        self.spin_aviso = QDoubleSpinBox(); self.spin_aviso.setRange(0.5, 10.0); self.spin_aviso.setValue(1.5); self.spin_aviso.setSuffix(" s")
        self.spin_acao = QDoubleSpinBox(); self.spin_acao.setRange(1.0, 15.0); self.spin_acao.setValue(3.0); self.spin_acao.setSuffix(" s")
        self.spin_repouso = QDoubleSpinBox(); self.spin_repouso.setRange(1.0, 15.0); self.spin_repouso.setValue(2.0); self.spin_repouso.setSuffix(" s")
        form_tempos.addRow("Aviso (Cruz ➕):", self.spin_aviso)
        form_tempos.addRow("Ação (Estímulo):", self.spin_acao)
        form_tempos.addRow("Repouso (Preta):", self.spin_repouso)
        group_tempos.setLayout(form_tempos)
        layout.addWidget(group_tempos)

        group_tl = QGroupBox("Estratégia de Calibração (Transfer Learning)")
        form_tl = QFormLayout()
        self.spin_calibracao = QSpinBox(); self.spin_calibracao.setRange(0, 200); self.spin_calibracao.setValue(20); self.spin_calibracao.setSuffix(" épocas/classe")
        self.spin_teste = QSpinBox(); self.spin_teste.setRange(1, 200); self.spin_teste.setValue(30); self.spin_teste.setSuffix(" épocas/classe")
        form_tl.addRow("1ª Fase (Calibração e Treino):", self.spin_calibracao)
        form_tl.addRow("2ª Fase (Teste Prático):", self.spin_teste)
        group_tl.setLayout(form_tl)
        layout.addWidget(group_tl)

        self.group_gabarito = QGroupBox("Gabarito de Playback")
        self.group_gabarito.setVisible(self.playback_mode)
        layout_gabarito = QVBoxLayout()
        self.line_gabarito = QLineEdit()
        self.line_gabarito.setPlaceholderText("Ex: 0, 1, 2, 0, 1")
        self.btn_carregar_gabarito = QPushButton("📂 Carregar _marcacoes.txt de Sessão Anterior")
        self.btn_carregar_gabarito.clicked.connect(self.carregar_gabarito_arquivo)
        layout_gabarito.addWidget(QLabel("Sequência Numérica de Classes:"))
        layout_gabarito.addWidget(self.line_gabarito)
        layout_gabarito.addWidget(self.btn_carregar_gabarito)
        self.group_gabarito.setLayout(layout_gabarito)
        layout.addWidget(self.group_gabarito)

        group_alvos = QGroupBox("Exibir estímulos em:")
        layout_alvos = QVBoxLayout()
        self.chk_python = QCheckBox("Interface Python (2D)"); self.chk_python.setChecked(True)
        self.chk_unity = QCheckBox("Ambiente Unity (3D)"); self.chk_unity.setChecked(unity_conectado); self.chk_unity.setEnabled(unity_conectado)
        if not unity_conectado: self.chk_unity.setText("Mostrar no Unity (Requer ligação)")
        layout_alvos.addWidget(self.chk_python); layout_alvos.addWidget(self.chk_unity)
        group_alvos.setLayout(layout_alvos)
        layout.addWidget(group_alvos)

        layout.addStretch()
        self.btn_iniciar = QPushButton("▶ APLICAR CONFIGURAÇÕES")
        self.btn_iniciar.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 15px;")
        self.btn_iniciar.clicked.connect(self.aceitar_configuracao)
        layout.addWidget(self.btn_iniciar)

    def toggle_modo_sessao(self, index):
        self.spin_calibracao.setEnabled(index == 0)
        if index == 1: 
            self.spin_calibracao.setValue(0)

    def carregar_gabarito_arquivo(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Carregar Marcações", "", "Arquivos Texto (*.txt)")
        if fname:
            try:
                sequencia = []
                with open(fname, "r") as f:
                    for line in f:
                        line = line.strip().replace('[', '').replace(']', '')
                        if not line: continue
                        parts = line.split(',')
                        if len(parts) >= 4:
                            sequencia.append(int(parts[3].strip()))
                self.line_gabarito.setText(", ".join(map(str, sequencia)))
                QMessageBox.information(self, "Sucesso", "Gabarito extraído com sucesso!")
            except Exception:
                QMessageBox.warning(self, "Erro", "Erro ao ler ficheiro de marcações.")

    def aceitar_configuracao(self):
        if not self.chk_python.isChecked() and not self.chk_unity.isChecked():
            return QMessageBox.warning(self, "Aviso", "Selecione pelo menos um local para exibir os estímulos!")
        
        gabarito_forçado = None
        if self.playback_mode:
            g_str = self.line_gabarito.text().replace(' ', '')
            if not g_str:
                return QMessageBox.warning(self, "Aviso", "O Playback exige a sequência numérica de Gabarito!")
            try:
                gabarito_forçado = [int(x) for x in g_str.split(',') if x != '']
                max_val = max(gabarito_forçado)
                if max_val >= self.num_classes:
                    return QMessageBox.warning(self, "Erro de Lógica", f"O Gabarito contém a classe '{max_val}', mas você só mapeou classes de 0 até {self.num_classes-1} na interface principal!")
            except ValueError:
                return QMessageBox.warning(self, "Erro", "A sequência deve conter apenas números inteiros separados por vírgula!")

        self.configs = {
            't_aviso': int(self.spin_aviso.value() * 1000), 't_acao': int(self.spin_acao.value() * 1000), 't_repouso': int(self.spin_repouso.value() * 1000), 
            'rep_calibracao': self.spin_calibracao.value(), 'rep_teste': self.spin_teste.value(),
            'usar_python': self.chk_python.isChecked(), 'usar_unity': self.chk_unity.isChecked(),
            'nomes_classes': self.nomes_classes, 'num_classes': self.num_classes,
            'gabarito_forçado': gabarito_forçado
        }
        self.accept()


class JanelaExecucaoParadigma(QDialog):
    sinal_extrair_dado = pyqtSignal(int) 
    sinal_iniciar_pausa = pyqtSignal()   
    sessao_concluida = pyqtSignal()      

    def __init__(self, configs, unity_sender=None):
        super().__init__()
        self.configs = configs; self.unity = unity_sender
        self.nomes_classes = configs['nomes_classes']
        self.num_classes = configs['num_classes']
        self.gabarito_forçado = configs['gabarito_forçado']
        
        if self.gabarito_forçado is not None:
            self.seq_calibracao = []
            divisao = self.configs['rep_calibracao'] * self.num_classes
            self.seq_calibracao = self.gabarito_forçado[:divisao]
            self.seq_teste = self.gabarito_forçado[divisao:]
        else:
            self.seq_calibracao = []; self.seq_teste = []
            for i in range(self.num_classes):
                self.seq_calibracao += [i] * configs['rep_calibracao']
                self.seq_teste += [i] * configs['rep_teste']
            random.shuffle(self.seq_calibracao)
            random.shuffle(self.seq_teste)
        
        self.fase_atual = "CALIBRACAO" if len(self.seq_calibracao) > 0 else "TESTE"
        self.trial_atual = 0
        self.total_fase = len(self.seq_calibracao) if self.fase_atual == "CALIBRACAO" else len(self.seq_teste)
        self.estado_atual = "STANDBY" 

        self.setWindowTitle("Coleta Visual (Cues)")
        self.resize(800, 600)
        self.setStyleSheet("background-color: black; color: white;")
        layout = QVBoxLayout(self)
        
        self.lbl_info = QLabel("A aguardar inicialização...")
        self.lbl_info.setStyleSheet("font-size: 16px; color: gray;")
        self.lbl_info.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.lbl_info)
        
        self.lbl_estimulo = QLabel("STANDBY")
        self.lbl_estimulo.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_estimulo.setStyleSheet("font-size: 60px; font-weight: bold; color: #444444;")
        layout.addWidget(self.lbl_estimulo, 1)
        
        self.timer_logica = QTimer(self)
        self.timer_logica.setSingleShot(True)
        self.timer_logica.timeout.connect(self.proximo_estado)
        self.contagem_inicial = 3

    def iniciar_paradigma(self):
        self.estado_atual = "INICIO"
        self.timer_contagem = QTimer(self)
        self.timer_contagem.timeout.connect(self.rotina_contagem)
        self.timer_contagem.start(1000)
        self.rotina_contagem()

    def rotina_contagem(self):
        if self.contagem_inicial > 0:
            self.desenhar_tela(str(self.contagem_inicial), 150, "#00bcd4", "Prepare-se...")
            self.contagem_inicial -= 1
        else:
            self.timer_contagem.stop()
            self.proximo_estado()

    def desenhar_tela(self, texto, tamanho, cor, info):
        if self.configs['usar_python']:
            self.lbl_estimulo.setText(texto)
            self.lbl_estimulo.setStyleSheet(f"font-size: {tamanho}px; color: {cor}; font-weight: bold;")
            self.lbl_info.setText(info)
        else:
            self.lbl_estimulo.setText("A transmitir...")
            self.lbl_estimulo.setStyleSheet("font-size: 40px; color: #aaaaaa;")
            self.lbl_info.setText(info)

    def proximo_estado(self):
        if self.estado_atual in ["CONCLUIDO", "PAUSA_TECNICA", "STANDBY"]: return

        lista_atual = self.seq_calibracao if self.fase_atual == "CALIBRACAO" else self.seq_teste
        
        if self.trial_atual >= len(lista_atual):
            self.finalizar_forçado()
            return

        classe_alvo = lista_atual[self.trial_atual]
        nome_classe_alvo = self.nomes_classes[classe_alvo]

        if self.estado_atual in ["INICIO", "REPOUSO"]:
            self.estado_atual = "AVISO"
            if self.configs['usar_unity'] and self.unity: 
                self.unity.send("CUE_CROSS") 
            self.desenhar_tela("➕", 150, "white", f"FASE: {self.fase_atual} | Estímulo {self.trial_atual + 1}/{self.total_fase}")
            self.timer_logica.start(self.configs['t_aviso'])
            
        elif self.estado_atual == "AVISO":
            self.estado_atual = "ACAO"
            icones = ["⬅️", "➡️", "🛑", "⬆️", "⬇️", "🌀", "⭐"]
            icone = icones[classe_alvo] if classe_alvo < len(icones) else f"[{classe_alvo}]"
            cores = ["#00bcd4", "#ff4081", "#ffeb3b", "#8bc34a", "#ff9800", "#9c27b0", "#ffffff"]
            cor = cores[classe_alvo] if classe_alvo < len(cores) else "white"

            if self.configs['usar_unity'] and self.unity:
                if classe_alvo == 0: self.unity.send("CUE_LEFT")
                elif classe_alvo == 1: self.unity.send("CUE_RIGHT")
                elif classe_alvo == 2: self.unity.send("CUE_REST")
                else: self.unity.send(f"CUE_{classe_alvo}")

            self.desenhar_tela(icone, 180, cor, f"AÇÃO: {nome_classe_alvo}")
            self.timer_logica.start(self.configs['t_acao'])
            
        elif self.estado_atual == "ACAO":
            self.estado_atual = "REPOUSO"
            self.sinal_extrair_dado.emit(classe_alvo)

            if self.configs['usar_unity'] and self.unity: 
                self.unity.send("CUE_REST") 
            self.desenhar_tela("", 150, "white", "Descanso...")
            
            self.trial_atual += 1
            if self.trial_atual >= self.total_fase:
                if self.fase_atual == "CALIBRACAO":
                    self.estado_atual = "PAUSA_TECNICA"
                    self.desenhar_tela("☕", 120, "yellow", "Descanso Técnico.\nAguarde um instante enquanto a IA se adapta...")
                    self.sinal_iniciar_pausa.emit() 
                else:
                    self.estado_atual = "CONCLUIDO"
                    self.desenhar_tela("Concluído!", 80, "#00e676", "Sessão finalizada.")
                    self.timer_logica.start(2000)
                    self.sessao_concluida.emit()
            else:
                self.timer_logica.start(self.configs['t_repouso'])

    def finalizar_forçado(self):
        self.estado_atual = "CONCLUIDO"
        self.desenhar_tela("Fim do Gabarito", 60, "yellow", "O arquivo de gabarito acabou. Finalizando...")
        self.timer_logica.start(2000)
        self.sessao_concluida.emit()

    def retomar_paradigma(self):
        self.fase_atual = "TESTE"
        self.trial_atual = 0
        self.total_fase = len(self.seq_teste)
        if self.total_fase == 0:
            self.estado_atual = "CONCLUIDO"
            self.desenhar_tela("Concluído!", 80, "#00e676", "Calibração finalizada.")
            self.timer_logica.start(2000)
            self.sessao_concluida.emit()
        else:
            self.estado_atual = "REPOUSO"
            self.proximo_estado()

# =============================================================================
# JANELA PRINCIPAL (HUB UNIFICADO)
# =============================================================================
class JanelaInicial(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1300, 850)
        self.setWindowTitle('BCI Control Center')
        aplicar_estilo_escuro(self)

        self.unity = None; self.inlet = None; self.model = None; self.worker_ia = None
        self.dados_arquivo = None; self.ponteiro_arquivo = 0
        
        self.canais = ['C3', 'C4', 'Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8','T7', 'T8', 'P7', 'P3', 'P4', 'P8', 'O1', 'O2']
        self.n_channels = len(self.canais) 
        self.x_size = 500 
        self.len_data = 1500

        self.salvar_dados = True
        self.dados_guardados = []; self.marcacoes = []
        self.buffer_dados_treino = []; self.buffer_labels_treino = []
        self.ocorreu_transfer_learning = False
        self.paradigma_win = None
        self.label_esperado = None 
        
        self.current_data_visual = np.zeros((self.x_size, self.n_channels))
        self.fs = 250.0  
        self.escala_visual = 150; self.escala_auto = False
        self.fft_smooth_factor = 0.0
        self.fft_buffer_history = np.zeros((self.n_channels, self.x_size//2))

        self.aquisicao = Aquisicao(len_data=self.len_data, num_canais=self.n_channels, xlim_FFT=self.x_size//2, smooth_factor=self.fft_smooth_factor)

        self.centralwidget = QWidget(self); self.setCentralWidget(self.centralwidget)
        self.main_layout = QHBoxLayout(self.centralwidget)
        
        self.panel_left = QFrame(); self.panel_left.setFixedWidth(400)
        self.layout_left = QVBoxLayout(self.panel_left)
        self.setup_painel_esquerdo()
        self.main_layout.addWidget(self.panel_left)
        
        self.panel_right = QWidget(); self.layout_right = QVBoxLayout(self.panel_right)
        self.tabs_graficos = QTabWidget(); self.setup_tabs_graficos()
        self.layout_right.addWidget(self.tabs_graficos)
        self.main_layout.addWidget(self.panel_right, 1)

        self.palette_vermelha = QtGui.QPalette(); self.palette_vermelha.setBrush(QtGui.QPalette.All, QtGui.QPalette.WindowText, QtGui.QBrush(QtGui.QColor(255, 0, 4)))
        self.palette_amarela = QtGui.QPalette(); self.palette_amarela.setBrush(QtGui.QPalette.All, QtGui.QPalette.WindowText, QtGui.QBrush(QtGui.QColor(150, 150, 0)))
        self.palette_verde = QtGui.QPalette(); self.palette_verde.setBrush(QtGui.QPalette.All, QtGui.QPalette.WindowText, QtGui.QBrush(QtGui.QColor(4, 150, 0)))

        self.timer_plot = QtCore.QTimer()
        self.timer_plot.timeout.connect(self.update_loop_continuo)
        self.timer_plot.start(20)

    def setup_painel_esquerdo(self):
        lbl_titulo = QLabel("PAINEL DE CONTROLE")
        lbl_titulo.setFont(QtGui.QFont("Segoe UI", 14, QtGui.QFont.Bold))
        lbl_titulo.setAlignment(QtCore.Qt.AlignCenter)
        self.layout_left.addWidget(lbl_titulo)

        self.tabs_controles = QTabWidget()
        
        # ABA 1: CONFIG
        self.tab_config = QWidget(); layout_config = QVBoxLayout(self.tab_config)
        
        group_conn = QGroupBox("Módulos & IA")
        form_conn = QFormLayout()
        self.lbl_lsl = QLabel("Desconectado"); self.lbl_lsl.setStyleSheet("color: #ff5555;")
        self.lbl_unity = QLabel("Desconectado"); self.lbl_unity.setStyleSheet("color: #ff5555;")
        self.modelo_infos = QLabel("Nenhum"); self.modelo_infos.setStyleSheet("color: gray;")
        form_conn.addRow("LSL:", self.lbl_lsl)
        form_conn.addRow("Unity:", self.lbl_unity)
        form_conn.addRow("IA:", self.modelo_infos)
        
        self.combo_tipo_modelo = QComboBox()
        self.combo_tipo_modelo.addItems(["Multiclasse (Categorical / Softmax)", "Binário (Binary / Sigmoid)"])
        form_conn.addRow("Tipo IA:", self.combo_tipo_modelo)
        group_conn.setLayout(form_conn)
        layout_config.addWidget(group_conn)

        row_botoes = QHBoxLayout()
        self.btn_lsl = QPushButton("📡 LSL"); self.btn_lsl.clicked.connect(self.conectar_LSL)
        self.btn_unity = QPushButton("🎮 Unity"); self.btn_unity.clicked.connect(self.conectar_Unity)
        self.btn_modelo = QPushButton("🤖 Modelo H5"); self.btn_modelo.clicked.connect(self.abrir_modelo)
        row_botoes.addWidget(self.btn_lsl); row_botoes.addWidget(self.btn_unity); row_botoes.addWidget(self.btn_modelo)
        layout_config.addLayout(row_botoes)

        # MAPEAMENTO DE CLASSES DINÂMICO
        group_classes = QGroupBox("Mapeamento de Classes (Gabarito)")
        layout_classes = QVBoxLayout()
        self.form_classes = QFormLayout()
        self.lista_lineedits = []
        
        # Inicia com 3 classes por padrão
        self.add_class_ui("Repouso"); self.add_class_ui("Mão Esquerda"); self.add_class_ui("Mão Direita")
        
        layout_classes.addLayout(self.form_classes)
        self.btn_add_classe = QPushButton("+ Adicionar Classe")
        self.btn_add_classe.clicked.connect(lambda: self.add_class_ui("Nova Classe"))
        layout_classes.addWidget(self.btn_add_classe)
        group_classes.setLayout(layout_classes)
        layout_config.addWidget(group_classes)

        group_shape = QGroupBox("Configuração de Hardware")
        form_shape = QFormLayout()
        self.spin_fs = QSpinBox(); self.spin_fs.setRange(1, 10000); self.spin_fs.setValue(250); self.spin_fs.setSuffix(" Hz")
        self.spin_shape_time = QSpinBox(); self.spin_shape_time.setRange(10, 5000); self.spin_shape_time.setValue(721); self.spin_shape_time.setSuffix(" pts")
        self.spin_shape_ch = QSpinBox(); self.spin_shape_ch.setRange(1, 128); self.spin_shape_ch.setValue(16); self.spin_shape_ch.setSuffix(" ch")
        self.spin_shape_ch.valueChanged.connect(lambda val: setattr(self, 'n_channels', val))
        form_shape.addRow("Amostragem (Hz):", self.spin_fs)
        form_shape.addRow("Shape Time:", self.spin_shape_time)
        form_shape.addRow("Canais:", self.spin_shape_ch)
        group_shape.setLayout(form_shape)
        layout_config.addWidget(group_shape)
        
        group_fonte = QGroupBox("Fonte de Dados")
        layout_fonte = QVBoxLayout()
        self.radio_lsl = QRadioButton("Placa LSL (Tempo Real)"); self.radio_lsl.setChecked(True)
        self.radio_csv = QRadioButton("Playback Offline (Ficheiro CSV)")
        self.radio_sim = QRadioButton("Sintético (Simulação Matemática)")
        self.btn_abrir_csv = QPushButton("📂 Abrir CSV..."); self.btn_abrir_csv.setEnabled(False)
        self.btn_abrir_csv.clicked.connect(self.abrir_arquivo_csv)
        self.radio_csv.toggled.connect(lambda state: self.btn_abrir_csv.setEnabled(state))

        layout_fonte.addWidget(self.radio_lsl); layout_fonte.addWidget(self.radio_csv)
        layout_fonte.addWidget(self.btn_abrir_csv); layout_fonte.addWidget(self.radio_sim)
        group_fonte.setLayout(layout_fonte)
        layout_config.addWidget(group_fonte)
        
        layout_config.addStretch(); self.tabs_controles.addTab(self.tab_config, "Configurações")

        # ABA 2: SESSÃO
        self.tab_experimento = QWidget(); layout_exp = QVBoxLayout(self.tab_experimento)
        self.btn_paradigma = QPushButton("🎯 PASSO 1: Configurar Protocolo")
        self.btn_paradigma.setStyleSheet("background-color: #ff9800; color: black; font-weight: bold; padding: 10px;")
        self.btn_paradigma.clicked.connect(self.abrir_gravacao_paradigma)
        layout_exp.addWidget(self.btn_paradigma)

        self.salvar_dados_checkbox = QCheckBox("Permitir Guardar Dados (CSV/TXT no final)")
        self.salvar_dados_checkbox.setChecked(True)
        self.salvar_dados_checkbox.stateChanged.connect(lambda state: setattr(self, 'salvar_dados', state == QtCore.Qt.Checked))
        layout_exp.addWidget(self.salvar_dados_checkbox)

        self.btn_iniciar_ia = QPushButton("▶ PASSO 2: INICIAR SESSÃO")
        self.btn_iniciar_ia.setStyleSheet("background-color: #2e7d32; font-weight: bold; padding: 15px; font-size: 14px;")
        self.btn_iniciar_ia.clicked.connect(self.iniciar_sessao_ml)
        layout_exp.addWidget(self.btn_iniciar_ia)

        group_mon = QGroupBox("Monitoramento em Tempo Real")
        layout_mon = QVBoxLayout()
        self.lbl_fase = QLabel("FASE: Parado")
        self.lbl_fase.setStyleSheet("color: yellow; font-weight: bold;")
        self.lbl_fase.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_predicao = QLabel("--")
        self.lbl_predicao.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        self.lbl_predicao.setAlignment(QtCore.Qt.AlignCenter)
        
        self.gauge = GaugeWidget()
        layout_mon.addWidget(self.lbl_fase)
        layout_mon.addWidget(self.lbl_predicao)
        layout_mon.addWidget(self.gauge)
        group_mon.setLayout(layout_mon)
        layout_exp.addWidget(group_mon)
        
        layout_exp.addStretch(); self.tabs_controles.addTab(self.tab_experimento, "Sessão & IA")
        self.layout_left.addWidget(self.tabs_controles)

    # ---------------- LÓGICA DE UI DAS CLASSES DINÂMICAS ----------------
    def add_class_ui(self, txt):
        idx = len(self.lista_lineedits)
        le = QLineEdit(txt)
        self.lista_lineedits.append(le)
        self.form_classes.addRow(f"{idx} ~ ", le)

    def clear_class_ui(self):
        while self.form_classes.rowCount() > 0:
            self.form_classes.removeRow(0)
        self.lista_lineedits.clear()

    # ---------------- GRÁFICOS ----------------
    def setup_tabs_graficos(self):
        self.tab_time = QWidget(); l_time = QVBoxLayout(self.tab_time)
        tb_time = QHBoxLayout()
        self.combo_scale = QComboBox(); self.combo_scale.addItems(["Auto", "50 uV", "100 uV", "200 uV", "400 uV"]); self.combo_scale.setCurrentText("200 uV")
        self.combo_scale.currentTextChanged.connect(lambda t: setattr(self, 'escala_auto', True) if t=="Auto" else (setattr(self, 'escala_auto', False), setattr(self, 'escala_visual', int(t.split()[0])), self.atualizar_limites_temporal()))
        tb_time.addWidget(QLabel("Escala:")); tb_time.addWidget(self.combo_scale); tb_time.addStretch()
        l_time.addLayout(tb_time)

        self.fig_time = Figure(figsize=(5,3), dpi=100); self.can_time = FigureCanvas(self.fig_time)
        self.setup_grafico_temporal(); l_time.addWidget(self.can_time); self.tabs_graficos.addTab(self.tab_time, "Série Temporal")

        self.tab_fft = QWidget(); l_fft = QVBoxLayout(self.tab_fft)
        tb_fft = QHBoxLayout()
        self.spin_smooth = QDoubleSpinBox(); self.spin_smooth.setRange(0, 0.99); self.spin_smooth.setSingleStep(0.1)
        self.spin_smooth.valueChanged.connect(self.mudar_smoothfactor)
        tb_fft.addWidget(QLabel("Smooth:")); tb_fft.addWidget(self.spin_smooth); tb_fft.addStretch()
        l_fft.addLayout(tb_fft)

        self.fig_fft = Figure(figsize=(5,3), dpi=100); self.can_fft = FigureCanvas(self.fig_fft)
        self.setup_grafico_fft(); l_fft.addWidget(self.can_fft); self.tabs_graficos.addTab(self.tab_fft, "FFT")

    def setup_grafico_temporal(self):
        self.ax_time = self.fig_time.add_subplot(111)
        self.fig_time.patch.set_facecolor('#2b2b2b'); self.ax_time.set_facecolor('#2b2b2b'); self.ax_time.tick_params(colors='#ffffff')
        self.ax_time.set_xlim(0, self.x_size); self.ax_time.set_yticks([])
        for spine in self.ax_time.spines.values(): spine.set_color('#555555')
        colors = ['#00bcd4', '#ff4081', '#71c671', '#e8c346', '#e68136', '#8959a8', '#d84e4e', '#8c564b']
        self.lines_time = []; self.rms_texts = []
        for i in range(self.n_channels):
            l, = self.ax_time.plot([],[], lw=1.2, color=colors[i%8]); self.lines_time.append(l)
            self.rms_texts.append(self.ax_time.text(self.x_size+10, 0, "", fontsize=9, color='#ffffff'))
        self.atualizar_limites_temporal()

    def setup_grafico_fft(self):
        self.ax_fft = self.fig_fft.add_subplot(111)
        self.fig_fft.patch.set_facecolor('#2b2b2b'); self.ax_fft.set_facecolor('#2b2b2b')
        self.ax_fft.tick_params(colors='#ffffff', which='both'); self.ax_fft.set_yscale('log')
        self.ax_fft.set_ylim(0.1, 100); self.ax_fft.set_xlim(0, 60); self.ax_fft.grid(True, which='both', color='#444444', alpha=0.8)
        self.ax_fft.set_xlabel('Freq (Hz)', color='#aaaaaa'); self.ax_fft.set_ylabel('uV', color='#aaaaaa')
        for spine in self.ax_fft.spines.values(): spine.set_color('#555555')
        colors = ['#00bcd4', '#ff4081', '#71c671', '#e8c346', '#e68136', '#8959a8', '#d84e4e', '#8c564b']
        self.lines_fft = [self.ax_fft.plot([],[], lw=1.5, alpha=0.8, color=colors[i%8])[0] for i in range(self.n_channels)]

    def atualizar_limites_temporal(self):
        top = self.n_channels * self.escala_visual
        self.ax_time.set_ylim(-self.escala_visual, top + self.escala_visual)

    def mudar_smoothfactor(self):
        self.aquisicao.fft_smooth_factor = self.spin_smooth.value()

    # ---------------- MONITORIZAÇÃO DE DADOS ----------------
    def update_loop_continuo(self):
        if self.radio_sim.isChecked():
            data = np.random.randn(3, self.n_channels) * 50 
            self.current_data_visual = np.roll(self.current_data_visual, -3, axis=0)
            self.current_data_visual[-3:, :] = data
        elif self.radio_csv.isChecked():
            pass 
        elif self.radio_lsl.isChecked():
            if not getattr(self.aquisicao, 'conectado', False): return 
            chunk = self.aquisicao.adquirir()
            if chunk is None: return

        if len(self.aquisicao.current_data) < self.x_size: return
        self.atualizar_graficos_visuais()

    def atualizar_dados_offline(self):
        fs_atual = self.spin_fs.value()
        intervalo_ms = 20
        chunk_size = max(1, int(fs_atual * (intervalo_ms / 1000.0))) 
        
        if self.ponteiro_arquivo + chunk_size < len(self.dados_arquivo):
            novos_dados = self.dados_arquivo[self.ponteiro_arquivo : self.ponteiro_arquivo + chunk_size]
            if len(self.aquisicao.current_data) == 0:
                self.aquisicao.current_data = np.array(novos_dados)
            else:
                self.aquisicao.current_data = np.vstack((self.aquisicao.current_data, novos_dados))
            if len(self.aquisicao.current_data) > self.aquisicao.max_len:
                self.aquisicao.current_data = self.aquisicao.current_data[-self.aquisicao.max_len:]
                
            self.aquisicao.len_data = len(self.aquisicao.current_data)
            self.ponteiro_arquivo += chunk_size
        else:
            self.timer_atualizacao_offline.stop()

    def atualizar_graficos_visuais(self):
        try:
            self.current_data_visual = self.aquisicao.current_data.copy()
            if len(self.current_data_visual.shape) < 2 or self.current_data_visual.shape[0] < self.x_size or self.current_data_visual.shape[1] < self.n_channels:
                return 

            if self.tabs_graficos.currentIndex() == 0: 
                if self.escala_auto:
                    amp = np.ptp(self.current_data_visual, axis=0).max()
                    if amp > 1: self.escala_visual = amp * 0.8; self.atualizar_limites_temporal()
                x = np.arange(self.x_size)
                for i, l in enumerate(self.lines_time):
                    off = i * self.escala_visual
                    recorte_y = self.current_data_visual[-self.x_size:, i]
                    y = recorte_y - np.mean(recorte_y)
                    l.set_data(x, y + off)
                    rms = np.sqrt(np.mean(y**2))
                    self.rms_texts[i].set_text(f"{rms:.2f} uVrms"); self.rms_texts[i].set_position((self.x_size+10, off))
                self.can_time.draw_idle()
                
            elif self.tabs_graficos.currentIndex() == 1: 
                fs_atual = self.spin_fs.value() 
                xf = np.linspace(0, fs_atual/2, self.x_size//2)
                if self.aquisicao.fft_buffer_history.shape[1] == self.x_size//2:
                    for i, l in enumerate(self.lines_fft):
                        l.set_data(xf, self.aquisicao.fft_buffer_history[i])
                    self.can_fft.draw_idle()
        except Exception:
            pass 

    # ---------------- PROTOCOLO BCI E EVENTOS ----------------
    def obter_nomes_classes(self):
        return [le.text().strip() if le.text().strip() else f"Classe {i}" for i, le in enumerate(self.lista_lineedits)]

    def abrir_gravacao_paradigma(self):
        nomes = self.obter_nomes_classes()
        if not nomes:
            return QMessageBox.warning(self, "Aviso", "Adicione pelo menos 1 classe no Gabarito!")
            
        win_config = JanelaConfiguracaoParadigma(
            self.unity is not None, 
            nomes, 
            playback_mode=self.radio_csv.isChecked()
        )
        
        if win_config.exec_() == QDialog.Accepted:
            self.paradigma_win = JanelaExecucaoParadigma(win_config.configs, unity_sender=self.unity)
            self.paradigma_win.sinal_extrair_dado.connect(self.processar_epoca_ia)
            self.paradigma_win.sinal_iniciar_pausa.connect(self.iniciar_pausa_tecnica)
            self.paradigma_win.sessao_concluida.connect(self.finalizar_sessao)
            self.paradigma_win.show()

    def iniciar_sessao_ml(self):
        if self.paradigma_win is None or not self.paradigma_win.isVisible():
            return QMessageBox.warning(self, "Aviso", "Configure o Protocolo (PASSO 1) primeiro!")
            
        if not self.unity and not self.radio_sim.isChecked():
            if QMessageBox.question(self, "Aviso", "Unity desligado. Continuar?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.No: return
        if not self.model or not self.worker_ia:
            if QMessageBox.question(self, "Aviso", "Nenhum modelo IA carregado. Simular predições?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.No: return
            
        if self.radio_lsl.isChecked() and not getattr(self.aquisicao, 'conectado', False):
            return QMessageBox.warning(self, "Aviso", "LSL ativo mas placa não conectada.")
        if self.radio_csv.isChecked() and self.dados_arquivo is None:
            return QMessageBox.warning(self, "Aviso", "Playback ativo mas CSV não carregado.")

        self.dados_guardados = []; self.marcacoes = []; self.buffer_dados_treino = []; self.buffer_labels_treino = []
        self.ocorreu_transfer_learning = False
        
        self.btn_iniciar_ia.setEnabled(False)
        self.btn_iniciar_ia.setText("A preparar buffer...")
        
        self.timer_largada = QTimer()
        self.timer_largada.timeout.connect(self.checar_largada)
        self.timer_largada.start(100)

    def checar_largada(self):
        target = self.spin_shape_time.value()
        if len(self.aquisicao.current_data) >= target or self.radio_sim.isChecked() or self.radio_csv.isChecked():
            self.timer_largada.stop()
            self.lbl_fase.setText("A INICIAR SESSÃO...")
            self.btn_iniciar_ia.setText("Sessão ativa")
            
            if self.radio_csv.isChecked():
                self.timer_atualizacao_offline = QtCore.QTimer()
                self.timer_atualizacao_offline.timeout.connect(self.atualizar_dados_offline)
                self.timer_atualizacao_offline.start(20) 
                
            self.paradigma_win.iniciar_paradigma()

    def processar_epoca_ia(self, label_real):
        target_time = self.spin_shape_time.value()
        target_ch = self.spin_shape_ch.value()
        
        dados_brutos = self.aquisicao.current_data[-target_time:, :target_ch]
        dados_norm = (dados_brutos - dados_brutos.min()) / (dados_brutos.max() - dados_brutos.min() + 1e-8)
        
        self.label_esperado = label_real

        if self.worker_ia and self.worker_ia.rodando:
            self.worker_ia.pedir_predicao(np.array([dados_norm]))
        else:
            nomes = self.obter_nomes_classes()
            self.receber_resposta_ia(random.randint(0, len(nomes)-1), [0.33, 0.33, 0.34], label_real)
            
        if self.paradigma_win.fase_atual == "CALIBRACAO":
            self.buffer_dados_treino.append(dados_norm)
            self.buffer_labels_treino.append(label_real)

    def receber_resposta_ia(self, pred, prob, label_real_backup=None):
        label_real = getattr(self, 'label_esperado', label_real_backup)
        if label_real is None: return

        nomes = self.obter_nomes_classes()
        nome_predito = nomes[pred] if pred < len(nomes) else f"Classe {pred}"
        cores_dinamicas = ["#00bcd4", "#ff4081", "#ffeb3b", "#8bc34a", "#ff9800", "#9c27b0", "#ffffff"]
        cor = cores_dinamicas[pred] if pred < len(cores_dinamicas) else "#ffffff"

        self.lbl_predicao.setText(nome_predito)
        self.lbl_predicao.setStyleSheet(f"color: {cor}")
        
        p0 = prob[0] if len(prob) > 0 else 0.0
        p1 = prob[1] if len(prob) > 1 else 0.0
        p2 = prob[2] if len(prob) > 2 else 0.0
        self.gauge.set_probabilities(p0, p1, p2)

        if self.unity:
            if pred == 0: self.unity.send("HAND_LEFT")
            elif pred == 1: self.unity.send("HAND_RIGHT")
            elif pred == 2: self.unity.send("HAND_REST")
            else: self.unity.send(f"CUE_{pred}")

        if self.salvar_dados:
            new_chunk = self.aquisicao.len_data if len(self.dados_guardados) == 0 else self.aquisicao.new_len
            self.marcacoes.append([len(self.dados_guardados), len(self.dados_guardados) + new_chunk, pred, label_real])
            self.dados_guardados += self.aquisicao.current_data.copy()[self.aquisicao.len_data - new_chunk:, :].tolist()

        self.label_esperado = None

    def iniciar_pausa_tecnica(self):
        self.lbl_fase.setText("A TREINAR MODELO (PAUSA TÉCNICA)...")
        self.lbl_fase.setStyleSheet("color: #ff9800;")
        self.ocorreu_transfer_learning = True
        if self.worker_ia:
            self.worker_ia.sinal_treino_concluido.connect(self.retornar_da_pausa)
            self.worker_ia.iniciar_transfer_learning(np.array(self.buffer_dados_treino), np.array(self.buffer_labels_treino))
        else:
            sleep(2)
            self.retornar_da_pausa() 

    def retornar_da_pausa(self):
        if self.worker_ia:
            try: 
                self.worker_ia.sinal_treino_concluido.disconnect(self.retornar_da_pausa)
            except TypeError: 
                pass
        self.lbl_fase.setText("A RETOMAR: TESTE PRÁTICO")
        self.lbl_fase.setStyleSheet("color: #00e676;")
        self.paradigma_win.retomar_paradigma()

    def finalizar_sessao(self):
        if hasattr(self, 'timer_atualizacao_offline'): 
            self.timer_atualizacao_offline.stop()
        self.lbl_fase.setText("SESSÃO CONCLUÍDA")
        self.btn_iniciar_ia.setEnabled(True)
        self.btn_iniciar_ia.setText("▶ PASSO 2: INICIAR SESSÃO")
        if self.unity: 
            self.unity.send("HAND_REST")

        if self.salvar_dados:
            self.janela_final = DialogoFimSessao(self.model, self, self.ocorreu_transfer_learning)
            self.janela_final.show()
        else:
            QMessageBox.information(self, "Fim", "Sessão concluída (Nenhum dado guardado).")

    # ---------------- CONEXÕES E CARREGAMENTO DE MODELOS ----------------
    def abrir_arquivo_csv(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Abrir Ficheiro Offline', '', "Ficheiros CSV (*.csv)")
        if fname:
            try:
                df = pd.read_csv(fname, comment='%')
                self.dados_arquivo = df.iloc[:, 1 : self.n_channels + 1].values
                self.ponteiro_arquivo = 0
                QMessageBox.information(self, "Sucesso", "CSV carregado para Playback!")
                self.radio_csv.setChecked(True)
            except Exception: 
                QMessageBox.critical(self, "Erro", "Erro no ficheiro CSV.")

    def conectar_Unity(self):
        try:
            self.unity = UnitySender()
            self.lbl_unity.setText("Ligado")
            self.lbl_unity.setStyleSheet(f"color: {self.palette_verde.color(QtGui.QPalette.WindowText).name()};")
        except Exception: 
            pass

    def conectar_LSL(self):
        self.lbl_lsl.setText('A procurar...')
        self.lbl_lsl.setStyleSheet(f"color: {self.palette_amarela.color(QtGui.QPalette.WindowText).name()};")
        QApplication.processEvents()
        self.aquisicao = Aquisicao(len_data=self.len_data, num_canais=self.n_channels, xlim_FFT=self.x_size//2, smooth_factor=self.fft_smooth_factor)
        self.aquisicao.channel_labels_by_file('gui\\channel_labels.txt')
        self.aquisicao.conectar()
        if self.aquisicao.conectado:
            self.inlet = self.aquisicao.inlet
            self.lbl_lsl.setText('Ligado!')
            self.lbl_lsl.setStyleSheet(f"color: {self.palette_verde.color(QtGui.QPalette.WindowText).name()};")
            self.radio_lsl.setChecked(True)
            
        else:
            self.lbl_lsl.setText('Falha LSL')
            self.lbl_lsl.setStyleSheet(f"color: {self.palette_vermelha.color(QtGui.QPalette.WindowText).name()};")

    def abrir_modelo(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Abrir Ficheiro IA', '../', "Model files (*.h5)")
        if fname:
            try:
                from keras.models import load_model
                from keras.optimizers import Adam
                self.model = load_model(fname)
                self.model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
                
                out_shape = self.model.output_shape
                self.modelo_infos.setText(f"Shape: {out_shape}")
                self.modelo_infos.setStyleSheet(f"color: {self.palette_verde.color(QtGui.QPalette.WindowText).name()};")
                
                if isinstance(out_shape, tuple) and len(out_shape) > 1 and out_shape[1] is not None:
                    n_out = out_shape[1]
                    self.clear_class_ui()
                    for i in range(n_out):
                        self.add_class_ui(f"Classe {i}")
                
                if self.worker_ia: 
                    self.worker_ia.parar()
                
                is_binary = (self.combo_tipo_modelo.currentIndex() == 1)
                self.worker_ia = WorkerIA(self.model, is_binary)
                self.worker_ia.sinal_predicao.connect(self.receber_resposta_ia)
                self.worker_ia.start()
                
            except Exception as e: 
                self.modelo_infos.setText("Erro ao carregar")
                self.modelo_infos.setStyleSheet(f"color: {self.palette_vermelha.color(QtGui.QPalette.WindowText).name()};")
                print(f"Erro ao carregar modelo: {e}")

    def closeEvent(self, event):
        if self.worker_ia: 
            self.worker_ia.parar()
            self.worker_ia.wait()
        event.accept()

# =============================================================================
# JANELA DE FIM DE SESSÃO (Gravação segura e Inteligente)
# =============================================================================
class DialogoFimSessao(QDialog):
    def __init__(self, model, hub_principal, ocorreu_tl):
        super().__init__()
        self.model = model
        self.hub = hub_principal
        self.ocorreu_tl = ocorreu_tl
        self.setWindowTitle('Guardar Sessão do Paciente')
        self.resize(500, 350)
        self.layout = QVBoxLayout(self)
        
        self.label = QLabel("Sessão concluída!")
        self.label.setFont(QtGui.QFont('Arial', 16))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.label)
        
        layoutshape = QFormLayout()
        self.line_paciente = QLineEdit(self)
        self.line_paciente.setPlaceholderText("Ex: Joao_Silva")
        layoutshape.addRow("Nome do Paciente:", self.line_paciente)

        self.epocas_separadas_checkBox = QCheckBox(self)
        self.epocas_separadas_checkBox.setChecked(True)
        layoutshape.addRow('Guardar Épocas Separadas (Pastas por Classe):', self.epocas_separadas_checkBox)
        
        self.salvar_apenas_acertos_checkBox = QCheckBox(self)
        layoutshape.addRow('Guardar Apenas Acertos (Predict = Label Real):', self.salvar_apenas_acertos_checkBox)
        self.layout.addLayout(layoutshape)
        
        self.button_salvar_dados = QPushButton("Guardar Registo, Dados Brutos e Modelo")
        self.button_salvar_dados.setStyleSheet("background-color: #00bcd4; font-weight: bold; padding: 12px; color: black;")
        self.button_salvar_dados.clicked.connect(self.salvar_dados)
        self.layout.addWidget(self.button_salvar_dados)

    def salvar_dados(self):
        paciente = self.line_paciente.text().strip()
        if not paciente: 
            paciente = "Paciente_Anonimo"
            
        data_str = datetime.now().strftime("%Y%m%d_%H%M")
        pasta_raiz = f"Sessao_{paciente}_{data_str}"

        nomes_finais = self.hub.obter_nomes_classes()
        num_outputs = len(nomes_finais)
        target_time = int(self.hub.spin_shape_time.value())

        pasta_selecionada = str(QFileDialog.getExistingDirectory(self, "Selecione o Diretório Principal"))
        if not pasta_selecionada: return
        
        caminho_completo = os.path.join(pasta_selecionada, pasta_raiz)
        os.makedirs(caminho_completo, exist_ok=True)

        if self.epocas_separadas_checkBox.isChecked():
            dados_separados = [[] for _ in range(num_outputs)]  
            
            for m in self.hub.marcacoes:
                start_idx, end_idx, pred, label_real = m
                if self.salvar_apenas_acertos_checkBox.isChecked() and int(pred) != int(label_real): 
                    continue
                    
                if 0 <= int(label_real) < num_outputs:
                    if int(end_idx) >= target_time:
                        epoch_bruta = self.hub.dados_guardados[int(end_idx) - target_time : int(end_idx)]
                        dados_separados[int(label_real)].append(epoch_bruta)
                    
            for i in range(num_outputs):
                nome_classe = nomes_finais[i] if i < num_outputs else str(i)
                pasta_classe = os.path.join(caminho_completo, f"output_{nome_classe}")
                
                if not os.path.exists(pasta_classe): 
                    os.makedirs(pasta_classe)
                for j, epoch in enumerate(dados_separados[i]):
                    ep_np = np.array(epoch)
                    ep_norm = (ep_np - ep_np.min()) / (ep_np.max() - ep_np.min() + 1e-8)
                    np.savetxt(os.path.join(pasta_classe, f"epoch_{j}.txt"), ep_norm)
        else:
            fname = os.path.join(caminho_completo, f"{paciente}_EEG_Agrupado.txt")
            with open(fname, "w") as f:
                for item in self.hub.dados_guardados: 
                    f.write("%s\n" % item)

        if self.ocorreu_tl and self.model:
            caminho_modelo = os.path.join(caminho_completo, f"Modelo_Adaptado_{paciente}.h5")
            self.model.save(caminho_modelo)
            mensagem_modelo = f"\n\n🤖 Novo Modelo Guardado:\nModelo_Adaptado_{paciente}.h5"
        else:
            mensagem_modelo = "\n\n(O modelo não foi alterado, pois não ocorreu Transfer Learning)."

        QMessageBox.information(self, "Sessão Concluída", f"Dados do paciente guardados em:\n{caminho_completo}{mensagem_modelo}")
        self.accept()

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    win = JanelaInicial()
    win.show()
    sys.exit(app.exec_())