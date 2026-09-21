import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

from dependencias import *
from aquisicao import Aquisicao
from unitysender import UnitySender
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QElapsedTimer
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QPolygonF, QFont
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QListWidget, QListWidgetItem, 
                             QPushButton, QLabel, QHBoxLayout, QMessageBox, 
                             QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox, 
                             QSpinBox, QLineEdit, QCheckBox, QRadioButton, 
                             QFileDialog, QWidget, QMainWindow, QFrame, QTabWidget, QApplication)
from time import sleep
import pandas as pd
import random
from datetime import datetime
import numpy as np

# BIBLIOTECA CIENTÍFICA PARA RESAMPLING (Proteção contra Efeito Fast-Forward)
from scipy.signal import resample

# =============================================================================
# TEMA VISUAL UNIFICADO (paleta única, inspirada no OpenBCI GUI)
# Centraliza todas as cores usadas na interface para manter coesão visual
# e para que qualquer ajuste de paleta seja feito em um único lugar.
# =============================================================================
class Tema:
    BG = "#14181c"
    PAINEL = "#1b2126"
    BORDA = "#2a323a"
    TEXTO = "#e6e9ec"
    TEXTO_MUTED = "#8b96a1"
    ESQUERDA = "#00bcd4"   # ciano — usado em todo lugar p/ classe "esquerda"
    DIREITA = "#ff4081"    # rosa/magenta — usado em todo lugar p/ classe "direita"
    REPOUSO = "#ffc93c"    # amarelo — usado em todo lugar p/ classe "repouso"
    OK = "#00e676"
    ALERTA = "#ff9800"
    ERRO = "#ff5252"
    NEUTRO = "#888888"

# QSS aplicado uma única vez na janela principal para dar aparência
# consistente de "painéis/cards" (bordas, títulos) em todos os GroupBox/Tabs,
# substituindo estilos inline repetidos espalhados pelo código.
QSS_PAINEIS = f"""
QGroupBox {{
    background-color: {Tema.PAINEL};
    border: 1px solid {Tema.BORDA};
    border-radius: 6px;
    margin-top: 12px;
    font-weight: bold;
    color: {Tema.TEXTO};
    padding-top: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {Tema.ESQUERDA};
}}
QTabWidget::pane {{
    border: 1px solid {Tema.BORDA};
    background-color: {Tema.BG};
}}
QTabBar::tab {{
    background-color: {Tema.PAINEL};
    color: {Tema.TEXTO_MUTED};
    padding: 6px 14px;
    border: 1px solid {Tema.BORDA};
    border-bottom: none;
}}
QTabBar::tab:selected {{
    background-color: {Tema.BG};
    color: {Tema.ESQUERDA};
    font-weight: bold;
}}
"""

def estilo_status(cor):
    """Estilo único e reutilizável para labels de status (LSL/Unity/IA),
    evitando criar objetos QPalette repetidos para a mesma finalidade."""
    return f"color: {cor}; font-weight: bold;"

# =============================================================================
# WIDGET 1: VELOCÍMETRO (Probabilidade Instantânea)
# =============================================================================
class GaugeWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(300, 150) 
        self.current_angle = 0.0
        self.target_angle = 0.0
        self.probs = [0.0, 0.0, 0.0]  
        
        # Motor de desenho a ~30ms de intervalo nominal. A velocidade da
        # animação NÃO depende mais desse valor nominal: usamos QElapsedTimer
        # para medir o dt real entre chamadas, então a suavização fica correta
        # mesmo se o timer atrasar (jitter do SO/Qt, carga na thread da GUI).
        self._relogio = QElapsedTimer()
        self._relogio.start()
        self.anim_timer = QtCore.QTimer(self)
        self.anim_timer.timeout.connect(self.update_animation)
        self.anim_timer.start(30) 

    def set_probabilities(self, prob_left, prob_right, prob_rest):
        self.target_angle = (prob_right * 60) + (prob_left * -60)
        self.probs = [prob_left, prob_right, prob_rest]

    def reset_position(self):
        self.target_angle = 0.0
        self.probs = [0.0, 0.0, 0.0]

    def update_animation(self):
        dt_ms = self._relogio.restart()
        # Fator de suavização calibrado para ~30ms; escalado pelo dt real
        # para que a velocidade angular seja igual independente de jitter.
        fator = 1.0 - (0.95 ** (dt_ms / 30.0)) if dt_ms > 0 else 0.05
        diff = self.target_angle - self.current_angle
        if abs(diff) > 0.1:
            self.current_angle += diff * fator
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        center_x, center_y = w / 2, h - 30 
        radius = min(w / 2, h) - 30

        rect = QtCore.QRectF(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        painter.setPen(QPen(QColor(Tema.ESQUERDA), 15, QtCore.Qt.SolidLine, QtCore.Qt.FlatCap))
        painter.drawArc(rect, 120 * 16, 60 * 16) 
        painter.setPen(QPen(QColor(Tema.NEUTRO), 15, QtCore.Qt.SolidLine, QtCore.Qt.FlatCap))
        painter.drawArc(rect, 60 * 16, 60 * 16)
        painter.setPen(QPen(QColor(Tema.DIREITA), 15, QtCore.Qt.SolidLine, QtCore.Qt.FlatCap))
        painter.drawArc(rect, 0 * 16, 60 * 16)

        painter.save()
        painter.translate(center_x, center_y)
        painter.rotate(self.current_angle)
        
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QBrush(QColor(Tema.TEXTO)))
        poly = QPolygonF([QtCore.QPointF(-4, 0), QtCore.QPointF(4, 0), QtCore.QPointF(0, -radius + 5)])
        painter.drawPolygon(poly)
        painter.setBrush(QBrush(QColor(Tema.REPOUSO)))
        painter.drawEllipse(QtCore.QPointF(-3, -3), 6, 6)
        painter.restore()

# =============================================================================
# WIDGET 2: CUBO ACUMULATIVO 1D (O Cabo de Guerra)
# =============================================================================
class CubeFeedbackWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(300, 150)
        self.center_x = 150
        self.current_x = 150
        self.target_direction = 0 
        self.confianca = 1.0  # 0..1, escala a velocidade proporcionalmente

        # Base de velocidade/decaimento calibrada para um passo de 30ms;
        # o dt real é medido via QElapsedTimer para que o movimento não
        # dependa da precisão do QTimer (evita variação de velocidade
        # perceptível quando a thread da GUI atrasa o disparo do timer).
        self._speed_base_px_ms = 5.0 / 30.0
        self._decay_base_px_ms = 1.5 / 30.0
        self._relogio = QElapsedTimer()
        self._relogio.start()

        # O Motor de Desenho: Desliza o cubo sem travar
        self.anim_timer = QtCore.QTimer(self)
        self.anim_timer.timeout.connect(self.update_physics)
        self.anim_timer.start(30)

    def reset_position(self):
        self.current_x = self.center_x
        self.target_direction = 0
        self.confianca = 1.0

    def set_decision(self, decision, confianca=1.0):
        """confianca (0..1): magnitude da predição (ex: WMA), usada para
        que o cubo se mova mais rápido quanto mais confiante a IA estiver,
        em vez de sempre à mesma velocidade fixa (informação antes perdida)."""
        self.confianca = max(0.0, min(1.0, confianca))
        if decision == "ESQUERDA": self.target_direction = -1
        elif decision == "DIREITA": self.target_direction = 1
        else: self.target_direction = 0

    def update_physics(self):
        dt_ms = self._relogio.restart()
        dt_ms = min(dt_ms, 100)  # evita "saltos" grandes após pausas longas (ex: dialogs)

        speed = self._speed_base_px_ms * dt_ms * (0.4 + 0.6 * self.confianca)
        decay = self._decay_base_px_ms * dt_ms

        if self.target_direction == -1: self.current_x -= speed
        elif self.target_direction == 1: self.current_x += speed
        else:
            if self.current_x > self.center_x + decay: self.current_x -= decay
            elif self.current_x < self.center_x - decay: self.current_x += decay
            else: self.current_x = self.center_x

        if self.current_x < 20: self.current_x = 20
        if self.current_x > 280: self.current_x = 280
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        
        painter.setPen(QPen(QColor(Tema.BORDA), 4))
        painter.drawLine(20, h//2, w-20, h//2)
        
        painter.setPen(QPen(QColor(Tema.TEXTO), 2))
        painter.drawLine(w//2, h//2 - 15, w//2, h//2 + 15)
        
        painter.setPen(Qt.NoPen)
        if self.current_x < self.center_x - 15: painter.setBrush(QBrush(QColor(Tema.ESQUERDA)))
        elif self.current_x > self.center_x + 15: painter.setBrush(QBrush(QColor(Tema.DIREITA)))
        else: painter.setBrush(QBrush(QColor(Tema.REPOUSO)))
        
        painter.drawRect(int(self.current_x - 15), h//2 - 15, 30, 30)

# =============================================================================
# WIDGET 3: CONSOLE DE TELEMETRIA (Exclusivo do Pesquisador)
# =============================================================================
class TelemetryWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(350, 100)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_explicacao = QLabel("🔍 CONJUNTO DE PREDIÇÕES DA AÇÃO (a cada 250ms):")
        self.lbl_explicacao.setStyleSheet(f"color: {Tema.ESQUERDA}; font-weight: bold; font-size: 11px;")

        self.lbl_log = QLabel("[ Aguardando... ]")
        self.lbl_log.setWordWrap(True)

        self.lbl_wma = QLabel("Média Ponderada: --")

        # Diagnóstico de tempo real: quantas épocas foram sobrescritas antes
        # de serem processadas pela IA (sinal de que o modelo está mais lento
        # que a cadência de 250ms da janela deslizante).
        self.lbl_diagnostico = QLabel("Épocas puladas (IA lenta): 0")

        for lbl in [self.lbl_explicacao, self.lbl_log, self.lbl_wma, self.lbl_diagnostico]:
            lbl.setStyleSheet(lbl.styleSheet() + f"font-family: Consolas, monospace; background-color: {Tema.PAINEL}; color: {Tema.TEXTO}; padding: 4px; border-radius: 4px;")
            layout.addWidget(lbl)

    def update_telemetry(self, log_lista, wma_prob, nomes_classes, epocas_puladas=0):
        log_str = " ➔ ".join(log_lista) if log_lista else "[ Vazio ]"
        self.lbl_log.setText(f"Sequência RAW: {log_str}")

        if wma_prob and nomes_classes:
            wma_str = " | ".join([f"{nomes_classes[i][:3].upper()}: {p*100:02.0f}%" for i, p in enumerate(wma_prob)])
            self.lbl_wma.setText(f"Média Atual: [{wma_str}]")
        else:
            self.lbl_wma.setText("Média Atual: --")

        cor_diag = Tema.ERRO if epocas_puladas > 0 else Tema.OK
        self.lbl_diagnostico.setText(f"Épocas puladas (IA lenta): {epocas_puladas}")
        self.lbl_diagnostico.setStyleSheet(
            f"font-family: Consolas, monospace; background-color: {Tema.PAINEL}; color: {cor_diag}; padding: 4px; border-radius: 4px; font-weight: bold;"
        )

# =============================================================================
# WORKER IA 
# =============================================================================
class WorkerIA(QThread):
    sinal_predicao = pyqtSignal(int, list, bool, object) 
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

        # Diagnóstico de tempo real (ver correção do bug de época perdida):
        # conta quantas épocas foram sobrescritas por uma nova antes de serem
        # processadas (indica que o predict() está mais lento que 250ms) e
        # guarda a duração da última inferência para calibração/monitoramento.
        self.epocas_puladas = 0
        self.ultimo_tempo_predict_ms = 0.0
        self._relogio_predict = QElapsedTimer()

    def run(self):
        while self.rodando:
            if self.modo_treino and self.dados_treino is not None:
                try: self.model.train_on_batch(self.dados_treino, self.labels_treino)
                except Exception as e: print(f"❌ Erro no treino: {e}")
                self.modo_treino = False
                self.dados_treino = None
                self.sinal_treino_concluido.emit()
            
            elif not self.modo_treino and self.dados_predicao is not None:
                # CORREÇÃO CRÍTICA: consome (lê e zera) a época ANTES de rodar
                # o predict(). Antes, o "self.dados_predicao = None" rodava só
                # depois do predict() e apagava incondicionalmente qualquer
                # época nova que a GUI tivesse colocado ali enquanto o predict
                # antigo ainda estava em andamento — descartando silenciosamente
                # janelas inteiras sempre que a inferência excedesse os 250ms
                # do timer de janela deslizante.
                dados, is_training, label_real = self.dados_predicao
                self.dados_predicao = None
                try:
                    self._relogio_predict.start()
                    res = self.model.predict(dados, verbose=0)[0]
                    self.ultimo_tempo_predict_ms = self._relogio_predict.elapsed()
                    if self.is_binary:
                        val = res[0]
                        pred = 0 if val < 0.5 else 1
                        prob = [1.0 - val, val]
                    else:
                        pred = int(np.argmax(res))
                        prob = res.tolist()
                    self.sinal_predicao.emit(pred, prob, is_training, label_real)
                except Exception as e:
                    print(f"❌ ERRO IA: {e}")
            
            sleep(0.01)

    def pedir_predicao(self, dados, is_training=False, label_real=None):
        if not self.modo_treino:
            if self.dados_predicao is not None:
                # Havia uma época pendente que nunca chegou a ser processada
                # (predict() ainda não tinha começado a consumi-la): conta
                # como pulada para diagnóstico em tela.
                self.epocas_puladas += 1
            self.dados_predicao = (dados, is_training, label_real)

    def iniciar_transfer_learning(self, dados, labels):
        self.dados_treino = dados
        self.labels_treino = labels
        self.modo_treino = True 

    def parar(self):
        self.rodando = False

class DialogoSelecaoCanaisIA(QDialog):
    def __init__(self, canais_disponiveis, canais_selecionados_atuais):
        super().__init__()
        self.setWindowTitle("Configurar Canais para IA")
        self.resize(350, 500)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        layout = QVBoxLayout(self)
        lbl = QLabel("Selecione os eletrodos:")
        lbl.setStyleSheet("font-weight: bold; color: #00bcd4;")
        layout.addWidget(lbl)
        self.lista = QListWidget()
        self.lista.setStyleSheet("background-color: #3b3b3b; color: white; font-size: 14px;")
        limpos_selecionados = [str(c).strip().upper() for c in canais_selecionados_atuais]
        for ch in canais_disponiveis:
            item = QListWidgetItem(str(ch))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            if str(ch).strip().upper() in limpos_selecionados: item.setCheckState(Qt.Checked)
            else: item.setCheckState(Qt.Unchecked)
            self.lista.addItem(item)
        layout.addWidget(self.lista)
        btn_box = QHBoxLayout()
        btn_ok = QPushButton("Aplicar"); btn_ok.setStyleSheet("background-color: #2e7d32; font-weight: bold; padding: 8px;"); btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancelar"); btn_cancel.setStyleSheet("background-color: #d32f2f; font-weight: bold; padding: 8px;"); btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_ok); btn_box.addWidget(btn_cancel)
        layout.addLayout(btn_box)

    def obter_selecionados(self):
        selecionados = []
        for i in range(self.lista.count()):
            item = self.lista.item(i)
            if item.checkState() == Qt.Checked: selecionados.append(item.text())
        return selecionados

class JanelaConfiguracaoParadigma(QDialog):
    def __init__(self, unity_conectado, nomes_classes, playback_mode=False):
        super().__init__()
        self.nomes_classes = nomes_classes
        self.num_classes = len(nomes_classes)
        self.playback_mode = playback_mode
        self.setWindowTitle("Configuração da Sessão")
        self.resize(500, 600)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        layout = QVBoxLayout(self)

        lbl_titulo = QLabel(f"Sessão BCI ({self.num_classes} Classes Detectadas)")
        lbl_titulo.setStyleSheet("font-size: 16px; font-weight: bold; color: #00bcd4;")
        lbl_titulo.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(lbl_titulo)

        # O SELETOR DE MODO CEGO / CUBO / VELOCÍMETRO
        group_fb = QGroupBox("Feedback Visual na Tela do Paciente")
        layout_fb = QVBoxLayout()
        self.combo_fb = QComboBox()
        self.combo_fb.addItems([
            "Cubo 1D Acumulativo (Recomendado)", 
            "Velocímetro (Probabilidade Instantânea)",
            "Nenhum (Modo Cego - Apenas Setas)"
        ])
        layout_fb.addWidget(self.combo_fb)
        group_fb.setLayout(layout_fb)
        layout.addWidget(group_fb)

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

        group_tl = QGroupBox("Calibração (Transfer Learning)")
        form_tl = QFormLayout()
        self.spin_calibracao = QSpinBox(); self.spin_calibracao.setRange(0, 200); self.spin_calibracao.setValue(20); self.spin_calibracao.setSuffix(" épocas/classe")
        self.spin_teste = QSpinBox(); self.spin_teste.setRange(1, 200); self.spin_teste.setValue(30); self.spin_teste.setSuffix(" épocas/classe")
        form_tl.addRow("1ª Fase (Calibração):", self.spin_calibracao)
        form_tl.addRow("2ª Fase (Teste):", self.spin_teste)
        group_tl.setLayout(form_tl)
        layout.addWidget(group_tl)

        self.group_gabarito = QGroupBox("Gabarito de Playback")
        self.group_gabarito.setVisible(self.playback_mode)
        layout_gabarito = QVBoxLayout()
        self.line_gabarito = QLineEdit()
        self.line_gabarito.setPlaceholderText("Ex: 0, 1, 2")
        self.btn_carregar_gabarito = QPushButton("📂 Carregar _marcacoes.txt")
        self.btn_carregar_gabarito.clicked.connect(self.carregar_gabarito_arquivo)
        layout_gabarito.addWidget(QLabel("Sequência:"))
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
        if index == 1: self.spin_calibracao.setValue(0)

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
                        if len(parts) >= 4: sequencia.append(int(parts[3].strip()))
                self.line_gabarito.setText(", ".join(map(str, sequencia)))
                QMessageBox.information(self, "Sucesso", "Gabarito extraído com sucesso!")
            except Exception:
                pass

    def aceitar_configuracao(self):
        if not self.chk_python.isChecked() and not self.chk_unity.isChecked():
            return QMessageBox.warning(self, "Aviso", "Selecione pelo menos um local para exibir os estímulos!")
        
        gabarito_forçado = None
        if self.playback_mode:
            g_str = self.line_gabarito.text().replace(' ', '')
            if not g_str: return QMessageBox.warning(self, "Aviso", "O Playback exige a sequência!")
            try:
                gabarito_forçado = [int(x) for x in g_str.split(',') if x != '']
                max_val = max(gabarito_forçado)
                if max_val >= self.num_classes: return QMessageBox.warning(self, "Erro", "Gabarito inválido!")
            except ValueError:
                return QMessageBox.warning(self, "Erro", "Sequência inválida!")

        self.configs = {
            't_aviso': int(self.spin_aviso.value() * 1000), 't_acao': int(self.spin_acao.value() * 1000), 't_repouso': int(self.spin_repouso.value() * 1000), 
            'rep_calibracao': self.spin_calibracao.value(), 'rep_teste': self.spin_teste.value(),
            'usar_python': self.chk_python.isChecked(), 'usar_unity': self.chk_unity.isChecked(),
            'nomes_classes': self.nomes_classes, 'num_classes': self.num_classes,
            'gabarito_forçado': gabarito_forçado,
            'tipo_feedback': self.combo_fb.currentIndex() # 0 = Cubo, 1 = Velocímetro, 2 = Cego
        }
        self.accept()

class JanelaExecucaoParadigma(QDialog):
    sinal_extrair_dado = pyqtSignal(int, bool) 
    sinal_iniciar_pausa = pyqtSignal()   
    sessao_concluida = pyqtSignal()   
    sinal_inicio_acao = pyqtSignal() 

    def __init__(self, configs, unity_sender=None):
        super().__init__()
        self.configs = configs
        self.unity = unity_sender
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

        # MONTAGEM DA INTERFACE CEGA OU COM FEEDBACK
        if self.configs['tipo_feedback'] == 0:
            self.widget_feedback = CubeFeedbackWidget()
            layout.addWidget(self.widget_feedback, alignment=QtCore.Qt.AlignHCenter)
        elif self.configs['tipo_feedback'] == 1:
            self.widget_feedback = GaugeWidget()
            layout.addWidget(self.widget_feedback, alignment=QtCore.Qt.AlignHCenter)
        else:
            self.widget_feedback = None # NÃO MOSTRA NADA!
        
        self.timer_logica = QTimer(self)
        self.timer_logica.setSingleShot(True)
        self.timer_logica.timeout.connect(self.proximo_estado)
        self.contagem_inicial = 3
        
        # O Motor Lógico que recorta o cérebro a cada 250ms
        self.timer_feedback = QTimer(self)
        self.timer_feedback.timeout.connect(self.enviar_janela_deslizante)

    def iniciar_paradigma(self):
        self.estado_atual = "INICIO"
        self.timer_contagem = QTimer(self)
        self.timer_contagem.timeout.connect(self.rotina_contagem)
        self.timer_contagem.start(1000)
        self.rotina_contagem()

    def rotina_contagem(self):
        if self.contagem_inicial > 0:
            self.desenhar_tela(str(self.contagem_inicial), 150, Tema.ESQUERDA, "Prepare-se...")
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

    def enviar_janela_deslizante(self):
        if self.estado_atual == "ACAO":
            lista_atual = self.seq_calibracao if self.fase_atual == "CALIBRACAO" else self.seq_teste
            classe_alvo = lista_atual[self.trial_atual]
            self.sinal_extrair_dado.emit(classe_alvo, False) # is_training = False

    def proximo_estado(self):
        if self.estado_atual in ["CONCLUIDO", "PAUSA_TECNICA", "STANDBY"]: return

        lista_atual = self.seq_calibracao if self.fase_atual == "CALIBRACAO" else self.seq_teste
        if self.trial_atual >= len(lista_atual): return self.finalizar_forçado()

        classe_alvo = lista_atual[self.trial_atual]
        nome_classe_alvo = self.nomes_classes[classe_alvo]
        nome_lower = nome_classe_alvo.lower()

        if self.estado_atual in ["INICIO", "REPOUSO"]:
            self.estado_atual = "AVISO"
            
            if self.widget_feedback and hasattr(self.widget_feedback, 'reset_position'):
                self.widget_feedback.reset_position()

            if self.configs['usar_unity'] and self.unity: self.unity.send("CUE_CROSS") 
            self.desenhar_tela("➕", 150, "white", f"FASE: {self.fase_atual} | Estímulo {self.trial_atual + 1}/{self.total_fase}")
            self.timer_logica.start(self.configs['t_aviso'])
            
        elif self.estado_atual == "AVISO":
            self.estado_atual = "ACAO"
            
            # Zera o histórico da telemetria para esse novo estímulo
            self.sinal_inicio_acao.emit()

            if "esquerda" in nome_lower: icone = "⬅️"; cor = Tema.ESQUERDA; comando_unity = "CUE_LEFT"
            elif "direita" in nome_lower: icone = "➡️"; cor = Tema.DIREITA; comando_unity = "CUE_RIGHT"
            elif "repouso" in nome_lower: icone = "🛑"; cor = Tema.REPOUSO; comando_unity = "CUE_REST"
            else: icone = f"[{classe_alvo}]"; cor = Tema.TEXTO; comando_unity = f"CUE_{classe_alvo}"

            if self.configs['usar_unity'] and self.unity: self.unity.send(comando_unity)

            self.desenhar_tela(icone, 180, cor, f"AÇÃO: {nome_classe_alvo}")
            
            # ATIVA A JANELA DESLIZANTE
            self.timer_feedback.start(250)
            self.timer_logica.start(self.configs['t_acao'])
            
        elif self.estado_atual == "ACAO":
            self.estado_atual = "REPOUSO"
            
            # DESLIGA A JANELA E TIRA A FOTO FINAL DO CÉREBRO
            self.timer_feedback.stop()
            self.sinal_extrair_dado.emit(classe_alvo, True) 

            if self.configs['usar_unity'] and self.unity: self.unity.send("CUE_CROSS") 
            self.desenhar_tela("", 150, "white", "Descanso...")
            
            self.trial_atual += 1
            if self.trial_atual >= self.total_fase:
                if self.fase_atual == "CALIBRACAO":
                    self.estado_atual = "PAUSA_TECNICA"
                    self.desenhar_tela("☕", 120, "yellow", "Descanso Técnico.\nAguarde...")
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
        self.desenhar_tela("Fim", 60, "yellow", "Arquivo finalizado.")
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
        # Camada de estilo adicional (painéis/abas) para visual coeso entre
        # todas as seções da interface, sem alterar o tema-base já aplicado.
        self.setStyleSheet(self.styleSheet() + QSS_PAINEIS)

        self.unity = None; self.inlet = None; self.model = None; self.worker_ia = None
        self.dados_arquivo = None; self.ponteiro_arquivo = 0
        
        self.canais_ia_default = ['C3', 'C4', 'Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8','T7', 'T8', 'P7', 'P3', 'P4', 'P8', 'O1', 'O2']
        self.canais_ia = self.canais_ia_default.copy()
        self.canais_base = self.canais_ia_default.copy()
        
        self.n_channels = 16 
        self.canais = self.canais_base[:self.n_channels]
        
        self.x_size = 1000 
        self.len_data = 1000
        self.ptr_visual = 0
        
        self.salvar_dados = True
        self.dados_guardados = []; self.marcacoes = []
        self.buffer_dados_treino = []; self.buffer_labels_treino = []
        self.ocorreu_transfer_learning = False
        self.paradigma_win = None
        
        self.historico_probs = [] 
        self.trial_predicts_log = [] 
        
        self.current_data_visual = np.zeros((self.x_size, self.n_channels))
        self.fs = 250.0  
        self.escala_visual = 200; self.escala_auto = False
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

        self.carregar_labels_txt(file_manual=False)

        # Otimização: os status (LSL/Unity/IA) antes criavam 3 objetos QPalette
        # só para extrair uma cor em hexadecimal via .color(...).name(). Isso
        # foi substituído por constantes de string do Tema (mesmas cores),
        # usadas diretamente em setStyleSheet — sem alocar QPalette.

        self.timer_plot = QtCore.QTimer()
        self.timer_plot.timeout.connect(self.update_loop_continuo)
        self.timer_plot.start(40)

    def setup_painel_esquerdo(self):
        lbl_titulo = QLabel("PAINEL DE CONTROLE")
        lbl_titulo.setFont(QtGui.QFont("Segoe UI", 14, QtGui.QFont.Bold))
        lbl_titulo.setAlignment(QtCore.Qt.AlignCenter)
        self.layout_left.addWidget(lbl_titulo)

        self.tabs_controles = QTabWidget()
        self.tab_config = QWidget(); layout_config = QVBoxLayout(self.tab_config)
        
        group_conn = QGroupBox("Módulos & IA")
        form_conn = QFormLayout()
        self.lbl_lsl = QLabel("Desconectado"); self.lbl_lsl.setStyleSheet(estilo_status(Tema.ERRO))
        self.lbl_unity = QLabel("Desconectado"); self.lbl_unity.setStyleSheet(estilo_status(Tema.ERRO))
        self.modelo_infos = QLabel("Nenhum"); self.modelo_infos.setStyleSheet(estilo_status(Tema.TEXTO_MUTED))
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

        group_classes = QGroupBox("Mapeamento de Classes (Gabarito)")
        layout_classes = QVBoxLayout()
        self.form_classes = QFormLayout()
        self.lista_lineedits = []
        
        self.add_class_ui("Repouso"); self.add_class_ui("Mão Esquerda"); self.add_class_ui("Mão Direita")
        
        layout_classes.addLayout(self.form_classes)
        self.btn_add_classe = QPushButton("+ Adicionar Classe")
        self.btn_add_classe.clicked.connect(lambda: self.add_class_ui("Nova Classe"))
        layout_classes.addWidget(self.btn_add_classe)
        group_classes.setLayout(layout_classes)
        layout_config.addWidget(group_classes)

        # CONFIGURAÇÕES AVANÇADAS: Resampling e Referência
        group_shape = QGroupBox("Configuração de Hardware e IA")
        form_shape = QFormLayout()
        
        self.spin_fs = QSpinBox(); self.spin_fs.setRange(1, 10000); self.spin_fs.setValue(250); self.spin_fs.setSuffix(" Hz (Equipamento)")
        self.spin_fs_modelo = QSpinBox(); self.spin_fs_modelo.setRange(1, 10000); self.spin_fs_modelo.setValue(250); self.spin_fs_modelo.setSuffix(" Hz (Modelo IA)")
        self.spin_shape_time = QSpinBox(); self.spin_shape_time.setRange(10, 5000); self.spin_shape_time.setValue(721); self.spin_shape_time.setSuffix(" pts (Tamanho IA)")
        
        self.spin_shape_ch = QSpinBox(); self.spin_shape_ch.setRange(1, 128); self.spin_shape_ch.setValue(16); self.spin_shape_ch.setSuffix(" ch (Visual)")
        self.spin_shape_ch.valueChanged.connect(self.mudar_numero_canais)
        
        self.btn_load_txt = QPushButton("📂 Arquivo de Montagem (.txt)")
        self.btn_load_txt.clicked.connect(self.carregar_labels_txt)
        
        self.btn_canais_ia = QPushButton("🧠 Canais Específicos para a IA")
        self.btn_canais_ia.setStyleSheet("background-color: #3f51b5; font-weight: bold;")
        self.btn_canais_ia.clicked.connect(self.abrir_selecao_canais_ia)

        self.combo_referencia = QComboBox()
        self.combo_referencia.addItems([
            "Nenhuma (Manter original da Placa)",
            "Nihon Kohden (0.55 * [C3 + C4])",
            "Média Bipolar (0.50 * [C3 + C4])",
            "CAR (Média Comum de Todos os Canais)"
        ])
        
        form_shape.addRow("Amostragem LSL:", self.spin_fs)
        form_shape.addRow("Amostragem IA:", self.spin_fs_modelo)
        form_shape.addRow("Shape Time (IA):", self.spin_shape_time)
        form_shape.addRow("Canais Display:", self.spin_shape_ch)
        form_shape.addRow("Montagem Real:", self.btn_load_txt)
        form_shape.addRow("Alimentar Modelo:", self.btn_canais_ia)
        form_shape.addRow("Re-referência IA:", self.combo_referencia) 
        
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

        # PAINEL DE MONITORAMENTO (PESQUISADOR)
        group_mon = QGroupBox("Monitoramento BCI (Pesquisador)")
        layout_mon = QVBoxLayout()
        self.lbl_fase = QLabel("FASE: Parado")
        self.lbl_fase.setStyleSheet(estilo_status(Tema.REPOUSO))
        self.lbl_fase.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_predicao = QLabel("--")
        self.lbl_predicao.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        self.lbl_predicao.setAlignment(QtCore.Qt.AlignCenter)
        
        self.gauge = GaugeWidget()
        self.telemetry = TelemetryWidget()

        layout_mon.addWidget(self.lbl_fase)
        layout_mon.addWidget(self.lbl_predicao)
        layout_mon.addWidget(self.gauge, alignment=QtCore.Qt.AlignHCenter)
        layout_mon.addWidget(self.telemetry)
        
        group_mon.setLayout(layout_mon)
        layout_exp.addWidget(group_mon)
        
        layout_exp.addStretch(); self.tabs_controles.addTab(self.tab_experimento, "Sessão & IA")
        self.layout_left.addWidget(self.tabs_controles)

    # Função Matemática para Re-referenciamento Espacial
    def aplicar_rereferenciamento(self, matriz_dados, canais_selecionados, tipo_ref):
        if tipo_ref == 0: return matriz_dados # Nenhuma
        
        matriz_ref = matriz_dados.copy()
        canais_upper = [c.strip().upper() for c in canais_selecionados]

        if tipo_ref == 1 or tipo_ref == 2:
            if 'C3' in canais_upper and 'C4' in canais_upper:
                idx_c3 = canais_upper.index('C3')
                idx_c4 = canais_upper.index('C4')
                
                fator = 0.55 if tipo_ref == 1 else 0.50
                sinal_referencia = fator * (matriz_dados[:, idx_c3] + matriz_dados[:, idx_c4])
                matriz_ref = matriz_dados - sinal_referencia[:, np.newaxis]
            else:
                print("⚠️ Aviso: C3 ou C4 não encontrados. Re-referenciamento ignorado.")
                
        elif tipo_ref == 3: # CAR
            sinal_referencia = np.mean(matriz_dados, axis=1)
            matriz_ref = matriz_dados - sinal_referencia[:, np.newaxis]

        return matriz_ref

    def carregar_labels_txt(self, file_manual=True):
        if file_manual:
            fname, _ = QFileDialog.getOpenFileName(self, "Carregar Arquivo de Montagem", "", "Arquivos de Texto (*.txt)")
            if fname:
                try: self.aquisicao.channel_labels_by_file(fname)
                except Exception as e: return QMessageBox.critical(self, "Erro", f"Falha ao ler arquivo: {str(e)}")
            else: return 
        else:
            try: self.aquisicao.channel_labels_by_file(os.path.join(os.path.dirname(__file__), 'channel_labels.txt'))
            except Exception: return
                
        if not self.aquisicao.channels: return

        max_idx = max(self.aquisicao.channels.values()) 
        total_canais = max_idx + 1
        
        while len(self.canais_base) < total_canais:
            self.canais_base.append(f"CH {len(self.canais_base)+1}")
            
        for label, idx in self.aquisicao.channels.items():
            self.canais_base[idx] = label
                
        self.spin_shape_ch.setValue(total_canais)
        self.mudar_numero_canais(total_canais)
        if file_manual: QMessageBox.information(self, "Sucesso", f"Montagem carregada com {total_canais} canais.")
  
    def abrir_selecao_canais_ia(self):
        canais_atuais = self.canais_base[:self.n_channels]
        dialogo = DialogoSelecaoCanaisIA(canais_atuais, self.canais_ia)
        if dialogo.exec_() == QDialog.Accepted:
            novos_canais = dialogo.obter_selecionados()
            if len(novos_canais) == 0: return QMessageBox.warning(self, "Aviso", "Selecione pelo menos 1 canal.")
            if self.model and hasattr(self.model, 'input_shape'):
                ch_esperados = self.model.input_shape[-1]
                if len(novos_canais) != ch_esperados:
                    QMessageBox.warning(self, "Erro de Dimensão", f"O modelo requer exatamente {ch_esperados} canais.\nVocê selecionou {len(novos_canais)}.")
                    return self.abrir_selecao_canais_ia() 
            self.canais_ia = novos_canais
            QMessageBox.information(self, "Canais IA", f"Canais definidos para a IA:\n" + ", ".join(self.canais_ia))

    def mudar_numero_canais(self, num_ch):
        self.n_channels = num_ch
        self.canais = self.canais_base[:self.n_channels] if self.n_channels <= len(self.canais_base) else self.canais_base + [f"CH {i+1}" for i in range(len(self.canais_base), self.n_channels)]
        
        self.ax_time.clear()
        self.ax_time.set_xlim(0, self.x_size)
        self.ax_time.set_yticks([])
        for spine in self.ax_time.spines.values(): spine.set_color('#555555')
        self.ax_time.spines['right'].set_visible(False)
        self.ax_time.spines['top'].set_visible(False)
        
        colors = ['#00bcd4', '#ff4081', '#71c671', '#e8c346', '#e68136', '#8959a8', '#d84e4e', '#8c564b']
        self.lines_time = []
        self.ch_texts = [] 
        
        for i in range(self.n_channels):
            l, = self.ax_time.plot([],[], lw=1.2, color=colors[i%8])
            self.lines_time.append(l)
            txt = self.ax_time.text(self.x_size+10, 0, "", fontsize=9, fontweight='bold', color='#ffffff', verticalalignment='center')
            self.ch_texts.append(txt)
            
        self.atualizar_limites_temporal()
        self.can_time.draw_idle()

        self.ax_fft.clear()
        self.ax_fft.tick_params(colors='#ffffff', which='both'); self.ax_fft.set_yscale('log')
        self.ax_fft.set_ylim(0.1, 100); self.ax_fft.set_xlim(0, 60); self.ax_fft.grid(True, which='both', color='#444444', alpha=0.8)
        self.ax_fft.set_xlabel('Freq (Hz)', color='#aaaaaa'); self.ax_fft.set_ylabel('uV', color='#aaaaaa')
        for spine in self.ax_fft.spines.values(): spine.set_color('#555555')
        self.lines_fft = [self.ax_fft.plot([],[], lw=1.5, alpha=0.8, color=colors[i%8])[0] for i in range(self.n_channels)]
        self.can_fft.draw_idle()

        self.current_data_visual = np.zeros((self.x_size, self.n_channels))
        self.fft_buffer_history = np.zeros((self.n_channels, self.x_size//2))
        self.ptr_visual = 0
        
        if hasattr(self, 'aquisicao') and self.aquisicao:
            self.aquisicao.num_canais = self.n_channels
            self.aquisicao.current_data = np.zeros((self.len_data, self.n_channels))

    def add_class_ui(self, txt):
        idx = len(self.lista_lineedits)
        le = QLineEdit(txt)
        self.lista_lineedits.append(le)
        self.form_classes.addRow(f"{idx} ~ ", le)

    def clear_class_ui(self):
        while self.form_classes.rowCount() > 0: self.form_classes.removeRow(0)
        self.lista_lineedits.clear()

    def setup_tabs_graficos(self):
        self.tab_time = QWidget(); l_time = QVBoxLayout(self.tab_time)
        tb_time = QHBoxLayout()
        
        tb_time.addWidget(QLabel("Escala:"))
        self.spin_escala_visual = QSpinBox()
        self.spin_escala_visual.setRange(10, 10000)
        self.spin_escala_visual.setValue(self.escala_visual)
        self.spin_escala_visual.setSuffix(" uV")
        self.spin_escala_visual.valueChanged.connect(self.mudar_escala_manual)
        tb_time.addWidget(self.spin_escala_visual)
        
        self.chk_escala_auto = QCheckBox("Auto")
        self.chk_escala_auto.stateChanged.connect(lambda state: setattr(self, 'escala_auto', state == QtCore.Qt.Checked))
        tb_time.addWidget(self.chk_escala_auto)
        tb_time.addStretch()
        l_time.addLayout(tb_time)

        self.fig_time = Figure(figsize=(5,3), dpi=100); self.can_time = FigureCanvas(self.fig_time)
        self.fig_time.subplots_adjust(left=0.02, right=0.92, top=0.98, bottom=0.08)
        self.ax_time = self.fig_time.add_subplot(111)
        self.fig_time.patch.set_facecolor('#2b2b2b'); self.ax_time.set_facecolor('#2b2b2b'); self.ax_time.tick_params(colors='#ffffff')
        l_time.addWidget(self.can_time); self.tabs_graficos.addTab(self.tab_time, "Série Temporal")

        self.tab_fft = QWidget(); l_fft = QVBoxLayout(self.tab_fft)
        tb_fft = QHBoxLayout()
        self.spin_smooth = QDoubleSpinBox(); self.spin_smooth.setRange(0, 0.99); self.spin_smooth.setSingleStep(0.1)
        self.spin_smooth.valueChanged.connect(self.mudar_smoothfactor)
        tb_fft.addWidget(QLabel("Smooth:")); tb_fft.addWidget(self.spin_smooth); tb_fft.addStretch()
        l_fft.addLayout(tb_fft)

        self.fig_fft = Figure(figsize=(5,3), dpi=100); self.can_fft = FigureCanvas(self.fig_fft)
        self.fig_fft.subplots_adjust(left=0.06, right=0.96, top=0.95, bottom=0.15)
        self.ax_fft = self.fig_fft.add_subplot(111)
        self.fig_fft.patch.set_facecolor('#2b2b2b'); self.ax_fft.set_facecolor('#2b2b2b')
        l_fft.addWidget(self.can_fft); self.tabs_graficos.addTab(self.tab_fft, "FFT")
        
    def mudar_escala_manual(self, valor):
        self.escala_visual = valor
        self.atualizar_limites_temporal()

    def atualizar_limites_temporal(self):
        fator_espacamento = 1.4
        top = self.n_channels * self.escala_visual * fator_espacamento
        self.ax_time.set_ylim(-self.escala_visual, top + self.escala_visual)

    def mudar_smoothfactor(self):
        self.aquisicao.fft_smooth_factor = self.spin_smooth.value()

    def aplicar_varredura_visual(self, chunk):
        if chunk is None or len(chunk) == 0: return
        n_samples = len(chunk)
        if n_samples > self.x_size: chunk = chunk[-self.x_size:]; n_samples = self.x_size
        end_ptr = self.ptr_visual + n_samples
        if end_ptr <= self.x_size:
            self.current_data_visual[self.ptr_visual:end_ptr, :] = chunk
        else:
            part1 = self.x_size - self.ptr_visual; part2 = n_samples - part1
            self.current_data_visual[self.ptr_visual:self.x_size, :] = chunk[:part1, :]
            self.current_data_visual[0:part2, :] = chunk[part1:, :]

        self.ptr_visual = (self.ptr_visual + n_samples) % self.x_size
        gap_size = int(self.spin_fs.value() * 0.02) 
        if gap_size > 0:
            gap_end = self.ptr_visual + gap_size
            if gap_end <= self.x_size: self.current_data_visual[self.ptr_visual:gap_end, :] = np.nan
            else:
                p1 = self.x_size - self.ptr_visual; p2 = gap_size - p1
                self.current_data_visual[self.ptr_visual:self.x_size, :] = np.nan
                self.current_data_visual[0:p2, :] = np.nan

    def update_loop_continuo(self):
        if self.radio_sim.isChecked():
            chunk = np.random.randn(3, self.n_channels) * 50 
            self.aplicar_varredura_visual(chunk)
            if self.paradigma_win is None or not self.paradigma_win.isVisible(): self.atualizar_graficos_visuais()
        elif self.radio_csv.isChecked(): pass 
        elif self.radio_lsl.isChecked():
            if not getattr(self.aquisicao, 'conectado', False): return 
            chunk = self.aquisicao.adquirir()
            if chunk is not None and len(chunk) > 0:
                chunk_np = np.array(chunk)[:, :self.n_channels]
                self.aplicar_varredura_visual(chunk_np)
                if self.paradigma_win is None or not self.paradigma_win.isVisible(): self.atualizar_graficos_visuais()

    def atualizar_dados_offline(self):
        fs_atual = self.spin_fs.value()
        intervalo_ms = 40
        chunk_size = max(1, int(fs_atual * (intervalo_ms / 1000.0))) 
        if self.ponteiro_arquivo + chunk_size < len(self.dados_arquivo):
            novos_dados = self.dados_arquivo[self.ponteiro_arquivo : self.ponteiro_arquivo + chunk_size]
            if len(self.aquisicao.current_data) == 0: self.aquisicao.current_data = np.array(novos_dados)
            else: self.aquisicao.current_data = np.vstack((self.aquisicao.current_data, novos_dados))
            if len(self.aquisicao.current_data) > self.aquisicao.max_len:
                self.aquisicao.current_data = self.aquisicao.current_data[-self.aquisicao.max_len:]
            self.aquisicao.len_data = len(self.aquisicao.current_data)
            self.aplicar_varredura_visual(novos_dados)
            if self.paradigma_win is None or not self.paradigma_win.isVisible(): self.atualizar_graficos_visuais()
            self.ponteiro_arquivo += chunk_size
        else: self.timer_atualizacao_offline.stop()

    def atualizar_graficos_visuais(self):
        try:
            if len(self.current_data_visual.shape) < 2 or self.current_data_visual.shape[0] < self.x_size or self.current_data_visual.shape[1] < self.n_channels: return 
            if self.tabs_graficos.currentIndex() == 0: 
                if self.escala_auto:
                    mins = np.nanmin(self.current_data_visual, axis=0)
                    maxs = np.nanmax(self.current_data_visual, axis=0)
                    amp = np.nanmax(maxs - mins)
                    if amp > 1:
                        self.spin_escala_visual.blockSignals(True)
                        self.spin_escala_visual.setValue(int(amp * 0.8))
                        self.spin_escala_visual.blockSignals(False)
                        self.escala_visual = int(amp * 0.8)
                        self.atualizar_limites_temporal()
                x = np.arange(self.x_size)
                for i, l in enumerate(self.lines_time):
                    if i >= self.n_channels: break 
                    off = i * self.escala_visual * 1.4
                    recorte_y = self.current_data_visual[:, i]
                    media = np.nanmean(recorte_y) if not np.all(np.isnan(recorte_y)) else 0
                    y = recorte_y - media
                    l.set_data(x, y + off)
                    self.ch_texts[i].set_text(self.canais[i])
                    self.ch_texts[i].set_color(l.get_color())
                    self.ch_texts[i].set_position((self.x_size+10, off))
                self.can_time.draw_idle()
            elif self.tabs_graficos.currentIndex() == 1: 
                fs_atual = self.spin_fs.value() 
                xf = np.linspace(0, fs_atual/2, self.x_size//2)
                if self.aquisicao.fft_buffer_history.shape[1] == self.x_size//2:
                    for i, l in enumerate(self.lines_fft):
                        if i >= self.n_channels: break
                        l.set_data(xf, self.aquisicao.fft_buffer_history[i])
                    self.can_fft.draw_idle()
        except Exception: pass 

    def obter_nomes_classes(self):
        return [le.text().strip() if le.text().strip() else f"Classe {i}" for i, le in enumerate(self.lista_lineedits)]

    def limpar_historico_trial(self):
        self.trial_predicts_log = []
        self.telemetry.update_telemetry([], [], [])

    def abrir_gravacao_paradigma(self):
        nomes = self.obter_nomes_classes()
        if not nomes: return QMessageBox.warning(self, "Aviso", "Adicione pelo menos 1 classe no Gabarito!")
            
        win_config = JanelaConfiguracaoParadigma(self.unity is not None, nomes, playback_mode=self.radio_csv.isChecked())
        if win_config.exec_() == QDialog.Accepted:
            self.paradigma_win = JanelaExecucaoParadigma(win_config.configs, unity_sender=self.unity)
            self.paradigma_win.sinal_extrair_dado.connect(self.processar_epoca_ia)
            self.paradigma_win.sinal_iniciar_pausa.connect(self.iniciar_pausa_tecnica)
            self.paradigma_win.sessao_concluida.connect(self.finalizar_sessao)
            self.paradigma_win.sinal_inicio_acao.connect(self.limpar_historico_trial)
            self.paradigma_win.show()

    def iniciar_sessao_ml(self):
        if self.paradigma_win is None or not self.paradigma_win.isVisible(): return QMessageBox.warning(self, "Aviso", "Configure o Protocolo primeiro!")
        if not self.unity and not self.radio_sim.isChecked():
            if QMessageBox.question(self, "Aviso", "Unity desligado. Continuar?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.No: return
        if not self.model or not self.worker_ia:
            if QMessageBox.question(self, "Aviso", "Nenhum modelo IA carregado. Simular predições?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.No: return
        if self.radio_lsl.isChecked() and not getattr(self.aquisicao, 'conectado', False): return QMessageBox.warning(self, "Aviso", "LSL ativo mas placa não conectada.")

        self.dados_guardados = []; self.marcacoes = []; self.buffer_dados_treino = []; self.buffer_labels_treino = []
        self.ocorreu_transfer_learning = False
        self.historico_probs = []
        self.trial_predicts_log = []
        
        self.btn_iniciar_ia.setEnabled(False)
        self.btn_iniciar_ia.setText("A preparar buffer...")
        
        self.timer_largada = QTimer()
        self.timer_largada.timeout.connect(self.checar_largada)
        self.timer_largada.start(100)

    def checar_largada(self):
        # A IA quer 'X' pontos, mas o LSL pode estar gerando noutra taxa. Vamos conferir quantos pontos físicos esperar.
        fs_eq = self.spin_fs.value()
        fs_ia = self.spin_fs_modelo.value()
        pts_ia = self.spin_shape_time.value()
        pts_reais = int(pts_ia * (fs_eq / fs_ia)) if fs_eq != fs_ia else pts_ia
        
        if len(self.aquisicao.current_data) >= pts_reais or self.radio_sim.isChecked() or self.radio_csv.isChecked():
            self.timer_largada.stop()
            self.lbl_fase.setText("A INICIAR SESSÃO...")
            self.btn_iniciar_ia.setText("Sessão ativa")
            
            if self.radio_csv.isChecked():
                self.timer_atualizacao_offline = QtCore.QTimer()
                self.timer_atualizacao_offline.timeout.connect(self.atualizar_dados_offline)
                self.timer_atualizacao_offline.start(40)
            self.paradigma_win.iniciar_paradigma()

    def processar_epoca_ia(self, label_real, is_training=False):
        # 1. Ajuste Dinâmico de Pontos (Resampling pre-check)
        fs_equip = self.spin_fs.value()
        fs_ia = self.spin_fs_modelo.value()
        pts_ia = self.spin_shape_time.value()
        
        pts_reais = int(pts_ia * (fs_equip / fs_ia)) if fs_equip != fs_ia else pts_ia
        
        dados_brutos = self.aquisicao.pegar_canais_especificos(self.canais_ia)[-pts_reais:, :]
        if dados_brutos.shape[0] < pts_reais: return # Buffer ainda enchendo
        
        # 2. Re-Referenciamento Espacial
        tipo_ref = self.combo_referencia.currentIndex()
        dados_ref = self.aplicar_rereferenciamento(dados_brutos, self.canais_ia, tipo_ref)
        
        # 3. Resampling (Efeito Sanfona)
        if fs_equip != fs_ia:
            dados_processados = resample(dados_ref, pts_ia, axis=0)
        else:
            dados_processados = dados_ref

        # 4. Trava de Segurança IA
        ch_esperados = self.model.input_shape[-1] if (self.model and hasattr(self.model, 'input_shape')) else len(self.canais_ia)
        if dados_processados.shape[1] != ch_esperados: return 
            
        # 5. Normalização
        dados_norm = (dados_processados - dados_processados.min()) / (dados_processados.max() - dados_processados.min() + 1e-8)
        
        if is_training and self.paradigma_win.fase_atual == "CALIBRACAO":
            self.buffer_dados_treino.append(dados_norm)
            self.buffer_labels_treino.append(label_real)

        if self.worker_ia and self.worker_ia.rodando:
            self.worker_ia.pedir_predicao(np.array([dados_norm]), is_training, label_real)
        else:
            nomes = self.obter_nomes_classes()
            self.receber_resposta_ia(random.randint(0, len(nomes)-1), [0.33, 0.33, 0.34], is_training, label_real)

    def receber_resposta_ia(self, pred, prob, is_training, label_real):
        nomes = self.obter_nomes_classes()

        raw_max_idx = prob.index(max(prob))
        nome_raw = nomes[raw_max_idx][:3].upper() 
        self.trial_predicts_log.append(nome_raw)

        self.historico_probs.append(prob)
        if len(self.historico_probs) > 4: self.historico_probs.pop(0)

        pesos_base = [0.05, 0.15, 0.30, 0.50]
        pesos_atuais = pesos_base[-len(self.historico_probs):]
        soma_pesos = sum(pesos_atuais)
        pesos_atuais = [p/soma_pesos for p in pesos_atuais]
        
        wma_prob = [0.0] * len(prob)
        for i, h_prob in enumerate(self.historico_probs):
            for c in range(len(h_prob)): wma_prob[c] += h_prob[c] * pesos_atuais[i]

        max_val = max(wma_prob)
        max_idx = wma_prob.index(max_val)
        nome_predito = nomes[max_idx].lower() if max_idx < len(nomes) else "indefinido"
        
        comando_movimento = "CENTRO" 
        if max_val >= 0.60:
            if "esquerda" in nome_predito: comando_movimento = "ESQUERDA"
            elif "direita" in nome_predito: comando_movimento = "DIREITA"

        epocas_puladas = self.worker_ia.epocas_puladas if self.worker_ia else 0
        self.telemetry.update_telemetry(self.trial_predicts_log, wma_prob, nomes, epocas_puladas)

        if self.paradigma_win and self.paradigma_win.isVisible() and self.paradigma_win.widget_feedback:
            if isinstance(self.paradigma_win.widget_feedback, CubeFeedbackWidget):
                # A velocidade do cubo agora escala com a confiança (max_val),
                # em vez de ser sempre a mesma independente de quão certa a IA está.
                self.paradigma_win.widget_feedback.set_decision(comando_movimento, confianca=max_val)
            elif isinstance(self.paradigma_win.widget_feedback, GaugeWidget):
                p0 = wma_prob[0] if len(wma_prob) > 0 else 0.0
                p1 = wma_prob[1] if len(wma_prob) > 1 else 0.0
                p2 = wma_prob[2] if len(wma_prob) > 2 else 0.0
                self.paradigma_win.widget_feedback.set_probabilities(p0, p1, p2)
        
        p0_main = wma_prob[0] if len(wma_prob) > 0 else 0.0
        p1_main = wma_prob[1] if len(wma_prob) > 1 else 0.0
        p2_main = wma_prob[2] if len(wma_prob) > 2 else 0.0
        self.gauge.set_probabilities(p0_main, p1_main, p2_main)
        
        self.lbl_predicao.setText(nomes[max_idx].upper())
        if "esquerda" in nome_predito: self.lbl_predicao.setStyleSheet(estilo_status(Tema.ESQUERDA))
        elif "direita" in nome_predito: self.lbl_predicao.setStyleSheet(estilo_status(Tema.DIREITA))
        else: self.lbl_predicao.setStyleSheet(estilo_status(Tema.REPOUSO))

        if self.unity:
            if "esquerda" in nome_predito: self.unity.send("HAND_LEFT")
            elif "direita" in nome_predito: self.unity.send("HAND_RIGHT")
            else: self.unity.send("HAND_REST")

        if is_training and self.salvar_dados:
            new_chunk = self.aquisicao.len_data if len(self.dados_guardados) == 0 else self.aquisicao.new_len
            self.marcacoes.append([len(self.dados_guardados), len(self.dados_guardados) + new_chunk, max_idx, label_real])
            self.dados_guardados += self.aquisicao.current_data.copy()[self.aquisicao.len_data - new_chunk:, :].tolist()
            self.historico_probs = []

    def iniciar_pausa_tecnica(self):
        self.lbl_fase.setText("A TREINAR MODELO (PAUSA TÉCNICA)...")
        self.lbl_fase.setStyleSheet(estilo_status(Tema.ALERTA))
        self.ocorreu_transfer_learning = True
        if self.worker_ia:
            self.worker_ia.sinal_treino_concluido.connect(self.retornar_da_pausa)
            self.worker_ia.iniciar_transfer_learning(np.array(self.buffer_dados_treino), np.array(self.buffer_labels_treino))
        else:
            sleep(2)
            self.retornar_da_pausa() 

    def retornar_da_pausa(self):
        if self.worker_ia:
            try: self.worker_ia.sinal_treino_concluido.disconnect(self.retornar_da_pausa)
            except TypeError: pass
        self.lbl_fase.setText("A RETOMAR: TESTE PRÁTICO")
        self.lbl_fase.setStyleSheet(estilo_status(Tema.OK))
        self.paradigma_win.retomar_paradigma()

    def finalizar_sessao(self):
        if hasattr(self, 'timer_atualizacao_offline'): self.timer_atualizacao_offline.stop()
        self.lbl_fase.setText("SESSÃO CONCLUÍDA")
        self.btn_iniciar_ia.setEnabled(True)
        self.btn_iniciar_ia.setText("▶ PASSO 2: INICIAR SESSÃO")
        if self.unity: self.unity.send("HAND_REST")
        if self.salvar_dados:
            self.janela_final = DialogoFimSessao(self.model, self, self.ocorreu_transfer_learning)
            self.janela_final.show()
        else:
            QMessageBox.information(self, "Fim", "Sessão concluída (Nenhum dado guardado).")

    def abrir_arquivo_csv(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Abrir Ficheiro Offline', '', "Ficheiros CSV (*.csv)")
        if fname:
            try:
                df = pd.read_csv(fname, comment='%')
                self.dados_arquivo = df.iloc[:, 1 : self.n_channels + 1].values
                self.ponteiro_arquivo = 0
                QMessageBox.information(self, "Sucesso", "CSV carregado!")
                self.radio_csv.setChecked(True)
            except Exception: pass

    def conectar_Unity(self):
        try:
            self.unity = UnitySender()
            self.lbl_unity.setText("Ligado")
            self.lbl_unity.setStyleSheet(estilo_status(Tema.OK))
        except Exception: pass

    def conectar_LSL(self):
        self.lbl_lsl.setText('A procurar...')
        self.lbl_lsl.setStyleSheet(estilo_status(Tema.ALERTA))
        QApplication.processEvents()
        
        self.aquisicao = Aquisicao(len_data=self.len_data, num_canais=self.spin_shape_ch.value(), xlim_FFT=self.x_size//2, smooth_factor=self.fft_smooth_factor)
        self.carregar_labels_txt(file_manual=False) 
        self.aquisicao.conectar()
        
        if self.aquisicao.conectado:
            self.inlet = self.aquisicao.inlet
            self.lbl_lsl.setText('Ligado!')
            self.lbl_lsl.setStyleSheet(estilo_status(Tema.OK))
            self.radio_lsl.setChecked(True)
            if hasattr(self.aquisicao, 'num_canais') and self.aquisicao.num_canais != self.n_channels:
                self.spin_shape_ch.setValue(self.aquisicao.num_canais)
        else:
            self.lbl_lsl.setText('Falha LSL')
            self.lbl_lsl.setStyleSheet(estilo_status(Tema.ERRO))

    def abrir_modelo(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Abrir Ficheiro IA', '../', "Model files (*.h5)")
        if fname:
            try:
                from keras.models import load_model
                from keras.optimizers import Adam
                self.model = load_model(fname)
                self.model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
                
                in_shape = getattr(self.model, 'input_shape', 'Indefinido')
                out_shape = self.model.output_shape
                
                if isinstance(in_shape, tuple) and len(in_shape) >= 3:
                    self.spin_shape_time.setValue(in_shape[1])
                    shape_info = f"IN: {in_shape} | OUT: {out_shape}"
                else: shape_info = f"Shape OUT: {out_shape}"

                self.modelo_infos.setText(shape_info)
                self.modelo_infos.setStyleSheet(estilo_status(Tema.OK))
                
                if isinstance(out_shape, tuple) and len(out_shape) > 1 and out_shape[1] is not None:
                    n_out = out_shape[1]
                    self.clear_class_ui()
                    for i in range(n_out): self.add_class_ui(f"Classe {i}")
                
                if self.worker_ia: self.worker_ia.parar()
                
                is_binary = (self.combo_tipo_modelo.currentIndex() == 1)
                self.worker_ia = WorkerIA(self.model, is_binary)
                self.worker_ia.sinal_predicao.connect(self.receber_resposta_ia)
                self.worker_ia.start()
            except Exception as e: 
                print(e)

    def closeEvent(self, event):
        if self.worker_ia: 
            self.worker_ia.parar()
            self.worker_ia.wait()
        event.accept()

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
        
        fs_equip = self.hub.spin_fs.value()
        fs_ia = self.hub.spin_fs_modelo.value()
        target_time = int(self.hub.spin_shape_time.value())
        pts_reais = int(target_time * (fs_equip / fs_ia)) if fs_equip != fs_ia else target_time

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
                    if int(end_idx) >= pts_reais:
                        # Extrai do log bruto
                        epoch_bruta_full = np.array(self.hub.dados_guardados[int(end_idx) - pts_reais : int(end_idx)])
                        
                        # Isola apenas os canais que a IA precisa
                        indices_ia = []
                        canais_limpos = {str(k).strip().upper(): v for k, v in self.hub.aquisicao.channels.items()}
                        for i in self.hub.canais_ia:
                            if i.strip().upper() in canais_limpos: indices_ia.append(canais_limpos[i.strip().upper()])
                        
                        if indices_ia:
                            epoch_bruta = epoch_bruta_full[:, indices_ia]
                            
                            # Refaz os mesmos passos que a IA viu para salvar idêntico!
                            tipo_ref = self.hub.combo_referencia.currentIndex()
                            epoch_ref = self.hub.aplicar_rereferenciamento(epoch_bruta, self.hub.canais_ia, tipo_ref)
                            epoch_resampled = resample(epoch_ref, target_time, axis=0) if fs_equip != fs_ia else epoch_ref
                            
                            dados_separados[int(label_real)].append(epoch_resampled)
                    
            for i in range(num_outputs):
                nome_classe = nomes_finais[i] if i < num_outputs else str(i)
                pasta_classe = os.path.join(caminho_completo, f"output_{nome_classe}")
                if not os.path.exists(pasta_classe): os.makedirs(pasta_classe)
                
                cabecalho_canais = ", ".join(self.hub.canais_base)
                for j, epoch in enumerate(dados_separados[i]):
                    ep_norm = (epoch - epoch.min()) / (epoch.max() - epoch.min() + 1e-8)
                    np.savetxt(os.path.join(pasta_classe, f"epoch_{j}.txt"), ep_norm, header=cabecalho_canais, comments='')
                with open(os.path.join(pasta_classe, "info.txt"), "w") as f:
                    f.write(f"Classe: {nome_classe}\n")
                    f.write(f"Descrição: {self.hub.combo_descricao.currentText()}\n")
                    f.write(f"Número de Épocas: {len(dados_separados[i])}\n")
                    f.write(f"Shape de cada Época: {target_time} x {len(self.hub.canais_ia)}\n")
                    f.write(f"Tempo de aviso entre estímulos: {self.hub.paradigma_win.spin_aviso.value()} s\n")
                    f.write(f"Tempo de ação: {self.hub.paradigma_win.spin_acao.value()} s\n")
                    f.write(f"Tempo de repouso entre estímulos: {self.hub.paradigma_win.spin_repouso.value()} s\n")
        else:
            fname = os.path.join(caminho_completo, f"{paciente}_EEG_Agrupado.txt")
            with open(fname, "w") as f:
                for item in self.hub.dados_guardados: f.write("%s\n" % item)

        if self.ocorreu_tl and self.model:
            caminho_modelo = os.path.join(caminho_completo, f"Modelo_Adaptado_{paciente}.h5")
            self.model.save(caminho_modelo)
            mensagem_modelo = f"\n\n🤖 Novo Modelo Guardado:\nModelo_Adaptado_{paciente}.h5"
        else:
            mensagem_modelo = "\n\n(O modelo não foi alterado, pois não ocorreu Transfer Learning)."

        QMessageBox.information(self, "Sessão Concluída", f"Dados guardados em:\n{caminho_completo}{mensagem_modelo}")
        self.accept()

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    win = JanelaInicial()
    win.show()
    sys.exit(app.exec_())