from dependencias import *




import os

from aquisicao import Aquisicao
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



class JanelaInicial(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1250,850)
        self.setWindowTitle('BCI Labios')
        self.aplicar_estilo_escuro()

        # --- Variáveis de Sistema ---
        self.unity = None
        self.inlet = None
        self.model = None
        self.conectado_unity = False
        self.sessao_iniciada = False
        self.sincronizado = False
        self.modo_teste_unity = False 
        
        # --- Configs Hardware ---
        self.canais = ['C3', 'C4', 'Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8','T7', 'T8', 'P7', 'P3', 'P4', 'P8', 'O1', 'O2']
        self.n_channels = len(self.canais) 
        self.x_size = 500 
        
        # --- BUFFER DA ESTEIRA ---
        self.buffer_sobra = [] 
        
        
        # --- VISUALIZAÇÃO ---
        self.current_data_visual = np.zeros((self.x_size, self.n_channels))
        self.fs = 250.0  
        self.escala_visual = 150 
        self.escala_auto = False
        self.fft_smooth_factor = 0.0
        self.fft_buffer_history = np.zeros((self.n_channels, self.x_size//2))

        self.aquisicao = Aquisicao(len_data=self.x_size,num_canais=self.n_channels,xlim_FFT=self.x_size//2,smooth_factor=self.fft_smooth_factor)

        # --- LAYOUT ---
        self.centralwidget = QWidget(self)
        self.setCentralWidget(self.centralwidget)
        self.main_layout = QHBoxLayout(self.centralwidget)
        
        self.panel_left = QFrame()
        self.panel_left.setFixedWidth(320)
        self.layout_left = QVBoxLayout(self.panel_left)
        self.setup_painel_esquerdo()
        self.main_layout.addWidget(self.panel_left)

        self.panel_right = QWidget()
        self.layout_right = QVBoxLayout(self.panel_right)
        self.tabs = QTabWidget()
        self.setup_tabs()
        self.layout_right.addWidget(self.tabs)
        self.main_layout.addWidget(self.panel_right, 1)

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
        #self.lbl_lsl.setPalette(self.palette_vermelha)

        self.setup_menu()

        


    def aplicar_estilo_escuro(self):
        qss = """
        QMainWindow, QWidget { background-color: #2b2b2b; color: #ffffff; font-family: 'Segoe UI', Arial; }
        QGroupBox { border: 1px solid #444; border-radius: 5px; margin-top: 10px; font-weight: bold; background-color: #2b2b2b; }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; background-color: #2b2b2b; color: #aaaaaa; }
        QPushButton { background-color: #3c3f41; border: 1px solid #555; border-radius: 4px; padding: 5px; color: white; }
        QPushButton:hover { background-color: #484b4d; }
        QTabWidget::pane { border: 1px solid #444; background-color: #2b2b2b; }
        QTabBar::tab { background: #2b2b2b; color: #888888; padding: 8px 25px; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; font-weight: bold; }
        QTabBar::tab:selected { background: #3c3f41; color: #ffffff; border-bottom: 3px solid #00bcd4; }
        QComboBox, QSpinBox, QDoubleSpinBox { background: #3c3f41; border: 1px solid #555; padding: 3px; color: white; }
        QProgressBar { border: 1px solid #555; text-align: center; color: white; }
        QProgressBar::chunk { background-color: #00bcd4; }
        QCheckBox { color: white; spacing: 5px; }
        QCheckBox::indicator { width: 15px; height: 15px; }
        """
        self.setStyleSheet(qss)

    def setup_ui_controls(self):
        self.lbl_lsl = QLabel('LSL: desconectado'); self.lbl_unity = QLabel('Unity: desconectado');
        self.lbl_model = QLabel('Modelo: nenhum')

        self.layout_left.addWidget(self.lbl_lsl)
        self.layout_left.addWidget(self.lbl_unity)
        self.layout_left.addWidget(self.lbl_model)

        self.btn_lsl = QPushButton('Conectar LSL'); self.btn_lsl.clicked.connect(self.conectar_LSL)
        self.btn_unity = QPushButton('Conectar Unity'); self.btn_unity.clicked.connect(self.conectar_Unity)
        self.btn_iniciar = QPushButton('Iniciar BCI'); self.btn_iniciar.clicked.connect(self.toggle_bci)

        self.layout_left.addWidget(self.btn_lsl)
        self.layout_left.addWidget(self.btn_unity)
        self.layout_left.addWidget(self.btn_iniciar)

        self.label_previsao = QLabel('--', alignment=QtCore.Qt.AlignCenter)
        self.label_previsao.setFont(QtGui.QFont('Arial', 20, QtGui.QFont.Bold))
        self.layout_left.addWidget(self.label_previsao)
        self.layout_left.addStretch()

    def setup_tabs(self):
        self.tab_time = QWidget()
        l_time = QVBoxLayout(self.tab_time)
        tb_time = QHBoxLayout()
        self.combo_scale = QComboBox(); self.combo_scale.addItems(["Auto", "50 uV", "100 uV", "200 uV", "400 uV"])
        self.combo_scale.setCurrentText("200 uV")
        self.combo_scale.currentTextChanged.connect(lambda t: setattr(self, 'escala_auto', True) if t=="Auto" else (setattr(self, 'escala_auto', False), setattr(self, 'escala_visual', int(t.split()[0])), self.atualizar_limites_temporal()))
        tb_time.addWidget(QLabel("Escala:")); tb_time.addWidget(self.combo_scale); tb_time.addStretch()
        l_time.addLayout(tb_time)

        self.fig_time = Figure(figsize=(5,3), dpi=100, facecolor='#ffffff')
        self.can_time = FigureCanvas(self.fig_time)
        self.setup_grafico_temporal()
        l_time.addWidget(self.can_time)
        self.tabs.addTab(self.tab_time, "Série Temporal")

        self.tab_fft = QWidget()
        l_fft = QVBoxLayout(self.tab_fft)
        tb_fft = QHBoxLayout()
        self.spin_smooth = QDoubleSpinBox(); self.spin_smooth.setRange(0, 0.99); self.spin_smooth.setSingleStep(0.1)
        self.spin_smooth.valueChanged.connect(self.mudar_smoothfactor)

        tb_fft.addWidget(QLabel("Smooth:")); tb_fft.addWidget(self.spin_smooth); tb_fft.addStretch()
        l_fft.addLayout(tb_fft)

        self.fig_fft = Figure(figsize=(5,3), dpi=100, facecolor='#ffffff')
        self.can_fft = FigureCanvas(self.fig_fft)
        self.setup_grafico_fft()
        l_fft.addWidget(self.can_fft)
        self.tabs.addTab(self.tab_fft, "FFT")

    def mudar_smoothfactor(self):
        self.aquisicao.fft_smooth_factor = self.spin_smooth.value()

    def setup_grafico_temporal(self):
        self.ax_time = self.fig_time.add_subplot(111)
        self.fig_time.patch.set_facecolor('#ffffff'); self.ax_time.set_facecolor('#ffffff')
        self.ax_time.tick_params(colors='#333333'); self.ax_time.set_xlim(0, self.x_size); self.ax_time.set_yticks([])
        for spine in self.ax_time.spines.values(): spine.set_color('#aaaaaa')
        colors = ['#555555', '#8959a8', '#3e999f', '#71c671', '#e8c346', '#e68136', '#d84e4e', '#8c564b']
        self.lines_time = []; self.rms_texts = []
        for i in range(self.n_channels):
            l, = self.ax_time.plot([],[], lw=1.2, color=colors[i%8])
            self.lines_time.append(l)
            self.rms_texts.append(self.ax_time.text(self.x_size+10, 0, "", fontsize=9, color='#333333'))
        self.atualizar_limites_temporal()

    def setup_grafico_fft(self):
        self.ax_fft = self.fig_fft.add_subplot(111)
        self.fig_fft.patch.set_facecolor('#ffffff'); self.ax_fft.set_facecolor('#ffffff')
        self.ax_fft.tick_params(colors='#333333', which='both'); self.ax_fft.set_yscale('log')
        self.ax_fft.set_ylim(0.1, 100); self.ax_fft.set_xlim(0, 60)
        self.ax_fft.grid(True, which='both', color='#dddddd', alpha=0.8)
        self.ax_fft.set_xlabel('Freq (Hz)', color='#555555'); self.ax_fft.set_ylabel('uV', color='#555555')
        for spine in self.ax_fft.spines.values(): spine.set_color('#aaaaaa')
        colors = ['#555555', '#8959a8', '#3e999f', '#71c671', '#e8c346', '#e68136', '#d84e4e', '#8c564b']
        self.lines_fft = [self.ax_fft.plot([],[], lw=1.5, alpha=0.8, color=colors[i%8])[0] for i in range(self.n_channels)]

    def atualizar_limites_temporal(self):
        top = self.n_channels * self.escala_visual
        self.ax_time.set_ylim(-self.escala_visual, top + self.escala_visual)
    def setup_graficos(self):
        self.fig_time = Figure(figsize=(6, 3), dpi=100, facecolor='#ffffff')
        self.canvas_time = FigureCanvas(self.fig_time)
        self.ax_time = self.fig_time.add_subplot(111)
        self.ax_time.set_xlim(0, self.x_size)
        self.ax_time.set_ylim(-self.escala_visual, self.n_channels * self.escala_visual)
        self.ax_time.set_yticks([])

        self.lines_time = []
        x = np.arange(self.x_size)
        for i in range(self.n_channels):
            line, = self.ax_time.plot(x, np.zeros(self.x_size), lw=1.0)
            self.lines_time.append(line)

        self.layout_right.addWidget(self.canvas_time)

        self.fig_fft = Figure(figsize=(6, 3), dpi=100, facecolor='#ffffff')
        self.canvas_fft = FigureCanvas(self.fig_fft)
        self.ax_fft = self.fig_fft.add_subplot(111)
        self.ax_fft.set_yscale('log')
        self.ax_fft.set_xlim(0, self.fs / 2)
        self.ax_fft.set_ylim(1e-2, 1e3)
        self.ax_fft.set_yticks([])

        self.lines_fft = []
        freq_bins = np.linspace(0, self.fs/2, self.aquisicao.fft_len//2+1)
        for _ in range(self.n_channels):
            line, = self.ax_fft.plot(freq_bins, np.zeros_like(freq_bins), alpha=0.5)
            self.lines_fft.append(line)

        self.layout_right.addWidget(self.canvas_fft)

    def setup_menu(self):
        menu = self.menuBar().addMenu('Arquivo')
        menu.addAction('Carregar Modelo').triggered.connect(self.abrir_modelo)

    def setup_painel_esquerdo(self):
        lbl_titulo = QLabel("CONTROLES")
        lbl_titulo.setFont(QtGui.QFont("Segoe UI", 12, QtGui.QFont.Bold))
        lbl_titulo.setAlignment(QtCore.Qt.AlignCenter)
        self.layout_left.addWidget(lbl_titulo)

        # Status
        group_conn = QGroupBox("Conexões")
        form_conn = QFormLayout()
        self.lbl_lsl = QLabel("Desconectado"); self.lbl_lsl.setStyleSheet("color: #ff5555;")
        self.lbl_unity = QLabel("Desconectado"); self.lbl_unity.setStyleSheet("color: #ff5555;")
        self.lbl_model = QLabel("Nenhum"); self.lbl_model.setStyleSheet("color: gray;")
        form_conn.addRow("LSL:", self.lbl_lsl)
        form_conn.addRow("Unity:", self.lbl_unity)
        form_conn.addRow("IA:", self.lbl_model)
        group_conn.setLayout(form_conn)
        self.layout_left.addWidget(group_conn)

        # --- SHAPE MANUAL ---
        group_shape = QGroupBox("Shape do Modelo")
        layout_shape = QFormLayout()
        self.spin_shape_time = QSpinBox(); self.spin_shape_time.setRange(10, 5000); self.spin_shape_time.setValue(721); self.spin_shape_time.setSuffix(" pts")
        self.spin_shape_ch = QSpinBox(); self.spin_shape_ch.setRange(1, 32); self.spin_shape_ch.setValue(16); self.spin_shape_ch.setSuffix(" ch")
        layout_shape.addRow("Time Steps:", self.spin_shape_time)
        layout_shape.addRow("Canais:", self.spin_shape_ch)
        group_shape.setLayout(layout_shape)
        self.layout_left.addWidget(group_shape)

        # --- INFOS DO MODELO ---
        group_modelo = QGroupBox("Informações do Modelo")
        self.modelo_infos = QLabel("Nenhum modelo carregado."); self.modelo_infos.setStyleSheet("color: gray;")
        group_modelo.setLayout(QVBoxLayout())
        group_modelo.layout().addWidget(self.modelo_infos)
        self.layout_left.addWidget(group_modelo)
        

        # --- CONTROLES E BOTÕES ---
        self.layout_left.addSpacing(10)
        
        self.chk_teste_unity = QCheckBox("Modo Teste Unity (Aleatório)")
        self.chk_teste_unity.setStyleSheet("color: #ff9800; font-weight: bold;")
        self.chk_teste_unity.setToolTip("Gera sinais aleatórios para testar o Unity sem precisar do LSL ou Capacete.")
        self.layout_left.addWidget(self.chk_teste_unity)

        self.btn_lsl = QPushButton("📡 1. Conectar LSL"); self.btn_lsl.clicked.connect(self.conectar_LSL)
        self.btn_unity = QPushButton("🎮 2. Conectar Unity"); self.btn_unity.clicked.connect(self.conectar_Unity)
        self.btn_iniciar = QPushButton("▶ 3. Iniciar Sessão"); self.btn_iniciar.setStyleSheet("background-color: #2e7d32; font-weight: bold;")
        self.btn_iniciar.clicked.connect(self.iniciar_sessao)
        #self.btn_iniciar.clicked.connect(self.toggle_bci)
        
        self.layout_left.addWidget(self.btn_lsl)
        self.layout_left.addWidget(self.btn_unity)
        self.layout_left.addWidget(self.btn_iniciar)


        # Resultado
        group_res = QGroupBox("Predição Atual")
        layout_res = QVBoxLayout()
        self.lbl_predicao = QLabel("--"); self.lbl_predicao.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold)); self.lbl_predicao.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_feedback = QLabel(""); self.lbl_feedback.setAlignment(QtCore.Qt.AlignCenter)
        layout_res.addWidget(self.lbl_predicao); layout_res.addWidget(self.lbl_feedback)
        group_res.setLayout(layout_res)
        self.layout_left.addWidget(group_res)

        self.layout_left.addStretch()

    def toggle_bci(self):
        if self.timer.isActive():
            self.timer.stop(); self.btn_iniciar.setText('Iniciar BCI')
        else:
            self.timer.start(20); self.btn_iniciar.setText('Parar BCI')

    def update_loop(self):
        if self.modo_teste_unity:
            data = np.random.randn(3, self.n_channels_hardware) * 50 
            self.sincronizado = True
            self.buffer_sobra.extend(data)
            self.current_data_visual = np.roll(self.current_data_visual, -3, axis=0)
            self.current_data_visual[-3:, :] = data
        else:
            chunk = self.aquisicao.adquirir()
            if chunk is None:
                return

        x = np.arange(self.x_size)
        for i, line in enumerate(self.lines_time):
            channel = self.aquisicao.current_data[:, i]
            offset = i * self.escala_visual
            line.set_data(x, channel + offset)

        #self.canvas_time.draw_idle()

        '''freq, fft_data = self.aquisicao.compute_fft()
        for i, line in enumerate(self.lines_fft):
            line.set_data(freq, fft_data[:, i])'''
        #self.canvas_fft.draw_idle()

        self.atualizar_graficos_visuais()

        if self.model is not None:
            epoch = self.aquisicao.get_epoch(self.epoch_size)
            norm = (epoch - epoch.min()) / (epoch.ptp() + 1e-8)
            try:
                pred = self.model.predict(norm[np.newaxis, ...], verbose=0)[0]
                if pred.shape[-1] == 3:
                    idx = int(np.argmax(pred))
                    label = ['ESQUERDA', 'DIREITA', 'REPOUSO'][idx]
                else:
                    idx = int(pred[0] > 0.5)
                    label = ['REPOUSO', 'MOV']['idx']
                self.label_previsao.setText(label)
                if self.unity:
                    commands = ['LEFT', 'RIGHT', 'REST']
                    self.unity.send(commands[idx if pred.shape[-1] == 3 else idx])
            except Exception:
                self.label_previsao.setText('Modelo erro')

    def iniciar_sessao(self):
        # MODO TESTE
        self.modo_teste_unity = self.chk_teste_unity.isChecked()
        
        if self.modo_teste_unity:
            if not self.conectado_unity:
                ret = QMessageBox.question(self, "Unity não conectado", "O Unity não está conectado. Deseja iniciar o teste assim mesmo?", QMessageBox.Yes | QMessageBox.No)
                if ret == QMessageBox.No: return
        else:
            if not self.inlet: return QMessageBox.warning(self, "Aviso", "Conecte o LSL ou ative o Modo Teste!")
        
        self.sessao_iniciada = True
        self.btn_iniciar.setEnabled(False)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(10)

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
        if self.aquisicao.adquirir() == None:
            return
        

        # 2. Atualização do Buffer Circular (Numpy é mais rápido que lista append/pop)
        # Desloca os dados antigos para a esquerda e insere os novos no final
        self.atualizar_graficos_visuais()
        #print(self.current_data.shape)
        #print(np.array([self.current_data]).shape)
    
        try:
            #pred = self.predict(np.array([self.current_data[self.x_size - self.epochsize:]])).numpy()[0][0]
            pred = self.predict(np.array([self.aquisicao.current_data[self.x_size - self.epochsize:]])).numpy()[0][0]
            if pred < self.limiar:
                label = 'T1'
            elif pred > 1 - self.limiar:
                label = 'T2'
            else:
                label = 'T0'
        except: pred = 'Modelo incompatível'


        self.label_previsao.setText(str(pred))
        try:
            self.enviarUnity()
        except:
            pass

        # 4. ATUALIZAÇÃO DO GRÁFICO (A mágica acontece aqui)
        # Eixo X é sempre o mesmo (0 a x_size)
        x_data = np.arange(self.x_size)
        
        for i, line in enumerate(self.lines):
            # Pegamos os dados do canal i
            '''channel_data = self.current_data[:, i]'''
            
            # Adicionamos um offset para criar o efeito "Waterfall" (um canal em cima do outro)
            # Sem isso, todos os 16 canais ficariam misturados no zero.
            offset = i * self.escala_visual
           
            '''line.set_data(x_data, channel_data + offset)'''
            line.set_data(x_data, self.aquisicao.current_data[:,i] + offset)

            '''segment_FFT = channel_data[-self.xlim_FFT*2:]
            fft_data = fft(segment_FFT)
            fft_mag = np.abs(fft_data)[:len(segment_FFT)//2]   # magnitude, só metade positiva'''
            freqs = np.arange(len(self.aquisicao.fft_data[:,i]))            # índices (ou converta p/ Hz com fs)
            self.lines_fft[i].set_data(freqs, self.aquisicao.fft_data[:,i]/self.normalizacaoFFT)

            if i == self.FFT_especifico_indice:
                FFT_espec_range_media = np.mean(self.aquisicao.fft_data[:,i][self.FFT_especifico_freq1:self.FFT_especifico_freq2])
                self.FFT_especifico_array.append(FFT_espec_range_media)
                if len(self.FFT_especifico_array) > self.FFT_especifico_frames:
                    self.FFT_especifico_array.pop(0)
            #self.lines_fft[i].set_data([i for i in range(len(self.aquisicao.fft_data[:,i]))],self.aquisicao.fft_data[:,i]/self.normalizacaoFFT)
        print(self.normalizacaoFFT)
        #self.line_FFT_especifico.set_data(np.arange(len(self.FFT_especifico_array)), np.array(self.FFT_especifico_array) / self.normalizacaoFFT_especifico)
        self.line_FFT_especifico.set_data(np.arange(len(self.FFT_especifico_array)), self.FFT_especifico_array)
        #print(len(self.FFT_especifico_array))
        self.canvas_FFT_especifico.draw_idle()
        # Redesenha apenas o canvas
        self.canvas.draw_idle()

        self.canvas_FFT.draw_idle()
    
    def conectar_Unity(self):
        self.unity = UnitySender()

    def enviarUnity(self):
            self.unity.send(str(self.label_previsao))



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

        self.lbl_lsl.setText('Procurando...')
        self.lbl_lsl.setStyleSheet(f"color: {self.palette_amarela.color(QtGui.QPalette.WindowText).name()};")
        QApplication.processEvents() # <--- isso aplica as mudanças antes da função acabar
        #self.streams = resolve_byprop('type', 'EEG',timeout=3)
        self.aquisicao.conectar()
        if self.aquisicao.conectado:
            self.inlet = self.aquisicao.inlet
            palette = QtGui.QPalette()
            palette.setBrush(QtGui.QPalette.All, QtGui.QPalette.WindowText,QtGui.QBrush(QtGui.QColor(4,150,0)))
            self.lbl_lsl.setText('Conectado!')
            self.lbl_lsl.setStyleSheet(f"color: {self.palette_verde.color(QtGui.QPalette.WindowText).name()};")
        else:
            print('Não achou conexão')
            self.lbl_lsl.setStyleSheet(f"color: {self.palette_vermelha.color(QtGui.QPalette.WindowText).name()};")
            self.lbl_lsl.setText('Desconectado')

    def abrir_janela_plot(self):
        widget_escolhido = self.comboBoxPlots.currentIndex()

        self.janela_plot = JanelaPlot(self,widget_escolhido)

    def atualizar_graficos_visuais(self):
        self.current_data_visual = self.aquisicao.current_data.copy()
        if self.tabs.currentIndex() == 0: 
            if self.escala_auto:
                amp = np.ptp(self.current_data_visual, axis=0).max()
                if amp > 1: self.escala_visual = amp * 0.8; self.atualizar_limites_temporal()
            x = np.arange(self.x_size)
            for i, l in enumerate(self.lines_time):
                off = i * self.escala_visual
                y = self.current_data_visual[:, i] - np.mean(self.current_data_visual[:, i])
                l.set_data(x, y + off)
                rms = np.sqrt(np.mean(y**2))
                self.rms_texts[i].set_text(f"{rms:.2f} uVrms"); self.rms_texts[i].set_position((self.x_size+10, off))
            self.can_time.draw_idle()
        elif self.tabs.currentIndex() == 1: 
            xf = np.linspace(0, self.fs/2, self.x_size//2)
            for i, l in enumerate(self.lines_fft):
                l.set_data(xf, self.aquisicao.fft_buffer_history[i])
            self.can_fft.draw_idle()

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
            case 2:
                self.layout1.addWidget(principal.canvas_FFT_especifico)

                self.channelText = QPlainTextEdit(self.centralwidget)
                self.channelText.setPlainText(str(principal.FFT_especifico_indice))
                self.channelButton = QPushButton('Mudar canal')
                self.channelButton.clicked.connect(self.escolher_canal)

                self.freqText1 = QPlainTextEdit(self.centralwidget)
                self.freqText2 = QPlainTextEdit(self.centralwidget)
                self.freqText1.setPlainText(str(principal.FFT_especifico_indice))
                self.freqText2.setPlainText(str(principal.FFT_especifico_indice))
                self.freqButton = QPushButton('Mudar frequencia')
                self.freqButton.clicked.connect(self.escolher_frequencia)

                self.layoutChannel = QHBoxLayout(self.centralwidget)
                self.layout1.addLayout(self.layoutChannel,1)
                self.layoutChannel.addWidget(self.channelText,1)
                self.layoutChannel.addWidget(self.channelButton,1)
                self.layoutChannel.addWidget(QWidget(self),8)

                self.layoutFreq = QHBoxLayout(self.centralwidget)
                self.layout1.addLayout(self.layoutFreq,2)

                self.layoutFreq.addWidget(self.freqText1,1)
                self.layoutFreq.addWidget(self.freqText2,1)
                self.layoutFreq.addWidget(self.freqButton,1)
                self.layoutFreq.addWidget(QWidget(self),8)

        
        self.show()

    def normalizar(self):
        num = self.normtext.toPlainText()
        #print(num)
        self.janelaPrincipal.normalizacaoFFT = float(num)

    def escolher_canal(self):
        num = self.channelText.toPlainText()
        self.janelaPrincipal.FFT_especifico_indice = int(num)
    
    def escolher_frequencia(self):
        num1 = self.freqText1.toPlainText()
        num2  = self.freqText2.toPlainText()
        self.janelaPrincipal.FFT_especifico_freq1 = int(num1)
        self.janelaPrincipal.FFT_especifico_freq2 = int(num2)





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

class UnitySender:
    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self.sender_loop, daemon=True).start()

        self.context = zmq.Context()
        self.socket_pub = self.context.socket(zmq.PUB)
        self.socket_pub.bind("tcp://*:5555")

    def send(self, msg):
        with self.lock:
            self.queue.append(msg)

    def sender_loop(self):
        import time
        while self.running:
            with self.lock:
                if self.queue:
                    msg = self.queue.pop(0)
                    self.socket_pub.send_string(msg)
            time.sleep(0.01)

    def stop(self):
        self.running = False



if __name__ == '__main__':
    def window():
        app = QApplication(sys.argv)
        win = JanelaInicial()
        win.show()

        sys.exit(app.exec_())
    window()