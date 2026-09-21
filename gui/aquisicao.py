from dependencias import *
from pylsl import StreamInlet, resolve_byprop
import socket
import json
from scipy.fft import fft, rfft
import numpy as np

class Aquisicao:
    def __init__(self, len_data, num_canais=16, xlim_FFT=200, smooth_factor=0, protocolo="LSL"):
        self.len_data = len_data
        self.num_canais = num_canais
        self.conectado = False
        self.current_data = np.zeros((self.len_data, self.num_canais))
        self.new_len = 0
        
        # Limpa strings como "UDP (OpenBCI)" pegando só a primeira palavra
        self.protocolo = str(protocolo).split()[0].upper()

        # --- Variáveis LSL ---
        self.inlet = None

        # --- Variáveis de Sockets (Rede Crua) ---
        self.ip_udp = "0.0.0.0"       # Ouve tudo na rede (Padrão OpenBCI)
        self.ip_tcp = "192.168.56.1"  # IP exato do Curry (como na sua imagem)
        self.porta = 4455             # Porta do Curry (Se usar OpenBCI, ele mandará na porta que vc configurar)
        self.sock = None

        self.xlim_FFT = xlim_FFT 
        self.fft_len = self.len_data * 2
        self.fft_smooth_factor = smooth_factor
        self.fft_buffer_history = np.zeros((self.num_canais, self.xlim_FFT))
        self.fft_data = np.zeros((self.xlim_FFT, self.num_canais))
        self.channels = {}

    def conectar(self):
        if self.protocolo == "LSL":
            print("Aguardando stream EEG via LSL...")
            self.streams = resolve_byprop('type', 'EEG', timeout=3)
            if self.streams:
                self.inlet = StreamInlet(self.streams[0])
                self.conectado = True
                self.get_channel_labels()
                print("Conectado ao LSL com sucesso!")
            else:
                print("Nenhum stream EEG via LSL encontrado.")
                self.conectado = False

        elif self.protocolo == "UDP":
            print(f"Aguardando dados UDP Broadcast na porta {self.porta}...")
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sock.bind((self.ip_udp, self.porta))
                self.sock.setblocking(False) 
                self.conectado = True
                print("Ouvinte UDP criado com sucesso!")
            except Exception as e:
                print(f"Erro ao abrir porta UDP: {e}")
                self.conectado = False
                
        elif self.protocolo == "TCP":
            print(f"Tentando bater na porta do Servidor Curry TCP ({self.ip_tcp}:{self.porta})...")
            try:
                # Cria a conexão Cliente TCP (Handshake) exigida pelo Curry
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.ip_tcp, self.porta))
                self.sock.setblocking(False) 
                self.conectado = True
                print("Conectado ao Curry TCP com sucesso! O stream começou.")
            except Exception as e:
                print(f"Erro ao conectar no Servidor TCP do Curry. Verifique o IP! Erro: {e}")
                self.conectado = False

    def adquirir(self):
        if not self.conectado:
            return None

        chunk = []

        if self.protocolo == "LSL":
            pulled_chunk, _ = self.inlet.pull_chunk(timeout=0.0)
            if pulled_chunk:
                chunk = pulled_chunk

        elif self.protocolo in ["UDP", "TCP"]:
            try:
                while True:
                    # O TCP usa recv(), o UDP usa recvfrom()
                    if self.protocolo == "TCP":
                        data = self.sock.recv(65536)
                    else:
                        data, addr = self.sock.recvfrom(65536)
                        
                    if not data:
                        break # Pacote vazio
                    
                    dados_raw = []
                    try:
                        # 1. TENTA LER COMO TEXTO / JSON (Padrão OpenBCI)
                        texto = data.decode('utf-8')
                        pacote = json.loads(texto)
                        
                        if isinstance(pacote, dict):
                            for chave in ["data", "chunk", "eeg", "timeSeriesRaw", "samples"]:
                                if chave in pacote:
                                    dados_raw = pacote[chave]
                                    break
                            if not dados_raw:
                                for val in pacote.values():
                                    if isinstance(val, list):
                                        dados_raw = val
                                        break
                        elif isinstance(pacote, list):
                            dados_raw = pacote
                            
                    except Exception:
                        # 2. CAI AQUI SE FOR CURRY (Pacote Binário TCP Puro)
                        try:
                            # Converte os bytes puros da Compumedics para floats do Python
                            amostra_bruta = np.frombuffer(data, dtype=np.float32).tolist()
                            
                            if len(amostra_bruta) > self.num_canais:
                                amostra_bruta = amostra_bruta[-self.num_canais:]
                                
                            if len(amostra_bruta) > 0:
                                dados_raw = [amostra_bruta]
                        except Exception:
                            continue
                        
                    if not dados_raw:
                        continue
                        
                    if not isinstance(dados_raw[0], list):
                        dados_raw = [dados_raw]
                        
                    for amostra in dados_raw:
                        amostra_formatada = list(amostra)
                        
                        if self.num_canais == 16 and len(amostra_formatada) == 8:
                            if not hasattr(self, 'buffer_daisy'):
                                self.buffer_daisy = []
                            if not self.buffer_daisy:
                                self.buffer_daisy = amostra_formatada 
                                continue 
                            else:
                                amostra_formatada = self.buffer_daisy + amostra_formatada 
                                self.buffer_daisy = [] 
                        else:
                            if len(amostra_formatada) < self.num_canais:
                                amostra_formatada += [0.0] * (self.num_canais - len(amostra_formatada))
                            elif len(amostra_formatada) > self.num_canais:
                                amostra_formatada = amostra_formatada[:self.num_canais]
                            
                        chunk.append(amostra_formatada)
                            
            except BlockingIOError:
                pass
            except Exception as e:
                print(f"Erro Rede: {e}")

        if not chunk: 
            self.new_len = 0
            return None
            
        try:
            chunk_np = np.array(chunk, dtype=float)
            if chunk_np.ndim == 1:
                chunk_np = np.expand_dims(chunk_np, axis=0)
        except Exception:
            return None 

        # ===========================================================
        # A MURALHA MATEMÁTICA (BLINDAGEM LSL/UDP/TCP)
        # ===========================================================
        ch_chegados = chunk_np.shape[1]
        if ch_chegados < self.num_canais:
            # Se vierem menos canais, preenche o vazio com zeros
            pad_zeros = np.zeros((chunk_np.shape[0], self.num_canais - ch_chegados))
            chunk_np = np.hstack((chunk_np, pad_zeros))
        elif ch_chegados > self.num_canais:
            # Se vierem mais canais (ex: 65 do LSL contra 64 da tela), CORTA o excesso!
            chunk_np = chunk_np[:, :self.num_canais]
        # ===========================================================

        if len(chunk_np) > self.len_data:  
            chunk_np = chunk_np[-self.len_data:]
            
        self.new_len = len(chunk_np)
        self.current_data = np.roll(self.current_data, -self.new_len, axis=0)
        
        # Agora o chunk_np e a current_data terão exatamente o mesmo tamanho!
        self.current_data[-self.new_len:, :] = chunk_np

        for i in range(self.num_canais):
            channel_data = self.current_data[:, i]
            segment_FFT = channel_data[-self.xlim_FFT*2:]
            fft_data = rfft(segment_FFT)
            fft_mag = fft_data[:len(segment_FFT)//2]
            fft_mag = 2.0/len(segment_FFT) * np.abs(fft_mag)
            f = self.fft_smooth_factor
            self.fft_buffer_history[i] = (self.fft_buffer_history[i]*f) + (fft_mag*(1-f))
            self.fft_data[:, i] = fft_mag[:self.xlim_FFT]

        return chunk_np.tolist()

    def predict(self, model):
        return model.predict(self.current_data[np.newaxis, :, :])
    
    def compute_fft(self, n_fft=None): 
        if n_fft is None:
            n_fft = self.fft_len
        segment = self.current_data[-self.xlim_FFT:, :]
        fft_vals = rfft(segment, axis=0)
        mag = np.abs(fft_vals)
        self.fft_data[:mag.shape[0], :] = mag
        freqs = np.linspace(0, self.fs / 2, mag.shape[0], endpoint=True)
        return freqs, self.fft_data
    
    def pegar_canais_especificos(self, nomes_canais):
        ch_solicitados = len(nomes_canais)
        tempo_total = self.current_data.shape[0]
        dados_out = np.zeros((tempo_total, ch_solicitados))

        if not self.channels:
            limite = min(ch_solicitados, self.current_data.shape[1])
            dados_out[:, :limite] = self.current_data[:, :limite]
            return dados_out
            
        try:
            canais_limpos = {}
            for k, v in self.channels.items():
                nome_limpo = str(k).strip().upper()
                if nome_limpo not in canais_limpos:
                    canais_limpos[nome_limpo] = v
                    
            for idx_ordem, nome in enumerate(nomes_canais):
                nome_buscado = str(nome).strip().upper()
                if nome_buscado in canais_limpos:
                    idx_real = canais_limpos[nome_buscado]
                    if idx_real < self.current_data.shape[1]:
                        dados_out[:, idx_ordem] = self.current_data[:, idx_real]
            
            return dados_out
            
        except Exception:
            return dados_out
    
    def get_channel_labels(self):
        """ Escaneia o XML da rede LSL ao vivo em busca dos nomes dos eletrodos """
        if self.protocolo == "LSL" and self.conectado and self.inlet is not None:
            try:
                # 1. PEGA O XML INTEIRO AO VIVO
                xml_completo = self.inlet.info().as_xml()
                
                # 2. IMPRIME NO TERMINAL PARA VOCÊ VER A VERDADE
                print("\n" + "="*50)
                print("📡 XML RECEBIDO AO VIVO DA PLACA:")
                print("="*50)
                print(xml_completo)
                print("="*50 + "\n")

                # 3. TENTA EXTRAIR PELO PADRÃO OFICIAL DO LSL
                channels_node = self.inlet.info().desc().child('channels')
                channel_labels = dict()
                canais_vistos = set() 
                
                for i in range(self.inlet.info().channel_count()):
                    # Tenta ler a tag <label>
                    label_str = channels_node.child("channel").child_value('label')
                    
                    # Se <label> estiver vazio, tenta ler <name> (alguns hardwares usam name)
                    if not label_str:
                        label_str = channels_node.child("channel").child_value('name')
                        
                    label_limpo = str(label_str).strip().upper() 
                    
                    # Só salva se ele realmente encontrou um nome válido
                    if label_limpo and label_limpo != "NONE" and label_limpo not in canais_vistos:
                        canais_vistos.add(label_limpo)
                        channel_labels[str(label_str).strip()] = i
                        
                    channels_node = channels_node.next_sibling()
                    
                if channel_labels: 
                    self.channels = channel_labels
                    print(f"✅ SUCESSO: {len(self.channels)} nomes extraídos ao vivo do XML!")
                else:
                    print("⚠️ AVISO: A placa não enviou os nomes nas tags padrão do XML.")
                    
            except Exception as e: 
                print(f"Erro ao ler XML ao vivo: {e}")
                
        return self.channels  
    def channel_labels_by_file(self, file_path):
        channel_labels = dict()
        canais_vistos = set() 
        try:
            with open(file_path, 'r') as f:
                for line in f.read().splitlines():
                    if ' - ' in line:
                        index_str, label_str = line.split(' - ', 1)
                        label_limpo = str(label_str).strip().upper() 
                        
                        # O SEGREDO MATEMÁTICO: Subtrai 1 para o Curry alinhar com o Python
                        idx_real = int(str(index_str).strip()) - 1
                        
                        if label_limpo not in canais_vistos:
                            canais_vistos.add(label_limpo)
                            channel_labels[str(label_str).strip()] = idx_real
                            
            self.channels = channel_labels
            return channel_labels
        except Exception as e:
            print(f"Erro ao carregar arquivo txt de labels: {e}")
            return {}
class AquisicaoOffline(Aquisicao):
    def __init__(self, len_data, num_canais=16, xlim_FFT=200, smooth_factor=0, data_source=None, len_chunk=3):
        super().__init__(len_data, num_canais, xlim_FFT, smooth_factor, protocolo="OFFLINE")
        self.conectado = True  
        self.data_source = data_source  
        self.current_index = 0  
        self.len_chunk = len_chunk  

    def adquirir(self):
        chunk = self.data_source[self.current_index:self.current_index + self.len_chunk]
        self.current_index += self.len_chunk
        self.current_data = np.roll(self.current_data, -self.len_chunk, axis=0)
        self.current_data[-self.len_chunk:, :] = chunk

        for i in range(self.num_canais):
            channel_data = self.current_data[:, i]
            segment_FFT = channel_data[-self.xlim_FFT*2:]
            fft_data = rfft(segment_FFT)
            fft_mag = fft_data[:len(segment_FFT)//2]
            fft_mag = 2.0/len(segment_FFT) * np.abs(fft_mag)
            f = self.fft_smooth_factor
            self.fft_buffer_history[i] = (self.fft_buffer_history[i]*f) + (fft_mag*(1-f))
            self.fft_data[:, i] = fft_mag[:self.xlim_FFT]
        return chunk
    
    def reset(self):
        self.current_index = 0
        self.current_data = np.zeros((self.len_data, self.num_canais))
        self.fft_buffer_history = np.zeros((self.num_canais, self.xlim_FFT))
        self.fft_data = np.zeros((self.xlim_FFT, self.num_canais))

    def data_from_file(self, file_path=None):
        if file_path: self.data_source = np.load(file_path)
        self.reset()