from pylsl import StreamInlet, resolve_stream, resolve_byprop
import numpy as np
from scipy.fft import fft, rfft

class Aquisicao:
    def __init__(self, len_data, num_canais=16, xlim_FFT=200,smooth_factor=0):
        self.len_data = len_data
        self.num_canais = num_canais
        self.conectado = False
        self.current_data = np.zeros((self.len_data, self.num_canais))

        self.xlim_FFT = xlim_FFT # nota: xlim_FFT é o número de pontos para o plot do FFT, mas deve ser o mesmo número que o número de pontos usados para calcular o FFT?
        self.fft_len = self.len_data * 2
        self.fft_smooth_factor = smooth_factor
        self.fft_buffer_history = np.zeros((self.num_canais, self.xlim_FFT))
        self.fft_data = np.zeros((self.xlim_FFT, self.num_canais))

    def conectar(self):
        print("Aguardando stream EEG...")
        self.streams = resolve_byprop('type', 'EEG',timeout=3)
        if self.streams:
           self.inlet = StreamInlet(self.streams[0])
           self.conectado = True
        else:
            print("Nenhum stream EEG encontrado.")
            self.conectado = False

    def adquirir(self):
        if self.conectado:
            chunk, _ = self.inlet.pull_chunk(timeout=0.0)
            if not chunk: return
            new_len = len(chunk)
            self.current_data = np.roll(self.current_data, -new_len, axis=0)
            self.current_data[-new_len:, :] = chunk

            for i in range(self.num_canais):
                # Pegamos os dados do canal i
                channel_data = self.current_data[:, i]

                segment_FFT = channel_data[-self.xlim_FFT*2:]
                fft_data = rfft(segment_FFT)
                fft_mag = fft_data[:len(segment_FFT)//2]   # magnitude, só metade positiva
                fft_mag = 2.0/len(segment_FFT) * np.abs(fft_mag)  # normalização
                f = self.fft_smooth_factor
                self.fft_buffer_history[i] = (self.fft_buffer_history[i]*f) + (fft_mag*(1-f))
                self.fft_data[:, i] = fft_mag[:self.xlim_FFT]

            return chunk

        else:
            print("Não conectado a nenhum stream EEG.")
            return None

    def predict(self,model):
        return model.predict(self.fft_data.reshape(1, -1))
    
    def compute_fft(self, n_fft=None): #ainda não utilizei mas veio da IA e posso implementar
        if n_fft is None:
            n_fft = self.fft_len
        segment = self.current_data[-self.xlim_FFT:, :]
        fft_vals = rfft(segment, axis=0)
        mag = np.abs(fft_vals)
        self.fft_data[:mag.shape[0], :] = mag
        freqs = np.linspace(0, self.fs / 2, mag.shape[0], endpoint=True)
        return freqs, self.fft_data
        

if __name__ == "__main__":
    aq = Aquisicao(len_data=721)
    aq.conectar()
    print(aq.fft_data)
    while True:
        amostra = aq.adquirir()
        if amostra is not None:
            print(aq.fft_data)  # Aqui você pode usar os dados FFT para visualização ou processamento adicional



