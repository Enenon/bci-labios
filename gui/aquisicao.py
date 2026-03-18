from pylsl import StreamInlet, resolve_byprop
import numpy as np
from scipy.fft import rfft
import time

class Aquisicao:
    def __init__(self, len_data=1000, num_canais=16, fs=250.0):
        self.len_data = len_data
        self.num_canais = num_canais
        self.fs = fs
        self.conectado = False
        self.current_data = np.zeros((len_data, num_canais), dtype=np.float32)
        self.fft_len = 512
        self.fft_data = np.zeros((self.fft_len // 2 + 1, num_canais), dtype=np.float32)
        self.inlet = None

    def conectar(self, timeout=3.0):
        streams = resolve_byprop('type', 'EEG', timeout=timeout)
        if not streams:
            self.conectado = False
            return False
        self.inlet = StreamInlet(streams[0])
        self.conectado = True
        return True

    def adquirir(self):
        if not self.conectado:
            return None
        chunk, _ = self.inlet.pull_chunk(timeout=0.0)
        if not chunk:
            return None

        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk.reshape(-1, self.num_canais)

        if chunk.shape[1] > self.num_canais:
            chunk = chunk[:, :self.num_canais]
        elif chunk.shape[1] < self.num_canais:
            toadd = np.zeros((chunk.shape[0], self.num_canais - chunk.shape[1]), dtype=np.float32)
            chunk = np.concatenate([chunk, toadd], axis=1)

        new_len = chunk.shape[0]
        self.current_data = np.roll(self.current_data, -new_len, axis=0)
        self.current_data[-new_len:, :] = chunk
        return chunk

    def get_epoch(self, epoch_size):
        if epoch_size > self.len_data:
            raise ValueError('epoch_size maior que len_data')
        return self.current_data[-epoch_size:, :]

    def compute_fft(self, n_fft=None):
        if n_fft is None:
            n_fft = self.fft_len
        segment = self.current_data[-n_fft:, :]
        fft_vals = rfft(segment, axis=0)
        mag = np.abs(fft_vals)
        self.fft_data[:mag.shape[0], :] = mag
        freqs = np.linspace(0, self.fs / 2, mag.shape[0], endpoint=True)
        return freqs, self.fft_data

if __name__ == '__main__':
    aq = Aquisicao(len_data=1000)
    if aq.conectar():
        while True:
            updated = aq.adquirir()
            if updated is not None:
                print('Chunk', updated.shape, 'FFT', aq.fft_data.shape)
            else:
                time.sleep(0.01)
